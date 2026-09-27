"""The render-to-SVG model: exact fast decoding, exact mirroring, a working step.

Pinned here: decoding that skips grammar-forced positions produces exactly the tokens a
position-by-position full-prefix decode would; every decoded program validates; the
token-space mirror is a true pixel mirror; and one training step runs and lowers loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mojidiff.learning.autoregressive import (
    SequenceLayout,
    flatten_program,
    legal_mask,
    unflatten_program,
)
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.render2svg import (
    DecodeStats,
    ModelConfig,
    RenderToProgram,
    TrainConfig,
    augment_program_tokens,
    greedy_decode,
    mirror_program_tokens,
    path_major_order,
    path_major_positions,
    permute_palette_tokens,
    pixel_error,
    render_program_rgb,
    render_trusted_rgb,
    shift_bounds,
    translate_program_tokens,
)
from mojidiff.representation.packed import (
    PackedTensorProgram,
    serialize_packed_svg,
    validate_packed_tensor_program,
)

_CONFIG = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")
_TINY = ModelConfig(
    image_size=32, d_model=32, heads=4, encoder_layers=1, decoder_layers=2, feedforward=64
)

Pieces = tuple[SequenceLayout, list[PackedTensorProgram]]


@pytest.fixture(scope="module")
def pieces() -> Pieces:
    config = load_openmoji_pilot_config(_CONFIG)
    codec = _selected_codec(config)
    by_split, _, _ = load_pilot_index(config)
    rows = _select_rows(by_split["primary/validation"], 3, config.seed + 1)
    programs = [_load_program(row, config, codec) for row in rows]
    layout = SequenceLayout(codec=codec, total_segment_slots=config.total_segment_slots)
    return layout, programs


_TINY_PATH = ModelConfig(
    image_size=32,
    d_model=32,
    heads=4,
    encoder_layers=1,
    decoder_layers=2,
    feedforward=64,
    metric=True,
    fourier=6,
    order="path",
)
_TINY_METRIC = ModelConfig(
    image_size=32,
    d_model=32,
    heads=4,
    encoder_layers=1,
    decoder_layers=2,
    feedforward=64,
    metric=True,
    fourier=6,
)


def _reference_decode(model: RenderToProgram, images: torch.Tensor) -> torch.Tensor:
    """Step by step, re-reading the whole prefix, no cache and no skipping."""

    layout = model.layout
    memory = model.encode(images)
    decoded = torch.zeros((1, layout.length), dtype=torch.long)
    path_order = model.config.order == "path"
    order = torch.arange(layout.length)[None].clone()
    walker = path_major_positions(decoded[0], layout)
    for step in range(layout.length):
        if path_order:
            order[0, step] = next(walker)
        position = int(order[0, step])
        mask = legal_mask(position, decoded[0], layout)
        if int(mask.sum()) == 1:
            decoded[0, position] = int(mask.to(torch.long).argmax())
            continue
        inputs, extras = model.step_inputs(decoded, order if path_order else None, 0, step + 1)
        logits, _ = model.decode(inputs, memory, extras=extras)
        decoded[0, position] = int(logits[0, -1].masked_fill(~mask, float("-inf")).argmax())
    return decoded


@pytest.mark.parametrize(
    "config", [_TINY, _TINY_METRIC, _TINY_PATH], ids=["plain", "metric", "path"]
)
def test_skipping_forced_positions_is_exact(pieces: Pieces, config: ModelConfig) -> None:
    layout, programs = pieces
    torch.manual_seed(0)
    model = RenderToProgram(layout, config).eval()
    images = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.uint8)
    stats = DecodeStats()
    fast = greedy_decode(model, images, stats=stats)
    assert torch.equal(fast, _reference_decode(model, images))
    assert stats.model_calls < layout.length
    validate_packed_tensor_program(
        unflatten_program(fast[0], programs[0], layout), layout.codec, layout.total_segment_slots
    )


def test_batched_decode_matches_single_decodes(pieces: Pieces) -> None:
    layout, _ = pieces
    torch.manual_seed(1)
    model = RenderToProgram(layout, _TINY).eval()
    images = torch.randint(0, 256, (3, 32, 32, 3), dtype=torch.uint8)
    batched = greedy_decode(model, images)
    for index in range(3):
        assert torch.equal(batched[index], greedy_decode(model, images[index : index + 1])[0])


def test_mirror_is_an_exact_pixel_mirror(pieces: Pieces) -> None:
    layout, programs = pieces
    for program in programs:
        tokens = flatten_program(program, layout)
        mirrored = mirror_program_tokens(tokens, layout)
        assert mirrored is not None
        back = mirror_program_tokens(mirrored, layout)
        assert back is not None and torch.equal(back, tokens)
        candidate = unflatten_program(mirrored, program, layout)
        validate_packed_tensor_program(candidate, layout.codec, layout.total_segment_slots)
        original = render_program_rgb(program, layout, 72)
        flipped = render_program_rgb(candidate, layout, 72)
        assert original is not None and flipped is not None
        # Antialiasing is symmetric on a pixel grid aligned to the box, so the mirror
        # is exact up to rasteriser rounding.
        assert pixel_error(flipped, np.ascontiguousarray(original[:, ::-1])) < 0.002
        assert pixel_error(flipped, original) > pixel_error(
            flipped, np.ascontiguousarray(original[:, ::-1])
        )


@pytest.mark.parametrize(
    "config", [_TINY, _TINY_METRIC, _TINY_PATH], ids=["plain", "metric", "path"]
)
def test_a_training_step_lowers_the_loss(pieces: Pieces, config: ModelConfig) -> None:
    layout, programs = pieces
    torch.manual_seed(2)
    model = RenderToProgram(layout, config)
    tokens = flatten_program(programs[0], layout)[None]
    masks = torch.stack([legal_mask(p, tokens[0], layout) for p in range(layout.length)])[None]
    images = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.uint8)
    free = masks.sum(dim=-1) > 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    order = path_major_order(tokens[0], layout)[None] if config.order == "path" else None
    for _ in range(5):
        logits = model(images, tokens, order)
        loss = F.cross_entropy(logits.masked_fill(~masks, float("-inf"))[free], tokens[free])
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]


def test_translation_is_an_exact_pixel_shift(pieces: Pieces) -> None:
    layout, programs = pieces
    for program in programs:
        tokens = flatten_program(program, layout)
        x_low, x_high, y_low, y_high = shift_bounds(tokens, layout)
        # Four quarter units is one view unit, which is one pixel at 72 px.
        dx, dy = (4 if x_high >= 4 else -4), (-4 if y_low <= -4 else 4)
        shifted = translate_program_tokens(tokens, layout, dx, dy)
        assert shifted is not None
        assert translate_program_tokens(tokens, layout, x_high + 1, 0) is None
        assert translate_program_tokens(tokens, layout, 0, y_low - 1) is None
        candidate = unflatten_program(shifted, program, layout)
        validate_packed_tensor_program(candidate, layout.codec, layout.total_segment_slots)
        original = render_program_rgb(program, layout, 72)
        moved = render_program_rgb(candidate, layout, 72)
        assert original is not None and moved is not None
        expected = np.full_like(original, 255)
        sx, sy = dx // 4, dy // 4
        h, w = original.shape[:2]
        expected[max(sy, 0) : h + min(sy, 0), max(sx, 0) : w + min(sx, 0)] = original[
            max(-sy, 0) : h + min(-sy, 0), max(-sx, 0) : w + min(-sx, 0)
        ]
        assert pixel_error(moved, expected) < 0.002


def test_trusted_renderer_matches_the_isolated_one(pieces: Pieces) -> None:
    layout, programs = pieces
    svg = serialize_packed_svg(programs[0], layout.codec, layout.total_segment_slots)
    isolated = render_program_rgb(programs[0], layout, 72)
    assert isolated is not None
    assert np.array_equal(render_trusted_rgb(svg, 72), isolated)


def test_augmented_variants_keep_the_source_masks_where_the_loss_reads(pieces: Pieces) -> None:
    layout, programs = pieces
    config = TrainConfig(augment_mirror=0.5, augment_colour=1.0, augment_max_shift=48)
    rng = np.random.default_rng(0)
    for program in programs:
        tokens = flatten_program(program, layout)
        source = torch.stack([legal_mask(p, tokens, layout) for p in range(layout.length)])
        free = source.sum(dim=-1) > 1
        for _ in range(3):
            variant = augment_program_tokens(tokens, layout, rng, config)
            assert not torch.equal(variant, tokens)
            validate_packed_tensor_program(
                unflatten_program(variant, program, layout),
                layout.codec,
                layout.total_segment_slots,
            )
            recomputed = torch.stack([legal_mask(p, variant, layout) for p in range(layout.length)])
            assert torch.equal(recomputed.sum(dim=-1) > 1, free)
            assert torch.equal(recomputed[free], source[free])
            # And every variant token is legal under the source masks at free positions.
            assert bool(source[free].gather(1, variant[free][:, None]).all())


def test_palette_permutation_recolours_consistently(pieces: Pieces) -> None:
    layout, programs = pieces
    tokens = flatten_program(programs[0], layout)
    identity = np.arange(len(layout.codec.palette))
    assert torch.equal(permute_palette_tokens(tokens, layout, identity), tokens)
    rolled = np.roll(identity, 1)
    recoloured = permute_palette_tokens(tokens, layout, rolled)
    back = permute_palette_tokens(recoloured, layout, np.argsort(rolled))
    assert torch.equal(back, tokens)


@pytest.mark.parametrize("config", [_TINY_METRIC, _TINY_PATH], ids=["metric", "path"])
def test_teacher_forced_logits_match_cached_decoding(pieces: Pieces, config: ModelConfig) -> None:
    """Chunked cached decoding equals one full forward, at the original positions."""

    layout, programs = pieces
    torch.manual_seed(3)
    model = RenderToProgram(layout, config).eval()
    tokens = flatten_program(programs[1], layout)[None]
    order = path_major_order(tokens[0], layout)[None] if config.order == "path" else None
    images = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.uint8)
    full = model(images, tokens, order)
    memory = model.encode(images)
    cache = None
    chunks = []
    for start, end in ((0, 500), (500, 501), (501, layout.length)):
        inputs, extras = model.step_inputs(tokens, order, start, end)
        logits, cache = model.decode(inputs, memory, cache, offset=start, extras=extras)
        chunks.append(logits)
    stepwise = torch.cat(chunks, 1)
    if order is not None:
        stepwise = stepwise.gather(
            1, torch.argsort(order, dim=1)[..., None].expand(-1, -1, stepwise.shape[-1])
        )
    assert torch.allclose(stepwise, full, atol=1e-4)


def test_path_major_order_groups_each_path_with_its_segments(pieces: Pieces) -> None:
    from mojidiff.learning.autoregressive import PATH_STRIDE, SEGMENT_STRIDE

    layout, programs = pieces
    program = programs[0]
    tokens = flatten_program(program, layout)
    order = path_major_order(tokens, layout)
    assert sorted(order.tolist()) == list(range(layout.length))
    lengths = [int(v) for v in program.path_length if int(v) > 0]
    step = 0
    offset = 0
    for path, length in enumerate(lengths):
        assert order[step : step + PATH_STRIDE].tolist() == list(
            range(path * PATH_STRIDE, (path + 1) * PATH_STRIDE)
        )
        step += PATH_STRIDE
        first = layout.path_positions + offset * SEGMENT_STRIDE
        assert order[step : step + length * SEGMENT_STRIDE].tolist() == list(
            range(first, first + length * SEGMENT_STRIDE)
        )
        step += length * SEGMENT_STRIDE
        offset += length


def test_coordinate_roles_follow_the_grammar(pieces: Pieces) -> None:
    from mojidiff.learning.render2svg import (
        ROLE_CONTROL,
        ROLE_ENDPOINT,
        _static_tables,
        coordinate_roles,
        coordinate_value,
    )

    layout, programs = pieces
    tokens = flatten_program(programs[0], layout)[None]
    role, axis = coordinate_roles(tokens, _static_tables(layout))
    program = programs[0]
    # Every active path's start point is an endpoint pair (x then y).
    first_start = 13
    assert int(role[0, first_start]) == ROLE_ENDPOINT and int(axis[0, first_start]) == 0
    assert int(axis[0, first_start + 1]) == 1
    value = coordinate_value(tokens, role)
    assert float(value[0, first_start]) == (int(program.start[0, 0]) - 1) * 0.25
    # Controls exist exactly where the program has quadratic or cubic segments.
    has_curves = bool(np.isin(program.segment_type, (2, 3)).any())
    assert bool((role == ROLE_CONTROL).any()) == has_curves


def test_online_augmentation_streams_exact_reproducible_variants(pieces: Pieces) -> None:
    from mojidiff.learning.render2svg import OnlineAugmentation

    layout, programs = pieces
    tokens = np.stack([flatten_program(p, layout).numpy() for p in programs]).astype(np.int16)
    images = np.stack(
        [render_trusted_rgb(serialize_packed_svg(p, layout.codec, 128), 32) for p in programs]
    )
    config = TrainConfig(augment_original=0.0, augment_seed=5)
    first = OnlineAugmentation(tokens, images, layout, programs[0], 32, config)
    stream = iter(first)
    samples = [next(stream) for _ in range(4)]
    again = iter(OnlineAugmentation(tokens, images, layout, programs[0], 32, config))
    for image, variant, index in samples:
        other_image, other_variant, other_index = next(again)
        assert index == other_index and torch.equal(variant, other_variant)
        assert torch.equal(image, other_image)
        program = unflatten_program(variant, programs[0], layout)
        validate_packed_tensor_program(program, layout.codec, layout.total_segment_slots)
        rendered = render_trusted_rgb(serialize_packed_svg(program, layout.codec, 128), 32)
        assert np.array_equal(image.numpy(), rendered)
