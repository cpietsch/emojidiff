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


def _reference_decode(model: RenderToProgram, images: torch.Tensor) -> torch.Tensor:
    """Position by position, re-reading the whole prefix, no cache and no skipping."""

    layout = model.layout
    memory = model.encode(images)
    decoded = torch.zeros((1, layout.length), dtype=torch.long)
    for position in range(layout.length):
        mask = legal_mask(position, decoded[0], layout)
        if int(mask.sum()) == 1:
            decoded[0, position] = int(mask.to(torch.long).argmax())
            continue
        shifted = torch.cat((torch.zeros((1, 1), dtype=torch.long), decoded[:, :position]), 1)
        logits, _ = model.decode(shifted, memory)
        decoded[0, position] = int(logits[0, -1].masked_fill(~mask, float("-inf")).argmax())
    return decoded


def test_skipping_forced_positions_is_exact(pieces: Pieces) -> None:
    layout, programs = pieces
    torch.manual_seed(0)
    model = RenderToProgram(layout, _TINY).eval()
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


def test_a_training_step_lowers_the_loss(pieces: Pieces) -> None:
    layout, programs = pieces
    torch.manual_seed(2)
    model = RenderToProgram(layout, _TINY)
    tokens = flatten_program(programs[0], layout)[None]
    masks = torch.stack([legal_mask(p, tokens[0], layout) for p in range(layout.length)])[None]
    images = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.uint8)
    free = masks.sum(dim=-1) > 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(5):
        logits = model(images, tokens)
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
