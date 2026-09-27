"""The graph decoder must decode exactly what the reference decoder decodes."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from mojidiff.learning.autoregressive import SequenceLayout, flatten_program
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
    _static_tables,
    coordinate_roles,
    greedy_decode,
    position_labels,
)

_CONFIG = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")


@pytest.fixture(scope="module")
def layout_and_tokens() -> tuple[SequenceLayout, torch.Tensor]:
    config = load_openmoji_pilot_config(_CONFIG)
    codec = _selected_codec(config)
    by_split, _, _ = load_pilot_index(config)
    rows = _select_rows(by_split["primary/validation"], 3, config.seed + 1)
    layout = SequenceLayout(codec=codec, total_segment_slots=config.total_segment_slots)
    tokens = torch.stack([flatten_program(_load_program(r, config, codec), layout) for r in rows])
    return layout, tokens


def test_cpu_rules_match_the_batched_rules(
    layout_and_tokens: tuple[SequenceLayout, torch.Tensor],
) -> None:
    """The graph decoder's per-position role and label rules equal the tensor versions."""

    from mojidiff.learning.fast_decode import GraphDecoder

    layout, tokens = layout_and_tokens
    role, axis = coordinate_roles(tokens, _static_tables(layout))
    field, path = position_labels(tokens, layout)
    helper = object.__new__(GraphDecoder)
    helper.layout = layout
    tables = _static_tables(layout)
    helper.slot, helper.kind_position, helper.start_axis = (t.tolist() for t in tables)
    for row in range(tokens.shape[0]):
        decoded = tokens[row].tolist()
        for position in range(layout.length):
            got_role, got_axis = helper._role(position, decoded)
            assert got_role == int(role[row, position])
            if got_role:
                assert got_axis == int(axis[row, position])
            assert helper._labels(position, decoded) == (
                int(field[row, position]),
                int(path[row, position]),
            )


@cuda
@pytest.mark.parametrize(
    "extra",
    [{}, {"metric": True, "fourier": 6}, {"metric": True, "fourier": 6, "order": "path"}],
    ids=["plain", "metric", "path"],
)
def test_graph_decoding_matches_the_reference(
    layout_and_tokens: tuple[SequenceLayout, torch.Tensor], extra: dict[str, object]
) -> None:
    from mojidiff.learning.fast_decode import GraphDecoder

    layout, _ = layout_and_tokens
    torch.manual_seed(0)
    config = ModelConfig(
        image_size=32,
        d_model=64,
        heads=4,
        encoder_layers=1,
        decoder_layers=2,
        feedforward=128,
        **extra,  # type: ignore[arg-type]
    )
    model = RenderToProgram(layout, config).cuda().eval()
    decoder = GraphDecoder(model, dtype=torch.float32)
    for seed in range(3):
        generator = torch.Generator().manual_seed(seed)
        image = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8, generator=generator)
        reference = greedy_decode(model, image[None].cuda())
        stats = DecodeStats()
        fast = decoder.decode(image, stats=stats)
        assert torch.equal(fast, reference)
        assert stats.model_calls > 0


@cuda
def test_batched_graph_decoding_keeps_greedy_first_and_every_row_valid(
    layout_and_tokens: tuple[SequenceLayout, torch.Tensor],
) -> None:
    from mojidiff.learning.autoregressive import unflatten_program
    from mojidiff.learning.fast_decode import GraphDecoder
    from mojidiff.learning.openmoji_pilot import _load_program
    from mojidiff.representation.packed import validate_packed_tensor_program

    layout, tokens = layout_and_tokens
    torch.manual_seed(0)
    config = ModelConfig(
        image_size=32,
        d_model=64,
        heads=4,
        encoder_layers=1,
        decoder_layers=2,
        feedforward=128,
        metric=True,
        fourier=6,
        order="path",
    )
    model = RenderToProgram(layout, config).cuda().eval()
    single = GraphDecoder(model, dtype=torch.float32)
    many = GraphDecoder(model, dtype=torch.float32, batch=4)
    image = torch.randint(0, 256, (32, 32, 3), dtype=torch.uint8)
    reference = greedy_decode(model, image[None].cuda())
    assert torch.equal(single.decode(image), reference)
    greedy_rows = many.decode_many(image)
    assert all(torch.equal(row, reference[0]) for row in greedy_rows)
    sampled = many.decode_many(image, temperature=1.0, generator=torch.Generator().manual_seed(1))
    assert torch.equal(sampled[0], reference[0])
    pilot = load_openmoji_pilot_config(_CONFIG)
    template = _load_program(load_pilot_index(pilot)[0]["primary/train"][0], pilot, layout.codec)
    for row in sampled:
        validate_packed_tensor_program(
            unflatten_program(row, template, layout), layout.codec, layout.total_segment_slots
        )
