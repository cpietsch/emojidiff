"""Gate I: the cached autoregressive baseline over the same codec.

The three properties the plan requires are the three this pins: the sequence is an exact
view of the program, KV-cached incremental decoding equals a full-sequence forward, and
legal-token masks make an invalid program unreachable by construction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mojidiff.learning.autoregressive import (
    CausalProgramModel,
    SequenceLayout,
    flatten_program,
    generate,
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
from mojidiff.representation.packed import PackedTensorProgram, validate_packed_tensor_program
from mojidiff.representation.program import CodecConfig

Pieces = tuple[
    object, CodecConfig, SequenceLayout, list[PackedTensorProgram], dict[str, int], dict[str, int]
]

_CONFIG = Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")


@pytest.fixture(scope="module")
def pieces() -> Pieces:
    config = load_openmoji_pilot_config(_CONFIG)
    codec = _selected_codec(config)
    by_split, groups, subgroups = load_pilot_index(config)
    rows = _select_rows(by_split["primary/validation"], 3, config.seed + 1)
    programs = [_load_program(row, config, codec) for row in rows]
    layout = SequenceLayout(codec=codec, total_segment_slots=config.total_segment_slots)
    return config, codec, layout, programs, groups, subgroups


def test_the_sequence_is_an_exact_view_of_the_program(pieces: Pieces) -> None:
    _, codec, layout, programs, _, _ = pieces
    assert layout.length == codec.max_paths * 15 + layout.total_segment_slots * 7

    for program in programs:
        tokens = flatten_program(program, layout)
        assert tokens.shape == (layout.length,)
        restored = unflatten_program(tokens, program, layout)
        for name in ("path_length", "layer", "fill", "stroke_width", "segment_type"):
            assert np.array_equal(getattr(restored, name), getattr(program, name))
        assert np.array_equal(restored.start, program.start)
        assert np.array_equal(restored.coordinates, program.coordinates)
        # And the round trip survives the packed validator, not just equality.
        validate_packed_tensor_program(restored, codec, layout.total_segment_slots)


def test_legal_masks_follow_the_segment_kind_decoded_earlier(pieces: Pieces) -> None:
    _, codec, layout, programs, _, _ = pieces
    tokens = flatten_program(programs[0], layout)

    # A path metadata position is statically bounded by its own vocabulary.
    first = legal_mask(0, tokens, layout)
    assert int(first.sum()) == codec.max_segments + 1

    # Coordinate legality is dynamic: it depends on the kind token already in the
    # sequence, which is what makes these masks worth having.
    seen = set()
    for segment in range(layout.total_segment_slots):
        kind = int(tokens[layout.segment_type_position(segment)])
        if kind in seen:
            continue
        seen.add(kind)
        control = legal_mask(layout.segment_type_position(segment) + 1, tokens, layout)
        endpoint = legal_mask(layout.segment_type_position(segment) + 6, tokens, layout)
        if kind == 3:  # CUBIC: slot 0 is a control handle, slot 5 an endpoint
            assert int(control.sum()) == codec.effective_control_coordinate_bins
            assert int(endpoint.sum()) == codec.coordinate_bins
        elif kind == 1:  # LINE: slot 0 is an endpoint, slot 5 is unused
            assert int(control.sum()) == codec.coordinate_bins
            assert int(endpoint.sum()) == 1 and bool(endpoint[0])
        elif kind == 0:  # padding carries no coordinates at all
            assert int(control.sum()) == 1 and bool(control[0])
    assert {0, 3} <= seen, "fixture must contain padding and cubic segments"


def _model(
    layout: SequenceLayout, groups: dict[str, int], subgroups: dict[str, int]
) -> CausalProgramModel:
    torch.manual_seed(0)
    return CausalProgramModel(
        layout, d_model=32, heads=4, layers=2, feedforward=64,
        group_vocab_size=len(groups) + 1, subgroup_vocab_size=len(subgroups) + 1,
    ).eval()


def test_kv_cached_decoding_equals_a_full_sequence_forward(pieces: Pieces) -> None:
    """The cache is a speed mechanism, so it must change nothing about the output.

    The plan is explicit that nominal step counts are not a speed result, which makes
    real incremental decoding a requirement - and a cache that quietly changed the
    logits would make every latency number meaningless.
    """

    _, _, layout, programs, groups, subgroups = pieces
    model = _model(layout, groups, subgroups)
    tokens = flatten_program(programs[0], layout)[None, :64]
    condition = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }

    full, _ = model(tokens, condition)

    cache = None
    stepwise = []
    for position in range(tokens.shape[1]):
        logits, cache = model(tokens[:, position : position + 1], condition, cache, offset=position)
        stepwise.append(logits[:, -1])
    incremental = torch.stack(stepwise, dim=1)

    assert torch.allclose(full, incremental, atol=1e-5), "cached decode diverged from full forward"
    # The cache holds one key/value pair per layer, grown to the full prefix.
    assert cache is not None and len(cache) == 2
    assert cache[0][0].shape[2] == tokens.shape[1]


def test_generation_is_legal_by_construction(pieces: Pieces) -> None:
    """An untrained model sampling freely must still emit a valid packed program."""

    config, codec, layout, programs, groups, subgroups = pieces
    model = _model(layout, groups, subgroups)
    condition = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }

    from mojidiff.representation.packed import serialize_packed_svg
    from mojidiff.representation.renderer import RenderLimits, render_typed_svg_isolated

    for seed in (3101, 7, 99):
        sampled, calls = generate(
            model, programs[0], condition, greedy=False, rng=np.random.default_rng(seed)
        )
        # One forward call per position: the cache means each costs a token, not a prefix.
        assert calls == layout.length
        validate_packed_tensor_program(sampled, codec, layout.total_segment_slots)
        # It is a real sample, not a copy of the template it borrowed dtypes from.
        assert not np.array_equal(sampled.coordinates, programs[0].coordinates)

    # Render-safe end to end, not merely structurally valid.
    svg = serialize_packed_svg(sampled, codec, layout.total_segment_slots)
    _, raster = render_typed_svg_isolated(
        svg, 72, RenderLimits(max_paths=codec.max_paths, timeout_seconds=20)
    )
    assert raster.shape == (72, 72, 4)


def test_no_position_is_ever_left_without_a_legal_token(pieces: Pieces) -> None:
    """An empty mask surfaces as NaN probabilities several steps later, not where it is.

    It is a real failure mode: forcing layers strictly increasing exhausts the layer
    vocabulary, and the sampler then produced NaN rather than an error. This walks the
    grammar over real programs and over sampled prefixes.
    """

    _, _, layout, programs, _, _ = pieces

    for program in programs:
        tokens = flatten_program(program, layout)
        for position in range(layout.length):
            assert bool(legal_mask(position, tokens, layout).any()), position

    # And along prefixes the sampler can actually reach, including the hardest case:
    # every path active and claiming the highest layer it is allowed.
    from mojidiff.learning.autoregressive import PATH_FIELDS, PATH_STRIDE

    decoded = torch.zeros(layout.length, dtype=torch.long)
    for path in range(layout.codec.max_paths):
        base = path * PATH_STRIDE
        for within in range(len(PATH_FIELDS) + 2):
            mask = legal_mask(base + within, decoded, layout)
            assert bool(mask.any()), (path, within)
            legal = torch.nonzero(mask).flatten()
            decoded[base + within] = int(legal[-1])  # always take the largest legal token
