"""Gate I: the cached autoregressive baseline over the same codec.

The three properties the plan requires are the three this pins: the sequence is an exact
view of the program, KV-cached incremental decoding equals a full-sequence forward, and
legal-token masks make an invalid program unreachable by construction.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from mojidiff.learning.ar_corpus import _Split
from mojidiff.learning.ar_overfit import _distinct_subgroup_rows
from mojidiff.learning.ar_sweep import _subset_view
from mojidiff.learning.autoregressive import (
    CausalProgramModel,
    SequenceLayout,
    flatten_program,
    generate,
    legal_mask,
    teacher_forcing_inputs,
    unflatten_program,
)
from mojidiff.learning.openmoji_pilot import (
    OpenMojiPilotError,
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
        layout,
        d_model=32,
        heads=4,
        layers=2,
        feedforward=64,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
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


def test_overfit_selection_draws_distinct_subgroups() -> None:
    """The conditioning must be able to tell the overfit icons apart.

    Exact reproduction is the study's strongest criterion, and it is only meetable if
    no two training icons share a prompt. This is the guard on that assumption: if the
    selector ever returned two icons from one subgroup, the criterion would be
    unsatisfiable and the study would read as a model failure.
    """

    rows = tuple(
        SimpleNamespace(source_path=f"icon-{index}.svg", subgroup=f"sub-{index % 3}")
        for index in range(12)
    )
    chosen = _distinct_subgroup_rows(rows, 3, seed=11)
    assert len({row.subgroup for row in chosen}) == 3
    assert _distinct_subgroup_rows(rows, 3, seed=11) == chosen
    with pytest.raises(OpenMojiPilotError):
        _distinct_subgroup_rows(rows, 4, seed=11)


def _metric_model(layout: SequenceLayout) -> CausalProgramModel:
    torch.manual_seed(17)
    return CausalProgramModel(
        layout, d_model=32, heads=4, layers=2, feedforward=64, metric_coordinates=6
    )


def test_metric_coordinates_order_the_coordinate_space(pieces: Pieces) -> None:
    """The property the categorical table did not have, pinned before it is relied on.

    Gate G measured Spearman -0.136 between its trained embedding distances and bin
    distance: the table was not weakly ordered, it was faintly anti-ordered. These
    features are a fixed function of the decoded value, so the ordering holds at
    initialisation and cannot be trained away.
    """

    _, _, layout, programs, _, _ = pieces
    model = _metric_model(layout)
    tokens = flatten_program(programs[0], layout)
    coordinate = next(
        position
        for position in range(layout.length)
        if layout.coordinate_slot_of(position) is not None
    )

    def embedded(value: int) -> torch.Tensor:
        probe = tokens.clone()
        probe[coordinate] = value
        shifted, kinds = teacher_forcing_inputs(probe[None], layout)
        hidden = model.token_embedding(shifted) + model.position_embedding(
            torch.arange(layout.length)
        )[None]
        positions = torch.arange(layout.length)
        with torch.no_grad():
            out = model._apply_metric_coordinates(hidden, shifted, positions, kinds)
        return out[0, coordinate + 1]

    anchor = embedded(100)
    distances = [float((embedded(100 + step) - anchor).norm()) for step in (1, 4, 16, 64)]
    assert distances == sorted(distances), f"not monotone in bin distance: {distances}"
    assert distances[0] < distances[-1] / 4


def test_cached_decode_matches_full_forward_under_metric_coordinates(pieces: Pieces) -> None:
    """The cache equality must survive an input path that depends on earlier tokens."""

    _, _, layout, programs, _, _ = pieces
    model = _metric_model(layout)
    model.eval()
    tokens = flatten_program(programs[0], layout)[None]
    shifted, kinds = teacher_forcing_inputs(tokens, layout)
    with torch.no_grad():
        full, _ = model(shifted, None, kinds=kinds)
        cache: list[tuple[torch.Tensor, torch.Tensor]] | None = None
        stepwise = []
        for position in range(layout.length):
            logits, cache = model(
                shifted[:, position : position + 1],
                None,
                cache,
                offset=position,
                kinds=kinds[:, position : position + 1],
            )
            stepwise.append(logits[:, -1])
    assert torch.allclose(full, torch.stack(stepwise, dim=1), atol=1e-5)


def test_generation_stays_legal_under_metric_coordinates(pieces: Pieces) -> None:
    _, codec, layout, programs, _, _ = pieces
    model = _metric_model(layout)
    model.eval()
    sampled, calls = generate(
        model, programs[0], None, greedy=False, rng=np.random.default_rng(3)
    )
    assert calls == layout.length
    validate_packed_tensor_program(sampled, codec, layout.total_segment_slots)


def test_the_earlier_arm_is_unaffected_by_the_coordinate_option(pieces: Pieces) -> None:
    """Adding metric coordinates must not move a run that did not ask for them.

    `ar-corpus-i2` is a committed negative result and the baseline the next arm is read
    against. Its training loop now goes through `teacher_forcing_inputs` and its model
    takes a `kinds` argument, so both have to be provably inert at zero - otherwise the
    comparison is against a baseline that quietly moved.
    """

    _, _, layout, programs, groups, subgroups = pieces
    tokens = torch.stack([flatten_program(program, layout) for program in programs])
    shifted, kinds = teacher_forcing_inputs(tokens, layout)
    previous = torch.cat((torch.zeros_like(tokens[:, :1]), tokens[:, :-1]), dim=1)
    assert torch.equal(shifted, previous), "the shift itself changed"

    model = _model(layout, groups, subgroups)
    condition = {
        "group": torch.zeros(len(programs), dtype=torch.long),
        "subgroup": torch.zeros(len(programs), dtype=torch.long),
    }
    with torch.no_grad():
        without, _ = model(shifted, condition)
        with_kinds, _ = model(shifted, condition, kinds=kinds)
        with_wrong, _ = model(shifted, condition, kinds=torch.full_like(kinds, 3))
    assert torch.equal(without, with_kinds)
    assert torch.equal(without, with_wrong), "kinds must be inert without the option"


def test_a_growing_prefix_forward_works_under_metric_coordinates(pieces: Pieces) -> None:
    """The uncached comparison path: one call per position, re-reading the whole prefix.

    This is the shape that broke. Full-sequence and single-token forwards were both
    covered and both passed, and the prefix path - only used to measure what the cache
    is worth - was not, so a run trained for eight minutes and then died on a
    diagnostic.
    """

    _, _, layout, programs, _, _ = pieces
    model = _metric_model(layout)
    model.eval()
    tokens = flatten_program(programs[0], layout)[None]
    shifted, kinds = teacher_forcing_inputs(tokens, layout)
    with torch.no_grad():
        full, _ = model(shifted, None, kinds=kinds)
        for position in (0, 1, 7, 64, layout.length - 1):
            logits, _ = model(
                shifted[:, : position + 1], None, kinds=kinds[:, : position + 1]
            )
            assert torch.allclose(logits[:, position], full[:, position], atol=1e-5)


def test_the_scaling_subsets_are_nested_and_the_floor_follows_them(pieces: Pieces) -> None:
    """Two properties the scaling verdict depends on, neither visible in the output.

    Nesting: a larger arm must be a strict superset of a smaller one, so size is the
    only thing that differs between arms. Three independent samples would differ in
    composition too, and a subset holding more flags would move the curve for a reason
    that has nothing to do with scale.

    Subsetting: `_subset_view` must carry the rows, tokens and packed masks together.
    Slicing one and not another would fit the floor on different icons than the arm
    trained on, silently.
    """

    _, _, layout, programs, _, _ = pieces
    split = _Split(tuple(range(len(programs))), layout)  # type: ignore[arg-type]
    split.tokens = torch.stack([flatten_program(program, layout) for program in programs])
    order = np.random.default_rng(7).permutation(len(programs))
    subsets = [np.sort(order[: max(int(round(f * len(programs))), 1)]) for f in (0.5, 1.0)]
    assert set(subsets[0]).issubset(set(subsets[1]))

    view = _subset_view(split, subsets[0])
    assert len(view.rows) == len(subsets[0])
    assert torch.equal(view.tokens, split.tokens[subsets[0]])
    assert view.packed.shape[0] == len(subsets[0])


def test_dropout_is_inert_at_zero_and_adds_no_parameters(pieces: Pieces) -> None:
    """A checkpoint written before dropout existed must still load, and the arms that
    ran without it must still reproduce.

    `nn.Dropout` carries no parameters, but constructing one at p=0 would still change
    the module tree, so it is omitted entirely rather than built and disabled.
    """

    _, _, layout, programs, groups, subgroups = pieces
    torch.manual_seed(0)
    plain = _model(layout, groups, subgroups)
    torch.manual_seed(0)
    zero = CausalProgramModel(
        layout,
        d_model=32,
        heads=4,
        layers=2,
        feedforward=64,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
        dropout=0.0,
    ).eval()
    assert sorted(plain.state_dict()) == sorted(zero.state_dict())

    tokens = flatten_program(programs[0], layout)[None]
    shifted, _ = teacher_forcing_inputs(tokens, layout)
    condition = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }
    with torch.no_grad():
        assert torch.equal(
            plain(shifted, condition)[0], zero(shifted, condition)[0]
        ), "dropout at zero moved the arithmetic"

    torch.manual_seed(0)
    dropped = CausalProgramModel(
        layout,
        d_model=32,
        heads=4,
        layers=2,
        feedforward=64,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
        dropout=0.5,
    )
    dropped.load_state_dict(plain.state_dict())
    dropped.eval()
    with torch.no_grad():
        assert torch.equal(plain(shifted, condition)[0], dropped(shifted, condition)[0]), (
            "dropout must be inert in eval mode, or every sample and every held-out "
            "number would be drawn from a different model than the one selected"
        )
