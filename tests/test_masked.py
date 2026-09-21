"""Gate L: the masked model's masks, loss, decoder and baseline, pinned before a GPU waits.

Every property here is one a defect could quietly violate while training looked fine:
a mask family that hides the wrong positions, a completion that is not a valid program,
a dropped path that shifts the others, a loss that pays out at the wrong positions.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from mojidiff.learning.autoregressive import (
    PATH_FIELDS,
    PATH_STRIDE,
    SequenceLayout,
    flatten_program,
    legal_mask,
    unflatten_program,
)
from mojidiff.learning.masked import (
    MASK_FAMILIES,
    MaskedProgramModel,
    MaskMixture,
    apply_mask,
    complete,
    coordinate_positions,
    drop_path,
    family_mask,
    marginal_logits,
    mask_token,
    masked_loss,
    masked_nll,
    model_logits,
    path_blocks,
    whole_path_mask,
)
from mojidiff.learning.masked_inpaint import (
    _paired,
    _t_critical,
    load_masked_study_config,
    run_masked_study,
)
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.representation.packed import (
    PackedTensorProgram,
    serialize_packed_svg,
    unpack_tensor_program,
    validate_packed_tensor_program,
)
from mojidiff.representation.program import CodecConfig
from mojidiff.representation.renderer import RenderLimits, render_typed_svg_isolated

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


def _model(
    layout: SequenceLayout, groups: dict[str, int], subgroups: dict[str, int], **extra: object
) -> MaskedProgramModel:
    torch.manual_seed(0)
    return MaskedProgramModel(
        layout,
        d_model=32,
        heads=4,
        layers=2,
        feedforward=64,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
        **extra,  # type: ignore[arg-type]
    ).eval()


def _legal(tokens: torch.Tensor, layout: SequenceLayout) -> torch.Tensor:
    return torch.stack([legal_mask(p, tokens, layout) for p in range(layout.length)])


def test_path_blocks_partition_the_active_program(pieces: Pieces) -> None:
    _, _, layout, programs, _, _ = pieces
    tokens = flatten_program(programs[0], layout)
    blocks = path_blocks(tokens, layout)
    assert len(blocks) == int((programs[0].path_length > 0).sum())
    assert sum(block.length for block in blocks) == int(programs[0].path_length.sum())
    # Segment blocks are contiguous, disjoint, and in painter order.
    covered = [position for block in blocks for position in block.segments]
    assert covered == sorted(covered) and len(set(covered)) == len(covered)
    assert covered[0] == layout.path_positions
    # Header blocks never include path_length, which the layout keeps fixed.
    for block in blocks:
        assert block.path * PATH_STRIDE not in block.header
        assert len(block.header) == PATH_STRIDE - 1


def test_every_mask_family_hides_what_it_says(pieces: Pieces) -> None:
    _, _, layout, programs, _, _ = pieces
    tokens = flatten_program(programs[0], layout)
    blocks = path_blocks(tokens, layout)
    mixture = MaskMixture(weights={family: 1.0 for family in MASK_FAMILIES})
    rng = np.random.default_rng(3)

    hide = family_mask("path", tokens, layout, mixture, rng)
    hidden_blocks = [block for block in blocks if hide[block.header[0]]]
    assert 1 <= len(hidden_blocks) <= mixture.max_paths
    for block in hidden_blocks:
        assert bool(hide[list(block.header)].all()) and bool(hide[list(block.segments)].all())
        assert not hide[block.path * PATH_STRIDE], "path_length stays visible"

    hide = family_mask("span", tokens, layout, mixture, rng)
    assert not hide[: layout.path_positions].any(), "a span touches no header"
    kinds = [
        layout.segment_type_position(s)
        for s in range(layout.total_segment_slots)
        if hide[layout.segment_type_position(s)]
    ]
    assert kinds == list(range(kinds[0], kinds[-1] + 1, 7)), "hidden segments are contiguous"

    hide = family_mask("style", tokens, layout, mixture, rng)
    assert not hide[layout.path_positions :].any(), "style touches no segment"
    for block in blocks:
        header = hide[[block.path * PATH_STRIDE + w for w in range(PATH_STRIDE)]]
        assert not header[0] and not header[1] and not header[-2:].any()

    hide = family_mask("geometry", tokens, layout, mixture, rng)
    for block in blocks:
        if hide[block.path * PATH_STRIDE + len(PATH_FIELDS)]:
            assert not any(hide[p] for p in block.kinds(layout)), "kinds stay visible"
            assert bool(hide[list(block.coordinates(layout))].all())

    hide = family_mask("random", tokens, layout, mixture, np.random.default_rng(0))
    assert 0 < int(hide.sum()) < layout.length

    # A capped span hides at most span_max segments, and exactly one at span_max 1.
    capped = MaskMixture(weights={"span": 1.0}, span_max=1)
    for seed in range(5):
        hide = family_mask("span", tokens, layout, capped, np.random.default_rng(seed))
        assert int(hide.sum()) == 7, "one segment block: its kind and six coordinates"


def test_the_loss_pays_out_only_at_targets_and_prefers_nearby_bins(pieces: Pieces) -> None:
    _, _, layout, programs, _, _ = pieces
    tokens = flatten_program(programs[0], layout)[None]
    legal = _legal(tokens[0], layout)[None]
    coordinates = coordinate_positions(layout)
    targets = whole_path_mask(tokens[0], layout, 0)[None] & (legal.sum(-1) > 1)
    # A coordinate position: logits one bin off the truth cost less than logits far off.
    position = next(p for p in range(layout.length) if targets[0, p] and coordinates[p])
    truth = int(tokens[0, position])
    near = torch.zeros(1, layout.length, layout.vocabulary)
    far = near.clone()
    near[0, position, truth + 1] = 8.0
    far[0, position, truth + 40] = 8.0
    single = torch.zeros_like(targets)
    single[0, position] = True
    assert masked_loss(near, tokens, legal, single, coordinates, tau=1.0) < masked_loss(
        far, tokens, legal, single, coordinates, tau=1.0
    )
    # The soft target is centred on the truth: logits peaked exactly there beat logits
    # peaked one bin either side. The first overfit run had every coordinate exactly
    # one bin off because the kernel was centred a token low, and this is the test that
    # would have caught it.
    exact = torch.zeros_like(near)
    exact[0, position, truth] = 8.0
    below = torch.zeros_like(near)
    below[0, position, truth - 1] = 8.0
    assert masked_loss(exact, tokens, legal, single, coordinates, tau=1.0) < masked_loss(
        near, tokens, legal, single, coordinates, tau=1.0
    )
    assert masked_loss(exact, tokens, legal, single, coordinates, tau=1.0) < masked_loss(
        below, tokens, legal, single, coordinates, tau=1.0
    )
    # A mixture of widths keeps the minimum at the truth and gives a far miss a
    # gradient it lacks under one narrow kernel: the difference between being 16 and
    # 40 bins off is invisible at tau 1 and visible at tau (1, 8).
    assert masked_loss(exact, tokens, legal, single, coordinates, tau=(1.0, 8.0)) < masked_loss(
        near, tokens, legal, single, coordinates, tau=(1.0, 8.0)
    )
    # For a spread prediction the loss is monotone in the miss under either kernel; a
    # computation on the record showed the narrow kernel already separates a coarse hit
    # from a wild miss, so the mixture is an option, not a fix.
    bins = torch.arange(layout.vocabulary).float()
    for tau in (1.0, (1.0, 8.0)):
        losses = []
        for offset in (0, 8, 16, 40):
            bump = torch.zeros_like(near)
            bump[0, position] = -(bins - (truth + offset)).abs() / 8.0
            losses.append(float(masked_loss(bump, tokens, legal, single, coordinates, tau=tau)))
        assert losses == sorted(losses), tau
    # With tau = 0 the target is exact and the two are equally wrong.
    assert torch.isclose(
        masked_loss(near, tokens, legal, single, coordinates, tau=0.0),
        masked_loss(far, tokens, legal, single, coordinates, tau=0.0),
    )
    # Logits at positions outside the target set change nothing.
    elsewhere = near.clone()
    elsewhere[0, (position + 1) % layout.length] = 5.0
    assert torch.isclose(
        masked_loss(near, tokens, legal, single, coordinates, tau=1.0),
        masked_loss(elsewhere, tokens, legal, single, coordinates, tau=1.0),
    )
    nll, hits, count = masked_nll(near, tokens, legal, single)
    assert count == 1 and hits == 0 and nll > 0


def test_completion_is_a_valid_program_for_every_family(pieces: Pieces) -> None:
    """An untrained model must still complete into a program the validator accepts."""

    _, codec, layout, programs, groups, subgroups = pieces
    model = _model(layout, groups, subgroups, metric_coordinates=4)
    condition = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }
    mixture = MaskMixture(weights={family: 1.0 for family in MASK_FAMILIES if family != "random"})
    tokens = flatten_program(programs[1], layout)
    for family in mixture.families:
        hide = family_mask(family, tokens, layout, mixture, np.random.default_rng(1))
        inputs = apply_mask(tokens, hide, layout)
        assert int((inputs == mask_token(layout)).sum()) == int(hide.sum())
        decoded, calls = complete(
            model_logits(model, condition),
            inputs,
            layout,
            greedy=False,
            rng=np.random.default_rng(7),
            iterations=3,
        )
        assert calls >= 1
        # Visible tokens are untouched; hidden ones are all filled.
        assert torch.equal(decoded[~hide], tokens[~hide])
        assert not bool((decoded == mask_token(layout)).any())
        program = unflatten_program(decoded, programs[1], layout)
        validate_packed_tensor_program(program, codec, layout.total_segment_slots)
        svg = serialize_packed_svg(program, codec, layout.total_segment_slots)
        _, raster = render_typed_svg_isolated(
            svg, 18, RenderLimits(max_paths=codec.max_paths, timeout_seconds=20)
        )
        assert raster.shape == (18, 18, 4)


def test_everything_masked_completes_into_a_valid_program(pieces: Pieces) -> None:
    """Generation is the same decoder with nothing visible."""

    _, codec, layout, programs, groups, subgroups = pieces
    model = _model(layout, groups, subgroups)
    condition = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }
    inputs = torch.full((layout.length,), mask_token(layout), dtype=torch.long)
    for seed in (1, 2):
        decoded, _ = complete(
            model_logits(model, condition),
            inputs,
            layout,
            greedy=False,
            rng=np.random.default_rng(seed),
            iterations=4,
        )
        program = unflatten_program(decoded, programs[0], layout)
        validate_packed_tensor_program(program, codec, layout.total_segment_slots)


def test_the_marginal_policy_completes_through_the_same_decoder(pieces: Pieces) -> None:
    _, codec, layout, programs, _, _ = pieces
    counts = torch.ones(layout.length, layout.vocabulary, dtype=torch.float64)
    for program in programs:
        tokens = flatten_program(program, layout)
        counts[torch.arange(layout.length), tokens] += 1.0
    tokens = flatten_program(programs[2], layout)
    hide = whole_path_mask(tokens, layout, 0)
    decoded, _ = complete(
        marginal_logits(counts), apply_mask(tokens, hide, layout), layout, greedy=True
    )
    validate_packed_tensor_program(
        unflatten_program(decoded, programs[2], layout), codec, layout.total_segment_slots
    )
    assert torch.equal(decoded[~hide], tokens[~hide])


def test_dropping_a_path_removes_exactly_that_path(pieces: Pieces) -> None:
    _, codec, layout, programs, _, _ = pieces
    program = programs[0]
    active = int((program.path_length > 0).sum())
    assert active >= 2
    dropped = drop_path(program, 0, codec, layout.total_segment_slots)
    validate_packed_tensor_program(dropped, codec, layout.total_segment_slots)
    assert int((dropped.path_length > 0).sum()) == active - 1
    before = unpack_tensor_program(program, codec, layout.total_segment_slots)
    after = unpack_tensor_program(dropped, codec, layout.total_segment_slots)
    # Every surviving path is the original one shifted up by one slot, unchanged.
    for name in ("path_length", "fill", "stroke", "layer", "segment_type", "coordinates"):
        assert np.array_equal(getattr(before, name)[1:active], getattr(after, name)[: active - 1])
    with pytest.raises(ValueError):
        drop_path(program, layout.codec.max_paths - 1, codec, layout.total_segment_slots)


def test_metric_coordinates_are_only_applied_to_known_coordinates(pieces: Pieces) -> None:
    """A masked coordinate, or one whose kind is masked, must fall back to the table."""

    _, _, layout, programs, groups, subgroups = pieces
    model = _model(layout, groups, subgroups, metric_coordinates=4)
    condition = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }
    tokens = flatten_program(programs[0], layout)[None]
    with torch.no_grad():
        base = model(tokens, condition)
        # Masking a segment's kind changes the logits somewhere, and does not raise.
        kind_position = layout.segment_type_position(0)
        masked = tokens.clone()
        masked[0, kind_position] = mask_token(layout)
        changed = model(masked, condition)
    assert base.shape == (1, layout.length, layout.vocabulary)
    assert not torch.allclose(base, changed)


def test_the_encoder_sees_a_coordinate_value_change(pieces: Pieces) -> None:
    """Moving one visible coordinate must move the hidden state at that position.

    The corpus arms ignored their neighbours entirely - a 40-bin shift of the previous
    endpoint moved the prediction for the hidden one by a median of zero bins. That
    could be the loss or the input; this pins the input side, at initialisation, so
    that a broken feature path can never hide behind a flat loss.
    """

    _, _, layout, programs, groups, subgroups = pieces
    model = _model(layout, groups, subgroups, metric_coordinates=8)
    tokens = flatten_program(programs[0], layout)[None]
    blocks = path_blocks(tokens[0], layout)
    kind_position = blocks[0].kinds(layout)[0]
    position = kind_position + 1  # slot 0 of the first segment, legal for any kind
    assert int(tokens[0, position]) > 0
    positions = torch.arange(layout.length)
    with torch.no_grad():

        def embed(value: int) -> torch.Tensor:
            probe = tokens.clone()
            probe[0, position] = value
            hidden = model.token_embedding(probe) + model.position_embedding(positions)[None]
            return model._apply_metric_coordinates(hidden, probe)[0, position]

        base = int(tokens[0, position])
        near = float((embed(base + 2) - embed(base)).norm())
        far = float((embed(base + 40) - embed(base)).norm())
    assert near > 0.0 and far > near, "the metric features must order the lattice"


def test_the_metric_head_changes_only_coordinate_logits_and_still_completes(pieces: Pieces) -> None:
    """Coordinate positions with a known role get lattice-aware logits; nothing else moves."""

    _, codec, layout, programs, groups, subgroups = pieces
    torch.manual_seed(0)
    plain = _model(layout, groups, subgroups, metric_coordinates=4)
    torch.manual_seed(0)
    metric = _model(layout, groups, subgroups, metric_coordinates=4, metric_head=True)
    metric.load_state_dict(plain.state_dict(), strict=False)
    tokens = flatten_program(programs[0], layout)[None]
    condition = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }
    with torch.no_grad():
        before = plain(tokens, condition)
        after = metric(tokens, condition)
    coordinates = coordinate_positions(layout)
    differs = ~torch.isclose(before, after).all(dim=-1)[0]
    assert not bool(differs[~coordinates].any()), "non-coordinate logits are untouched"
    assert bool(differs[coordinates].any()), "coordinate logits come from the metric head"
    # A coordinate whose kind is masked keeps the categorical head.
    masked = tokens.clone()
    kind_position = layout.segment_type_position(0)
    masked[0, kind_position] = mask_token(layout)
    with torch.no_grad():
        plain_masked = plain(masked, condition)
        metric_masked = metric(masked, condition)
    assert torch.allclose(plain_masked[0, kind_position + 1], metric_masked[0, kind_position + 1])
    # Under a span mask the kind is hidden with its coordinates, so the head has no
    # role to read from the input - and the first metric-head arm trained every step
    # that way, exercising nothing. Supplying the clean kinds, as the loss supplies its
    # legal masks, turns the head on at exactly those positions.
    from mojidiff.learning.autoregressive import coordinate_kind_tokens

    span = family_mask(
        "span", tokens[0], layout, MaskMixture(weights={"span": 1.0}), np.random.default_rng(2)
    )
    hidden_inputs = apply_mask(tokens[0], span, layout)[None]
    hidden_coordinates = span & coordinates
    assert bool(hidden_coordinates.any())
    with torch.no_grad():
        without = metric(hidden_inputs, condition)
        with_kinds = metric(hidden_inputs, condition, kinds=coordinate_kind_tokens(tokens, layout))
        plain_hidden = plain(hidden_inputs, condition)
    assert torch.allclose(without[0][hidden_coordinates], plain_hidden[0][hidden_coordinates]), (
        "with the kind masked and no kinds supplied, the categorical head is used"
    )
    assert not torch.allclose(
        with_kinds[0][hidden_coordinates], plain_hidden[0][hidden_coordinates]
    ), "with the clean kinds supplied, the metric head fires at the hidden coordinates"
    # The head's extra parameters are one small projection.
    extra = 32 * (2 * 4 + 2) + (2 * 4 + 2)
    assert (
        sum(p.numel() for p in metric.parameters())
        == sum(p.numel() for p in plain.parameters()) + extra
    )
    # And the decoder still produces a valid program through it.
    hide = whole_path_mask(tokens[0], layout, 0)
    decoded, _ = complete(
        model_logits(metric, condition), apply_mask(tokens[0], hide, layout), layout, iterations=3
    )
    validate_packed_tensor_program(
        unflatten_program(decoded, programs[0], layout), codec, layout.total_segment_slots
    )
    with pytest.raises(ValueError):
        _model(layout, groups, subgroups, metric_head=True)


def test_start_features_carry_the_previous_endpoint_and_know_when_it_is_hidden(
    pieces: Pieces,
) -> None:
    """Every segment block sees its own start point, and only when that point is visible."""

    _, codec, layout, programs, groups, subgroups = pieces
    model = _model(layout, groups, subgroups, metric_coordinates=4, start_features=True)
    tokens = flatten_program(programs[0], layout)[None]
    blocks = path_blocks(tokens[0], layout)
    first = blocks[0]
    with torch.no_grad():
        features = model._start_point_features(tokens)
    width = 2 * (2 * 4 + 2) + 1
    assert features.shape == (1, layout.length, width)
    assert not bool(features[0, : layout.path_positions].any()), "headers carry nothing"
    # The first segment's start is the header's start point.
    kind0 = first.kinds(layout)[0]
    start_x = (
        float(tokens[0, first.path * PATH_STRIDE + len(PATH_FIELDS)] - 1)
        * 72.0
        / (codec.coordinate_bins - 1)
    )
    assert torch.isclose(features[0, kind0, 0], torch.tensor(start_x / 72.0))
    assert float(features[0, kind0, -1]) == 1.0
    assert torch.equal(features[0, kind0], features[0, kind0 + 6]), "all seven positions agree"
    # A later segment's start is the previous segment's endpoint, whichever slots its
    # kind keeps that in.
    if first.length >= 2:
        kind1 = first.kinds(layout)[1]
        previous_kind = int(tokens[0, kind0])
        offset = {1: 0, 2: 2, 3: 4}[previous_kind]
        previous_x = float(tokens[0, kind0 + 1 + offset] - 1) * 72.0 / (codec.coordinate_bins - 1)
        assert torch.isclose(features[0, kind1, 0], torch.tensor(previous_x / 72.0))
        # Hiding the previous block hides the start: zero features, zero flag.
        hidden = tokens.clone()
        hidden[0, kind0 : kind0 + 7] = mask_token(layout)
        with torch.no_grad():
            masked = model._start_point_features(hidden)
        assert float(masked[0, kind1, -1]) == 0.0 and not bool(masked[0, kind1].any())
        # But the first block still knows its header start.
        assert float(masked[0, kind0, -1]) == 1.0
    condition = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }
    with torch.no_grad():
        assert model(tokens, condition).shape == (1, layout.length, layout.vocabulary)
    with pytest.raises(ValueError):
        _model(layout, groups, subgroups, start_features=True)


def test_chain_order_commits_a_segment_only_after_its_start(pieces: Pieces) -> None:
    """Under chain order no coordinate is committed while the segment before it is a hole."""

    from mojidiff.learning.masked import _start_known

    _, codec, layout, programs, groups, subgroups = pieces
    model = _model(layout, groups, subgroups, metric_coordinates=4, start_features=True)
    condition = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }
    tokens = flatten_program(programs[0], layout)
    block = max(path_blocks(tokens, layout), key=lambda item: item.length)
    assert block.length >= 3
    hide = whole_path_mask(tokens, layout, block.path)
    inputs = apply_mask(tokens, hide, layout)
    snapshots: list[torch.Tensor] = []
    inner = model_logits(model, condition)

    def recording(decoded: torch.Tensor) -> torch.Tensor:
        snapshots.append(decoded.clone())
        return inner(decoded)

    decoded, calls = complete(recording, inputs, layout, greedy=True, chain_order=True)
    assert calls == len(snapshots) and not bool((decoded == mask_token(layout)).any())
    validate_packed_tensor_program(
        unflatten_program(decoded, programs[0], layout), codec, layout.total_segment_slots
    )
    # Every snapshot in which a later segment's coordinates are visible also shows the
    # previous segment's coordinates visible: the chain was never skipped.
    kinds = block.kinds(layout)
    for snapshot in snapshots:
        for index in range(1, block.length):
            later = snapshot[kinds[index] + 1 : kinds[index] + 7]
            earlier = snapshot[kinds[index - 1] + 1 : kinds[index - 1] + 7]
            if bool((later != mask_token(layout)).any()):
                assert not bool((earlier == mask_token(layout)).any()), index
    # The chain took at least as many passes as the path has segments minus one.
    assert calls >= block.length
    # A whole-path mask hides the header's start point too, so before tier one nothing
    # in the path has a known start; once the header is back, the first segment does
    # and the second does not while the first is still a hole.
    assert not _start_known(kinds[0] + 1, inputs, layout)
    header_back = inputs.clone()
    header_back[list(block.header)] = tokens[list(block.header)]
    assert _start_known(kinds[0] + 1, header_back, layout)
    assert not _start_known(kinds[1] + 1, header_back, layout)


def test_joining_a_span_collapses_it_to_its_entry_and_stays_valid(pieces: Pieces) -> None:
    from mojidiff.learning.masked import join_span, span_mask

    _, codec, layout, programs, _, _ = pieces
    tokens = flatten_program(programs[0], layout)
    block = max(path_blocks(tokens, layout), key=lambda item: item.length)
    assert block.length >= 4
    first, count = 1, 2
    hide = span_mask(tokens, layout, block, first, count)
    assert int(hide.sum()) == 7 * count
    joined = join_span(tokens, layout, block, first, count)
    assert torch.equal(joined[~hide], tokens[~hide]), "only the span changes"
    previous = layout.segment_type_position(block.offset)
    offset = {1: 0, 2: 2, 3: 4}[int(tokens[previous])]
    entry = (int(tokens[previous + 1 + offset]), int(tokens[previous + 2 + offset]))
    for slot in range(block.offset + first, block.offset + first + count):
        base = layout.segment_type_position(slot)
        assert int(joined[base]) == 1
        assert (int(joined[base + 1]), int(joined[base + 2])) == entry
        assert not bool(joined[base + 3 : base + 7].any())
    validate_packed_tensor_program(
        unflatten_program(joined, programs[0], layout), codec, layout.total_segment_slots
    )
    with pytest.raises(ValueError):
        join_span(tokens, layout, block, 0, 1)
    with pytest.raises(ValueError):
        span_mask(tokens, layout, block, block.length - 1, 2)


def test_dropout_adds_no_parameters_and_is_inert_at_zero(pieces: Pieces) -> None:
    _, _, layout, programs, groups, subgroups = pieces
    plain = _model(layout, groups, subgroups)
    regularised = _model(layout, groups, subgroups, dropout=0.2)
    assert sum(p.numel() for p in plain.parameters()) == sum(
        p.numel() for p in regularised.parameters()
    )
    tokens = flatten_program(programs[0], layout)[None]
    condition = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }
    plain.train()
    with torch.no_grad():
        assert torch.allclose(plain(tokens, condition), plain(tokens, condition))


def test_segment_ownership_follows_the_visible_lengths(pieces: Pieces) -> None:
    """Every segment knows its path and its place in it, until a length is hidden."""

    from mojidiff.learning.masked import segment_ownership

    _, _, layout, programs, groups, subgroups = pieces
    tokens = flatten_program(programs[0], layout)
    owner, within = segment_ownership(tokens[None], layout)
    blocks = path_blocks(tokens, layout)
    for block in blocks:
        for position in block.header:
            assert int(owner[0, position]) == block.path
            assert int(within[0, position]) == layout.codec.max_segments
        for index, kind_position in enumerate(block.kinds(layout)):
            for offset in range(7):
                assert int(owner[0, kind_position + offset]) == block.path
                assert int(within[0, kind_position + offset]) == index
    # Slots past every declared length carry the sentinels.
    used = sum(block.length for block in blocks)
    if used < layout.total_segment_slots:
        tail = layout.segment_type_position(used)
        assert int(owner[0, tail]) == layout.codec.max_paths
        assert int(within[0, tail]) == layout.codec.max_segments
    # Masking the second path's length leaves the first path's segments known and every
    # later segment unknown - the binding never guesses.
    hidden = tokens.clone()
    hidden[1 * PATH_STRIDE] = mask_token(layout)
    owner, within = segment_ownership(hidden[None], layout)
    first = blocks[0]
    assert all(int(owner[0, p]) == 0 for p in first.segments)
    later = blocks[1].segments[0]
    assert int(owner[0, later]) == layout.codec.max_paths
    assert int(within[0, later]) == layout.codec.max_segments
    # Header positions keep their path regardless.
    assert int(owner[0, 1 * PATH_STRIDE + 3]) == 1

    # With binding on, the model forwards and differs from the unbound model; off, the
    # parameter count is exactly the earlier arms'.
    bound = _model(layout, groups, subgroups, path_binding=True)
    plain = _model(layout, groups, subgroups)
    extra = (layout.codec.max_paths + 1 + layout.codec.max_segments + 1) * 32
    assert (
        sum(p.numel() for p in bound.parameters())
        == sum(p.numel() for p in plain.parameters()) + extra
    )
    condition = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }
    with torch.no_grad():
        assert bound(tokens[None], condition).shape == (1, layout.length, layout.vocabulary)


def test_paired_interval_and_t_table() -> None:
    assert _t_critical(1) == 12.706 and _t_critical(30) == 2.042 and _t_critical(200) == 1.96
    assert _t_critical(35) == _t_critical(30), "between rows, the larger (conservative) value"
    clear = _paired(np.array([0.5, 0.6, 0.4, 0.55, 0.45]))
    assert clear["interval_excludes_zero_above"] and clear["positive"] == 5
    noisy = _paired(np.array([0.5, -0.6, 0.4, -0.55, 0.45]))
    assert not noisy["interval_excludes_zero_above"]
    assert not _paired(np.array([1.0]))["interval_excludes_zero_above"]


@pytest.mark.parametrize("task", ["path", "span"])
def test_the_study_runs_end_to_end_on_the_cpu(tmp_path: Path, task: str) -> None:
    """The whole harness - train, select, checkpoint, inpaint, render, summarise."""

    config_path = tmp_path / "study.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "study_version": "test-masked",
                "pilot_config": str(_CONFIG),
                "report_root": str(tmp_path / "report"),
                "checkpoint_root": str(tmp_path / "checkpoint"),
                "data": {
                    "train_icons": 2,
                    "distinct_subgroups": True,
                    "evaluate_on": "train",
                    "evaluation_icons": 2,
                    "inpaint_icons": 2,
                    "task": task,
                    "span_length": 1,
                },
                "model": {
                    "d_model": 32,
                    "heads": 4,
                    "layers": 1,
                    "feedforward": 64,
                    "metric_coordinates": 4,
                    "dropout": 0.1,
                },
                "masks": {"random": 0.2, "path": 0.4, "span": 0.2, "style": 0.1, "geometry": 0.1},
                "training": {
                    "steps": 2,
                    "batch_size": 2,
                    "learning_rate": 0.001,
                    "seed": 5,
                    "eval_every": 1,
                    "patience_evals": 4,
                    "coordinate_tau": 1.0,
                },
                "decoding": {
                    "iterations": 2,
                    "samples": 1,
                    "render_size": 18,
                    "render_timeout_seconds": 20,
                },
                "criteria": {
                    "min_masked_token_accuracy": 0.99,
                    "min_loss_reduction_factor": 100.0,
                    "min_exact_path_reproduction_rate": 1.0,
                    "min_close_path_reproduction_rate": 1.0,
                    "beats_drop_baseline": True,
                    "beats_marginal_baseline": True,
                    "min_median_recovery": 0.3,
                    "require_all_valid": True,
                },
            }
        )
    )
    summary = run_masked_study(load_masked_study_config(config_path), config_path)
    assert summary["train_icons"] == 2 and summary["steps_run"] == 2
    assert summary["checkpoint_round_trip"]
    assert summary["inpainting"]["all_valid"]
    assert summary["inpainting"]["icons"] >= 1
    assert summary["inpainting"]["task"] == task
    assert summary["inpainting"]["identity_policy"] == (
        "join_span" if task == "span" else "drop_path"
    )
    assert (
        0.0
        <= summary["inpainting"]["exact_reproduction_rate"]
        <= summary["inpainting"]["close_reproduction_rate"]
        <= 1.0
    )
    assert set(summary["continuity"]) == {"model", "copy_previous", "marginal"}
    assert summary["continuity"]["copy_previous"]["n"] >= 1
    assert summary["coordinate_tau"] == [1.0]
    assert set(summary["checks"]) == {
        "masked_token_accuracy",
        "loss_reduction",
        "exact_path_reproduction",
        "close_path_reproduction",
        "beats_drop_baseline",
        "beats_marginal_baseline",
        "median_recovery",
        "all_valid",
    }
    assert summary["predeclared_outcome"] == "falsified", "two steps cannot memorise anything"
    sheet = (tmp_path / "report" / "inpainting.png").read_bytes()
    assert sheet[:8] == b"\x89PNG\r\n\x1a\n"
    assert hashlib.sha256(sheet).hexdigest() == summary["inpainting"]["sheet_sha256"]
    written = json.loads((tmp_path / "report" / "summary.json").read_text())
    assert written == summary
    trace = [
        json.loads(line)
        for line in (tmp_path / "report" / "metrics.jsonl").read_text().splitlines()
    ]

    # The same checkpoint evaluated on the other task, without training.
    other = "span" if task == "path" else "path"
    frozen_path = tmp_path / "frozen.yaml"
    frozen = yaml.safe_load(config_path.read_text())
    frozen["study_version"] = "test-masked-frozen"
    frozen["report_root"] = str(tmp_path / "frozen-report")
    frozen["checkpoint_root"] = str(tmp_path / "frozen-checkpoint")
    frozen["data"]["task"] = other
    frozen["training"]["checkpoint"] = str(tmp_path / "checkpoint" / "checkpoint.zip")
    frozen["training"]["checkpoint_sha256"] = summary["checkpoint_sha256"]
    frozen["criteria"] = {"require_all_valid": True}
    frozen_path.write_text(yaml.safe_dump(frozen))
    evaluated = run_masked_study(load_masked_study_config(frozen_path), frozen_path)
    assert evaluated["trained_here"] is False
    assert evaluated["inpainting"]["inpaint_seed"] == 5, "defaults to the training seed"
    pinned = yaml.safe_load(frozen_path.read_text())
    pinned["data"]["inpaint_seed"] = 99
    pinned["report_root"] = str(tmp_path / "pinned-report")
    frozen_path.write_text(yaml.safe_dump(pinned))
    repinned = run_masked_study(load_masked_study_config(frozen_path), frozen_path)
    assert repinned["inpainting"]["inpaint_seed"] == 99
    assert evaluated["selected_step"] == summary["selected_step"]
    assert evaluated["inpainting"]["task"] == other
    assert evaluated["loss_reduction_factor"] is None
    assert evaluated["checks"] == {"all_valid": True}
    wrong = yaml.safe_load(frozen_path.read_text())
    wrong["training"]["checkpoint_sha256"] = "0" * 64
    wrong["report_root"] = str(tmp_path / "wrong-report")
    frozen_path.write_text(yaml.safe_dump(wrong))
    with pytest.raises(Exception, match="hash mismatch"):
        run_masked_study(load_masked_study_config(frozen_path), frozen_path)
    assert [row["step"] for row in trace] == [0, 1, 2], "the untrained model is evaluated first"
    assert (
        summary["loss_reduction_factor"]
        == trace[0]["held_out_nll"] / summary["held_out_nll_per_masked_token"]
    )
    assert all("marginal_nll" in row and "held_out_nll" in row for row in trace)
