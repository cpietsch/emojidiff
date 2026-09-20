from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mojidiff.learning.geometry import (
    GeometryDenoiser,
    corrupt_factorized_geometry,
    corrupt_geometry,
    corrupt_path_correlated_geometry,
    corrupt_whole_path_geometry_from_pool,
    corrupt_whole_path_geometry_replacement,
    geometry_accuracy_by_corruption,
    geometry_loss_and_accuracy,
    packed_batch,
    predict_clean_geometry,
    whole_path_pool_support,
)
from mojidiff.learning.tiny_study import (
    TinyLearningConfig,
    _load_checkpoint,
    _model_hash,
    _new_training,
    _save_checkpoint,
)
from mojidiff.representation.packed import pack_tensor_program, validate_packed_tensor_program
from mojidiff.representation.program import (
    CodecConfig,
    FloatContour,
    FloatProgram,
    FloatSegment,
    SegmentType,
    encode_program,
)


def _codec() -> CodecConfig:
    return CodecConfig(
        max_paths=4,
        max_segments=8,
        coordinate_bins=289,
        control_coordinate_bins=417,
        control_coordinate_min=-8.0,
        control_coordinate_max=96.0,
        palette=("#000000", "#ffffff"),
        stroke_widths=(1.0, 2.0),
        dash_patterns=((2.0, 4.0),),
        miter_limits=(4.0, 10.0),
        opacities=(1.0,),
        max_serialized_bytes=20_000,
    )


def _program() -> FloatProgram:
    return FloatProgram(
        (
            FloatContour(
                layer=1,
                fill="#ffffff",
                stroke=None,
                stroke_width=1.0,
                linecap="butt",
                linejoin="miter",
                miter_limit=4.0,
                dash_pattern=(),
                fill_rule="nonzero",
                opacity=1.0,
                fill_opacity=1.0,
                stroke_opacity=None,
                start=(4.0, 4.0),
                segments=(
                    FloatSegment(SegmentType.LINE, (20.0, 4.0)),
                    FloatSegment(SegmentType.CUBIC, (24.0, 4.0, 24.0, 20.0, 20.0, 20.0)),
                    FloatSegment(SegmentType.LINE, (4.0, 20.0)),
                    FloatSegment(SegmentType.CLOSE, ()),
                ),
            ),
        )
    )


def test_geometry_denoiser_forward_backward_and_legal_prediction() -> None:
    torch.manual_seed(7)
    codec = _codec()
    dense, _ = encode_program(_program(), codec)
    clean = pack_tensor_program(dense, codec, total_segment_slots=16)
    noisy = corrupt_geometry(clean, codec, 0.5, np.random.default_rng(7))
    model = GeometryDenoiser(
        codec,
        16,
        d_model=32,
        heads=4,
        layers=1,
        feedforward=64,
    )
    clean_batch = packed_batch([clean], torch.device("cpu"))
    noisy_batch = packed_batch([noisy], torch.device("cpu"))

    logits = model(noisy_batch)
    loss, counts = geometry_loss_and_accuracy(logits, clean_batch, codec)
    split_counts = geometry_accuracy_by_corruption(logits, noisy_batch, clean_batch, codec)
    loss.backward()  # type: ignore[no-untyped-call]
    prediction = predict_clean_geometry(noisy, logits, codec)

    assert logits[0].shape == (1, 4, 2, 290)
    assert logits[1].shape == (1, 16, 6, 418)
    assert torch.isfinite(loss)
    assert counts["total"] == 12
    assert split_counts["changed"]["total"] + split_counts["retained"]["total"] == 12
    assert any(parameter.grad is not None for parameter in model.parameters())
    validate_packed_tensor_program(prediction, codec, 16)


def test_fixed_topology_corruption_is_seeded_and_preserves_typed_padding() -> None:
    codec = _codec()
    dense, _ = encode_program(_program(), codec)
    clean = pack_tensor_program(dense, codec, total_segment_slots=16)

    first = corrupt_geometry(clean, codec, 1.0, np.random.default_rng(11))
    second = corrupt_geometry(clean, codec, 1.0, np.random.default_rng(11))

    for field in clean.__dataclass_fields__:
        assert np.array_equal(getattr(first, field), getattr(second, field))
    assert np.array_equal(first.path_length, clean.path_length)
    assert np.array_equal(first.segment_type, clean.segment_type)
    assert np.all(first.segment_type[4:] == 0)
    assert np.all(first.coordinates[4:] == 0)
    validate_packed_tensor_program(first, codec, 16)


def test_factorized_and_path_correlated_contracts_preserve_grammar() -> None:
    codec = _codec()
    dense, _ = encode_program(_program(), codec)
    clean = pack_tensor_program(dense, codec, total_segment_slots=16)
    factorized = corrupt_factorized_geometry(clean, codec, 1.0, np.random.default_rng(31))
    correlated = corrupt_path_correlated_geometry(clean, codec, 1.0, np.random.default_rng(31))

    for noisy in (factorized, correlated):
        assert np.array_equal(noisy.path_length, clean.path_length)
        assert np.array_equal(noisy.segment_type, clean.segment_type)
        assert np.all(noisy.start[0] != clean.start[0])
        for index, kind in enumerate(clean.segment_type[:4]):
            coordinate_count = {1: 2, 2: 4, 3: 6, 4: 0}[int(kind)]
            assert np.all(
                noisy.coordinates[index, :coordinate_count]
                != clean.coordinates[index, :coordinate_count]
            )
        assert np.all(noisy.segment_type[4:] == 0)
        assert np.all(noisy.coordinates[4:] == 0)
        validate_packed_tensor_program(noisy, codec, 16)


def test_factorized_corruption_and_prediction_preserve_locked_paths() -> None:
    codec = _codec()
    dense, _ = encode_program(_program(), codec)
    clean = pack_tensor_program(dense, codec, total_segment_slots=16)
    locks = np.zeros((codec.max_paths,), dtype=np.bool_)
    locks[0] = True
    noisy = corrupt_factorized_geometry(
        clean,
        codec,
        1.0,
        np.random.default_rng(37),
        locked_paths=locks,
    )
    model = GeometryDenoiser(
        codec,
        16,
        d_model=32,
        heads=4,
        layers=1,
        feedforward=64,
    )
    prediction = predict_clean_geometry(
        noisy,
        model(packed_batch([noisy], torch.device("cpu"))),
        codec,
        locked_paths=locks,
    )

    assert np.array_equal(noisy.start[0], clean.start[0])
    assert np.array_equal(noisy.coordinates[:4], clean.coordinates[:4])
    assert np.array_equal(prediction.start[0], clean.start[0])
    assert np.array_equal(prediction.coordinates[:4], clean.coordinates[:4])
    validate_packed_tensor_program(prediction, codec, 16)


def test_structured_conditioning_requires_and_accepts_group_tokens() -> None:
    codec = _codec()
    dense, _ = encode_program(_program(), codec)
    clean = pack_tensor_program(dense, codec, total_segment_slots=16)
    batch = packed_batch([clean], torch.device("cpu"))
    model = GeometryDenoiser(
        codec,
        16,
        d_model=32,
        heads=4,
        layers=1,
        feedforward=64,
        group_vocab_size=3,
        subgroup_vocab_size=5,
    )

    with np.testing.assert_raises(ValueError):
        model(batch)
    logits = model(
        batch,
        {
            "group": torch.tensor([1], dtype=torch.long),
            "subgroup": torch.tensor([2], dtype=torch.long),
        },
    )

    assert logits[0].shape == (1, 4, 2, 290)
    assert logits[1].shape == (1, 16, 6, 418)


def test_path_correlated_gate_changes_complete_paths_or_none() -> None:
    codec = _codec()
    dense, _ = encode_program(_program(), codec)
    clean = pack_tensor_program(dense, codec, total_segment_slots=16)

    retained = corrupt_path_correlated_geometry(clean, codec, 0.0, np.random.default_rng(13))
    changed = corrupt_path_correlated_geometry(clean, codec, 1.0, np.random.default_rng(13))

    assert np.array_equal(retained.start, clean.start)
    assert np.array_equal(retained.coordinates, clean.coordinates)
    assert np.all(changed.start[0] != clean.start[0])
    for index, kind in enumerate(clean.segment_type[:4]):
        coordinate_count = {1: 2, 2: 4, 3: 6, 4: 0}[int(kind)]
        assert np.all(
            changed.coordinates[index, :coordinate_count]
            != clean.coordinates[index, :coordinate_count]
        )


def test_whole_path_replacement_copies_only_compatible_legal_geometry() -> None:
    codec = _codec()
    dense, _ = encode_program(_program(), codec)
    clean = pack_tensor_program(dense, codec, total_segment_slots=16)
    donor = corrupt_factorized_geometry(clean, codec, 1.0, np.random.default_rng(41))
    replaced = corrupt_whole_path_geometry_replacement(
        clean, donor, codec, 1.0, np.random.default_rng(42)
    )

    assert np.array_equal(replaced.start, donor.start)
    assert np.array_equal(replaced.coordinates, donor.coordinates)
    assert np.array_equal(replaced.path_length, clean.path_length)
    assert np.array_equal(replaced.segment_type, clean.segment_type)
    validate_packed_tensor_program(replaced, codec, 16)


def test_whole_path_pool_uses_compatible_paths_across_different_slots() -> None:
    codec = _codec()
    dense, _ = encode_program(_program(), codec)
    clean = pack_tensor_program(dense, codec, total_segment_slots=16)
    donor = corrupt_factorized_geometry(clean, codec, 1.0, np.random.default_rng(43))
    replaced = corrupt_whole_path_geometry_from_pool(
        clean, [clean, donor], codec, 1.0, np.random.default_rng(44)
    )

    assert np.array_equal(replaced.start, donor.start)
    assert np.array_equal(replaced.coordinates, donor.coordinates)
    assert whole_path_pool_support([clean, donor]) == {
        "active_paths": 2,
        "eligible_paths": 2,
        "geometry_fields": 24,
        "eligible_geometry_fields": 24,
    }
    validate_packed_tensor_program(replaced, codec, 16)


def test_checkpoint_is_byte_stable_and_restores_training_state() -> None:
    codec = _codec()
    config = TinyLearningConfig(
        version="test",
        source_revision="revision",
        raw_root=Path("raw"),
        fixture=Path("fixture"),
        fixture_sha256="0" * 64,
        palette_path=Path("palette"),
        palette_sha256="0" * 64,
        style_summary=Path("style"),
        style_summary_sha256="0" * 64,
        style_candidate="candidate",
        report_root=Path("report"),
        derived_root=Path("derived"),
        max_paths=codec.max_paths,
        max_segments=codec.max_segments,
        total_segment_slots=16,
        d_model=32,
        heads=4,
        layers=1,
        feedforward=64,
        learning_rate=0.001,
        corruption_probability=0.5,
        cases=(),
        render_sizes=(18,),
        render_timeout_seconds=1,
    )
    model, optimizer = _new_training(config, codec, seed=19)
    dense, _ = encode_program(_program(), codec)
    clean = pack_tensor_program(dense, codec, total_segment_slots=16)
    batch = packed_batch([clean], torch.device("cpu"))
    loss, _ = geometry_loss_and_accuracy(model(batch), batch, codec)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()

    first = _save_checkpoint(model, optimizer, step=1)
    second = _save_checkpoint(model, optimizer, step=1)
    restored_model, restored_optimizer, step = _load_checkpoint(first, config, codec)

    assert first == second
    assert first.startswith(b"PK")
    assert step == 1
    assert _model_hash(restored_model) == _model_hash(model)
    assert _save_checkpoint(restored_model, restored_optimizer, step) == first


def test_slot_binding_is_what_lets_the_encoder_distinguish_coordinate_slots() -> None:
    """A segment's six coordinates must not collapse into an unordered bag.

    The encoder sums six lookups from one shared embedding table into a single vector
    per segment slot. Without slot binding a program and its coordinate-swapped variant
    have byte-identical representations, so the model provably cannot tell which
    coordinate held which value while still having to predict all six separately -
    which is why it could not copy its own input. This pins both halves: the defect the
    unbound path still has, and that binding is what removes it.
    """

    import copy

    import torch

    from mojidiff.learning.geometry import GeometryDenoiser, packed_batch
    from mojidiff.learning.openmoji_pilot import (
        _load_program,
        _select_rows,
        _selected_codec,
        load_openmoji_pilot_config,
        load_pilot_index,
    )

    config = load_openmoji_pilot_config(
        Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    )
    codec = _selected_codec(config)
    by_split, _, _ = load_pilot_index(config)
    row = _select_rows(by_split["primary/validation"], 1, config.seed + 1)[0]
    program = _load_program(row, config, codec)

    swapped = copy.deepcopy(program)
    changed = 0
    for index in range(swapped.coordinates.shape[0]):
        first, second = int(swapped.coordinates[index, 0]), int(swapped.coordinates[index, 1])
        if first != second and first != 0 and second != 0:
            swapped.coordinates[index, 0], swapped.coordinates[index, 1] = second, first
            changed += 1
    assert changed > 0, "fixture must contain segments with two distinct coordinates"

    device = torch.device("cpu")
    results = {}
    for binding in (False, True):
        torch.manual_seed(0)
        model = (
            GeometryDenoiser(
                codec,
                config.total_segment_slots,
                d_model=32,
                heads=4,
                layers=1,
                feedforward=64,
                slot_binding=binding,
            )
            .eval()
            .double()
        )
        with torch.no_grad():
            _, original = model(packed_batch([program], device))
            _, permuted = model(packed_batch([swapped], device))
        # float64, so neither outcome can rest on summation-order noise.
        results[binding] = torch.equal(original, permuted)

    assert results[False] is True, "the unbound encoder is permutation-invariant by construction"
    assert results[True] is False, "slot binding must make the encoder slot-aware"


def test_edit_mask_makes_the_identity_policy_exactly_representable() -> None:
    """Predicting keep everywhere must reproduce the input byte for byte.

    The identity policy outscores every trained Gate G model, and without an edit mask
    the model can only express it by reconstructing all 26,030 uncorrupted tokens
    through a 289- or 417-way softmax. This pins that the gated decode makes it free.
    """

    import numpy as np
    import torch

    from mojidiff.learning.geometry import (
        corrupt_factorized_geometry,
        edit_mask_predictions,
        predict_clean_geometry,
    )
    from mojidiff.learning.openmoji_pilot import (
        _load_program,
        _select_rows,
        _selected_codec,
        load_openmoji_pilot_config,
        load_pilot_index,
    )

    config = load_openmoji_pilot_config(
        Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    )
    codec = _selected_codec(config)
    by_split, _, _ = load_pilot_index(config)
    row = _select_rows(by_split["primary/validation"], 1, config.seed + 1)[0]
    clean = _load_program(row, config, codec)
    noisy = corrupt_factorized_geometry(clean, codec, 0.35, np.random.default_rng(7))
    assert not np.array_equal(noisy.coordinates, clean.coordinates)

    # Value logits that would overwrite everything with token 1, and a keep decision
    # that says keep everywhere. The keep decision must win.
    start_logits = torch.zeros(1, codec.max_paths, 2, codec.coordinate_bins + 1)
    coordinate_logits = torch.zeros(
        1, config.total_segment_slots, 6, codec.effective_control_coordinate_bins + 1
    )
    keep = (
        torch.tensor([0.0, 1.0]).expand(1, codec.max_paths, 2, 2).contiguous(),
        torch.tensor([0.0, 1.0]).expand(1, config.total_segment_slots, 6, 2).contiguous(),
    )

    identity = predict_clean_geometry(noisy, (start_logits, coordinate_logits), codec, keep=keep)

    assert np.array_equal(identity.coordinates, noisy.coordinates)
    assert np.array_equal(identity.start, noisy.start)
    # And the field-level helper agrees with the program-level decode.
    values = torch.zeros(4, 10)
    tokens = torch.tensor([3, 5, 7, 9])
    assert torch.equal(
        edit_mask_predictions(values, torch.tensor([[0.0, 1.0]]).expand(4, 2), tokens), tokens
    )
    assert torch.equal(
        edit_mask_predictions(values, torch.tensor([[1.0, 0.0]]).expand(4, 2), tokens),
        torch.ones(4, dtype=tokens.dtype),
    )


def test_the_continuity_statistic_separates_a_planted_outlier() -> None:
    """The detectability probe must actually detect a break in continuity.

    Gate G's conclusion turns on whether corrupted fields are identifiable at all, so
    the instrument used to answer that has to be shown to work on a case where the
    answer is known.
    """

    import copy

    import numpy as np
    import pytest

    from mojidiff.learning.detectability import roc_auc, separation
    from mojidiff.learning.openmoji_pilot import (
        _load_program,
        _select_rows,
        _selected_codec,
        load_openmoji_pilot_config,
        load_pilot_index,
    )

    config = load_openmoji_pilot_config(
        Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    )
    codec = _selected_codec(config)
    by_split, _, _ = load_pilot_index(config)
    row = _select_rows(by_split["primary/validation"], 1, config.seed + 1)[0]
    clean = _load_program(row, config, codec)

    # An identical program has no corrupted fields, so there is nothing to separate.
    corrupted, retained = separation(clean, clean, codec)
    assert corrupted == []
    assert retained

    # Plant far-away values in a handful of fields; a continuity statistic must rank
    # them above the untouched ones.
    planted = copy.deepcopy(clean)
    changed = 0
    # Every fifth segment, so a planted field always has legitimate neighbours. Planting
    # into adjacent segments would make the outliers smooth with respect to each other.
    for segment_index in range(0, planted.coordinates.shape[0], 5):
        token = int(planted.coordinates[segment_index, 0])
        if token > 0 and changed < 12:
            planted.coordinates[segment_index, 0] = 1 if token > 144 else codec.coordinate_bins
            changed += 1
    assert changed == 12

    corrupted, retained = separation(clean, planted, codec)
    assert len(corrupted) == 12
    assert roc_auc(corrupted, retained) > 0.9

    # A statistic with no signal scores 0.5, including when every value ties.
    assert roc_auc([1.0, 1.0], [1.0, 1.0]) == pytest.approx(0.5)
    assert roc_auc([2.0, 3.0], [0.0, 1.0]) == pytest.approx(1.0)
    assert np.isnan(roc_auc([], [1.0]))
