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


def test_the_detector_reference_learns_from_local_features_alone() -> None:
    """The cheap-feature reference must beat chance, and must never see the clean program.

    It exists to bound how much detection headroom there is above the zero-parameter
    continuity statistic, so if it silently had access to the answer its numbers would
    be meaningless.
    """

    import inspect

    import numpy as np

    from mojidiff.learning import detector_reference
    from mojidiff.learning.detectability import roc_auc
    from mojidiff.learning.detector_reference import (
        FEATURE_NAMES,
        features_and_labels,
        fit_logistic,
        score,
    )
    from mojidiff.learning.geometry import corrupt_factorized_geometry
    from mojidiff.learning.openmoji_pilot import (
        _load_program,
        _select_rows,
        _selected_codec,
        load_openmoji_pilot_config,
        load_pilot_index,
    )

    # Features are read from the corrupted program only; `clean` is used solely for the
    # label. Pin that structurally rather than trusting the comment.
    source = inspect.getsource(detector_reference.features_and_labels)
    body = source.split("offset = 0", 1)[1]
    assert "clean." not in body.replace("clean.coordinates[segment_index, slot])", "")

    config = load_openmoji_pilot_config(
        Path("configs/learning/openmoji-g1-dominant-bucket-train-v2-data-scale.yaml")
    )
    codec = _selected_codec(config)
    by_split, _, _ = load_pilot_index(config)
    rows = _select_rows(by_split["primary/validation"], 16, config.seed + 1)

    features, labels = [], []
    for index, row in enumerate(rows):
        clean = _load_program(row, config, codec)
        noisy = corrupt_factorized_geometry(
            clean, codec, 0.35, np.random.default_rng(config.seed + 9_000_000 + index)
        )
        f, y = features_and_labels(clean, noisy, codec)
        features.append(f)
        labels.append(y)
    x = np.concatenate(features)
    y = np.concatenate(labels)
    assert x.shape[1] == len(FEATURE_NAMES)
    assert 0.2 < y.mean() < 0.5

    weight, bias, norm = fit_logistic(x, y, steps=100)
    s = score(x, weight, bias, norm)
    assert roc_auc(list(s[y == 1]), list(s[y == 0])) > 0.65


def test_group_pooled_loss_lets_four_fields_dominate_and_field_pooling_fixes_it() -> None:
    """QUAD is 0.93% of the corpus, so its groups hold one field and carry 4/13 of the loss.

    `_legal_fields` emits one group per (segment kind, coordinate slot), and the default
    loss averages over groups. On the held-out draw that gives four single-field QUAD
    groups the same weight as six 5,287-field CUBIC groups. Because the evaluation path
    uses the same function, held-out loss - the checkpoint-selection and early-stopping
    signal - inherits it.
    """

    import numpy as np
    import torch

    from mojidiff.learning.geometry import (
        _legal_fields,
        corrupt_factorized_geometry,
        geometry_loss_and_accuracy,
        packed_batch,
    )
    from mojidiff.learning.openmoji_pilot import (
        _load_program,
        _select_rows,
        _selected_codec,
        load_openmoji_pilot_config,
        load_pilot_index,
    )

    config = load_openmoji_pilot_config(
        Path("configs/learning/openmoji-g1-dominant-bucket-train-v2-data-scale.yaml")
    )
    codec = _selected_codec(config)
    by_split, _, _ = load_pilot_index(config)
    rows = _select_rows(by_split["primary/validation"], 24, config.seed + 1)
    clean = [_load_program(row, config, codec) for row in rows]
    noisy = [
        corrupt_factorized_geometry(
            program, codec, 0.35, np.random.default_rng(config.seed + 9_000_000 + index)
        )
        for index, program in enumerate(clean)
    ]
    device = torch.device("cpu")
    noisy_batch, clean_batch = packed_batch(noisy, device), packed_batch(clean, device)

    sizes = [
        int(targets.numel())
        for _, _, targets, _ in _legal_fields(
            (
                torch.zeros(*clean_batch["start"].shape, codec.coordinate_bins + 1),
                torch.zeros(
                    *clean_batch["coordinates"].shape,
                    codec.effective_control_coordinate_bins + 1,
                ),
            ),
            None,
            noisy_batch,
            clean_batch,
            codec,
        )
    ]
    # The defect in one line: the smallest group is orders of magnitude smaller than the
    # largest, yet the group average gives them identical weight.
    assert max(sizes) / min(sizes) > 100

    torch.manual_seed(0)
    logits = (
        torch.randn(*clean_batch["start"].shape, codec.coordinate_bins + 1),
        torch.randn(
            *clean_batch["coordinates"].shape, codec.effective_control_coordinate_bins + 1
        ),
    )
    grouped, counts_grouped = geometry_loss_and_accuracy(logits, clean_batch, codec)
    pooled, counts_pooled = geometry_loss_and_accuracy(
        logits, clean_batch, codec, pool_over_fields=True
    )
    # Accuracy counting is untouched; only the weighting changes.
    assert counts_grouped == counts_pooled
    assert torch.isfinite(grouped) and torch.isfinite(pooled)
    assert not torch.allclose(grouped, pooled)


def test_padding_is_hidden_from_attention_only_when_asked() -> None:
    """Typed padding is about 29% of the sequence and was always attended to."""

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
    batch = packed_batch([_load_program(row, config, codec)], torch.device("cpu"))
    padding = int((batch["path_length"] == 0).sum() + (batch["segment_type"] == 0).sum())
    assert padding > 0, "fixture must contain typed padding"

    outputs = {}
    for mask_padding in (False, True):
        torch.manual_seed(0)
        model = GeometryDenoiser(
            codec,
            config.total_segment_slots,
            d_model=32,
            heads=4,
            layers=1,
            feedforward=64,
            mask_padding=mask_padding,
        ).eval()
        with torch.no_grad():
            _, coordinates = model(batch)
        # No NaN even though most of the sequence is masked out.
        assert torch.isfinite(coordinates).all()
        outputs[mask_padding] = coordinates

    assert not torch.allclose(outputs[False], outputs[True])


def test_marginal_corruption_closes_the_density_leak() -> None:
    """Uniform replacement makes corrupted tokens detectable without any geometry.

    Corruption draws from the full legal vocabulary, so many replacements land on tokens
    real icons never use, and a detector knowing only the corpus marginal reaches 2.84
    precision lift with no context at all. Marginal-respecting replacement removes that
    shortcut while leaving the genuine local-continuity signal in place.
    """

    import numpy as np

    from mojidiff.learning.detectability import roc_auc, separation
    from mojidiff.learning.geometry import (
        corrupt_factorized_geometry,
        corrupt_marginal_geometry,
        role_token_marginals,
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
        validate_packed_tensor_program,
    )

    config = load_openmoji_pilot_config(
        Path("configs/learning/openmoji-g1-dominant-bucket-train-v2-data-scale.yaml")
    )
    codec = _selected_codec(config)
    by_split, _, _ = load_pilot_index(config)
    train = [
        _load_program(row, config, codec)
        for row in _select_rows(by_split["primary/train"], 64, config.seed)
    ]
    marginal = role_token_marginals(train, codec)
    assert marginal.shape == (5, 6, codec.effective_control_coordinate_bins + 1)
    # Every distribution either sums to one or is an unused (kind, slot) pair.
    totals = marginal.sum(axis=2)
    assert np.all((np.isclose(totals, 1.0)) | np.isclose(totals, 0.0))

    clean = [
        _load_program(row, config, codec)
        for row in _select_rows(by_split["primary/validation"], 12, config.seed + 1)
    ]
    logp = np.log(np.clip(marginal, 1e-12, None))

    def density_and_continuity(
        noisy_programs: list[PackedTensorProgram],
    ) -> tuple[float, float]:
        scores, labels = [], []
        for reference, noisy in zip(clean, noisy_programs, strict=True):
            validate_packed_tensor_program(noisy, codec, config.total_segment_slots)
            for segment in range(noisy.coordinates.shape[0]):
                kind = int(noisy.segment_type[segment])
                if kind not in (1, 2, 3):
                    continue
                for slot in range(6):
                    token = int(noisy.coordinates[segment, slot])
                    if token <= 0:
                        continue
                    scores.append(-logp[kind, slot, token])
                    labels.append(int(token != int(reference.coordinates[segment, slot])))
        density = roc_auc(
            [s for s, y in zip(scores, labels, strict=True) if y],
            [s for s, y in zip(scores, labels, strict=True) if not y],
        )
        positive, negative = [], []
        for reference, noisy in zip(clean, noisy_programs, strict=True):
            a, b = separation(reference, noisy, codec)
            positive += a
            negative += b
        return density, roc_auc(positive, negative)

    uniform_density, uniform_continuity = density_and_continuity(
        [
            corrupt_factorized_geometry(
                p, codec, 0.35, np.random.default_rng(config.seed + 9_000_000 + i)
            )
            for i, p in enumerate(clean)
        ]
    )
    marginal_density, marginal_continuity = density_and_continuity(
        [
            corrupt_marginal_geometry(
                p, codec, 0.35, np.random.default_rng(config.seed + 9_000_000 + i), marginal
            )
            for i, p in enumerate(clean)
        ]
    )

    # The leak: uniform replacement is detectable from the marginal alone.
    assert uniform_density > 0.70
    # Closed: knowing the marginal no longer identifies corrupted fields.
    assert marginal_density < 0.60
    # But the geometric signal is still there to be learned.
    assert marginal_continuity > 0.65
    assert uniform_continuity > 0.65


def test_metric_coordinates_encode_magnitude_and_shrink_the_model() -> None:
    """Under metric encoding, nearby lattice bins must produce nearby inputs.

    A token table gives the encoder no way to subtract two coordinates: on a trained
    checkpoint the Spearman between embedding distance and bin distance is -0.136 and
    adjacent bins sit 0.989 of the all-pairs mean apart, i.e. no order at all. The
    metric path decodes each token to its view-unit value instead, so proximity in the
    lattice becomes proximity in the input by construction.
    """

    import torch

    from mojidiff.learning.geometry import GeometryDenoiser
    from mojidiff.learning.openmoji_pilot import (
        _selected_codec,
        load_openmoji_pilot_config,
    )

    config = load_openmoji_pilot_config(
        Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    )
    codec = _selected_codec(config)
    torch.manual_seed(0)
    metric = GeometryDenoiser(
        codec, config.total_segment_slots, d_model=96, heads=4, layers=2, feedforward=192,
        metric_coordinates=8,
    )
    torch.manual_seed(0)
    categorical = GeometryDenoiser(
        codec, config.total_segment_slots, d_model=96, heads=4, layers=2, feedforward=192,
    )

    # Strictly fewer parameters, so this cannot be confounded with a capacity change.
    metric_count = sum(p.numel() for p in metric.parameters())
    categorical_count = sum(p.numel() for p in categorical.parameters())
    assert metric_count < categorical_count
    assert metric.coordinate_embedding is None and categorical.coordinate_embedding is not None

    # One CUBIC segment whose slot 5 (an endpoint) takes three values: 100, 102, 280.
    def features_for(token: int) -> torch.Tensor:
        tokens = torch.zeros(1, config.total_segment_slots, 6, dtype=torch.long)
        tokens[0, 0, :] = 100
        tokens[0, 0, 5] = token
        kinds = torch.zeros(1, config.total_segment_slots, dtype=torch.long)
        kinds[0, 0] = 3
        return metric._metric_features(tokens, kinds)[0, 0]

    near = (features_for(100) - features_for(102)).norm()
    far = (features_for(100) - features_for(280)).norm()
    assert near < far, "adjacent lattice bins must be closer than distant ones"

    # A control slot and an endpoint slot decode with different affine maps, so the
    # same token means different values in the two roles.
    tokens = torch.zeros(1, config.total_segment_slots, 6, dtype=torch.long)
    tokens[0, 0, :] = 100
    cubic = torch.zeros(1, config.total_segment_slots, dtype=torch.long)
    cubic[0, 0] = 3
    per_slot = metric._metric_features(tokens, cubic)[0, 0].reshape(6, -1)
    assert not torch.allclose(per_slot[0], per_slot[5])


def test_the_break_even_threshold_matches_its_decision_rule() -> None:
    """Changing a field pays only above 1/(1+q) in the value head's accuracy.

    Keeping a retained field is always right; changing one is right only if the value
    head re-predicts the same token, which is negligible. The threshold is therefore
    derivable from a training-split estimate of q rather than swept on the data it is
    reported on - which is what made the first identity win require a post-hoc sweep.
    """

    import pytest
    import torch

    from mojidiff.learning.geometry import break_even_threshold, edit_mask_predictions

    assert break_even_threshold(0.0) == 1.0
    assert break_even_threshold(1.0) == pytest.approx(0.5)
    assert break_even_threshold(0.1233) == pytest.approx(1 / 1.1233)
    with pytest.raises(ValueError):
        break_even_threshold(1.5)

    # A field the model calls "change" with probability 0.8 is edited under argmax but
    # kept under a threshold derived from a weak value head.
    keep_logits = torch.log(torch.tensor([[0.8, 0.2]]))
    values = torch.zeros(1, 10)
    values[0, 4] = 5.0
    tokens = torch.tensor([9])
    assert edit_mask_predictions(values, keep_logits, tokens).item() == 5
    assert edit_mask_predictions(values, keep_logits, tokens, 0.5).item() == 5
    cautious = break_even_threshold(0.12)
    assert edit_mask_predictions(values, keep_logits, tokens, cautious).item() == 9

    # Confident enough, and it edits again.
    confident = torch.log(torch.tensor([[0.95, 0.05]]))
    assert edit_mask_predictions(values, confident, tokens, cautious).item() == 5
