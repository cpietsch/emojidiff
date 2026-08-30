from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mojidiff.learning.geometry import (
    GeometryDenoiser,
    corrupt_geometry,
    geometry_loss_and_accuracy,
    packed_batch,
    predict_clean_geometry,
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
    loss.backward()  # type: ignore[no-untyped-call]
    prediction = predict_clean_geometry(noisy, logits, codec)

    assert logits[0].shape == (1, 4, 2, 290)
    assert logits[1].shape == (1, 16, 6, 418)
    assert torch.isfinite(loss)
    assert counts["total"] == 12
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
