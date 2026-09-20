"""Bounded Gate G pilot over the dominant exact OpenMoji packed bucket.

This pilot deliberately keeps topology and style fixed.  It scales the selected
factorized geometry denoiser from four icons to family-disjoint OpenMoji train and
validation rows, adds structured group/subgroup conditioning, and enforces exact
locked-path editing.  It is not an unconditional-generation result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import torch
import yaml
from picosvg.svg import SVG

from mojidiff.learning.geometry import (
    GeometryDenoiser,
    corrupt_factorized_geometry,
    geometry_accuracy_by_corruption,
    geometry_loss_and_accuracy,
    packed_batch,
    predict_clean_geometry,
)
from mojidiff.learning.tiny_study import (
    _decode_checkpoint,
    _model_hash,
    _save_checkpoint,
)
from mojidiff.representation.codec_study import _palette, _write_bytes_artifact
from mojidiff.representation.normalizer import normalize_svg
from mojidiff.representation.packed import (
    PackedTensorProgram,
    pack_tensor_program,
    validate_packed_tensor_program,
)
from mojidiff.representation.program import CodecConfig, encode_program


class OpenMojiPilotError(RuntimeError):
    """The bounded Gate G pilot is invalid or failed closed."""


@dataclass(frozen=True)
class PilotRow:
    source_path: str
    source_svg_sha256: str
    hexcode: str
    split: str
    variant_family_id: str
    group: str
    subgroup: str
    selected_representation: str
    bucket: str


@dataclass(frozen=True)
class OpenMojiPilotConfig:
    version: str
    source_revision: str
    raw_root: Path
    hybrid: Path
    hybrid_sha256: str
    assignments: Path
    assignments_sha256: str
    palette: Path
    palette_sha256: str
    style_summary: Path
    style_summary_sha256: str
    style_candidate: str
    report_root: Path
    checkpoint_root: Path
    expected_icons: int
    bucket: str
    expected_bucket_icons: int
    max_paths: int
    max_segments: int
    total_segment_slots: int
    d_model: int
    heads: int
    layers: int
    feedforward: int
    train_samples: int
    validation_samples: int
    batch_size: int
    steps: int
    learning_rate: float
    corruption_probability: float
    seed: int
    device: str


def load_openmoji_pilot_config(path: Path) -> OpenMojiPilotConfig:
    root = _mapping(yaml.safe_load(path.read_bytes()), "root")
    if root.get("schema_version") != 1:
        raise OpenMojiPilotError("openmoji pilot schema_version must be 1")
    inputs = _mapping(root.get("inputs"), "inputs")
    selection = _mapping(root.get("selection"), "selection")
    codec = _mapping(root.get("codec"), "codec")
    model = _mapping(root.get("model"), "model")
    training = _mapping(root.get("training"), "training")
    result = OpenMojiPilotConfig(
        version=_string(root, "study_version"),
        source_revision=_git_revision(_string(root, "source_revision")),
        raw_root=Path(_string(inputs, "raw_root")),
        hybrid=Path(_string(inputs, "hybrid")),
        hybrid_sha256=_sha256(_string(inputs, "hybrid_sha256")),
        assignments=Path(_string(inputs, "assignments")),
        assignments_sha256=_sha256(_string(inputs, "assignments_sha256")),
        palette=Path(_string(inputs, "palette")),
        palette_sha256=_sha256(_string(inputs, "palette_sha256")),
        style_summary=Path(_string(inputs, "style_summary")),
        style_summary_sha256=_sha256(_string(inputs, "style_summary_sha256")),
        style_candidate=_string(inputs, "style_candidate"),
        report_root=Path(_string(root, "report_root")),
        checkpoint_root=Path(_string(root, "checkpoint_root")),
        expected_icons=_positive_int(inputs.get("expected_icons"), "expected_icons"),
        bucket=_string(selection, "bucket"),
        expected_bucket_icons=_positive_int(
            selection.get("expected_bucket_icons"), "expected_bucket_icons"
        ),
        max_paths=_positive_int(codec.get("max_paths"), "max_paths"),
        max_segments=_positive_int(codec.get("max_segments"), "max_segments"),
        total_segment_slots=_positive_int(
            codec.get("total_segment_slots"), "total_segment_slots"
        ),
        d_model=_positive_int(model.get("d_model"), "d_model"),
        heads=_positive_int(model.get("heads"), "heads"),
        layers=_positive_int(model.get("layers"), "layers"),
        feedforward=_positive_int(model.get("feedforward"), "feedforward"),
        train_samples=_positive_int(training.get("train_samples"), "train_samples"),
        validation_samples=_positive_int(
            training.get("validation_samples"), "validation_samples"
        ),
        batch_size=_positive_int(training.get("batch_size"), "batch_size"),
        steps=_positive_int(training.get("steps"), "steps"),
        learning_rate=_positive_float(training.get("learning_rate"), "learning_rate"),
        corruption_probability=_probability(
            training.get("corruption_probability"), "corruption_probability"
        ),
        seed=_nonnegative_int(training.get("seed"), "seed"),
        device=_string(training, "device"),
    )
    if result.d_model % result.heads:
        raise OpenMojiPilotError("model.d_model must be divisible by model.heads")
    if result.batch_size > result.train_samples:
        raise OpenMojiPilotError("batch_size cannot exceed train_samples")
    if result.device not in {"auto", "cpu", "cuda"}:
        raise OpenMojiPilotError("training.device must be auto, cpu, or cuda")
    return result


def load_pilot_index(
    config: OpenMojiPilotConfig,
) -> tuple[dict[str, tuple[PilotRow, ...]], dict[str, int], dict[str, int]]:
    """Verify and join the immutable hybrid and capacity-assignment ledgers."""

    _verify(config.hybrid, config.hybrid_sha256, "hybrid ledger")
    _verify(config.assignments, config.assignments_sha256, "bucket assignments")
    hybrid_rows = _jsonl(config.hybrid)
    assignment_rows = _jsonl(config.assignments)
    if len(hybrid_rows) != config.expected_icons or len(assignment_rows) != config.expected_icons:
        raise OpenMojiPilotError("input ledgers do not have the expected complete-corpus size")

    assignments: dict[str, str] = {}
    for item in assignment_rows:
        source_path = _safe_source_path(_string(item, "source_path"))
        if source_path in assignments:
            raise OpenMojiPilotError(f"duplicate assignment: {source_path}")
        assignments[source_path] = _string(item, "bucket")

    rows: list[PilotRow] = []
    seen: set[str] = set()
    family_splits: dict[str, set[str]] = {}
    for item in hybrid_rows:
        if item.get("source_revision") != config.source_revision:
            raise OpenMojiPilotError("hybrid source revision mismatch")
        source_path = _safe_source_path(_string(item, "source_path"))
        if source_path in seen or source_path not in assignments:
            raise OpenMojiPilotError(f"hybrid/assignment join is not one-to-one: {source_path}")
        seen.add(source_path)
        split = _string(item, "split")
        if split not in {"primary/train", "primary/validation", "primary/test"}:
            raise OpenMojiPilotError(f"unexpected primary split: {split}")
        family = _string(item, "split_family_cluster")
        family_splits.setdefault(family, set()).add(split)
        representation = _string(item, "selected_representation")
        if representation not in {"semantic", "outlined"}:
            raise OpenMojiPilotError(f"unsupported selected representation: {representation}")
        rows.append(
            PilotRow(
                source_path=source_path,
                source_svg_sha256=_sha256(_string(item, "source_svg_sha256")),
                hexcode=_string(item, "hexcode"),
                split=split,
                variant_family_id=family,
                group=_string(item, "group"),
                subgroup=_string(item, "subgroup"),
                selected_representation=representation,
                bucket=assignments[source_path],
            )
        )
    if set(assignments) != seen:
        raise OpenMojiPilotError("bucket assignments contain rows absent from the hybrid ledger")
    if any(len(splits) != 1 for splits in family_splits.values()):
        raise OpenMojiPilotError("variant-family leakage crosses primary splits")

    selected = [row for row in rows if row.bucket == config.bucket]
    if len(selected) != config.expected_bucket_icons:
        raise OpenMojiPilotError("selected packed bucket count does not match its pinned census")
    groups = {name: index + 1 for index, name in enumerate(sorted({row.group for row in rows}))}
    subgroups = {
        name: index + 1 for index, name in enumerate(sorted({row.subgroup for row in rows}))
    }
    by_split = {
        split: tuple(sorted((row for row in selected if row.split == split), key=_row_key))
        for split in ("primary/train", "primary/validation", "primary/test")
    }
    return by_split, groups, subgroups


def run_openmoji_pilot(config: OpenMojiPilotConfig, config_path: Path) -> dict[str, Any]:
    """Run a bounded one- or few-step pipeline smoke and write compact evidence."""

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    _verify(config.palette, config.palette_sha256, "palette")
    _verify(config.style_summary, config.style_summary_sha256, "style summary")
    by_split, groups, subgroups = load_pilot_index(config)
    train_rows = _select_rows(by_split["primary/train"], config.train_samples, config.seed)
    validation_rows = _select_rows(
        by_split["primary/validation"], config.validation_samples, config.seed + 1
    )
    codec = _selected_codec(config)
    train_programs = [_load_program(row, config, codec) for row in train_rows]
    validation_programs = [_load_program(row, config, codec) for row in validation_rows]
    device = _device(config.device)
    model = _new_model(config, codec, groups, subgroups).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    metrics: list[dict[str, Any]] = []
    for step in range(1, config.steps + 1):
        indices = [
            (step * config.batch_size + offset) % len(train_programs)
            for offset in range(config.batch_size)
        ]
        clean = [train_programs[index] for index in indices]
        noisy = [
            corrupt_factorized_geometry(
                program,
                codec,
                config.corruption_probability,
                np.random.default_rng(config.seed + step * 100_000 + index),
            )
            for index, program in zip(indices, clean, strict=True)
        ]
        condition = _condition([train_rows[index] for index in indices], groups, subgroups, device)
        clean_batch = packed_batch(clean, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(packed_batch(noisy, device), condition)
        loss, counts = geometry_loss_and_accuracy(logits, clean_batch, codec)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        metrics.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "train_token_accuracy": counts["correct"] / counts["total"],
            }
        )

    validation = _evaluate(
        model,
        validation_rows,
        validation_programs,
        groups,
        subgroups,
        codec,
        config,
        device,
    )
    lock_verified = _verify_locked_path(
        model,
        validation_rows[0],
        validation_programs[0],
        groups,
        subgroups,
        codec,
        config,
        device,
    )
    checkpoint = _save_checkpoint(model, optimizer, config.steps)
    restored = _new_model(config, codec, groups, subgroups).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=config.learning_rate)
    document = _decode_checkpoint(checkpoint)
    restored.load_state_dict(document["model"])
    restored_optimizer.load_state_dict(document["optimizer"])
    checkpoint_round_trip = _model_hash(restored) == _model_hash(model)
    if int(document["step"]) != config.steps or not checkpoint_round_trip:
        raise OpenMojiPilotError("canonical checkpoint did not restore the trained model")

    checkpoint_sha256 = hashlib.sha256(checkpoint).hexdigest()
    metrics_payload = _jsonl_bytes(metrics)
    summary = {
        "schema_version": 1,
        "study_version": config.version,
        "scope": "dominant-bucket fixed-topology geometry pilot; not unconditional generation",
        "config_sha256": _file_sha256(config_path),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": True,
        "corruption": "factorized_role_uniform_geometry",
        "bucket": config.bucket,
        "bucket_icons": config.expected_bucket_icons,
        "selected_train_rows": [row.source_path for row in train_rows],
        "selected_validation_rows": [row.source_path for row in validation_rows],
        "group_vocabulary_size": len(groups) + 1,
        "subgroup_vocabulary_size": len(subgroups) + 1,
        "model_parameters": parameter_count,
        "steps": config.steps,
        "final_train": metrics[-1],
        "validation": validation,
        "locked_path_exact": lock_verified,
        "checkpoint_bytes": len(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_round_trip": checkpoint_round_trip,
        "metrics_sha256": hashlib.sha256(metrics_payload).hexdigest(),
    }
    _write_bytes_artifact(config.report_root / "metrics.jsonl", metrics_payload)
    _write_bytes_artifact(config.report_root / "summary.json", _json_bytes(summary))
    _write_bytes_artifact(config.report_root / "README.md", _markdown(summary).encode())
    _write_bytes_artifact(config.checkpoint_root / "checkpoint.zip", checkpoint)
    return summary


def _new_model(
    config: OpenMojiPilotConfig,
    codec: CodecConfig,
    groups: dict[str, int],
    subgroups: dict[str, int],
) -> GeometryDenoiser:
    return GeometryDenoiser(
        codec,
        config.total_segment_slots,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        feedforward=config.feedforward,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
    )


def _evaluate(
    model: GeometryDenoiser,
    rows: tuple[PilotRow, ...],
    clean: list[PackedTensorProgram],
    groups: dict[str, int],
    subgroups: dict[str, int],
    codec: CodecConfig,
    config: OpenMojiPilotConfig,
    device: torch.device,
) -> dict[str, float | int | None]:
    noisy = [
        corrupt_factorized_geometry(
            program,
            codec,
            config.corruption_probability,
            np.random.default_rng(config.seed + 9_000_000 + index),
        )
        for index, program in enumerate(clean)
    ]
    with torch.no_grad():
        logits = model(packed_batch(noisy, device), _condition(rows, groups, subgroups, device))
        loss, counts = geometry_loss_and_accuracy(logits, packed_batch(clean, device), codec)
        split = geometry_accuracy_by_corruption(
            logits, packed_batch(noisy, device), packed_batch(clean, device), codec
        )
    return {
        "loss": float(loss),
        "accuracy": counts["correct"] / counts["total"],
        "changed_accuracy": (
            split["changed"]["correct"] / split["changed"]["total"]
            if split["changed"]["total"]
            else None
        ),
        "changed_total": split["changed"]["total"],
        "retained_accuracy": (
            split["retained"]["correct"] / split["retained"]["total"]
            if split["retained"]["total"]
            else None
        ),
        "retained_total": split["retained"]["total"],
    }


def _verify_locked_path(
    model: GeometryDenoiser,
    row: PilotRow,
    clean: PackedTensorProgram,
    groups: dict[str, int],
    subgroups: dict[str, int],
    codec: CodecConfig,
    config: OpenMojiPilotConfig,
    device: torch.device,
) -> bool:
    locks = np.zeros((codec.max_paths,), dtype=np.bool_)
    locks[0] = True
    noisy = corrupt_factorized_geometry(
        clean,
        codec,
        1.0,
        np.random.default_rng(config.seed + 10_000_000),
        locked_paths=locks,
    )
    with torch.no_grad():
        logits = model(
            packed_batch([noisy], device),
            _condition((row,), groups, subgroups, device),
        )
    prediction = predict_clean_geometry(noisy, logits, codec, locked_paths=locks)
    length = int(clean.path_length[0])
    exact = bool(
        np.array_equal(prediction.start[0], clean.start[0])
        and np.array_equal(prediction.coordinates[:length], clean.coordinates[:length])
    )
    if not exact:
        raise OpenMojiPilotError("locked path changed during corruption or prediction")
    validate_packed_tensor_program(prediction, codec, config.total_segment_slots)
    return True


def _condition(
    rows: tuple[PilotRow, ...] | list[PilotRow],
    groups: dict[str, int],
    subgroups: dict[str, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {
        "group": torch.tensor([groups[row.group] for row in rows], dtype=torch.long, device=device),
        "subgroup": torch.tensor(
            [subgroups[row.subgroup] for row in rows], dtype=torch.long, device=device
        ),
    }


def _load_program(
    row: PilotRow, config: OpenMojiPilotConfig, codec: CodecConfig
) -> PackedTensorProgram:
    source = (config.raw_root / row.source_path).read_bytes()
    if hashlib.sha256(source).hexdigest() != row.source_svg_sha256:
        raise OpenMojiPilotError(f"source hash mismatch: {row.source_path}")
    normalized_source = source if row.selected_representation == "semantic" else _outline(source)
    dense, report = encode_program(normalize_svg(normalized_source).program, codec)
    if report.dropped_contours or report.dropped_segments or report.clamped_coordinates:
        raise OpenMojiPilotError(f"selected bucket requires projection: {row.source_path}")
    packed = pack_tensor_program(dense, codec, config.total_segment_slots)
    validate_packed_tensor_program(packed, codec, config.total_segment_slots)
    return packed


def _selected_codec(config: OpenMojiPilotConfig) -> CodecConfig:
    style = _mapping(json.loads(config.style_summary.read_bytes()), "style summary")
    candidate = _mapping(
        _mapping(style.get("width_vocabulary_analytics"), "width analytics").get(
            config.style_candidate
        ),
        "style candidate",
    )
    widths = tuple(float(value) for value in _sequence(candidate.get("values"), "widths"))
    return CodecConfig(
        max_paths=config.max_paths,
        max_segments=config.max_segments,
        coordinate_bins=289,
        control_coordinate_bins=417,
        control_coordinate_min=-8.0,
        control_coordinate_max=96.0,
        palette=_palette(config.palette),
        stroke_widths=widths,
        dash_patterns=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.1598, 5.1598),
            (0.0, 0.0, 3.9396, 3.9396),
            (0.0, 6.7346, 0.0, 0.0, 0.0, 0.0),
            (2.0, 4.0),
            (5.2132, 5.2132),
            (6.1156, 4.5867),
        ),
        miter_limits=(1.5, 2.0, 4.0, 7.0, 10.0),
        opacities=(0.25, 0.4, 0.5, 0.502, 0.6, 0.9969, 0.997, 0.999, 1.0),
        max_serialized_bytes=2_000_000,
    )


def _select_rows(rows: tuple[PilotRow, ...], count: int, seed: int) -> tuple[PilotRow, ...]:
    if count > len(rows):
        raise OpenMojiPilotError("sample request exceeds available split rows")
    return tuple(
        sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}\0{row.source_path}".encode()
            ).digest(),
        )[:count]
    )


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise OpenMojiPilotError("CUDA was required but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _outline(source: bytes) -> bytes:
    try:
        value = SVG.fromstring(source).topicosvg().tostring()  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise OpenMojiPilotError(f"PicoSVG outline failed: {type(exc).__name__}") from exc
    return value.encode("utf-8") if isinstance(value, str) else value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            raise OpenMojiPilotError(f"blank JSONL row in {path}:{line_number}")
        try:
            rows.append(_mapping(json.loads(line), f"{path}:{line_number}"))
        except json.JSONDecodeError as exc:
            raise OpenMojiPilotError(f"invalid JSONL row in {path}:{line_number}") from exc
    return rows


def _safe_source_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".svg"
        or path.as_posix() != value
    ):
        raise OpenMojiPilotError(f"unsafe source path: {value}")
    return value


def _row_key(row: PilotRow) -> tuple[str, str]:
    return row.source_path, row.hexcode


def _verify(path: Path, expected: str, label: str) -> None:
    if _file_sha256(path) != expected:
        raise OpenMojiPilotError(f"{label} hash mismatch")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _markdown(summary: dict[str, Any]) -> str:
    validation = cast(dict[str, Any], summary["validation"])
    return "\n".join(
        [
            "# OpenMoji Gate G dominant-bucket pipeline smoke",
            "",
            f"Device: `{summary['device']}`; parameters: {summary['model_parameters']:,}; "
            f"steps: {summary['steps']}.",
            "",
            f"Validation accuracy: {validation['accuracy']:.4f}; changed-token accuracy: "
            f"{validation['changed_accuracy']:.4f}.",
            "",
            f"Canonical checkpoint round trip: {summary['checkpoint_round_trip']}; "
            f"locked-path exactness: {summary['locked_path_exact']}.",
            "",
            "This is a bounded fixed-topology geometry pipeline smoke over the dominant exact "
            "packed bucket. It is not evidence for unconditional generation.",
            "",
        ]
    )


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OpenMojiPilotError(f"{field} must be a mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise OpenMojiPilotError(f"{field} must be a list")
    return value


def _string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise OpenMojiPilotError(f"{field} must be a non-empty string")
    return value


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise OpenMojiPilotError("expected a lowercase SHA-256 value")
    return value


def _git_revision(value: str) -> str:
    if not 7 <= len(value) <= 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise OpenMojiPilotError("expected a lowercase hexadecimal Git revision")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenMojiPilotError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenMojiPilotError(f"{field} must be a nonnegative integer")
    return value


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise OpenMojiPilotError(f"{field} must be positive")
    return float(value)


def _probability(value: object, field: str) -> float:
    result = _positive_float(value, field)
    if result > 1:
        raise OpenMojiPilotError(f"{field} must not exceed 1")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_openmoji_pilot_config(args.config)
    print(json.dumps(run_openmoji_pilot(config, args.config), sort_keys=True))


if __name__ == "__main__":
    main()
