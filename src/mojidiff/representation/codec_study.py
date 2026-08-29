"""Measured round-trip study for semantic-stroke and outlined typed codecs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from picosvg.svg import SVG

from mojidiff.curation.audit import _render
from mojidiff.representation.normalizer import NormalizationError, normalize_svg
from mojidiff.representation.program import (
    VIEWBOX_SIZE,
    CodecConfig,
    CodecError,
    EncodingReport,
    FloatProgram,
    SegmentType,
    TensorProgram,
    decode_program,
    encode_program,
    serialize_float_svg,
    serialize_svg,
)
from mojidiff.representation.study import _similarity


class CodecStudyError(RuntimeError):
    """A typed-codec study cannot be reproduced from its recorded inputs."""


@dataclass(frozen=True)
class CodecBudget:
    name: str
    path_slots: int
    segments_per_path: int


@dataclass(frozen=True)
class CodecStudyConfig:
    schema_version: int
    version: str
    source_revision: str
    raw_root: Path
    fixture_manifest: Path
    palette_path: Path
    derived_root: Path
    report_root: Path
    coordinate_bins: tuple[int, ...]
    budgets: tuple[CodecBudget, ...]
    stroke_widths: tuple[float, ...]
    dash_patterns: tuple[tuple[float, ...], ...]
    miter_limits: tuple[float, ...]
    opacities: tuple[float, ...]
    max_serialized_bytes: int
    render_sizes: tuple[int, ...]
    render_timeout_seconds: int
    allow_truncation: bool
    allow_clamping: bool
    compare_unclamped_coordinates: bool


def load_codec_study_config(path: Path) -> CodecStudyConfig:
    """Load an exact versioned study config without implicit research defaults."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(document, "root")
    raw_schema_version = root.get("schema_version")
    if (
        isinstance(raw_schema_version, bool)
        or not isinstance(raw_schema_version, int)
        or raw_schema_version not in {1, 2}
    ):
        raise CodecStudyError("codec study schema_version must be 1 or 2")
    schema_version = raw_schema_version
    codec = _mapping(root.get("codec"), "codec")
    opacities: tuple[float, ...]
    if schema_version == 1:
        if "opacities" in codec:
            raise CodecStudyError("schema_version 1 cannot define codec.opacities")
        opacities = (1.0,)
        allow_truncation = True
        allow_clamping = True
    else:
        opacities = _float_tuple(codec.get("opacities"), "codec.opacities")
        projection = _mapping(root.get("projection"), "projection")
        allow_truncation = _boolean(projection.get("allow_truncation"), "allow_truncation")
        allow_clamping = _boolean(projection.get("allow_clamping"), "allow_clamping")
    raw_analysis = root.get("analysis")
    if raw_analysis is None:
        compare_unclamped_coordinates = False
    else:
        analysis = _mapping(raw_analysis, "analysis")
        compare_unclamped_coordinates = _boolean(
            analysis.get("compare_unclamped_coordinates"),
            "analysis.compare_unclamped_coordinates",
        )
    render = _mapping(root.get("render"), "render")
    raw_budgets = _sequence(codec.get("budgets"), "codec.budgets")
    budgets: list[CodecBudget] = []
    for index, raw_budget in enumerate(raw_budgets):
        budget = _mapping(raw_budget, f"codec.budgets[{index}]")
        budgets.append(
            CodecBudget(
                name=_string(budget, "name"),
                path_slots=_positive_int(budget.get("path_slots"), "path_slots"),
                segments_per_path=_positive_int(
                    budget.get("segments_per_path"), "segments_per_path"
                ),
            )
        )
    if not budgets or len({item.name for item in budgets}) != len(budgets):
        raise CodecStudyError("codec budgets must have unique names")
    sizes = tuple(
        _positive_int(value, "render.sizes")
        for value in _sequence(render.get("sizes"), "render.sizes")
    )
    if sizes != (72, 18):
        raise CodecStudyError("render sizes must be exactly [72, 18]")
    bins = tuple(
        _positive_int(value, "codec.coordinate_bins")
        for value in _sequence(codec.get("coordinate_bins"), "codec.coordinate_bins")
    )
    if not bins or len(set(bins)) != len(bins) or tuple(sorted(bins)) != bins:
        raise CodecStudyError("coordinate bins must contain unique increasing values")
    if schema_version == 1 and len(bins) != 2:
        raise CodecStudyError("schema_version 1 requires exactly two coordinate bins")
    return CodecStudyConfig(
        schema_version=schema_version,
        version=_string(root, "probe_version"),
        source_revision=_string(root, "source_revision"),
        raw_root=Path(_string(root, "raw_root")),
        fixture_manifest=Path(_string(root, "fixture_manifest")),
        palette_path=Path(_string(root, "palette_path")),
        derived_root=Path(_string(root, "derived_root")),
        report_root=Path(_string(root, "report_root")),
        coordinate_bins=bins,
        budgets=tuple(budgets),
        stroke_widths=_float_tuple(codec.get("stroke_widths"), "codec.stroke_widths"),
        dash_patterns=tuple(
            _float_tuple(value, f"codec.dash_patterns[{index}]")
            for index, value in enumerate(
                _sequence(codec.get("dash_patterns"), "codec.dash_patterns")
            )
        ),
        miter_limits=_float_tuple(codec.get("miter_limits"), "codec.miter_limits"),
        opacities=opacities,
        max_serialized_bytes=_positive_int(
            codec.get("max_serialized_bytes"), "codec.max_serialized_bytes"
        ),
        render_sizes=sizes,
        render_timeout_seconds=_positive_int(
            render.get("timeout_seconds"), "render.timeout_seconds"
        ),
        allow_truncation=allow_truncation,
        allow_clamping=allow_clamping,
        compare_unclamped_coordinates=compare_unclamped_coordinates,
    )


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CodecStudyError(f"{field} must be a mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise CodecStudyError(f"{field} must be a list")
    return value


def _string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise CodecStudyError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CodecStudyError(f"{field} must be a positive integer")
    return value


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise CodecStudyError(f"{field} must be a boolean")
    return value


def _float_tuple(value: object, field: str) -> tuple[float, ...]:
    items = _sequence(value, field)
    result: list[float] = []
    for item in items:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CodecStudyError(f"{field} must contain numbers")
        number = float(item)
        if not math.isfinite(number):
            raise CodecStudyError(f"{field} must contain finite numbers")
        result.append(number)
    return tuple(result)


def _palette(path: Path) -> tuple[str, ...]:
    document = _mapping(json.loads(path.read_text(encoding="utf-8")), "palette")
    values: list[str] = []
    for value in _sequence(document.get("colors"), "palette.colors"):
        if not isinstance(value, str):
            raise CodecStudyError("palette colors must be strings")
        values.append(value.lower())
    for group in _mapping(document.get("skintones"), "palette.skintones").values():
        for value in _sequence(group, "palette.skintones entry"):
            if not isinstance(value, str):
                raise CodecStudyError("palette colors must be strings")
            values.append(value.lower())
    return tuple(dict.fromkeys(values))


def _codec_config(
    config: CodecStudyConfig, budget: CodecBudget, bins: int, palette: tuple[str, ...]
) -> CodecConfig:
    return CodecConfig(
        max_paths=budget.path_slots,
        max_segments=budget.segments_per_path,
        coordinate_bins=bins,
        palette=palette,
        stroke_widths=config.stroke_widths,
        dash_patterns=config.dash_patterns,
        miter_limits=config.miter_limits,
        opacities=config.opacities,
        max_serialized_bytes=config.max_serialized_bytes,
    )


def run_codec_study(config: CodecStudyConfig, config_path: Path) -> dict[str, Any]:
    """Execute the fixed CPU fixture study and write compact reproducible evidence."""

    code_identity = _code_identity()
    fixture_bytes = config.fixture_manifest.read_bytes()
    fixture = _mapping(json.loads(fixture_bytes), "fixture")
    if config.schema_version == 2:
        if fixture.get("source_revision") != config.source_revision:
            raise CodecStudyError("schema_version 2 fixture source_revision mismatch")
        selection = _mapping(fixture.get("selection"), "fixture.selection")
        _verify_fixture_parent(selection, "parent_summary")
        if "parent_hybrid" in selection or "parent_hybrid_sha256" in selection:
            _verify_fixture_parent(selection, "parent_hybrid")
    rows = _sequence(fixture.get("rows"), "fixture.rows")
    palette = _palette(config.palette_path)
    normalization_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    used_colors: set[str] = set()

    for raw_row in rows:
        row = _mapping(raw_row, "fixture row")
        relative = Path(_string(row, "source_path"))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".svg":
            raise CodecStudyError(f"unsafe fixture path: {relative}")
        source_path = config.raw_root / relative
        source_bytes = source_path.read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        if source_sha != _string(row, "source_svg_sha256"):
            raise CodecStudyError(f"source hash mismatch for {relative}")
        baselines = {
            size: _render(source_bytes, size, config.render_timeout_seconds)[1]
            for size in config.render_sizes
        }
        outlined = _outline(source_bytes)
        outline_path = config.derived_root / "outlined" / relative
        _write_bytes_artifact(outline_path, outlined)

        for representation, candidate_source in (
            ("semantic", source_bytes),
            ("outlined", outlined),
        ):
            identity = {
                "hexcode": _string(row, "hexcode"),
                "group": _string(row, "group"),
                "source_path": str(relative),
                "source_sha256": source_sha,
                "representation": representation,
            }
            try:
                normalized = normalize_svg(candidate_source)
            except NormalizationError as exc:
                normalization_rows.append(
                    {**identity, "ok": False, "error_code": exc.code, "error": str(exc)}
                )
                continue
            program = normalized.program
            for contour in program.contours:
                if contour.fill is not None:
                    used_colors.add(contour.fill)
                if contour.stroke is not None:
                    used_colors.add(contour.stroke)
            normalization_rows.append(
                {
                    **identity,
                    "ok": True,
                    "error_code": None,
                    "error": None,
                    **asdict(normalized.report),
                    "max_segments_per_contour": max(
                        (len(contour.segments) for contour in program.contours), default=0
                    ),
                    "opacity_values": sorted({contour.opacity for contour in program.contours}),
                    "fill_opacity_values": sorted(
                        {
                            contour.fill_opacity
                            for contour in program.contours
                            if contour.fill_opacity is not None
                        }
                    ),
                    "stroke_opacity_values": sorted(
                        {
                            contour.stroke_opacity
                            for contour in program.contours
                            if contour.stroke_opacity is not None
                        }
                    ),
                    "stroke_width_values": sorted(
                        {
                            contour.stroke_width
                            for contour in program.contours
                            if contour.stroke is not None
                        }
                    ),
                    "miter_limit_values": sorted(
                        {
                            contour.miter_limit
                            for contour in program.contours
                            if contour.stroke is not None
                        }
                    ),
                    "dash_patterns": sorted(
                        {
                            contour.dash_pattern
                            for contour in program.contours
                            if contour.stroke is not None and contour.dash_pattern
                        }
                    ),
                    "coordinate_excursions": _coordinate_excursions(program),
                }
            )
            for budget in config.budgets:
                for bins in config.coordinate_bins:
                    codec = _codec_config(config, budget, bins, palette)
                    metric_rows.append(
                        _round_trip(
                            identity,
                            program,
                            codec,
                            budget,
                            bins,
                            baselines,
                            config,
                            relative,
                        )
                    )

    normalization_payload = _jsonl_bytes(normalization_rows)
    metrics_payload = _jsonl_bytes(metric_rows)
    summary = _summarize(
        normalization_rows,
        metric_rows,
        config,
        config_path,
        palette,
        used_colors,
        code_identity,
    )
    summary["normalization_sha256"] = hashlib.sha256(normalization_payload).hexdigest()
    summary["metrics_sha256"] = hashlib.sha256(metrics_payload).hexdigest()
    if _code_identity() != code_identity:
        raise CodecStudyError("codec study source identity changed during the run")
    _write_bytes_artifact(config.report_root / "normalization.jsonl", normalization_payload)
    _write_bytes_artifact(config.report_root / "metrics.jsonl", metrics_payload)
    _write_bytes_artifact(
        config.report_root / "summary.json",
        (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(),
    )
    _write_bytes_artifact(config.report_root / "README.md", _markdown(summary).encode())
    return summary


def _outline(source_bytes: bytes) -> bytes:
    try:
        text = SVG.fromstring(source_bytes).topicosvg().tostring()  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise CodecStudyError(f"PicoSVG outline failed: {exc}") from exc
    return text.encode("utf-8") if isinstance(text, str) else text


def _round_trip(
    identity: dict[str, Any],
    program: FloatProgram,
    codec: CodecConfig,
    budget: CodecBudget,
    bins: int,
    baselines: dict[int, np.ndarray[Any, Any]],
    config: CodecStudyConfig,
    relative: Path,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        **identity,
        "budget": budget.name,
        "path_slots": budget.path_slots,
        "segments_per_path": budget.segments_per_path,
        "coordinate_bins": bins,
        "allow_truncation": config.allow_truncation,
        "allow_clamping": config.allow_clamping,
        "segments_dropped_by_path_budget": sum(
            len(contour.segments) for contour in program.contours[budget.path_slots :]
        ),
        "segments_dropped_by_segment_budget": sum(
            max(0, len(contour.segments) - budget.segments_per_path)
            for contour in program.contours[: budget.path_slots]
        ),
    }
    strict_encode_ok = True
    strict_lossless = False
    strict_error: str | None = None
    try:
        _strict_tensor, strict_report = encode_program(program, codec)
        strict_lossless = strict_report.lossless
    except CodecError as exc:
        strict_encode_ok = False
        strict_error = f"{type(exc).__name__}:{exc}"[:500]
    try:
        tensor, report = encode_program(
            program,
            codec,
            allow_truncation=config.allow_truncation,
            allow_clamping=config.allow_clamping,
        )
        svg_bytes = serialize_svg(tensor, codec)
        decoded = decode_program(tensor, codec)
        reencoded, _reencode_report = encode_program(decoded, codec)
        stable = _tensor_hash(tensor) == _tensor_hash(reencoded)
        if not stable or svg_bytes != serialize_svg(reencoded, codec):
            raise CodecStudyError("encode/decode/serialize identity is not stable")
        output_path = (
            config.derived_root
            / str(identity["representation"])
            / budget.name
            / str(bins)
            / relative
        )
        _write_bytes_artifact(output_path, svg_bytes)
        counterfactual_bytes: bytes | None = None
        counterfactual_renders: dict[int, np.ndarray[Any, Any]] = {}
        if config.compare_unclamped_coordinates:
            counterfactual = _quantize_unclamped(program, bins)
            counterfactual_bytes = serialize_float_svg(
                counterfactual,
                max_serialized_bytes=config.max_serialized_bytes,
            )
            counterfactual_path = (
                config.derived_root
                / "unclamped-counterfactual"
                / str(identity["representation"])
                / str(bins)
                / relative
            )
            _write_bytes_artifact(counterfactual_path, counterfactual_bytes)
            counterfactual_renders = {
                size: _render(counterfactual_bytes, size, config.render_timeout_seconds)[1]
                for size in config.render_sizes
            }
        result.update(
            {
                "ok": True,
                "error": None,
                "tensor_sha256": _tensor_hash(tensor),
                "svg_sha256": hashlib.sha256(svg_bytes).hexdigest(),
                "serialized_bytes": len(svg_bytes),
                "stable_round_trip": stable,
                "strict_encode_ok": strict_encode_ok,
                "strict_lossless": strict_lossless,
                "strict_error": strict_error,
                "safety_projection_applied": not report.lossless,
                "unclamped_counterfactual_sha256": (
                    None
                    if counterfactual_bytes is None
                    else hashlib.sha256(counterfactual_bytes).hexdigest()
                ),
                "unclamped_counterfactual_bytes": (
                    None if counterfactual_bytes is None else len(counterfactual_bytes)
                ),
                **_report_fields(report),
            }
        )
        for size, baseline in baselines.items():
            candidate = _render(svg_bytes, size, config.render_timeout_seconds)[1]
            for field, value in _similarity(baseline, candidate).items():
                result[f"{field}_{size}"] = value
            if counterfactual_renders:
                unclamped = counterfactual_renders[size]
                for field, value in _similarity(baseline, unclamped).items():
                    result[f"unclamped_{field}_{size}"] = value
                for field, value in _similarity(unclamped, candidate).items():
                    result[f"clamp_increment_{field}_{size}"] = value
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error": f"{type(exc).__name__}:{exc}"[:500],
                "stable_round_trip": False,
                "strict_encode_ok": strict_encode_ok,
                "strict_lossless": strict_lossless,
                "strict_error": strict_error,
            }
        )
    return result


def _coordinate_excursions(program: FloatProgram) -> list[dict[str, Any]]:
    """Describe every coordinate outside the tensor vocabulary by geometric role."""

    excursions: list[dict[str, Any]] = []
    for contour_index, contour in enumerate(program.contours):
        for coordinate_index, value in enumerate(contour.start):
            if value < 0 or value > VIEWBOX_SIZE:
                excursions.append(
                    _excursion(
                        contour_index=contour_index,
                        layer=contour.layer,
                        segment_index=None,
                        segment_type="move",
                        role="endpoint",
                        coordinate_index=coordinate_index,
                        value=value,
                    )
                )
        for segment_index, segment in enumerate(contour.segments):
            endpoint_start = {
                SegmentType.LINE: 0,
                SegmentType.QUAD: 2,
                SegmentType.CUBIC: 4,
                SegmentType.CLOSE: 0,
            }[segment.kind]
            for coordinate_index, value in enumerate(segment.coords):
                if value < 0 or value > VIEWBOX_SIZE:
                    excursions.append(
                        _excursion(
                            contour_index=contour_index,
                            layer=contour.layer,
                            segment_index=segment_index,
                            segment_type=segment.kind.name.lower(),
                            role="endpoint" if coordinate_index >= endpoint_start else "control",
                            coordinate_index=coordinate_index,
                            value=value,
                        )
                    )
    return excursions


def _excursion(
    *,
    contour_index: int,
    layer: int,
    segment_index: int | None,
    segment_type: str,
    role: str,
    coordinate_index: int,
    value: float,
) -> dict[str, Any]:
    bound = 0.0 if value < 0 else VIEWBOX_SIZE
    return {
        "contour_index": contour_index,
        "layer": layer,
        "segment_index": segment_index,
        "segment_type": segment_type,
        "role": role,
        "coordinate_index": coordinate_index,
        "axis": "x" if coordinate_index % 2 == 0 else "y",
        "side": "below" if value < 0 else "above",
        "value": value,
        "distance_outside": abs(value - bound),
    }


def _quantize_unclamped(program: FloatProgram, bins: int) -> FloatProgram:
    """Extend the configured lattice beyond 0..72 for an analysis-only counterfactual."""

    step = VIEWBOX_SIZE / (bins - 1)

    def quantize(values: tuple[float, ...]) -> tuple[float, ...]:
        return tuple(math.floor(value / step + 0.5) * step for value in values)

    return FloatProgram(
        tuple(
            replace(
                contour,
                start=cast(tuple[float, float], quantize(contour.start)),
                segments=tuple(
                    replace(segment, coords=quantize(segment.coords))
                    for segment in contour.segments
                ),
            )
            for contour in program.contours
        )
    )


def _report_fields(report: EncodingReport) -> dict[str, Any]:
    fields = asdict(report)
    fields["partial_layers"] = list(report.partial_layers)
    fields["encoding_lossless"] = report.lossless
    return fields


def _tensor_hash(program: TensorProgram) -> str:
    digest = hashlib.sha256()
    for field in program.__dataclass_fields__:
        array = getattr(program, field)
        digest.update(field.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.astype("<i8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()


def _write_bytes_artifact(path: Path, payload: bytes) -> None:
    """Atomically create an artifact, accepting only an exact prior copy."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise CodecStudyError(f"refusing to replace symlink artifact: {path}")
    if path.exists() and not path.is_file():
        raise CodecStudyError(f"artifact target is not a regular file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise CodecStudyError(f"artifact target is not a regular file: {path}") from None
            if not _files_equal(temporary, path):
                raise CodecStudyError(f"refusing to replace differing artifact: {path}") from None
        except OSError as exc:
            raise CodecStudyError(f"artifact could not be installed: {path}: {exc}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _files_equal(left: Path, right: Path) -> bool:
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as left_handle, right.open("rb") as right_handle:
        while True:
            left_chunk = left_handle.read(1 << 20)
            right_chunk = right_handle.read(1 << 20)
            if left_chunk != right_chunk:
                return False
            if not left_chunk:
                return True


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_fixture_parent(selection: dict[str, Any], field: str) -> None:
    parent = Path(_string(selection, field))
    if parent.is_absolute() or ".." in parent.parts:
        raise CodecStudyError(f"fixture {field} must be a safe repository path")
    expected_sha = _string(selection, f"{field}_sha256")
    if _file_sha256(parent) != expected_sha:
        raise CodecStudyError(f"fixture {field} hash mismatch")


def _code_identity() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise CodecStudyError(f"could not identify Git commit: {exc}") from exc
    sources = {
        "codec_study_sha256": repository / "src/mojidiff/representation/codec_study.py",
        "normalizer_sha256": repository / "src/mojidiff/representation/normalizer.py",
        "program_sha256": repository / "src/mojidiff/representation/program.py",
    }
    return {"git_commit": commit, **{name: _file_sha256(path) for name, path in sources.items()}}


def _summarize(
    normalization: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    config: CodecStudyConfig,
    config_path: Path,
    palette: tuple[str, ...],
    used_colors: set[str],
    code_identity: dict[str, str],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        key = f"{row['representation']}|{row['budget']}|q{row['coordinate_bins']}"
        groups[key].append(row)
    comparisons: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        successful = [row for row in rows if row.get("ok")]
        metric_fields = [
            "rgba_mae_72",
            "rgba_mae_18",
            "alpha_iou_72",
            "alpha_iou_18",
            "pixel_exact_fraction_72",
            "pixel_exact_fraction_18",
            "serialized_bytes",
        ]
        if config.compare_unclamped_coordinates:
            metric_fields.extend(
                (
                    "unclamped_rgba_mae_72",
                    "unclamped_rgba_mae_18",
                    "clamp_increment_rgba_mae_72",
                    "clamp_increment_rgba_mae_18",
                    "clamp_increment_alpha_iou_72",
                    "clamp_increment_alpha_iou_18",
                    "clamp_increment_pixel_exact_fraction_72",
                    "clamp_increment_pixel_exact_fraction_18",
                )
            )
        comparisons[key] = {
            "attempted": len(rows),
            "successful": len(successful),
            "failures": len(rows) - len(successful),
            "strict_encode_failures": sum(not bool(row.get("strict_encode_ok")) for row in rows),
            "strict_lossy_encodings": sum(
                bool(row.get("strict_encode_ok")) and not bool(row.get("strict_lossless"))
                for row in rows
            ),
            "icons_with_safety_projection": sum(
                bool(row.get("safety_projection_applied")) for row in successful
            ),
            "strict_projection_cases": [
                {
                    "hexcode": row["hexcode"],
                    "source_path": row["source_path"],
                    "strict_error": row.get("strict_error"),
                    "clamped_coordinates": row.get("clamped_coordinates"),
                    "approximated_stroke_widths": row.get("approximated_stroke_widths"),
                    "approximated_miter_limits": row.get("approximated_miter_limits"),
                }
                for row in successful
                if not bool(row.get("strict_lossless"))
            ],
            "icons_with_path_truncation": sum(
                row.get("dropped_contours", 0) > 0 for row in successful
            ),
            "icons_with_segment_truncation": sum(
                row.get("segments_dropped_by_segment_budget", 0) > 0 for row in successful
            ),
            "dropped_contours": sum(int(row.get("dropped_contours", 0)) for row in successful),
            "dropped_segments": sum(int(row.get("dropped_segments", 0)) for row in successful),
            "segments_dropped_by_path_budget": sum(
                int(row.get("segments_dropped_by_path_budget", 0)) for row in successful
            ),
            "segments_dropped_by_segment_budget": sum(
                int(row.get("segments_dropped_by_segment_budget", 0)) for row in successful
            ),
            "icons_with_coordinate_clamping": sum(
                row.get("clamped_coordinates", 0) > 0 for row in successful
            ),
            "metrics": {
                field: _stats(successful, field)
                for field in metric_fields
            },
            "worst_rgba_mae_18": _worst(successful, "rgba_mae_18", 8),
            "worst_clamp_increment_rgba_mae_18": (
                _worst(successful, "clamp_increment_rgba_mae_18", 8)
                if config.compare_unclamped_coordinates
                else []
            ),
            "errors": [
                {"hexcode": row["hexcode"], "error": row.get("error")}
                for row in rows
                if not row.get("ok")
            ],
        }
    normalization_summary: dict[str, Any] = {}
    for representation in ("semantic", "outlined"):
        rows = [row for row in normalization if row["representation"] == representation]
        successful = [row for row in rows if row.get("ok")]
        excursions = [
            excursion
            for row in successful
            for excursion in cast(list[dict[str, Any]], row.get("coordinate_excursions", []))
        ]
        normalization_summary[representation] = {
            "attempted": len(rows),
            "successful": len(successful),
            "failures": [
                {
                    "hexcode": row["hexcode"],
                    "error_code": row.get("error_code"),
                    "error": row.get("error"),
                }
                for row in rows
                if not row.get("ok")
            ],
            "contours": _stats(successful, "contours"),
            "segments": _stats(successful, "segments"),
            "max_segments_per_contour": _stats(successful, "max_segments_per_contour"),
            "icons_with_compound_layers": sum(
                row.get("compound_layers", 0) > 0 for row in successful
            ),
            "icons_with_out_of_bounds_coordinates": sum(
                row.get("out_of_bounds_coordinates", 0) > 0 for row in successful
            ),
            "coordinate_excursions": {
                "count": len(excursions),
                "by_role": _counts(excursions, "role"),
                "by_segment_type": _counts(excursions, "segment_type"),
                "by_axis": _counts(excursions, "axis"),
                "by_side": _counts(excursions, "side"),
                "distance_outside": _stats(excursions, "distance_outside"),
            },
            "icons_with_partial_opacity": sum(
                row.get("partially_opaque_layers", 0) > 0 for row in successful
            ),
            "partially_opaque_layers": sum(
                int(row.get("partially_opaque_layers", 0)) for row in successful
            ),
            "observed_opacity_values": {
                field: sorted(
                    {
                        float(value)
                        for row in successful
                        for value in cast(list[float], row.get(field, []))
                    }
                )
                for field in (
                    "opacity_values",
                    "fill_opacity_values",
                    "stroke_opacity_values",
                )
            },
            "observed_stroke_widths": sorted(
                {
                    float(value)
                    for row in successful
                    for value in cast(list[float], row.get("stroke_width_values", []))
                }
            ),
            "observed_miter_limits": sorted(
                {
                    float(value)
                    for row in successful
                    for value in cast(list[float], row.get("miter_limit_values", []))
                }
            ),
        }
    return {
        "schema_version": config.schema_version,
        "probe_version": config.version,
        "code_identity": code_identity,
        "source_revision": config.source_revision,
        "fixture_manifest": str(config.fixture_manifest),
        "fixture_manifest_sha256": hashlib.sha256(config.fixture_manifest.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "picosvg_version": version("picosvg"),
        "cairosvg_version": version("CairoSVG"),
        "fixture_count": len({row["source_path"] for row in normalization}),
        "projection_policy": {
            "allow_truncation": config.allow_truncation,
            "allow_clamping": config.allow_clamping,
        },
        "analysis": {
            "compare_unclamped_coordinates": config.compare_unclamped_coordinates,
            "unclamped_counterfactual": (
                "same coordinate lattice extended beyond 0..72; analysis-only and not a "
                "model vocabulary"
                if config.compare_unclamped_coordinates
                else None
            ),
        },
        "codec": {
            "coordinate_bins": list(config.coordinate_bins),
            "budgets": [asdict(budget) for budget in config.budgets],
            "opacities": list(config.opacities),
            "stroke_widths": list(config.stroke_widths),
            "dash_patterns": [list(pattern) for pattern in config.dash_patterns],
            "miter_limits": list(config.miter_limits),
        },
        "palette": {
            "configured_entries": len(palette),
            "used_entries": len(used_colors),
            "outliers": sorted(used_colors - set(palette)),
        },
        "normalization": normalization_summary,
        "comparisons": comparisons,
    }


def _stats(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = np.asarray(
        [float(row[field]) for row in rows if row.get(field) is not None], dtype=np.float64
    )
    values = values[np.isfinite(values)]
    return {
        "count": len(values),
        "min": float(values.min()) if len(values) else None,
        "median": float(np.median(values)) if len(values) else None,
        "p95": float(np.quantile(values, 0.95)) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        counts[str(row[field])] += 1
    return dict(sorted(counts.items()))


def _worst(rows: list[dict[str, Any]], field: str, count: int) -> list[dict[str, Any]]:
    ranked = sorted(
        (row for row in rows if row.get(field) is not None),
        key=lambda row: (-float(row[field]), str(row["source_path"])),
    )
    return [
        {
            "hexcode": row["hexcode"],
            "source_path": row["source_path"],
            field: row[field],
            "dropped_contours": row.get("dropped_contours"),
            "dropped_segments": row.get("dropped_segments"),
            "clamped_coordinates": row.get("clamped_coordinates"),
        }
        for row in ranked[:count]
    ]


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Typed semantic-stroke versus outlined codec probe",
        "",
        f"Fixture: {summary['fixture_count']} icons. PicoSVG {summary['picosvg_version']}; "
        f"CairoSVG {summary['cairosvg_version']}.",
        "",
        "| representation / budget / bins | ok | MAE 72 median | MAE 18 median | "
        "alpha IoU 18 median | truncated icons (P/S) | projected icons |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, item in summary["comparisons"].items():
        metrics = item["metrics"]
        display_key = key.replace("|", " / ")
        lines.append(
            f"| {display_key} | {item['successful']}/{item['attempted']} | "
            f"{_format(metrics['rgba_mae_72']['median'])} | "
            f"{_format(metrics['rgba_mae_18']['median'])} | "
            f"{_format(metrics['alpha_iou_18']['median'])} | "
            f"{item['icons_with_path_truncation']}/{item['icons_with_segment_truncation']} |"
            f" {item['icons_with_safety_projection']} |"
        )
    lines.extend(
        (
            "",
            "Normalization failures and exact worst-case lists are retained in `summary.json`;",
            "per-icon evidence is in `normalization.jsonl` and `metrics.jsonl`.",
            "",
        )
    )
    if summary["analysis"]["compare_unclamped_coordinates"]:
        lines.extend(
            (
                "## Out-of-bounds coordinate projection",
                "",
                "The unclamped comparison extends the same coordinate lattice outside 0..72 "
                "for rendering only. It is not a proposed model vocabulary.",
                "",
                "| representation / budget / bins | clamp MAE 72 median/max | "
                "clamp MAE 18 median/max |",
                "|---|---:|---:|",
            )
        )
        for key, item in summary["comparisons"].items():
            metrics = item["metrics"]
            at_72 = metrics["clamp_increment_rgba_mae_72"]
            at_18 = metrics["clamp_increment_rgba_mae_18"]
            lines.append(
                f"| {key.replace('|', ' / ')} | {_format(at_72['median'])}/"
                f"{_format(at_72['max'])} | {_format(at_18['median'])}/"
                f"{_format(at_18['max'])} |"
            )
        lines.append("")
    return "\n".join(lines)


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mojidiff-codec-study")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = run_codec_study(load_codec_study_config(args.config), args.config)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
