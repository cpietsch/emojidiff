"""Measured round-trip study for semantic-stroke and outlined typed codecs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from picosvg.svg import SVG

from mojidiff.curation.audit import _render
from mojidiff.representation.normalizer import NormalizationError, normalize_svg
from mojidiff.representation.program import (
    CodecConfig,
    EncodingReport,
    FloatProgram,
    TensorProgram,
    decode_program,
    encode_program,
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
    max_serialized_bytes: int
    render_sizes: tuple[int, ...]
    render_timeout_seconds: int


def load_codec_study_config(path: Path) -> CodecStudyConfig:
    """Load an exact versioned study config without implicit research defaults."""

    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = _mapping(document, "root")
    if root.get("schema_version") != 1:
        raise CodecStudyError("codec study schema_version must be 1")
    codec = _mapping(root.get("codec"), "codec")
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
    if len(bins) != 2 or len(set(bins)) != 2 or tuple(sorted(bins)) != bins:
        raise CodecStudyError("coordinate bins must contain two distinct increasing values")
    return CodecStudyConfig(
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
        max_serialized_bytes=_positive_int(
            codec.get("max_serialized_bytes"), "codec.max_serialized_bytes"
        ),
        render_sizes=sizes,
        render_timeout_seconds=_positive_int(
            render.get("timeout_seconds"), "render.timeout_seconds"
        ),
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
        max_serialized_bytes=config.max_serialized_bytes,
    )


def run_codec_study(config: CodecStudyConfig, config_path: Path) -> dict[str, Any]:
    """Execute the fixed CPU fixture study and write compact reproducible evidence."""

    fixture = _mapping(json.loads(config.fixture_manifest.read_text(encoding="utf-8")), "fixture")
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
        outline_path.parent.mkdir(parents=True, exist_ok=True)
        outline_path.write_bytes(outlined)

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

    summary = _summarize(
        normalization_rows,
        metric_rows,
        config,
        config_path,
        palette,
        used_colors,
    )
    config.report_root.mkdir(parents=True, exist_ok=True)
    _write_jsonl(config.report_root / "normalization.jsonl", normalization_rows)
    _write_jsonl(config.report_root / "metrics.jsonl", metric_rows)
    (config.report_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (config.report_root / "README.md").write_text(_markdown(summary), encoding="utf-8")
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
        "segments_dropped_by_path_budget": sum(
            len(contour.segments) for contour in program.contours[budget.path_slots :]
        ),
        "segments_dropped_by_segment_budget": sum(
            max(0, len(contour.segments) - budget.segments_per_path)
            for contour in program.contours[: budget.path_slots]
        ),
    }
    try:
        tensor, report = encode_program(program, codec, allow_truncation=True, allow_clamping=True)
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
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(svg_bytes)
        result.update(
            {
                "ok": True,
                "error": None,
                "tensor_sha256": _tensor_hash(tensor),
                "svg_sha256": hashlib.sha256(svg_bytes).hexdigest(),
                "serialized_bytes": len(svg_bytes),
                "stable_round_trip": stable,
                **_report_fields(report),
            }
        )
        for size, baseline in baselines.items():
            candidate = _render(svg_bytes, size, config.render_timeout_seconds)[1]
            for field, value in _similarity(baseline, candidate).items():
                result[f"{field}_{size}"] = value
    except Exception as exc:
        result.update(
            {
                "ok": False,
                "error": f"{type(exc).__name__}:{exc}"[:500],
                "stable_round_trip": False,
            }
        )
    return result


def _report_fields(report: EncodingReport) -> dict[str, Any]:
    fields = asdict(report)
    fields["partial_layers"] = list(report.partial_layers)
    return fields


def _tensor_hash(program: TensorProgram) -> str:
    digest = hashlib.sha256()
    for field in program.__dataclass_fields__:
        array = getattr(program, field)
        digest.update(field.encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.astype("<i8", copy=False).tobytes(order="C"))
    return digest.hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def _summarize(
    normalization: list[dict[str, Any]],
    metrics: list[dict[str, Any]],
    config: CodecStudyConfig,
    config_path: Path,
    palette: tuple[str, ...],
    used_colors: set[str],
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in metrics:
        key = f"{row['representation']}|{row['budget']}|q{row['coordinate_bins']}"
        groups[key].append(row)
    comparisons: dict[str, Any] = {}
    for key, rows in sorted(groups.items()):
        successful = [row for row in rows if row.get("ok")]
        comparisons[key] = {
            "attempted": len(rows),
            "successful": len(successful),
            "failures": len(rows) - len(successful),
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
                for field in (
                    "rgba_mae_72",
                    "rgba_mae_18",
                    "alpha_iou_72",
                    "alpha_iou_18",
                    "pixel_exact_fraction_72",
                    "pixel_exact_fraction_18",
                    "serialized_bytes",
                )
            },
            "worst_rgba_mae_18": _worst(successful, "rgba_mae_18", 8),
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
        }
    return {
        "schema_version": 1,
        "probe_version": config.version,
        "source_revision": config.source_revision,
        "fixture_manifest": str(config.fixture_manifest),
        "fixture_manifest_sha256": hashlib.sha256(config.fixture_manifest.read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "picosvg_version": version("picosvg"),
        "cairosvg_version": version("CairoSVG"),
        "fixture_count": len({row["source_path"] for row in normalization}),
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
        "alpha IoU 18 median | truncated icons (P/S) |",
        "|---|---:|---:|---:|---:|---:|",
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
        )
    lines.extend(
        (
            "",
            "Normalization failures and exact worst-case lists are retained in `summary.json`;",
            "per-icon evidence is in `normalization.jsonl` and `metrics.jsonl`.",
            "",
        )
    )
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
