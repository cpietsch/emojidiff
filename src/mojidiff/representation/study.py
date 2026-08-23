"""Small falsification probe for semantic-stroke and outlined representation proxies."""

from __future__ import annotations

import hashlib
import io
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow.parquet as pq
import yaml
from defusedxml import ElementTree
from picosvg.svg import SVG
from PIL import Image
from svg.path import Move, parse_path

from mojidiff.curation.audit import AuditConfig, _render, _safe_parse, _style_and_geometry
from mojidiff.curation.source import OpenMojiSource


class StudyError(RuntimeError):
    """A representation probe cannot be reproduced safely."""


@dataclass(frozen=True)
class StudyConfig:
    version: str
    source_revision: str
    raw_root: Path
    primary_manifest: Path
    fixture_manifest: Path
    derived_root: Path
    report_root: Path
    selection_version: str
    per_group: int
    path_budgets: tuple[int, ...]
    segment_budgets: tuple[int, ...]
    render_sizes: tuple[int, ...]
    render_timeout_seconds: int


def load_study_config(path: Path) -> StudyConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise StudyError("study config schema_version must be 1")
    selection = _map(document.get("selection"), "selection")
    budgets = _map(document.get("budgets"), "budgets")
    render = _map(document.get("render"), "render")
    sizes = tuple(_positive_int(value, "render.sizes") for value in _list(render, "sizes"))
    if sizes != (72, 18):
        raise StudyError("render sizes must be exactly [72, 18]")
    return StudyConfig(
        version=_string(document, "probe_version"),
        source_revision=_string(document, "source_revision"),
        raw_root=Path(_string(document, "raw_root")),
        primary_manifest=Path(_string(document, "primary_manifest")),
        fixture_manifest=Path(_string(document, "fixture_manifest")),
        derived_root=Path(_string(document, "derived_root")),
        report_root=Path(_string(document, "report_root")),
        selection_version=_string(selection, "version"),
        per_group=_positive_int(selection.get("per_group"), "selection.per_group"),
        path_budgets=tuple(
            _positive_int(value, "budgets.path_slots") for value in _list(budgets, "path_slots")
        ),
        segment_budgets=tuple(
            _positive_int(value, "budgets.segments_per_path")
            for value in _list(budgets, "segments_per_path")
        ),
        render_sizes=sizes,
        render_timeout_seconds=_positive_int(
            render.get("timeout_seconds"), "render.timeout_seconds"
        ),
    )


def _map(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StudyError(f"{field} must be a mapping")
    return value


def _list(mapping: dict[str, Any], field: str) -> list[Any]:
    value = mapping.get(field)
    if not isinstance(value, list):
        raise StudyError(f"{field} must be a list")
    return value


def _string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise StudyError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StudyError(f"{field} must be a positive integer")
    return value


def select_fixture(rows: list[dict[str, Any]], config: StudyConfig) -> list[dict[str, Any]]:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group"])].append(row)
    selected: dict[str, dict[str, Any]] = {}
    for group, members in sorted(by_group.items()):
        priorities: list[dict[str, Any]] = []
        for field, reverse in (
            ("segment_count", False),
            ("segment_count", True),
            ("ink_fraction_72", False),
            ("ink_fraction_72", True),
        ):
            priorities.append(
                sorted(
                    members,
                    key=lambda row: (float(row.get(field, 0)), row["source_path"]),
                    reverse=reverse,
                )[0]
            )
        hashed = sorted(
            members,
            key=lambda row: hashlib.sha256(
                f"{config.selection_version}:{group}:{row['source_path']}".encode()
            ).hexdigest(),
        )
        for row in (*priorities, *hashed):
            if sum(item["group"] == group for item in selected.values()) >= config.per_group:
                break
            selected[str(row["source_path"])] = row
    return [selected[key] for key in sorted(selected)]


def _audit_config(config: StudyConfig) -> AuditConfig:
    source = OpenMojiSource(
        name="OpenMoji",
        url="https://github.com/hfg-gmuend/openmoji.git",
        tag="17.0.0",
        revision=config.source_revision,
        license="CC-BY-SA-4.0",
        license_path="LICENSE.txt",
        metadata_path="data/openmoji.csv",
        palette_path="data/color-palette.json",
        svg_glob="color/svg/*.svg",
        raw_root=config.raw_root,
        source_manifest=Path("data/manifests/openmoji-17.0.0-source.json"),
    )
    return AuditConfig(
        version=config.version,
        source=source,
        output_root=config.derived_root,
        sizes=(72, 18),
        timeout_seconds=config.render_timeout_seconds,
        max_svg_bytes=2_000_000,
        max_elements=20_000,
        decisions={},
    )


def _program_counts(svg_bytes: bytes) -> tuple[int, list[int]]:
    root = ElementTree.fromstring(svg_bytes)
    per_path: list[int] = []
    for element in root.iter():
        name = element.tag.rsplit("}", 1)[-1]
        if name == "path":
            try:
                parsed = parse_path(element.attrib.get("d", ""))
                per_path.append(sum(not isinstance(segment, Move) for segment in parsed))
            except Exception:
                per_path.append(0)
        elif name in {"circle", "ellipse"}:
            per_path.append(5)
        elif name == "rect":
            per_path.append(5)
        elif name == "line":
            per_path.append(1)
        elif name in {"polygon", "polyline"}:
            points = element.attrib.get("points", "").replace(",", " ").split()
            count = max(0, len(points) // 2 - 1)
            per_path.append(count + (name == "polygon" and count > 1))
    return len(per_path), per_path


def _rgba(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"))


def _similarity(original: np.ndarray, candidate: np.ndarray) -> dict[str, float | None]:
    delta = np.abs(original.astype(np.int16) - candidate.astype(np.int16))
    original_alpha = original[:, :, 3] > 0
    candidate_alpha = candidate[:, :, 3] > 0
    union = np.count_nonzero(original_alpha | candidate_alpha)
    intersection = np.count_nonzero(original_alpha & candidate_alpha)
    mse = float(np.mean(np.square(delta.astype(np.float64))))
    return {
        "rgba_mae": float(delta.mean() / 255),
        "rgba_mse": mse / (255**2),
        "pixel_exact_fraction": float(np.all(delta == 0, axis=2).mean()),
        "alpha_iou": float(intersection / union) if union else 1.0,
        "psnr_db": float(20 * math.log10(255 / math.sqrt(mse))) if mse else None,
    }


def probe_one(row: dict[str, Any], config: StudyConfig) -> dict[str, Any]:
    source_path = config.raw_root / str(row["source_path"])
    source_bytes = source_path.read_bytes()
    audit_config = _audit_config(config)
    result: dict[str, Any] = {
        "hexcode": row["hexcode"],
        "annotation": row["annotation"],
        "group": row["group"],
        "source_path": row["source_path"],
        "source_sha256": row["source_svg_sha256"],
        "source_bytes": len(source_bytes),
        "source_transform_count": row["transform_count"],
        "source_stroke_element_count": row["stroke_element_count"],
    }
    source_paths, source_segments = _program_counts(source_bytes)
    result.update(
        {
            "semantic_proxy_path_count": source_paths,
            "semantic_proxy_segment_count": sum(source_segments),
            "semantic_proxy_max_segments_per_path": max(source_segments, default=0),
        }
    )
    try:
        outlined_text = SVG.fromstring(source_bytes).topicosvg().tostring()  # type: ignore[no-untyped-call]
        outlined_bytes = outlined_text.encode() if isinstance(outlined_text, str) else outlined_text
        _root, elements = _safe_parse(outlined_bytes, audit_config)
        outlined_style = _style_and_geometry(elements)
        outlined_paths, outlined_segments = _program_counts(outlined_bytes)
        result.update(
            {
                "outline_ok": True,
                "outline_error": None,
                "outline_bytes": len(outlined_bytes),
                "outline_sha256": hashlib.sha256(outlined_bytes).hexdigest(),
                "outline_path_count": outlined_paths,
                "outline_segment_count": sum(outlined_segments),
                "outline_max_segments_per_path": max(outlined_segments, default=0),
                "outline_stroke_element_count": outlined_style["stroke_element_count"],
                "path_expansion_ratio": outlined_paths / source_paths if source_paths else None,
                "segment_expansion_ratio": (
                    sum(outlined_segments) / sum(source_segments) if sum(source_segments) else None
                ),
            }
        )
        outline_path = config.derived_root / "outlined" / str(row["source_path"])
        outline_path.parent.mkdir(parents=True, exist_ok=True)
        outline_path.write_bytes(outlined_bytes)
        for size in config.render_sizes:
            original_png, original_rgba = _render(source_bytes, size, config.render_timeout_seconds)
            outlined_png, outlined_rgba = _render(
                outlined_bytes, size, config.render_timeout_seconds
            )
            del original_png, outlined_png
            for key, value in _similarity(original_rgba, outlined_rgba).items():
                result[f"{key}_{size}"] = value
    except Exception as exc:
        result["outline_ok"] = False
        result["outline_error"] = f"{type(exc).__name__}:{exc}"[:500]
    return result


def run_study(config: StudyConfig) -> dict[str, Any]:
    rows = cast(list[dict[str, Any]], pq.read_table(config.primary_manifest).to_pylist())
    fixture = select_fixture(rows, config)
    fixture_document = {
        "schema_version": 1,
        "probe_version": config.version,
        "selection_version": config.selection_version,
        "primary_manifest": str(config.primary_manifest),
        "primary_manifest_sha256": hashlib.sha256(config.primary_manifest.read_bytes()).hexdigest(),
        "rows": [
            {
                "hexcode": row["hexcode"],
                "group": row["group"],
                "source_path": row["source_path"],
                "source_svg_sha256": row["source_svg_sha256"],
            }
            for row in fixture
        ],
    }
    config.fixture_manifest.parent.mkdir(parents=True, exist_ok=True)
    config.fixture_manifest.write_text(
        json.dumps(fixture_document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    results = [probe_one(row, config) for row in fixture]
    summary = _summarize(results, config)
    config.report_root.mkdir(parents=True, exist_ok=True)
    with (config.report_root / "metrics.jsonl").open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, sort_keys=True, allow_nan=False) + "\n")
    (config.report_root / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )
    (config.report_root / "README.md").write_text(_markdown(summary), encoding="utf-8")
    return summary


def _finite_values(results: list[dict[str, Any]], field: str) -> np.ndarray:
    values = [float(result[field]) for result in results if result.get(field) is not None]
    return np.asarray([value for value in values if math.isfinite(value)], dtype=np.float64)


def _stats(results: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    values = _finite_values(results, field)
    return {
        "count": len(values),
        "median": float(np.median(values)) if len(values) else None,
        "p95": float(np.quantile(values, 0.95)) if len(values) else None,
        "max": float(values.max()) if len(values) else None,
    }


def _summarize(results: list[dict[str, Any]], config: StudyConfig) -> dict[str, Any]:
    successful = [result for result in results if result.get("outline_ok")]
    summary: dict[str, Any] = {
        "schema_version": 1,
        "probe_version": config.version,
        "source_revision": config.source_revision,
        "picosvg_version": version("picosvg"),
        "fixture_count": len(results),
        "group_counts": dict(sorted(_counts(result["group"] for result in results).items())),
        "outline_success_count": len(successful),
        "outline_failures": [
            {"hexcode": result["hexcode"], "error": result.get("outline_error")}
            for result in results
            if not result.get("outline_ok")
        ],
        "metrics": {
            field: _stats(successful, field)
            for field in (
                "semantic_proxy_path_count",
                "outline_path_count",
                "path_expansion_ratio",
                "semantic_proxy_segment_count",
                "outline_segment_count",
                "segment_expansion_ratio",
                "rgba_mae_72",
                "rgba_mae_18",
                "alpha_iou_72",
                "alpha_iou_18",
                "pixel_exact_fraction_72",
                "pixel_exact_fraction_18",
            )
        },
        "truncation": {},
    }
    for representation, path_field, segment_field in (
        ("semantic_proxy", "semantic_proxy_path_count", "semantic_proxy_max_segments_per_path"),
        ("outline", "outline_path_count", "outline_max_segments_per_path"),
    ):
        summary["truncation"][representation] = {
            "path_slots": {
                str(budget): sum(result.get(path_field, 0) > budget for result in successful)
                for budget in config.path_budgets
            },
            "segments_per_path": {
                str(budget): sum(result.get(segment_field, 0) > budget for result in successful)
                for budget in config.segment_budgets
            },
        }
    return summary


def _counts(values: Any) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for value in values:
        counts[str(value)] += 1
    return dict(counts)


def _markdown(summary: dict[str, Any]) -> str:
    metrics = summary["metrics"]
    lines = [
        "# Semantic-stroke versus outlined representation probe",
        "",
        f"Fixture: {summary['fixture_count']} primary-manifest icons; "
        f"PicoSVG successes: {summary['outline_success_count']}.",
        "",
        "This is a pre-codec structural proxy, not a representation selection. Source SVG",
        "primitives approximate semantic paths; PicoSVG materializes shapes, transforms,",
        "clip paths, and strokes into filled paths.",
        "",
        "| metric | median | p95 | max |",
        "|---|---:|---:|---:|",
    ]
    for field in (
        "semantic_proxy_path_count",
        "outline_path_count",
        "path_expansion_ratio",
        "semantic_proxy_segment_count",
        "outline_segment_count",
        "segment_expansion_ratio",
        "rgba_mae_72",
        "rgba_mae_18",
        "alpha_iou_72",
        "alpha_iou_18",
    ):
        stats = metrics[field]
        lines.append(
            f"| {field} | {_format(stats['median'])} | {_format(stats['p95'])} | "
            f"{_format(stats['max'])} |"
        )
    lines.extend(
        (
            "",
            "## Truncation counts",
            "",
            "```json",
            json.dumps(summary["truncation"], indent=2, sort_keys=True),
            "```",
            "",
        )
    )
    if summary["outline_failures"]:
        lines.extend(
            (
                "## Normalization failures",
                "",
                "```json",
                json.dumps(summary["outline_failures"], indent=2),
                "```",
                "",
            )
        )
    return "\n".join(lines)


def _format(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"
