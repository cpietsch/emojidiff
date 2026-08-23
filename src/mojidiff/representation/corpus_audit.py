"""Deterministic, no-render structural census for a curated SVG manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, BinaryIO, cast

import numpy as np
import pyarrow.parquet as pq
import yaml
from picosvg.svg import SVG

from mojidiff.representation.normalizer import NormalizationError, normalize_svg
from mojidiff.representation.program import FloatProgram, SegmentType

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_COLOR = re.compile(r"#[0-9a-f]{6}\Z")
_MAX_CONFIG_BYTES = 1_000_000
_MANIFEST_COLUMNS = (
    "source_revision",
    "source_path",
    "source_svg_sha256",
    "hexcode",
    "group",
    "subgroup",
    "variant_family_id",
    "curation_status",
    "split",
    "split_family_cluster",
)


class CorpusAuditError(RuntimeError):
    """The structural census cannot be reproduced from its declared inputs."""


@dataclass(frozen=True)
class AuditLimits:
    max_manifest_bytes: int
    max_manifest_rows: int
    max_source_bytes: int
    max_outlined_bytes: int
    max_error_chars: int
    tail_count: int


@dataclass(frozen=True)
class CorpusAuditConfig:
    """All inputs and bounds that define one structural census."""

    version: str
    source_revision: str
    raw_root: Path
    primary_manifest: Path
    primary_manifest_sha256: str
    expected_manifest_rows: int
    palette_path: Path
    palette_sha256: str
    report_root: Path
    path_slots: tuple[int, ...]
    segments_per_path: tuple[int, ...]
    stroke_widths: tuple[float, ...]
    miter_limits: tuple[float, ...]
    dash_patterns: tuple[tuple[float, ...], ...]
    limits: AuditLimits


@dataclass(frozen=True)
class ManifestRow:
    source_revision: str
    source_path: str
    source_svg_sha256: str
    hexcode: str
    group: str
    subgroup: str
    variant_family_id: str
    curation_status: str
    split: str
    split_family_cluster: str

    def identity(self) -> dict[str, Any]:
        return asdict(self)


def load_corpus_audit_config(path: Path) -> CorpusAuditConfig:
    """Load an exact bounded config; no research defaults are inferred."""

    raw = _read_bounded(path, _MAX_CONFIG_BYTES, "config")
    try:
        document = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        raise CorpusAuditError(f"invalid YAML config: {exc}") from exc
    root = _mapping(document, "root")
    if root.get("schema_version") != 1:
        raise CorpusAuditError("corpus audit schema_version must be 1")
    capacity = _mapping(root.get("capacity"), "capacity")
    style = _mapping(root.get("style_vocabulary"), "style_vocabulary")
    raw_limits = _mapping(root.get("limits"), "limits")
    limits = AuditLimits(
        max_manifest_bytes=_positive_int(
            raw_limits.get("max_manifest_bytes"), "limits.max_manifest_bytes"
        ),
        max_manifest_rows=_positive_int(
            raw_limits.get("max_manifest_rows"), "limits.max_manifest_rows"
        ),
        max_source_bytes=_positive_int(
            raw_limits.get("max_source_bytes"), "limits.max_source_bytes"
        ),
        max_outlined_bytes=_positive_int(
            raw_limits.get("max_outlined_bytes"), "limits.max_outlined_bytes"
        ),
        max_error_chars=_positive_int(raw_limits.get("max_error_chars"), "limits.max_error_chars"),
        tail_count=_positive_int(raw_limits.get("tail_count"), "limits.tail_count"),
    )
    paths = _int_tuple(capacity.get("path_slots"), "capacity.path_slots")
    segments = _int_tuple(capacity.get("segments_per_path"), "capacity.segments_per_path")
    if tuple(sorted(set(paths))) != paths or tuple(sorted(set(segments))) != segments:
        raise CorpusAuditError("capacity candidates must be unique and increasing")
    if paths[-1] > 512 or segments[-1] > 1024 or paths[-1] * segments[-1] > 100_000:
        raise CorpusAuditError("capacity candidates exceed renderer-safe codec bounds")
    expected_rows = _positive_int(root.get("expected_manifest_rows"), "expected_manifest_rows")
    if expected_rows > limits.max_manifest_rows:
        raise CorpusAuditError("expected manifest rows exceed the configured row bound")
    config = CorpusAuditConfig(
        version=_string(root, "audit_version"),
        source_revision=_string(root, "source_revision"),
        raw_root=Path(_string(root, "raw_root")),
        primary_manifest=Path(_string(root, "primary_manifest")),
        primary_manifest_sha256=_sha256(
            _string(root, "primary_manifest_sha256"), "primary_manifest_sha256"
        ),
        expected_manifest_rows=expected_rows,
        palette_path=Path(_string(root, "palette_path")),
        palette_sha256=_sha256(_string(root, "palette_sha256"), "palette_sha256"),
        report_root=Path(_string(root, "report_root")),
        path_slots=paths,
        segments_per_path=segments,
        stroke_widths=_positive_floats(
            style.get("stroke_widths"), "style_vocabulary.stroke_widths"
        ),
        miter_limits=_positive_floats(style.get("miter_limits"), "style_vocabulary.miter_limits"),
        dash_patterns=tuple(
            _nonnegative_floats(item, f"style_vocabulary.dash_patterns[{index}]")
            for index, item in enumerate(
                _sequence(style.get("dash_patterns"), "style_vocabulary.dash_patterns")
            )
        ),
        limits=limits,
    )
    if len(set(config.stroke_widths)) != len(config.stroke_widths) or len(
        set(config.miter_limits)
    ) != len(config.miter_limits):
        raise CorpusAuditError("style vocabulary values must be unique")
    if len(set(config.dash_patterns)) != len(config.dash_patterns):
        raise CorpusAuditError("dash patterns must be unique")
    if any(
        not item or len(item) % 2 or len(item) > 32 or not any(item)
        for item in config.dash_patterns
    ):
        raise CorpusAuditError("dash patterns must be nonzero even tuples of at most 32 values")
    raw_root = config.raw_root.resolve()
    report_root = config.report_root.resolve()
    if (
        report_root == raw_root
        or raw_root in report_root.parents
        or report_root in raw_root.parents
    ):
        raise CorpusAuditError("report_root must be separate from the immutable raw tree")
    return config


def run_corpus_audit(config: CorpusAuditConfig, config_path: Path) -> dict[str, Any]:
    """Normalize every manifest row twice and write deterministic compact evidence."""

    config_sha = hashlib.sha256(_read_bounded(config_path, _MAX_CONFIG_BYTES, "config")).hexdigest()
    source_code_hashes = _source_code_hashes()
    code_identity = {**_git_identity(), **source_code_hashes}
    palette = _load_palette(config)
    manifest_rows, manifest_sha = _load_manifest(config)
    attempts: list[dict[str, Any]] = []
    hybrids: list[dict[str, Any]] = []
    source_set = hashlib.sha256()

    for item in manifest_rows:
        source = _read_source(config, item)
        source_set.update(
            item.source_path.encode("utf-8")
            + b"\0"
            + item.source_svg_sha256.encode("ascii")
            + b"\n"
        )
        identity = item.identity()
        semantic = _normalize_attempt(
            identity, "semantic", source, item.source_svg_sha256, config, palette
        )
        attempts.append(semantic)
        outlined = _outlined_attempt(identity, source, config, palette)
        attempts.append(outlined)
        hybrids.append(_hybrid_row(identity, semantic, outlined))

    attempts.sort(
        key=lambda row: (str(row["source_path"]), 0 if row["representation"] == "semantic" else 1)
    )
    hybrids.sort(key=lambda row: str(row["source_path"]))
    if len(attempts) != 2 * len(manifest_rows) or len(hybrids) != len(manifest_rows):
        raise CorpusAuditError("internal row accounting violated the two-attempt contract")
    if _source_code_hashes() != source_code_hashes:
        raise CorpusAuditError("audit source code changed while the census was running")

    config.report_root.mkdir(parents=True, exist_ok=True)
    attempts_path = config.report_root / "attempts.jsonl"
    hybrids_path = config.report_root / "hybrid.jsonl"
    _write_jsonl(attempts_path, attempts)
    _write_jsonl(hybrids_path, hybrids)
    summary = _summarize(
        config,
        config_sha,
        manifest_sha,
        source_set.hexdigest(),
        palette,
        attempts,
        hybrids,
        attempts_path,
        hybrids_path,
        code_identity,
    )
    _write_text_artifact(
        config.report_root / "summary.json",
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
    )
    _write_text_artifact(config.report_root / "README.md", _markdown(summary))
    return summary


def _normalize_attempt(
    identity: dict[str, Any],
    representation: str,
    candidate: bytes,
    candidate_sha256: str,
    config: CorpusAuditConfig,
    palette: tuple[str, ...],
) -> dict[str, Any]:
    base = {
        **identity,
        "representation": representation,
        "prepare_ok": True,
        "candidate_sha256": candidate_sha256,
    }
    try:
        normalized = normalize_svg(candidate)
    except NormalizationError as exc:
        return _failed_attempt(
            identity,
            representation,
            prepare_ok=True,
            candidate_sha256=candidate_sha256,
            stage="normalize",
            code=exc.code,
            error=str(exc),
            limit=config.limits.max_error_chars,
        )
    return {
        **base,
        "normalize_ok": True,
        "error_stage": None,
        "error_code": None,
        "error": None,
        **_program_features(normalized.program, asdict(normalized.report), config, palette),
    }


def _outlined_attempt(
    identity: dict[str, Any],
    source: bytes,
    config: CorpusAuditConfig,
    palette: tuple[str, ...],
) -> dict[str, Any]:
    try:
        rendered = SVG.fromstring(source).topicosvg().tostring()  # type: ignore[no-untyped-call]
        outlined = rendered.encode("utf-8") if isinstance(rendered, str) else rendered
        if not isinstance(outlined, bytes):
            raise TypeError("PicoSVG returned a non-bytes/non-text value")
        if len(outlined) > config.limits.max_outlined_bytes:
            return _failed_attempt(
                identity,
                "outlined",
                prepare_ok=False,
                candidate_sha256=None,
                stage="prepare",
                code="outlined_size_limit",
                error=f"outlined SVG has {len(outlined)} bytes",
                limit=config.limits.max_error_chars,
            )
    except Exception as exc:
        return _failed_attempt(
            identity,
            "outlined",
            prepare_ok=False,
            candidate_sha256=None,
            stage="prepare",
            code="outline_error",
            error=f"{type(exc).__name__}: {exc}",
            limit=config.limits.max_error_chars,
        )
    digest = hashlib.sha256(outlined).hexdigest()
    return _normalize_attempt(identity, "outlined", outlined, digest, config, palette)


def _failed_attempt(
    identity: dict[str, Any],
    representation: str,
    *,
    prepare_ok: bool,
    candidate_sha256: str | None,
    stage: str,
    code: str,
    error: str,
    limit: int,
) -> dict[str, Any]:
    return {
        **identity,
        "representation": representation,
        "prepare_ok": prepare_ok,
        "candidate_sha256": candidate_sha256,
        "normalize_ok": False,
        "error_stage": stage,
        "error_code": code,
        "error": error[:limit],
    }


def _program_features(
    program: FloatProgram,
    report: dict[str, Any],
    config: CorpusAuditConfig,
    palette: tuple[str, ...],
) -> dict[str, Any]:
    lengths = [len(contour.segments) for contour in program.contours]
    layers = [contour.layer for contour in program.contours]
    kinds = Counter(segment.kind for contour in program.contours for segment in contour.segments)
    x_values: list[float] = []
    y_values: list[float] = []
    color_counts: Counter[str] = Counter()
    for contour in program.contours:
        x_values.append(contour.start[0])
        y_values.append(contour.start[1])
        for segment in contour.segments:
            x_values.extend(segment.coords[0::2])
            y_values.extend(segment.coords[1::2])
        if contour.fill is not None:
            color_counts[contour.fill] += 1
        if contour.stroke is not None:
            color_counts[contour.stroke] += 1
    colors = sorted(color_counts)
    palette_outliers = sorted(set(colors) - set(palette))
    widths = Counter(
        contour.stroke_width for contour in program.contours if contour.stroke is not None
    )
    miters = Counter(
        contour.miter_limit for contour in program.contours if contour.stroke is not None
    )
    dashes = Counter(
        contour.dash_pattern
        for contour in program.contours
        if contour.stroke is not None and contour.dash_pattern
    )
    return {
        **report,
        "layers": len(set(layers)),
        "contour_segment_lengths": lengths,
        "contour_layers": layers,
        "max_segments_per_contour": max(lengths, default=0),
        "segment_type_counts": {
            "line": kinds[SegmentType.LINE],
            "quad": kinds[SegmentType.QUAD],
            "cubic": kinds[SegmentType.CUBIC],
            "close": kinds[SegmentType.CLOSE],
        },
        "open_contours": sum(
            not contour.segments or contour.segments[-1].kind != SegmentType.CLOSE
            for contour in program.contours
        ),
        "closed_contours": sum(
            bool(contour.segments) and contour.segments[-1].kind == SegmentType.CLOSE
            for contour in program.contours
        ),
        "coordinate_scalar_count": len(x_values) + len(y_values),
        "coordinate_min": min((*x_values, *y_values), default=None),
        "coordinate_max": max((*x_values, *y_values), default=None),
        "coordinate_x_min": min(x_values, default=None),
        "coordinate_x_max": max(x_values, default=None),
        "coordinate_y_min": min(y_values, default=None),
        "coordinate_y_max": max(y_values, default=None),
        "colors": colors,
        "color_counts": dict(sorted(color_counts.items())),
        "palette_outliers": palette_outliers,
        "fill_colors": sorted(
            {contour.fill for contour in program.contours if contour.fill is not None}
        ),
        "stroke_colors": sorted(
            {contour.stroke for contour in program.contours if contour.stroke is not None}
        ),
        "stroke_width_counts": _numeric_counts(widths),
        "miter_limit_counts": _numeric_counts(miters),
        "dash_pattern_counts": [
            {"value": list(value), "count": count} for value, count in sorted(dashes.items())
        ],
        "linecap_counts": dict(
            sorted(
                Counter(
                    contour.linecap for contour in program.contours if contour.stroke is not None
                ).items()
            )
        ),
        "linejoin_counts": dict(
            sorted(
                Counter(
                    contour.linejoin for contour in program.contours if contour.stroke is not None
                ).items()
            )
        ),
        "fill_rule_counts": dict(
            sorted(
                Counter(
                    contour.fill_rule for contour in program.contours if contour.fill is not None
                ).items()
            )
        ),
        "style_vocabulary": {
            "stroke_width": _vocabulary_fit(widths, config.stroke_widths),
            "miter_limit": _vocabulary_fit(miters, config.miter_limits),
            "dash_patterns_outside": [
                list(value) for value in sorted(set(dashes) - set(config.dash_patterns))
            ],
            "exact": not (
                set(widths) - set(config.stroke_widths)
                or set(miters) - set(config.miter_limits)
                or set(dashes) - set(config.dash_patterns)
            ),
        },
    }


def _numeric_counts(values: Counter[float]) -> list[dict[str, Any]]:
    return [{"value": value, "count": count} for value, count in sorted(values.items())]


def _vocabulary_fit(values: Counter[float], vocabulary: tuple[float, ...]) -> dict[str, Any]:
    outside = sorted(set(values) - set(vocabulary))
    errors = [
        (
            abs(value - min(vocabulary, key=lambda candidate: (abs(candidate - value), candidate))),
            value,
        )
        for value, count in values.items()
        for _ in range(count)
    ]
    return {
        "contours": sum(values.values()),
        "nonexact_contours": sum(
            count for value, count in values.items() if value not in vocabulary
        ),
        "outside_values": outside,
        "max_absolute_error": max((error for error, _value in errors), default=0.0),
        "max_relative_error": max((error / value for error, value in errors), default=0.0),
    }


def _hybrid_row(
    identity: dict[str, Any], semantic: dict[str, Any], outlined: dict[str, Any]
) -> dict[str, Any]:
    if semantic["normalize_ok"]:
        selected, route = semantic, "semantic_native"
    elif outlined["normalize_ok"]:
        selected = outlined
        route = f"outlined_fallback:{semantic['error_code']}"
    else:
        return {
            **identity,
            "route": "unsupported_both",
            "selected_representation": None,
            "semantic_error_code": semantic["error_code"],
            "outlined_error_code": outlined["error_code"],
        }
    metadata = {
        *identity,
        "representation",
        "prepare_ok",
        "normalize_ok",
        "error_stage",
        "error_code",
        "error",
    }
    return {
        **identity,
        "route": route,
        "selected_representation": selected["representation"],
        "semantic_error_code": semantic["error_code"],
        "outlined_error_code": outlined["error_code"],
        **{key: value for key, value in selected.items() if key not in metadata},
    }


def _capacity_loss(row: dict[str, Any], paths: int, segments: int) -> dict[str, Any]:
    lengths = cast(list[int], row["contour_segment_lengths"])
    layers = cast(list[int], row["contour_layers"])
    dropped_p = sum(lengths[paths:])
    dropped_s = sum(max(0, length - segments) for length in lengths[:paths])
    omitted_layers = {
        layer
        for index, (length, layer) in enumerate(zip(lengths, layers, strict=True))
        if index >= paths or length > segments
    }
    retained_layers = {
        layer
        for index, layer in enumerate(layers)
        if index < paths and min(lengths[index], segments) > 0
    }
    retained = sum(min(length, segments) for length in lengths[:paths])
    return {
        "dropped_contours": max(0, len(lengths) - paths),
        "segments_dropped_by_path_budget": dropped_p,
        "segments_dropped_by_segment_budget": dropped_s,
        "dropped_segments": dropped_p + dropped_s,
        "retained_segments": retained,
        "damaged_layers": len(omitted_layers),
        "partial_layers": len(omitted_layers & retained_layers),
        "fully_dropped_layers": len(omitted_layers - retained_layers),
    }


def _summarize(
    config: CorpusAuditConfig,
    config_sha: str,
    manifest_sha: str,
    source_set_sha: str,
    palette: tuple[str, ...],
    attempts: list[dict[str, Any]],
    hybrids: list[dict[str, Any]],
    attempts_path: Path,
    hybrids_path: Path,
    code_identity: dict[str, Any],
) -> dict[str, Any]:
    representations = {
        name: _representation_summary(
            [row for row in attempts if row["representation"] == name], config, palette
        )
        for name in ("semantic", "outlined")
    }
    semantic = {
        str(row["source_path"]): row
        for row in attempts
        if row["representation"] == "semantic" and row["normalize_ok"]
    }
    outlined = {
        str(row["source_path"]): row
        for row in attempts
        if row["representation"] == "outlined" and row["normalize_ok"]
    }
    paired_paths = sorted(set(semantic) & set(outlined))
    paired = {
        "intersection_count": len(paired_paths),
        "contour_expansion": _stats(
            [
                float(outlined[path]["contours"]) / float(semantic[path]["contours"])
                for path in paired_paths
                if semantic[path]["contours"]
            ]
        ),
        "segment_expansion": _stats(
            [
                float(outlined[path]["segments"]) / float(semantic[path]["segments"])
                for path in paired_paths
                if semantic[path]["segments"]
            ]
        ),
    }
    hybrid_success = [row for row in hybrids if row["selected_representation"] is not None]
    return {
        "schema_version": 1,
        "audit_version": config.version,
        "source_revision": config.source_revision,
        "config_sha256": config_sha,
        "primary_manifest": str(config.primary_manifest),
        "primary_manifest_sha256": manifest_sha,
        "manifest_rows": len(hybrids),
        "source_set_sha256": source_set_sha,
        "palette_path": str(config.palette_path),
        "palette_sha256": config.palette_sha256,
        "picosvg_version": version("picosvg"),
        "numpy_version": version("numpy"),
        "pyarrow_version": version("pyarrow"),
        "code_identity": code_identity,
        "attempts_sha256": _file_sha256(attempts_path),
        "hybrid_sha256": _file_sha256(hybrids_path),
        "representations": representations,
        "paired_expansion": paired,
        "hybrid": {
            "denominator": len(hybrids),
            "routes": dict(sorted(Counter(str(row["route"]) for row in hybrids).items())),
            "successful": len(hybrid_success),
            "unsupported": len(hybrids) - len(hybrid_success),
            "structure": _success_summary(hybrid_success, config, palette),
            "by_group": _breakdown(hybrids, "group"),
            "by_split": _breakdown(hybrids, "split"),
        },
    }


def _representation_summary(
    rows: list[dict[str, Any]], config: CorpusAuditConfig, palette: tuple[str, ...]
) -> dict[str, Any]:
    successful = [row for row in rows if row["normalize_ok"]]
    failures = [row for row in rows if not row["normalize_ok"]]
    return {
        "attempted": len(rows),
        "successful": len(successful),
        "error_codes": dict(sorted(Counter(str(row["error_code"]) for row in failures).items())),
        "failures": [
            {
                "source_path": row["source_path"],
                "stage": row["error_stage"],
                "code": row["error_code"],
            }
            for row in failures
        ],
        "structure": _success_summary(successful, config, palette),
        "by_group": _breakdown(rows, "group"),
        "by_split": _breakdown(rows, "split"),
    }


def _success_summary(
    rows: list[dict[str, Any]], config: CorpusAuditConfig, palette: tuple[str, ...]
) -> dict[str, Any]:
    colors: Counter[str] = Counter()
    linecaps: Counter[str] = Counter()
    linejoins: Counter[str] = Counter()
    fill_rules: Counter[str] = Counter()
    outlier_paths: defaultdict[str, list[str]] = defaultdict(list)
    for row in rows:
        colors.update(cast(dict[str, int], row.get("color_counts", {})))
        linecaps.update(cast(dict[str, int], row.get("linecap_counts", {})))
        linejoins.update(cast(dict[str, int], row.get("linejoin_counts", {})))
        fill_rules.update(cast(dict[str, int], row.get("fill_rule_counts", {})))
        for color in cast(list[str], row.get("palette_outliers", [])):
            outlier_paths[color].append(str(row["source_path"]))
    return {
        "icons": len(rows),
        "distributions": {
            field: _stats([float(row[field]) for row in rows])
            for field in (
                "contours",
                "layers",
                "segments",
                "max_segments_per_contour",
                "out_of_bounds_coordinates",
            )
        },
        "segment_type_counts": {
            kind: sum(int(cast(dict[str, int], row["segment_type_counts"])[kind]) for row in rows)
            for kind in ("line", "quad", "cubic", "close")
        }
        if rows and "segment_type_counts" in rows[0]
        else {},
        "coordinates": {
            "min": min(
                (
                    float(row["coordinate_min"])
                    for row in rows
                    if row.get("coordinate_min") is not None
                ),
                default=None,
            ),
            "max": max(
                (
                    float(row["coordinate_max"])
                    for row in rows
                    if row.get("coordinate_max") is not None
                ),
                default=None,
            ),
            "icons_out_of_bounds": sum(int(row["out_of_bounds_coordinates"]) > 0 for row in rows),
            "scalar_values_out_of_bounds": sum(
                int(row["out_of_bounds_coordinates"]) for row in rows
            ),
        },
        "palette": {
            "configured_entries": len(palette),
            "used_entries": len(set(colors) & set(palette)),
            "colors": dict(sorted(colors.items())),
            "outlier_paths": {key: sorted(value) for key, value in sorted(outlier_paths.items())},
        },
        "icons_with_nonexact_style_vocabulary": sum(
            not cast(dict[str, Any], row["style_vocabulary"])["exact"] for row in rows
        ),
        "style_counts": {
            "linecaps": dict(sorted(linecaps.items())),
            "linejoins": dict(sorted(linejoins.items())),
            "fill_rules": dict(sorted(fill_rules.items())),
            "stroke_widths": _aggregate_value_counts(rows, "stroke_width_counts"),
            "miter_limits": _aggregate_value_counts(rows, "miter_limit_counts"),
            "dash_patterns": _aggregate_value_counts(rows, "dash_pattern_counts"),
        },
        "capacity": _capacity_summary(rows, config),
        "tails": {
            field: _worst(rows, field, config.limits.tail_count)
            for field in ("contours", "segments", "max_segments_per_contour")
        },
    }


def _aggregate_value_counts(rows: list[dict[str, Any]], field: str) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    values: dict[str, object] = {}
    for row in rows:
        for item in cast(list[dict[str, Any]], row.get(field, [])):
            value = item["value"]
            key = json.dumps(value, sort_keys=True, allow_nan=False)
            values[key] = value
            counts[key] += int(item["count"])
    return [{"value": values[key], "count": counts[key]} for key in sorted(counts)]


def _capacity_summary(rows: list[dict[str, Any]], config: CorpusAuditConfig) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for paths in config.path_slots:
        for segments in config.segments_per_path:
            losses = [(row, _capacity_loss(row, paths, segments)) for row in rows]
            total_segments = sum(int(row["segments"]) for row in rows)
            dropped = sum(int(loss["dropped_segments"]) for _row, loss in losses)
            affected = [
                (row, loss)
                for row, loss in losses
                if loss["dropped_segments"] or loss["dropped_contours"]
            ]
            result[f"p{paths}-s{segments}"] = {
                "path_slots": paths,
                "segments_per_path": segments,
                "program_slots": paths * segments,
                "icons": len(rows),
                "zero_loss_icons": len(rows) - len(affected),
                "icons_with_path_loss": sum(
                    int(loss["dropped_contours"]) > 0 for _row, loss in losses
                ),
                "icons_with_segment_loss": sum(
                    int(loss["segments_dropped_by_segment_budget"]) > 0 for _row, loss in losses
                ),
                "dropped_contours": sum(int(loss["dropped_contours"]) for _row, loss in losses),
                "segments_dropped_by_path_budget": sum(
                    int(loss["segments_dropped_by_path_budget"]) for _row, loss in losses
                ),
                "segments_dropped_by_segment_budget": sum(
                    int(loss["segments_dropped_by_segment_budget"]) for _row, loss in losses
                ),
                "dropped_segments": dropped,
                "dropped_segment_fraction": dropped / total_segments if total_segments else 0.0,
                "slot_utilization": sum(int(loss["retained_segments"]) for _row, loss in losses)
                / (len(rows) * paths * segments)
                if rows
                else 0.0,
                "damaged_layers": sum(int(loss["damaged_layers"]) for _row, loss in losses),
                "partial_layers": sum(int(loss["partial_layers"]) for _row, loss in losses),
                "fully_dropped_layers": sum(
                    int(loss["fully_dropped_layers"]) for _row, loss in losses
                ),
                "affected_icon_dropped_segments": _stats(
                    [float(loss["dropped_segments"]) for _row, loss in affected]
                ),
                "worst": [
                    {"source_path": row["source_path"], "hexcode": row["hexcode"], **loss}
                    for row, loss in sorted(
                        affected,
                        key=lambda item: (
                            -int(item[1]["dropped_segments"]),
                            str(item[0]["source_path"]),
                        ),
                    )[: config.limits.tail_count]
                ],
            }
    return result


def _breakdown(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        key: {
            "rows": len(items),
            "successful": sum(
                bool(item.get("normalize_ok", item.get("selected_representation") is not None))
                for item in items
            ),
            "errors_or_unsupported": sum(
                not bool(item.get("normalize_ok", item.get("selected_representation") is not None))
                for item in items
            ),
        }
        for key, items in sorted(groups.items())
    }


def _stats(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    return {
        "count": len(array),
        "min": float(array.min()) if len(array) else None,
        "median": float(np.quantile(array, 0.50, method="higher")) if len(array) else None,
        "p90": float(np.quantile(array, 0.90, method="higher")) if len(array) else None,
        "p95": float(np.quantile(array, 0.95, method="higher")) if len(array) else None,
        "p99": float(np.quantile(array, 0.99, method="higher")) if len(array) else None,
        "p99_5": float(np.quantile(array, 0.995, method="higher")) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def _worst(rows: list[dict[str, Any]], field: str, count: int) -> list[dict[str, Any]]:
    return [
        {"source_path": row["source_path"], "hexcode": row["hexcode"], field: row[field]}
        for row in sorted(rows, key=lambda row: (-float(row[field]), str(row["source_path"])))[
            :count
        ]
    ]


def _load_manifest(config: CorpusAuditConfig) -> tuple[list[ManifestRow], str]:
    raw = _read_bounded(
        config.primary_manifest, config.limits.max_manifest_bytes, "primary manifest"
    )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != config.primary_manifest_sha256:
        raise CorpusAuditError("primary manifest hash mismatch")
    try:
        table = pq.read_table(config.primary_manifest, columns=list(_MANIFEST_COLUMNS))
    except Exception as exc:
        raise CorpusAuditError(f"primary manifest could not be read: {exc}") from exc
    if (
        table.num_rows != config.expected_manifest_rows
        or table.num_rows > config.limits.max_manifest_rows
    ):
        raise CorpusAuditError("primary manifest row count does not match the pinned config")
    rows: list[ManifestRow] = []
    seen: set[str] = set()
    for index, raw_row in enumerate(table.to_pylist()):
        row = _mapping(raw_row, f"manifest row {index}")
        values = {name: _required_row_string(row, name, index) for name in _MANIFEST_COLUMNS}
        item = ManifestRow(**values)
        if item.source_revision != config.source_revision:
            raise CorpusAuditError(f"manifest row {index} has the wrong source revision")
        _sha256(item.source_svg_sha256, f"manifest row {index} source hash")
        if item.source_path in seen:
            raise CorpusAuditError(f"duplicate source path in manifest: {item.source_path}")
        if (
            item.group == "flags"
            or not item.split.startswith("primary/")
            or item.curation_status not in {"include", "include_override"}
        ):
            raise CorpusAuditError(
                f"manifest row is not in the curated primary dataset: {item.source_path}"
            )
        seen.add(item.source_path)
        rows.append(item)
    if (
        hashlib.sha256(
            _read_bounded(
                config.primary_manifest, config.limits.max_manifest_bytes, "primary manifest"
            )
        ).hexdigest()
        != digest
    ):
        raise CorpusAuditError("primary manifest changed while it was being read")
    return sorted(rows, key=lambda item: item.source_path), digest


def _read_source(config: CorpusAuditConfig, row: ManifestRow) -> bytes:
    relative = Path(row.source_path)
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".svg":
        raise CorpusAuditError(f"unsafe source path in manifest: {row.source_path}")
    raw_root = config.raw_root.resolve(strict=True)
    source_path = (raw_root / relative).resolve(strict=True)
    try:
        source_path.relative_to(raw_root)
    except ValueError as exc:
        raise CorpusAuditError(f"source path escapes raw root: {row.source_path}") from exc
    source = _read_bounded(source_path, config.limits.max_source_bytes, f"source {row.source_path}")
    if hashlib.sha256(source).hexdigest() != row.source_svg_sha256:
        raise CorpusAuditError(f"source hash mismatch: {row.source_path}")
    return source


def _load_palette(config: CorpusAuditConfig) -> tuple[str, ...]:
    raw = _read_bounded(config.palette_path, config.limits.max_source_bytes, "palette")
    if hashlib.sha256(raw).hexdigest() != config.palette_sha256:
        raise CorpusAuditError("palette hash mismatch")
    try:
        document = _mapping(json.loads(raw), "palette")
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise CorpusAuditError(f"invalid palette JSON: {exc}") from exc
    values: list[str] = []
    for value in _sequence(document.get("colors"), "palette.colors"):
        values.append(_color(value, "palette.colors"))
    for group in _mapping(document.get("skintones"), "palette.skintones").values():
        for value in _sequence(group, "palette.skintones entry"):
            values.append(_color(value, "palette.skintones entry"))
    result = tuple(dict.fromkeys(values))
    if not result:
        raise CorpusAuditError("palette is empty")
    return result


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    def write(handle: BinaryIO) -> None:
        for row in rows:
            handle.write((json.dumps(row, sort_keys=True, allow_nan=False) + "\n").encode("utf-8"))

    _install_artifact(path, write)


def _write_text_artifact(path: Path, value: str) -> None:
    payload = value.encode("utf-8")

    def write(handle: BinaryIO) -> None:
        handle.write(payload)

    _install_artifact(path, write)


def _install_artifact(path: Path, write: Callable[[BinaryIO], None]) -> None:
    """Atomically create one immutable artifact, accepting an exact prior copy."""

    if path.is_symlink():
        raise CorpusAuditError(f"refusing to replace symlink artifact: {path}")
    if path.exists() and not path.is_file():
        raise CorpusAuditError(f"artifact target is not a regular file: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            write(handle)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.chmod(0o644)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or not path.is_file():
                raise CorpusAuditError(f"artifact target is not a regular file: {path}") from None
            if not _files_equal(temporary, path):
                raise CorpusAuditError(f"refusing to replace differing artifact: {path}") from None
        except OSError as exc:
            raise CorpusAuditError(f"artifact could not be installed: {path}: {exc}") from exc
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


def _markdown(summary: dict[str, Any]) -> str:
    semantic = summary["representations"]["semantic"]
    outlined = summary["representations"]["outlined"]
    count = summary["manifest_rows"]
    semantic_structure = semantic["structure"]["distributions"]
    outlined_structure = outlined["structure"]["distributions"]
    return "\n".join(
        (
            "# Full primary structural codec census",
            "",
            f"Pinned manifest: {count} icons; exactly {2 * count} representation attempts.",
            "",
            "| representation | successful | failed | median contours | median segments |",
            "|---|---:|---:|---:|---:|",
            f"| semantic | {semantic['successful']} | "
            f"{semantic['attempted'] - semantic['successful']} | "
            f"{semantic_structure['contours']['median']} | "
            f"{semantic_structure['segments']['median']} |",
            f"| outlined | {outlined['successful']} | "
            f"{outlined['attempted'] - outlined['successful']} | "
            f"{outlined_structure['contours']['median']} | "
            f"{outlined_structure['segments']['median']} |",
            "",
            "`attempts.jsonl` preserves ordered contour lengths/layers and exact styles; "
            "`hybrid.jsonl` records semantic-native versus outlined-fallback routing. "
            "Candidate capacity loss is exact in `summary.json` (CLOSE consumes a slot). "
            "Percentiles use higher order statistics and are not integer safety bounds.",
            "",
        )
    )


def _git_identity() -> dict[str, Any]:
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
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": None, "git_dirty_at_start": None}
    return {
        "git_commit": commit,
        "git_dirty_at_start": bool(status),
        "git_status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
    }


def _source_code_hashes() -> dict[str, str]:
    representation_root = Path(__file__).parent
    return {
        "corpus_audit_sha256": _file_sha256(Path(__file__)),
        "normalizer_sha256": _file_sha256(representation_root / "normalizer.py"),
        "program_sha256": _file_sha256(representation_root / "program.py"),
    }


def _read_bounded(path: Path, limit: int, field: str) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise CorpusAuditError(f"{field} is unavailable: {exc}") from exc
    if size > limit:
        raise CorpusAuditError(f"{field} exceeds its {limit}-byte bound")
    try:
        result = path.read_bytes()
    except OSError as exc:
        raise CorpusAuditError(f"{field} could not be read: {exc}") from exc
    if len(result) > limit:
        raise CorpusAuditError(f"{field} exceeds its {limit}-byte bound")
    return result


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CorpusAuditError(f"{field} must be a mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise CorpusAuditError(f"{field} must be a list")
    return value


def _string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise CorpusAuditError(f"{field} must be a non-empty string")
    return value


def _required_row_string(row: dict[str, Any], field: str, index: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value:
        raise CorpusAuditError(f"manifest row {index} field {field} must be a non-empty string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CorpusAuditError(f"{field} must be a positive integer")
    return value


def _int_tuple(value: object, field: str) -> tuple[int, ...]:
    result = tuple(_positive_int(item, field) for item in _sequence(value, field))
    if not result:
        raise CorpusAuditError(f"{field} cannot be empty")
    return result


def _positive_floats(value: object, field: str) -> tuple[float, ...]:
    result = _nonnegative_floats(value, field)
    if not result or any(item <= 0 for item in result):
        raise CorpusAuditError(f"{field} must contain positive values")
    return result


def _nonnegative_floats(value: object, field: str) -> tuple[float, ...]:
    result: list[float] = []
    for item in _sequence(value, field):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise CorpusAuditError(f"{field} must contain numbers")
        number = float(item)
        if not math.isfinite(number) or number < 0:
            raise CorpusAuditError(f"{field} must contain finite nonnegative values")
        result.append(number)
    return tuple(result)


def _sha256(value: str, field: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise CorpusAuditError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _color(value: object, field: str) -> str:
    if not isinstance(value, str) or _COLOR.fullmatch(value.lower()) is None:
        raise CorpusAuditError(f"{field} contains an invalid color")
    return value.lower()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mojidiff.representation.corpus_audit")
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args(argv)
    summary = run_corpus_audit(load_corpus_audit_config(args.config), args.config)
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
