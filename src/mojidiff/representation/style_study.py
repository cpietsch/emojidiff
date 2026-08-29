"""Deterministic corpus audit and render probe for categorical SVG styles."""

from __future__ import annotations

import argparse
import bisect
import hashlib
import io
import json
import math
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml
from picosvg.svg import SVG
from PIL import Image, ImageDraw

from mojidiff.curation.audit import _render
from mojidiff.representation.codec_study import _palette, _tensor_hash, _write_bytes_artifact
from mojidiff.representation.normalizer import normalize_svg
from mojidiff.representation.program import (
    CodecConfig,
    decode_program,
    encode_program,
    serialize_svg,
)
from mojidiff.representation.study import _similarity

_MAX_INPUT_BYTES = 64 * 1024 * 1024


class StyleStudyError(RuntimeError):
    """A style-vocabulary study cannot be reproduced from its declared inputs."""


@dataclass(frozen=True)
class WidthCandidate:
    name: str
    tokens: int


@dataclass(frozen=True)
class StyleStudyConfig:
    schema_version: int
    version: str
    source_revision: str
    raw_root: Path
    parent_summary: Path
    parent_summary_sha256: str
    parent_hybrid: Path
    parent_hybrid_sha256: str
    expected_parent_rows: int
    palette_path: Path
    palette_sha256: str
    derived_root: Path
    report_root: Path
    coordinate_bins: int
    control_coordinate_bins: int
    control_coordinate_min: float
    control_coordinate_max: float
    max_paths: int
    max_segments: int
    max_serialized_bytes: int
    width_candidates: tuple[WidthCandidate, ...]
    miter_limits: tuple[float, ...]
    dash_patterns: tuple[tuple[float, ...], ...]
    opacities: tuple[float, ...]
    top_relative_per_candidate: int
    top_absolute_per_candidate: int
    max_fixture_icons: int
    render_sizes: tuple[int, ...]
    render_timeout_seconds: int


def load_style_study_config(path: Path) -> StyleStudyConfig:
    """Load an explicit bounded style study configuration."""

    root = _mapping(yaml.safe_load(_read_bounded(path, 1_000_000, "config")), "root")
    if root.get("schema_version") != 1:
        raise StyleStudyError("style study schema_version must be 1")
    parent = _mapping(root.get("parent"), "parent")
    codec = _mapping(root.get("codec"), "codec")
    controls = _mapping(codec.get("control_coordinates"), "codec.control_coordinates")
    styles = _mapping(root.get("style_vocabulary"), "style_vocabulary")
    selection = _mapping(root.get("fixture_selection"), "fixture_selection")
    render = _mapping(root.get("render"), "render")
    candidates = tuple(
        WidthCandidate(
            name=_string(_mapping(item, "width candidate"), "name"),
            tokens=_positive_int(_mapping(item, "width candidate").get("tokens"), "tokens"),
        )
        for item in _sequence(styles.get("stroke_width_candidates"), "stroke candidates")
    )
    if not candidates or len({item.name for item in candidates}) != len(candidates):
        raise StyleStudyError("stroke width candidates must have unique names")
    if _string(styles, "objective") != "weighted-relative-l1":
        raise StyleStudyError("only weighted-relative-l1 is supported")
    sizes = tuple(
        _positive_int(value, "render size")
        for value in _sequence(render.get("sizes"), "render.sizes")
    )
    if sizes != (72, 18):
        raise StyleStudyError("render sizes must be exactly [72, 18]")
    config = StyleStudyConfig(
        schema_version=1,
        version=_string(root, "study_version"),
        source_revision=_string(root, "source_revision"),
        raw_root=Path(_string(root, "raw_root")),
        parent_summary=Path(_string(parent, "summary")),
        parent_summary_sha256=_sha256(_string(parent, "summary_sha256"), "summary_sha256"),
        parent_hybrid=Path(_string(parent, "hybrid")),
        parent_hybrid_sha256=_sha256(_string(parent, "hybrid_sha256"), "hybrid_sha256"),
        expected_parent_rows=_positive_int(parent.get("expected_rows"), "expected_rows"),
        palette_path=Path(_string(root, "palette_path")),
        palette_sha256=_sha256(_string(root, "palette_sha256"), "palette_sha256"),
        derived_root=Path(_string(root, "derived_root")),
        report_root=Path(_string(root, "report_root")),
        coordinate_bins=_positive_int(codec.get("coordinate_bins"), "coordinate_bins"),
        control_coordinate_bins=_positive_int(controls.get("bins"), "control bins"),
        control_coordinate_min=_finite_number(controls.get("minimum"), "control minimum"),
        control_coordinate_max=_finite_number(controls.get("maximum"), "control maximum"),
        max_paths=_positive_int(codec.get("max_paths"), "max_paths"),
        max_segments=_positive_int(codec.get("max_segments"), "max_segments"),
        max_serialized_bytes=_positive_int(
            codec.get("max_serialized_bytes"), "max_serialized_bytes"
        ),
        width_candidates=candidates,
        miter_limits=_positive_floats(styles.get("miter_limits"), "miter_limits"),
        dash_patterns=tuple(
            _nonnegative_floats(value, f"dash_patterns[{index}]")
            for index, value in enumerate(_sequence(styles.get("dash_patterns"), "dash_patterns"))
        ),
        opacities=_positive_floats(styles.get("opacities"), "opacities"),
        top_relative_per_candidate=_positive_int(
            selection.get("top_relative_per_candidate"), "top_relative_per_candidate"
        ),
        top_absolute_per_candidate=_positive_int(
            selection.get("top_absolute_per_candidate"), "top_absolute_per_candidate"
        ),
        max_fixture_icons=_positive_int(selection.get("max_icons"), "max_icons"),
        render_sizes=sizes,
        render_timeout_seconds=_positive_int(
            render.get("timeout_seconds"), "render.timeout_seconds"
        ),
    )
    if config.control_coordinate_min > 0 or config.control_coordinate_max < 72:
        raise StyleStudyError("control bounds must cover 0..72")
    if any(item.tokens > 128 for item in candidates):
        raise StyleStudyError("stroke width candidate exceeds 128 tokens")
    if len(set(config.miter_limits)) != len(config.miter_limits):
        raise StyleStudyError("miter limits must be unique")
    if len(set(config.dash_patterns)) != len(config.dash_patterns):
        raise StyleStudyError("dash patterns must be unique")
    if any(
        not item or len(item) % 2 or len(item) > 32 or not any(item)
        for item in config.dash_patterns
    ):
        raise StyleStudyError("dash patterns must be nonzero even tuples of at most 32 values")
    return config


def optimal_relative_l1_vocabulary(
    counts: Mapping[float, int], token_count: int
) -> tuple[float, ...]:
    """Return the exact discrete weighted-relative-L1 one-dimensional k-medians."""

    values = tuple(sorted(counts))
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise StyleStudyError("stroke widths must be finite and positive")
    if token_count <= 0 or token_count > len(values):
        raise StyleStudyError("token count must be between one and the distinct value count")
    weights = tuple(float(counts[value]) / value for value in values)
    prefix_weight = [0.0]
    prefix_weighted_value = [0.0]
    for value, weight in zip(values, weights, strict=True):
        prefix_weight.append(prefix_weight[-1] + weight)
        prefix_weighted_value.append(prefix_weighted_value[-1] + weight * value)

    costs = [[0.0] * len(values) for _ in values]
    medians = [[0] * len(values) for _ in values]
    for start in range(len(values)):
        for stop in range(start, len(values)):
            target = prefix_weight[start] + (prefix_weight[stop + 1] - prefix_weight[start]) / 2
            median = min(stop, bisect.bisect_left(prefix_weight, target, start + 1, stop + 2) - 1)
            center = values[median]
            left = center * (prefix_weight[median] - prefix_weight[start]) - (
                prefix_weighted_value[median] - prefix_weighted_value[start]
            )
            right = (
                prefix_weighted_value[stop + 1] - prefix_weighted_value[median + 1]
            ) - center * (prefix_weight[stop + 1] - prefix_weight[median + 1])
            costs[start][stop] = left + right
            medians[start][stop] = median

    infinity = float("inf")
    dp = [[infinity] * (len(values) + 1) for _ in range(token_count + 1)]
    previous = [[-1] * (len(values) + 1) for _ in range(token_count + 1)]
    dp[0][0] = 0.0
    for groups in range(1, token_count + 1):
        for stop in range(groups, len(values) + 1):
            for split in range(groups - 1, stop):
                candidate = dp[groups - 1][split] + costs[split][stop - 1]
                if candidate < dp[groups][stop] - 1e-12:
                    dp[groups][stop] = candidate
                    previous[groups][stop] = split

    centers: list[float] = []
    stop = len(values)
    for groups in range(token_count, 0, -1):
        split = previous[groups][stop]
        if split < 0:
            raise StyleStudyError("stroke-width optimizer failed")
        centers.append(values[medians[split][stop - 1]])
        stop = split
    return tuple(reversed(centers))


def run_style_study(config: StyleStudyConfig, config_path: Path) -> dict[str, Any]:
    """Audit the full distribution and render its pinned worst style tails."""

    _verify_file(config.parent_summary, config.parent_summary_sha256, "parent summary")
    _verify_file(config.parent_hybrid, config.parent_hybrid_sha256, "parent hybrid")
    _verify_file(config.palette_path, config.palette_sha256, "palette")
    parent_summary = _mapping(
        json.loads(_read_bounded(config.parent_summary, _MAX_INPUT_BYTES, "summary")), "summary"
    )
    if parent_summary.get("source_revision") != config.source_revision:
        raise StyleStudyError("parent summary source revision mismatch")
    parent_rows = [
        _mapping(json.loads(line), "hybrid row")
        for line in _read_bounded(config.parent_hybrid, _MAX_INPUT_BYTES, "hybrid").splitlines()
    ]
    if len(parent_rows) != config.expected_parent_rows:
        raise StyleStudyError("parent hybrid row count mismatch")

    width_counts = _numeric_counts(parent_rows, "stroke_width_counts")
    miter_counts = _numeric_counts(parent_rows, "miter_limit_counts")
    dash_counts = _tuple_counts(parent_rows, "dash_pattern_counts")
    vocabularies = {
        candidate.name: optimal_relative_l1_vocabulary(width_counts, candidate.tokens)
        for candidate in config.width_candidates
    }
    analytics = {
        name: _vocabulary_metrics(parent_rows, vocabulary)
        for name, vocabulary in vocabularies.items()
    }
    selected, selection_reasons = _select_fixture(parent_rows, vocabularies, config)
    if len(selected) > config.max_fixture_icons:
        raise StyleStudyError(
            f"fixture selection produced {len(selected)} icons, exceeding cap "
            f"{config.max_fixture_icons}"
        )

    exact_widths = tuple(sorted(width_counts))
    exact_miters = tuple(sorted(miter_counts))
    exact_dashes = tuple(sorted(dash_counts))
    candidate_widths = {"exact-observed": exact_widths, **vocabularies}
    palette = _palette(config.palette_path)
    metric_rows: list[dict[str, Any]] = []
    sheet_images: dict[tuple[str, str], np.ndarray[Any, Any]] = {}

    for row in selected:
        relative = Path(_string(row, "source_path"))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".svg":
            raise StyleStudyError(f"unsafe source path: {relative}")
        source_bytes = (config.raw_root / relative).read_bytes()
        source_sha = hashlib.sha256(source_bytes).hexdigest()
        if source_sha != _string(row, "source_svg_sha256"):
            raise StyleStudyError(f"source hash mismatch: {relative}")
        representation = _string(row, "selected_representation")
        normalized_source = source_bytes if representation == "semantic" else _outline(source_bytes)
        program = normalize_svg(normalized_source).program
        source_renders = {
            size: _render(source_bytes, size, config.render_timeout_seconds)[1]
            for size in config.render_sizes
        }
        sheet_images[(str(relative), "source")] = source_renders[72]
        exact_renders: dict[int, np.ndarray[Any, Any]] = {}

        for candidate_name, widths in candidate_widths.items():
            codec = CodecConfig(
                max_paths=config.max_paths,
                max_segments=config.max_segments,
                coordinate_bins=config.coordinate_bins,
                palette=palette,
                stroke_widths=widths,
                dash_patterns=exact_dashes
                if candidate_name == "exact-observed"
                else config.dash_patterns,
                miter_limits=exact_miters
                if candidate_name == "exact-observed"
                else config.miter_limits,
                opacities=config.opacities,
                max_serialized_bytes=config.max_serialized_bytes,
                control_coordinate_bins=config.control_coordinate_bins,
                control_coordinate_min=config.control_coordinate_min,
                control_coordinate_max=config.control_coordinate_max,
            )
            tensor, report = encode_program(program, codec)
            svg_bytes = serialize_svg(tensor, codec)
            decoded = decode_program(tensor, codec)
            reencoded, _ = encode_program(decoded, codec)
            stable = _tensor_hash(tensor) == _tensor_hash(reencoded)
            if not stable or svg_bytes != serialize_svg(reencoded, codec):
                raise StyleStudyError(f"unstable round trip: {relative} / {candidate_name}")
            output = config.derived_root / candidate_name / relative
            _write_bytes_artifact(output, svg_bytes)
            result: dict[str, Any] = {
                "hexcode": row["hexcode"],
                "group": row["group"],
                "source_path": str(relative),
                "source_sha256": source_sha,
                "representation": representation,
                "candidate": candidate_name,
                "selection_reasons": sorted(selection_reasons[str(relative)]),
                "svg_sha256": hashlib.sha256(svg_bytes).hexdigest(),
                "tensor_sha256": _tensor_hash(tensor),
                "stable_round_trip": stable,
                **asdict(report),
            }
            candidate_renders: dict[int, np.ndarray[Any, Any]] = {}
            for size, source_render in source_renders.items():
                candidate_render = _render(svg_bytes, size, config.render_timeout_seconds)[1]
                candidate_renders[size] = candidate_render
                for field, value in _similarity(source_render, candidate_render).items():
                    result[f"source_{field}_{size}"] = value
                if candidate_name != "exact-observed":
                    for field, value in _similarity(exact_renders[size], candidate_render).items():
                        result[f"style_delta_{field}_{size}"] = value
            if candidate_name == "exact-observed":
                exact_renders = candidate_renders
            if candidate_name in {"exact-observed", *(vocabularies.keys())}:
                sheet_images[(str(relative), candidate_name)] = candidate_renders[72]
            metric_rows.append(result)

    fixture = _fixture_document(config, selected, selection_reasons, vocabularies)
    metrics_payload = _jsonl(metric_rows)
    summary = _summarize(
        config,
        config_path,
        parent_rows,
        selected,
        width_counts,
        miter_counts,
        dash_counts,
        vocabularies,
        analytics,
        metric_rows,
    )
    fixture_payload = _json(fixture)
    summary["fixture_sha256"] = hashlib.sha256(fixture_payload).hexdigest()
    summary["metrics_sha256"] = hashlib.sha256(metrics_payload).hexdigest()
    _write_bytes_artifact(config.report_root / "fixture.json", fixture_payload)
    _write_bytes_artifact(config.report_root / "metrics.jsonl", metrics_payload)
    _write_bytes_artifact(config.report_root / "summary.json", _json(summary))
    _write_bytes_artifact(config.report_root / "README.md", _markdown(summary).encode())
    _write_bytes_artifact(
        config.report_root / "worst-style-delta-18.png",
        _contact_sheet(summary, metric_rows, sheet_images, tuple(candidate_widths)),
    )
    return summary


def _select_fixture(
    rows: list[dict[str, Any]],
    vocabularies: Mapping[str, tuple[float, ...]],
    config: StyleStudyConfig,
) -> tuple[list[dict[str, Any]], dict[str, set[str]]]:
    tail = [
        row for row in rows if not bool(_mapping(row.get("style_vocabulary"), "style").get("exact"))
    ]
    reasons: dict[str, set[str]] = {}

    def choose(row: dict[str, Any], reason: str) -> None:
        reasons.setdefault(_string(row, "source_path"), set()).add(reason)

    for name, vocabulary in vocabularies.items():
        scored = [(_row_width_errors(row, vocabulary), row) for row in tail]
        by_relative = sorted(
            scored, key=lambda item: (-item[0][0], _string(item[1], "source_path"))
        )
        by_absolute = sorted(
            scored, key=lambda item: (-item[0][1], _string(item[1], "source_path"))
        )
        for _, row in by_relative[: config.top_relative_per_candidate]:
            choose(row, f"{name}:top-relative")
        for _, row in by_absolute[: config.top_absolute_per_candidate]:
            choose(row, f"{name}:top-absolute")
    miter_set = set(config.miter_limits)
    dash_set = set(config.dash_patterns)
    for row in tail:
        if _sequence(row.get("dash_pattern_counts"), "dash counts"):
            choose(row, "all-dashed")
        if any(
            float(item["value"]) not in miter_set
            for item in _sequence(row.get("miter_limit_counts"), "miter counts")
        ):
            choose(row, "all-nonexact-miter")
        if any(
            tuple(float(v) for v in cast(list[float], item["value"])) not in dash_set
            for item in _sequence(row.get("dash_pattern_counts"), "dash counts")
        ):
            choose(row, "dash-vocabulary-check")
    selected = sorted(
        (row for row in tail if _string(row, "source_path") in reasons),
        key=lambda row: _string(row, "source_path"),
    )
    return selected, reasons


def _vocabulary_metrics(
    rows: list[dict[str, Any]], vocabulary: tuple[float, ...]
) -> dict[str, Any]:
    absolute: list[float] = []
    relative: list[float] = []
    affected_icons = 0
    affected_contours = 0
    for row in rows:
        affected = False
        for item in _sequence(row.get("stroke_width_counts"), "stroke widths"):
            value = float(item["value"])
            count = int(item["count"])
            nearest = _nearest(value, vocabulary)
            error = abs(value - nearest)
            absolute.extend([error] * count)
            relative.extend([error / value] * count)
            if error:
                affected = True
                affected_contours += count
        affected_icons += int(affected)
    return {
        "tokens": len(vocabulary),
        "values": list(vocabulary),
        "affected_icons": affected_icons,
        "affected_contours": affected_contours,
        "absolute_error": _stats(absolute),
        "relative_error": _stats(relative),
    }


def _row_width_errors(row: dict[str, Any], vocabulary: tuple[float, ...]) -> tuple[float, float]:
    errors = [
        (
            abs(float(item["value"]) - _nearest(float(item["value"]), vocabulary))
            / float(item["value"]),
            abs(float(item["value"]) - _nearest(float(item["value"]), vocabulary)),
        )
        for item in _sequence(row.get("stroke_width_counts"), "stroke widths")
    ]
    return (
        max((item[0] for item in errors), default=0.0),
        max((item[1] for item in errors), default=0.0),
    )


def _nearest(value: float, vocabulary: tuple[float, ...]) -> float:
    return min(vocabulary, key=lambda candidate: (abs(value - candidate), candidate))


def _summarize(
    config: StyleStudyConfig,
    config_path: Path,
    parent_rows: list[dict[str, Any]],
    selected: list[dict[str, Any]],
    width_counts: Counter[float],
    miter_counts: Counter[float],
    dash_counts: Counter[tuple[float, ...]],
    vocabularies: Mapping[str, tuple[float, ...]],
    analytics: dict[str, Any],
    metrics: list[dict[str, Any]],
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    for name in ("exact-observed", *vocabularies):
        rows = [row for row in metrics if row["candidate"] == name]
        item: dict[str, Any] = {
            "attempted": len(rows),
            "stable": sum(bool(row["stable_round_trip"]) for row in rows),
            "icons_with_style_approximation": sum(
                int(row["approximated_stroke_widths"]) > 0
                or int(row["approximated_miter_limits"]) > 0
                for row in rows
            ),
            "approximated_stroke_contours": sum(
                int(row["approximated_stroke_widths"]) for row in rows
            ),
            "approximated_miter_contours": sum(
                int(row["approximated_miter_limits"]) for row in rows
            ),
            "source_rgba_mae_72": _metric_stats(rows, "source_rgba_mae_72"),
            "source_rgba_mae_18": _metric_stats(rows, "source_rgba_mae_18"),
        }
        if name != "exact-observed":
            item["style_delta_rgba_mae_72"] = _metric_stats(rows, "style_delta_rgba_mae_72")
            item["style_delta_rgba_mae_18"] = _metric_stats(rows, "style_delta_rgba_mae_18")
            item["worst_style_delta_18"] = _worst(rows, "style_delta_rgba_mae_18", 12)
        comparisons[name] = item
    return {
        "schema_version": 1,
        "study_version": config.version,
        "source_revision": config.source_revision,
        "code_identity": _code_identity(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "parent_summary": str(config.parent_summary),
        "parent_summary_sha256": config.parent_summary_sha256,
        "parent_hybrid": str(config.parent_hybrid),
        "parent_hybrid_sha256": config.parent_hybrid_sha256,
        "parent_rows": len(parent_rows),
        "parent_style_tail_icons": sum(
            not bool(_mapping(row.get("style_vocabulary"), "style").get("exact"))
            for row in parent_rows
        ),
        "fixture_icons": len(selected),
        "observed": {
            "stroke_width_values": len(width_counts),
            "stroke_width_contours": sum(width_counts.values()),
            "miter_limit_values": [
                {"value": value, "count": count} for value, count in sorted(miter_counts.items())
            ],
            "dash_patterns": [
                {"value": list(value), "count": count}
                for value, count in sorted(dash_counts.items())
            ],
        },
        "width_vocabulary_analytics": analytics,
        "configured_miter_limits": list(config.miter_limits),
        "configured_dash_patterns": [list(item) for item in config.dash_patterns],
        "codec": {
            "coordinate_bins": config.coordinate_bins,
            "control_coordinate_bins": config.control_coordinate_bins,
            "control_coordinate_bounds": [
                config.control_coordinate_min,
                config.control_coordinate_max,
            ],
            "max_paths": config.max_paths,
            "max_segments": config.max_segments,
        },
        "comparisons": comparisons,
    }


def _fixture_document(
    config: StyleStudyConfig,
    rows: list[dict[str, Any]],
    reasons: Mapping[str, set[str]],
    vocabularies: Mapping[str, tuple[float, ...]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "manifest_version": f"{config.version}-fixture",
        "source_revision": config.source_revision,
        "selection": {
            "parent_hybrid": str(config.parent_hybrid),
            "parent_hybrid_sha256": config.parent_hybrid_sha256,
            "rule": (
                "union of each candidate's top relative and absolute width-error tails, "
                "all dashed icons, and all nonexact-miter icons within the parent's "
                "347-icon style tail"
            ),
            "width_vocabularies": {name: list(values) for name, values in vocabularies.items()},
        },
        "rows": [
            {
                "hexcode": row["hexcode"],
                "group": row["group"],
                "subgroup": row["subgroup"],
                "split": row["split"],
                "source_path": row["source_path"],
                "source_svg_sha256": row["source_svg_sha256"],
                "selected_representation": row["selected_representation"],
                "selection_reasons": sorted(reasons[_string(row, "source_path")]),
            }
            for row in rows
        ],
    }


def _contact_sheet(
    summary: dict[str, Any],
    rows: list[dict[str, Any]],
    images: Mapping[tuple[str, str], np.ndarray[Any, Any]],
    candidates: tuple[str, ...],
) -> bytes:
    leading = candidates[-1]
    ranked = _worst(
        [row for row in rows if row["candidate"] == leading], "style_delta_rgba_mae_18", 8
    )
    labels = ("source", *candidates)
    tile = 96
    header = 28
    label_width = 220
    canvas = Image.new(
        "RGB", (label_width + tile * len(labels), header + tile * len(ranked)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((label_width + column * tile + 4, 7), label, fill="black")
    for row_index, item in enumerate(ranked):
        path = str(item["source_path"])
        y = header + row_index * tile
        draw.text(
            (4, y + 6),
            f"{item['hexcode']}\nMAE18 {item['style_delta_rgba_mae_18']:.6f}",
            fill="black",
        )
        for column, label in enumerate(labels):
            rgba = Image.fromarray(images[(path, label)].astype(np.uint8), "RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            white.alpha_composite(rgba)
            canvas.paste(
                white.convert("RGB").resize((72, 72)),
                (label_width + column * tile + 12, y + 12),
            )
    payload = io.BytesIO()
    canvas.save(payload, format="PNG", optimize=True)
    return payload.getvalue()


def _metric_stats(rows: list[dict[str, Any]], field: str) -> dict[str, float | int | None]:
    return _stats([float(row[field]) for row in rows])


def _stats(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(array),
        "mean": float(array.mean()) if len(array) else None,
        "median": float(np.median(array)) if len(array) else None,
        "p95": float(np.quantile(array, 0.95)) if len(array) else None,
        "p99": float(np.quantile(array, 0.99)) if len(array) else None,
        "max": float(array.max()) if len(array) else None,
    }


def _worst(rows: list[dict[str, Any]], field: str, count: int) -> list[dict[str, Any]]:
    ranked = sorted(rows, key=lambda row: (-float(row[field]), str(row["source_path"])))
    return [
        {"hexcode": row["hexcode"], "source_path": row["source_path"], field: row[field]}
        for row in ranked[:count]
    ]


def _numeric_counts(rows: list[dict[str, Any]], field: str) -> Counter[float]:
    result: Counter[float] = Counter()
    for row in rows:
        for item in _sequence(row.get(field), field):
            result[float(item["value"])] += int(item["count"])
    return result


def _tuple_counts(rows: list[dict[str, Any]], field: str) -> Counter[tuple[float, ...]]:
    result: Counter[tuple[float, ...]] = Counter()
    for row in rows:
        for item in _sequence(row.get(field), field):
            result[tuple(float(value) for value in cast(list[float], item["value"]))] += int(
                item["count"]
            )
    return result


def _outline(source: bytes) -> bytes:
    try:
        text = SVG.fromstring(source).topicosvg().tostring()  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise StyleStudyError(f"PicoSVG outline failed: {exc}") from exc
    return text.encode() if isinstance(text, str) else text


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
        raise StyleStudyError(f"cannot identify Git commit: {exc}") from exc
    return {
        "git_commit": commit,
        "style_study_sha256": _file_sha256(Path(__file__)),
        "normalizer_sha256": _file_sha256(repository / "src/mojidiff/representation/normalizer.py"),
        "program_sha256": _file_sha256(repository / "src/mojidiff/representation/program.py"),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Categorical style-vocabulary study",
        "",
        f"Full-corpus analytics cover {summary['parent_rows']} hybrid programs; the render "
        f"fixture contains {summary['fixture_icons']} pinned style-tail icons.",
        "",
        "| candidate | tokens | affected icons | max relative width error | "
        "median style-only MAE 18 | max style-only MAE 18 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, analytics in summary["width_vocabulary_analytics"].items():
        render = summary["comparisons"][name]["style_delta_rgba_mae_18"]
        lines.append(
            f"| {name} | {analytics['tokens']} | {analytics['affected_icons']} | "
            f"{analytics['relative_error']['max']:.6g} | {render['median']:.6g} | "
            f"{render['max']:.6g} |"
        )
    lines.extend(
        [
            "",
            "`exact-observed` holds the role-typed coordinate lattice and all observed "
            "styles exact; candidate-to-control render metrics therefore isolate style "
            "approximation. Per-icon evidence and the deterministic fixture are retained "
            "beside this report.",
            "",
        ]
    )
    return "\n".join(lines)


def _read_bounded(path: Path, limit: int, label: str) -> bytes:
    if path.stat().st_size > limit:
        raise StyleStudyError(f"{label} exceeds {limit} bytes")
    return path.read_bytes()


def _verify_file(path: Path, expected: str, label: str) -> None:
    if _file_sha256(path) != expected:
        raise StyleStudyError(f"{label} hash mismatch")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StyleStudyError(f"{field} must be a mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise StyleStudyError(f"{field} must be a list")
    return value


def _string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise StyleStudyError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StyleStudyError(f"{field} must be a positive integer")
    return value


def _finite_number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise StyleStudyError(f"{field} must be finite")
    return float(value)


def _positive_floats(value: object, field: str) -> tuple[float, ...]:
    result = tuple(_finite_number(item, field) for item in _sequence(value, field))
    if not result or any(item <= 0 for item in result):
        raise StyleStudyError(f"{field} must contain positive numbers")
    return result


def _nonnegative_floats(value: object, field: str) -> tuple[float, ...]:
    result = tuple(_finite_number(item, field) for item in _sequence(value, field))
    if any(item < 0 for item in result):
        raise StyleStudyError(f"{field} must contain nonnegative numbers")
    return result


def _sha256(value: str, field: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise StyleStudyError(f"{field} must be lowercase SHA-256")
    return value


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mojidiff-style-study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_style_study(load_style_study_config(args.config), args.config),
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
