"""Evidence-derived, reversible curation decisions and family-aware splits."""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageDraw, ImageFont


@dataclass(frozen=True)
class DecisionPaths:
    output_root: Path
    report_root: Path
    manifest_prefix: Path
    thresholds: dict[str, Any]
    review_manifest: Path | None


def load_decision_paths(config_path: Path) -> DecisionPaths:
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        raise ValueError("audit configuration must be a mapping")
    decisions = document.get("decisions")
    if not isinstance(decisions, dict):
        raise ValueError("decisions must be a mapping")
    return DecisionPaths(
        output_root=Path(_string(document, "output_root")),
        report_root=Path(_string(document, "report_root")),
        manifest_prefix=Path(_string(document, "manifest_prefix")),
        thresholds=decisions,
        review_manifest=(
            Path(str(document["review_manifest"])) if document.get("review_manifest") else None
        ),
    )


def _string(mapping: dict[str, Any], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a string")
    return value


def read_rows(path: Path) -> list[dict[str, Any]]:
    return cast(list[dict[str, Any]], pq.read_table(path).to_pylist())


def distribution_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "alpha_fraction_72",
        "ink_fraction_72",
        "bbox_width_72",
        "bbox_height_72",
        "bbox_area_72",
        "alpha_fraction_18",
        "segment_count",
        "path_count",
        "component_count_72",
        "total_path_length_estimate",
    )
    quantiles = (0, 0.001, 0.005, 0.01, 0.05, 0.5, 0.95, 0.99, 0.995, 0.999, 1)
    summary: dict[str, Any] = {"row_count": len(rows), "metrics": {}}
    for field in fields:
        values = np.array(
            [float(row[field]) for row in rows if row.get(field) is not None], dtype=np.float64
        )
        values = values[np.isfinite(values)]
        summary["metrics"][field] = {
            "count": int(len(values)),
            "min": float(values.min()) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
            "mean": float(values.mean()) if len(values) else None,
            "quantiles": {
                f"q{quantile:g}": float(np.quantile(values, quantile)) if len(values) else None
                for quantile in quantiles
            },
        }
    summary["groups"] = dict(sorted(_counts(row.get("group", "") for row in rows).items()))
    summary["parse_failures"] = sum(not row.get("parse_ok", False) for row in rows)
    summary["render_72_failures"] = sum(not row.get("render_72_ok", False) for row in rows)
    summary["render_18_failures"] = sum(not row.get("render_18_ok", False) for row in rows)
    return summary


def _counts(values: Any) -> dict[str, int]:
    result: dict[str, int] = defaultdict(int)
    for value in values:
        result[str(value)] += 1
    return dict(result)


def write_distribution_report(paths: DecisionPaths, rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = distribution_summary(rows)
    paths.report_root.mkdir(parents=True, exist_ok=True)
    (paths.report_root / "distributions.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    for field in summary["metrics"]:
        values = [
            float(row[field])
            for row in rows
            if row.get(field) is not None and np.isfinite(float(row[field]))
        ]
        _histogram(values, paths.report_root / f"hist-{field}.png", field)
    threshold_version = paths.thresholds.get("threshold_version")
    threshold_note = (
        "Numeric quarantine thresholds are intentionally unset in the initial pass. "
        "The committed `distributions.json` contains observed quantiles used to select them."
        if threshold_version == "observe-only"
        else (
            f"Curation thresholds `{threshold_version}` were selected from the observed "
            "0.5% corpus tails and nominate review candidates without automatic exclusion."
        )
    )
    markdown = [
        "# OpenMoji corpus audit distributions",
        "",
        f"Audited rows: {summary['row_count']}",
        "",
        f"Parse failures: {summary['parse_failures']}",
        f"72 px render failures: {summary['render_72_failures']}",
        f"18 px render failures: {summary['render_18_failures']}",
        "",
        threshold_note,
        "",
        "| metric | min | q0.1% | q0.5% | median | q99.5% | max |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for field, metric in summary["metrics"].items():
        q = metric["quantiles"]
        markdown.append(
            f"| {field} | {_fmt(metric['min'])} | {_fmt(q['q0.001'])} | "
            f"{_fmt(q['q0.005'])} | {_fmt(q['q0.5'])} | {_fmt(q['q0.995'])} | "
            f"{_fmt(metric['max'])} |"
        )
    (paths.report_root / "README.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.6g}"


def _histogram(values: list[float], output: Path, title: str) -> None:
    width, height = 800, 420
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((16, 12), title, fill="black", font=font)
    if values:
        counts, edges = np.histogram(np.asarray(values), bins=50)
        maximum = max(int(counts.max()), 1)
        left, top, right, bottom = 50, 40, width - 20, height - 45
        draw.line((left, bottom, right, bottom), fill="black")
        draw.line((left, top, left, bottom), fill="black")
        bar_width = (right - left) / len(counts)
        for index, count in enumerate(counts):
            x0 = left + index * bar_width
            x1 = left + (index + 1) * bar_width
            y0 = bottom - (int(count) / maximum) * (bottom - top)
            draw.rectangle((x0, y0, x1, bottom), fill="#61b2e4")
        draw.text((left, bottom + 8), f"min {edges[0]:.6g}", fill="black", font=font)
        draw.text((right - 130, bottom + 8), f"max {edges[-1]:.6g}", fill="black", font=font)
        draw.text((left + 4, top + 4), f"max bin {maximum}", fill="black", font=font)
    image.save(output, format="PNG", optimize=False)


def apply_decisions(paths: DecisionPaths, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    canonical_by_cluster = _canonical_exact_duplicates(rows)
    decided: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        reasons: list[str] = []
        status = "include"
        if not row.get("parse_ok"):
            status, reasons = "exclude_defect", ["PARSE_ERROR"]
        elif not row.get("render_72_ok") or not row.get("render_18_ok"):
            status, reasons = "exclude_defect", ["RENDER_ERROR"]
        elif not row.get("alpha_pixel_count_72") and not row.get("alpha_pixel_count_18"):
            status, reasons = "exclude_defect", ["NO_VISIBLE_PAINT"]
        elif (
            row.get("geometry_sample_count", 0) > 0
            and row.get("geometry_inside_sample_count") == 0
            and not row.get("geometry_estimate_has_unapplied_transforms")
        ):
            status, reasons = "exclude_defect", ["OUTSIDE_VIEWBOX"]
        elif row.get("group") == "flags":
            status, reasons = "exclude_policy", ["SCOPE_FLAGS"]
        elif (
            row.get("exact_raster_cluster")
            and canonical_by_cluster.get(str(row["exact_raster_cluster"])) != row["hexcode"]
        ):
            status, reasons = "exclude_policy", ["EXACT_DUPLICATE_ALIAS"]
        else:
            reasons = _quarantine_reasons(row, paths.thresholds)
            if reasons:
                status = "quarantine_auto"
        row.update(
            {
                "curation_status": status,
                "reason_codes": reasons,
                "rule_version": paths.thresholds.get("threshold_version"),
                "reviewer": None,
                "review_note": None,
                "split": _policy_split(status, reasons),
            }
        )
        decided.append(row)
    _assign_family_splits(decided)
    return decided


def candidate_set_sha256(rows: list[dict[str, Any]]) -> str:
    candidates = [
        {"hexcode": row["hexcode"], "reason_codes": row["reason_codes"]}
        for row in rows
        if row["curation_status"] == "quarantine_auto"
    ]
    serialized = json.dumps(
        sorted(candidates, key=lambda item: str(item["hexcode"])),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(serialized).hexdigest()


def apply_review(rows: list[dict[str, Any]], review_path: Path) -> list[dict[str, Any]]:
    review = yaml.safe_load(review_path.read_text(encoding="utf-8"))
    if not isinstance(review, dict) or review.get("schema_version") != 1:
        raise ValueError("review manifest schema_version must be 1")
    base_path = review.get("base_decisions_path")
    base_hash = review.get("base_decisions_sha256")
    if (
        not isinstance(base_path, str)
        or not Path(base_path).is_file()
        or hashlib.sha256(Path(base_path).read_bytes()).hexdigest() != base_hash
    ):
        raise ValueError("review base decision artifact hash does not match")
    sheet_root = review.get("contact_sheet_root")
    sheet_hashes = review.get("contact_sheets_sha256")
    if not isinstance(sheet_root, str) or not isinstance(sheet_hashes, dict):
        raise ValueError("review contact-sheet evidence is incomplete")
    for name, expected in sheet_hashes.items():
        sheet_path = Path(sheet_root) / str(name)
        if (
            not sheet_path.is_file()
            or hashlib.sha256(sheet_path.read_bytes()).hexdigest() != expected
        ):
            raise ValueError(f"review contact-sheet hash does not match: {name}")
    expected_count = review.get("candidate_count")
    expected_hash = review.get("candidate_set_sha256")
    candidates = [row for row in rows if row["curation_status"] == "quarantine_auto"]
    if expected_count != len(candidates) or expected_hash != candidate_set_sha256(rows):
        raise ValueError("review manifest candidate set does not match current automatic decisions")
    decision = review.get("decision")
    if decision not in {"include_override", "exclude_defect", "quarantine_manual"}:
        raise ValueError("review decision is unsupported")
    reviewer = review.get("reviewer")
    note = review.get("note")
    if not isinstance(reviewer, str) or not reviewer or not isinstance(note, str) or not note:
        raise ValueError("reviewer and note must be non-empty strings")
    for row in candidates:
        row["curation_status"] = decision
        row["reviewer"] = reviewer
        row["review_note"] = note
        row["split"] = _policy_split(decision, row["reason_codes"])
    _assign_family_splits(rows)
    return rows


def _canonical_exact_duplicates(rows: list[dict[str, Any]]) -> dict[str, str]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cluster = row.get("exact_raster_cluster")
        if cluster and row.get("parse_ok") and row.get("render_72_ok") and row.get("render_18_ok"):
            clusters[str(cluster)].append(row)

    def priority(row: dict[str, Any]) -> tuple[bool, bool, int, str]:
        group = str(row.get("group", ""))
        hexcode = str(row["hexcode"])
        return (group == "flags", group.startswith("extras-"), len(hexcode), hexcode)

    return {
        cluster: str(min(members, key=priority)["hexcode"]) for cluster, members in clusters.items()
    }


def _quarantine_reasons(row: dict[str, Any], thresholds: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    comparisons = (
        ("low_ink_fraction_72_max", "ink_fraction_72", "LOW_INK", "maximum"),
        ("thin_bbox_width_72_max", "bbox_width_72", "THIN_BBOX", "maximum"),
        ("thin_bbox_height_72_max", "bbox_height_72", "THIN_BBOX", "maximum"),
        ("small_bbox_area_72_max", "bbox_area_72", "THIN_BBOX", "maximum"),
        ("excessive_paths_min", "path_count", "EXCESSIVE_COMPLEXITY", "minimum"),
        ("excessive_segments_min", "segment_count", "EXCESSIVE_COMPLEXITY", "minimum"),
    )
    for threshold_field, metric_field, reason, direction in comparisons:
        threshold = thresholds.get(threshold_field)
        metric = row.get(metric_field)
        matches = False
        if threshold is not None and metric is not None:
            matches = (
                float(metric) <= float(threshold)
                if direction == "maximum"
                else float(metric) >= float(threshold)
            )
        if matches:
            reasons.append(reason)
    if row.get("alpha_pixel_count_72", 0) and not row.get("alpha_pixel_count_18", 0):
        reasons.append("FAILS_AT_18PX")
    if (
        row.get("path_count") == 1
        and row.get("open_path_count") == 1
        and row.get("fill_element_count") == 0
    ):
        reasons.append("SINGLE_OPEN_STROKE")
    return sorted(set(reasons))


def _policy_split(status: str, reasons: list[str]) -> str:
    if "SCOPE_FLAGS" in reasons:
        return "excluded/flags"
    if "EXACT_DUPLICATE_ALIAS" in reasons:
        return "excluded/exact-duplicates"
    if status.startswith("quarantine"):
        return "quarantine"
    if status == "exclude_defect":
        return "excluded/defects"
    return "primary/pending-family-split"


def _assign_family_splits(rows: list[dict[str, Any]]) -> None:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    family_by_exact: dict[str, str] = {}
    for row in rows:
        family = str(row.get("variant_family_id") or row["hexcode"])
        find(family)
        exact = row.get("exact_raster_cluster")
        if exact:
            previous = family_by_exact.setdefault(str(exact), family)
            union(family, previous)
    for row in rows:
        if row["curation_status"] not in {"include", "include_override"}:
            continue
        family = find(str(row.get("variant_family_id") or row["hexcode"]))
        bucket = int(hashlib.sha256(f"split-v1:{family}".encode()).hexdigest()[:8], 16) % 10000
        split = "train" if bucket < 8000 else "validation" if bucket < 9000 else "test"
        row["split"] = f"primary/{split}"
        row["split_family_cluster"] = family


def write_manifests(paths: DecisionPaths, rows: list[dict[str, Any]]) -> dict[str, Any]:
    paths.manifest_prefix.parent.mkdir(parents=True, exist_ok=True)
    subsets = {
        "decisions": rows,
        "primary": [row for row in rows if str(row["split"]).startswith("primary/")],
        "flags": [row for row in rows if row["split"] == "excluded/flags"],
        "exact_duplicates": [row for row in rows if row["split"] == "excluded/exact-duplicates"],
        "quarantine": [row for row in rows if row["split"] == "quarantine"],
        "defects": [row for row in rows if row["split"] == "excluded/defects"],
    }
    hashes: dict[str, str] = {}
    for name, subset in subsets.items():
        path = Path(f"{paths.manifest_prefix}-{name}.parquet")
        pq.write_table(pa.Table.from_pylist(subset), path, compression="zstd")
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    mapping_rows = [
        {
            "hexcode": row["hexcode"],
            "variant_family_id": row.get("variant_family_id"),
            "split_family_cluster": row.get("split_family_cluster"),
            "exact_raster_cluster": row.get("exact_raster_cluster"),
            "perceptual_hash_cluster": row.get("perceptual_hash_cluster"),
        }
        for row in rows
    ]
    mapping_path = Path(f"{paths.manifest_prefix}-families.parquet")
    pq.write_table(pa.Table.from_pylist(mapping_rows), mapping_path, compression="zstd")
    hashes["families"] = hashlib.sha256(mapping_path.read_bytes()).hexdigest()
    summary = {
        "total": len(rows),
        "status_counts": _counts(row["curation_status"] for row in rows),
        "split_counts": _counts(row["split"] for row in rows),
        "reason_counts": _counts(reason for row in rows for reason in row["reason_codes"]),
        "manifest_sha256": hashes,
    }
    summary_path = Path(f"{paths.manifest_prefix}-summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return summary


def write_contact_sheets(paths: DecisionPaths, rows: list[dict[str, Any]]) -> list[Path]:
    by_reason: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["curation_status"] == "quarantine_auto":
            for reason in row["reason_codes"]:
                by_reason[reason].append(row)
    outputs: list[Path] = []
    sheet_root = paths.output_root / "contact-sheets"
    sheet_root.mkdir(parents=True, exist_ok=True)
    for reason, members in sorted(by_reason.items()):
        for page, offset in enumerate(range(0, len(members), 24), start=1):
            output = sheet_root / f"{reason.lower()}-{page:03d}.png"
            _contact_page(paths, reason, members[offset : offset + 24], output)
            outputs.append(output)
    return outputs


def _contact_page(
    paths: DecisionPaths, reason: str, rows: list[dict[str, Any]], output: Path
) -> None:
    cell_width, cell_height, columns = 420, 180, 4
    page = Image.new(
        "RGB", (columns * cell_width, 40 + math_ceil(len(rows) / columns) * cell_height), "white"
    )
    draw = ImageDraw.Draw(page)
    font = ImageFont.load_default()
    draw.text((10, 10), f"Quarantine reason: {reason}", fill="black", font=font)
    for index, row in enumerate(rows):
        x = (index % columns) * cell_width
        y = 40 + (index // columns) * cell_height
        stem = Path(row["source_path"]).stem
        render72 = Image.open(paths.output_root / "renders" / "72" / f"{stem}.png").convert("RGBA")
        render18 = Image.open(paths.output_root / "renders" / "18" / f"{stem}.png").convert("RGBA")
        page.paste(
            render72.resize((96, 96), Image.Resampling.NEAREST),
            (x + 8, y + 8),
            render72.resize((96, 96), Image.Resampling.NEAREST),
        )
        enlarged18 = render18.resize((96, 96), Image.Resampling.NEAREST)
        page.paste(enlarged18, (x + 112, y + 8), enlarged18)
        text = (
            f"{row['hexcode']} {row['annotation'][:28]}\n"
            f"{row['group']}/{row['subgroup']}\n"
            f"ink={row.get('ink_fraction_72', 0):.4g} "
            f"bbox={row.get('bbox_width_72', 0)}x{row.get('bbox_height_72', 0)}\n"
            f"paths={row.get('path_count', 0)} segs={row.get('segment_count', 0)}"
        )
        draw.multiline_text((x + 214, y + 10), text, fill="black", font=font, spacing=4)
        draw.rectangle((x, y, x + cell_width - 1, y + cell_height - 1), outline="#999999")
    page.save(output, format="PNG", optimize=False)


def math_ceil(value: float) -> int:
    integer = int(value)
    return integer if integer == value else integer + 1
