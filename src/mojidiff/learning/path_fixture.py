"""Select a family-distinct, capacity-bounded whole-path donor fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from mojidiff.learning.path_signature_census import _coordinate_count, _outline
from mojidiff.learning.tiny_study import _mapping, _string
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.normalizer import normalize_svg
from mojidiff.representation.program import SegmentType


class PathFixtureError(RuntimeError):
    """The whole-path fixture selection cannot reproduce its declared inputs."""


@dataclass(frozen=True)
class FixtureConfig:
    source_revision: str
    raw_root: Path
    hybrid_manifest: Path
    hybrid_manifest_sha256: str
    expected_rows: int
    report_root: Path
    fixture_path: Path
    max_paths: int
    max_segments_per_path: int
    max_total_segments: int
    icon_count: int
    per_group_candidates: int
    min_eligible_fraction: float


def load_fixture_config(path: Path) -> FixtureConfig:
    root = _mapping(yaml.safe_load(path.read_bytes()), "root")
    if root.get("schema_version") != 1:
        raise PathFixtureError("fixture schema_version must be 1")
    selection = _mapping(root.get("selection"), "selection")
    value = selection.get("min_eligible_fraction")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise PathFixtureError("min_eligible_fraction must be within (0, 1]")
    return FixtureConfig(
        source_revision=_string(root, "source_revision"),
        raw_root=Path(_string(root, "raw_root")),
        hybrid_manifest=Path(_string(root, "hybrid_manifest")),
        hybrid_manifest_sha256=_sha256(_string(root, "hybrid_manifest_sha256")),
        expected_rows=_positive_int(root.get("expected_rows")),
        report_root=Path(_string(root, "report_root")),
        fixture_path=Path(_string(root, "fixture_path")),
        max_paths=_positive_int(selection.get("max_paths")),
        max_segments_per_path=_positive_int(selection.get("max_segments_per_path")),
        max_total_segments=_positive_int(selection.get("max_total_segments")),
        icon_count=_positive_int(selection.get("icon_count")),
        per_group_candidates=_positive_int(selection.get("per_group_candidates")),
        min_eligible_fraction=float(value),
    )


def run_fixture_selection(config: FixtureConfig) -> dict[str, Any]:
    if _sha256_file(config.hybrid_manifest) != config.hybrid_manifest_sha256:
        raise PathFixtureError("hybrid manifest hash mismatch")
    rows = [
        _mapping(json.loads(line), "hybrid row")
        for line in config.hybrid_manifest.read_text().splitlines()
    ]
    if len(rows) != config.expected_rows:
        raise PathFixtureError("hybrid manifest row count mismatch")
    candidates: list[dict[str, Any]] = []
    for row in rows:
        item = _candidate(row, config)
        if item is not None:
            candidates.append(item)
    global_signatures: Counter[tuple[int, ...]] = Counter(
        signature for item in candidates for signature, _ in item["paths"]
    )
    for item in candidates:
        total = sum(fields for _, fields in item["paths"])
        supported = sum(fields for sig, fields in item["paths"] if global_signatures[sig] > 1)
        item["global_fraction"] = supported / total if total else 0.0
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in candidates:
        grouped[item["group"]].append(item)
    shortlist: list[dict[str, Any]] = []
    for group in sorted(grouped):
        ranked = sorted(
            grouped[group],
            key=lambda item: (-item["global_fraction"], -item["fields"], item["hexcode"]),
        )
        shortlist.extend(ranked[: config.per_group_candidates])
    selected = _greedy_select(shortlist, config)
    score, coverage = _score(selected)
    if len(selected) != config.icon_count or any(
        item["eligible_fraction"] < config.min_eligible_fraction for item in coverage
    ):
        raise PathFixtureError("no selection satisfies count and donor coverage constraints")
    manifest = {
        "schema_version": 1,
        "source_revision": config.source_revision,
        "rows": [
            {
                key: item[key]
                for key in (
                    "hexcode",
                    "source_path",
                    "source_svg_sha256",
                    "group",
                    "subgroup",
                    "variant_family_id",
                    "selected_representation",
                )
            }
            for item in selected
        ],
    }
    summary = {
        "schema_version": 1,
        "selected_score": score,
        "icons": coverage,
        "candidate_count": len(candidates),
        "shortlist_count": len(shortlist),
        "fixture_sha256": hashlib.sha256(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
        ).hexdigest(),
    }
    _write_bytes_artifact(
        config.fixture_path, (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    )
    _write_bytes_artifact(
        config.report_root / "summary.json",
        (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode(),
    )
    return summary


def _candidate(row: dict[str, Any], config: FixtureConfig) -> dict[str, Any] | None:
    source_path = Path(_string(row, "source_path"))
    source = (config.raw_root / source_path).read_bytes()
    if hashlib.sha256(source).hexdigest() != _string(row, "source_svg_sha256"):
        raise PathFixtureError(f"source hash mismatch: {source_path}")
    representation = _string(row, "selected_representation")
    program = normalize_svg(source if representation == "semantic" else _outline(source)).program
    paths = []
    segments = 0
    for contour in program.contours:
        signature = tuple(int(segment.kind) for segment in contour.segments)
        segments += len(signature)
        paths.append(
            (signature, 2 + sum(_coordinate_count(SegmentType(kind)) for kind in signature))
        )
    if (
        len(paths) > config.max_paths
        or segments > config.max_total_segments
        or any(len(signature) > config.max_segments_per_path for signature, _ in paths)
    ):
        return None
    return {
        "hexcode": _string(row, "hexcode"),
        "source_path": str(source_path),
        "source_svg_sha256": _string(row, "source_svg_sha256"),
        "group": _string(row, "group"),
        "subgroup": _string(row, "subgroup"),
        "variant_family_id": _string(row, "variant_family_id"),
        "selected_representation": representation,
        "paths": paths,
        "fields": sum(field for _, field in paths),
    }


def _greedy_select(candidates: list[dict[str, Any]], config: FixtureConfig) -> list[dict[str, Any]]:
    best: list[dict[str, Any]] = []
    for first in candidates:
        chosen = [first]
        while len(chosen) < config.icon_count:
            options = [
                item
                for item in candidates
                if item not in chosen
                and item["group"] not in {x["group"] for x in chosen}
                and item["variant_family_id"] not in {x["variant_family_id"] for x in chosen}
            ]
            if not options:
                break
            chosen.append(
                max(
                    options,
                    key=lambda item: (_score(chosen + [item])[0], item["fields"], item["hexcode"]),
                )
            )
        if len(chosen) == config.icon_count and _score(chosen)[0] > _score(best)[0]:
            best = chosen
    return best


def _score(items: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]]]:
    counts: Counter[tuple[int, ...]] = Counter(sig for item in items for sig, _ in item["paths"])
    coverage = []
    score = 0
    for item in items:
        total = item["fields"]
        eligible = sum(fields for sig, fields in item["paths"] if counts[sig] > 1)
        score += eligible
        coverage.append(
            {
                "hexcode": item["hexcode"],
                "group": item["group"],
                "variant_family_id": item["variant_family_id"],
                "active_paths": len(item["paths"]),
                "geometry_fields": total,
                "eligible_geometry_fields": eligible,
                "eligible_fraction": eligible / total if total else 0.0,
            }
        )
    return score, coverage


def _sha256(value: str) -> str:
    if len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise PathFixtureError("invalid SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _positive_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PathFixtureError("expected positive integer")
    return value


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    a = p.parse_args(argv)
    print(
        json.dumps(run_fixture_selection(load_fixture_config(a.config)), indent=2, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
