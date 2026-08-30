"""Reproducible full-primary census of exact SVG path-kind signatures."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml
from picosvg.svg import SVG

from mojidiff.learning.tiny_study import _mapping, _string
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.normalizer import normalize_svg
from mojidiff.representation.program import SegmentType


class PathSignatureCensusError(RuntimeError):
    """A path-signature census cannot reproduce its bounded declared input."""


@dataclass(frozen=True)
class PathSignatureCensusConfig:
    version: str
    source_revision: str
    raw_root: Path
    hybrid_manifest: Path
    hybrid_manifest_sha256: str
    expected_rows: int
    report_root: Path
    max_source_bytes: int
    candidate_limit: int


def load_path_signature_census_config(path: Path) -> PathSignatureCensusConfig:
    try:
        root = _mapping(yaml.safe_load(path.read_bytes()), "root")
    except yaml.YAMLError as error:
        raise PathSignatureCensusError("invalid path-signature census YAML") from error
    if root.get("schema_version") != 1:
        raise PathSignatureCensusError("path-signature census schema_version must be 1")
    limits = _mapping(root.get("limits"), "limits")
    expected_rows = _positive_int(root.get("expected_rows"), "expected_rows")
    candidate_limit = _positive_int(limits.get("candidate_limit"), "limits.candidate_limit")
    return PathSignatureCensusConfig(
        version=_string(root, "study_version"),
        source_revision=_string(root, "source_revision"),
        raw_root=Path(_string(root, "raw_root")),
        hybrid_manifest=Path(_string(root, "hybrid_manifest")),
        hybrid_manifest_sha256=_sha256(_string(root, "hybrid_manifest_sha256")),
        expected_rows=expected_rows,
        report_root=Path(_string(root, "report_root")),
        max_source_bytes=_positive_int(limits.get("max_source_bytes"), "limits.max_source_bytes"),
        candidate_limit=candidate_limit,
    )


def run_path_signature_census(config: PathSignatureCensusConfig) -> dict[str, Any]:
    if _sha256_file(config.hybrid_manifest) != config.hybrid_manifest_sha256:
        raise PathSignatureCensusError("hybrid manifest hash mismatch")
    rows = _read_jsonl(config.hybrid_manifest)
    if len(rows) != config.expected_rows:
        raise PathSignatureCensusError("hybrid manifest row count mismatch")
    signature_icons: dict[tuple[int, ...], set[str]] = defaultdict(set)
    signature_paths: Counter[tuple[int, ...]] = Counter()
    icon_paths: dict[str, list[tuple[tuple[int, ...], int]]] = {}
    routes: Counter[str] = Counter()
    for row in rows:
        source_path = Path(_string(row, "source_path"))
        if source_path.is_absolute() or ".." in source_path.parts or source_path.suffix != ".svg":
            raise PathSignatureCensusError(f"unsafe source path: {source_path}")
        if _string(row, "source_revision") != config.source_revision:
            raise PathSignatureCensusError("source revision mismatch")
        source = (config.raw_root / source_path).read_bytes()
        if len(source) > config.max_source_bytes:
            raise PathSignatureCensusError(f"source exceeds byte limit: {source_path}")
        if hashlib.sha256(source).hexdigest() != _string(row, "source_svg_sha256"):
            raise PathSignatureCensusError(f"source hash mismatch: {source_path}")
        representation = _string(row, "selected_representation")
        if representation not in {"semantic", "outlined"}:
            raise PathSignatureCensusError("hybrid row lacks selected representation")
        normalized_source = source if representation == "semantic" else _outline(source)
        program = normalize_svg(normalized_source).program
        hexcode = _string(row, "hexcode")
        paths: list[tuple[tuple[int, ...], int]] = []
        for contour in program.contours:
            signature = tuple(int(segment.kind) for segment in contour.segments)
            fields = 2 + sum(_coordinate_count(SegmentType(kind)) for kind in signature)
            paths.append((signature, fields))
            signature_icons[signature].add(hexcode)
            signature_paths[signature] += 1
        if hexcode in icon_paths:
            raise PathSignatureCensusError(f"duplicate icon identity: {hexcode}")
        icon_paths[hexcode] = paths
        routes[representation] += 1
    eligible_by_icon: list[dict[str, Any]] = []
    all_fields = 0
    eligible_fields = 0
    for hexcode, paths in icon_paths.items():
        icon_all = sum(fields for _, fields in paths)
        icon_eligible = sum(
            fields for signature, fields in paths if len(signature_icons[signature] - {hexcode}) > 0
        )
        all_fields += icon_all
        eligible_fields += icon_eligible
        eligible_by_icon.append(
            {
                "hexcode": hexcode,
                "active_paths": len(paths),
                "geometry_fields": icon_all,
                "eligible_geometry_fields": icon_eligible,
                "eligible_fraction": icon_eligible / icon_all if icon_all else 0.0,
            }
        )
    candidates: list[dict[str, Any]] = []
    for signature, occurrences in signature_paths.items():
        icons = sorted(signature_icons[signature])
        if len(icons) < 2:
            continue
        fields = 2 + sum(_coordinate_count(SegmentType(kind)) for kind in signature)
        candidates.append(
            {
                "segment_types": list(signature),
                "segments": len(signature),
                "geometry_fields_per_path": fields,
                "path_occurrences": occurrences,
                "icon_occurrences": len(icons),
                "icon_examples": icons[:20],
            }
        )
    candidates.sort(key=_candidate_sort_key)
    eligible_by_icon.sort(
        key=lambda item: (
            -float(item["eligible_fraction"]),
            -int(item["geometry_fields"]),
            item["hexcode"],
        )
    )
    result = {
        "schema_version": 1,
        "study_version": config.version,
        "hybrid_manifest": str(config.hybrid_manifest),
        "hybrid_manifest_sha256": config.hybrid_manifest_sha256,
        "icons": len(rows),
        "routes": dict(sorted(routes.items())),
        "total_geometry_fields": all_fields,
        "externally_compatible_geometry_fields": eligible_fields,
        "externally_compatible_geometry_field_fraction": eligible_fields / all_fields,
        "signature_count": len(signature_paths),
        "signatures_with_external_donors": len(candidates),
        "candidate_signatures": candidates[: config.candidate_limit],
        "top_icons_by_external_donor_coverage": eligible_by_icon[: config.candidate_limit],
    }
    _write_bytes_artifact(
        config.report_root / "summary.json",
        (json.dumps(result, indent=2, sort_keys=True) + "\n").encode(),
    )
    _write_bytes_artifact(config.report_root / "README.md", _markdown(result).encode())
    return result


def _outline(source: bytes) -> bytes:
    try:
        text = SVG.fromstring(source).topicosvg().tostring()  # type: ignore[no-untyped-call]
        return cast(str, text).encode()
    except Exception as error:
        raise PathSignatureCensusError("outlined fallback failed") from error


def _coordinate_count(kind: SegmentType) -> int:
    return {
        SegmentType.LINE: 2,
        SegmentType.QUAD: 4,
        SegmentType.CUBIC: 6,
        SegmentType.CLOSE: 0,
    }[kind]


def _candidate_sort_key(item: dict[str, Any]) -> tuple[int, int, int, list[int]]:
    return (
        -cast(int, item["geometry_fields_per_path"]) * cast(int, item["icon_occurrences"]),
        -cast(int, item["icon_occurrences"]),
        -cast(int, item["segments"]),
        cast(list[int], item["segment_types"]),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [_mapping(json.loads(line), "hybrid row") for line in path.read_text().splitlines()]
    except json.JSONDecodeError as error:
        raise PathSignatureCensusError("invalid hybrid JSONL") from error


def _markdown(result: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# Full-primary path signature census",
            "",
            f"Enumerated {result['icons']} hybrid-selected OpenMoji programs and "
            f"{result['signature_count']} exact segment-kind signatures.",
            "",
            f"External exact donors cover {result['externally_compatible_geometry_fields']} of "
            f"{result['total_geometry_fields']} legal geometry fields "
            f"({result['externally_compatible_geometry_field_fraction']:.2%}).",
            "",
            "Candidate signatures and icon-level coverage are in `summary.json`.",
            "",
        ]
    )


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PathSignatureCensusError("expected lowercase SHA-256")
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PathSignatureCensusError(f"{field} must be a positive integer")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mojidiff-path-signature-census")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_path_signature_census(load_path_signature_census_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
