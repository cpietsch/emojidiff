"""Reproducible dense-versus-packed capacity analysis for typed SVG programs."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from mojidiff.representation.codec_study import _write_bytes_artifact


class CapacityStudyError(RuntimeError):
    """A capacity study cannot be reproduced from its pinned inputs."""


@dataclass(frozen=True)
class DenseCapacity:
    name: str
    path_slots: int
    segments_per_path: int


@dataclass(frozen=True)
class PackedCapacity:
    name: str
    path_slots: int
    total_segment_slots: int


@dataclass(frozen=True)
class CapacityStudyConfig:
    version: str
    source_revision: str
    parent_hybrid: Path
    parent_hybrid_sha256: str
    expected_rows: int
    report_root: Path
    dense: tuple[DenseCapacity, ...]
    packed: tuple[PackedCapacity, ...]
    buckets: tuple[PackedCapacity, ...]
    tail_count: int


def load_capacity_study_config(path: Path) -> CapacityStudyConfig:
    root = _mapping(yaml.safe_load(path.read_bytes()), "root")
    if root.get("schema_version") != 1:
        raise CapacityStudyError("capacity study schema_version must be 1")
    parent = _mapping(root.get("parent"), "parent")
    candidates = _mapping(root.get("candidates"), "candidates")
    dense = tuple(
        DenseCapacity(
            _string(item, "name"),
            _positive_int(item.get("path_slots"), "path_slots"),
            _positive_int(item.get("segments_per_path"), "segments_per_path"),
        )
        for item in (
            _mapping(value, "dense candidate")
            for value in _sequence(candidates.get("dense"), "candidates.dense")
        )
    )
    packed = tuple(
        _packed(value, "packed candidate")
        for value in _sequence(candidates.get("packed"), "candidates.packed")
    )
    buckets = tuple(
        _packed(value, "adaptive bucket")
        for value in _sequence(candidates.get("adaptive_buckets"), "adaptive_buckets")
    )
    names = (
        [item.name for item in dense]
        + [item.name for item in packed]
        + [item.name for item in buckets]
    )
    if not dense or not packed or not buckets or len(set(names)) != len(names):
        raise CapacityStudyError("capacity candidates must exist and have unique names")
    if any(
        left.path_slots > right.path_slots
        or left.total_segment_slots > right.total_segment_slots
        for left, right in zip(buckets, buckets[1:], strict=False)
    ):
        raise CapacityStudyError("adaptive buckets must be nondecreasing")
    return CapacityStudyConfig(
        version=_string(root, "study_version"),
        source_revision=_string(root, "source_revision"),
        parent_hybrid=Path(_string(parent, "hybrid")),
        parent_hybrid_sha256=_sha256(_string(parent, "hybrid_sha256")),
        expected_rows=_positive_int(parent.get("expected_rows"), "expected_rows"),
        report_root=Path(_string(root, "report_root")),
        dense=dense,
        packed=packed,
        buckets=buckets,
        tail_count=_positive_int(root.get("tail_count"), "tail_count"),
    )


def dense_retention(lengths: Sequence[int], capacity: DenseCapacity) -> tuple[int, int]:
    """Return retained contours/segments under independent dense path truncation."""

    retained = lengths[: capacity.path_slots]
    return len(retained), sum(min(length, capacity.segments_per_path) for length in retained)


def packed_retention(lengths: Sequence[int], capacity: PackedCapacity) -> tuple[int, int]:
    """Retain a whole-contour prefix so painter order and contour validity remain intact."""

    contours = 0
    segments = 0
    for length in lengths[: capacity.path_slots]:
        if segments + length > capacity.total_segment_slots:
            break
        contours += 1
        segments += length
    return contours, segments


def run_capacity_study(config: CapacityStudyConfig, config_path: Path) -> dict[str, Any]:
    if _file_sha256(config.parent_hybrid) != config.parent_hybrid_sha256:
        raise CapacityStudyError("parent hybrid hash mismatch")
    rows = [
        _mapping(json.loads(line), "hybrid row")
        for line in config.parent_hybrid.read_text(encoding="utf-8").splitlines()
    ]
    if len(rows) != config.expected_rows:
        raise CapacityStudyError("parent row count mismatch")
    if any(row.get("source_revision") != config.source_revision for row in rows):
        raise CapacityStudyError("parent source revision mismatch")
    lengths_by_row = [_lengths(row) for row in rows]
    if any(
        sum(lengths) != int(row["segments"])
        for row, lengths in zip(rows, lengths_by_row, strict=True)
    ):
        raise CapacityStudyError("contour lengths do not balance total segments")

    comparisons: dict[str, Any] = {}
    detail_rows: list[dict[str, Any]] = []
    for dense_capacity in config.dense:
        results = [dense_retention(lengths, dense_capacity) for lengths in lengths_by_row]
        comparisons[dense_capacity.name] = _summarize_candidate(
            rows, lengths_by_row, results, dense_capacity, config.tail_count
        )
        detail_rows.extend(_detail_rows(rows, results, dense_capacity.name, "dense"))
    for packed_capacity in config.packed:
        results = [packed_retention(lengths, packed_capacity) for lengths in lengths_by_row]
        comparisons[packed_capacity.name] = _summarize_candidate(
            rows, lengths_by_row, results, packed_capacity, config.tail_count
        )
        detail_rows.extend(_detail_rows(rows, results, packed_capacity.name, "packed"))

    bucket_counts = {bucket.name: 0 for bucket in config.buckets}
    allocated_slots = 0
    used_slots = 0
    assignments: list[dict[str, Any]] = []
    for row, lengths in zip(rows, lengths_by_row, strict=True):
        for bucket in config.buckets:
            retained = packed_retention(lengths, bucket)
            if retained == (len(lengths), sum(lengths)):
                bucket_counts[bucket.name] += 1
                allocated_slots += bucket.path_slots + bucket.total_segment_slots
                used_slots += len(lengths) + sum(lengths)
                assignments.append(
                    {
                        "hexcode": row["hexcode"],
                        "source_path": row["source_path"],
                        "bucket": bucket.name,
                    }
                )
                break
        else:
            raise CapacityStudyError(f"adaptive buckets do not cover {row['source_path']}")

    detail_payload = _jsonl(detail_rows)
    assignments_payload = _jsonl(assignments)
    summary = {
        "schema_version": 1,
        "study_version": config.version,
        "source_revision": config.source_revision,
        "code_identity": _code_identity(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "parent_hybrid": str(config.parent_hybrid),
        "parent_hybrid_sha256": config.parent_hybrid_sha256,
        "icons": len(rows),
        "distribution": {
            "paths": _stats([len(lengths) for lengths in lengths_by_row]),
            "total_segments": _stats([sum(lengths) for lengths in lengths_by_row]),
            "segments_per_contour_max": _stats(
                [max(lengths, default=0) for lengths in lengths_by_row]
            ),
        },
        "comparisons": comparisons,
        "adaptive_buckets": {
            "policy": [asdict(bucket) for bucket in config.buckets],
            "counts": bucket_counts,
            "allocated_logical_slots": allocated_slots,
            "used_logical_slots": used_slots,
            "utilization": used_slots / allocated_slots,
            "mean_allocated_slots_per_icon": allocated_slots / len(rows),
        },
        "detail_sha256": hashlib.sha256(detail_payload).hexdigest(),
        "assignments_sha256": hashlib.sha256(assignments_payload).hexdigest(),
    }
    _write_bytes_artifact(config.report_root / "detail.jsonl", detail_payload)
    _write_bytes_artifact(config.report_root / "bucket-assignments.jsonl", assignments_payload)
    _write_bytes_artifact(config.report_root / "summary.json", _json(summary))
    _write_bytes_artifact(config.report_root / "README.md", _markdown(summary).encode())
    return summary


def _summarize_candidate(
    rows: list[dict[str, Any]],
    lengths: list[tuple[int, ...]],
    results: list[tuple[int, int]],
    capacity: DenseCapacity | PackedCapacity,
    tail_count: int,
) -> dict[str, Any]:
    losses = [
        (len(item) - retained_paths, sum(item) - retained_segments)
        for item, (retained_paths, retained_segments) in zip(lengths, results, strict=True)
    ]
    segment_slots = (
        capacity.path_slots * capacity.segments_per_path
        if isinstance(capacity, DenseCapacity)
        else capacity.total_segment_slots
    )
    logical_slots = capacity.path_slots + segment_slots
    exact = sum(loss == (0, 0) for loss in losses)
    ranked = sorted(
        zip(rows, losses, strict=True),
        key=lambda item: (-item[1][1], -item[1][0], str(item[0]["source_path"])),
    )
    return {
        "layout": "dense" if isinstance(capacity, DenseCapacity) else "packed",
        "capacity": asdict(capacity),
        "segment_slots": segment_slots,
        "logical_slots": logical_slots,
        "exact_icons": exact,
        "lossy_icons": len(rows) - exact,
        "dropped_contours": sum(item[0] for item in losses),
        "dropped_segments": sum(item[1] for item in losses),
        "partial_contours": (
            sum(
                length > capacity.segments_per_path
                for item in lengths
                for length in item[: capacity.path_slots]
            )
            if isinstance(capacity, DenseCapacity)
            else 0
        ),
        "mean_utilization": float(
            np.mean(
                [
                    (retained_paths + retained_segments) / logical_slots
                    for retained_paths, retained_segments in results
                ]
            )
        ),
        "tail": [
            {
                "hexcode": row["hexcode"],
                "source_path": row["source_path"],
                "dropped_contours": loss[0],
                "dropped_segments": loss[1],
            }
            for row, loss in ranked[:tail_count]
            if loss != (0, 0)
        ],
    }


def _detail_rows(
    rows: list[dict[str, Any]],
    results: list[tuple[int, int]],
    candidate: str,
    layout: str,
) -> list[dict[str, Any]]:
    return [
        {
            "candidate": candidate,
            "layout": layout,
            "hexcode": row["hexcode"],
            "source_path": row["source_path"],
            "input_contours": row["contours"],
            "input_segments": row["segments"],
            "retained_contours": retained[0],
            "retained_segments": retained[1],
            "dropped_contours": int(row["contours"]) - retained[0],
            "dropped_segments": int(row["segments"]) - retained[1],
        }
        for row, retained in zip(rows, results, strict=True)
    ]


def _lengths(row: dict[str, Any]) -> tuple[int, ...]:
    values = tuple(int(value) for value in _sequence(row.get("contour_segment_lengths"), "lengths"))
    if len(values) != int(row["contours"]) or any(value <= 0 for value in values):
        raise CapacityStudyError("invalid contour lengths")
    return values


def _stats(values: Sequence[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    return {
        "min": int(array.min()),
        "median": float(np.median(array)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": int(array.max()),
    }


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Dense versus packed capacity study",
        "",
        f"Complete hybrid corpus: {summary['icons']} icons.",
        "",
        "| candidate | layout | logical slots | exact icons | dropped contours | "
        "dropped segments | mean utilization |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["comparisons"].items():
        lines.append(
            f"| {name} | {item['layout']} | {item['logical_slots']} | "
            f"{item['exact_icons']} | {item['dropped_contours']} | "
            f"{item['dropped_segments']} | {item['mean_utilization']:.4%} |"
        )
    adaptive = summary["adaptive_buckets"]
    lines.extend(
        [
            "",
            f"Adaptive packed buckets are exact for every icon, allocate "
            f"{adaptive['mean_allocated_slots_per_icon']:.2f} logical slots per icon on "
            f"average, and use {adaptive['utilization']:.2%} of them.",
            "",
            "Packed truncation retains a whole-contour prefix; it never creates a partial "
            "contour merely to fit the total-segment budget.",
            "",
        ]
    )
    return "\n".join(lines)


def _packed(value: object, label: str) -> PackedCapacity:
    item = _mapping(value, label)
    return PackedCapacity(
        _string(item, "name"),
        _positive_int(item.get("path_slots"), "path_slots"),
        _positive_int(item.get("total_segment_slots"), "total_segment_slots"),
    )


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
        raise CapacityStudyError(f"cannot identify Git commit: {exc}") from exc
    return {"git_commit": commit, "capacity_study_sha256": _file_sha256(Path(__file__))}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise CapacityStudyError(f"{field} must be a mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise CapacityStudyError(f"{field} must be a list")
    return value


def _string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise CapacityStudyError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CapacityStudyError(f"{field} must be a positive integer")
    return value


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise CapacityStudyError("hash must be lowercase SHA-256")
    return value


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mojidiff-capacity-study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_capacity_study(load_capacity_study_config(args.config), args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
