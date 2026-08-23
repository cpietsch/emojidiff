"""Independent invariants for acquired source and reviewed curation artifacts."""

from __future__ import annotations

import hashlib
import json
import stat
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from mojidiff.curation.source import OpenMojiSource, sha256_file


class ValidationError(RuntimeError):
    """A curation artifact violates a traceability or leakage invariant."""


def validate_curation(source: OpenMojiSource, decisions_path: Path, output: Path) -> dict[str, Any]:
    source_manifest = json.loads(source.source_manifest.read_text(encoding="utf-8"))
    source_rows = source_manifest.get("svgs")
    if not isinstance(source_rows, list):
        raise ValidationError("source manifest SVG rows are missing")
    decisions = pq.read_table(decisions_path).to_pylist()

    checks: dict[str, Any] = {}
    manifest_paths = {str(row["path"]) for row in source_rows}
    decision_paths = {str(row["source_path"]) for row in decisions}
    checks["source_row_count"] = len(source_rows)
    checks["decision_row_count"] = len(decisions)
    checks["source_paths_unique"] = len(manifest_paths) == len(source_rows)
    checks["decision_paths_unique"] = len(decision_paths) == len(decisions)
    checks["source_decision_path_sets_equal"] = manifest_paths == decision_paths

    mismatched_hashes: list[str] = []
    writable_paths: list[str] = []
    for row in source_rows:
        relative = str(row["path"])
        path = source.raw_root / relative
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            mismatched_hashes.append(relative)
        if path.exists() and stat.S_IMODE(path.stat().st_mode) & 0o222:
            writable_paths.append(relative)
    for path in (source.raw_root, *source.raw_root.rglob("*")):
        if stat.S_IMODE(path.stat().st_mode) & 0o222:
            writable_paths.append(path.relative_to(source.raw_root).as_posix() or ".")
    checks["source_hash_mismatches"] = mismatched_hashes
    checks["writable_raw_paths"] = sorted(set(writable_paths))
    checks["raw_git_clean"] = _git_clean(source.raw_root)

    flags = [row for row in decisions if row.get("group") == "flags"]
    checks["flag_count"] = len(flags)
    checks["all_metadata_flags_reversible"] = all(
        row.get("split") == "excluded/flags"
        and row.get("curation_status") == "exclude_policy"
        and row.get("reason_codes") == ["SCOPE_FLAGS"]
        for row in flags
    )
    checks["primary_contains_no_metadata_flags"] = not any(
        row.get("group") == "flags" and str(row.get("split", "")).startswith("primary/")
        for row in decisions
    )
    checks["unresolved_quarantine_count"] = sum(
        str(row.get("curation_status", "")).startswith("quarantine") for row in decisions
    )

    primary = [row for row in decisions if str(row.get("split", "")).startswith("primary/")]
    by_family: dict[str, set[str]] = defaultdict(set)
    by_exact: dict[str, set[str]] = defaultdict(set)
    for row in primary:
        by_family[str(row.get("split_family_cluster"))].add(str(row["split"]))
        if row.get("exact_raster_cluster"):
            by_exact[str(row["exact_raster_cluster"])].add(str(row["hexcode"]))
    checks["family_clusters_crossing_splits"] = sorted(
        family for family, splits in by_family.items() if len(splits) > 1
    )
    checks["exact_clusters_with_multiple_primary_rows"] = {
        cluster: sorted(hexcodes) for cluster, hexcodes in by_exact.items() if len(hexcodes) > 1
    }
    checks["primary_count"] = len(primary)
    checks["reviewed_override_count"] = sum(
        row.get("curation_status") == "include_override" for row in decisions
    )

    required_true = (
        "source_paths_unique",
        "decision_paths_unique",
        "source_decision_path_sets_equal",
        "raw_git_clean",
        "all_metadata_flags_reversible",
        "primary_contains_no_metadata_flags",
    )
    failures = [name for name in required_true if checks[name] is not True]
    failures.extend(
        name
        for name in (
            "source_hash_mismatches",
            "writable_raw_paths",
            "family_clusters_crossing_splits",
            "exact_clusters_with_multiple_primary_rows",
        )
        if checks[name]
    )
    if checks["unresolved_quarantine_count"]:
        failures.append("unresolved_quarantine_count")
    result = {
        "schema_version": 1,
        "source_revision": source.revision,
        "source_manifest_sha256": hashlib.sha256(source.source_manifest.read_bytes()).hexdigest(),
        "decisions_path": str(decisions_path),
        "decisions_sha256": hashlib.sha256(decisions_path.read_bytes()).hexdigest(),
        "checks": checks,
        "passed": not failures,
        "failures": sorted(set(failures)),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise ValidationError(f"curation validation failed: {', '.join(sorted(set(failures)))}")
    return result


def _git_clean(root: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"],
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
        shell=False,
    )
    return completed.returncode == 0 and not completed.stdout.strip()
