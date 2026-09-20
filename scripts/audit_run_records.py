#!/usr/bin/env python3
"""Check that the run registry and the per-run records agree.

The run registry is what makes this research reproducible, and it is maintained by
hand across many sessions, so it drifts. This checks the invariants that matter:

* every `runs/*/run.yaml` appears in `state/runs.jsonl`;
* its declared state matches the latest state the registry recorded for it;
* a completed run that predeclared criteria records an outcome somewhere - either a
  `predeclared_outcome` block or explicit `*_passed` flags under `result`;
* the registry parses as strict JSONL, one object per line, no blanks.

Exits non-zero when anything fails, so it can gate a commit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent


def registry_states(path: Path) -> tuple[dict[str, str], list[str]]:
    """Latest recorded state per run id, plus any format problems."""

    problems: list[str] = []
    latest: dict[str, str] = {}
    text = path.read_text()
    if not text.endswith("\n"):
        problems.append(f"{path}: does not end with a newline")
    for number, line in enumerate(text.split("\n")[:-1], 1):
        if not line.strip():
            problems.append(f"{path}:{number}: blank line in an append-only JSONL log")
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            problems.append(f"{path}:{number}: invalid JSON ({error.msg})")
            continue
        if not isinstance(row, dict):
            problems.append(f"{path}:{number}: row is not an object")
            continue
        ids = [row["run_id"]] if isinstance(row.get("run_id"), str) else []
        ids += [item for item in row.get("run_ids", []) if isinstance(item, str)]
        if not ids:
            problems.append(f"{path}:{number}: row names no run_id or run_ids")
        state = row.get("state")
        for run_id in ids:
            if isinstance(state, str):
                latest[run_id] = state
            else:
                latest.setdefault(run_id, "?")
    return latest, problems


def records_outcome(record: dict[str, Any]) -> bool:
    if record.get("predeclared_outcome"):
        return True
    result = record.get("result")
    if isinstance(result, dict):
        return any(key.endswith("passed") for key in result)
    return False


def main() -> int:
    latest, problems = registry_states(_REPO_ROOT / "state" / "runs.jsonl")
    records = sorted((_REPO_ROOT / "runs").glob("*/run.yaml"))
    for path in records:
        record = yaml.safe_load(path.read_text()) or {}
        run_id = record.get("run_id", path.parent.name)
        declared = record.get("state")
        registered = latest.get(run_id)
        if registered is None:
            problems.append(f"{run_id}: run.yaml says {declared} but the registry has no row")
        elif declared != registered:
            problems.append(f"{run_id}: run.yaml says {declared}, registry says {registered}")
        if declared == "completed" and record.get("predeclared_criteria"):
            if not records_outcome(record):
                problems.append(
                    f"{run_id}: completed with predeclared criteria but records no outcome"
                )

    print(f"run records: {len(records)}   registry run ids: {len(latest)}")
    counts = {s: sum(1 for v in latest.values() if v == s) for s in sorted(set(latest.values()))}
    print("states: " + json.dumps(counts))
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        return 1
    print("\nconsistent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
