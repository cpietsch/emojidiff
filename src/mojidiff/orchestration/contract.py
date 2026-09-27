"""The run contract's machine-checked parts, importable without torch.

`scripts/audit_run_records.py` enforces these on `runs/*/run.yaml`; the learning code
writes them through `mojidiff.learning.telemetry`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

RESOURCE_REQUIRED_FROM = "2026-09-27"
"""Completed runs registered on or after this date must carry `resource` and a baseline."""

RESOURCE_REQUIRED_KEYS = ("device", "peak_vram_gib", "train_seconds", "inference_ms_per_icon")
"""Keys the `resource` block must have; a value may be null only with a stated reason."""

BASELINE_KEYS = ("baseline", "baselines")
"""A run carries its zero-parameter or identity comparison under one of these keys."""

_DATE_KEYS = ("planned_at", "staged_at", "registered_at", "started_at", "completed_at")


def registration_date(record: Mapping[str, Any]) -> str | None:
    """The earliest ISO date the record names, as YYYY-MM-DD, or None."""

    dates = [str(record[key])[:10] for key in _DATE_KEYS if record.get(key)]
    return min(dates) if dates else None


def contract_problems(record: Mapping[str, Any]) -> list[str]:
    """Why a completed run record violates the contract, empty when it satisfies it.

    Runs registered before `RESOURCE_REQUIRED_FROM` are history and are not judged.
    """

    if record.get("state") != "completed":
        return []
    date = registration_date(record)
    if date is None or date < RESOURCE_REQUIRED_FROM:
        return []
    run_id = str(record.get("run_id", "?"))
    problems: list[str] = []
    resource = record.get("resource")
    if not isinstance(resource, Mapping):
        problems.append(f"{run_id}: completed without a `resource` block (latency, VRAM, device)")
    else:
        missing = [key for key in RESOURCE_REQUIRED_KEYS if key not in resource]
        if missing:
            problems.append(f"{run_id}: `resource` lacks {', '.join(missing)}")
        for key in RESOURCE_REQUIRED_KEYS:
            if key in resource and resource[key] is None and not resource.get(f"{key}_null_reason"):
                problems.append(f"{run_id}: `resource.{key}` is null without `{key}_null_reason`")
    if not any(
        isinstance(record.get(key), Mapping | list) and record.get(key) for key in BASELINE_KEYS
    ):
        problems.append(
            f"{run_id}: completed without a `baseline`/`baselines` block "
            "(identity, zero-parameter, or prior-run comparison)"
        )
    return problems
