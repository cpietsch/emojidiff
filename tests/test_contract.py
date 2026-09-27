from __future__ import annotations

from mojidiff.learning.telemetry import RESOURCE_REQUIRED_KEYS as TELEMETRY_KEYS
from mojidiff.orchestration.contract import (
    RESOURCE_REQUIRED_FROM,
    RESOURCE_REQUIRED_KEYS,
    contract_problems,
)

_RESOURCE = {
    "device": "NVIDIA GeForce RTX 4080",
    "peak_vram_gib": 1.9,
    "train_seconds": 300.0,
    "inference_ms_per_icon": 12.5,
}


def test_telemetry_and_contract_agree_on_required_keys() -> None:
    assert TELEMETRY_KEYS == RESOURCE_REQUIRED_KEYS


def test_history_before_cutoff_is_not_judged() -> None:
    record = {"run_id": "old", "state": "completed", "planned_at": "2026-09-21T10:00:00Z"}
    assert contract_problems(record) == []


def test_unfinished_runs_are_not_judged() -> None:
    record = {
        "run_id": "r",
        "state": "running",
        "planned_at": f"{RESOURCE_REQUIRED_FROM}T01:00:00Z",
    }
    assert contract_problems(record) == []


def test_new_completed_run_needs_resource_and_baseline() -> None:
    record = {
        "run_id": "r",
        "state": "completed",
        "planned_at": f"{RESOURCE_REQUIRED_FROM}T01:00:00Z",
    }
    problems = contract_problems(record)
    assert any("`resource` block" in p for p in problems)
    assert any("baseline" in p for p in problems)


def test_missing_and_unexplained_null_keys_are_named() -> None:
    record = {
        "run_id": "r",
        "state": "completed",
        "completed_at": "2026-10-01T00:00:00Z",
        "resource": {"device": "cpu", "peak_vram_gib": None, "train_seconds": 1.0},
        "baselines": {"identity": {"render_recovery": 0.0}},
    }
    problems = contract_problems(record)
    assert problems == [
        "r: `resource` lacks inference_ms_per_icon",
        "r: `resource.peak_vram_gib` is null without `peak_vram_gib_null_reason`",
    ]


def test_explained_null_and_baseline_list_satisfy_the_contract() -> None:
    record = {
        "run_id": "r",
        "state": "completed",
        "planned_at": "2026-10-01T00:00:00Z",
        "resource": {
            **_RESOURCE,
            "peak_vram_gib": None,
            "peak_vram_gib_null_reason": "CPU-only fixture run",
        },
        "baseline": [{"name": "zero-parameter marginal", "nats_per_token": 3.929}],
    }
    assert contract_problems(record) == []
