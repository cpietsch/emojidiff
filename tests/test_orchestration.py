from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from mojidiff.orchestration.adapters import (
    SSH_OPTIONS,
    AdapterError,
    ProbeOnlySshAdapter,
    UnsupportedOperation,
    adapter_for,
)
from mojidiff.orchestration.config import WorkerSpec, load_inventory
from mojidiff.orchestration.local_probe import collect_local_probe
from mojidiff.orchestration.registry import append_event


def test_example_inventory_fails_closed() -> None:
    inventory = load_inventory(Path("hosts.example.yaml"))

    assert inventory.orchestrator_name == "gtc"
    assert inventory.artifact_store.missing_fields() == (
        "artifact_store.type",
        "artifact_store.uri",
        "artifact_store.credentials_source",
    )
    assert inventory.enabled_workers() == ()
    assert all(not worker.compute_authorized for worker in inventory.workers.values())
    assert inventory.workers["vast_5090"].missing_cap_fields() == (
        "workers.vast_5090.resource_cap.max_steps",
        "workers.vast_5090.resource_cap.max_spend_usd",
        "workers.vast_5090.resource_cap.max_storage_gb",
    )


def test_probe_plan_is_noninteractive_and_constant() -> None:
    worker = load_inventory(Path("hosts.example.yaml")).workers["vast_5090"]
    plan = adapter_for(worker).probe()
    description = plan.safe_description()

    assert plan.argv[: 1 + len(SSH_OPTIONS)] == ("ssh", *SSH_OPTIONS)
    assert plan.argv[-4:] == ("--", "vast-5090", "python3", "-")
    assert plan.stdin is not None
    assert "stdin_sha256" in description
    assert "stdin" not in description
    assert "shell" not in description


def test_probe_execute_uses_no_shell() -> None:
    worker = load_inventory(Path("hosts.example.yaml")).workers["owned_gpu"]
    plan = adapter_for(worker).probe()
    calls: list[dict[str, Any]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, '{"hostname":"fixture"}\n', "")

    assert plan.execute(runner=runner) == {"hostname": "fixture"}
    assert calls[0]["shell"] is False
    assert calls[0]["input"] == plan.stdin
    assert calls[0]["argv"] == list(plan.argv)


def test_invalid_alias_is_rejected_before_subprocess() -> None:
    worker = WorkerSpec(
        name="bad",
        enabled=False,
        ssh_alias="host; touch /tmp/not-safe",
        kind="owned-persistent",
        execution="docker",
        workspace_root="/srv/mojidiff",
        image="example.invalid/image@sha256:abc",
        resource_cap={"max_steps": 1, "max_storage_gb": 1},
        raw={},
    )
    with pytest.raises(AdapterError, match="invalid ssh_alias"):
        ProbeOnlySshAdapter(worker).probe()


def test_mutating_operations_fail_closed() -> None:
    worker = load_inventory(Path("hosts.example.yaml")).workers["a100_cluster"]
    adapter = adapter_for(worker)

    for operation in (
        adapter.stage,
        adapter.smoke,
        adapter.launch,
        adapter.status,
        adapter.sync,
        adapter.cancel,
    ):
        with pytest.raises(UnsupportedOperation):
            operation()


def test_registry_appends_without_rewriting(tmp_path: Path) -> None:
    registry = tmp_path / "runs.jsonl"
    first = {"run_id": "fixture-a", "state": "planned", "timestamp": "2026-08-23T00:00:00Z"}
    second = {"run_id": "fixture-a", "state": "failed", "timestamp": "2026-08-23T00:00:01Z"}

    append_event(registry, first)
    original = registry.read_bytes()
    append_event(registry, second)
    combined = registry.read_bytes()

    assert combined.startswith(original)
    assert [json.loads(line) for line in combined.splitlines()] == [first, second]
    assert registry.stat().st_mode & 0o777 == 0o600


def test_local_probe_is_sanitized() -> None:
    probe = collect_local_probe(Path.cwd())
    serialized = json.dumps(probe)

    assert probe["hostname"] == "gtc"
    assert probe["repo_path"].endswith("/mojidiff")
    assert "environment" not in probe
    assert "DOCKER_CERT_PATH" not in serialized
    assert "token" not in serialized.lower()
