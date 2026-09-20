from __future__ import annotations

import base64
import io
import json
import os
import shlex
import subprocess
import tarfile
from pathlib import Path
from typing import Any

import pytest

from mojidiff.orchestration.adapters import (
    REMOTE_OWNED_DOCKER_SMOKE,
    SSH_OPTIONS,
    AdapterError,
    ProbeOnlySshAdapter,
    SmokeRequest,
    StageRequest,
    UnsupportedOperation,
    adapter_for,
    snapshot_identity,
)
from mojidiff.orchestration.config import ArtifactStore, WorkerSpec, load_inventory
from mojidiff.orchestration.local_probe import collect_local_probe
from mojidiff.orchestration.registry import append_event


def _snapshot_archive(files: dict[str, bytes] | None = None) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w") as archive:
        for name, payload in sorted((files or {"README.md": b"fixture\n"}).items()):
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            member.mode = 0o644
            member.mtime = 0
            archive.addfile(member, io.BytesIO(payload))
    return buffer.getvalue()


def _authorized_vast(workspace_root: str = "/workspace/mojidiff-runs") -> WorkerSpec:
    return WorkerSpec(
        name="vast_fixture",
        enabled=True,
        ssh_alias="vast-fixture",
        kind="vast-ephemeral",
        execution="container-shell",
        workspace_root=workspace_root,
        image="example.invalid/pytorch@sha256:abc",
        resource_cap={"max_steps": 2, "max_spend_usd": 1.0, "max_storage_gb": 1},
        raw={},
    )


def _authorized_owned(workspace_root: str = "/home/hans/mojidiff-runs") -> WorkerSpec:
    return WorkerSpec(
        name="owned_fixture",
        enabled=True,
        ssh_alias="owned-fixture",
        kind="owned-persistent",
        execution="docker",
        workspace_root=workspace_root,
        image="nvcr.io/nvidia/pytorch:25.06-py3",
        resource_cap={"max_steps": 2, "max_storage_gb": 1},
        raw={
            "workspace_volume": "code-server-gpu_gpubox-workspace",
            "artifact_volume": "code-server-gpu_gpubox-home",
            "artifact_volume_subpath": ".cache",
        },
    )


def _artifact_store(path: str = "/durable/mojidiff") -> ArtifactStore:
    return ArtifactStore(
        type="worker-filesystem",
        uri=f"file://{path}",
        credentials_source="pre-mounted-read-write-volume",
    )


def _stage_request() -> StageRequest:
    return StageRequest(
        run_id="fixture-run",
        archive=_snapshot_archive(),
        git_revision="a" * 40,
        config_sha256="b" * 64,
    )


def _decoded_remote_config(command: str) -> dict[str, Any]:
    remote = shlex.split(command)
    encoded = remote[-1]
    padding = "=" * (-len(encoded) % 4)
    decoded = json.loads(base64.urlsafe_b64decode(encoded + padding))
    assert isinstance(decoded, dict)
    return decoded


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

    with pytest.raises(UnsupportedOperation):
        adapter.stage(_stage_request())
    with pytest.raises(UnsupportedOperation):
        adapter.smoke(SmokeRequest("fixture-run", snapshot_identity(_stage_request())))
    for operation in (adapter.launch, adapter.status, adapter.sync, adapter.cancel):
        with pytest.raises(UnsupportedOperation):
            operation()


def test_vast_mutations_require_enabled_complete_caps_and_artifact_store() -> None:
    disabled = load_inventory(Path("hosts.example.yaml")).workers["vast_5090"]
    request = _stage_request()

    with pytest.raises(AdapterError, match=r"enabled=true.*max_steps.*artifact_store"):
        adapter_for(disabled).stage(request)
    with pytest.raises(AdapterError, match="artifact_store"):
        adapter_for(_authorized_vast()).stage(request)


def test_vast_stage_plan_is_bounded_binary_and_redacted() -> None:
    request = _stage_request()
    plan = adapter_for(_authorized_vast(), _artifact_store()).stage(request)
    description = plan.safe_description()

    assert plan.argv[: 1 + len(SSH_OPTIONS)] == ("ssh", *SSH_OPTIONS)
    assert plan.argv[-2] == "vast-fixture"
    assert plan.stdin == request.archive
    assert description["stdin_sha256"] == snapshot_identity(request).archive_sha256
    assert description["stdin_bytes"] == len(request.archive)
    assert description["argv"][-1].startswith("<redacted:")
    assert request.git_revision not in json.dumps(description)
    remote = shlex.split(plan.argv[-1])
    assert remote[:3] == ["python3", "-I", "-c"]
    config = _decoded_remote_config(plan.argv[-1])
    identity = snapshot_identity(request)
    assert config["archive_sha256"] == identity.archive_sha256
    assert config["tree_sha256"] == identity.tree_sha256
    assert config["workspace_root"] == "/workspace/mojidiff-runs"
    assert config["max_archive_bytes"] == 512 * 1024 * 1024


def test_vast_stage_remote_program_verifies_and_is_idempotent(tmp_path: Path) -> None:
    request = _stage_request()
    workspace = tmp_path / "worker-runs"
    plan = adapter_for(
        _authorized_vast(str(workspace)), _artifact_store(str(tmp_path / "durable"))
    ).stage(request)
    command = shlex.split(plan.argv[-1])

    first = subprocess.run(
        command,
        input=request.archive,
        capture_output=True,
        check=False,
        shell=False,
    )
    second = subprocess.run(
        command,
        input=request.archive,
        capture_output=True,
        check=False,
        shell=False,
    )

    assert first.returncode == second.returncode == 0
    result = json.loads(first.stdout)
    assert result["tree_sha256"] == snapshot_identity(request).tree_sha256
    source = workspace / request.run_id / "source"
    assert (source / "README.md").read_bytes() == b"fixture\n"
    assert (source / ".mojidiff-stage.json").is_file()
    assert source.stat().st_mode & 0o222 == 0


def test_vast_stage_rejects_unsafe_tar_members() -> None:
    archive = _snapshot_archive({"../escape": b"unsafe"})
    request = StageRequest("fixture-run", archive, "a" * 40, "b" * 64)

    with pytest.raises(AdapterError, match="escapes"):
        adapter_for(_authorized_vast(), _artifact_store()).stage(request)


def test_vast_stage_binary_execute_uses_no_shell() -> None:
    plan = adapter_for(_authorized_vast(), _artifact_store()).stage(_stage_request())
    calls: list[dict[str, Any]] = []

    def runner(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        calls.append({"argv": argv, **kwargs})
        return subprocess.CompletedProcess(argv, 0, b'{"operation":"stage"}\n', b"")

    assert plan.execute(runner=runner) == {"operation": "stage"}
    assert calls[0]["shell"] is False
    assert calls[0]["text"] is False
    assert calls[0]["input"] == plan.stdin


def test_vast_smoke_rejects_tampered_stage_before_project_imports(tmp_path: Path) -> None:
    smoke_script = Path("scripts/remote/vast_tiny_smoke.py").read_bytes()
    request = StageRequest(
        run_id="fixture-run",
        archive=_snapshot_archive(
            {
                "README.md": b"original\n",
                "scripts/remote/vast_tiny_smoke.py": smoke_script,
            }
        ),
        git_revision="a" * 40,
        config_sha256="b" * 64,
    )
    workspace = tmp_path / "worker-runs"
    artifact = tmp_path / "durable"
    artifact.mkdir()
    adapter = adapter_for(_authorized_vast(str(workspace)), _artifact_store(str(artifact)))
    stage = adapter.stage(request)
    staged = subprocess.run(
        shlex.split(stage.argv[-1]),
        input=request.archive,
        capture_output=True,
        check=False,
        shell=False,
    )
    assert staged.returncode == 0
    readme = workspace / request.run_id / "source/README.md"
    readme.chmod(0o644)
    readme.write_bytes(b"tampered\n")

    smoke = adapter.smoke(SmokeRequest(request.run_id, snapshot_identity(request)))
    attempted = subprocess.run(
        shlex.split(smoke.argv[-1]),
        capture_output=True,
        check=False,
        shell=False,
        text=True,
    )

    assert attempted.returncode == 1
    assert "tree hash does not match" in attempted.stderr
    assert not (workspace / request.run_id / "smoke").exists()


def test_vast_smoke_plan_is_staged_quote_safe_and_redacted() -> None:
    request = _stage_request()
    identity = snapshot_identity(request)
    plan = adapter_for(_authorized_vast(), _artifact_store()).smoke(
        SmokeRequest(request.run_id, identity)
    )
    description = plan.safe_description()
    remote = shlex.split(plan.argv[-1])

    assert remote[:3] == ["python3", "-I", "-B"]
    assert remote[3].endswith("/fixture-run/source/scripts/remote/vast_tiny_smoke.py")
    assert remote[4] == "--config"
    config = _decoded_remote_config(plan.argv[-1])
    assert config["artifact_root"] == "/durable/mojidiff"
    assert config["archive_sha256"] == identity.archive_sha256
    assert config["max_steps"] == 2
    assert description["argv"][-1].startswith("<redacted:")
    assert "durable/mojidiff" not in json.dumps(description)


def test_vast_smoke_rejects_unimplemented_or_ephemeral_sink() -> None:
    request = SmokeRequest("fixture-run", snapshot_identity(_stage_request()))
    unsupported = ArtifactStore("s3-compatible", "s3://bucket/prefix", "worker-secret-file")
    nested = _artifact_store("/workspace/mojidiff-runs/artifacts")
    ancestor = _artifact_store("/workspace")

    with pytest.raises(UnsupportedOperation, match="worker-filesystem"):
        adapter_for(_authorized_vast(), unsupported).smoke(request)
    with pytest.raises(AdapterError, match="must be disjoint"):
        adapter_for(_authorized_vast(), nested).smoke(request)
    with pytest.raises(AdapterError, match="must be disjoint"):
        adapter_for(_authorized_vast(), ancestor).smoke(request)


def test_owned_smoke_plan_uses_a_redacted_docker_launcher() -> None:
    request = _stage_request()
    identity = snapshot_identity(request)
    plan = adapter_for(_authorized_owned(), _artifact_store("/home/hans/mojidiff-artifacts")).smoke(
        SmokeRequest(request.run_id, identity)
    )
    description = plan.safe_description()
    config = _decoded_remote_config(plan.argv[-1])

    assert plan.operation == "smoke"
    assert plan.timeout_seconds == 600
    assert description["argv"][-1].startswith("<redacted:")
    assert config["workspace_root"] == "/home/hans/mojidiff-runs"
    assert config["artifact_root"] == "/home/hans/mojidiff-artifacts"
    assert config["workspace_volume"] == "code-server-gpu_gpubox-workspace"
    assert config["artifact_volume"] == "code-server-gpu_gpubox-home"
    assert config["artifact_volume_subpath"] == ".cache"
    assert config["image"] == "nvcr.io/nvidia/pytorch:25.06-py3"
    assert config["run_id"] == request.run_id


def test_owned_smoke_requires_safe_named_volume_mapping() -> None:
    request = SmokeRequest("fixture-run", snapshot_identity(_stage_request()))
    worker = _authorized_owned()
    missing = WorkerSpec(
        name=worker.name,
        enabled=worker.enabled,
        ssh_alias=worker.ssh_alias,
        kind=worker.kind,
        execution=worker.execution,
        workspace_root=worker.workspace_root,
        image=worker.image,
        resource_cap=worker.resource_cap,
        raw={},
    )
    unsafe = WorkerSpec(
        name=worker.name,
        enabled=worker.enabled,
        ssh_alias=worker.ssh_alias,
        kind=worker.kind,
        execution=worker.execution,
        workspace_root=worker.workspace_root,
        image=worker.image,
        resource_cap=worker.resource_cap,
        raw={
            "workspace_volume": "valid-volume",
            "artifact_volume": "also-valid",
            "artifact_volume_subpath": "../escape",
        },
    )

    with pytest.raises(AdapterError, match="workspace_volume"):
        adapter_for(missing, _artifact_store()).smoke(request)
    with pytest.raises(AdapterError, match="artifact_volume_subpath"):
        adapter_for(unsafe, _artifact_store()).smoke(request)


def test_owned_smoke_remote_launcher_uses_valid_scoped_volume_mounts(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    artifact = tmp_path / "home" / ".cache"
    (workspace / "fixture-run").mkdir(parents=True)
    artifact.mkdir(parents=True)
    log = tmp_path / "docker-argv.log"
    fake_docker = tmp_path / "docker"
    fake_docker.write_text(
        "#!/usr/bin/env python3\n"
        "import os, sys\n"
        "with open(os.environ['FAKE_DOCKER_LOG'], 'a', encoding='utf-8') as stream:\n"
        "    stream.write('\\0'.join(sys.argv[1:]) + '\\n')\n",
        encoding="utf-8",
    )
    fake_docker.chmod(0o755)
    config = {
        "artifact_root": str(artifact),
        "artifact_volume": "artifact-volume",
        "artifact_volume_subpath": ".cache",
        "archive_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "git_revision": "c" * 40,
        "image": "example.invalid/pytorch:fixed",
        "max_steps": 2,
        "max_storage_bytes": 1_000_000,
        "run_id": "fixture-run",
        "smoke_id": "tiny-smoke-v1",
        "tree_sha256": "d" * 64,
        "workspace_root": str(workspace),
        "workspace_volume": "workspace-volume",
    }
    encoded = base64.urlsafe_b64encode(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    environment = os.environ.copy()
    environment["FAKE_DOCKER_LOG"] = str(log)
    environment["PATH"] = f"{tmp_path}:{environment['PATH']}"

    result = subprocess.run(
        ["python3", "-I", "-c", REMOTE_OWNED_DOCKER_SMOKE, encoded],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    invocations = log.read_text(encoding="utf-8").splitlines()
    assert len(invocations) == 2
    run_argv = invocations[1].split("\0")
    mounts = [run_argv[index + 1] for index, value in enumerate(run_argv) if value == "--mount"]
    assert mounts == [
        "type=volume,src=workspace-volume,dst=/mojidiff/workspace/fixture-run,"
        "volume-subpath=fixture-run,volume-nocopy",
        "type=volume,src=artifact-volume,dst=/mojidiff/artifacts/fixture-run,"
        "volume-subpath=.cache/fixture-run,volume-nocopy",
    ]
    entrypoint = run_argv.index("--entrypoint")
    assert run_argv[entrypoint : entrypoint + 4] == [
        "--entrypoint",
        "python3",
        "example.invalid/pytorch:fixed",
        "-I",
    ]
    assert "python3" not in run_argv[entrypoint + 3 :]
    assert run_argv[-3].endswith("/scripts/remote/vast_tiny_smoke.py")

    config["smoke_id"] = "openmoji-g1-pipeline-v1"
    encoded = base64.urlsafe_b64encode(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    ).decode().rstrip("=")
    result = subprocess.run(
        ["python3", "-I", "-c", REMOTE_OWNED_DOCKER_SMOKE, encoded],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    pipeline_argv = log.read_text(encoding="utf-8").splitlines()[-1].split("\0")
    assert pipeline_argv[-3].endswith("/scripts/remote/openmoji_pilot_smoke.py")


def test_owned_smoke_requires_docker_execution() -> None:
    worker = _authorized_owned()
    invalid = WorkerSpec(
        name=worker.name,
        enabled=worker.enabled,
        ssh_alias=worker.ssh_alias,
        kind=worker.kind,
        execution="container-shell",
        workspace_root=worker.workspace_root,
        image=worker.image,
        resource_cap=worker.resource_cap,
        raw=worker.raw,
    )
    with pytest.raises(AdapterError, match="execution=docker"):
        adapter_for(invalid, _artifact_store()).smoke(
            SmokeRequest("fixture-run", snapshot_identity(_stage_request()))
        )


def test_vast_dynamic_identifiers_are_validated_before_command_planning() -> None:
    bad = StageRequest(
        run_id="fixture; touch /tmp/unsafe",
        archive=_snapshot_archive(),
        git_revision="a" * 40,
        config_sha256="b" * 64,
    )

    with pytest.raises(AdapterError, match="run_id"):
        adapter_for(_authorized_vast(), _artifact_store()).stage(bad)


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
