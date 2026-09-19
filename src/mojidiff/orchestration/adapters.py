"""Authorization-aware worker plans with a fail-closed Vast implementation."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import re
import shlex
import tarfile
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Never
from urllib.parse import unquote, urlsplit

from mojidiff.orchestration.config import ArtifactStore, WorkerSpec
from mojidiff.orchestration.plans import CommandPlan


class AdapterError(RuntimeError):
    """A worker operation cannot be planned safely."""


class UnsupportedOperation(AdapterError):
    """The adapter contract includes an operation not implemented yet."""


_SSH_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION = re.compile(r"[0-9a-f]{7,64}\Z")
_MAX_SNAPSHOT_BYTES = 512 * 1024 * 1024
_MAX_SNAPSHOT_MEMBERS = 50_000
_STAGE_MANIFEST = ".mojidiff-stage.json"
_FILESYSTEM_SINK = "worker-filesystem"
_CONTAINER_IMAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._/@:+-]{0,511}\Z")

SSH_OPTIONS = (
    "-o",
    "BatchMode=yes",
    "-o",
    "ForwardAgent=no",
    "-o",
    "ConnectTimeout=15",
    "-o",
    "ServerAliveInterval=30",
    "-o",
    "ServerAliveCountMax=3",
)

REMOTE_PROBE = r"""import json
import os
import platform
import shutil
import socket
import subprocess
import sys


def command(argv):
    try:
        result = subprocess.run(
            argv,
            text=True,
            capture_output=True,
            check=False,
            timeout=15,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": True,
        "exit_code": result.returncode,
        "stdout": result.stdout.strip()[:8000],
        "stderr": result.stderr.strip()[-1000:],
    }


disk = shutil.disk_usage("/")
probe = {
    "schema_version": 1,
    "hostname": socket.gethostname(),
    "platform": platform.platform(),
    "python": {
        "version": platform.python_version(),
        "executable": sys.executable,
    },
    "disk_root": {"total": disk.total, "used": disk.used, "free": disk.free},
    "gpu": command([
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
        "--format=csv,noheader,nounits",
    ]),
    "nvidia_smi": command(["nvidia-smi", "--version"]),
    "container_markers": {
        key: os.environ.get(key)
        for key in ("NVIDIA_BUILD_ID", "NVIDIA_PYTORCH_VERSION", "CUDA_VERSION")
        if os.environ.get(key)
    },
    "docker_client": command(["docker", "version", "--format", "{{json .Client.Version}}"]),
    "kubectl_client": command(["kubectl", "version", "--client=true", "-o", "json"]),
}

try:
    import torch
except Exception as exc:
    probe["torch"] = {"available": False, "error": type(exc).__name__}
else:
    probe["torch"] = {
        "available": True,
        "version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
    }

print(json.dumps(probe, sort_keys=True))
"""


REMOTE_STAGE = r"""import hashlib
import io
import json
import os
import re
import sys
import tarfile
from pathlib import Path, PurePosixPath

RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
GIT_REVISION = re.compile(r"[0-9a-f]{7,64}\Z")
STAGE_MANIFEST = ".mojidiff-stage.json"
MAX_MEMBERS = 50000


def fail(message):
    raise SystemExit(message)


def clean_root(value):
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        fail("invalid workspace root")
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value.rstrip("/")
    ):
        fail("workspace root must be an absolute normalized POSIX path")
    return Path(path)


def member_name(raw):
    if not isinstance(raw, str) or not raw or "\x00" in raw or "\\" in raw:
        fail("snapshot contains an invalid member name")
    name = raw.rstrip("/")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        fail("snapshot member escapes its root")
    if name == STAGE_MANIFEST:
        fail("snapshot contains the reserved stage manifest")
    return path.as_posix()


def inspect_snapshot(payload, maximum):
    if len(payload) > maximum:
        fail("snapshot archive exceeds the configured bound")
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:*")
    except tarfile.TarError as exc:
        fail(f"invalid snapshot archive: {type(exc).__name__}")
    records = []
    seen = set()
    total = 0
    with archive:
        for member_count, member in enumerate(archive, start=1):
            if member_count > MAX_MEMBERS:
                fail("snapshot contains too many members")
            name = member_name(member.name)
            if name in seen:
                fail("snapshot contains duplicate member paths")
            seen.add(name)
            if not (member.isdir() or member.isreg()):
                fail("snapshot contains links or special files")
            if member.isdir():
                continue
            total += member.size
            if total > maximum:
                fail("expanded snapshot exceeds the configured bound")
            source = archive.extractfile(member)
            if source is None:
                fail("snapshot member could not be read")
            digest = hashlib.sha256()
            read_bytes = 0
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                read_bytes += len(chunk)
                digest.update(chunk)
            if read_bytes != member.size:
                fail("snapshot member size changed while reading")
            records.append((name, member.size, digest.digest(), member.mode))
    if not records:
        fail("snapshot contains no regular files")
    regular = {item[0] for item in records}
    for name in seen:
        for parent in PurePosixPath(name).parents:
            parent_name = parent.as_posix()
            if parent_name == ".":
                break
            if parent_name in regular:
                fail("a regular file is the parent of another snapshot member")
    tree = hashlib.sha256(b"mojidiff-tree-v1\0")
    for name, size, digest, _mode in sorted(records):
        tree.update(name.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(size).encode("ascii"))
        tree.update(b"\0")
        tree.update(digest)
    return tree.hexdigest(), records


def inspect_directory(root):
    records = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            fail("staged source contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            fail("staged source contains a special file")
        name = path.relative_to(root).as_posix()
        if name == STAGE_MANIFEST:
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as source:
            while True:
                chunk = source.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
        records.append((name, size, digest.digest()))
    tree = hashlib.sha256(b"mojidiff-tree-v1\0")
    for name, size, digest in records:
        tree.update(name.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(size).encode("ascii"))
        tree.update(b"\0")
        tree.update(digest)
    return tree.hexdigest()


def stage(config, payload):
    required = {
        "archive_sha256",
        "config_sha256",
        "git_revision",
        "max_archive_bytes",
        "run_id",
        "tree_sha256",
        "workspace_root",
    }
    if not isinstance(config, dict) or set(config) != required:
        fail("invalid stage configuration fields")
    run_id = config["run_id"]
    if not isinstance(run_id, str) or RUN_ID.fullmatch(run_id) is None:
        fail("invalid run_id")
    for key in ("archive_sha256", "config_sha256", "tree_sha256"):
        if not isinstance(config[key], str) or SHA256.fullmatch(config[key]) is None:
            fail(f"invalid {key}")
    revision = config["git_revision"]
    if not isinstance(revision, str) or GIT_REVISION.fullmatch(revision) is None:
        fail("invalid git_revision")
    maximum = config["max_archive_bytes"]
    if isinstance(maximum, bool) or not isinstance(maximum, int) or maximum <= 0:
        fail("invalid archive bound")
    archive_hash = hashlib.sha256(payload).hexdigest()
    if archive_hash != config["archive_sha256"]:
        fail("snapshot transfer hash mismatch")
    tree_hash, records = inspect_snapshot(payload, maximum)
    if tree_hash != config["tree_sha256"]:
        fail("snapshot tree hash mismatch")

    root = clean_root(config["workspace_root"])
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir() or root.resolve() != root:
        fail("workspace root must be a real directory without symbolic-link parents")
    run_root = root / run_id
    if run_root.is_symlink() or (run_root.exists() and not run_root.is_dir()):
        fail("run root is not a real directory")
    run_root.mkdir(exist_ok=True)
    destination = run_root / "source"
    expected = {
        "archive_sha256": archive_hash,
        "config_sha256": config["config_sha256"],
        "git_revision": revision,
        "run_id": run_id,
        "schema_version": 1,
        "tree_sha256": tree_hash,
    }
    manifest_path = destination / STAGE_MANIFEST
    if destination.is_symlink():
        fail("existing stage is a symbolic link")
    if destination.exists():
        if manifest_path.is_symlink():
            fail("existing stage manifest is a symbolic link")
        try:
            observed = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            fail(f"existing stage is incomplete: {type(exc).__name__}")
        if observed != expected:
            fail("existing stage has a different immutable identity")
        if inspect_directory(destination) != tree_hash:
            fail("existing staged source tree hash mismatch")
        print(
            json.dumps(
                {"operation": "stage", "path": str(destination), **expected}, sort_keys=True
            )
        )
        return

    destination.mkdir()
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:*")
        modes = {name: mode for name, _size, _digest, mode in records}
        directories = []
        with archive:
            for member in archive.getmembers():
                name = member_name(member.name)
                target = destination.joinpath(*PurePosixPath(name).parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                    directories.append(target)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    fail("snapshot member could not be read during extraction")
                with target.open("xb") as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        output.write(chunk)
                    output.flush()
                    os.fsync(output.fileno())
                target.chmod(0o555 if modes[name] & 0o111 else 0o444)
        manifest_path.write_text(json.dumps(expected, sort_keys=True) + "\n", encoding="utf-8")
        manifest_path.chmod(0o444)
        directories.extend(path for path in destination.rglob("*") if path.is_dir())
        for directory in sorted(set(directories), key=lambda item: len(item.parts), reverse=True):
            directory.chmod(0o555)
        if inspect_directory(destination) != tree_hash:
            fail("extracted source tree hash mismatch")
        destination.chmod(0o555)
    except BaseException:
        # Preserve the partial directory as failure evidence; never overwrite or delete it.
        raise
    print(json.dumps({"operation": "stage", "path": str(destination), **expected}, sort_keys=True))


if len(sys.argv) != 2:
    fail("stage expects one encoded configuration argument")
try:
    padding = "=" * (-len(sys.argv[1]) % 4)
    decoded = __import__("base64").urlsafe_b64decode(sys.argv[1] + padding)
    configuration = json.loads(decoded)
except (ValueError, UnicodeError) as exc:
    fail(f"invalid encoded stage configuration: {type(exc).__name__}")
limit = configuration.get("max_archive_bytes", 0) if isinstance(configuration, dict) else 0
if isinstance(limit, bool) or not isinstance(limit, int) or limit <= 0:
    fail("invalid archive bound")
snapshot = sys.stdin.buffer.read(limit + 1)
stage(configuration, snapshot)
"""


REMOTE_OWNED_DOCKER_SMOKE = r"""import base64
import json
import os
import subprocess
import sys


def fail(message):
    raise SystemExit(message)


if len(sys.argv) != 2:
    fail("owned smoke expects one encoded configuration argument")
try:
    encoded = sys.argv[1]
    padding = "=" * (-len(encoded) % 4)
    config = json.loads(base64.urlsafe_b64decode(encoded + padding))
except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
    fail("invalid owned smoke configuration: " + type(exc).__name__)

required = {
    "artifact_root", "archive_sha256", "config_sha256", "git_revision", "image",
    "max_steps", "max_storage_bytes", "run_id", "smoke_id", "tree_sha256",
    "workspace_root",
}
if not isinstance(config, dict) or set(config) != required:
    fail("invalid owned smoke configuration fields")

workspace_host = config["workspace_root"]
artifact_host = config["artifact_root"]
if not all(
    isinstance(value, str) and value.startswith("/") and "\x00" not in value and "\n" not in value
    for value in (workspace_host, artifact_host)
):
    fail("invalid owned smoke path")
if not isinstance(config["image"], str) or "\x00" in config["image"] or "\n" in config["image"]:
    fail("invalid owned smoke image")
workspace_container = "/mojidiff/workspace"
artifact_container = "/mojidiff/artifacts"
source_root = workspace_container + "/" + config["run_id"] + "/source"
inner = {
    "artifact_root": artifact_container,
    "archive_sha256": config["archive_sha256"],
    "config_sha256": config["config_sha256"],
    "git_revision": config["git_revision"],
    "max_steps": config["max_steps"],
    "max_storage_bytes": config["max_storage_bytes"],
    "run_id": config["run_id"],
    "smoke_id": config["smoke_id"],
    "source_root": source_root,
    "tree_sha256": config["tree_sha256"],
    "workspace_root": workspace_container,
}
inner_payload = json.dumps(inner, sort_keys=True, separators=(",", ":")).encode()
inner_encoded = base64.urlsafe_b64encode(inner_payload).decode().rstrip("=")
command = [
    "docker", "run", "--rm", "--pull=never", "--network=none", "--read-only",
    "--tmpfs", "/tmp:rw,noexec,nosuid,size=64m", "--cap-drop", "ALL", "--gpus", "all",
    "--user", str(os.getuid()) + ":" + str(os.getgid()),
    "--mount", "type=bind,src=" + workspace_host + ",dst=" + workspace_container + ",rw",
    "--mount", "type=bind,src=" + artifact_host + ",dst=" + artifact_container + ",rw",
    config["image"], "python3", "-I", "-B",
    source_root + "/scripts/remote/vast_tiny_smoke.py", "--config", inner_encoded,
]
raise SystemExit(subprocess.run(command, check=False, shell=False).returncode)
"""


@dataclass(frozen=True)
class StageRequest:
    """An exact code/config snapshot to install under one immutable run identity."""

    run_id: str
    archive: bytes
    git_revision: str
    config_sha256: str


@dataclass(frozen=True)
class SnapshotIdentity:
    """Hashes required to bind a smoke run to a previously staged snapshot."""

    archive_sha256: str
    tree_sha256: str
    git_revision: str
    config_sha256: str


@dataclass(frozen=True)
class SmokeRequest:
    """A bounded tiny smoke tied to an immutable stage."""

    run_id: str
    snapshot: SnapshotIdentity
    smoke_id: str = "tiny-smoke-v1"


class WorkerAdapter(ABC):
    """Conceptual operations shared by every worker implementation."""

    def __init__(self, worker: WorkerSpec, artifact_store: ArtifactStore | None = None) -> None:
        self.worker = worker
        self.artifact_store = artifact_store

    @abstractmethod
    def probe(self) -> CommandPlan:
        """Plan a read-only environment and connectivity probe."""

    @abstractmethod
    def stage(self, request: StageRequest) -> CommandPlan:
        """Plan transfer and verification of an immutable snapshot."""

    @abstractmethod
    def smoke(self, request: SmokeRequest) -> CommandPlan:
        """Plan the complete tiny worker smoke contract."""

    @abstractmethod
    def launch(self) -> CommandPlan:
        """Plan a bounded detached run returning a stable identity."""

    @abstractmethod
    def status(self) -> CommandPlan:
        """Plan a read-only job status query."""

    @abstractmethod
    def sync(self) -> CommandPlan:
        """Plan selected artifact transfer and hash verification."""

    @abstractmethod
    def cancel(self) -> CommandPlan:
        """Plan graceful checkpoint-and-stop."""


class ProbeOnlySshAdapter(WorkerAdapter):
    """Safe initial adapter: probe works; mutating operations fail closed."""

    def probe(self) -> CommandPlan:
        alias = self.worker.ssh_alias
        if alias is None:
            raise AdapterError(f"worker {self.worker.name} has no ssh_alias")
        if not _SSH_ALIAS.fullmatch(alias):
            raise AdapterError(f"worker {self.worker.name} has an invalid ssh_alias")
        return CommandPlan(
            operation="probe",
            worker=self.worker.name,
            argv=("ssh", *SSH_OPTIONS, "--", alias, "python3", "-"),
            stdin=REMOTE_PROBE,
        )

    def stage(self, request: StageRequest) -> CommandPlan:
        return self._unsupported("stage")

    def smoke(self, request: SmokeRequest) -> CommandPlan:
        return self._unsupported("smoke")

    def launch(self) -> CommandPlan:
        return self._unsupported("launch")

    def status(self) -> CommandPlan:
        return self._unsupported("status")

    def sync(self) -> CommandPlan:
        return self._unsupported("sync")

    def cancel(self) -> CommandPlan:
        return self._unsupported("cancel")

    def _unsupported(self, operation: str) -> Never:
        raise UnsupportedOperation(
            f"{operation} is fail-closed until its bounded implementation and tests exist"
        )


class VastSshAdapter(ProbeOnlySshAdapter):
    """Vast container-shell adapter with bounded stage and tiny smoke plans."""

    def stage(self, request: StageRequest) -> CommandPlan:
        self._require_mutation_authorized("stage")
        _validate_run_id(request.run_id)
        _validate_git_revision(request.git_revision)
        _validate_sha256(request.config_sha256, "config_sha256")
        if not isinstance(request.archive, bytes):
            raise AdapterError("snapshot archive must be bytes")
        maximum = self._snapshot_byte_limit()
        if not request.archive or len(request.archive) > maximum:
            raise AdapterError("snapshot archive is empty or exceeds the configured storage bound")
        tree_sha256 = _inspect_snapshot(request.archive, maximum)
        archive_sha256 = hashlib.sha256(request.archive).hexdigest()
        workspace_root = _workspace_root(self.worker)
        config = {
            "archive_sha256": archive_sha256,
            "config_sha256": request.config_sha256,
            "git_revision": request.git_revision,
            "max_archive_bytes": maximum,
            "run_id": request.run_id,
            "tree_sha256": tree_sha256,
            "workspace_root": workspace_root,
        }
        command = _remote_python_command(REMOTE_STAGE, config)
        argv = ("ssh", *SSH_OPTIONS, "--", self._alias(), command)
        return CommandPlan(
            operation="stage",
            worker=self.worker.name,
            argv=argv,
            stdin=request.archive,
            timeout_seconds=180,
            redact_argv=(len(argv) - 1,),
        )

    def smoke(self, request: SmokeRequest) -> CommandPlan:
        self._require_mutation_authorized("smoke")
        _validate_run_id(request.run_id)
        _validate_run_id(request.smoke_id, field="smoke_id")
        _validate_snapshot_identity(request.snapshot)
        workspace_root = _workspace_root(self.worker)
        artifact_root = self._filesystem_artifact_root(workspace_root)
        source_root = str(PurePosixPath(workspace_root) / request.run_id / "source")
        script_path = str(PurePosixPath(source_root) / "scripts/remote/vast_tiny_smoke.py")
        cap = self.worker.resource_cap
        config = {
            "artifact_root": artifact_root,
            "archive_sha256": request.snapshot.archive_sha256,
            "config_sha256": request.snapshot.config_sha256,
            "git_revision": request.snapshot.git_revision,
            "max_steps": int(cap["max_steps"]),
            "max_storage_bytes": self._storage_cap_bytes(),
            "run_id": request.run_id,
            "smoke_id": request.smoke_id,
            "source_root": source_root,
            "tree_sha256": request.snapshot.tree_sha256,
            "workspace_root": workspace_root,
        }
        encoded = _encode_config(config)
        remote_argv = ("python3", "-I", "-B", script_path, "--config", encoded)
        command = shlex.join(remote_argv)
        argv = ("ssh", *SSH_OPTIONS, "--", self._alias(), command)
        return CommandPlan(
            operation="smoke",
            worker=self.worker.name,
            argv=argv,
            timeout_seconds=300,
            redact_argv=(len(argv) - 1,),
        )

    def _require_mutation_authorized(self, operation: str) -> None:
        missing: list[str] = []
        if not self.worker.enabled:
            missing.append(f"workers.{self.worker.name}.enabled=true")
        missing.extend(self.worker.missing_launch_fields())
        if self.artifact_store is None:
            missing.append("artifact_store(configuration unavailable)")
        else:
            missing.extend(self.artifact_store.missing_fields())
        if missing:
            raise AdapterError(f"{operation} is not authorized; missing: {', '.join(missing)}")

    def _alias(self) -> str:
        alias = self.worker.ssh_alias
        if alias is None:
            raise AdapterError(f"worker {self.worker.name} has no ssh_alias")
        if _SSH_ALIAS.fullmatch(alias) is None:
            raise AdapterError(f"worker {self.worker.name} has an invalid ssh_alias")
        return alias

    def _storage_cap_bytes(self) -> int:
        value = self.worker.resource_cap.get("max_storage_gb")
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise AdapterError("max_storage_gb is not a positive numeric cap")
        return int(value * 1_000_000_000)

    def _snapshot_byte_limit(self) -> int:
        return min(self._storage_cap_bytes(), _MAX_SNAPSHOT_BYTES)

    def _filesystem_artifact_root(self, workspace_root: str) -> str:
        store = self.artifact_store
        if store is None:  # guarded by _require_mutation_authorized
            raise AdapterError("artifact store is unavailable")
        if store.type != _FILESYSTEM_SINK:
            raise UnsupportedOperation(
                "smoke currently supports only artifact_store.type=worker-filesystem"
            )
        assert store.uri is not None  # missing_fields was checked above
        parsed = urlsplit(store.uri)
        if (
            parsed.scheme != "file"
            or parsed.netloc
            or parsed.query
            or parsed.fragment
            or unquote(parsed.path) != parsed.path
        ):
            raise AdapterError("worker-filesystem artifact URI must be file:///absolute/path")
        root = _absolute_posix_path(parsed.path, "artifact_store.uri")
        workspace = PurePosixPath(workspace_root)
        artifact = PurePosixPath(root)
        if artifact == workspace or workspace in artifact.parents or artifact in workspace.parents:
            raise AdapterError("artifact root and ephemeral worker workspace must be disjoint")
        return root


class OwnedDockerAdapter(VastSshAdapter):
    """Owned-host adapter that runs the tiny smoke inside a GPU Docker container."""

    def smoke(self, request: SmokeRequest) -> CommandPlan:
        self._require_mutation_authorized("smoke")
        if self.worker.execution != "docker":
            raise AdapterError("owned worker smoke requires execution=docker")
        _validate_run_id(request.run_id)
        _validate_run_id(request.smoke_id, field="smoke_id")
        _validate_snapshot_identity(request.snapshot)
        workspace_root = _workspace_root(self.worker)
        artifact_root = self._filesystem_artifact_root(workspace_root)
        image = _container_image(self.worker.image)
        cap = self.worker.resource_cap
        config = {
            "artifact_root": artifact_root,
            "archive_sha256": request.snapshot.archive_sha256,
            "config_sha256": request.snapshot.config_sha256,
            "git_revision": request.snapshot.git_revision,
            "image": image,
            "max_steps": int(cap["max_steps"]),
            "max_storage_bytes": self._storage_cap_bytes(),
            "run_id": request.run_id,
            "smoke_id": request.smoke_id,
            "tree_sha256": request.snapshot.tree_sha256,
            "workspace_root": workspace_root,
        }
        command = _remote_python_command(REMOTE_OWNED_DOCKER_SMOKE, config)
        argv = ("ssh", *SSH_OPTIONS, "--", self._alias(), command)
        return CommandPlan(
            operation="smoke",
            worker=self.worker.name,
            argv=argv,
            timeout_seconds=600,
            redact_argv=(len(argv) - 1,),
        )


def adapter_for(worker: WorkerSpec, artifact_store: ArtifactStore | None = None) -> WorkerAdapter:
    if worker.kind not in {"vast-ephemeral", "owned-persistent", "cluster"}:
        raise AdapterError(f"unsupported worker kind: {worker.kind}")
    if worker.kind == "vast-ephemeral":
        return VastSshAdapter(worker, artifact_store)
    if worker.kind == "owned-persistent":
        return OwnedDockerAdapter(worker, artifact_store)
    return ProbeOnlySshAdapter(worker, artifact_store)


def snapshot_identity(request: StageRequest) -> SnapshotIdentity:
    """Compute the identity a successful stage returns, without executing a command."""

    _validate_run_id(request.run_id)
    _validate_git_revision(request.git_revision)
    _validate_sha256(request.config_sha256, "config_sha256")
    if not isinstance(request.archive, bytes) or not request.archive:
        raise AdapterError("snapshot archive must be non-empty bytes")
    tree_sha256 = _inspect_snapshot(request.archive, _MAX_SNAPSHOT_BYTES)
    return SnapshotIdentity(
        archive_sha256=hashlib.sha256(request.archive).hexdigest(),
        tree_sha256=tree_sha256,
        git_revision=request.git_revision,
        config_sha256=request.config_sha256,
    )


def _validate_snapshot_identity(identity: SnapshotIdentity) -> None:
    _validate_sha256(identity.archive_sha256, "archive_sha256")
    _validate_sha256(identity.tree_sha256, "tree_sha256")
    _validate_git_revision(identity.git_revision)
    _validate_sha256(identity.config_sha256, "config_sha256")


def _validate_run_id(value: str, *, field: str = "run_id") -> None:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise AdapterError(f"{field} must match {_RUN_ID.pattern!r}")


def _validate_git_revision(value: str) -> None:
    if not isinstance(value, str) or _GIT_REVISION.fullmatch(value) is None:
        raise AdapterError("git_revision must be a 7-64 character lowercase hex revision")


def _validate_sha256(value: str, field: str) -> None:
    if not isinstance(value, str) or _HEX_SHA256.fullmatch(value) is None:
        raise AdapterError(f"{field} must be a lowercase SHA-256 hex digest")


def _workspace_root(worker: WorkerSpec) -> str:
    if worker.workspace_root is None:
        raise AdapterError(f"worker {worker.name} has no workspace_root")
    return _absolute_posix_path(worker.workspace_root, f"workers.{worker.name}.workspace_root")


def _absolute_posix_path(value: str, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise AdapterError(f"{field} must be a safe absolute POSIX path")
    path = PurePosixPath(value)
    if not path.is_absolute() or ".." in path.parts or path.as_posix() != value.rstrip("/"):
        raise AdapterError(f"{field} must be a normalized absolute POSIX path")
    return path.as_posix()


def _container_image(value: str | None) -> str:
    if not isinstance(value, str) or _CONTAINER_IMAGE.fullmatch(value) is None:
        raise AdapterError("worker image must be a safe container image reference")
    return value


def _remote_python_command(program: str, config: dict[str, object]) -> str:
    return shlex.join(("python3", "-I", "-c", program, _encode_config(config)))


def _encode_config(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")


def _snapshot_member_name(raw: str) -> str:
    if not raw or "\x00" in raw or "\\" in raw:
        raise AdapterError("snapshot contains an invalid member name")
    name = raw.rstrip("/")
    path = PurePosixPath(name)
    if not name or path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise AdapterError("snapshot member escapes its root")
    if name == _STAGE_MANIFEST:
        raise AdapterError("snapshot contains the reserved stage manifest")
    return path.as_posix()


def _inspect_snapshot(payload: bytes, maximum: int) -> str:
    if len(payload) > maximum:
        raise AdapterError("snapshot archive exceeds the configured bound")
    try:
        archive = tarfile.open(fileobj=io.BytesIO(payload), mode="r:*")
    except (tarfile.TarError, OSError) as exc:
        raise AdapterError(f"invalid snapshot archive: {type(exc).__name__}") from exc
    records: list[tuple[str, int, bytes]] = []
    seen: set[str] = set()
    total = 0
    with archive:
        for member_count, member in enumerate(archive, start=1):
            if member_count > _MAX_SNAPSHOT_MEMBERS:
                raise AdapterError("snapshot contains too many members")
            name = _snapshot_member_name(member.name)
            if name in seen:
                raise AdapterError("snapshot contains duplicate member paths")
            seen.add(name)
            if not (member.isdir() or member.isreg()):
                raise AdapterError("snapshot contains links or special files")
            if member.isdir():
                continue
            total += member.size
            if total > maximum:
                raise AdapterError("expanded snapshot exceeds the configured bound")
            source = archive.extractfile(member)
            if source is None:
                raise AdapterError("snapshot member could not be read")
            content_digest = hashlib.sha256()
            read_bytes = 0
            while chunk := source.read(1024 * 1024):
                read_bytes += len(chunk)
                content_digest.update(chunk)
            if read_bytes != member.size:
                raise AdapterError("snapshot member size changed while reading")
            records.append((name, member.size, content_digest.digest()))
    if not records:
        raise AdapterError("snapshot contains no regular files")
    regular = {item[0] for item in records}
    for name in seen:
        for parent in PurePosixPath(name).parents:
            parent_name = parent.as_posix()
            if parent_name == ".":
                break
            if parent_name in regular:
                raise AdapterError("a regular file is the parent of another snapshot member")
    tree = hashlib.sha256(b"mojidiff-tree-v1\0")
    for name, size, digest_bytes in sorted(records):
        tree.update(name.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(size).encode("ascii"))
        tree.update(b"\0")
        tree.update(digest_bytes)
    return tree.hexdigest()
