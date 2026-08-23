"""Common worker interface and safe read-only SSH probe implementation."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Never

from mojidiff.orchestration.config import WorkerSpec
from mojidiff.orchestration.plans import CommandPlan


class AdapterError(RuntimeError):
    """A worker operation cannot be planned safely."""


class UnsupportedOperation(AdapterError):
    """The adapter contract includes an operation not implemented yet."""


_SSH_ALIAS = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")

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


class WorkerAdapter(ABC):
    """Conceptual operations shared by every worker implementation."""

    def __init__(self, worker: WorkerSpec) -> None:
        self.worker = worker

    @abstractmethod
    def probe(self) -> CommandPlan:
        """Plan a read-only environment and connectivity probe."""

    @abstractmethod
    def stage(self) -> CommandPlan:
        """Plan transfer and verification of an immutable snapshot."""

    @abstractmethod
    def smoke(self) -> CommandPlan:
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

    def stage(self) -> CommandPlan:
        return self._unsupported("stage")

    def smoke(self) -> CommandPlan:
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


def adapter_for(worker: WorkerSpec) -> WorkerAdapter:
    if worker.kind not in {"vast-ephemeral", "owned-persistent", "cluster"}:
        raise AdapterError(f"unsupported worker kind: {worker.kind}")
    return ProbeOnlySshAdapter(worker)
