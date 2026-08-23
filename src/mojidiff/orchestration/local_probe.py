"""Sanitized read-only probe for the persistent control plane."""

from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _command(argv: Sequence[str], timeout: int = 15) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            list(argv),
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
            shell=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"available": False, "error": type(exc).__name__}
    return {
        "available": True,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip()[:4000],
        "stderr": completed.stderr.strip()[-500:],
    }


def collect_local_probe(repo_path: Path) -> dict[str, Any]:
    workspace = Path("/home/dev/workspace")
    disk = shutil.disk_usage(workspace)
    socket_paths = (
        Path("/var/run/docker.sock"),
        Path("/run/docker.sock"),
        Path("/run/host-services/docker.sock"),
    )
    tools = {
        name: shutil.which(name)
        for name in ("docker", "git", "gh", "kubectl", "python3", "rg", "ssh", "tmux", "uv")
    }
    return {
        "schema_version": 1,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "repo_path": str(repo_path.resolve()),
        "workspace": {
            "path": str(workspace),
            "total_bytes": disk.total,
            "used_bytes": disk.used,
            "free_bytes": disk.free,
        },
        "docker": {
            "host": os.environ.get("DOCKER_HOST"),
            "tls_verify": os.environ.get("DOCKER_TLS_VERIFY") == "1",
            "context": _command(("docker", "context", "show")),
            "info": _command(
                (
                    "docker",
                    "info",
                    "--format",
                    "{{json .Name}} {{json .DockerRootDir}} {{json .Driver}}",
                )
            ),
            "host_socket_paths": {str(path): path.exists() for path in socket_paths},
        },
        "tools": tools,
        "git": {
            "status": _command(("git", "-C", str(repo_path), "status", "--short", "--branch")),
            "head": _command(("git", "-C", str(repo_path), "rev-parse", "HEAD")),
        },
    }


def write_probe(probe: dict[str, Any], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(probe, indent=2, sort_keys=True) + "\n"
    output.write_text(serialized, encoding="utf-8")
