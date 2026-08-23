"""Non-interactive subprocess plans with sanitized dry-run descriptions."""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any


class CommandFailed(RuntimeError):
    """A planned command failed or returned malformed output."""


Runner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class CommandPlan:
    operation: str
    worker: str
    argv: tuple[str, ...]
    stdin: str | None = None
    timeout_seconds: int = 45

    def safe_description(self) -> dict[str, Any]:
        """Describe a plan without logging the transferred program or config values."""

        description: dict[str, Any] = {
            "operation": self.operation,
            "worker": self.worker,
            "argv": list(self.argv),
            "timeout_seconds": self.timeout_seconds,
        }
        if self.stdin is not None:
            description["stdin_sha256"] = hashlib.sha256(self.stdin.encode()).hexdigest()
            description["stdin_bytes"] = len(self.stdin.encode())
        return description

    def execute(self, runner: Runner = subprocess.run) -> dict[str, Any]:
        completed = runner(
            list(self.argv),
            input=self.stdin,
            text=True,
            capture_output=True,
            check=False,
            timeout=self.timeout_seconds,
            shell=False,
        )
        if completed.returncode != 0:
            stderr_tail = completed.stderr[-1000:].strip()
            raise CommandFailed(
                f"{self.operation} failed with exit {completed.returncode}: {stderr_tail}"
            )
        try:
            result = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise CommandFailed(f"{self.operation} returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise CommandFailed(f"{self.operation} returned non-object JSON")
        return result


def argv_tuple(parts: Sequence[str]) -> tuple[str, ...]:
    return tuple(parts)
