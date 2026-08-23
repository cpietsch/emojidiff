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


Runner = Callable[..., subprocess.CompletedProcess[Any]]


@dataclass(frozen=True)
class CommandPlan:
    operation: str
    worker: str
    argv: tuple[str, ...]
    stdin: str | bytes | None = None
    timeout_seconds: int = 45
    redact_argv: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if not self.argv or any(
            not isinstance(value, str) or not value or "\x00" in value for value in self.argv
        ):
            raise ValueError("argv must contain non-empty NUL-free strings")
        if isinstance(self.timeout_seconds, bool) or self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if len(set(self.redact_argv)) != len(self.redact_argv) or any(
            index < 0 or index >= len(self.argv) for index in self.redact_argv
        ):
            raise ValueError("redact_argv contains an invalid index")

    def safe_description(self) -> dict[str, Any]:
        """Describe a plan without logging the transferred program or config values."""

        argv: list[str] = []
        redacted = set(self.redact_argv)
        for index, value in enumerate(self.argv):
            if index in redacted:
                digest = hashlib.sha256(value.encode()).hexdigest()
                argv.append(f"<redacted:{len(value.encode())} bytes:sha256:{digest}>")
            else:
                argv.append(value)
        description: dict[str, Any] = {
            "operation": self.operation,
            "worker": self.worker,
            "argv": argv,
            "timeout_seconds": self.timeout_seconds,
        }
        if self.stdin is not None:
            payload = self.stdin.encode() if isinstance(self.stdin, str) else self.stdin
            description["stdin_sha256"] = hashlib.sha256(payload).hexdigest()
            description["stdin_bytes"] = len(payload)
        return description

    def execute(self, runner: Runner = subprocess.run) -> dict[str, Any]:
        text_mode = not isinstance(self.stdin, bytes)
        completed = runner(
            list(self.argv),
            input=self.stdin,
            text=text_mode,
            capture_output=True,
            check=False,
            timeout=self.timeout_seconds,
            shell=False,
        )
        stderr = _decode_output(completed.stderr)
        if completed.returncode != 0:
            stderr_tail = stderr[-1000:].strip()
            raise CommandFailed(
                f"{self.operation} failed with exit {completed.returncode}: {stderr_tail}"
            )
        stdout = _decode_output(completed.stdout)
        try:
            result = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise CommandFailed(f"{self.operation} returned invalid JSON") from exc
        if not isinstance(result, dict):
            raise CommandFailed(f"{self.operation} returned non-object JSON")
        return result


def argv_tuple(parts: Sequence[str]) -> tuple[str, ...]:
    return tuple(parts)


def _decode_output(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        return value
    return "" if value is None else str(value)
