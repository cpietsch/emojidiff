"""Append-only, crash-resistant run-state registry."""

from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def append_event(path: Path, event: Mapping[str, Any]) -> None:
    """Append exactly one JSON event while holding an advisory exclusive lock."""

    required = ("run_id", "state", "timestamp")
    missing = [field for field in required if not event.get(field)]
    if missing:
        raise ValueError(f"run event missing required fields: {', '.join(missing)}")
    line = json.dumps(dict(event), sort_keys=True, separators=(",", ":")) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, line.encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
