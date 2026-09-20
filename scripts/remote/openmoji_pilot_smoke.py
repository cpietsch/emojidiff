#!/usr/bin/env python3
"""Run the exact bounded Gate G OpenMoji pilot inside an owned GPU container."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(_SOURCE_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import vast_tiny_smoke as base  # noqa: E402

from mojidiff.learning.openmoji_pilot import (  # noqa: E402
    OpenMojiPilotConfig,
    load_openmoji_pilot_config,
    run_openmoji_pilot,
)

_SMOKE_ID = "openmoji-g1-pipeline-v1"
_CONFIG_RELATIVE = Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
_MIN_FREE_OUTPUT_BYTES = 256 * 1024 * 1024


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _run(config: dict[str, Any]) -> dict[str, Any]:
    if config["smoke_id"] != _SMOKE_ID:
        raise base.SmokeError("unexpected OpenMoji pipeline smoke id")
    source = base._path(config["source_root"], "source_root")
    artifact = base._path(config["artifact_root"], "artifact_root")
    stage, staged_bytes = base._verify_stage(source, config)
    config_path = source / _CONFIG_RELATIVE
    if hashlib.sha256(config_path.read_bytes()).hexdigest() != config["config_sha256"]:
        raise base.SmokeError("pipeline config hash does not match staged identity")
    pilot = load_openmoji_pilot_config(config_path)
    if pilot.steps > base._positive_int(config["max_steps"], "max_steps"):
        raise base.SmokeError("pipeline smoke exceeds the configured step cap")
    storage_cap = base._positive_int(config["max_storage_bytes"], "max_storage_bytes")
    existing = staged_bytes + base._directory_storage_bytes(artifact / config["run_id"])
    if existing + _MIN_FREE_OUTPUT_BYTES > storage_cap:
        raise base.SmokeError("pipeline smoke lacks bounded artifact capacity")

    output = artifact / config["run_id"] / "smoke" / _SMOKE_ID
    if output.is_symlink():
        raise base.SmokeError("pipeline smoke output root is a symbolic link")
    summary = run_openmoji_pilot_with_roots(pilot, config_path, output)
    if not str(summary["device"]).startswith("cuda"):
        raise base.SmokeError("pipeline smoke did not execute on CUDA")
    final_size = staged_bytes + base._directory_storage_bytes(artifact / config["run_id"])
    if final_size > storage_cap:
        raise base.SmokeError("pipeline smoke exceeded the configured storage cap")
    return {
        "artifact_root": str(output),
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "device": summary["device"],
        "locked_path_exact": summary["locked_path_exact"],
        "operation": "pipeline-smoke",
        "run_id": config["run_id"],
        "schema_version": 1,
        "smoke_id": _SMOKE_ID,
        "stage": stage,
        "summary_sha256": hashlib.sha256((output / "summary.json").read_bytes()).hexdigest(),
    }


def run_openmoji_pilot_with_roots(
    pilot: OpenMojiPilotConfig,
    config_path: Path,
    output: Path,
) -> dict[str, Any]:
    configured = replace(
        pilot,
        report_root=output / "report",
        checkpoint_root=output / "checkpoint",
    )
    return run_openmoji_pilot(configured, config_path)


def main() -> int:
    try:
        config = base._validated_config(_parse_args().config)
        result = _run(config)
    except (base.SmokeError, OSError, ValueError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
