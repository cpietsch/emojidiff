"""Command-line access to authorization inventory and read-only probes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from mojidiff.orchestration.adapters import adapter_for
from mojidiff.orchestration.config import load_inventory
from mojidiff.orchestration.local_probe import collect_local_probe, write_probe


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mojidiff-worker")
    parser.add_argument(
        "--hosts", type=Path, default=Path("state/hosts.local.yaml"), help="authorization record"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("inventory", help="show sanitized authorization readiness")

    local = commands.add_parser("local-probe", help="probe the control plane")
    local.add_argument("--repo", type=Path, default=Path.cwd())
    local.add_argument("--output", type=Path)

    remote = commands.add_parser("probe", help="probe a configured worker over SSH")
    remote.add_argument("--worker", required=True)
    remote.add_argument("--dry-run", action="store_true")
    return parser


def _inventory(args: argparse.Namespace) -> dict[str, Any]:
    inventory = load_inventory(args.hosts)
    return {
        "orchestrator": inventory.orchestrator_name,
        "artifact_store_missing": list(inventory.artifact_store.missing_fields()),
        "workers": {
            name: {
                "enabled": worker.enabled,
                "compute_authorized": worker.compute_authorized,
                "missing_launch_fields": list(worker.missing_launch_fields()),
            }
            for name, worker in inventory.workers.items()
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "local-probe":
        result = collect_local_probe(args.repo)
        if args.output:
            write_probe(result, args.output)
    elif args.command == "inventory":
        result = _inventory(args)
    elif args.command == "probe":
        inventory = load_inventory(args.hosts)
        worker = inventory.workers.get(args.worker)
        if worker is None:
            raise SystemExit(f"unknown worker: {args.worker}")
        plan = adapter_for(worker).probe()
        result = plan.safe_description() if args.dry_run else plan.execute()
    else:  # pragma: no cover - argparse enforces this
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
