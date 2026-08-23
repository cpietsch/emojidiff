"""Command-line entry point for representation evidence probes."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mojidiff.representation.study import load_study_config, run_study


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mojidiff-representation")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_study(load_study_config(args.config))
    print(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
