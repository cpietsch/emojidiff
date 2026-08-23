"""Reproducible OpenMoji source, audit, and reversible-decision commands."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from mojidiff.curation.audit import load_audit_config, run_audit
from mojidiff.curation.decisions import (
    apply_decisions,
    apply_review,
    load_decision_paths,
    read_rows,
    write_contact_sheets,
    write_distribution_report,
    write_manifests,
)
from mojidiff.curation.source import acquire, load_source_config
from mojidiff.curation.validate import validate_curation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mojidiff-curate")
    commands = parser.add_subparsers(dest="command", required=True)

    source = commands.add_parser("acquire", help="acquire and freeze the pinned raw source")
    source.add_argument("--config", type=Path, required=True)

    audit = commands.add_parser("audit", help="compute raw metrics and deterministic renders")
    audit.add_argument("--config", type=Path, required=True)

    curate = commands.add_parser("curate", help="report distributions and apply reversible rules")
    curate.add_argument("--config", type=Path, required=True)

    verify = commands.add_parser("verify", help="verify raw and reviewed curation invariants")
    verify.add_argument("--source-config", type=Path, required=True)
    verify.add_argument("--decisions", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "acquire":
        result = acquire(load_source_config(args.config))
        output = {
            "source": result["source"],
            "revision": result["revision"],
            "svg_count": result["svg_count"],
            "source_manifest": str(load_source_config(args.config).source_manifest),
        }
    elif args.command == "audit":
        config = load_audit_config(args.config)
        rows = run_audit(config)
        output = {
            "audit_version": config.version,
            "rows": len(rows),
            "output_root": str(config.output_root),
        }
    elif args.command == "curate":
        paths = load_decision_paths(args.config)
        rows = read_rows(paths.output_root / "audit.parquet")
        distributions = write_distribution_report(paths, rows)
        decisions = apply_decisions(paths, rows)
        if paths.review_manifest is not None:
            decisions = apply_review(decisions, paths.review_manifest)
        summary = write_manifests(paths, decisions)
        sheets = write_contact_sheets(paths, decisions)
        output = {
            "rows": distributions["row_count"],
            "summary": summary,
            "contact_sheets": [str(path) for path in sheets],
        }
    elif args.command == "verify":
        output = validate_curation(
            load_source_config(args.source_config), args.decisions, args.output
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
