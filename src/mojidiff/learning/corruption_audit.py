"""Bounded reproducible audits for structured corruption support."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import yaml

from mojidiff.learning.geometry import whole_path_pool_support
from mojidiff.learning.tiny_study import (
    TinyLearningError,
    _codec,
    _file_sha256,
    _load_program,
    _mapping,
    _sequence,
    _string,
    load_tiny_learning_config,
)
from mojidiff.representation.codec_study import _palette, _write_bytes_artifact


class CorruptionAuditError(RuntimeError):
    """A corruption-support audit is malformed or cannot reproduce its input."""


@dataclass(frozen=True)
class WholePathAuditConfig:
    version: str
    learning_config: Path
    learning_config_sha256: str
    report_root: Path
    target_marginal_probability: float


def load_whole_path_audit_config(path: Path) -> WholePathAuditConfig:
    try:
        root = _mapping(yaml.safe_load(path.read_bytes()), "root")
    except yaml.YAMLError as error:
        raise CorruptionAuditError("invalid whole-path audit YAML") from error
    if root.get("schema_version") != 1:
        raise CorruptionAuditError("whole-path audit schema_version must be 1")
    target = root.get("target_marginal_probability")
    if isinstance(target, bool) or not isinstance(target, (int, float)) or not 0 < target <= 1:
        raise CorruptionAuditError("target_marginal_probability must be within (0, 1]")
    expected = _string(root, "learning_config_sha256")
    if len(expected) != 64 or any(character not in "0123456789abcdef" for character in expected):
        raise CorruptionAuditError("learning_config_sha256 must be a lowercase SHA-256")
    return WholePathAuditConfig(
        version=_string(root, "study_version"),
        learning_config=Path(_string(root, "learning_config")),
        learning_config_sha256=expected,
        report_root=Path(_string(root, "report_root")),
        target_marginal_probability=float(target),
    )


def run_whole_path_audit(config: WholePathAuditConfig) -> dict[str, Any]:
    if _file_sha256(config.learning_config) != config.learning_config_sha256:
        raise CorruptionAuditError("learning config hash mismatch")
    learning = load_tiny_learning_config(config.learning_config)
    fixture = _mapping(json.loads(learning.fixture.read_bytes()), "fixture")
    rows = [_mapping(row, "fixture row") for row in _sequence(fixture.get("rows"), "rows")]
    style = _mapping(json.loads(learning.style_summary.read_bytes()), "style summary")
    candidate = _mapping(
        _mapping(style.get("width_vocabulary_analytics"), "width analytics").get(
            learning.style_candidate
        ),
        "style candidate",
    )
    widths = tuple(float(value) for value in _sequence(candidate.get("values"), "widths"))
    codec = _codec(learning, _palette(learning.palette_path), widths)
    try:
        programs = [_load_program(row, learning, codec) for row in rows]
    except TinyLearningError as error:
        raise CorruptionAuditError("cannot load pinned whole-path fixture") from error
    support = whole_path_pool_support(programs)
    field_fraction = support["eligible_geometry_fields"] / support["geometry_fields"]
    path_fraction = support["eligible_paths"] / support["active_paths"]
    result = {
        "schema_version": 1,
        "study_version": config.version,
        "learning_config": str(config.learning_config),
        "learning_config_sha256": config.learning_config_sha256,
        "fixture": str(learning.fixture),
        "fixture_sha256": learning.fixture_sha256,
        "icons": len(programs),
        "support": support,
        "eligible_path_fraction": path_fraction,
        "eligible_geometry_field_fraction": field_fraction,
        "target_marginal_probability": config.target_marginal_probability,
        "maximum_expected_changed_field_fraction": field_fraction,
        "can_match_target_marginal_probability": (
            field_fraction >= config.target_marginal_probability
        ),
    }
    payload = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode()
    _write_bytes_artifact(config.report_root / "summary.json", payload)
    _write_bytes_artifact(config.report_root / "README.md", _markdown(result).encode())
    return result


def _markdown(result: dict[str, Any]) -> str:
    support = cast(dict[str, int], result["support"])
    decision = (
        "can support the requested marginal probability"
        if result["can_match_target_marginal_probability"]
        else "cannot support the requested marginal probability"
    )
    return "\n".join(
        [
            "# Whole-path donor support audit",
            "",
            f"Pinned fixture: {result['icons']} icons; {support['active_paths']} active paths.",
            "",
            f"Compatible external donors cover {support['eligible_paths']} paths "
            f"({result['eligible_path_fraction']:.2%}) and "
            f"{support['eligible_geometry_fields']} of {support['geometry_fields']} legal "
            f"geometry fields ({result['eligible_geometry_field_fraction']:.2%}).",
            "",
            f"At a path gate of 1.0 the operator {decision} "
            f"{result['target_marginal_probability']:.2%}; its maximum expected changed-field "
            f"fraction is {result['maximum_expected_changed_field_fraction']:.2%}.",
            "",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mojidiff-whole-path-audit")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_whole_path_audit(load_whole_path_audit_config(args.config))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
