from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from mojidiff.representation.corpus_audit import (
    CorpusAuditError,
    _capacity_loss,
    load_corpus_audit_config,
    run_corpus_audit,
)


def test_versioned_full_primary_config_pins_inputs_and_capacity_grid() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_corpus_audit_config(root / "configs/codec/full-primary-structure-v1.yaml")

    assert config.expected_manifest_rows == 4006
    assert config.path_slots == (32, 48, 64, 96, 128)
    assert config.segments_per_path == (64, 128, 192, 256, 384, 512)
    assert len(config.primary_manifest_sha256) == 64
    assert len(config.palette_sha256) == 64


def test_capacity_loss_separates_path_and_retained_contour_segment_loss() -> None:
    row: dict[str, Any] = {
        "contour_segment_lengths": [5, 2, 7],
        "contour_layers": [1, 1, 2],
    }

    loss = _capacity_loss(row, paths=2, segments=3)

    assert loss["dropped_contours"] == 1
    assert loss["segments_dropped_by_path_budget"] == 7
    assert loss["segments_dropped_by_segment_budget"] == 2
    assert loss["dropped_segments"] == 9
    assert loss["damaged_layers"] == 2
    assert loss["partial_layers"] == 1
    assert loss["fully_dropped_layers"] == 1


def test_tiny_census_emits_two_attempts_and_one_hybrid_per_manifest_row(
    tmp_path: Path,
) -> None:
    raw_root = tmp_path / "raw"
    (raw_root / "color/svg").mkdir(parents=True)
    valid = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">
    <path fill="#ffffff" d="M1 1L10 1L10 10Z M20 20L30 20L30 30Z"/></svg>"""
    fallback = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">
    <path fill="#ffffff" paint-order="stroke" d="M1 1L10 1L10 10Z"/></svg>"""
    sources = {"color/svg/A.svg": valid, "color/svg/B.svg": fallback}
    for relative, source in sources.items():
        (raw_root / relative).write_bytes(source)

    palette_path = tmp_path / "palette.json"
    palette_path.write_text(
        json.dumps({"colors": ["#ffffff", "#000000"], "skintones": {}}),
        encoding="utf-8",
    )
    revision = "a" * 40
    manifest_rows = [
        {
            "source_revision": revision,
            "source_path": relative,
            "source_svg_sha256": hashlib.sha256(source).hexdigest(),
            "hexcode": name,
            "group": "symbols",
            "subgroup": "test",
            "variant_family_id": name,
            "curation_status": "include",
            "split": "primary/train",
            "split_family_cluster": name,
        }
        for name, (relative, source) in zip(("A", "B"), sources.items(), strict=True)
    ]
    manifest_path = tmp_path / "primary.parquet"
    pq.write_table(pa.Table.from_pylist(manifest_rows), manifest_path)
    report_root = tmp_path / "report"
    config_document: dict[str, Any] = {
        "schema_version": 1,
        "audit_version": "tiny-structure-v1",
        "source_revision": revision,
        "raw_root": str(raw_root),
        "primary_manifest": str(manifest_path),
        "primary_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "expected_manifest_rows": 2,
        "palette_path": str(palette_path),
        "palette_sha256": hashlib.sha256(palette_path.read_bytes()).hexdigest(),
        "report_root": str(report_root),
        "capacity": {"path_slots": [1, 2], "segments_per_path": [1, 4]},
        "style_vocabulary": {
            "stroke_widths": [2.0],
            "miter_limits": [4.0],
            "dash_patterns": [[2.0, 4.0]],
        },
        "limits": {
            "max_manifest_bytes": 1_000_000,
            "max_manifest_rows": 10,
            "max_source_bytes": 10_000,
            "max_outlined_bytes": 100_000,
            "max_error_chars": 200,
            "tail_count": 3,
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config_document, sort_keys=False), encoding="utf-8")
    config = load_corpus_audit_config(config_path)

    first = run_corpus_audit(config, config_path)
    attempts = [
        json.loads(line) for line in (report_root / "attempts.jsonl").read_text().splitlines()
    ]
    hybrids = [json.loads(line) for line in (report_root / "hybrid.jsonl").read_text().splitlines()]
    second = run_corpus_audit(config, config_path)

    assert len(attempts) == 4
    assert len(hybrids) == 2
    assert [(row["source_path"], row["representation"]) for row in attempts] == [
        ("color/svg/A.svg", "semantic"),
        ("color/svg/A.svg", "outlined"),
        ("color/svg/B.svg", "semantic"),
        ("color/svg/B.svg", "outlined"),
    ]
    assert attempts[0]["contour_segment_lengths"] == [3, 3]
    assert attempts[2]["error_stage"] == "normalize"
    assert attempts[2]["error_code"] == "unsupported_presentation"
    assert hybrids[1]["route"] == "outlined_fallback:unsupported_presentation"
    assert first["attempts_sha256"] == second["attempts_sha256"]
    assert first["hybrid_sha256"] == second["hybrid_sha256"]
    assert (
        first["representations"]["semantic"]["structure"]["capacity"]["p1-s1"][
            "segments_dropped_by_path_budget"
        ]
        == 3
    )
    assert (
        first["representations"]["semantic"]["structure"]["capacity"]["p1-s1"][
            "fully_dropped_layers"
        ]
        == 0
    )
    attempts_path = report_root / "attempts.jsonl"
    attempts_path.write_text("pre-existing operator evidence\n", encoding="utf-8")
    original = attempts_path.read_bytes()
    with pytest.raises(CorpusAuditError, match="refusing to replace differing artifact"):
        run_corpus_audit(config, config_path)
    assert attempts_path.read_bytes() == original
