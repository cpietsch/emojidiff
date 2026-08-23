from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from mojidiff.curation.audit import AuditConfig, AuditError, _safe_parse, audit_one
from mojidiff.curation.decisions import (
    DecisionPaths,
    apply_decisions,
    apply_review,
    candidate_set_sha256,
    distribution_summary,
)
from mojidiff.curation.source import OpenMojiSource

SIMPLE_SVG = b"""<svg viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg">
<circle cx="36" cy="36" r="20" fill="#fcea2b" stroke="#000" stroke-width="2"/>
</svg>"""


def _config(tmp_path: Path) -> AuditConfig:
    source = OpenMojiSource(
        name="fixture",
        url="https://github.com/example/fixture.git",
        tag="1.0.0",
        revision="a" * 40,
        license="CC-BY-SA-4.0",
        license_path="LICENSE",
        metadata_path="metadata.csv",
        palette_path="palette.json",
        svg_glob="color/svg/*.svg",
        raw_root=tmp_path / "raw",
        source_manifest=tmp_path / "source.json",
    )
    return AuditConfig(
        version="fixture-v1",
        source=source,
        output_root=tmp_path / "audit",
        sizes=(72, 18),
        timeout_seconds=2,
        max_svg_bytes=100_000,
        max_elements=100,
        decisions={},
    )


def test_svg_audit_renders_both_sizes(tmp_path: Path) -> None:
    config = _config(tmp_path)
    svg = config.source.raw_root / "color/svg/1F600.svg"
    svg.parent.mkdir(parents=True)
    svg.write_bytes(SIMPLE_SVG)

    row = audit_one(
        svg,
        {"hexcode": "1F600", "group": "smileys-emotion"},
        config,
        config.output_root / "renders",
    )

    assert row["parse_ok"] is True
    assert row["render_72_ok"] is True
    assert row["render_18_ok"] is True
    assert row["alpha_pixel_count_72"] > row["alpha_pixel_count_18"] > 0
    assert row["geometry_outside_sample_count"] == 0
    assert (config.output_root / "renders/72/1F600.png").is_file()


def test_svg_audit_rejects_external_reference(tmp_path: Path) -> None:
    config = _config(tmp_path)
    unsafe = b"""<svg xmlns="http://www.w3.org/2000/svg"><image href="https://example.com/x.png"/></svg>"""

    with pytest.raises(AuditError, match="UNSAFE_TAG"):
        _safe_parse(unsafe, config)


def test_svg_audit_allows_internal_fragment_reference(tmp_path: Path) -> None:
    config = _config(tmp_path)
    internal = b"""<svg xmlns="http://www.w3.org/2000/svg">
    <defs><clipPath id="safe"><circle cx="1" cy="1" r="1"/></clipPath></defs>
    <g clip-path="url(#safe)"><path d="M0 0L1 1"/></g></svg>"""

    root, elements = _safe_parse(internal, config)

    assert root is not None
    assert len(elements) == 6


def test_svg_audit_rejects_external_css_url(tmp_path: Path) -> None:
    config = _config(tmp_path)
    unsafe = b"""<svg xmlns="http://www.w3.org/2000/svg">
    <path fill="url(https://example.com/colors.svg#x)" d="M0 0L1 1"/></svg>"""

    with pytest.raises(AuditError, match="EXTERNAL_OR_EMBEDDED_RESOURCE"):
        _safe_parse(unsafe, config)


def test_flags_are_reversible_policy_split(tmp_path: Path) -> None:
    paths = DecisionPaths(
        output_root=tmp_path,
        report_root=tmp_path,
        manifest_prefix=tmp_path / "manifest",
        thresholds={"threshold_version": "observe-only"},
        review_manifest=None,
    )
    row = {
        "hexcode": "1F1E6-1F1E8",
        "group": "flags",
        "parse_ok": True,
        "render_72_ok": True,
        "render_18_ok": True,
        "alpha_pixel_count_72": 100,
        "alpha_pixel_count_18": 10,
        "geometry_sample_count": 2,
        "geometry_inside_sample_count": 2,
        "geometry_estimate_has_unapplied_transforms": False,
        "variant_family_id": "flag-family",
        "exact_raster_cluster": None,
    }

    decision = apply_decisions(paths, [row])[0]

    assert decision["curation_status"] == "exclude_policy"
    assert decision["split"] == "excluded/flags"
    assert decision["reason_codes"] == ["SCOPE_FLAGS"]


def test_exact_duplicates_keep_one_canonical_alias(tmp_path: Path) -> None:
    paths = DecisionPaths(
        output_root=tmp_path,
        report_root=tmp_path,
        manifest_prefix=tmp_path / "manifest",
        thresholds={"threshold_version": "observe-only"},
        review_manifest=None,
    )
    base = {
        "group": "objects",
        "parse_ok": True,
        "render_72_ok": True,
        "render_18_ok": True,
        "alpha_pixel_count_72": 100,
        "alpha_pixel_count_18": 10,
        "geometry_sample_count": 2,
        "geometry_inside_sample_count": 2,
        "geometry_estimate_has_unapplied_transforms": False,
        "exact_raster_cluster": "same-render",
    }
    rows = [
        {**base, "hexcode": "A", "variant_family_id": "family-a"},
        {**base, "hexcode": "B", "variant_family_id": "family-b"},
    ]

    decisions = apply_decisions(paths, rows)

    assert [row["curation_status"] for row in decisions] == ["include", "exclude_policy"]
    assert decisions[0]["split"].startswith("primary/")
    assert decisions[1]["split"] == "excluded/exact-duplicates"
    assert decisions[1]["reason_codes"] == ["EXACT_DUPLICATE_ALIAS"]


def test_distribution_summary_preserves_raw_extremes() -> None:
    rows = [
        {
            "parse_ok": True,
            "render_72_ok": True,
            "render_18_ok": True,
            "group": "a",
            "ink_fraction_72": value,
        }
        for value in (0.01, 0.2, 0.9)
    ]

    summary = distribution_summary(rows)

    assert summary["metrics"]["ink_fraction_72"]["min"] == 0.01
    assert summary["metrics"]["ink_fraction_72"]["max"] == 0.9


def test_review_is_anchored_to_exact_candidate_set(tmp_path: Path) -> None:
    rows = [
        {
            "hexcode": "A",
            "curation_status": "quarantine_auto",
            "reason_codes": ["LOW_INK"],
            "variant_family_id": "family-a",
            "exact_raster_cluster": None,
        }
    ]
    review = tmp_path / "review.yaml"
    base = tmp_path / "base.parquet"
    base.write_bytes(b"fixture base")
    sheets = tmp_path / "sheets"
    sheets.mkdir()
    sheet = sheets / "fixture.png"
    sheet.write_bytes(b"fixture sheet")
    review.write_text(
        "\n".join(
            (
                "schema_version: 1",
                f"base_decisions_path: {base}",
                f"base_decisions_sha256: {hashlib.sha256(base.read_bytes()).hexdigest()}",
                "candidate_count: 1",
                f"candidate_set_sha256: {candidate_set_sha256(rows)}",
                "reviewer: fixture-reviewer",
                "decision: include_override",
                "note: visibly intentional fixture",
                f"contact_sheet_root: {sheets}",
                "contact_sheets_sha256:",
                f"  fixture.png: {hashlib.sha256(sheet.read_bytes()).hexdigest()}",
            )
        ),
        encoding="utf-8",
    )

    reviewed = apply_review(rows, review)

    assert reviewed[0]["curation_status"] == "include_override"
    assert reviewed[0]["split"].startswith("primary/")

    rows[0]["curation_status"] = "quarantine_auto"
    rows[0]["reason_codes"] = ["THIN_BBOX"]
    with pytest.raises(ValueError, match="candidate set"):
        apply_review(rows, review)
