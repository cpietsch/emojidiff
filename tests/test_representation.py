from __future__ import annotations

from pathlib import Path

import numpy as np

from mojidiff.representation.study import (
    StudyConfig,
    _program_counts,
    _similarity,
    probe_one,
    select_fixture,
)


def _config(tmp_path: Path) -> StudyConfig:
    return StudyConfig(
        version="fixture-probe",
        source_revision="a" * 40,
        raw_root=tmp_path / "raw",
        primary_manifest=tmp_path / "primary.parquet",
        fixture_manifest=tmp_path / "fixture.json",
        derived_root=tmp_path / "derived",
        report_root=tmp_path / "report",
        selection_version="fixture-selection-v1",
        per_group=2,
        path_budgets=(1, 2),
        segment_budgets=(4, 8),
        render_sizes=(72, 18),
        render_timeout_seconds=2,
    )


def test_program_counts_source_primitives() -> None:
    svg = b"""<svg xmlns="http://www.w3.org/2000/svg">
    <circle cx="5" cy="5" r="2"/><line x1="0" y1="0" x2="1" y2="1"/>
    <path d="M0 0 L1 1 Z"/></svg>"""

    paths, segments = _program_counts(svg)

    assert paths == 3
    assert segments == [5, 1, 2]


def test_fixture_selection_is_group_stratified_and_deterministic(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rows = [
        {
            "group": group,
            "source_path": f"color/svg/{group}-{index}.svg",
            "segment_count": index,
            "ink_fraction_72": index / 10,
        }
        for group in ("a", "b")
        for index in range(5)
    ]

    first = select_fixture(rows, config)
    second = select_fixture(list(reversed(rows)), config)

    assert [row["source_path"] for row in first] == [row["source_path"] for row in second]
    assert {group: sum(row["group"] == group for row in first) for group in ("a", "b")} == {
        "a": 2,
        "b": 2,
    }


def test_outlined_probe_preserves_simple_render(tmp_path: Path) -> None:
    config = _config(tmp_path)
    source = config.raw_root / "color/svg/A.svg"
    source.parent.mkdir(parents=True)
    source.write_text(
        """<svg viewBox="0 0 72 72" xmlns="http://www.w3.org/2000/svg">
        <circle cx="36" cy="36" r="20" fill="#fcea2b" stroke="#000" stroke-width="2"/>
        </svg>""",
        encoding="utf-8",
    )
    row = {
        "hexcode": "A",
        "annotation": "fixture",
        "group": "objects",
        "source_path": "color/svg/A.svg",
        "source_svg_sha256": "fixture",
        "transform_count": 0,
        "stroke_element_count": 1,
    }

    result = probe_one(row, config)

    assert result["outline_ok"] is True
    assert result["outline_stroke_element_count"] == 0
    assert result["alpha_iou_72"] > 0.99
    assert result["rgba_mae_72"] < 0.01


def test_similarity_is_exact_for_identical_rasters() -> None:
    raster = np.zeros((4, 4, 4), dtype=np.uint8)

    metrics = _similarity(raster, raster.copy())

    assert metrics["rgba_mae"] == 0
    assert metrics["pixel_exact_fraction"] == 1
    assert metrics["alpha_iou"] == 1
    assert metrics["psnr_db"] is None
