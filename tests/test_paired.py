from __future__ import annotations

import json
from pathlib import Path

import pytest

from mojidiff.learning.paired import load_errors, paired_reduction


def _write(run: Path, errors: dict[str, float]) -> Path:
    run.mkdir()
    rows = [
        {"hexcode": h, "model_pixel_error": e, "baseline_pixel_error": 0.1}
        for h, e in errors.items()
    ]
    (run / "per_icon.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return run


def test_a_uniformly_better_candidate_excludes_zero(tmp_path: Path) -> None:
    parent = {f"{i:04X}": 0.08 + 0.001 * (i % 7) for i in range(100)}
    candidate = {h: e - 0.01 for h, e in parent.items()}
    result = paired_reduction(
        load_errors(_write(tmp_path / "c", candidate)), load_errors(_write(tmp_path / "p", parent))
    )
    assert result["icons"] == 100
    assert result["parent_minus_candidate"][0] == pytest.approx(0.01)
    assert result["candidate_better_icons"] == 100
    assert result["interval_excludes_zero_in_favour"]


def test_identical_runs_do_not_pass_and_mismatched_icons_raise(tmp_path: Path) -> None:
    errors = {"1F600": 0.05, "1F601": 0.07}
    same = paired_reduction(errors, dict(errors))
    assert same["parent_minus_candidate"] == [0.0, 0.0, 0.0]
    assert not same["interval_excludes_zero_in_favour"]
    with pytest.raises(ValueError):
        paired_reduction(errors, {"1F600": 0.05})
