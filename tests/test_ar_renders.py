"""The sample sheet must run end to end before a GPU is waiting on it.

The latency diagnostic taught this lesson expensively: a path that only runs after
training, and that no test exercises, fails after the expensive part is done. This
builds a small model, writes a checkpoint, and drives the whole module - load, sample,
validate, render, compose the sheet, write the artifacts - on the CPU in a few seconds.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml

from mojidiff.learning.ar_renders import load_ar_render_config, run_ar_renders
from mojidiff.learning.autoregressive import CausalProgramModel, SequenceLayout
from mojidiff.learning.openmoji_pilot import (
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.tiny_study import _save_checkpoint

_PILOT = Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")


def test_the_sample_sheet_renders_end_to_end(tmp_path: Path) -> None:
    pilot = load_openmoji_pilot_config(_PILOT)
    codec = _selected_codec(pilot)
    _, groups, subgroups = load_pilot_index(pilot)
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)
    shape = {"d_model": 32, "heads": 4, "layers": 2, "feedforward": 64}
    torch.manual_seed(5)
    model = CausalProgramModel(
        layout,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
        **shape,
    )
    payload = _save_checkpoint(model, torch.optim.AdamW(model.parameters(), lr=1e-3), 1)
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(payload)

    config_path = tmp_path / "renders.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "study_version": "test-renders",
                "pilot_config": str(_PILOT),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
                "report_root": str(tmp_path / "out"),
                "model": shape,
                "subgroups": 2,
                "samples_per_subgroup": 1,
                "seed": 11,
                "render": {"size": 18, "timeout_seconds": 20},
            }
        )
    )

    summary = run_ar_renders(load_ar_render_config(config_path), config_path)
    assert len(summary["subgroups"]) == 2
    assert len(set(summary["subgroups"])) == 2, "one exemplar per subgroup"
    assert len(summary["samples"]) == 2
    sheet = (tmp_path / "out" / "samples.png").read_bytes()
    assert sheet[:8] == b"\x89PNG\r\n\x1a\n"
    assert hashlib.sha256(sheet).hexdigest() == summary["sheet_sha256"]
    written = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert written == summary


def test_a_wrong_checkpoint_hash_is_refused(tmp_path: Path) -> None:
    """The sheet names a checkpoint; it must be the checkpoint that was measured."""

    config_path = tmp_path / "renders.yaml"
    checkpoint = tmp_path / "checkpoint.zip"
    checkpoint.write_bytes(b"not a checkpoint")
    config_path.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "study_version": "test-renders",
                "pilot_config": str(_PILOT),
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": "0" * 64,
                "report_root": str(tmp_path / "out"),
                "model": {"d_model": 32, "heads": 4, "layers": 2, "feedforward": 64},
                "subgroups": 1,
                "samples_per_subgroup": 1,
                "seed": 1,
                "render": {"size": 18, "timeout_seconds": 20},
            }
        )
    )
    try:
        run_ar_renders(load_ar_render_config(config_path), config_path)
    except Exception as error:  # noqa: BLE001 - the type is the module's own error
        assert "hash mismatch" in str(error)
    else:  # pragma: no cover - the guard must fire
        raise AssertionError("a mismatched checkpoint hash was accepted")
