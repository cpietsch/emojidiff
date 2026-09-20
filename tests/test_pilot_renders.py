"""Contract tests for the read-only Gate G render probe.

The probe's actual renders depend on a multi-megabyte checkpoint that lives on the
durable artifact sink rather than in the repository, so these tests cover the parts
whose correctness does not depend on that artifact: configuration validation, the
refusal to render from a checkpoint whose bytes do not match the recorded hash, and
the reported aggregate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from mojidiff.learning.pilot_renders import (
    PilotRenderError,
    _aggregate,
    _markdown,
    load_pilot_render_config,
    run_pilot_renders,
)

_CONFIG = Path("configs/learning/openmoji-g1-train-v2-renders.yaml")


def test_the_committed_probe_config_loads() -> None:
    config = load_pilot_render_config(_CONFIG)

    assert config.render_sizes == (72, 18)
    assert config.icons > 0
    assert config.pilot_config.is_file()


def test_a_checkpoint_whose_bytes_changed_is_refused(tmp_path: Path) -> None:
    source = yaml.safe_load(_CONFIG.read_bytes())
    impostor = tmp_path / "checkpoint.zip"
    impostor.write_bytes(b"not the recorded checkpoint")
    source["checkpoint"] = str(impostor)
    source["report_root"] = str(tmp_path / "report")
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(source))

    # Renders are evidence about a specific trained model. Rendering from whatever
    # happens to be at that path would silently attribute one model's output to another.
    with pytest.raises(PilotRenderError, match="checkpoint hash mismatch"):
        run_pilot_renders(load_pilot_render_config(path), path)
    assert not (tmp_path / "report").exists()


def test_config_validation_rejects_a_bad_render_size(tmp_path: Path) -> None:
    source = yaml.safe_load(_CONFIG.read_bytes())
    source["render_sizes"] = []
    path = tmp_path / "probe.yaml"
    path.write_text(yaml.safe_dump(source))

    with pytest.raises(PilotRenderError, match="render_sizes"):
        load_pilot_render_config(path)


def test_the_aggregate_is_grouped_by_state_and_size() -> None:
    rows = [
        {"state": "x_t", "size": 72, "rgba_mae": 0.2},
        {"state": "x_t", "size": 72, "rgba_mae": 0.4},
        {"state": "x_hat_0", "size": 72, "rgba_mae": 0.1},
    ]

    aggregate = _aggregate(rows)

    assert aggregate["x_t-72"] == {"median_rgba_mae": pytest.approx(0.3), "count": 2.0}
    assert aggregate["x_hat_0-72"] == {"median_rgba_mae": pytest.approx(0.1), "count": 1.0}


def test_the_written_summary_labels_the_raw_corrupted_state() -> None:
    markdown = _markdown(
        {
            "checkpoint_step": 840,
            "icons": 12,
            "corruption_probability": 0.35,
            "aggregate": {"x_t-72": {"median_rgba_mae": 0.17, "count": 12.0}},
        }
    )

    # The project's reporting rule: the raw corrupted state is x_t and the prediction
    # is x_hat_0, and any projection would have to be labelled.
    assert "`x_t` the raw corrupted state" in markdown
    assert "No safety projection or constrained decoding is applied." in markdown
