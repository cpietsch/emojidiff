import hashlib
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import yaml

from mojidiff.learning.openmoji_pilot import (
    TRACE_FILENAME,
    OpenMojiPilotConfig,
    OpenMojiPilotError,
    SelectionPolicy,
    load_openmoji_pilot_config,
    load_pilot_index,
    run_openmoji_pilot,
)
from mojidiff.learning.tiny_study import _decode_checkpoint


def test_pinned_openmoji_pilot_index_is_complete_and_family_disjoint() -> None:
    config = load_openmoji_pilot_config(
        Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    )
    splits, groups, subgroups = load_pilot_index(config)

    assert {name: len(rows) for name, rows in splits.items()} == {
        "primary/train": 2681,
        "primary/validation": 339,
        "primary/test": 339,
    }
    assert sum(len(rows) for rows in splits.values()) == 3359
    assert len(groups) == 11
    assert len(subgroups) == 117
    families = {
        split: {row.variant_family_id for row in rows} for split, rows in splits.items()
    }
    assert families["primary/train"].isdisjoint(families["primary/validation"])
    assert families["primary/train"].isdisjoint(families["primary/test"])
    assert families["primary/validation"].isdisjoint(families["primary/test"])


def test_validation_tracing_is_metric_only_and_does_not_perturb_training(
    tmp_path: Path,
) -> None:
    config_path = Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    config = load_openmoji_pilot_config(config_path)
    assert config.eval_every == 0

    plain_root = tmp_path / "plain"
    plain = run_openmoji_pilot(
        replace(
            config,
            report_root=plain_root / "report",
            checkpoint_root=plain_root / "checkpoint",
        ),
        config_path,
    )
    traced_root = tmp_path / "traced"
    traced = run_openmoji_pilot(
        replace(
            config,
            report_root=traced_root / "report",
            checkpoint_root=traced_root / "checkpoint",
            eval_every=1,
        ),
        config_path,
    )

    # Tracing must be observationally inert for the trained model itself.
    assert traced["checkpoint_sha256"] == plain["checkpoint_sha256"]
    assert traced["validation"] == plain["validation"]
    assert not (plain_root / "report" / TRACE_FILENAME).exists()
    assert "validation_untrained" not in plain

    rows = [
        json.loads(line)
        for line in (traced_root / "report" / TRACE_FILENAME).read_text().splitlines()
    ]
    assert [row["step"] for row in rows] == [0, config.steps]
    assert rows[0]["trained"] is False and rows[-1]["trained"] is True
    # The untrained control must see the identical corruption draw, so it is a fair
    # same-input baseline rather than a differently corrupted evaluation.
    assert rows[0]["changed_total"] == rows[-1]["changed_total"]
    assert rows[0]["retained_total"] == rows[-1]["retained_total"]


def _selection_config(
    tmp_path: Path, name: str, **overrides: Any
) -> tuple[OpenMojiPilotConfig, Path, Path]:
    config_path = Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    root = tmp_path / name
    config = replace(
        load_openmoji_pilot_config(config_path),
        report_root=root / "report",
        checkpoint_root=root / "checkpoint",
        **overrides,
    )
    return config, config_path, root


def _trace(root: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (root / "report" / TRACE_FILENAME).read_text().splitlines()
    ]


def test_absent_selection_policy_leaves_the_reported_run_shape_unchanged(
    tmp_path: Path,
) -> None:
    config, config_path, _ = _selection_config(tmp_path, "plain", steps=3, eval_every=1)
    assert config.selection_policy is None

    summary = run_openmoji_pilot(config, config_path)

    # The v1 data-scale baseline was produced by this path, so it must stay a
    # fixed-budget run that reports its last trained step and nothing extra.
    assert "selection" not in summary
    assert "validation_final_step" not in summary
    assert summary["steps"] == 3


def test_selection_policy_reports_and_writes_the_lowest_held_out_loss_checkpoint(
    tmp_path: Path,
) -> None:
    config, config_path, root = _selection_config(
        tmp_path,
        "select",
        steps=4,
        eval_every=1,
        selection_policy=SelectionPolicy(
            objective="held_out_loss", patience_evals=99, min_delta=0.0
        ),
    )

    summary = run_openmoji_pilot(config, config_path)
    selection = summary["selection"]

    trained = [row for row in _trace(root) if row["trained"]]
    best = min(trained, key=lambda row: row["loss"])
    assert selection["stopped_early"] is False
    assert selection["completed_steps"] == 4
    assert selection["selected_step"] == best["step"]
    assert selection["selected_held_out_loss"] == best["loss"]
    # The reported held-out block must describe the selected checkpoint, while the
    # last trained step stays available for the overfitting comparison.
    assert summary["validation"]["loss"] == best["loss"]
    assert summary["validation_final_step"] == {
        key: value for key, value in trained[-1].items() if key not in {"step", "trained"}
    }
    written = (root / "checkpoint" / "checkpoint.zip").read_bytes()
    assert hashlib.sha256(written).hexdigest() == summary["checkpoint_sha256"]
    assert _decode_checkpoint(written)["step"] == best["step"]


def test_selection_policy_stops_after_patience_evaluations_without_improvement(
    tmp_path: Path,
) -> None:
    # A min_delta no evaluation can ever clear makes the stall deterministic: the first
    # trained evaluation is selected and every later one counts against patience.
    config, config_path, root = _selection_config(
        tmp_path,
        "stall",
        steps=8,
        eval_every=1,
        selection_policy=SelectionPolicy(
            objective="held_out_loss", patience_evals=2, min_delta=1e6
        ),
    )

    summary = run_openmoji_pilot(config, config_path)
    selection = summary["selection"]

    assert selection["stopped_early"] is True
    assert selection["selected_step"] == 1
    assert selection["completed_steps"] == 3
    assert selection["evals_without_improvement"] == 2
    assert [row["step"] for row in _trace(root)] == [0, 1, 2, 3]
    assert _decode_checkpoint((root / "checkpoint" / "checkpoint.zip").read_bytes())["step"] == 1


def test_selection_policy_requires_periodic_evaluation() -> None:
    config_path = Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    source = yaml.safe_load(config_path.read_bytes())
    source["training"]["selection_policy"] = {
        "objective": "held_out_loss",
        "patience_evals": 2,
    }
    broken = Path(tempfile.mkdtemp()) / "no-eval.yaml"
    broken.write_text(yaml.safe_dump(source))

    with pytest.raises(OpenMojiPilotError, match="nonzero training.eval_every"):
        load_openmoji_pilot_config(broken)


def test_a_model_without_noise_conditioning_rejects_a_noise_level() -> None:
    from mojidiff.learning.geometry import GeometryDenoiser
    from mojidiff.learning.openmoji_pilot import _selected_codec

    config = load_openmoji_pilot_config(
        Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    )
    codec = _selected_codec(config)
    plain = GeometryDenoiser(
        codec, config.total_segment_slots, d_model=32, heads=4, layers=1, feedforward=64,
        group_vocab_size=3, subgroup_vocab_size=3,
    )
    conditioned = GeometryDenoiser(
        codec, config.total_segment_slots, d_model=32, heads=4, layers=1, feedforward=64,
        group_vocab_size=3, subgroup_vocab_size=3, noise_level_features=8,
    )

    # The condition contract is exact in both directions, so a config and a checkpoint
    # cannot silently disagree about whether the model is told its corruption level.
    assert plain.noise_level_projection is None
    assert conditioned.noise_level_projection is not None
    import torch

    base = {
        "group": torch.zeros(1, dtype=torch.long),
        "subgroup": torch.zeros(1, dtype=torch.long),
    }
    with pytest.raises(ValueError, match="condition must contain exactly"):
        plain({"path_length": torch.zeros(1, config.max_paths, dtype=torch.long)},
              {**base, "noise_level": torch.zeros(1)})
    with pytest.raises(ValueError, match="condition must contain exactly"):
        conditioned({"path_length": torch.zeros(1, config.max_paths, dtype=torch.long)}, base)


def test_corruption_level_sampling_is_fixed_without_a_range_and_within_it_otherwise() -> None:
    import numpy as np

    from mojidiff.learning.openmoji_pilot import sample_corruption_level

    config = load_openmoji_pilot_config(
        Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    )
    assert config.corruption_probability_max is None
    fixed = [
        sample_corruption_level(config, np.random.default_rng(seed)) for seed in range(5)
    ]
    assert fixed == [config.corruption_probability] * 5

    ranged = replace(config, corruption_probability=0.05, corruption_probability_max=0.5)
    drawn = [
        sample_corruption_level(ranged, np.random.default_rng(seed)) for seed in range(20)
    ]
    assert all(0.05 <= value <= 0.5 for value in drawn)
    assert len(set(drawn)) > 1
    # Deterministic in the generator, so an example is a pure function of its seed.
    assert sample_corruption_level(ranged, np.random.default_rng(3)) == drawn[3]


def test_a_corruption_range_must_be_ordered(tmp_path: Path) -> None:
    config_path = Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    source = yaml.safe_load(config_path.read_bytes())
    source["training"]["corruption_probability_max"] = 0.1
    source["training"]["corruption_probability"] = 0.35
    broken = tmp_path / "range.yaml"
    broken.write_text(yaml.safe_dump(source))

    with pytest.raises(OpenMojiPilotError, match="corruption_probability_max must exceed"):
        load_openmoji_pilot_config(broken)


def test_the_evaluation_level_defaults_to_the_training_level() -> None:
    config = load_openmoji_pilot_config(
        Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")
    )

    assert config.evaluation_probability == config.corruption_probability
    assert replace(config, evaluation_corruption_probability=0.1).evaluation_probability == 0.1
