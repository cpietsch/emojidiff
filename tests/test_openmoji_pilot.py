import json
from dataclasses import replace
from pathlib import Path

from mojidiff.learning.openmoji_pilot import (
    TRACE_FILENAME,
    load_openmoji_pilot_config,
    load_pilot_index,
    run_openmoji_pilot,
)


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
