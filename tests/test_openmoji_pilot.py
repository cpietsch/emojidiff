from pathlib import Path

from mojidiff.learning.openmoji_pilot import (
    load_openmoji_pilot_config,
    load_pilot_index,
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
