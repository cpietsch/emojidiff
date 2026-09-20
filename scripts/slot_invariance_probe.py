#!/usr/bin/env python3
"""Evidence that the unbound encoder cannot tell which coordinate is which.

`GeometryDenoiser` without slot binding sums six coordinate lookups - all from one
shared embedding table - into a single vector per segment slot. A sum is commutative,
so a segment is represented by an unordered bag of its coordinate values and a program
is indistinguishable from its coordinate-swapped variant.

This probe measures that directly, in float64 so the result cannot rest on
summation-order noise. It is read-only: it builds an untrained model, because the
invariance is a property of the encoder rather than of anything learned.

    python scripts/slot_invariance_probe.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mojidiff.learning.geometry import packed_batch  # noqa: E402
from mojidiff.learning.openmoji_pilot import (  # noqa: E402
    _condition,
    _load_program,
    _new_model,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)

CONFIG = (
    _REPO_ROOT / "configs" / "learning" / "openmoji-g1-dominant-bucket-train-v2-data-scale.yaml"
)
ICONS = 6


def main() -> None:
    config = load_openmoji_pilot_config(CONFIG)
    by_split, groups, subgroups = load_pilot_index(config)
    codec = _selected_codec(config)
    device = torch.device("cpu")
    rows = _select_rows(by_split["primary/validation"], config.validation_samples, config.seed + 1)

    torch.manual_seed(0)
    model = _new_model(config, codec, groups, subgroups).eval().double()

    results: list[dict[str, Any]] = []
    for index in range(ICONS):
        row = rows[index]
        program = _load_program(row, config, codec)
        swapped = copy.deepcopy(program)
        changed = 0
        for slot in range(swapped.coordinates.shape[0]):
            first = int(swapped.coordinates[slot, 0])
            second = int(swapped.coordinates[slot, 1])
            if first != second and first != 0 and second != 0:
                swapped.coordinates[slot, 0] = second
                swapped.coordinates[slot, 1] = first
                changed += 1
        condition = _condition((row,), groups, subgroups, device)
        with torch.no_grad():
            start, coordinates = model(packed_batch([program], device), condition)
            start_swapped, coordinates_swapped = model(packed_batch([swapped], device), condition)
        results.append(
            {
                "hexcode": row.hexcode,
                "segments_swapped": changed,
                "max_abs_logit_difference": float((coordinates - coordinates_swapped).abs().max()),
                "identical_prediction": bool(
                    torch.equal(coordinates.argmax(-1), coordinates_swapped.argmax(-1))
                    and torch.equal(start.argmax(-1), start_swapped.argmax(-1))
                ),
            }
        )

    print(
        json.dumps(
            {
                "precision": "float64",
                "slot_binding": config.slot_binding,
                "config": str(CONFIG.relative_to(_REPO_ROOT)),
                "rows": results,
                "total_segments_swapped": sum(row["segments_swapped"] for row in results),
                "all_predictions_identical": all(row["identical_prediction"] for row in results),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
