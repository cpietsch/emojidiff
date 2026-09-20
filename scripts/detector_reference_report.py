#!/usr/bin/env python3
"""Fit and report the cheap-feature detection reference.

Bounds how much corrupted-field detection headroom exists above the zero-parameter
continuity statistic, using a logistic regression on eight local features computed from
the corrupted program alone. Trains only that 9-parameter model; it touches no
checkpoint and writes no learning artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mojidiff.learning.detectability import roc_auc  # noqa: E402
from mojidiff.learning.detector_reference import (  # noqa: E402
    FEATURE_NAMES,
    features_and_labels,
    fit_logistic,
    score,
)
from mojidiff.learning.openmoji_pilot import (  # noqa: E402
    CORRUPTION_PROCESSES,
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)

MATCHED_FLAG_RATE = 3430 / 40008


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-icons", type=int, default=256)
    args = parser.parse_args()

    config = load_openmoji_pilot_config(args.config)
    by_split, _, _ = load_pilot_index(config)
    codec = _selected_codec(config)
    corrupt = CORRUPTION_PROCESSES[config.corruption_process]
    probability = config.evaluation_probability

    def build(rows: tuple, seed_base: int) -> tuple[np.ndarray, np.ndarray]:
        features, labels = [], []
        for index, row in enumerate(rows):
            clean = _load_program(row, config, codec)
            noisy = corrupt(clean, codec, probability, np.random.default_rng(seed_base + index))
            f, y = features_and_labels(clean, noisy, codec)
            features.append(f)
            labels.append(y)
        return np.concatenate(features), np.concatenate(labels)

    train_x, train_y = build(
        _select_rows(by_split["primary/train"], args.train_icons, config.seed),
        config.seed + 5_000_000,
    )
    test_x, test_y = build(
        _select_rows(by_split["primary/validation"], config.validation_samples, config.seed + 1),
        config.seed + 9_000_000,
    )
    weight, bias, norm = fit_logistic(train_x, train_y)
    scores = score(test_x, weight, bias, norm)
    base = float(test_y.mean())

    points = {}
    for name, rate in (("matched_flag_rate", MATCHED_FLAG_RATE), ("flag_base_rate", base)):
        count = max(1, int(round(rate * len(test_y))))
        chosen = np.argsort(-scores)[:count]
        precision = float(test_y[chosen].mean())
        points[name] = {
            "flag_rate": count / len(test_y),
            "precision": precision,
            "lift_over_base_rate": precision / base,
            "recall": float(test_y[chosen].sum() / test_y.sum()),
        }
    print(
        json.dumps(
            {
                "config": str(args.config),
                "corruption_process": config.corruption_process,
                "evaluation_corruption_probability": probability,
                "parameters": len(FEATURE_NAMES) + 1,
                "train_icons": args.train_icons,
                "train_fields": int(len(train_y)),
                "heldout_fields": int(len(test_y)),
                "base_rate": base,
                "auc": roc_auc(list(scores[test_y == 1]), list(scores[test_y == 0])),
                "standardised_weights": dict(zip(FEATURE_NAMES, weight.tolist(), strict=True)),
                "operating_points": points,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
