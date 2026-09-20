#!/usr/bin/env python3
"""Does the trained encoder represent the corruption signal at all?

The keep head is already a linear readout of the encoder output, so fitting a fresh
linear probe on the SAME frozen features separates two very different failures:

* probe much better than the keep head -> the encoder carries the signal and the head,
  or its optimisation, is at fault;
* probe no better -> the encoder never represented it, and the problem is upstream in
  how geometry enters the network.

Read-only: it restores a checkpoint, freezes it, and fits only the probe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from mojidiff.learning.detectability import roc_auc  # noqa: E402
from mojidiff.learning.detector_reference import fit_logistic, score  # noqa: E402
from mojidiff.learning.geometry import packed_batch  # noqa: E402
from mojidiff.learning.openmoji_pilot import (  # noqa: E402
    CORRUPTION_PROCESSES,
    _condition,
    _load_program,
    _new_model,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.tiny_study import _decode_checkpoint  # noqa: E402

MATCHED_FLAG_RATE = 3430 / 40008


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--train-icons", type=int, default=96)
    args = parser.parse_args()

    config = load_openmoji_pilot_config(args.config)
    by_split, groups, subgroups = load_pilot_index(config)
    codec = _selected_codec(config)
    device = torch.device("cpu")
    model = _new_model(config, codec, groups, subgroups).to(device)
    model.load_state_dict(_decode_checkpoint(args.checkpoint.read_bytes())["model"])
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def hook(_module: object, _inputs: object, output: torch.Tensor) -> None:
        captured["encoded"] = output.detach()

    model.encoder.register_forward_hook(hook)
    corrupt = CORRUPTION_PROCESSES[config.corruption_process]
    probability = config.evaluation_probability

    def collect(rows: tuple, seed_base: int) -> tuple[np.ndarray, np.ndarray]:
        features: list[np.ndarray] = []
        labels: list[int] = []
        for index, row in enumerate(rows):
            clean = _load_program(row, config, codec)
            noisy = corrupt(clean, codec, probability, np.random.default_rng(seed_base + index))
            noisy_batch = packed_batch([noisy], device)
            clean_batch = packed_batch([clean], device)
            with torch.no_grad():
                model.forward_with_edits(
                    noisy_batch, _condition((row,), groups, subgroups, device)
                )
            segments = captured["encoded"][0][config.max_paths :].numpy()
            noisy_coordinates = noisy_batch["coordinates"][0].numpy()
            clean_coordinates = clean_batch["coordinates"][0].numpy()
            for segment in range(segments.shape[0]):
                for slot in range(6):
                    if noisy_coordinates[segment, slot] <= 0:
                        continue
                    marker = np.zeros(6)
                    marker[slot] = 1.0
                    features.append(np.concatenate([segments[segment], marker]))
                    labels.append(
                        int(noisy_coordinates[segment, slot] != clean_coordinates[segment, slot])
                    )
        return np.asarray(features, dtype=np.float64), np.asarray(labels, dtype=np.int64)

    train_x, train_y = collect(
        _select_rows(by_split["primary/train"], args.train_icons, config.seed),
        config.seed + 5_000_000,
    )
    test_x, test_y = collect(
        _select_rows(by_split["primary/validation"], config.validation_samples, config.seed + 1),
        config.seed + 9_000_000,
    )
    weight, bias, norm = fit_logistic(train_x, train_y)
    scores = score(test_x, weight, bias, norm)
    base = float(test_y.mean())
    count = max(1, int(round(MATCHED_FLAG_RATE * len(test_y))))
    chosen = np.argsort(-scores)[:count]
    precision = float(test_y[chosen].mean())
    print(
        json.dumps(
            {
                "config": str(args.config),
                "checkpoint": str(args.checkpoint),
                "probe_dimensions": int(train_x.shape[1]),
                "train_rows": int(len(train_y)),
                "heldout_rows": int(len(test_y)),
                "base_rate": base,
                "auc": roc_auc(list(scores[test_y == 1]), list(scores[test_y == 0])),
                "matched_flag_rate": {
                    "precision": precision,
                    "lift_over_base_rate": precision / base,
                    "recall": float(test_y[chosen].sum() / test_y.sum()),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
