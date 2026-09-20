#!/usr/bin/env python3
"""Measure a trained keep head as a corrupted-field detector.

Reports precision, recall and lift over the base rate at the model's own operating
point, and again at the 8.573% flag rate used to compare against the training-free
local-continuity statistic. Read-only: it restores a checkpoint and trains nothing.
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

from mojidiff.learning.geometry import _legal_fields, packed_batch  # noqa: E402
from mojidiff.learning.openmoji_pilot import (  # noqa: E402
    _condition,
    _load_program,
    _new_model,
    _select_rows,
    _selected_codec,
    corruption_operator,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.tiny_study import _decode_checkpoint  # noqa: E402

MATCHED_FLAG_RATE = 3430 / 40008
"""The v6 keep head's own flag rate, so lifts are comparable across detectors."""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    args = parser.parse_args()

    config = load_openmoji_pilot_config(args.config)
    by_split, groups, subgroups = load_pilot_index(config)
    codec = _selected_codec(config)
    device = torch.device("cpu")
    rows = _select_rows(by_split["primary/validation"], config.validation_samples, config.seed + 1)
    model = _new_model(config, codec, groups, subgroups).to(device)
    model.load_state_dict(_decode_checkpoint(args.checkpoint.read_bytes())["model"])
    model.eval()

    clean = [_load_program(row, config, codec) for row in rows]
    # Bind through the pilot's own operator so the corruption draw is identical to the
    # one the run trained and evaluated against; `marginal` needs the train programs.
    probability = config.evaluation_probability
    train_programs = [
        _load_program(row, config, codec)
        for row in _select_rows(by_split["primary/train"], 512, config.seed)
    ]
    corrupt = corruption_operator(config, codec, train_programs)
    noisy = [
        corrupt(program, codec, probability, np.random.default_rng(config.seed + 9_000_000 + index))
        for index, program in enumerate(clean)
    ]
    noisy_batch, clean_batch = packed_batch(noisy, device), packed_batch(clean, device)
    with torch.no_grad():
        start_logits, coordinate_logits, keep = model.forward_with_edits(
            noisy_batch,
            _condition(
                rows,
                groups,
                subgroups,
                device,
                [probability] * len(rows) if config.noise_level_features else None,
            ),
        )
    if keep is None:
        raise SystemExit("this checkpoint has no keep head")

    scores: list[float] = []
    labels: list[int] = []
    for _, keep_logits, targets, noisy_tokens in _legal_fields(
        (start_logits, coordinate_logits), keep, noisy_batch, clean_batch, codec
    ):
        if keep_logits is None or not targets.numel():
            continue
        probabilities = torch.softmax(keep_logits.float(), dim=-1)[:, 0]
        scores.extend(probabilities.tolist())
        labels.extend((noisy_tokens != targets).long().tolist())

    score = np.asarray(scores)
    label = np.asarray(labels)
    base = float(label.mean())
    report = {
        "config": str(args.config),
        "corruption_process": config.corruption_process,
        "evaluation_corruption_probability": probability,
        "detection_only": config.detection_only,
        "fields": int(label.size),
        "base_rate": base,
        "operating_points": {},
    }
    flagged = score > 0.5
    report["operating_points"]["model_own_threshold"] = _point(flagged, label, base)
    for name, rate in (("matched_flag_rate", MATCHED_FLAG_RATE), ("flag_base_rate", base)):
        count = max(1, int(round(rate * label.size)))
        cutoff = np.argsort(-score)[:count]
        mask = np.zeros_like(label, dtype=bool)
        mask[cutoff] = True
        report["operating_points"][name] = _point(mask, label, base)
    print(json.dumps(report, indent=2, sort_keys=True))


def _point(flagged: np.ndarray, label: np.ndarray, base: float) -> dict[str, float]:
    true_positive = int((flagged & (label == 1)).sum())
    precision = true_positive / int(flagged.sum()) if flagged.sum() else float("nan")
    return {
        "flag_rate": float(flagged.mean()),
        "precision": precision,
        "lift_over_base_rate": precision / base if base else float("nan"),
        "recall": true_positive / int((label == 1).sum()),
        "flagged": int(flagged.sum()),
        "true_positive": true_positive,
    }


if __name__ == "__main__":
    main()
