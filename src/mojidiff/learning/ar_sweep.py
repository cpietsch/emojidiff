"""Is the autoregressive model short of data, or short of something else?

Two arms are in: at 2,681 icons the causal model clears a zero-parameter position-
marginal floor by 7.7% in nats against a predeclared 50%, and correcting the coordinate
encoding - the defect that was decisive for the denoiser on this same codec - moves it
nowhere. Both peak within two to four epochs and then get worse while their training
loss keeps falling.

That signature has two readings and they call for different work, so guessing between
them is the expensive mistake. This measures instead. Three nested training sets, one
quarter, one half and all of it, everything else held fixed, and a criterion declared
before the runs: if held-out likelihood improves monotonically with data and the second
doubling still buys a meaningful fraction of what the first did, the constraint is data
volume and the next work is Gate H. If the curve is flat or saturating, more OpenMoji
icons will not help and the constraint is the representation or the model.

Nesting the subsets matters. Three independent samples would differ in composition as
well as in size, and a subset that happened to hold more flags would move the curve for
a reason that has nothing to do with scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from mojidiff.learning.ar_corpus import (
    _build_split,
    _evaluate,
    _free_mask,
    _marginal_nll,
    _position_marginals,
    _Split,
)
from mojidiff.learning.autoregressive import (
    CausalProgramModel,
    SequenceLayout,
    teacher_forcing_inputs,
)
from mojidiff.learning.openmoji_pilot import (
    OpenMojiPilotError,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.representation.codec_study import _write_bytes_artifact


@dataclass(frozen=True)
class Arm:
    """One point on whichever axis this study sweeps."""

    label: str
    fraction: float
    d_model: int
    heads: int
    layers: int
    feedforward: int
    metric_coordinates: int


@dataclass(frozen=True)
class ARScalingConfig:
    version: str
    axis: str
    pilot_config: Path
    report_root: Path
    arms: tuple[Arm, ...]
    steps: int
    batch_size: int
    learning_rate: float
    seed: int
    eval_every: int
    patience_evals: int
    marginal_alpha: float
    min_second_doubling_share: float
    max_best_ratio: float


def load_ar_scaling_config(path: Path) -> ARScalingConfig:
    root = yaml.safe_load(path.read_bytes())
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise OpenMojiPilotError("ar scaling schema_version must be 1")
    base = root["model"]
    training = root["training"]
    arms = tuple(
        Arm(
            label=str(entry["label"]),
            fraction=float(entry.get("fraction", 1.0)),
            d_model=int(entry.get("d_model", base["d_model"])),
            heads=int(entry.get("heads", base["heads"])),
            layers=int(entry.get("layers", base["layers"])),
            feedforward=int(entry.get("feedforward", base["feedforward"])),
            metric_coordinates=int(
                entry.get("metric_coordinates", base.get("metric_coordinates", 0))
            ),
        )
        for entry in root["arms"]
    )
    if len(arms) != 3:
        raise OpenMojiPilotError("this study reads three arms, ordered small to large")
    return ARScalingConfig(
        version=str(root["study_version"]),
        axis=str(root["axis"]),
        pilot_config=Path(str(root["pilot_config"])),
        report_root=Path(str(root["report_root"])),
        arms=arms,
        steps=int(training["steps"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        seed=int(training["seed"]),
        eval_every=int(training["eval_every"]),
        patience_evals=int(training["patience_evals"]),
        marginal_alpha=float(training["marginal_alpha"]),
        min_second_doubling_share=float(root["criteria"]["min_second_doubling_share"]),
        max_best_ratio=float(root["criteria"]["max_best_ratio"]),
    )


def run_ar_scaling(config: ARScalingConfig, config_path: Path) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    pilot = load_openmoji_pilot_config(config.pilot_config)
    by_split, groups, subgroups = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)

    started = time.perf_counter()
    train = _build_split(by_split["primary/train"], pilot, codec, layout)
    validation = _build_split(by_split["primary/validation"], pilot, codec, layout)
    prepare_seconds = time.perf_counter() - started

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    validation_condition = {
        "group": torch.tensor(
            [groups[row.group] for row in validation.rows], dtype=torch.long
        ),
        "subgroup": torch.tensor(
            [subgroups[row.subgroup] for row in validation.rows], dtype=torch.long
        ),
    }
    train_condition = {
        "group": torch.tensor([groups[row.group] for row in train.rows], dtype=torch.long),
        "subgroup": torch.tensor(
            [subgroups[row.subgroup] for row in train.rows], dtype=torch.long
        ),
    }

    # The subsets are nested, so a larger arm is a strict superset of a smaller one and
    # the only thing that differs between arms is how many icons there are.
    order = np.random.default_rng(config.seed).permutation(len(train.rows))
    results: list[dict[str, Any]] = []
    for arm in config.arms:
        count = max(int(round(arm.fraction * len(train.rows))), config.batch_size)
        subset = np.sort(order[:count])
        results.append(
            _train_arm(
                config,
                arm,
                layout,
                train,
                subset,
                train_condition,
                validation,
                validation_condition,
                groups,
                subgroups,
                device,
            )
        )
    arms = results

    improvements = [
        results[0]["held_out_nll"] - results[1]["held_out_nll"],
        results[1]["held_out_nll"] - results[2]["held_out_nll"],
    ]
    share = improvements[1] / improvements[0] if improvements[0] > 0 else 0.0
    monotone = all(value > 0 for value in improvements)
    best_ratio = min(float(arm["ratio_to_own_floor"]) for arm in results)
    checks = {
        f"monotone_in_{config.axis}": monotone,
        "second_step_still_buys": monotone and share >= config.min_second_doubling_share,
        # i4's lesson, written into the instrument. Its criterion asked whether returns
        # had stopped and passed while the returns were far too small to matter, which
        # is the Gate G failure of a metric that reads well and points the wrong way.
        # A direction without a magnitude does not identify a lever.
        "best_arm_clears_magnitude_bar": best_ratio <= config.max_best_ratio,
    }

    summary = {
        "schema_version": 1,
        "study_version": config.version,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "deterministic_algorithms": True,
        "axis": config.axis,
        "prepare_seconds": prepare_seconds,
        "validation_icons": len(validation.rows),
        "arms": arms,
        "improvement_first_doubling": improvements[0],
        "improvement_second_doubling": improvements[1],
        "second_doubling_share": share,
        "best_ratio_to_own_floor": best_ratio,
        "criteria": {
            "min_second_doubling_share": config.min_second_doubling_share,
            "max_best_ratio": config.max_best_ratio,
        },
        "checks": checks,
        "predeclared_outcome": (
            f"limited_by_{config.axis}" if all(checks.values()) else f"not_limited_by_{config.axis}"
        ),
    }
    _write_bytes_artifact(config.report_root / "summary.json", _json(summary))
    _write_bytes_artifact(
        config.report_root / "metrics.jsonl",
        b"".join(_json(row) for arm in arms for row in arm["trace"]),
    )
    return summary


def _train_arm(
    config: ARScalingConfig,
    arm: Arm,
    layout: SequenceLayout,
    train: _Split,
    subset: np.ndarray,
    train_condition: dict[str, torch.Tensor],
    validation: _Split,
    validation_condition: dict[str, torch.Tensor],
    groups: dict[str, int],
    subgroups: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    model = CausalProgramModel(
        layout,
        d_model=arm.d_model,
        heads=arm.heads,
        layers=arm.layers,
        feedforward=arm.feedforward,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
        metric_coordinates=arm.metric_coordinates,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # The floor is refitted on this arm's icons. A policy given the whole corpus while
    # the model sees a quarter of it is not a floor, it is a different experiment.
    marginals = _position_marginals(
        _subset_view(train, subset), layout, config.marginal_alpha
    )
    floor_total = 0.0
    floor_count = 0
    for start in range(0, len(validation.rows), 32):
        window = np.arange(start, min(start + 32, len(validation.rows)))
        masks = validation.masks(window)
        total, count = _marginal_nll(
            marginals, validation.tokens[window], masks, _free_mask(masks)
        )
        floor_total += total
        floor_count += count

    rng = np.random.default_rng(config.seed + 1)
    trace: list[dict[str, Any]] = []
    best = float("inf")
    best_step = 0
    stale = 0
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        indices = subset[rng.choice(len(subset), size=config.batch_size, replace=False)]
        tokens = train.tokens[indices].to(device)
        masks = train.masks(indices).to(device)
        shifted, kinds = teacher_forcing_inputs(tokens, layout)
        free = _free_mask(masks)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(
            shifted,
            {key: value[indices].to(device) for key, value in train_condition.items()},
            kinds=kinds,
        )
        loss = torch.nn.functional.cross_entropy(
            logits.masked_fill(~masks, float("-inf"))[free], tokens[free]
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if step % config.eval_every == 0 or step == config.steps:
            held_out = _evaluate(model, validation, validation_condition, device)
            trace.append(
                {
                    "label": arm.label,
                    "fraction": arm.fraction,
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "held_out_nll": held_out,
                    "marginal_nll": floor_total / floor_count,
                }
            )
            if held_out < best:
                best, best_step, stale = held_out, step, 0
            else:
                stale += 1
                if stale >= config.patience_evals:
                    break
    return {
        "label": arm.label,
        "fraction": arm.fraction,
        "icons": len(subset),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "held_out_nll": best,
        "marginal_nll": floor_total / floor_count,
        "ratio_to_own_floor": best / (floor_total / floor_count),
        "selected_step": best_step,
        "steps_run": trace[-1]["step"] if trace else 0,
        "epochs_to_best": best_step * config.batch_size / len(subset),
        "train_seconds": time.perf_counter() - started,
        "trace": trace,
    }


def _subset_view(split: _Split, subset: np.ndarray) -> _Split:
    view = _Split(tuple(split.rows[index] for index in subset), split.layout)
    view.tokens = split.tokens[subset]
    view.packed = split.packed[subset]
    return view


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_ar_scaling_config(args.config)
    print(json.dumps(run_ar_scaling(config, args.config), sort_keys=True))


if __name__ == "__main__":
    main()
