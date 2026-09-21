"""Gate L: train the masked model and measure it on the edit an editor actually makes.

One harness serves both of the gate's first two runs, because the discipline is the
same and only the scale and the criteria differ. The overfit run trains on a handful of
icons and asks whether the objective, the masks and the decoder are wired correctly -
memorisation, a loss that actually falls, and exact reproduction of a masked path. The
corpus run trains on the family-disjoint split and asks the question that matters:
given a held-out icon with one path removed, does the model draw that path back better
than leaving it out, and better than a zero-parameter sampler that knows the corpus?

Both are scored on renders as well as tokens. Gate G's whole history says why.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from mojidiff.learning.ar_corpus import _build_split, _free_mask, _position_marginals, _Split
from mojidiff.learning.ar_overfit import _distinct_subgroup_rows
from mojidiff.learning.autoregressive import (
    SequenceLayout,
    unflatten_program,
)
from mojidiff.learning.masked import (
    MaskedProgramModel,
    MaskMixture,
    apply_mask,
    complete,
    coordinate_positions,
    drop_path,
    marginal_logits,
    masked_loss,
    masked_nll,
    model_logits,
    path_blocks,
    sample_mask,
    whole_path_mask,
)
from mojidiff.learning.openmoji_pilot import (
    OpenMojiPilotError,
    PilotRow,
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.tiny_study import _decode_checkpoint, _save_checkpoint
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.packed import (
    PackedTensorProgram,
    serialize_packed_svg,
    validate_packed_tensor_program,
)
from mojidiff.representation.renderer import RenderLimits, render_typed_svg_isolated
from mojidiff.representation.study import _similarity

SHEET_FILENAME = "inpainting.png"
_EVAL_MASK_SEED = 4_000_000
_INPAINT_SEED = 5_000_000


@dataclass(frozen=True)
class MaskedStudyConfig:
    version: str
    pilot_config: Path
    report_root: Path
    checkpoint_root: Path
    train_icons: int | None
    distinct_subgroups: bool
    evaluate_on: str
    evaluation_icons: int
    inpaint_icons: int
    d_model: int
    heads: int
    layers: int
    feedforward: int
    metric_coordinates: int
    dropout: float
    mixture: MaskMixture
    steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    seed: int
    eval_every: int
    patience_evals: int
    marginal_alpha: float
    coordinate_tau: float
    iterations: int
    samples: int
    render_size: int
    render_timeout_seconds: int
    criteria: dict[str, float | bool]


def load_masked_study_config(path: Path) -> MaskedStudyConfig:
    root = yaml.safe_load(path.read_bytes())
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise OpenMojiPilotError("masked study schema_version must be 1")
    data = root["data"]
    model = root["model"]
    masks = dict(root["masks"])
    training = root["training"]
    decoding = root["decoding"]
    evaluate_on = str(data.get("evaluate_on", "validation"))
    if evaluate_on not in {"train", "validation"}:
        raise OpenMojiPilotError("data.evaluate_on must be train or validation")
    mixture = MaskMixture(
        weights={key: float(masks[key]) for key in masks if key in _FAMILY_KEYS},
        random_rate_min=float(masks.get("random_rate_min", 0.05)),
        random_rate_max=float(masks.get("random_rate_max", 0.95)),
        max_paths=int(masks.get("max_paths", 2)),
    )
    criteria = {str(key): value for key, value in dict(root["criteria"]).items()}
    unknown = set(criteria) - _CRITERIA
    if unknown:
        raise OpenMojiPilotError(f"unknown criteria: {sorted(unknown)}")
    train_icons = data.get("train_icons")
    return MaskedStudyConfig(
        version=str(root["study_version"]),
        pilot_config=Path(str(root["pilot_config"])),
        report_root=Path(str(root["report_root"])),
        checkpoint_root=Path(str(root["checkpoint_root"])),
        train_icons=None if train_icons is None else int(train_icons),
        distinct_subgroups=bool(data.get("distinct_subgroups", False)),
        evaluate_on=evaluate_on,
        evaluation_icons=int(data["evaluation_icons"]),
        inpaint_icons=int(data["inpaint_icons"]),
        d_model=int(model["d_model"]),
        heads=int(model["heads"]),
        layers=int(model["layers"]),
        feedforward=int(model["feedforward"]),
        metric_coordinates=int(model.get("metric_coordinates", 0)),
        dropout=float(model.get("dropout", 0.0)),
        mixture=mixture,
        steps=int(training["steps"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.01)),
        seed=int(training["seed"]),
        eval_every=int(training["eval_every"]),
        patience_evals=int(training["patience_evals"]),
        marginal_alpha=float(training.get("marginal_alpha", 1.0)),
        coordinate_tau=float(training.get("coordinate_tau", 0.0)),
        iterations=int(decoding.get("iterations", 8)),
        samples=int(decoding.get("samples", 0)),
        render_size=int(decoding.get("render_size", 72)),
        render_timeout_seconds=int(decoding.get("render_timeout_seconds", 20)),
        criteria=criteria,
    )


_FAMILY_KEYS = {"random", "path", "span", "style", "geometry"}
_CRITERIA = {
    "min_masked_token_accuracy",
    "min_loss_reduction_factor",
    "min_exact_path_reproduction_rate",
    "beats_drop_baseline",
    "beats_marginal_baseline",
    "min_median_recovery",
    "require_all_valid",
}


def run_masked_study(config: MaskedStudyConfig, config_path: Path) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    pilot = load_openmoji_pilot_config(config.pilot_config)
    by_split, groups, subgroups = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)

    train_rows = by_split["primary/train"]
    if config.train_icons is not None:
        train_rows = (
            _distinct_subgroup_rows(train_rows, config.train_icons, pilot.seed)
            if config.distinct_subgroups
            else _select_rows(train_rows, config.train_icons, pilot.seed)
        )
    source_rows = train_rows if config.evaluate_on == "train" else by_split["primary/validation"]
    evaluation_rows = _select_rows(
        source_rows, min(config.evaluation_icons, len(source_rows)), pilot.seed + 1
    )

    prepared = time.perf_counter()
    train = _build_split(train_rows, pilot, codec, layout)
    evaluation = _build_split(evaluation_rows, pilot, codec, layout)
    prepare_seconds = time.perf_counter() - prepared

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MaskedProgramModel(
        layout,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        feedforward=config.feedforward,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
        metric_coordinates=config.metric_coordinates,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    condition = {
        name: _condition(split.rows, groups, subgroups)
        for name, split in (("train", train), ("evaluation", evaluation))
    }
    coordinates = coordinate_positions(layout).to(device)

    # One fixed mask draw per evaluation icon, reused at every evaluation, so the
    # held-out trace measures the model and not the dice.
    evaluation_hide = torch.stack(
        [
            sample_mask(
                evaluation.tokens[index],
                layout,
                config.mixture,
                np.random.default_rng(config.seed + _EVAL_MASK_SEED + index),
            )[1]
            for index in range(len(evaluation.rows))
        ]
    )
    marginals = _position_marginals(train, layout, config.marginal_alpha)
    floor_nll, floor_accuracy = _floor(marginals, evaluation, evaluation_hide)

    rng = np.random.default_rng(config.seed)
    # The untrained model is evaluated first, so the loss-reduction criterion reads an
    # exact-token likelihood before and after training rather than the soft training
    # loss, whose floor is the entropy of its own spread target and never reaches zero.
    initial_nll, initial_accuracy = _evaluate(
        model, evaluation, evaluation_hide, condition["evaluation"], device
    )
    metrics: list[dict[str, Any]] = [
        {
            "step": 0,
            "held_out_nll": initial_nll,
            "marginal_nll": floor_nll,
            "masked_token_accuracy": initial_accuracy,
        }
    ]
    families: dict[str, int] = {}
    best_step = 0
    best_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    first_loss: float | None = None
    stale = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        indices = rng.choice(
            len(train.rows), size=min(config.batch_size, len(train.rows)), replace=False
        )
        tokens = train.tokens[indices]
        legal = train.masks(indices)
        hide = []
        for index in indices:
            family, mask = sample_mask(train.tokens[index], layout, config.mixture, rng)
            families[family] = families.get(family, 0) + 1
            hide.append(mask)
        hidden = torch.stack(hide)
        inputs = apply_mask(tokens, hidden, layout).to(device)
        targets = (hidden & _free_mask(legal)).to(device)
        batch_condition = {
            key: value[indices].to(device) for key, value in condition["train"].items()
        }
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs, batch_condition)
        loss = masked_loss(
            logits, tokens.to(device), legal.to(device), targets, coordinates, config.coordinate_tau
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if first_loss is None:
            first_loss = float(loss.detach())

        if step % config.eval_every == 0 or step == config.steps:
            held_out_nll, accuracy = _evaluate(
                model, evaluation, evaluation_hide, condition["evaluation"], device
            )
            metrics.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "held_out_nll": held_out_nll,
                    "marginal_nll": floor_nll,
                    "masked_token_accuracy": accuracy,
                }
            )
            if held_out_nll < best_nll:
                best_step, best_nll, stale = step, held_out_nll, 0
                best_state = {
                    key: value.detach().cpu().clone() for key, value in model.state_dict().items()
                }
            else:
                stale += 1
                if stale >= config.patience_evals:
                    break
    train_seconds = time.perf_counter() - started
    peak_vram = float(torch.cuda.max_memory_allocated()) / 2**30 if device.type == "cuda" else None
    if best_state is None or first_loss is None:
        raise OpenMojiPilotError("no evaluation ran, so no checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()
    last_loss = metrics[-1]["train_loss"]
    best_accuracy = next(
        row["masked_token_accuracy"] for row in metrics if row["step"] == best_step
    )

    # The trace and the checkpoint are the expensive part; they are written before any
    # diagnostic that can still fail runs, which Gate I learned at the cost of a run.
    _write_bytes_artifact(config.report_root / "metrics.jsonl", _jsonl(metrics))
    _write_bytes_artifact(
        config.checkpoint_root / "checkpoint.zip", _save_checkpoint(model, optimizer, best_step)
    )

    inpainting = _inpaint(
        model,
        config,
        pilot,
        codec,
        layout,
        evaluation,
        condition["evaluation"],
        marginals,
        groups,
        subgroups,
    )

    loss_reduction = initial_nll / max(best_nll, 1e-12)
    checks: dict[str, bool] = {}
    criteria = config.criteria
    if "min_masked_token_accuracy" in criteria:
        checks["masked_token_accuracy"] = best_accuracy >= float(
            criteria["min_masked_token_accuracy"]
        )
    if "min_loss_reduction_factor" in criteria:
        checks["loss_reduction"] = loss_reduction >= float(criteria["min_loss_reduction_factor"])
    if "min_exact_path_reproduction_rate" in criteria:
        checks["exact_path_reproduction"] = inpainting["exact_reproduction_rate"] >= float(
            criteria["min_exact_path_reproduction_rate"]
        )
    if "beats_drop_baseline" in criteria and bool(criteria["beats_drop_baseline"]):
        checks["beats_drop_baseline"] = inpainting["paired"]["drop_minus_model"][
            "interval_excludes_zero_above"
        ]
    if "beats_marginal_baseline" in criteria and bool(criteria["beats_marginal_baseline"]):
        checks["beats_marginal_baseline"] = inpainting["paired"]["marginal_minus_model"][
            "interval_excludes_zero_above"
        ]
    if "min_median_recovery" in criteria:
        checks["median_recovery"] = inpainting["model"]["median_recovery"] >= float(
            criteria["min_median_recovery"]
        )
    if "require_all_valid" in criteria and bool(criteria["require_all_valid"]):
        checks["all_valid"] = inpainting["all_valid"]

    checkpoint = (config.checkpoint_root / "checkpoint.zip").read_bytes()
    summary = {
        "schema_version": 1,
        "study_version": config.version,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "deterministic_algorithms": True,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "sequence_length": layout.length,
        "vocabulary": layout.vocabulary,
        "train_icons": len(train.rows),
        "evaluation_split": config.evaluate_on,
        "evaluation_icons": len(evaluation.rows),
        "mask_mixture": {
            "weights": dict(config.mixture.weights),
            "random_rate": [config.mixture.random_rate_min, config.mixture.random_rate_max],
            "max_paths": config.mixture.max_paths,
            "drawn": families,
        },
        "coordinate_tau": config.coordinate_tau,
        "icon_presentations": metrics[-1]["step"] * config.batch_size,
        "selected_step": best_step,
        "steps_run": metrics[-1]["step"],
        "held_out_nll_per_masked_token": best_nll,
        "marginal_nll_per_masked_token": floor_nll,
        "nll_ratio_to_marginal": best_nll / floor_nll,
        "masked_token_accuracy": best_accuracy,
        "marginal_masked_token_accuracy": floor_accuracy,
        "initial_held_out_nll": initial_nll,
        "first_train_loss": first_loss,
        "last_train_loss": last_loss,
        "loss_reduction_factor": loss_reduction,
        "inpainting": inpainting,
        "timing": {
            "prepare_seconds": prepare_seconds,
            "train_seconds": train_seconds,
            "peak_vram_gib": peak_vram,
        },
        "criteria": criteria,
        "checks": checks,
        "predeclared_outcome": "passed" if all(checks.values()) else "falsified",
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "checkpoint_round_trip": _decode_checkpoint(checkpoint)["step"] == best_step,
        "metrics_sha256": hashlib.sha256(_jsonl(metrics)).hexdigest(),
    }
    _write_bytes_artifact(config.report_root / "summary.json", _json(summary))
    return summary


def _condition(
    rows: tuple[PilotRow, ...], groups: dict[str, int], subgroups: dict[str, int]
) -> dict[str, torch.Tensor]:
    return {
        "group": torch.tensor([groups[row.group] for row in rows], dtype=torch.long),
        "subgroup": torch.tensor([subgroups[row.subgroup] for row in rows], dtype=torch.long),
    }


def _floor(marginals: torch.Tensor, split: _Split, hide: torch.Tensor) -> tuple[float, float]:
    """The position-marginal policy's likelihood and accuracy on the evaluation masks."""

    total, hits, count = 0.0, 0, 0
    scores = torch.log(marginals.to(torch.float32))
    for start in range(0, len(split.rows), 32):
        window = np.arange(start, min(start + 32, len(split.rows)))
        legal = split.masks(window)
        targets = hide[window] & _free_mask(legal)
        logits = scores[None].expand(len(window), -1, -1)
        nll, batch_hits, batch_count = masked_nll(logits, split.tokens[window], legal, targets)
        total, hits, count = total + nll, hits + batch_hits, count + batch_count
    return total / max(count, 1), hits / max(count, 1)


@torch.no_grad()
def _evaluate(
    model: MaskedProgramModel,
    split: _Split,
    hide: torch.Tensor,
    condition: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total, hits, count = 0.0, 0, 0
    for start in range(0, len(split.rows), 16):
        window = np.arange(start, min(start + 16, len(split.rows)))
        tokens = split.tokens[window]
        legal = split.masks(window)
        inputs = apply_mask(tokens, hide[window], split.layout).to(device)
        logits = model(inputs, {key: value[window].to(device) for key, value in condition.items()})
        targets = (hide[window] & _free_mask(legal)).to(device)
        nll, batch_hits, batch_count = masked_nll(
            logits, tokens.to(device), legal.to(device), targets
        )
        total, hits, count = total + nll, hits + batch_hits, count + batch_count
    model.train()
    return total / max(count, 1), hits / max(count, 1)


def _inpaint(
    model: MaskedProgramModel,
    config: MaskedStudyConfig,
    pilot: Any,
    codec: Any,
    layout: SequenceLayout,
    evaluation: _Split,
    condition: dict[str, torch.Tensor],
    marginals: torch.Tensor,
    groups: dict[str, int],
    subgroups: dict[str, int],
) -> dict[str, Any]:
    """Remove one path from each evaluation icon and ask three policies to put it back.

    `drop` leaves the hole, which is what an editor shows before the fill and the
    identity policy of this task. `marginal` fills it from corpus statistics through the
    same grammar-ordered decoder. `model` is the trained completion, greedy so that the
    paired comparison is deterministic; sampled completions are rendered beside it for
    the eye, not the criterion.
    """

    limits = RenderLimits(max_paths=codec.max_paths, timeout_seconds=config.render_timeout_seconds)
    floor = marginal_logits(marginals)
    rows: list[dict[str, Any]] = []
    tiles: list[tuple[str, list[np.ndarray[Any, Any]]]] = []
    exact = 0
    valid = 0
    attempted = 0
    for index in range(min(config.inpaint_icons, len(evaluation.rows))):
        row = evaluation.rows[index]
        tokens = evaluation.tokens[index]
        blocks = path_blocks(tokens, layout)
        if len(blocks) < 2:
            continue  # a one-path icon with its path removed is blank, and blank is not a task
        chooser = np.random.default_rng(config.seed + _INPAINT_SEED + index)
        block = blocks[int(chooser.integers(len(blocks)))]
        clean = _load_program(row, pilot, codec)
        hide = whole_path_mask(tokens, layout, block.path)
        inputs = apply_mask(tokens, hide, layout)
        icon_condition = {
            "group": condition["group"][index : index + 1].to(next(model.parameters()).device),
            "subgroup": condition["subgroup"][index : index + 1].to(
                next(model.parameters()).device
            ),
        }
        attempted += 1

        completions: dict[str, PackedTensorProgram] = {}
        greedy, _ = complete(
            model_logits(model, icon_condition),
            inputs,
            layout,
            greedy=True,
            iterations=config.iterations,
        )
        completions["model"] = unflatten_program(greedy, clean, layout)
        exact += int(bool(torch.equal(greedy[hide], tokens[hide])))
        marginal, _ = complete(floor, inputs, layout, greedy=True, iterations=config.iterations)
        completions["marginal"] = unflatten_program(marginal, clean, layout)
        for sample in range(config.samples):
            sampled, _ = complete(
                model_logits(model, icon_condition),
                inputs,
                layout,
                greedy=False,
                rng=np.random.default_rng(config.seed + _INPAINT_SEED + 1000 * index + sample),
                iterations=config.iterations,
            )
            completions[f"sample_{sample}"] = unflatten_program(sampled, clean, layout)
        for program in completions.values():
            validate_packed_tensor_program(program, codec, pilot.total_segment_slots)
            valid += 1
        dropped = drop_path(clean, block.path, codec, pilot.total_segment_slots)

        images = {"clean": _raster(clean, codec, pilot, limits, config.render_size)}
        images["drop"] = _raster(dropped, codec, pilot, limits, config.render_size)
        for name, program in completions.items():
            images[name] = _raster(program, codec, pilot, limits, config.render_size)
        errors = {
            name: float(cast(float, _similarity(images["clean"], image)["rgba_mae"]))
            for name, image in images.items()
            if name != "clean"
        }
        rows.append(
            {
                "hexcode": row.hexcode,
                "source_path": row.source_path,
                "subgroup": row.subgroup,
                "masked_path": block.path,
                "masked_segments": block.length,
                "active_paths": len(blocks),
                "masked_positions": int(hide.sum()),
                "rgba_mae": errors,
                "exact_reproduction": bool(torch.equal(greedy[hide], tokens[hide])),
            }
        )
        order = (
            ["clean", "drop", "model"]
            + [f"sample_{s}" for s in range(config.samples)]
            + ["marginal"]
        )
        tiles.append((row.hexcode, [images[name] for name in order]))

    if not rows:
        raise OpenMojiPilotError("no evaluation icon has two or more active paths")
    drop = np.array([r["rgba_mae"]["drop"] for r in rows])
    model_error = np.array([r["rgba_mae"]["model"] for r in rows])
    marginal_error = np.array([r["rgba_mae"]["marginal"] for r in rows])
    recovery = (drop - model_error) / np.maximum(drop, 1e-12)
    marginal_recovery = (drop - marginal_error) / np.maximum(drop, 1e-12)
    sheet = _sheet(tiles, config)
    _write_bytes_artifact(config.report_root / "inpainting.jsonl", _jsonl(rows))
    _write_bytes_artifact(config.report_root / SHEET_FILENAME, sheet)
    return {
        "icons": len(rows),
        "attempted": attempted,
        "all_valid": valid == attempted * (2 + config.samples),
        "exact_reproduction_rate": exact / len(rows),
        "decoding": {
            "iterations": config.iterations,
            "greedy_for_criteria": True,
            "samples": config.samples,
        },
        "drop": {"median_rgba_mae": float(np.median(drop))},
        "model": {
            "median_rgba_mae": float(np.median(model_error)),
            "mean_recovery": float(recovery.mean()),
            "median_recovery": float(np.median(recovery)),
            "helped": int((model_error < drop).sum()),
        },
        "marginal": {
            "median_rgba_mae": float(np.median(marginal_error)),
            "mean_recovery": float(marginal_recovery.mean()),
            "median_recovery": float(np.median(marginal_recovery)),
            "helped": int((marginal_error < drop).sum()),
        },
        "paired": {
            "drop_minus_model": _paired(drop - model_error),
            "marginal_minus_model": _paired(marginal_error - model_error),
        },
        "sheet_sha256": hashlib.sha256(sheet).hexdigest(),
        "rows_sha256": hashlib.sha256(_jsonl(rows)).hexdigest(),
    }


def _paired(differences: np.ndarray) -> dict[str, Any]:
    """Mean paired difference with a 95% t-interval. On paired data, compare the pairs."""

    n = len(differences)
    mean = float(differences.mean())
    if n < 2:
        return {
            "n": n,
            "mean": mean,
            "interval": [mean, mean],
            "interval_excludes_zero_above": False,
        }
    half = _t_critical(n - 1) * float(differences.std(ddof=1)) / math.sqrt(n)
    return {
        "n": n,
        "mean": mean,
        "interval": [mean - half, mean + half],
        "interval_excludes_zero_above": mean - half > 0.0,
        "positive": int((differences > 0).sum()),
    }


_T_TABLE = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
    40: 2.021,
    60: 2.000,
    120: 1.980,
}


def _t_critical(df: int) -> float:
    """Two-sided 97.5% Student t quantile, tabulated; conservative between rows."""

    if df in _T_TABLE:
        return _T_TABLE[df]
    lower = max(key for key in _T_TABLE if key < df) if df > 1 else 1
    return _T_TABLE[lower] if df < 120 else 1.960


def _raster(
    program: PackedTensorProgram, codec: Any, pilot: Any, limits: RenderLimits, size: int
) -> np.ndarray[Any, Any]:
    svg = serialize_packed_svg(program, codec, pilot.total_segment_slots)
    _, raster = render_typed_svg_isolated(svg, size, limits)
    return raster


def _sheet(tiles: list[tuple[str, list[np.ndarray[Any, Any]]]], config: MaskedStudyConfig) -> bytes:
    labels = (
        ["clean", "path dropped", "model"]
        + [f"sample {s}" for s in range(config.samples)]
        + ["marginal"]
    )
    tile = config.render_size + 16
    label_width = 150
    header = 30
    canvas = Image.new(
        "RGB", (label_width + tile * len(labels), header + tile * len(tiles)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    for column, label in enumerate(labels):
        draw.text((label_width + tile * column + 4, 8), label, fill="black")
    for row_index, (hexcode, images) in enumerate(tiles):
        y = header + row_index * tile
        draw.text((4, y + tile // 2), hexcode[:22], fill="black")
        for column, raster in enumerate(images):
            rgba = Image.fromarray(raster.astype(np.uint8), "RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            white.alpha_composite(rgba)
            canvas.paste(white.convert("RGB"), (label_width + column * tile + 8, y + 8))
    payload = io.BytesIO()
    canvas.save(payload, format="PNG", optimize=True)
    return payload.getvalue()


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_json(row) for row in rows)


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_masked_study_config(args.config)
    print(json.dumps(run_masked_study(config, args.config), sort_keys=True))


if __name__ == "__main__":
    main()
