"""Score transcriber checkpoints on degraded ("doodle") validation renders.

    python -m mojidiff.learning.doodle_eval --run <run_id> [--run ...] [--seed 1] [--kinds ...]

Each kind in `doodle.EVAL_KINDS` degrades the 339 validation renders once, reproducibly per
icon (`doodle.degrade_images`, seed 1 by default; training runs select checkpoints on seed 0).
Every run's best.pt decodes the degraded renders greedily in float32 and is scored against the
clean 72 px targets with `render2svg.pixel_error`. The zero-parameter reference is the
degraded input itself at 72 px (`input_as_output`): what a perfectly faithful tracer returns.
Writes `runs/<run_id>/eval-doodle.json` per run and the degraded sets (uint8 npz, with the
per-icon parameters) under `CACHE_ROOT/doodle-eval/seed<seed>/` for tracer and visual checks.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from mojidiff.learning.doodle import EVAL_KINDS, degrade_images
from mojidiff.learning.render2svg import (
    CACHE_ROOT,
    REPO_ROOT,
    ModelConfig,
    RenderToProgram,
    _pilot_rows,
    bootstrap_mean_interval,
    decode_and_score,
    load_config,
    load_corpus,
    pixel_error,
)

SPLIT = "primary/validation"


def downsample(image: np.ndarray, size: int) -> np.ndarray:
    """The degraded input at the target size, area-averaged: the input-as-output policy."""

    return np.asarray(Image.fromarray(image).resize((size, size), Image.Resampling.BOX))


def degraded_sets(
    images: np.ndarray, hexcodes: list[str], kinds: tuple[str, ...], seed: int, out: Path
) -> dict[str, np.ndarray]:
    """Build (or reuse) one degraded copy of `images` per kind, saved under `out`."""

    out.mkdir(parents=True, exist_ok=True)
    sets: dict[str, np.ndarray] = {}
    for kind in kinds:
        path = out / f"{kind}.npz"
        if path.exists():
            sets[kind] = np.load(path)["images"]
            continue
        degraded, params = degrade_images(images, hexcodes, (kind,), seed)
        np.savez_compressed(path, images=degraded, hexcodes=np.array(hexcodes))
        (out / f"{kind}.params.json").write_text(json.dumps(params))
        sets[kind] = degraded
    return sets


def evaluate(runs: list[str], kinds: tuple[str, ...], seed: int) -> dict[str, Any]:
    record = yaml.safe_load((REPO_ROOT / "runs" / runs[0] / "run.yaml").read_text())
    config = load_config(REPO_ROOT / record["config"])
    plain, _, layout, pilot, dataset_hash = load_corpus(
        config.pilot_config, config.model.image_size
    )
    from mojidiff.learning.openmoji_pilot import _load_program

    template = _load_program(_pilot_rows(pilot)["primary/train"][0], pilot, layout.codec)
    data = plain[SPLIT]
    targets = data.targets.numpy()
    out_dir = CACHE_ROOT / "doodle-eval" / f"seed{seed}"
    sets = degraded_sets(data.images.numpy(), list(data.hexcodes), kinds, seed, out_dir)
    reference = {
        kind: [
            pixel_error(downsample(image, targets.shape[1]), target)
            for image, target in zip(sets[kind], targets, strict=True)
        ]
        for kind in kinds
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary: dict[str, Any] = {"seed": seed, "split": SPLIT, "dataset": dataset_hash, "runs": {}}
    for run_id in runs:
        run_record = yaml.safe_load((REPO_ROOT / "runs" / run_id / "run.yaml").read_text())
        if run_record["dataset"]["cache_sha256"] != dataset_hash:
            raise SystemExit(f"{run_id} trained on another corpus cache")
        state = torch.load(CACHE_ROOT / "runs" / run_id / "best.pt", map_location=device)
        model = RenderToProgram(layout, ModelConfig(**state["config"])).to(device).eval()
        model.load_state_dict(state["model"])
        result: dict[str, Any] = {
            "run_id": run_id,
            "checkpoint_step": int(state["step"]),
            "seed": seed,
            "split": SPLIT,
            "icons": len(targets),
            "precision": "float32",
            "decoder": "render2svg.greedy_decode via decode_and_score, batch 64",
            "kinds": {},
        }
        for kind in kinds:
            started = time.perf_counter()
            split = type(data)(
                hexcodes=data.hexcodes,
                tokens=data.tokens,
                masks=data.masks,
                images=torch.from_numpy(sets[kind]),
                targets=data.targets,
                shape=data.shape,
            )
            errors, _, _, _ = decode_and_score(
                model, split, template, device, len(targets), bf16=False
            )
            result["kinds"][kind] = {
                "model_pixel_error": bootstrap_mean_interval(errors),
                "input_as_output_pixel_error": bootstrap_mean_interval(reference[kind]),
                "per_icon": [float(e) for e in errors],
                "input_as_output_per_icon": [float(e) for e in reference[kind]],
                "seconds": time.perf_counter() - started,
            }
            print(
                json.dumps(
                    {
                        "run": run_id,
                        "kind": kind,
                        "model": float(np.mean(errors)),
                        "input_as_output": float(np.mean(reference[kind])),
                    }
                ),
                flush=True,
            )
        (REPO_ROOT / "runs" / run_id / "eval-doodle.json").write_text(json.dumps(result) + "\n")
        summary["runs"][run_id] = {k: v["model_pixel_error"] for k, v in result["kinds"].items()}
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--run", action="append", required=True, help="run id (repeatable)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--kinds", nargs="+", default=list(EVAL_KINDS))
    args = parser.parse_args()
    print(json.dumps(evaluate(args.run, tuple(args.kinds), args.seed)))


if __name__ == "__main__":
    main()
