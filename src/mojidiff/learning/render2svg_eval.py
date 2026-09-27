"""Evaluate a trained render-to-SVG checkpoint the same way for every run.

Training runs score themselves with batched bfloat16 decoding, which is quick and
comparable within the phase. This re-scores a saved checkpoint in float32, where the
batched decoder and the CUDA-graph decoder produce identical programs (tested), so the
quality and the latency it reports describe one and the same decoder:

* pixel error at 72 px against the nearest training icon, paired, with intervals;
* CLIP top-1 on Gate N's 32 validation icons (validation split only);
* single-icon latency through `fast_decode.GraphDecoder`, and batched throughput.

    python -m mojidiff.learning.render2svg_eval --run <run_id> [--split test]

Writes `runs/<run_id>/eval-<split>.json` and `eval-<split>.png`.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np
import torch

from mojidiff.learning.fast_decode import GraphDecoder
from mojidiff.learning.render2svg import (
    CACHE_ROOT,
    REPO_ROOT,
    DecodeStats,
    ModelConfig,
    RenderToProgram,
    _pilot_rows,
    bootstrap_mean_interval,
    clip_retrieval,
    contact_sheet,
    decode_and_score,
    greedy_decode,
    load_config,
    load_corpus,
    nearest_training_icon,
    parameter_count,
)
from mojidiff.learning.telemetry import measure_latency, resource_summary

SPLITS = {"validation": "primary/validation", "test": "primary/test"}


def evaluate_run(run_id: str, split: str, icons: int | None = None) -> dict[str, Any]:
    import yaml

    run_dir = REPO_ROOT / "runs" / run_id
    record = yaml.safe_load((run_dir / "run.yaml").read_text())
    config = load_config(REPO_ROOT / record["config"])
    device = torch.device("cuda")
    plain, _, layout, pilot, dataset_hash = load_corpus(
        config.pilot_config, config.model.image_size
    )
    if dataset_hash != record["dataset"]["cache_sha256"]:
        raise SystemExit("the cached corpus differs from the one this run trained on")
    state = torch.load(CACHE_ROOT / "runs" / run_id / "best.pt", map_location=device)
    model = RenderToProgram(layout, ModelConfig(**state["config"])).to(device).eval()
    model.load_state_dict(state["model"])

    from mojidiff.learning.openmoji_pilot import _load_program, _select_rows

    rows = _pilot_rows(pilot)
    template = _load_program(rows["primary/train"][0], pilot, layout.codec)
    data = plain[SPLITS[split]]
    count = min(icons or len(data.tokens), len(data.tokens))
    data = data.subset(count)
    library = plain["primary/train"]

    started = time.perf_counter()
    errors, tokens, renders, _ = decode_and_score(model, data, template, device, count, bf16=False)
    decode_seconds = time.perf_counter() - started
    nearest, nearest_errors = nearest_training_icon(data.targets, library.targets, device)
    baseline = [float(v) for v in nearest_errors]
    reduction = [b - m for m, b in zip(errors, baseline, strict=True)]
    result: dict[str, Any] = {
        "run_id": run_id,
        "split": SPLITS[split],
        "icons": count,
        "precision": "float32",
        "selected_step": int(state["step"]),
        "model_parameters": parameter_count(model),
        "model_pixel_error": bootstrap_mean_interval(errors),
        "model_pixel_error_median": float(np.median(errors)),
        "nearest_training_icon_pixel_error": bootstrap_mean_interval(baseline),
        "model_minus_baseline_error_reduction": bootstrap_mean_interval(reduction),
        "icons_model_beats_baseline": sum(1 for d in reduction if d > 0),
        "exact_program_rate": float(
            np.mean([torch.equal(tokens[i], data.tokens[i]) for i in range(count)])
        ),
        "rendered_rate": sum(1 for r in renders if r is not None) / count,
        "batched_decode_seconds": decode_seconds,
    }

    if split == "validation":
        selected = _select_rows(rows["primary/validation"], 32, pilot.seed + 1)
        full = plain["primary/validation"]
        positions = {h: i for i, h in enumerate(full.hexcodes)}
        chosen = [positions[row.hexcode] for row in selected]
        from mojidiff.learning.render2svg import _programs_to_renders

        clip_tokens = greedy_decode(model, full.images[chosen].to(device))
        clip_renders = _programs_to_renders(clip_tokens, layout, template)
        references = [full.targets[i].numpy() for i in chosen]
        clip_nearest, _ = nearest_training_icon(full.targets[chosen], library.targets, device)
        result["clip_retrieval"] = {
            "model_greedy": clip_retrieval(clip_renders, references, "cuda"),
            "nearest_training_icon": clip_retrieval(
                [library.targets[int(i)].numpy() for i in clip_nearest], references, "cuda"
            ),
            "gate_n_omnisvg_zero_shot_top1": 39 / 64,
        }

    graph = GraphDecoder(model, dtype=torch.float32)
    agreement = sum(
        int(torch.equal(graph.decode(data.images[i]), tokens[i : i + 1])) for i in range(8)
    )
    stats = DecodeStats()
    graph.decode(data.images[0], stats=stats)
    latency = measure_latency(lambda: graph.decode(data.images[0]), device=device, repeats=10)
    many = data.images[: min(64, count)].to(device)
    batched = measure_latency(
        lambda: greedy_decode(model, many), device=device, icons_per_call=len(many), repeats=3
    )
    resource = resource_summary(device, train_seconds=None, latency=latency).as_record()
    resource.update(
        train_seconds_null_reason="evaluation of a trained checkpoint",
        decoder="fast_decode.GraphDecoder, float32, batch 1",
        graph_matches_batched_programs=f"{agreement} of 8",
        decoder_calls_first_icon=stats.model_calls,
        batched_ms_per_icon=batched.median_ms_per_icon,
        batched_icons_per_call=batched.icons_per_call,
        excludes="rasterising the output SVG",
    )
    result["resource"] = resource

    sheet_rows = [
        [data.targets[i].numpy(), renders[i], library.targets[int(nearest[i])].numpy()]
        for i in range(min(24, count))
    ]
    (run_dir / f"eval-{split}.png").write_bytes(contact_sheet(sheet_rows))
    (run_dir / f"eval-{split}.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def latency_only(run_id: str, icons: int = 16) -> dict[str, Any]:
    """Graph-decoder latency on `icons` validation icons, one at a time, on an idle GPU.

    Quality does not depend on load; latency does. This is the measurement to quote,
    run when nothing else holds the GPU, and it records whether anything did.
    """

    import subprocess

    import yaml

    run_dir = REPO_ROOT / "runs" / run_id
    record = yaml.safe_load((run_dir / "run.yaml").read_text())
    config = load_config(REPO_ROOT / record["config"])
    device = torch.device("cuda")
    plain, _, layout, _, _ = load_corpus(config.pilot_config, config.model.image_size)
    state = torch.load(CACHE_ROOT / "runs" / run_id / "best.pt", map_location=device)
    model = RenderToProgram(layout, ModelConfig(**state["config"])).to(device).eval()
    model.load_state_dict(state["model"])
    graph = GraphDecoder(model, dtype=torch.float32)
    data = plain["primary/validation"]
    others = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.split()
    per_icon: list[float] = []
    calls: list[int] = []
    for index in range(icons):
        stats = DecodeStats()
        graph.decode(data.images[index], stats=stats)
        report = measure_latency(
            lambda index=index: graph.decode(data.images[index]),  # type: ignore[misc]
            device=device,
            warmup=1,
            repeats=3,
        )
        per_icon.append(report.median_ms_per_icon)
        calls.append(stats.model_calls)
    result = {
        "run_id": run_id,
        "decoder": "fast_decode.GraphDecoder, float32, batch 1",
        "icons": icons,
        "median_ms_per_icon": float(np.median(per_icon)),
        "p90_ms_per_icon": float(np.quantile(per_icon, 0.9)),
        "median_decoder_calls": float(np.median(calls)),
        "ms_per_decoder_call": sum(per_icon) / max(sum(calls), 1),
        "gpu_processes_during_measurement": len(others),
        "device": torch.cuda.get_device_name(device),
        "excludes": "rasterising the output SVG",
    }
    (run_dir / "latency.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--split", choices=sorted(SPLITS), default="validation")
    parser.add_argument("--icons", type=int, default=None)
    parser.add_argument("--latency-only", action="store_true")
    args = parser.parse_args()
    if args.latency_only:
        print(json.dumps(latency_only(args.run, args.icons or 16)))
        return
    result = evaluate_run(args.run, args.split, args.icons)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
