#!/usr/bin/env python3
"""Direction 3, first evidence: is OmniSVG a better source of programs than the student?

Distilling a big model into a small one pays only if the teacher's outputs are better
than what the student already produces, on the inputs it would be taught with. For the
render-to-SVG task this compares, on Gate N's 32 validation icons (the same icons, the
same 72 px references): OmniSVG 4B's saved image-conditioned drawings from Gate N
(zero-shot, best-of-6, best-of-12) against the student's greedy and best-of-8 decodes.
Reported per system: pixel error, share of outputs the codec accepts, seconds per
drawing as measured.

    python scripts/distill_probe.py --student runs/<run_id>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mojidiff.learning.omnisvg_study import render_raw  # noqa: E402
from mojidiff.learning.openmoji_pilot import _load_program, _select_rows  # noqa: E402
from mojidiff.learning.render2svg import (  # noqa: E402
    CACHE_ROOT,
    ModelConfig,
    RenderToProgram,
    _pilot_rows,
    _programs_to_renders,
    bootstrap_mean_interval,
    greedy_decode,
    load_corpus,
    pixel_error,
    rerank_decode,
)

TEACHERS = {
    "omnisvg zero-shot": "omnisvg-n1-image-control",
    "omnisvg best of 6": "omnisvg-n1-image-control-rerank",
    "omnisvg best of 12": "omnisvg-n1-image-control-rerank12",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", required=True, help="run id of a render2svg run")
    parser.add_argument(
        "--out", type=Path, default=_REPO_ROOT / "reports/learning/distill-probe-v1"
    )
    args = parser.parse_args()
    plain, _, layout, pilot, _ = load_corpus(
        _REPO_ROOT / "configs/learning/openmoji-g1-geometric-gate-v16.yaml", 144
    )
    rows = _pilot_rows(pilot)
    selected = _select_rows(rows["primary/validation"], 32, pilot.seed + 1)
    validation = plain["primary/validation"]
    index = {h: i for i, h in enumerate(validation.hexcodes)}
    chosen = [index[row.hexcode] for row in selected]
    references = {row.hexcode: validation.targets[index[row.hexcode]].numpy() for row in selected}
    report: dict[str, object] = {"icons": len(selected), "systems": {}}

    for name, directory in TEACHERS.items():
        path = _REPO_ROOT / "reports/learning" / directory
        drawings = [json.loads(line) for line in (path / "drawings.jsonl").open()]
        summary = json.loads((path / "summary.json").read_text())
        errors = []
        for drawing in drawings:
            reference = references.get(drawing["hexcode"])
            if reference is None:
                continue
            try:
                image = np.asarray(render_raw(drawing["svg"], 72)) if drawing.get("svg") else None
            except Exception:  # noqa: BLE001 - an unrenderable drawing scores as a miss
                image = None
            errors.append(pixel_error(image, reference))
        report["systems"][name] = {  # type: ignore[index]
            "drawings": len(errors),
            "pixel_error": bootstrap_mean_interval(errors),
            "codec_valid_rate": summary["codec_valid_rate"],
            "clip_top1": summary["reference_rank"]["top1_rate"],
            "seconds_per_drawing": summary["timing"]["seconds_per_drawing"],
            "parameters": "about 4B",
        }

    device = torch.device("cuda")
    state = torch.load(CACHE_ROOT / "runs" / args.student / "best.pt", map_location=device)
    model = RenderToProgram(layout, ModelConfig(**state["config"])).to(device).eval()
    model.load_state_dict(state["model"])
    template = _load_program(rows["primary/train"][0], pilot, layout.codec)
    images = validation.images[chosen].to(device)
    import time

    started = time.perf_counter()
    greedy = greedy_decode(model, images)
    greedy_seconds = (time.perf_counter() - started) / len(chosen)
    started = time.perf_counter()
    reranked = torch.cat(
        [
            rerank_decode(model, images[i], template, candidates=8, temperature=0.7, seed=i)[0]
            for i in range(len(chosen))
        ]
    )
    rerank_seconds = (time.perf_counter() - started) / len(chosen)
    for name, tokens, seconds in (
        ("student greedy", greedy, greedy_seconds),
        ("student best of 8", reranked, rerank_seconds),
    ):
        renders = _programs_to_renders(tokens.cpu(), layout, template)
        errors = [
            pixel_error(render, references[row.hexcode])
            for render, row in zip(renders, selected, strict=True)
        ]
        report["systems"][name] = {  # type: ignore[index]
            "drawings": len(errors),
            "pixel_error": bootstrap_mean_interval(errors),
            "codec_valid_rate": 1.0,
            "seconds_per_drawing": seconds,
            "seconds_note": "batched reference decoder, float32, whatever else was on the GPU",
            "parameters": sum(p.numel() for p in model.parameters()),
        }
    report["student_run"] = args.student
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
