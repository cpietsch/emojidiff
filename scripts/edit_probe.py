#!/usr/bin/env python3
"""Edit pixels, re-vectorise: Gate N's edit test, run on the small render-to-SVG model.

Same 32 validation icons, same three exact edits (recolour the commonest non-black,
non-white fill to red; erase the middle path; move the last path 8 units right), same
scores as `learning/omnisvg_edit.py`: an edit is reflected when the re-vectorised
drawing is closer in CLIP space to the edited render than to the original, and the
edited icon should rank first among all edited renders of that kind. Adds pixel error
to the edited render.

    python scripts/edit_probe.py --run <run_id> [--rerank 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

from mojidiff.learning.omnisvg_edit import edited_program  # noqa: E402
from mojidiff.learning.omnisvg_study import Clip  # noqa: E402
from mojidiff.learning.openmoji_pilot import _load_program, _select_rows  # noqa: E402
from mojidiff.learning.render2svg import (  # noqa: E402
    CACHE_ROOT,
    ModelConfig,
    RenderToProgram,
    _pilot_rows,
    _programs_to_renders,
    greedy_decode,
    load_corpus,
    pixel_error,
    render_trusted_rgb,
    rerank_decode,
)
from mojidiff.representation.packed import serialize_packed_svg  # noqa: E402

EDITS = {"recolour": {"target": "#ea5a47"}, "erase": {}, "move": {"dx": 8, "dy": 0}}
GATE_N = {"omnisvg zero-shot": (0.699, 0.548, 19.5), "omnisvg best of 6": (0.699, 0.710, 114.7)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True)
    parser.add_argument("--rerank", type=int, default=0)
    args = parser.parse_args()
    plain, _, layout, pilot, _ = load_corpus(
        _REPO_ROOT / "configs/learning/openmoji-g1-geometric-gate-v16.yaml", 144
    )
    codec, slots = layout.codec, layout.total_segment_slots
    rows = _pilot_rows(pilot)
    selected = _select_rows(rows["primary/validation"], 32, pilot.seed + 1)
    originals = [_load_program(row, pilot, codec) for row in selected]
    template = originals[0]
    device = torch.device("cuda")
    state = torch.load(CACHE_ROOT / "runs" / args.run / "best.pt", map_location=device)
    model = RenderToProgram(layout, ModelConfig(**state["config"])).to(device).eval()
    model.load_state_dict(state["model"])
    clip = Clip("cuda")

    def render(program: object, size: int) -> np.ndarray:
        return render_trusted_rgb(serialize_packed_svg(program, codec, slots), size)  # type: ignore[arg-type]

    original_renders = [render(p, 72) for p in originals]
    original_features = clip.image([Image.fromarray(r) for r in original_renders])
    summary: dict[str, object] = {"run": args.run, "rerank": args.rerank, "edits": {}}
    all_reflected: list[bool] = []
    all_top1: list[bool] = []
    seconds = 0.0
    for kind, settings in EDITS.items():
        targets = []
        for index, program in enumerate(originals):
            try:
                edited, _ = edited_program(kind, program, codec, slots, settings)
            except ValueError:
                continue
            targets.append((index, edited))
        edited_renders = [render(e, 72) for _, e in targets]
        edited_features = clip.image([Image.fromarray(r) for r in edited_renders])
        reflected: list[bool] = []
        top1: list[bool] = []
        errors: list[float] = []
        for position, (index, edited) in enumerate(targets):
            condition = torch.from_numpy(render(edited, 144)).to(device)
            clock = time.perf_counter()
            if args.rerank > 1:
                tokens, _ = rerank_decode(
                    model,
                    condition,
                    template,
                    candidates=args.rerank,
                    temperature=0.7,
                    seed=index,
                )
            else:
                tokens = greedy_decode(model, condition[None])
            seconds += time.perf_counter() - clock
            image = _programs_to_renders(tokens.cpu(), layout, template)[0]
            if image is None:
                reflected.append(False)
                top1.append(False)
                errors.append(1.0)
                continue
            feature = clip.image([Image.fromarray(image)])
            to_edited = (feature @ edited_features.T)[0]
            own_edited = float(to_edited[position])
            own_original = float((feature @ original_features[index : index + 1].T)[0, 0])
            reflected.append(own_edited > own_original)
            top1.append(int((to_edited > own_edited).sum()) == 0)
            errors.append(pixel_error(image, edited_renders[position]))
        summary["edits"][kind] = {  # type: ignore[index]
            "icons": len(targets),
            "edit_reflected_rate": float(np.mean(reflected)),
            "edited_top1_rate": float(np.mean(top1)),
            "pixel_error_to_edited": float(np.mean(errors)),
        }
        all_reflected += reflected
        all_top1 += top1
    summary["edit_reflected_rate"] = float(np.mean(all_reflected))
    summary["edited_top1_rate"] = float(np.mean(all_top1))
    summary["drawings"] = len(all_reflected)
    summary["seconds_per_drawing"] = seconds / max(len(all_reflected), 1)
    summary["seconds_note"] = "batched reference decoder, float32, GPU shared with training"
    summary["gate_n_omnisvg"] = {
        name: {"edit_reflected_rate": a, "edited_top1_rate": b, "seconds_per_drawing": c}
        for name, (a, b, c) in GATE_N.items()
    }
    out = _REPO_ROOT / "reports/learning/edit-probe-v1"
    out.mkdir(parents=True, exist_ok=True)
    name = f"{args.run}{'-rerank' + str(args.rerank) if args.rerank > 1 else ''}.json"
    (out / name).write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
