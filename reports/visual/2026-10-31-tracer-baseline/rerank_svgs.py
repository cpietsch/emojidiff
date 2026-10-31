"""Best-of-8 SVGs of the final model on all 339 validation icons, as render2svg_eval --rerank 8
decodes them (same rerank_decode, temperature 0.7, seed = icon index), saved for visual grading."""
import json, sys
from pathlib import Path
import torch
import yaml
sys.path.insert(0, "/home/dev/workspace/mojidiff/src")
from mojidiff.learning.render2svg import (CACHE_ROOT, ModelConfig, RenderToProgram, load_config, load_corpus,
                                          pixel_error, rerank_decode, render_trusted_rgb)
from mojidiff.learning.render2svg_eval import _pilot_rows
from mojidiff.learning.openmoji_pilot import _load_program
from mojidiff.learning.autoregressive import unflatten_program
from mojidiff.representation.packed import serialize_packed_svg

RUN = "r2s-cur-d02-s120k-seed2-53b3108-46294452-47646604"
OUT = Path("/home/dev/.cache/mojidiff/visual-review-2026-10-31/rerank8")
repo = Path("/home/dev/workspace/mojidiff")
record = yaml.safe_load((repo / "runs" / RUN / "run.yaml").read_text())
config = load_config(repo / record["config"])
plain, _, layout, pilot, dataset_hash = load_corpus(config.pilot_config, config.model.image_size)
assert dataset_hash == record["dataset"]["cache_sha256"]
device = torch.device("cuda")
state = torch.load(CACHE_ROOT / "runs" / RUN / "best.pt", map_location=device)
model = RenderToProgram(layout, ModelConfig(**state["config"])).to(device).eval()
model.load_state_dict(state["model"])
template = _load_program(_pilot_rows(pilot)["primary/train"][0], pilot, layout.codec)
data = plain["primary/validation"]
rows = []
for index in range(len(data.tokens)):
    row, info = rerank_decode(model, data.images[index].to(device), template, candidates=8, temperature=0.7, seed=index)
    program = unflatten_program(row.cpu()[0] if row.dim() > 1 else row.cpu(), template, layout)
    svg = serialize_packed_svg(program, layout.codec, layout.total_segment_slots)
    target = data.targets[index].numpy()
    error = pixel_error(render_trusted_rgb(svg, target.shape[0]), target)
    rows.append({"index": index, "hexcode": data.hexcodes[index], "pixel_error": float(error), "chosen": int(info["chosen"]), "svg": svg.decode() if isinstance(svg, bytes) else svg})
    if index % 50 == 0:
        print(index, f"{error:.4f}", flush=True)
(OUT / "rerank8.json").write_text(json.dumps(rows))
import numpy as np
print("mean pixel error", np.mean([r["pixel_error"] for r in rows]), "chose a sample", np.mean([r["chosen"] > 0 for r in rows]))
