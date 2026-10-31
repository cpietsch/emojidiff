"""Greedy float32 SVGs of a run's best.pt for the 130 QuickDraw doodles. Usage: qd_decode.py <run_id> <tag>"""
import json, sys
from pathlib import Path
import numpy as np, torch, yaml
sys.path.insert(0, "/home/dev/workspace/mojidiff/src")
import os; os.chdir("/home/dev/workspace/mojidiff")
from mojidiff.learning.render2svg import CACHE_ROOT, ModelConfig, RenderToProgram, greedy_decode, load_config, load_corpus, _pilot_rows
from mojidiff.learning.openmoji_pilot import _load_program
from mojidiff.learning.autoregressive import unflatten_program
from mojidiff.representation.packed import serialize_packed_svg
run, tag = sys.argv[1], sys.argv[2]
Q = Path("/home/dev/.cache/mojidiff/quickdraw")
record = yaml.safe_load(open(f"runs/{run}/run.yaml"))
config = load_config(Path(record["config"]))
plain, _, layout, pilot, h = load_corpus(config.pilot_config, 144)
template = _load_program(_pilot_rows(pilot)["primary/train"][0], pilot, layout.codec)
device = torch.device("cuda")
state = torch.load(CACHE_ROOT / "runs" / run / "best.pt", map_location=device)
model = RenderToProgram(layout, ModelConfig(**state["config"])).to(device).eval(); model.load_state_dict(state["model"])
images = torch.from_numpy(np.load(Q / "images.npy"))
svgs = []
with torch.no_grad():
    for start in range(0, len(images), 64):
        tokens = greedy_decode(model, images[start:start + 64].to(device))
        for row in tokens.cpu():
            try: svgs.append(serialize_packed_svg(unflatten_program(row, template, layout), layout.codec, layout.total_segment_slots).decode())
            except Exception as e: svgs.append(None)
(Q / f"svgs-{tag}.json").write_text(json.dumps({"run": run, "step": int(state["step"]), "svgs": svgs}))
print(tag, len(svgs), "decoded,", sum(s is None for s in svgs), "failed")
