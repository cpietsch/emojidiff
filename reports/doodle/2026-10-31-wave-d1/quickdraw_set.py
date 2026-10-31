"""Real doodles for the doodle model: the first 5 recognised QuickDraw drawings of 26 categories
(CC BY 4.0, Google Creative Lab), drawn black with a 4 px round pen and fitted into OpenMoji's
22-122 px content frame at 144 px - as the demo's drawing pad would."""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
sys.path.insert(0, "/home/dev/workspace/mojidiff/src")
Q = Path("/home/dev/.cache/mojidiff/quickdraw")
EMOJI = {"umbrella": "2602", "apple": "1F34E", "pizza": "1F355", "bird": "1F426", "hand": "270B", "camera": "1F4F7",
         "airplane": "2708", "mushroom": "1F344", "rabbit": "1F430", "teapot": "1FAD6", "violin": "1F3BB", "hedgehog": "1F994",
         "stop sign": "1F6D1", "smiley face": "1F600", "sun": "2600", "moon": "1F319", "star": "2B50", "cloud": "2601",
         "house": "1F3E0", "cat": "1F431", "fish": "1F41F", "tree": "1F333", "key": "1F511", "light bulb": "1F4A1",
         "snowman": "2603", "crown": "1F451"}
meta_all = {e["hexcode"]: e for e in json.load(open("/home/dev/workspace/mojidiff/data/raw/openmoji/17.0.0/data/openmoji.json"))}
import yaml
from mojidiff.learning.render2svg import load_corpus, load_config
RUN = "r2s-cur-d02-s120k-seed2-53b3108-46294452-47646604"
record = yaml.safe_load(open(f"/home/dev/workspace/mojidiff/runs/{RUN}/run.yaml"))
import os; os.chdir("/home/dev/workspace/mojidiff")
plain, _, layout, pilot, h = load_corpus(load_config(Path(record["config"])).pilot_config, 144)
where = {hx: split for split in ("primary/train", "primary/validation", "primary/test") for hx in plain[split].hexcodes}

def draw(strokes, size=144, lo=22, hi=122, pen=4):
    xs = np.concatenate([s[0] for s in strokes]); ys = np.concatenate([s[1] for s in strokes])
    w, h = xs.max() - xs.min(), ys.max() - ys.min()
    scale = (hi - lo) / max(w, h, 1)
    ox = lo + ((hi - lo) - w * scale) / 2 - xs.min() * scale; oy = lo + ((hi - lo) - h * scale) / 2 - ys.min() * scale
    im = Image.new("RGB", (size * 4, size * 4), "white"); d = ImageDraw.Draw(im)  # 4x supersampled
    for sx, sy in strokes:
        pts = [((x * scale + ox) * 4, (y * scale + oy) * 4) for x, y in zip(sx, sy)]
        if len(pts) == 1: pts = pts * 2
        d.line(pts, fill="black", width=pen * 4, joint="curve")
        for x, y in (pts[0], pts[-1]): d.ellipse([x - pen * 2, y - pen * 2, x + pen * 2, y + pen * 2], fill="black")
    return np.asarray(im.resize((size, size), Image.Resampling.LANCZOS))

images, rows = [], []
for cat, hexcode in EMOJI.items():
    found = 0
    for line in open(Q / "raw" / f"{cat.replace(' ', '_')}.ndjson"):
        try: r = json.loads(line)
        except json.JSONDecodeError: break  # the range request cut the last line
        if not r["recognized"]: continue
        images.append(draw(r["drawing"]))
        rows.append({"category": cat, "key_id": r["key_id"], "hexcode": hexcode, "annotation": meta_all[hexcode]["annotation"],
                     "icon_split": where.get(hexcode, "not in corpus")})
        found += 1
        if found == 5: break
np.save(Q / "images.npy", np.stack(images)); (Q / "rows.json").write_text(json.dumps(rows, indent=1))
import collections
print(len(images), "drawings;", collections.Counter(r["icon_split"] for r in rows))
