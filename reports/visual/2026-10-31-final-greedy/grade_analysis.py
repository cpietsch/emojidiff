"""Join the visual grades with pixel error and decoder calls; agreement; by-grade spot-check sheets.
Usage: python grade_analysis.py <workflow result json>"""
import collections
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = Path("/home/dev/.cache/mojidiff/visual-review-2026-10-31")
result = json.loads(Path(sys.argv[1]).read_text())
key = json.loads((OUT / "key.json").read_text())
sheets = json.loads((OUT / "sheets.json").read_text())
grades = {g["id"]: g for g in result["icons"]}
missing = sorted(set(key) - set(grades))
extra = sorted(set(grades) - set(key))
print(f"graded {len(grades)} of {len(key)}; missing {missing[:10]}{'...' if len(missing) > 10 else ''}; unknown ids {extra[:5]}")
ids = [i for i in key if i in grades]
g = np.array([grades[i]["grade"] for i in ids])
pe = np.array([key[i]["pixel_error"] for i in ids])
calls = np.array([key[i]["calls"] for i in ids])
counts = collections.Counter(g.tolist())
print("grade distribution:", {k: f"{counts[k]} ({counts[k] / len(g):.0%})" for k in (3, 2, 1, 0)})
print(f"recognisable (2-3): {(g >= 2).mean():.0%}; not recognisable (0-1): {(g <= 1).mean():.0%}")
for k in (3, 2, 1, 0):
    m = g == k
    if m.any():
        print(f"  grade {k}: pixel error median {np.median(pe[m]):.3f} (IQR {np.percentile(pe[m], 25):.3f}-{np.percentile(pe[m], 75):.3f}), calls median {np.median(calls[m]):.0f}")


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


print(f"Spearman(grade, pixel error) {spearman(g, pe):+.2f}; Spearman(grade, calls) {spearman(g, calls):+.2f}")
for lo, hi in [(0, 300), (300, 450), (450, 600), (600, 2000)]:
    m = (calls >= lo) & (calls < hi)
    print(f"  calls {lo}-{hi}: n={m.sum()}, recognisable {(g[m] >= 2).mean():.0%}, faithful {(g[m] == 3).mean():.0%}")
defects = collections.Counter(d for i in ids for d in grades[i]["defects"] if d != "none")
print("defects:", defects.most_common())


def agreement(a, b, label):
    common = sorted(set(a) & set(b))
    x = np.array([a[i] for i in common]); y = np.array([b[i] for i in common])
    exact = (x == y).mean(); within = (abs(x - y) <= 1).mean()
    rec = ((x >= 2) == (y >= 2)).mean()
    print(f"{label}: n={len(common)}, exact {exact:.0%}, within 1 {within:.0%}, same recognisable/not {rec:.0%}, mean {x.mean():.2f} vs {y.mean():.2f}")


if result.get("blind"):
    agreement({i: grades[i]["grade"] for i in grades}, {b["id"]: b["grade"] for b in result["blind"]}, "grader vs blind re-grade (sheets 3+11)")
mine = json.loads((OUT / "my-grades-sheets-00-07.json").read_text())
agreement(mine, {i: grades[i]["grade"] for i in grades}, "my grades vs graders (sheets 0+7)")

# Spot-check sheets: 12 random icons per grade, input | model, with grade and pixel error.
images = None
sys.path.insert(0, str(Path(__file__).parent))
import sheets as base  # noqa: E402  (re-renders from the bundle)

font = ImageFont.load_default(size=14)
rng = random.Random(1)
for k in (3, 2, 1, 0):
    pool = [i for i in ids if grades[i]["grade"] == k]
    pick = rng.sample(pool, min(12, len(pool)))
    S, cols = 144, 4
    canvas = Image.new("RGB", (cols * (2 * S + 24) + 10, ((len(pick) + cols - 1) // cols) * (S + 34) + 10), (236, 236, 232))
    draw = ImageDraw.Draw(canvas)
    for j, i in enumerate(pick):
        x, y = 10 + (j % cols) * (2 * S + 24), 10 + (j // cols) * (S + 34)
        index = key[i]["index"]
        draw.text((x, y), f"{i} grade {k}  px {key[i]['pixel_error']:.3f}", fill=(40, 40, 40), font=font)
        canvas.paste(Image.fromarray(base.images[index]), (x, y + 20))
        canvas.paste(base.render(base.ref["icons"][index]["svg"]), (x + S + 6, y + 20))
    canvas.save(OUT / f"by-grade-{k}.png")
merged = {i: {**key[i], **grades[i]} for i in ids}
(OUT / "grades.json").write_text(json.dumps(merged, indent=1))
print("wrote", OUT / "grades.json", "and by-grade-*.png")
