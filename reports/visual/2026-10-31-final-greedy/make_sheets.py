"""Contact sheets for a visual review: validation render | model greedy SVG, neutral ids only."""
import io, json, random
from pathlib import Path
import numpy as np
import cairosvg
from PIL import Image, ImageDraw, ImageFont

RUN = "r2s-cur-d02-s120k-seed2-53b3108-46294452-47646604"
B = Path(f"/home/dev/.cache/mojidiff/web-reference/{RUN}/validation-339")
OUT = Path("/home/dev/.cache/mojidiff/visual-review-2026-10-31")
ref = json.loads((B / "reference.json").read_text())
images = np.fromfile(B / "images.u8", dtype=np.uint8).reshape(-1, 144, 144, 3)
per_icon = [json.loads(l) for l in open(f"/home/dev/workspace/mojidiff/runs/{RUN}/per_icon.jsonl")]
assert [p["hexcode"] for p in per_icon] == [i["hexcode"] for i in ref["icons"]]

def render(svg: str, size: int = 144) -> Image.Image:
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
    rgba = Image.open(io.BytesIO(png)).convert("RGBA")
    white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(white, rgba).convert("RGB")

order = list(range(len(ref["icons"])))
random.Random(0).shuffle(order)
ids = {k: f"#{n:03d}" for n, k in enumerate(order)}  # neutral id = position in the shuffled order
font = ImageFont.load_default(size=15)
S, GAP, LABEL, COLS, ROWS = 144, 6, 22, 4, 6
cell_w, cell_h = 2 * S + GAP + 18, S + LABEL + 10
manifest = []
for sheet, start in enumerate(range(0, len(order), COLS * ROWS)):
    chunk = order[start:start + COLS * ROWS]
    canvas = Image.new("RGB", (COLS * cell_w + 10, ((len(chunk) + COLS - 1) // COLS) * cell_h + 10), (236, 236, 232))
    draw = ImageDraw.Draw(canvas)
    for j, k in enumerate(chunk):
        x, y = 10 + (j % COLS) * cell_w, 10 + (j // COLS) * cell_h
        draw.text((x, y), f"{ids[k]}   left: input   right: model", fill=(40, 40, 40), font=font)
        canvas.paste(Image.fromarray(images[k]), (x, y + LABEL))
        canvas.paste(render(ref["icons"][k]["svg"]), (x + S + GAP, y + LABEL))
        draw.rectangle([x - 1, y + LABEL - 1, x + 2 * S + GAP, y + LABEL + S], outline=(180, 180, 175))
        draw.line([x + S + GAP // 2, y + LABEL, x + S + GAP // 2, y + LABEL + S], fill=(150, 150, 150), width=2)
    path = OUT / "sheets" / f"sheet-{sheet:02d}.png"
    canvas.save(path)
    manifest.append({"sheet": path.name, "ids": [ids[k] for k in chunk]})
key = {ids[k]: {"index": k, "hexcode": ref["icons"][k]["hexcode"], "pixel_error": per_icon[k]["model_pixel_error"],
                "retrieval_error": per_icon[k]["baseline_pixel_error"], "calls": ref["icons"][k]["model_calls"]} for k in order}
(OUT / "key.json").write_text(json.dumps(key, indent=1))
(OUT / "sheets.json").write_text(json.dumps(manifest, indent=1))
print(len(manifest), "sheets;", sum(len(m["ids"]) for m in manifest), "icons")
