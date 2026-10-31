"""Contact sheets and GIFs from rendered frames: one row per icon, input then N frames.

    python sheet.py <omniprobe dir> <frames dir> <out dir> <mode> [N=8] [per sheet=15]
"""
import json, sys
from pathlib import Path
from PIL import Image, ImageDraw

P, frames, out, mode = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
N = int(sys.argv[5]) if len(sys.argv) > 5 else 8
PER = int(sys.argv[6]) if len(sys.argv) > 6 else 15
C, LABEL = 160, 150
out.mkdir(parents=True, exist_ok=True)
items = [json.loads(line) for line in open(P / "items.jsonl")]
summary = json.loads((frames / "render-summary.json").read_text())
blank = Image.new("RGB", (C, C), (235, 235, 235))
for start in range(0, len(items), PER):
    chunk = items[start:start + PER]
    sheet = Image.new("RGB", (LABEL + C * (N + 1), C * len(chunk)), "white")
    draw = ImageDraw.Draw(sheet)
    for r, it in enumerate(chunk):
        name = f"{it['hexcode']}-{mode}"
        src = Image.open(P / "images" / f"{it['hexcode']}.png").convert("RGBA")
        bg = Image.new("RGBA", src.size, "white"); bg.alpha_composite(src)
        sheet.paste(bg.convert("RGB").resize((C, C)), (LABEL, r * C))
        ok = summary.get(name, {}).get("ok")
        draw.text((6, r * C + 6), f"{it['hexcode']}\n{it['annotation'][:22]}\n{'' if ok else 'NO OUTPUT'}",
                  fill="black")
        gif = []
        for i in range(N):
            f = frames / f"{name}-{i}.png"
            im = Image.open(f).convert("RGB").resize((C, C)) if f.exists() else blank
            sheet.paste(im, (LABEL + C * (i + 1), r * C))
            gif.append(im)
        if ok:
            gif[0].save(out / f"{name}.gif", save_all=True, append_images=gif[1:], duration=150, loop=0)
        draw.line([(0, r * C), (sheet.width, r * C)], fill=(200, 200, 200))
    sheet.save(out / f"sheet-{mode}-{start // PER}.png")
print("sheets written")
