#!/usr/bin/env python3
"""Build-time checks and caches for a MojiDiff Space, so the container starts fast.

    python space/warm_cache.py gallery|vectorise

Both: the icon cache (`data/processed/kitbash`, built just before by
`python -m mojidiff.vectorise.icon_cache`) has its index, and every checkpoint in
`space/checkpoints.json` loads with this image's torch (CPU-only 2.8.0; the checkpoints
were saved by a 2.14 nightly). Gallery only: the render-to-SVG corpus at the gallery's
render size, `render2svg.load_corpus`, which the gallery reads at startup for the
held-out programs and the latent encoders' images; built here it is a cache hit at run
time instead of minutes of rendering.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def main(argv: list[str]) -> int:
    if len(argv) != 2 or argv[1] not in ("gallery", "vectorise"):
        print(__doc__, file=sys.stderr)
        return 2
    import torch

    from mojidiff.vectorise.server import ICON_CACHE

    if not (ICON_CACHE / "index.json").is_file():
        print(f"no icon cache at {ICON_CACHE}", file=sys.stderr)
        return 1
    spec = json.loads((ROOT / "space" / "checkpoints.json").read_text())
    for name in sorted(spec["files"]):
        state = torch.load(Path(spec["target"]) / name, map_location="cpu")
        print(f"{name}: loads with torch {torch.__version__}, step {state.get('step')}")
    if argv[1] == "gallery":
        from mojidiff.gallery.server import RENDER_SIZE
        from mojidiff.learning.render2svg import load_corpus
        from mojidiff.vectorise.server import PILOT_CONFIG

        started = time.perf_counter()
        workers = max(1, min(8, os.cpu_count() or 1))
        plain, _, _, _, dataset_hash = load_corpus(PILOT_CONFIG, RENDER_SIZE, workers=workers)
        icons = {name: len(split.hexcodes) for name, split in plain.items()}
        print(
            f"corpus at {RENDER_SIZE} px: {icons}, dataset sha256 {dataset_hash}, "
            f"{time.perf_counter() - started:.0f} s with {workers} workers"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
