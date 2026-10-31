"""vtracer (compact preset, as in reports/visual/2026-10-31-tracer-baseline) on each degraded validation set.
Run with the tool venv: python -I trace_degraded.py <doodle-eval seed dir> <out dir> <kind>..."""
import sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import vtracer
src, out = Path(sys.argv[1]), Path(sys.argv[2])
compact = dict(colormode="color", filter_speckle=8, color_precision=4, layer_difference=32, corner_threshold=60,
               length_threshold=6.0, splice_threshold=45, path_precision=2)
for kind in sys.argv[3:]:
    images = np.load(src / f"{kind}.npz")["images"]
    (out / kind).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        for i, image in enumerate(images):
            png = Path(tmp) / f"{i:03d}.png"
            Image.fromarray(image).save(png)
            vtracer.convert_image_to_svg_py(str(png), str(out / kind / f"{i:03d}.svg"), **compact)
    print(kind, len(images))
