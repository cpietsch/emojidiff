"""Trace the 339 validation input renders (144 px, what the model sees) with vtracer, two presets."""
import sys, time
from pathlib import Path
import vtracer
V = Path(sys.argv[1])
presets = {
    "default": dict(colormode="color"),
    # Fewer, larger shapes: closer to an icon's structure (coarser colours, bigger speckle filter).
    "compact": dict(colormode="color", filter_speckle=8, color_precision=4, layer_difference=32, corner_threshold=60, length_threshold=6.0, splice_threshold=45, path_precision=2),
}
for name, kw in presets.items():
    t0 = time.perf_counter()
    for png in sorted((V / "inputs").glob("*.png")):
        vtracer.convert_image_to_svg_py(str(png), str(V / name / (png.stem + ".svg")), **kw)
    print(name, f"{(time.perf_counter() - t0) / 339 * 1000:.1f} ms per icon (CPU)")
