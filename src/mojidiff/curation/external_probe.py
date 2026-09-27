"""How much of another emoji set does the OpenMoji codec take, and how faithfully?

Wider vector data is the only route the first phase found to shape priors a small model
cannot get from 2,681 icons. Before any curation, this measures the fit: each source SVG
goes through one adapter -

* PicoSVG flattens CSS, groups, transforms and strokes into filled paths;
* each gradient becomes the mean of its stop colours (lossy, and counted);
* every colour snaps to OpenMoji's palette (lossy, and measured);
* the box is scaled into the codec's 72 units;

- and then through the project's own normaliser, codec and packer at the P32/T128
bucket the render-to-SVG model uses. Reported per set: how many parse, how many fit the
bucket without projection, why the rest fail, and pixel error between the source's own
render and the codec program's render at 72 px. OpenMoji goes through the same adapter
as the calibration row, so adapter loss and style mismatch can be told apart.

    python -m mojidiff.curation.external_probe --sample 600
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import subprocess
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTERNAL = REPO_ROOT / "data/raw/external"
SOURCES: dict[str, dict[str, str]] = {
    "twemoji": {"glob": "twemoji/assets/svg/*.svg", "license": "CC-BY-4.0 (graphics)"},
    "noto-emoji": {"glob": "noto-emoji/2D/svg/*.svg", "license": "Apache-2.0"},
    "blobmoji": {"glob": "blobmoji/svg/*.svg", "license": "Apache-2.0"},
    "openmoji": {"glob": "", "license": "CC-BY-SA-4.0"},
}
_RGB = re.compile(r"rgb\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)")


def _hex(colour: str) -> str | None:
    colour = colour.strip().lower()
    match = _RGB.fullmatch(colour)
    if match:
        return "#" + "".join(f"{int(v):02x}" for v in match.groups())
    if re.fullmatch(r"#[0-9a-f]{6}", colour):
        return colour
    if re.fullmatch(r"#[0-9a-f]{3}", colour):
        return "#" + "".join(c * 2 for c in colour[1:])
    named = {"black": "#000000", "white": "#ffffff", "red": "#ff0000"}
    return named.get(colour)


def _gradient_colours(pico_text: str) -> dict[str, str]:
    """Gradient id -> mean of its stop colours."""

    from defusedxml import ElementTree

    root = ElementTree.fromstring(pico_text)
    means: dict[str, str] = {}
    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        if tag not in ("linearGradient", "radialGradient"):
            continue
        stops = []
        for stop in element:
            colour = _hex(stop.attrib.get("stop-color", "#000000"))
            if colour:
                stops.append([int(colour[i : i + 2], 16) for i in (1, 3, 5)])
        if stops and "id" in element.attrib:
            mean = np.mean(np.asarray(stops, dtype=np.float64), axis=0)
            means[element.attrib["id"]] = "#" + "".join(f"{int(round(v)):02x}" for v in mean)
    return means


def adapt(source: bytes, palette: tuple[str, ...]) -> tuple[bytes, dict[str, Any]]:
    """A source SVG as a 72-box, fill-only SVG in palette colours, with what it cost."""

    from picosvg.svg import SVG

    from mojidiff.learning.omnisvg import snap_to_palette

    pico = SVG.fromstring(source.decode("utf-8", errors="replace")).topicosvg()  # type: ignore[no-untyped-call]
    text = pico.tostring()
    gradients = _gradient_colours(text)
    view_box = pico.view_box()
    if view_box is None:
        raise ValueError("no viewBox")
    x, y, width, height = view_box.x, view_box.y, view_box.w, view_box.h
    scale = 72.0 / max(width, height)
    offset_x = (72.0 - width * scale) / 2 - x * scale
    offset_y = (72.0 - height * scale) / 2 - y * scale
    paths: list[str] = []
    gradient_fills = 0
    for shape in pico.shapes():
        fill = str(shape.fill)
        if fill.startswith("url("):
            key = fill[fill.index("#") + 1 : fill.rindex(")")]
            colour = gradients.get(key)
            gradient_fills += 1
        else:
            colour = _hex(fill)
        if colour is None:
            continue
        opacity = float(shape.fill_opacity) * float(shape.opacity)
        if opacity <= 0:
            continue
        attributes = f'd="{shape.d}" fill="{colour}"'
        if opacity < 1:
            attributes += f' fill-opacity="{opacity:.4f}"'
        if getattr(shape, "fill_rule", "nonzero") == "evenodd":
            attributes += ' fill-rule="evenodd"'
        paths.append(f"<path {attributes}/>")
    body, snap = snap_to_palette("".join(paths), palette)
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">'
        f'<g transform="matrix({scale} 0 0 {scale} {offset_x} {offset_y})">{body}</g></svg>'
    )
    return svg.encode(), {
        "shapes": len(paths),
        "gradient_fills": gradient_fills,
        **{f"snap_{k}": v for k, v in snap.items()},
    }


def _render_source(source: bytes, size: int = 72) -> np.ndarray:
    import cairosvg
    from PIL import Image

    png = cairosvg.svg2png(bytestring=source, output_width=size, output_height=size, unsafe=False)
    with Image.open(io.BytesIO(png)) as image:
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, "white")
        white.alpha_composite(rgba)
        return np.asarray(white.convert("RGB"), dtype=np.uint8)


def probe_one(args: tuple[str, str, bytes]) -> dict[str, Any]:
    from mojidiff.learning.omnisvg import parse_into_codec
    from mojidiff.learning.openmoji_pilot import _selected_codec, load_openmoji_pilot_config
    from mojidiff.learning.render2svg import pixel_error, render_trusted_rgb
    from mojidiff.representation.packed import serialize_packed_svg

    name, key, source = args
    pilot = load_openmoji_pilot_config(
        REPO_ROOT / "configs/learning/openmoji-g1-geometric-gate-v16.yaml"
    )
    codec = _selected_codec(pilot)
    record: dict[str, Any] = {"set": name, "key": key}
    try:
        reference = _render_source(source)
    except Exception as error:  # noqa: BLE001 - a source the renderer refuses is a result
        record["failure"] = f"source_render:{type(error).__name__}"
        return record
    try:
        adapted, info = adapt(source, codec.palette)
    except Exception as error:  # noqa: BLE001
        record["failure"] = f"adapt:{type(error).__name__}"
        return record
    record.update(info)
    try:
        record["adapted_pixel_error"] = pixel_error(_render_source(adapted), reference)
    except Exception:  # noqa: BLE001
        record["adapted_pixel_error"] = None
    program, parsed = parse_into_codec(adapted, codec, pilot.total_segment_slots)
    record.update({f"codec_{k}": v for k, v in parsed.items()})
    if program is None:
        record["failure"] = str(parsed.get("failure", "codec"))
        return record
    svg = serialize_packed_svg(program, codec, pilot.total_segment_slots)
    record["fits"] = True
    record["codec_pixel_error"] = pixel_error(render_trusted_rgb(svg, 72), reference)
    return record


def _sources(name: str, sample: int, seed: int) -> list[tuple[str, str, bytes]]:
    if name == "openmoji":
        from mojidiff.learning.openmoji_pilot import load_openmoji_pilot_config, load_pilot_index

        pilot = load_openmoji_pilot_config(
            REPO_ROOT / "configs/learning/openmoji-g1-geometric-gate-v16.yaml"
        )
        by_split, _, _ = load_pilot_index(pilot)
        rows = [row for rows in by_split.values() for row in rows]
        paths = sorted({str(pilot.raw_root / row.source_path) for row in rows})
    else:
        paths = sorted(str(p) for p in EXTERNAL.glob(SOURCES[name]["glob"]))
    rng = np.random.default_rng(seed)
    chosen = sorted(rng.choice(len(paths), size=min(sample, len(paths)), replace=False))
    return [(name, Path(paths[i]).name, Path(paths[i]).read_bytes()) for i in chosen]


def _commit(name: str) -> str | None:
    if name == "openmoji":
        return "17.0.0"
    return subprocess.run(
        ["git", "-C", str(EXTERNAL / name), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout.strip()


def summarise(records: list[dict[str, Any]]) -> dict[str, Any]:
    fits = [r for r in records if r.get("fits")]
    errors = [r["codec_pixel_error"] for r in fits]
    adapted = [
        r["adapted_pixel_error"] for r in records if r.get("adapted_pixel_error") is not None
    ]
    return {
        "probed": len(records),
        "fit_rate": len(fits) / max(len(records), 1),
        "failures": dict(Counter(r["failure"] for r in records if "failure" in r).most_common(8)),
        "codec_pixel_error_median": float(np.median(errors)) if errors else None,
        "codec_pixel_error_p90": float(np.quantile(errors, 0.9)) if errors else None,
        "adapted_pixel_error_median": float(np.median(adapted)) if adapted else None,
        "with_gradients_rate": float(np.mean([r.get("gradient_fills", 0) > 0 for r in records])),
        "mean_snap_distance_rgb": float(
            np.mean([r["snap_mean_snap_distance_rgb"] for r in records if "snap_fills" in r])
        ),
        "median_segments_when_fit": float(np.median([r["codec_segments"] for r in fits]))
        if fits
        else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sample", type=int, default=600)
    parser.add_argument("--seed", type=int, default=2809)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "reports/corpus/external-probe-v1")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"sample_per_set": args.sample, "seed": args.seed, "sets": {}}
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name in SOURCES:
            jobs = _sources(name, args.sample, args.seed)
            records = list(pool.map(probe_one, jobs, chunksize=4))
            rows.extend(records)
            digest = hashlib.sha256(b"".join(source for _, _, source in jobs)).hexdigest()
            report["sets"][name] = {
                "commit": _commit(name),
                "license": SOURCES[name]["license"],
                "sample_sha256": digest,
                **summarise(records),
            }
            print(json.dumps({name: report["sets"][name]}), flush=True)
    (args.out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (args.out / "records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
