"""Reference decodes for the browser's end-to-end checks (`web/e2e`).

The JavaScript decoder (`web/src`: grammar, greedy loop, codec) driving an exported model
(`onnx_export`) must return what the Python ONNX Runtime decoder (`onnx_decode.OnnxDecoder`)
returns on the same pixels: the same 1,376 tokens and, through the JavaScript codec, the
same SVG bytes as `packed.serialize_packed_svg`. This module writes that reference for a
spread of validation icons as a small bundle both the Node check and the page read:

    python -m mojidiff.learning.web_e2e --run <run_id> [--icons 32] [--model DIR] [--out DIR]

* `images.u8`: the renders the model decodes, (icons, S, S, 3) uint8 RGB, exactly the
  corpus pixels (`load_corpus`), so browser and Python see identical inputs.
* `reference.json`: per icon its validation index and hexcode, the reference tokens in
  original positions, the decoder calls, the SVG text and its SHA-256, plus the model
  files' and dataset's hashes.

Icons are spread evenly over the validation split (`spread_indices`), not taken from its
start: the first icons are light (the first 16 need a median of 260 calls against 425
over all 339). Run it from the repository root; the pilot configuration's paths are
relative to it. The default output is `CACHE_ROOT/web-reference/<run_id>/validation-<N>/`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mojidiff.learning.autoregressive import SequenceLayout
from mojidiff.learning.onnx_decode import OnnxDecoder
from mojidiff.learning.onnx_export import MANIFEST_FILE
from mojidiff.learning.render2svg import CACHE_ROOT, DecodeStats, SplitData
from mojidiff.learning.web_reference import program_svg
from mojidiff.representation.packed import PackedTensorProgram

BUNDLE_FORMAT = 1
IMAGES_FILE = "images.u8"
REFERENCE_FILE = "reference.json"
SPLIT = "primary/validation"


def spread_indices(total: int, count: int) -> list[int]:
    """`count` distinct indices from 0 to `total - 1`, evenly spaced (both ends included)."""

    if total < 1 or count < 1:
        raise ValueError("need at least one item and one index")
    return sorted({int(i) for i in np.linspace(0, total - 1, min(count, total)).round()})


def reference_icons(
    decoder: OnnxDecoder,
    data: SplitData,
    indices: Sequence[int],
    template: PackedTensorProgram,
    layout: SequenceLayout,
) -> list[dict[str, Any]]:
    """Decode each chosen render with ONNX Runtime; tokens, calls and SVG per icon."""

    icons = []
    for index in indices:
        stats = DecodeStats()
        started = time.perf_counter()
        tokens = decoder.decode(data.images[index].numpy(), stats=stats)
        seconds = time.perf_counter() - started
        svg = program_svg(tokens, template, layout)
        if svg is None:
            raise ValueError(f"the serializer rejects the decode of icon {index}")
        icons.append(
            {
                "index": int(index),
                "hexcode": data.hexcodes[index],
                "model_calls": stats.model_calls,
                "tokens": tokens.tolist(),
                "svg": svg,
                "svg_sha256": hashlib.sha256(svg.encode()).hexdigest(),
                "python_ort_cpu_ms": round(seconds * 1000, 1),
            }
        )
    return icons


def write_bundle(
    out: Path,
    model_dir: Path,
    data: SplitData,
    indices: Sequence[int],
    icons: list[dict[str, Any]],
    provenance: dict[str, Any],
) -> dict[str, Any]:
    """Write `images.u8` and `reference.json` to `out`; the reference record."""

    manifest = json.loads((model_dir / MANIFEST_FILE).read_text())
    images = np.ascontiguousarray(data.images[list(indices)].numpy(), dtype=np.uint8)
    out.mkdir(parents=True, exist_ok=True)
    (out / IMAGES_FILE).write_bytes(images.tobytes())
    import onnxruntime

    record: dict[str, Any] = {
        "format": BUNDLE_FORMAT,
        "generated_by": "python -m mojidiff.learning.web_e2e",
        "decoder": "mojidiff.learning.onnx_decode.OnnxDecoder, CPUExecutionProvider",
        "onnxruntime": onnxruntime.__version__,
        "run_id": manifest.get("run_id"),
        "model": {
            "directory": str(model_dir),
            "checkpoint_sha256": manifest.get("checkpoint", {}).get("sha256"),
            **{f"{role}_sha256": entry["sha256"] for role, entry in manifest["files"].items()},
        },
        **provenance,
        "split": SPLIT,
        "split_size": len(data.hexcodes),
        "images": {
            "file": IMAGES_FILE,
            "dtype": "uint8",
            "shape": list(images.shape),
            "layout": "NHWC RGB, the corpus render the model reads",
            "sha256": hashlib.sha256(images.tobytes()).hexdigest(),
        },
        "icons": icons,
    }
    (out / REFERENCE_FILE).write_text(json.dumps(record) + "\n")
    return record


def build_reference(
    model_dir: Path,
    out: Path,
    *,
    icons: int = 32,
    pilot_config: Path | None = None,
    threads: int | None = None,
) -> dict[str, Any]:
    """Load the corpus and the exported model, decode `icons` spread validation icons and
    write the bundle."""

    from mojidiff.learning.openmoji_pilot import _load_program, load_pilot_index
    from mojidiff.learning.render2svg import load_corpus

    manifest = json.loads((model_dir / MANIFEST_FILE).read_text())
    pilot_config = pilot_config or Path(str(manifest["pilot_config"]))
    image_size = int(manifest["constants"]["image_size"])
    plain, _, layout, pilot, dataset_hash = load_corpus(pilot_config, image_size)
    by_split, _, _ = load_pilot_index(pilot)
    template = _load_program(by_split["primary/train"][0], pilot, layout.codec)
    data = plain[SPLIT]
    indices = spread_indices(len(data.hexcodes), icons)
    decoder = OnnxDecoder(model_dir, layout, threads=threads)
    rows = reference_icons(decoder, data, indices, template, layout)
    provenance = {"pilot_config": str(pilot_config), "dataset_sha256": dataset_hash}
    return write_bundle(out, model_dir, data, indices, rows, provenance)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--run", required=True, help="run id of the exported model")
    parser.add_argument("--model", type=Path, help="export directory (CACHE_ROOT/onnx/<run>)")
    parser.add_argument("--icons", type=int, default=32, help="validation icons, spread evenly")
    parser.add_argument("--out", type=Path, help="bundle directory")
    parser.add_argument("--pilot-config", type=Path, help="pilot config (from the manifest)")
    parser.add_argument("--threads", type=int, help="ONNX Runtime intra-op threads")
    args = parser.parse_args(argv)
    model_dir = args.model or CACHE_ROOT / "onnx" / args.run
    out = args.out or CACHE_ROOT / "web-reference" / args.run / f"validation-{args.icons}"
    torch.set_grad_enabled(False)
    record = build_reference(
        model_dir, out, icons=args.icons, pilot_config=args.pilot_config, threads=args.threads
    )
    calls = [icon["model_calls"] for icon in record["icons"]]
    print(
        json.dumps(
            {
                "wrote": str(out),
                "icons": len(calls),
                "indices": [icon["index"] for icon in record["icons"]],
                "model_calls_median": float(np.median(calls)),
                "model_calls_range": [min(calls), max(calls)],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
