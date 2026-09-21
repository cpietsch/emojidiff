"""OmniSVG on held-out OpenMoji captions: the zero-shot control, and later the fine-tune.

Every held-out icon's caption is given to the model; every sample it returns is decoded
with OmniSVG's own tokenizer, converted into the project's codec, rendered by the
project's renderer where the codec accepts it and by cairosvg regardless, and scored
against the held-out icon's own render and against its caption with a pinned CLIP. The
zero-shot reading is the floor the fine-tuned model is read against, and it is not
nothing: the model draws.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from mojidiff.learning.omnisvg import (
    OMNISVG_REPO,
    OMNISVG_REVISION,
    OmniSVG,
    parse_into_codec,
    to_project_svg,
)
from mojidiff.learning.openmoji_pilot import (
    OpenMojiPilotError,
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.packed import serialize_packed_svg
from mojidiff.representation.renderer import RenderLimits, render_typed_svg_isolated

CLIP_REPO = "openai/clip-vit-base-patch32"
CLIP_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def captions(raw_root: Path) -> dict[str, str]:
    """OpenMoji's own annotation for every icon, by hexcode."""

    entries = json.loads((raw_root / "data" / "openmoji.json").read_text())
    return {entry["hexcode"]: str(entry.get("annotation", "")).strip() for entry in entries}


def caption_prompt(annotation: str) -> str:
    return f"{annotation}, flat emoji icon"


class Clip:
    def __init__(self, device: str) -> None:
        from transformers import CLIPModel, CLIPProcessor

        model: Any = CLIPModel.from_pretrained(CLIP_REPO, revision=CLIP_REVISION)
        self.model = model.to(device).eval()
        self.processor = CLIPProcessor.from_pretrained(CLIP_REPO, revision=CLIP_REVISION)
        self.device = device

    @torch.inference_mode()
    def image(self, images: list[Image.Image]) -> torch.Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        features = self.model.get_image_features(**inputs)
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu()

    @torch.inference_mode()
    def text(self, texts: list[str]) -> torch.Tensor:
        inputs = self.processor(text=texts, return_tensors="pt", padding=True).to(self.device)
        features = self.model.get_text_features(**inputs)
        return torch.nn.functional.normalize(features.float(), dim=-1).cpu()


def render_raw(svg: str, size: int) -> Image.Image:
    """Render OmniSVG's own SVG text without the codec, for the eye and for CLIP."""

    import cairosvg

    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=size, output_height=size)
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    return background.convert("RGB")


def render_program(
    program: Any, codec: Any, slots: int, limits: RenderLimits, size: int
) -> Image.Image:
    svg = serialize_packed_svg(program, codec, slots)
    _, raster = render_typed_svg_isolated(svg, size, limits)
    image = Image.fromarray(raster.astype(np.uint8), "RGBA")
    background = Image.new("RGBA", image.size, "white")
    background.alpha_composite(image)
    return background.convert("RGB")


def run_omnisvg_study(config_path: Path) -> dict[str, Any]:
    root = yaml.safe_load(config_path.read_text())
    if root.get("schema_version") != 1:
        raise OpenMojiPilotError("omnisvg study schema_version must be 1")
    version = str(root["study_version"])
    report_root = Path(str(root["report_root"]))
    pilot = load_openmoji_pilot_config(Path(str(root["pilot_config"])))
    by_split, _, _ = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    data = root["data"]
    generation = root["generation"]
    icons = int(data["icons"])
    samples = int(generation["samples"])
    max_new_tokens = int(generation["max_new_tokens"])
    seed = int(generation["seed"])
    size = int(root.get("render", {}).get("size", 72))
    limits = RenderLimits(max_paths=codec.max_paths, timeout_seconds=20)

    rows_selected = _select_rows(
        by_split[str(data.get("split", "primary/validation"))], icons, pilot.seed + 1
    )
    annotations = captions(pilot.raw_root)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    started = time.perf_counter()
    model = OmniSVG.load(device)
    load_seconds = time.perf_counter() - started
    clip = Clip(device)

    rows: list[dict[str, Any]] = []
    tiles: list[tuple[str, list[Image.Image]]] = []
    generate_seconds = 0.0
    for index, row in enumerate(rows_selected):
        annotation = annotations.get(row.hexcode, row.hexcode)
        prompt = caption_prompt(annotation)
        reference = render_program(
            _load_program(row, pilot, codec), codec, pilot.total_segment_slots, limits, size
        )
        reference_feature = clip.image([reference])
        caption_feature = clip.text([annotation])
        clock = time.perf_counter()
        drawings = model.generate(
            prompt, samples=samples, max_new_tokens=max_new_tokens, seed=seed + index
        )
        generate_seconds += time.perf_counter() - clock
        row_tiles = [reference]
        for sample_index, tokens in enumerate(drawings):
            record: dict[str, Any] = {
                "hexcode": row.hexcode,
                "annotation": annotation,
                "prompt": prompt,
                "sample": sample_index,
            }
            svg, decode_info = model.tokens_to_svg(tokens)
            record.update({f"decode_{key}": value for key, value in decode_info.items()})
            image: Image.Image | None = None
            if svg is not None:
                projected, snap = to_project_svg(svg, codec.palette)
                record.update({f"snap_{key}": value for key, value in snap.items()})
                program, parse_info = parse_into_codec(projected, codec, pilot.total_segment_slots)
                record.update({f"codec_{key}": value for key, value in parse_info.items()})
                record["codec_valid"] = program is not None
                try:
                    image = render_raw(svg, size)
                    record["raw_render"] = True
                except Exception as error:  # noqa: BLE001
                    record["raw_render"] = False
                    record["raw_render_error"] = type(error).__name__
                if program is not None:
                    image = render_program(program, codec, pilot.total_segment_slots, limits, size)
            else:
                record["codec_valid"] = False
            if image is not None:
                feature = clip.image([image])
                record["clip_to_reference"] = float((feature @ reference_feature.T)[0, 0])
                record["clip_to_caption"] = float((feature @ caption_feature.T)[0, 0])
            row_tiles.append(
                image if image is not None else Image.new("RGB", (size, size), "white")
            )
            rows.append(record)
        tiles.append((f"{row.hexcode} {annotation[:22]}", row_tiles))

    sheet = _sheet(tiles, samples, size)
    rows_payload = b"".join(
        (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for record in rows
    )
    decoded = [record for record in rows if record.get("decode_paths")]
    valid = [record for record in rows if record.get("codec_valid")]
    scored = [record for record in rows if "clip_to_reference" in record]
    summary = {
        "schema_version": 1,
        "study_version": version,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model": {
            "repo": OMNISVG_REPO,
            "revision": OMNISVG_REVISION,
            "checkpoint_sha256": model.checkpoint_sha256,
            "parameters": model.parameters,
            "fine_tuned": False,
        },
        "clip": {"repo": CLIP_REPO, "revision": CLIP_REVISION},
        "icons": len(rows_selected),
        "samples_per_icon": samples,
        "max_new_tokens": max_new_tokens,
        "drawings": len(rows),
        "decoded_rate": len(decoded) / len(rows),
        "ended_rate": sum(1 for record in rows if record.get("decode_ended")) / len(rows),
        "codec_valid_rate": len(valid) / len(rows),
        "failures": _count(rows, "decode_failure") | _count(rows, "codec_failure"),
        "median_tokens": float(np.median([record["decode_tokens"] for record in rows])),
        "median_paths": float(np.median([record["decode_paths"] for record in decoded]))
        if decoded
        else None,
        "mean_snap_distance_rgb": float(
            np.mean(
                [
                    record["snap_mean_snap_distance_rgb"]
                    for record in rows
                    if "snap_mean_snap_distance_rgb" in record
                ]
            )
        )
        if decoded
        else None,
        "clip_to_reference": _stats([record["clip_to_reference"] for record in scored]),
        "clip_to_caption": _stats([record["clip_to_caption"] for record in scored]),
        "timing": {
            "load_seconds": load_seconds,
            "generate_seconds": generate_seconds,
            "seconds_per_drawing": generate_seconds / len(rows),
            "peak_vram_gib": float(torch.cuda.max_memory_allocated()) / 2**30
            if device == "cuda"
            else None,
        },
        "sheet_sha256": hashlib.sha256(sheet).hexdigest(),
        "rows_sha256": hashlib.sha256(rows_payload).hexdigest(),
    }
    _write_bytes_artifact(report_root / "drawings.jsonl", rows_payload)
    _write_bytes_artifact(report_root / "samples.png", sheet)
    _write_bytes_artifact(
        report_root / "summary.json",
        (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    return summary


def _count(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for record in rows:
        value = record.get(key)
        if isinstance(value, str):
            counts[value] = counts.get(value, 0) + 1
    return counts


def _stats(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "median": None}
    array = np.array(values)
    return {"n": len(values), "mean": float(array.mean()), "median": float(np.median(array))}


def _sheet(tiles: list[tuple[str, list[Image.Image]]], samples: int, size: int) -> bytes:
    tile = size + 16
    label_width = 190
    header = 30
    columns = 1 + samples
    canvas = Image.new("RGB", (label_width + tile * columns, header + tile * len(tiles)), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((label_width + 4, 8), "held-out icon", fill="black")
    for index in range(samples):
        draw.text((label_width + tile * (index + 1) + 4, 8), f"sample {index}", fill="black")
    for row_index, (label, images) in enumerate(tiles):
        y = header + row_index * tile
        draw.text((4, y + tile // 2), label[:30], fill="black")
        for column, image in enumerate(images):
            canvas.paste(image.resize((size, size)), (label_width + column * tile + 8, y + 8))
    payload = io.BytesIO()
    canvas.save(payload, format="PNG", optimize=True)
    return payload.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = run_omnisvg_study(args.config)
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "drawings",
                    "decoded_rate",
                    "ended_rate",
                    "codec_valid_rate",
                    "failures",
                    "median_tokens",
                    "clip_to_reference",
                    "clip_to_caption",
                    "timing",
                )
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
