"""Gate N, step three: an edit made in pixels, handed back to the prior for the program.

An icon's program is edited exactly - a fill recoloured, a part erased, a part moved -
and the edited program is rendered. The render is what the prior sees; what it returns
is decoded, converted into the codec, rendered, and read against the edited icon's own
render and against the unedited one: a drawing that reflects the edit is closer to the
edited render than to the original, and the right edited icon should be retrieved
first among all edited held-out renders. Because the edited program is exact, every
measure has a ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image

from mojidiff.learning.omnisvg import (
    OMNISVG_REPO,
    OMNISVG_REVISION,
    OmniSVG,
    parse_into_codec,
    to_project_svg,
)
from mojidiff.learning.omnisvg_study import (
    CLIP_REPO,
    CLIP_REVISION,
    STEP_SECONDS,
    Clip,
    StepTimeout,
    captions,
    guarded,
    render_program,
    render_raw,
)
from mojidiff.learning.openmoji_pilot import (
    OpenMojiPilotError,
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.prior import dumps
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.packed import PackedTensorProgram, serialize_packed_svg
from mojidiff.representation.renderer import RenderLimits

NONE = 1  # the codec's "no paint" fill token; palette colour i is token i + 2
_PATH = re.compile(r"<path\b[^>]*/>", re.IGNORECASE)


# ---------------------------------------------------------------------------- edits


def recolour(
    program: PackedTensorProgram, codec: Any, slots: int, target: str
) -> tuple[bytes, dict[str, Any]]:
    """The most used painted fill that is not black or white becomes `target`."""

    painted = [
        int(f) for f, n in zip(program.fill, program.path_length, strict=True) if n > 0 and f > NONE
    ]
    black, white = codec.palette.index("#000000") + 2, codec.palette.index("#ffffff") + 2
    target_token = codec.palette.index(target) + 2
    candidates = [f for f in painted if f not in (black, white, target_token)]
    if not candidates:
        raise ValueError("no recolourable fill")
    source = max(set(candidates), key=candidates.count)
    fill = program.fill.copy()
    fill[fill == source] = target_token
    edited = PackedTensorProgram(**{**program.__dict__, "fill": fill})
    return serialize_packed_svg(edited, codec, slots), {
        "from": codec.palette[source - 2],
        "to": target,
        "paths_changed": int((program.fill == source).sum()),
    }


def erase(program: PackedTensorProgram, codec: Any, slots: int) -> tuple[bytes, dict[str, Any]]:
    """The middle painted path is removed from the drawing."""

    svg = serialize_packed_svg(program, codec, slots).decode()
    paths = _PATH.findall(svg)
    if len(paths) < 3:
        raise ValueError("too few paths to erase one")
    index = len(paths) // 2
    edited = svg.replace(paths[index], "", 1)
    return edited.encode(), {"path_index": index, "paths": len(paths)}


def move(
    program: PackedTensorProgram, codec: Any, slots: int, dx: float, dy: float
) -> tuple[bytes, dict[str, Any]]:
    """The last painted path is translated by (dx, dy) units of the 72 box."""

    svg = serialize_packed_svg(program, codec, slots).decode()
    paths = _PATH.findall(svg)
    if len(paths) < 2:
        raise ValueError("too few paths to move one")
    index = len(paths) - 1
    moved = f'<g transform="translate({dx} {dy})">{paths[index]}</g>'
    edited = svg.replace(paths[index], moved, 1)
    return edited.encode(), {"path_index": index, "dx": dx, "dy": dy}


def edited_program(
    kind: str, program: PackedTensorProgram, codec: Any, slots: int, settings: dict[str, Any]
) -> tuple[PackedTensorProgram, dict[str, Any]]:
    """The edited program back through the codec, so the ground truth is a valid program."""

    if kind == "recolour":
        svg, info = recolour(program, codec, slots, str(settings.get("target", "#ea5a47")))
    elif kind == "erase":
        svg, info = erase(program, codec, slots)
    elif kind == "move":
        svg, info = move(
            program, codec, slots, float(settings.get("dx", 8)), float(settings.get("dy", 0))
        )
    else:
        raise ValueError(f"unknown edit {kind!r}")
    parsed, parse_info = parse_into_codec(svg, codec, slots)
    if parsed is None:
        raise ValueError(f"edited program does not parse: {parse_info.get('failure')}")
    return parsed, info


# ---------------------------------------------------------------------------- study


def run_omnisvg_edit_study(config_path: Path) -> dict[str, Any]:
    root = yaml.safe_load(config_path.read_text())
    if root.get("schema_version") != 1:
        raise OpenMojiPilotError("omnisvg edit study schema_version must be 1")
    version = str(root["study_version"])
    report_root = Path(str(root["report_root"]))
    pilot = load_openmoji_pilot_config(Path(str(root["pilot_config"])))
    by_split, _, _ = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    slots = pilot.total_segment_slots
    data = root["data"]
    generation = root["generation"]
    edits: dict[str, dict[str, Any]] = root["edits"]
    size = int(root.get("render", {}).get("size", 72))
    limits = RenderLimits(max_paths=codec.max_paths, timeout_seconds=20)
    rows = _select_rows(
        by_split[str(data.get("split", "primary/validation"))], int(data["icons"]), pilot.seed + 1
    )
    annotations = captions(pilot.raw_root)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sampling = {
        key: generation[key]
        for key in ("temperature", "top_p", "top_k", "repetition_penalty", "greedy")
        if key in generation
    }

    started = time.perf_counter()
    model = OmniSVG.load(device)
    adapter = root.get("adapter")
    adapter_sha256 = model.attach_adapter(Path(str(adapter))) if adapter else None
    load_seconds = time.perf_counter() - started
    clip = Clip(device)

    originals = [_load_program(row, pilot, codec) for row in rows]
    original_renders = [render_program(p, codec, slots, limits, size) for p in originals]
    original_features = clip.image(original_renders)

    records: list[dict[str, Any]] = []
    tiles: list[tuple[str, list[Image.Image]]] = []
    generate_seconds = 0.0
    for kind, settings in edits.items():
        targets: list[tuple[int, PackedTensorProgram, dict[str, Any]]] = []
        for index, program in enumerate(originals):
            try:
                edited, info = edited_program(kind, program, codec, slots, settings)
            except ValueError as error:
                records.append(
                    {"hexcode": rows[index].hexcode, "edit": kind, "skipped": str(error)[:80]}
                )
                continue
            targets.append((index, edited, info))
        edited_renders = [
            render_program(edited, codec, slots, limits, size) for _, edited, _ in targets
        ]
        edited_features = clip.image(edited_renders)
        for position, (index, edited, info) in enumerate(targets):
            row = rows[index]
            condition = render_program(edited, codec, slots, limits, 448)
            clock = time.perf_counter()
            drawings = model.generate(
                f"edited render of {row.hexcode}",
                samples=int(generation["samples"]),
                max_new_tokens=int(generation["max_new_tokens"]),
                seed=int(generation["seed"]) + 100 * index + list(edits).index(kind),
                style="image",
                image=condition,
                **sampling,
            )
            generate_seconds += time.perf_counter() - clock
            row_tiles = [original_renders[index], edited_renders[position]]
            for sample_index, tokens in enumerate(drawings):
                record: dict[str, Any] = {
                    "hexcode": row.hexcode,
                    "annotation": annotations.get(row.hexcode, row.hexcode),
                    "edit": kind,
                    "edit_info": info,
                    "sample": sample_index,
                }
                svg, decode_info = model.tokens_to_svg(tokens)
                record.update({f"decode_{key}": value for key, value in decode_info.items()})
                image: Image.Image | None = None
                if svg is not None:
                    record["svg"] = svg
                    projected, snap = to_project_svg(svg, codec.palette)
                    record.update({f"snap_{key}": value for key, value in snap.items()})
                    try:
                        parsed, parse_info = guarded(
                            STEP_SECONDS, parse_into_codec, projected, codec, slots
                        )
                    except StepTimeout:
                        parsed, parse_info = None, {"failure": "parse_timeout"}
                    record.update({f"codec_{key}": value for key, value in parse_info.items()})
                    record["codec_valid"] = parsed is not None
                    try:
                        image = guarded(STEP_SECONDS, render_raw, svg, size)
                        record["raw_render"] = True
                    except Exception as error:  # noqa: BLE001
                        record["raw_render"] = False
                        record["raw_render_error"] = type(error).__name__
                    if parsed is not None:
                        image = render_program(parsed, codec, slots, limits, size)
                else:
                    record["codec_valid"] = False
                if image is not None:
                    feature = clip.image([image])
                    to_edited = (feature @ edited_features.T)[0]
                    own_edited = float(to_edited[position])
                    own_original = float((feature @ original_features[index : index + 1].T)[0, 0])
                    record["clip_to_edited"] = own_edited
                    record["clip_to_original"] = own_original
                    record["edit_reflected"] = own_edited > own_original
                    record["edited_rank"] = 1 + int((to_edited > own_edited).sum())
                    record["rgba_mae_to_edited"] = float(
                        np.abs(
                            np.asarray(image, dtype=np.float32)
                            - np.asarray(edited_renders[position], dtype=np.float32)
                        ).mean()
                        / 255.0
                    )
                row_tiles.append(
                    image if image is not None else Image.new("RGB", (size, size), "white")
                )
                records.append(record)
            tiles.append((f"{kind} {row.hexcode}", row_tiles))
            print(
                json.dumps(
                    {
                        "edit": kind,
                        "icon": position + 1,
                        "of": len(targets),
                        "seconds": round(generate_seconds),
                    }
                ),
                flush=True,
            )

    sheet = _edit_sheet(tiles, int(generation["samples"]), size)
    payload = b"".join(dumps(record) for record in records)
    _write_bytes_artifact(report_root / "drawings.jsonl", payload)
    _write_bytes_artifact(report_root / "samples.png", sheet)
    drawn = [r for r in records if "edit" in r and "skipped" not in r]
    scored = [r for r in drawn if "clip_to_edited" in r]
    per_edit: dict[str, Any] = {}
    for kind in edits:
        kind_drawn = [r for r in drawn if r["edit"] == kind]
        kind_scored = [r for r in scored if r["edit"] == kind]
        n_targets = len({r["hexcode"] for r in kind_drawn})
        per_edit[kind] = {
            "icons": n_targets,
            "skipped": sum(1 for r in records if r.get("edit") == kind and "skipped" in r),
            "drawings": len(kind_drawn),
            "ended_rate": _rate(kind_drawn, "decode_ended"),
            "codec_valid_rate": _rate(kind_drawn, "codec_valid"),
            "edit_reflected_rate": sum(1 for r in kind_scored if r["edit_reflected"])
            / max(len(kind_drawn), 1),
            "edited_top1_rate": sum(1 for r in kind_scored if r["edited_rank"] == 1)
            / max(len(kind_drawn), 1),
            "clip_to_edited_mean": float(np.mean([r["clip_to_edited"] for r in kind_scored]))
            if kind_scored
            else None,
            "clip_to_original_mean": float(np.mean([r["clip_to_original"] for r in kind_scored]))
            if kind_scored
            else None,
            "rgba_mae_to_edited_median": float(
                np.median([r["rgba_mae_to_edited"] for r in kind_scored])
            )
            if kind_scored
            else None,
            "chance_top1": 1.0 / max(n_targets, 1),
        }
    summary: dict[str, Any] = {
        "schema_version": 1,
        "study_version": version,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model": {
            "repo": OMNISVG_REPO,
            "revision": OMNISVG_REVISION,
            "checkpoint_sha256": model.checkpoint_sha256,
            "fine_tuned": adapter is not None,
            "adapter": str(adapter) if adapter else None,
            "adapter_sha256": adapter_sha256,
        },
        "clip": {"repo": CLIP_REPO, "revision": CLIP_REVISION},
        "icons": len(rows),
        "edits": edits,
        "sampling": sampling,
        "drawings": len(drawn),
        "per_edit": per_edit,
        "edit_reflected_rate": sum(1 for r in scored if r["edit_reflected"]) / max(len(drawn), 1),
        "edited_top1_rate": sum(1 for r in scored if r["edited_rank"] == 1) / max(len(drawn), 1),
        "ended_rate": _rate(drawn, "decode_ended"),
        "codec_valid_rate": _rate(drawn, "codec_valid"),
        "timing": {
            "load_seconds": load_seconds,
            "generate_seconds": generate_seconds,
            "seconds_per_drawing": generate_seconds / max(len(drawn), 1),
        },
        "sheet_sha256": hashlib.sha256(sheet).hexdigest(),
        "rows_sha256": hashlib.sha256(payload).hexdigest(),
    }
    if "criteria" in root:
        criteria = root["criteria"]
        checks: dict[str, bool] = {}
        if "min_edit_reflected_rate" in criteria:
            checks["edit_reflected_rate"] = summary["edit_reflected_rate"] >= float(
                criteria["min_edit_reflected_rate"]
            )
        if "min_edited_top1_rate" in criteria:
            checks["edited_top1_rate"] = summary["edited_top1_rate"] >= float(
                criteria["min_edited_top1_rate"]
            )
        if "min_ended_rate" in criteria:
            checks["ended_rate"] = summary["ended_rate"] >= float(criteria["min_ended_rate"])
        summary["criteria"] = criteria
        summary["checks"] = checks
        summary["predeclared_outcome"] = "passed" if all(checks.values()) else "falsified"
    _write_bytes_artifact(report_root / "summary.json", dumps(summary))
    return summary


def _rate(rows: list[dict[str, Any]], key: str) -> float:
    return sum(1 for r in rows if r.get(key)) / max(len(rows), 1)


def _edit_sheet(tiles: list[tuple[str, list[Image.Image]]], samples: int, size: int) -> bytes:
    import io

    from PIL import ImageDraw

    tile, label_width, header = size + 16, 150, 30
    columns = 2 + samples
    canvas = Image.new("RGB", (label_width + tile * columns, header + tile * len(tiles)), "white")
    draw = ImageDraw.Draw(canvas)
    for column, name in enumerate(
        ["original", "edited"] + [f"drawing {i}" for i in range(samples)]
    ):
        draw.text((label_width + tile * column + 4, 8), name, fill="black")
    for row_index, (label, images) in enumerate(tiles):
        y = header + row_index * tile
        draw.text((4, y + tile // 2), label[:24], fill="black")
        for column, image in enumerate(images):
            canvas.paste(image.resize((size, size)), (label_width + column * tile + 8, y + 8))
    payload = io.BytesIO()
    canvas.save(payload, format="PNG", optimize=True)
    return payload.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = run_omnisvg_edit_study(args.config)
    keys = (
        "drawings",
        "per_edit",
        "edit_reflected_rate",
        "edited_top1_rate",
        "ended_rate",
        "codec_valid_rate",
        "checks",
        "predeclared_outcome",
        "timing",
    )
    print(json.dumps({key: summary[key] for key in keys if key in summary}, indent=1))


if __name__ == "__main__":
    main()
