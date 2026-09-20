"""What the autoregressive model actually draws, next to what it was asked to draw.

Every number Gate I produces - held-out likelihood, a ratio against a marginal floor,
a distinctness rate - can be satisfied by a model that emits plausible token statistics
and no icon. Gate G's whole history is the argument for looking: its scores improved for
months while the renders showed nothing a person would recognise, and the moment the
renders were put beside the identity baseline the picture changed completely.

So this renders samples beside real icons from the same subgroup, at the same size, on
the same white ground. The comparison is the point: a row where the samples are visibly
not icons is a result, and so is a row where they are.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from mojidiff.learning.autoregressive import CausalProgramModel, SequenceLayout, generate
from mojidiff.learning.openmoji_pilot import (
    OpenMojiPilotError,
    PilotRow,
    _load_program,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.tiny_study import _decode_checkpoint
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.packed import serialize_packed_svg, validate_packed_tensor_program
from mojidiff.representation.renderer import RenderLimits, render_typed_svg_isolated

SHEET_FILENAME = "samples.png"


@dataclass(frozen=True)
class ARRenderConfig:
    version: str
    pilot_config: Path
    checkpoint: Path
    checkpoint_sha256: str
    report_root: Path
    d_model: int
    heads: int
    layers: int
    feedforward: int
    subgroups: int
    samples_per_subgroup: int
    seed: int
    render_size: int
    render_timeout_seconds: int


def load_ar_render_config(path: Path) -> ARRenderConfig:
    root = yaml.safe_load(path.read_bytes())
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise OpenMojiPilotError("ar render schema_version must be 1")
    model = root["model"]
    render = root["render"]
    return ARRenderConfig(
        version=str(root["study_version"]),
        pilot_config=Path(str(root["pilot_config"])),
        checkpoint=Path(str(root["checkpoint"])),
        checkpoint_sha256=str(root["checkpoint_sha256"]),
        report_root=Path(str(root["report_root"])),
        d_model=int(model["d_model"]),
        heads=int(model["heads"]),
        layers=int(model["layers"]),
        feedforward=int(model["feedforward"]),
        subgroups=int(root["subgroups"]),
        samples_per_subgroup=int(root["samples_per_subgroup"]),
        seed=int(root["seed"]),
        render_size=int(render["size"]),
        render_timeout_seconds=int(render["timeout_seconds"]),
    )


def run_ar_renders(config: ARRenderConfig, config_path: Path) -> dict[str, Any]:
    payload = config.checkpoint.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != config.checkpoint_sha256:
        raise OpenMojiPilotError(
            f"checkpoint hash mismatch: {digest} != {config.checkpoint_sha256}"
        )
    torch.use_deterministic_algorithms(True)
    pilot = load_openmoji_pilot_config(config.pilot_config)
    by_split, groups, subgroups = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalProgramModel(
        layout,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        feedforward=config.feedforward,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
    ).to(device)
    model.load_state_dict(_decode_checkpoint(payload)["model"])
    model.eval()

    chosen = _chosen_subgroups(by_split["primary/train"], config.subgroups, config.seed)
    limits = RenderLimits(
        max_paths=codec.max_paths, timeout_seconds=config.render_timeout_seconds
    )
    rows: list[tuple[str, list[np.ndarray[Any, Any]]]] = []
    records: list[dict[str, Any]] = []
    for row_index, (subgroup, exemplar) in enumerate(chosen):
        program = _load_program(exemplar, pilot, codec)
        tiles = [_raster(program, codec, pilot, limits, config.render_size)]
        for sample_index in range(config.samples_per_subgroup):
            seed = config.seed + 1000 * row_index + sample_index
            sampled, _ = generate(
                model,
                program,
                {
                    "group": torch.tensor(
                        [groups[exemplar.group]], dtype=torch.long, device=device
                    ),
                    "subgroup": torch.tensor(
                        [subgroups[subgroup]], dtype=torch.long, device=device
                    ),
                },
                greedy=False,
                rng=np.random.default_rng(seed),
            )
            validate_packed_tensor_program(sampled, codec, pilot.total_segment_slots)
            raster = _raster(sampled, codec, pilot, limits, config.render_size)
            tiles.append(raster)
            records.append(
                {
                    "subgroup": subgroup,
                    "sample": sample_index,
                    "seed": seed,
                    "ink_coverage": float((raster[:, :, 3] > 0).mean()),
                    "active_paths": int((sampled.path_length > 0).sum()),
                }
            )
        rows.append((subgroup, tiles))

    sheet = _sheet(rows, config)
    exemplar_coverage = [
        float((tiles[0][:, :, 3] > 0).mean()) for _, tiles in rows
    ]
    sample_coverage = [record["ink_coverage"] for record in records]
    summary = {
        "schema_version": 1,
        "study_version": config.version,
        "scope": "renders only; the numbers this sits beside are in the training run",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "checkpoint_sha256": digest,
        "device": str(device),
        "subgroups": [subgroup for subgroup, _ in rows],
        "samples": records,
        "median_sample_ink_coverage": float(np.median(sample_coverage)),
        "median_exemplar_ink_coverage": float(np.median(exemplar_coverage)),
        "median_sample_active_paths": float(
            np.median([record["active_paths"] for record in records])
        ),
        "sheet_sha256": hashlib.sha256(sheet).hexdigest(),
    }
    _write_bytes_artifact(config.report_root / SHEET_FILENAME, sheet)
    _write_bytes_artifact(
        config.report_root / "summary.json",
        (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    return summary


def _chosen_subgroups(
    rows: tuple[PilotRow, ...], count: int, seed: int
) -> list[tuple[str, PilotRow]]:
    """Subgroups picked by hash, with one real icon each to stand beside the samples.

    Picking by hash rather than by frequency matters: the most populous subgroups are
    the flags and the skin-tone people, which are the easiest sequences in the corpus
    and would make any model look better than it is.
    """

    ordered = sorted(
        rows, key=lambda row: hashlib.sha256(f"{seed}\0{row.source_path}".encode()).digest()
    )
    exemplars: dict[str, PilotRow] = {}
    for row in ordered:
        exemplars.setdefault(row.subgroup, row)
        if len(exemplars) == count:
            break
    if len(exemplars) < count:
        raise OpenMojiPilotError(f"only {len(exemplars)} subgroups available")
    return list(exemplars.items())


def _raster(
    program: Any, codec: Any, pilot: Any, limits: RenderLimits, size: int
) -> np.ndarray[Any, Any]:
    svg = serialize_packed_svg(program, codec, pilot.total_segment_slots)
    _, raster = render_typed_svg_isolated(svg, size, limits)
    return raster


def _sheet(
    rows: list[tuple[str, list[np.ndarray[Any, Any]]]], config: ARRenderConfig
) -> bytes:
    tile = config.render_size + 16
    label_width = 180
    header = 30
    columns = 1 + config.samples_per_subgroup
    canvas = Image.new(
        "RGB", (label_width + tile * columns, header + tile * len(rows)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((label_width + 4, 8), "real icon", fill="black")
    for index in range(config.samples_per_subgroup):
        draw.text((label_width + tile * (index + 1) + 4, 8), f"sample {index}", fill="black")
    for row_index, (subgroup, tiles) in enumerate(rows):
        y = header + row_index * tile
        draw.text((4, y + tile // 2), subgroup[:26], fill="black")
        for column, raster in enumerate(tiles):
            rgba = Image.fromarray(raster.astype(np.uint8), "RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            white.alpha_composite(rgba)
            canvas.paste(white.convert("RGB"), (label_width + column * tile + 8, y + 8))
    payload = io.BytesIO()
    canvas.save(payload, format="PNG", optimize=True)
    return payload.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_ar_render_config(args.config)
    print(json.dumps(run_ar_renders(config, args.config), sort_keys=True))


if __name__ == "__main__":
    main()
