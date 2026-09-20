"""Render what a trained Gate G pilot checkpoint actually predicts on held-out icons.

The pilot reports scalars - loss and per-token accuracy - and nothing you can look at.
At 0.0573 changed-token recovery the scalars alone cannot answer the question a reader
will ask first: is the prediction a recognizable icon with some fields wrong, or is it
rubble?  This probe answers that by rendering the triple the project labels explicitly:

* `x_0`     - the clean program;
* `x_t`     - the raw corrupted state the model is given, never relabelled;
* `x_hat_0` - the model's predicted clean state.

It is strictly read-only with respect to training.  It restores a committed checkpoint,
reuses the pilot's own held-out corruption draw so the renders show the same inputs the
reported metrics were computed on, and writes only renders and render metrics.  It
trains nothing and changes no learning artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml

from mojidiff.learning.geometry import (
    corrupt_factorized_geometry,
    packed_batch,
    predict_clean_geometry,
)
from mojidiff.learning.openmoji_pilot import (
    OpenMojiPilotConfig,
    PilotRow,
    _condition,
    _device,
    _load_program,
    _new_model,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.tiny_study import _contact_sheet, _decode_checkpoint
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.packed import serialize_packed_svg
from mojidiff.representation.program import CodecConfig
from mojidiff.representation.renderer import RenderLimits, render_typed_svg_isolated
from mojidiff.representation.study import _similarity

SUMMARY_FILENAME = "summary.json"
SHEET_FILENAME = "trajectory.png"
METRICS_FILENAME = "render-metrics.jsonl"

_EVAL_SEED_OFFSET = 9_000_000
"""Matches `openmoji_pilot._evaluate`, so renders show the evaluated corruption draw."""


class PilotRenderError(RuntimeError):
    """The render probe is invalid or failed closed."""


@dataclass(frozen=True)
class PilotRenderConfig:
    version: str
    pilot_config: Path
    checkpoint: Path
    checkpoint_sha256: str
    report_root: Path
    icons: int
    render_sizes: tuple[int, ...]
    render_timeout_seconds: int
    corruption_probability: float | None = None
    """Evaluate at a different corruption level than the checkpoint trained at.

    Held-out recovery is reported at whatever probability the training config used, so
    a model trained at 0.35 has only ever been measured on a task where a third of its
    geometry is gone. Overriding this sweeps one fixed checkpoint across corruption
    levels, which separates what the model cannot do from what the corruption regime
    has already destroyed. `None` keeps the training probability, so existing probe
    configs are unaffected.
    """


def load_pilot_render_config(path: Path) -> PilotRenderConfig:
    root = yaml.safe_load(path.read_bytes())
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise PilotRenderError("pilot render schema_version must be 1")
    sizes = root.get("render_sizes")
    if not isinstance(sizes, list) or not sizes:
        raise PilotRenderError("render_sizes must be a non-empty list")
    return PilotRenderConfig(
        version=_string(root, "study_version"),
        pilot_config=Path(_string(root, "pilot_config")),
        checkpoint=Path(_string(root, "checkpoint")),
        checkpoint_sha256=_hex(_string(root, "checkpoint_sha256")),
        report_root=Path(_string(root, "report_root")),
        icons=_positive_int(root.get("icons"), "icons"),
        render_sizes=tuple(_positive_int(size, "render_size") for size in sizes),
        render_timeout_seconds=_positive_int(
            root.get("render_timeout_seconds", 20), "render_timeout_seconds"
        ),
        corruption_probability=_optional_probability(root.get("corruption_probability")),
    )


def _optional_probability(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 1:
        raise PilotRenderError("corruption_probability must be within (0, 1]")
    return float(value)


def run_pilot_renders(config: PilotRenderConfig, config_path: Path) -> dict[str, Any]:
    """Restore a pilot checkpoint and render its held-out predictions."""

    pilot = load_openmoji_pilot_config(config.pilot_config)
    payload = config.checkpoint.read_bytes()
    actual = hashlib.sha256(payload).hexdigest()
    if actual != config.checkpoint_sha256:
        raise PilotRenderError(
            f"checkpoint hash mismatch: expected {config.checkpoint_sha256}, got {actual}"
        )
    document = _decode_checkpoint(payload)

    by_split, groups, subgroups = load_pilot_index(pilot)
    validation_rows = _select_rows(
        by_split["primary/validation"], pilot.validation_samples, pilot.seed + 1
    )
    if config.icons > len(validation_rows):
        raise PilotRenderError("icons exceeds the pilot's held-out draw")
    codec = _selected_codec(pilot)
    device = _device("cpu")
    model = _new_model(pilot, codec, groups, subgroups).to(device)
    model.load_state_dict(document["model"])
    model.eval()

    limits = RenderLimits(
        max_paths=pilot.max_paths, timeout_seconds=config.render_timeout_seconds
    )
    probability = (
        pilot.corruption_probability
        if config.corruption_probability is None
        else config.corruption_probability
    )
    rows: list[dict[str, Any]] = []
    tiles: list[tuple[str, dict[str, np.ndarray[Any, Any]]]] = []
    for index in range(config.icons):
        row = validation_rows[index]
        clean = _load_program(row, pilot, codec)
        # The identical generator the pilot's own held-out evaluation uses, so these
        # renders are of the very inputs the reported scalars were measured on.
        noisy = corrupt_factorized_geometry(
            clean,
            codec,
            probability,
            np.random.default_rng(pilot.seed + _EVAL_SEED_OFFSET + index),
        )
        with torch.no_grad():
            logits = model(
                packed_batch([noisy], device),
                _condition(
                    (row,),
                    groups,
                    subgroups,
                    device,
                    # A checkpoint trained with noise-level conditioning must be told
                    # the level it is being evaluated at, including in a sweep.
                    [probability] if pilot.noise_level_features else None,
                ),
            )
        prediction = predict_clean_geometry(noisy, logits, codec)
        rows.extend(
            _render_one(
                row, {"x_0": clean, "x_t": noisy, "x_hat_0": prediction},
                codec, pilot, config, limits, tiles,
            )
        )

    metrics_payload = b"".join(
        (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode() for row in rows
    )
    sheet = _contact_sheet(tiles, config.render_sizes)
    summary = {
        "schema_version": 1,
        "study_version": config.version,
        "scope": "read-only render probe over a trained pilot checkpoint; trains nothing",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "pilot_config": str(config.pilot_config),
        "pilot_config_sha256": hashlib.sha256(config.pilot_config.read_bytes()).hexdigest(),
        "checkpoint_sha256": config.checkpoint_sha256,
        "checkpoint_step": int(document["step"]),
        "icons": config.icons,
        "render_sizes": list(config.render_sizes),
        "corruption_probability": probability,
        "trained_corruption_probability": pilot.corruption_probability,
        "trained_corruption_probability_max": pilot.corruption_probability_max,
        "noise_level_conditioned": pilot.noise_level_features > 0,
        "corruption_probability_overridden": config.corruption_probability is not None,
        "corruption": "factorized_role_uniform_geometry, the pilot's held-out draw",
        "rendered_rows": [row.source_path for row in validation_rows[: config.icons]],
        "aggregate": _aggregate(rows),
        "render_metrics_sha256": hashlib.sha256(metrics_payload).hexdigest(),
        "sheet_sha256": hashlib.sha256(sheet).hexdigest(),
    }
    _write_bytes_artifact(config.report_root / METRICS_FILENAME, metrics_payload)
    _write_bytes_artifact(config.report_root / SHEET_FILENAME, sheet)
    _write_bytes_artifact(
        config.report_root / SUMMARY_FILENAME,
        (json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n").encode(),
    )
    _write_bytes_artifact(config.report_root / "README.md", _markdown(summary).encode())
    return summary


def _render_one(
    row: PilotRow,
    programs: dict[str, Any],
    codec: CodecConfig,
    pilot: OpenMojiPilotConfig,
    config: PilotRenderConfig,
    limits: RenderLimits,
    tiles: list[tuple[str, dict[str, np.ndarray[Any, Any]]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    images: dict[str, np.ndarray[Any, Any]] = {}
    for size in config.render_sizes:
        rendered = {
            name: render_typed_svg_isolated(
                serialize_packed_svg(program, codec, pilot.total_segment_slots), size, limits
            )[1]
            for name, program in programs.items()
        }
        images.update({f"{name}-{size}": image for name, image in rendered.items()})
        for name in ("x_t", "x_hat_0"):
            rows.append(
                {
                    "hexcode": row.hexcode,
                    "source_path": row.source_path,
                    "group": row.group,
                    "size": size,
                    "state": name,
                    **_similarity(rendered["x_0"], rendered[name]),
                }
            )
    tiles.append((row.hexcode, images))
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Median RGBA error per state and size, so the sheet has a numeric companion."""

    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(f"{row['state']}-{row['size']}", []).append(
            float(cast(float, row["rgba_mae"]))
        )
    return {
        key: {"median_rgba_mae": float(np.median(values)), "count": float(len(values))}
        for key, values in sorted(grouped.items())
    }


def _markdown(summary: dict[str, Any]) -> str:
    aggregate = cast(dict[str, Any], summary["aggregate"])
    lines = [
        "# Held-out render probe",
        "",
        f"Checkpoint step {summary['checkpoint_step']}, {summary['icons']} held-out icons, "
        f"corruption probability {summary['corruption_probability']}"
        + (
            f" (trained at {summary['trained_corruption_probability']})."
            if summary.get("corruption_probability_overridden")
            else "."
        ),
        "",
        "`x_0` is the clean program, `x_t` the raw corrupted state the model is given, "
        "and `x_hat_0` the model's predicted clean state. No safety projection or "
        "constrained decoding is applied.",
        "",
        "| state and size | median RGBA MAE against x_0 | renders |",
        "| --- | ---: | ---: |",
    ]
    for key, values in aggregate.items():
        lines.append(f"| {key} | {values['median_rgba_mae']:.6f} | {int(values['count'])} |")
    lines.extend(["", "This probe trains nothing and changes no learning artifact.", ""])
    return "\n".join(lines)


def _string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise PilotRenderError(f"{field} must be a non-empty string")
    return value


def _hex(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PilotRenderError("expected a lowercase SHA-256 value")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PilotRenderError(f"{field} must be a positive integer")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_pilot_render_config(args.config)
    print(json.dumps(run_pilot_renders(config, args.config), sort_keys=True))


if __name__ == "__main__":
    main()
