"""Gate O: merging icons as a vector-to-vector task, with OpenMoji's own ground truth.

OpenMoji's ZWJ sequences are merged icons by definition - person surfing + female sign
= woman surfing, mushroom + brown square = brown mushroom - and every one whose
components are in the corpus is a pair of source programs and a target program in the
typed codec. This study writes each as text: the target's annotation, then each
component's annotation and compact canonical SVG, then the merged icon's SVG as the
suffix the model must write. The Qwen3.5-2B text prior is fine-tuned with LoRA on the
training targets and read on held-out targets, family-disjoint as always.

Two baselines that use no model set the floor: the components overlaid on one box in
order, and the first component alone. Every output is parsed by the codec, rendered,
and read against the true merged icon's render - the pixel error, CLIP similarity and
the rank of the true target among the held-out targets - so the question is exact:
does the prior learn OpenMoji's own composition rules better than stacking the parts?
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

from mojidiff.learning.omnisvg import parse_into_codec
from mojidiff.learning.omnisvg_study import (
    STEP_SECONDS,
    Clip,
    StepTimeout,
    _sheet,
    captions,
    guarded,
    render_program,
    render_raw,
)
from mojidiff.learning.openmoji_pilot import (
    OpenMojiPilotError,
    PilotRow,
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.prior import (
    SVG_CLOSE,
    SVG_OPEN,
    Example,
    Prior,
    adapter_bytes,
    compact_svg,
    dumps,
    extract_svg,
    sha256,
)
from mojidiff.learning.prior_study import train_lora
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.renderer import RenderLimits

_PATH = re.compile(r"<path\b[^>]*/>", re.IGNORECASE)


# ------------------------------------------------------------------------- pairs


def merge_pairs(
    by_split: dict[str, tuple[PilotRow, ...]],
) -> dict[str, list[tuple[PilotRow, list[PilotRow]]]]:
    """Every ZWJ target whose components are all in the corpus, keyed by the target's split."""

    rows = {row.hexcode: row for split in by_split.values() for row in split}

    def find(part: str) -> PilotRow | None:
        return rows.get(part) or rows.get(part + "-FE0F")

    pairs: dict[str, list[tuple[PilotRow, list[PilotRow]]]] = {}
    for split, split_rows in by_split.items():
        for row in split_rows:
            if "200D" not in row.hexcode:
                continue
            parts = [find(p.replace("-FE0F", "")) for p in row.hexcode.split("-200D-")]
            if any(part is None for part in parts):
                continue
            pairs.setdefault(split, []).append((row, [part for part in parts if part]))
    return pairs


def merge_prompt(target: str, components: list[tuple[str, str]]) -> str:
    """The text the model continues: what to make, then each part with its program."""

    lines = [f"<!-- merge: {target} -->"]
    for annotation, svg in components:
        lines.append(f"<!-- part: {annotation} -->")
        lines.append(svg)
    lines.append("<!-- merged -->")
    return "\n".join(lines) + "\n"


def build_merge_example(tokenizer: Any, hexcode: str, prompt: str, target_svg: str) -> Example:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + target_svg, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"the prompt's tokens change at the boundary for {hexcode}")
    eos = tokenizer.eos_token_id
    return Example(
        hexcode,
        prompt,
        target_svg,
        tuple(full_ids) + ((eos,) if eos is not None else ()),
        len(prompt_ids),
    )


def overlay(component_svgs: list[str]) -> str:
    """The no-model merge: every component's paths on one box, in order."""

    paths = "".join("".join(_PATH.findall(svg)) for svg in component_svgs)
    return SVG_OPEN + paths + SVG_CLOSE


# ------------------------------------------------------------------------- study


def run_merge_study(config_path: Path) -> dict[str, Any]:
    root = yaml.safe_load(config_path.read_text())
    if root.get("schema_version") != 1:
        raise OpenMojiPilotError("merge study schema_version must be 1")
    version = str(root["study_version"])
    mode = str(root.get("mode", "finetune"))
    report_root = Path(str(root["report_root"]))
    checkpoint_root = Path(str(root.get("checkpoint_root", report_root)))
    pilot = load_openmoji_pilot_config(Path(str(root["pilot_config"])))
    by_split, _, _ = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    slots = pilot.total_segment_slots
    model_cfg = root["model"]
    data = root["data"]
    generation = root["generation"]
    training = root.get("training", {})
    size = int(root.get("render", {}).get("size", 72))
    limits = RenderLimits(max_paths=codec.max_paths, timeout_seconds=20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    annotations = captions(pilot.raw_root)
    pairs = merge_pairs(by_split)

    texts: dict[str, str] = {}

    def text_of(row: PilotRow) -> str:
        if row.hexcode not in texts:
            texts[row.hexcode] = compact_svg(_load_program(row, pilot, codec), codec, slots)
        return texts[row.hexcode]

    def prompt_of(target: PilotRow, components: list[PilotRow]) -> str:
        return merge_prompt(
            annotations.get(target.hexcode, target.hexcode),
            [(annotations.get(c.hexcode, c.hexcode), text_of(c)) for c in components],
        )

    eval_pairs = _select_pairs(
        pairs.get("primary/validation", []), int(data["targets"]), pilot.seed + 1
    )
    eval_hexcodes = {target.hexcode for target, _ in eval_pairs}
    select_pairs = [
        pair
        for pair in _select_pairs(
            pairs.get("primary/validation", []),
            len(pairs.get("primary/validation", [])),
            pilot.seed + 2,
        )
        if pair[0].hexcode not in eval_hexcodes
    ][: int(training.get("held_out_targets", 32))]
    train_pairs = pairs.get("primary/train", [])
    if data.get("train_targets") is not None:
        train_pairs = _select_pairs(train_pairs, int(data["train_targets"]), pilot.seed)

    started = time.perf_counter()
    prior = Prior.load(
        str(model_cfg["source"]),
        str(model_cfg["revision"]),
        device=device,
        lora=training.get("lora") if mode == "finetune" else None,
    )
    load_seconds = time.perf_counter() - started

    train_report: dict[str, Any] = {}
    metrics: list[dict[str, Any]] = []
    if mode == "finetune":
        max_tokens = int(training["max_tokens"])
        train, excluded_train = _examples(prior, train_pairs, prompt_of, text_of, max_tokens)
        select, excluded_select = _examples(prior, select_pairs, prompt_of, text_of, max_tokens)
        train_report, metrics = train_lora(prior, train, select, training, device)
        train_report["training"].update(
            {
                "train_targets": len(train),
                "excluded_train_over_max_tokens": excluded_train,
                "selection_targets": len(select),
                "excluded_selection_over_max_tokens": excluded_select,
                "max_tokens": max_tokens,
            }
        )
        _write_bytes_artifact(
            report_root / "metrics.jsonl", b"".join(dumps(row) for row in metrics)
        )
        payload = adapter_bytes(prior.model)
        _write_bytes_artifact(checkpoint_root / "adapter.zip", payload)
        train_report["adapter_sha256"] = sha256(payload)
        prior.model.eval()

    train_targets = {text_of(target) for target, _ in pairs.get("primary/train", [])}
    evaluation, rows, sheet = _evaluate(
        prior,
        eval_pairs,
        prompt_of,
        text_of,
        annotations,
        pilot,
        codec,
        limits,
        size,
        device,
        generation,
        train_targets,
        mode,
    )
    rows_payload = b"".join(dumps(record) for record in rows)
    _write_bytes_artifact(report_root / "drawings.jsonl", rows_payload)
    _write_bytes_artifact(report_root / "samples.png", sheet)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "study_version": version,
        "mode": mode,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model": {
            "source": prior.source,
            "revision": prior.revision,
            "parameters": prior.parameters,
            "trainable_parameters": prior.trainable_parameters if mode == "finetune" else 0,
            "fine_tuned": mode == "finetune",
        },
        "pairs": {split: len(items) for split, items in pairs.items()},
        "timing": {"load_seconds": load_seconds},
        **train_report,
        **evaluation,
        "sheet_sha256": hashlib.sha256(sheet).hexdigest(),
        "rows_sha256": hashlib.sha256(rows_payload).hexdigest(),
    }
    if "criteria" in root:
        summary["criteria"] = root["criteria"]
        summary["checks"] = _checks(summary, root["criteria"], rows)
        summary["predeclared_outcome"] = (
            "passed" if all(summary["checks"].values()) else "falsified"
        )
    _write_bytes_artifact(report_root / "summary.json", dumps(summary))
    return summary


def _select_pairs(
    pairs: list[tuple[PilotRow, list[PilotRow]]], count: int, seed: int
) -> list[tuple[PilotRow, list[PilotRow]]]:
    by_target = {target.hexcode: (target, components) for target, components in pairs}
    chosen = _select_rows(tuple(target for target, _ in pairs), min(count, len(pairs)), seed)
    return [by_target[row.hexcode] for row in chosen]


def _examples(
    prior: Prior,
    pairs: list[tuple[PilotRow, list[PilotRow]]],
    prompt_of: Any,
    text_of: Any,
    max_tokens: int,
) -> tuple[list[Example], int]:
    examples, excluded = [], 0
    for target, components in pairs:
        example = build_merge_example(
            prior.tokenizer, target.hexcode, prompt_of(target, components), text_of(target)
        )
        if example.length > max_tokens:
            excluded += 1
            continue
        examples.append(example)
    return examples, excluded


def _evaluate(
    prior: Prior,
    eval_pairs: list[tuple[PilotRow, list[PilotRow]]],
    prompt_of: Any,
    text_of: Any,
    annotations: dict[str, str],
    pilot: Any,
    codec: Any,
    limits: RenderLimits,
    size: int,
    device: str,
    generation: dict[str, Any],
    train_targets: set[str],
    mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]], bytes]:
    """The model's merge and the two baselines, each read against the true merged icon."""

    slots = pilot.total_segment_slots
    clip = Clip(device)
    target_renders = [
        render_program(_load_program(target, pilot, codec), codec, slots, limits, size)
        for target, _ in eval_pairs
    ]
    target_features = clip.image(target_renders)
    max_new_tokens = int(generation["max_new_tokens"])
    seed = int(generation["seed"])
    arms = ["model"] if mode == "finetune" else []
    arms += ["overlay", "first_component"]
    rows: list[dict[str, Any]] = []
    tiles: list[tuple[str, list[Image.Image]]] = []
    generate_seconds = 0.0
    for index, (target, components) in enumerate(eval_pairs):
        annotation = annotations.get(target.hexcode, target.hexcode)
        target_array = np.asarray(target_renders[index], dtype=np.float32)
        row_tiles = [target_renders[index]]
        candidates: dict[str, str | None] = {}
        record_extra: dict[str, dict[str, Any]] = {}
        if "model" in arms:
            clock = time.perf_counter()
            text, tokens, ended = prior.generate(
                prompt_of(target, components) + SVG_OPEN,
                max_new_tokens=max_new_tokens,
                seed=seed + index,
                greedy=True,
            )
            generate_seconds += time.perf_counter() - clock
            candidates["model"] = extract_svg(SVG_OPEN + text)
            record_extra["model"] = {"tokens": tokens, "ended": ended}
        candidates["overlay"] = overlay([text_of(c) for c in components])
        candidates["first_component"] = text_of(components[0])
        for arm in arms:
            svg = candidates[arm]
            record: dict[str, Any] = {
                "hexcode": target.hexcode,
                "annotation": annotation,
                "components": [c.hexcode for c in components],
                "arm": arm,
                **record_extra.get(arm, {}),
            }
            image: Image.Image | None = None
            if svg is None:
                record["failure"] = "no_closed_svg"
                record["codec_valid"] = False
            else:
                record["svg"] = svg
                record["memorised_exactly"] = arm == "model" and svg in train_targets
                try:
                    program, info = guarded(
                        STEP_SECONDS, parse_into_codec, svg.encode(), codec, slots
                    )
                except StepTimeout:
                    program, info = None, {"failure": "parse_timeout"}
                record.update({f"codec_{key}": value for key, value in info.items()})
                record["codec_valid"] = program is not None
                try:
                    image = guarded(STEP_SECONDS, render_raw, svg, size)
                except Exception as error:  # noqa: BLE001
                    record["raw_render_error"] = type(error).__name__
                if program is not None:
                    image = render_program(program, codec, slots, limits, size)
            if image is not None:
                feature = clip.image([image])
                similarities = (feature @ target_features.T)[0]
                own = float(similarities[index])
                record["clip_to_target"] = own
                record["target_rank"] = 1 + int((similarities > own).sum())
                record["rgba_mae_to_target"] = float(
                    np.abs(np.asarray(image, dtype=np.float32) - target_array).mean() / 255.0
                )
            row_tiles.append(
                image if image is not None else Image.new("RGB", (size, size), "white")
            )
            rows.append(record)
        tiles.append((f"{target.hexcode} {annotation[:20]}", row_tiles))
        print(
            json.dumps(
                {"target": index + 1, "of": len(eval_pairs), "seconds": round(generate_seconds)}
            ),
            flush=True,
        )
    sheet = _sheet(tiles, len(arms), size)
    per_arm: dict[str, Any] = {}
    for arm in arms:
        arm_rows = [r for r in rows if r["arm"] == arm]
        scored = [r for r in arm_rows if "clip_to_target" in r]
        per_arm[arm] = {
            "targets": len(arm_rows),
            "codec_valid_rate": sum(1 for r in arm_rows if r.get("codec_valid"))
            / max(len(arm_rows), 1),
            "rendered_rate": len(scored) / max(len(arm_rows), 1),
            "rgba_mae_to_target_median": float(np.median([r["rgba_mae_to_target"] for r in scored]))
            if scored
            else None,
            "rgba_mae_to_target_mean": float(np.mean([r["rgba_mae_to_target"] for r in scored]))
            if scored
            else None,
            "clip_to_target_mean": float(np.mean([r["clip_to_target"] for r in scored]))
            if scored
            else None,
            "target_top1_rate": sum(1 for r in scored if r["target_rank"] == 1)
            / max(len(arm_rows), 1),
            "chance_top1": 1.0 / max(len(eval_pairs), 1),
        }
        if arm == "model":
            per_arm[arm]["ended_rate"] = sum(1 for r in arm_rows if r.get("ended")) / max(
                len(arm_rows), 1
            )
            per_arm[arm]["memorised_exactly"] = sum(
                1 for r in arm_rows if r.get("memorised_exactly")
            )
            per_arm[arm]["median_tokens"] = (
                float(np.median([r["tokens"] for r in arm_rows])) if arm_rows else None
            )
    summary: dict[str, Any] = {
        "targets": len(eval_pairs),
        "arms": per_arm,
        "generation": {
            "max_new_tokens": max_new_tokens,
            "seconds_per_merge": generate_seconds / max(len(eval_pairs), 1),
        },
    }
    if "model" in arms:
        summary["model_vs_baselines"] = {
            arm: _paired_error_gain(rows, "model", arm) for arm in ("overlay", "first_component")
        }
    return summary, rows, sheet


def _paired_error_gain(
    rows: list[dict[str, Any]], arm: str, against: str, resamples: int = 10_000
) -> dict[str, Any]:
    """How much lower the arm's pixel error is than the baseline's, paired by target."""

    def per_target(name: str) -> dict[str, float]:
        return {
            r["hexcode"]: float(r["rgba_mae_to_target"])
            for r in rows
            if r["arm"] == name and "rgba_mae_to_target" in r
        }

    ours, theirs = per_target(arm), per_target(against)
    common = sorted(set(ours) & set(theirs))
    if not common:
        return {"paired_targets": 0}
    diffs = np.array([theirs[h] - ours[h] for h in common])  # positive: the arm is closer
    rng = np.random.default_rng(0)
    means = np.array(
        [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(resamples)]
    )
    return {
        "paired_targets": len(common),
        "error_reduction_mean": float(diffs.mean()),
        "error_reduction_ci95": [
            float(np.quantile(means, 0.025)),
            float(np.quantile(means, 0.975)),
        ],
        "targets_closer": int((diffs > 0).sum()),
    }


def _checks(
    summary: dict[str, Any], criteria: dict[str, Any], rows: list[dict[str, Any]]
) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    model = summary["arms"].get("model", {})
    if "min_codec_valid_rate" in criteria:
        checks["codec_valid_rate"] = model.get("codec_valid_rate", 0.0) >= float(
            criteria["min_codec_valid_rate"]
        )
    if criteria.get("closer_than_overlay_ci_excludes_zero"):
        gain = summary.get("model_vs_baselines", {}).get("overlay", {})
        checks["closer_than_overlay"] = bool(gain) and gain.get("error_reduction_ci95", [0])[0] > 0
    if criteria.get("closer_than_first_component_ci_excludes_zero"):
        gain = summary.get("model_vs_baselines", {}).get("first_component", {})
        checks["closer_than_first_component"] = (
            bool(gain) and gain.get("error_reduction_ci95", [0])[0] > 0
        )
    if "min_target_top1_rate" in criteria:
        checks["target_top1_rate"] = model.get("target_top1_rate", 0.0) >= float(
            criteria["min_target_top1_rate"]
        )
    if "max_memorised_exactly" in criteria:
        checks["memorisation"] = model.get("memorised_exactly", 0) <= int(
            criteria["max_memorised_exactly"]
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = run_merge_study(args.config)
    keys = (
        "mode",
        "pairs",
        "targets",
        "arms",
        "model_vs_baselines",
        "training",
        "checks",
        "predeclared_outcome",
        "generation",
    )
    print(json.dumps({key: summary[key] for key in keys if key in summary}, indent=1))


if __name__ == "__main__":
    main()
