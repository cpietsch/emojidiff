"""Gate M, experiment two: the zero-shot control and the LoRA fine-tune of a text prior.

Two modes under one config. `control` prompts the unfine-tuned model with each held-out
icon's caption and the open tag and lets it write the icon. `finetune` trains LoRA
adapters on the training split with the loss on the SVG only, selects on held-out
suffix likelihood, saves the adapter as one hashed archive, and then generates for the
same held-out icons. Both modes score every output the same way as OmniSVG's study:
parsed by the typed codec, rendered, and compared with a pinned CLIP against the
held-out icon's render and against its caption, with the sheet beside the numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
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
    Example,
    Prior,
    adapter_bytes,
    batch_tensors,
    build_example,
    compact_svg,
    control_prefix,
    dumps,
    extract_svg,
    sha256,
)
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.renderer import RenderLimits


def run_prior_study(config_path: Path) -> dict[str, Any]:
    root = yaml.safe_load(config_path.read_text())
    if root.get("schema_version") != 1:
        raise OpenMojiPilotError("prior study schema_version must be 1")
    version = str(root["study_version"])
    mode = str(root["mode"])
    if mode not in {"control", "finetune", "evaluate"}:
        raise OpenMojiPilotError("mode must be control, finetune or evaluate")
    report_root = Path(str(root["report_root"]))
    checkpoint_root = Path(str(root.get("checkpoint_root", report_root)))
    pilot = load_openmoji_pilot_config(Path(str(root["pilot_config"])))
    by_split, _, _ = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    model_cfg = root["model"]
    data = root["data"]
    generation = root["generation"]
    training = root.get("training", {})
    size = int(root.get("render", {}).get("size", 72))
    limits = RenderLimits(max_paths=codec.max_paths, timeout_seconds=20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    annotations = captions(pilot.raw_root)

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
    if mode == "evaluate":
        # A saved adapter, read again: the same evaluation without training.
        from mojidiff.learning.prior import load_adapter

        payload = Path(str(root["adapter"])).read_bytes()
        prior.model = load_adapter(prior.model, payload).eval()
        train_report["adapter_sha256"] = sha256(payload)
        train_report["adapter"] = str(root["adapter"])
    if mode == "finetune":
        train_report, metrics = _finetune(
            prior, pilot, codec, by_split, annotations, data, training, device
        )
        _write_bytes_artifact(
            report_root / "metrics.jsonl", b"".join(dumps(row) for row in metrics)
        )
        payload = adapter_bytes(prior.model)
        _write_bytes_artifact(checkpoint_root / "adapter.zip", payload)
        train_report["adapter_sha256"] = sha256(payload)
        prior.model.eval()

    evaluation = _evaluate(
        prior,
        pilot,
        codec,
        by_split,
        annotations,
        data,
        generation,
        mode,
        limits,
        size,
        device,
        report_root,
    )
    summary = {
        "schema_version": 1,
        "study_version": version,
        "mode": mode,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model": {
            "source": prior.source,
            "revision": prior.revision,
            "parameters": prior.parameters,
            "trainable_parameters": prior.trainable_parameters if mode == "finetune" else 0,
            "fine_tuned": mode != "control",
        },
        "timing": {"load_seconds": load_seconds},
        **train_report,
        **evaluation,
    }
    if "control_report_root" in root:
        from mojidiff.learning.omnisvg_finetune import compare_to_control

        rows_drawn = [
            json.loads(line)
            for line in (report_root / "drawings.jsonl").read_text().splitlines()
            if line.strip()
        ]
        summary["control"] = compare_to_control(rows_drawn, Path(str(root["control_report_root"])))
    if "criteria" in root:
        summary["criteria"] = root["criteria"]
        summary["checks"] = _checks(summary, root["criteria"])
        summary["predeclared_outcome"] = (
            "passed" if all(summary["checks"].values()) else "falsified"
        )
    _write_bytes_artifact(report_root / "summary.json", dumps(summary))
    return summary


def _examples(
    prior: Prior,
    rows: tuple[PilotRow, ...],
    pilot: Any,
    codec: Any,
    annotations: dict[str, str],
    max_tokens: int,
) -> tuple[list[Example], int]:
    examples, excluded = [], 0
    for row in rows:
        svg = compact_svg(_load_program(row, pilot, codec), codec, pilot.total_segment_slots)
        example = build_example(
            prior.tokenizer, row.hexcode, annotations.get(row.hexcode, row.hexcode), svg
        )
        if example.length > max_tokens:
            excluded += 1  # never train on a silently truncated icon
            continue
        examples.append(example)
    return examples, excluded


def _finetune(
    prior: Prior,
    pilot: Any,
    codec: Any,
    by_split: dict[str, tuple[PilotRow, ...]],
    annotations: dict[str, str],
    data: dict[str, Any],
    training: dict[str, Any],
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    max_tokens = int(training["max_tokens"])
    train_rows = by_split["primary/train"]
    if data.get("train_icons") is not None:
        train_rows = _select_rows(train_rows, int(data["train_icons"]), pilot.seed)
    train, excluded_train = _examples(prior, train_rows, pilot, codec, annotations, max_tokens)
    held_rows = _select_rows(
        by_split["primary/validation"], int(training["held_out_icons"]), pilot.seed + 1
    )
    held, excluded_held = _examples(prior, held_rows, pilot, codec, annotations, max_tokens)
    if not train or not held:
        raise OpenMojiPilotError("no examples fit under max_tokens")

    steps = int(training["steps"])
    accumulate = int(training.get("accumulate", 8))
    chunk = int(training.get("loss_chunk", 1024))
    eval_every = int(training["eval_every"])
    patience = int(training.get("patience_evals", 8))
    optimizer = torch.optim.AdamW(
        [p for p in prior.model.parameters() if p.requires_grad],
        lr=float(training["learning_rate"]),
        weight_decay=float(training.get("weight_decay", 0.0)),
    )
    warmup = int(training.get("warmup_steps", 50))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min(1.0, (step + 1) / warmup)
    )
    rng = np.random.default_rng(int(training["seed"]))
    torch.manual_seed(int(training["seed"]))

    def held_out_nll() -> float:
        prior.model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for example in held:
                ids, labels = batch_tensors(example, device)
                loss, n = prior.suffix_loss(ids, labels, chunk)
                total, count = total + float(loss), count + n
        prior.model.train()
        return total / max(count, 1)

    metrics: list[dict[str, Any]] = [{"step": 0, "held_out_nll": held_out_nll()}]
    best_nll, best_step, best_state, stale = metrics[0]["held_out_nll"], 0, None, 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    prior.model.train()
    order = rng.permutation(len(train))
    cursor = 0
    running = 0.0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        tokens_in_step = 0
        for _ in range(accumulate):
            if cursor >= len(order):
                order, cursor = rng.permutation(len(train)), 0
            example = train[order[cursor]]
            cursor += 1
            ids, labels = batch_tensors(example, device)
            loss, n = prior.suffix_loss(ids, labels, chunk)
            (loss / (accumulate * max(n, 1))).backward()  # type: ignore[no-untyped-call]
            running += float(loss.detach())
            tokens_in_step += n
        torch.nn.utils.clip_grad_norm_(
            [p for p in prior.model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        schedule.step()
        if step % eval_every == 0 or step == steps:
            nll = held_out_nll()
            metrics.append(
                {
                    "step": step,
                    "train_loss": running / max(tokens_in_step, 1) / 1.0,
                    "held_out_nll": nll,
                }
            )
            running = 0.0
            if nll < best_nll:
                best_nll, best_step, stale = nll, step, 0
                best_state = {
                    k: v.detach().cpu().clone()
                    for k, v in prior.model.state_dict().items()
                    if "lora" in k.lower()
                }
            else:
                stale += 1
                if stale >= patience:
                    break
    train_seconds = time.perf_counter() - started
    if best_state is not None:
        prior.model.load_state_dict(best_state, strict=False)
    return {
        "training": {
            "train_icons": len(train),
            "excluded_train_over_max_tokens": excluded_train,
            "held_out_icons": len(held),
            "excluded_held_over_max_tokens": excluded_held,
            "max_tokens": max_tokens,
            "steps_run": metrics[-1]["step"],
            "sequences_per_step": accumulate,
            "selected_step": best_step,
            "initial_held_out_nll": metrics[0]["held_out_nll"],
            "held_out_nll": best_nll,
            "train_seconds": train_seconds,
            "peak_vram_gib": float(torch.cuda.max_memory_allocated()) / 2**30
            if device == "cuda"
            else None,
        }
    }, metrics


def _evaluate(
    prior: Prior,
    pilot: Any,
    codec: Any,
    by_split: dict[str, tuple[PilotRow, ...]],
    annotations: dict[str, str],
    data: dict[str, Any],
    generation: dict[str, Any],
    mode: str,
    limits: RenderLimits,
    size: int,
    device: str,
    report_root: Path,
) -> dict[str, Any]:
    icons = int(data["icons"])
    samples = int(generation["samples"])
    greedy_first = bool(generation.get("greedy_first", True))
    max_new_tokens = int(generation["max_new_tokens"])
    seed = int(generation["seed"])
    rows = _select_rows(by_split["primary/validation"], icons, pilot.seed + 1)
    train_texts = set()
    if mode != "control":
        for row in by_split["primary/train"]:
            train_texts.add(
                compact_svg(_load_program(row, pilot, codec), codec, pilot.total_segment_slots)
            )
    clip = Clip(device)
    texts = [annotations.get(row.hexcode, row.hexcode) for row in rows]
    references = [
        render_program(
            _load_program(row, pilot, codec), codec, pilot.total_segment_slots, limits, size
        )
        for row in rows
    ]
    reference_features = clip.image(references)
    caption_features = clip.text(texts)
    records: list[dict[str, Any]] = []
    tiles: list[tuple[str, list[Image.Image]]] = []
    generate_seconds = 0.0
    for index, row in enumerate(rows):
        annotation = texts[index]
        row_tiles = [references[index]]
        for sample in range(samples):
            greedy = greedy_first and sample == 0
            clock = time.perf_counter()
            text, tokens, ended = prior.generate(
                control_prefix(annotation),
                max_new_tokens=max_new_tokens,
                seed=seed + 1000 * index + sample,
                greedy=greedy,
            )
            generate_seconds += time.perf_counter() - clock
            record: dict[str, Any] = {
                "hexcode": row.hexcode,
                "annotation": annotation,
                "sample": sample,
                "greedy": greedy,
                "tokens": tokens,
                "ended": ended,
            }
            svg = extract_svg(control_prefix(annotation) + text)
            image: Image.Image | None = None
            if svg is None:
                record["failure"] = "no_closed_svg"
                record["codec_valid"] = False
            else:
                record["svg"] = svg
                record["svg_chars"] = len(svg)
                record["memorised_exactly"] = svg in train_texts
                try:
                    program, info = guarded(
                        STEP_SECONDS,
                        parse_into_codec,
                        svg.encode(),
                        codec,
                        pilot.total_segment_slots,
                    )
                except StepTimeout:
                    program, info = None, {"failure": "parse_timeout"}
                record.update({f"codec_{key}": value for key, value in info.items()})
                record["codec_valid"] = program is not None
                try:
                    image = guarded(STEP_SECONDS, render_raw, svg, size)
                    record["raw_render"] = True
                except Exception as error:  # noqa: BLE001
                    record["raw_render"] = False
                    record["raw_render_error"] = type(error).__name__
                if program is not None:
                    image = render_program(program, codec, pilot.total_segment_slots, limits, size)
            if image is not None:
                feature = clip.image([image])
                similarities = (feature @ reference_features.T)[0]
                own = float(similarities[index])
                record["clip_to_reference"] = own
                record["reference_rank"] = 1 + int((similarities > own).sum())
                record["clip_to_caption"] = float(
                    (feature @ caption_features[index : index + 1].T)[0, 0]
                )
            row_tiles.append(
                image if image is not None else Image.new("RGB", (size, size), "white")
            )
            records.append(record)
        tiles.append((f"{row.hexcode} {annotation[:22]}", row_tiles))
        print(
            json.dumps(
                {"icon": index + 1, "hexcode": row.hexcode, "seconds": round(generate_seconds)}
            ),
            flush=True,
        )
    sheet = _sheet(tiles, samples, size)
    payload = b"".join(dumps(record) for record in records)
    _write_bytes_artifact(report_root / "drawings.jsonl", payload)
    _write_bytes_artifact(report_root / "samples.png", sheet)
    scored = [r for r in records if "clip_to_reference" in r]
    ranks = [r["reference_rank"] for r in scored]
    return {
        "icons": len(rows),
        "samples_per_icon": samples,
        "drawings": len(records),
        "closed_svg_rate": sum(1 for r in records if "svg_chars" in r) / len(records),
        "codec_valid_rate": sum(1 for r in records if r.get("codec_valid")) / len(records),
        "raw_render_rate": sum(1 for r in records if r.get("raw_render")) / len(records),
        "ended_rate": sum(1 for r in records if r.get("ended")) / len(records),
        "memorised_exactly": sum(1 for r in records if r.get("memorised_exactly")),
        "failures": _count(records, "failure") | _count(records, "codec_failure"),
        "median_tokens": float(np.median([r["tokens"] for r in records])),
        "clip_to_reference": _stats([r["clip_to_reference"] for r in scored]),
        "clip_to_caption": _stats([r["clip_to_caption"] for r in scored]),
        "reference_rank": {
            "scored": len(ranks),
            "top1_rate": sum(1 for rank in ranks if rank == 1) / len(records),
            "top5_rate": sum(1 for rank in ranks if rank <= 5) / len(records),
            "mean": float(np.mean(ranks)) if ranks else None,
            "chance_top1": 1.0 / len(rows),
        },
        "generation": {
            "max_new_tokens": max_new_tokens,
            "seconds_per_drawing": generate_seconds / len(records),
        },
        "sheet_sha256": hashlib.sha256(sheet).hexdigest(),
        "rows_sha256": hashlib.sha256(payload).hexdigest(),
    }


def _checks(summary: dict[str, Any], criteria: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if "min_codec_valid_rate" in criteria:
        checks["codec_valid_rate"] = summary["codec_valid_rate"] >= float(
            criteria["min_codec_valid_rate"]
        )
    if "min_clip_to_reference_mean" in criteria:
        value = summary["clip_to_reference"]["mean"]
        checks["clip_to_reference"] = value is not None and value >= float(
            criteria["min_clip_to_reference_mean"]
        )
    if "max_memorised_exactly" in criteria:
        checks["memorisation"] = summary["memorised_exactly"] <= int(
            criteria["max_memorised_exactly"]
        )
    if criteria.get("clip_gain_over_control_ci_excludes_zero"):
        control: dict[str, Any] = summary.get("control") or {}
        checks["clip_gain_over_control"] = bool(control) and float(control["gain_ci95"][0]) > 0
    if "min_reference_top1_rate" in criteria:
        checks["reference_top1_rate"] = summary["reference_rank"]["top1_rate"] >= float(
            criteria["min_reference_top1_rate"]
        )
    if "min_ended_rate" in criteria:
        checks["ended_rate"] = summary["ended_rate"] >= float(criteria["min_ended_rate"])
    return checks


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = run_prior_study(args.config)
    keys = (
        "mode",
        "drawings",
        "closed_svg_rate",
        "codec_valid_rate",
        "raw_render_rate",
        "failures",
        "median_tokens",
        "clip_to_reference",
        "clip_to_caption",
        "reference_rank",
        "memorised_exactly",
        "training",
        "control",
        "checks",
        "predeclared_outcome",
    )
    print(json.dumps({key: summary[key] for key in keys if key in summary}, indent=1))


if __name__ == "__main__":
    main()
