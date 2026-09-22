"""Gate M, experiment one, step two: OmniSVG 1.1 4B fine-tuned with LoRA on OpenMoji.

The model is trained in its own token language - OpenMoji icons outlined and encoded
by `omnisvg_encode`, which round-trips through OmniSVG's released decoder exactly -
under the trainer's own prompt, with the loss on the drawing tokens only. Adapters go
on the language model's linear projections; the vocabulary, embeddings and vision
tower stay as released. Selection is by held-out likelihood on validation icons that
are not the evaluation icons. The evaluation is the zero-shot study's, with the
training-style prompt, so every number reads directly against the control: the same
held-out icons, decoder, codec, renderer and CLIP.

Three predeclared readings decide the run: the codec takes a stated share of the
drawings; the per-icon gain in similarity to the held-out render over the control
has a paired bootstrap interval that excludes zero; and the right icon is retrieved
first among the held-out renders at a stated rate, far above chance. A memorisation
bound - drawings identical to a training icon, token for token - is read beside them.
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

from mojidiff.learning.omnisvg import IMAGE_SIZE, OMNISVG_REPO, OMNISVG_REVISION, OmniSVG
from mojidiff.learning.omnisvg_encode import encode_rows
from mojidiff.learning.omnisvg_study import (
    CLIP_REPO,
    CLIP_REVISION,
    Clip,
    captions,
    evaluate,
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
from mojidiff.learning.prior import adapter_bytes, dumps, sha256
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.renderer import RenderLimits

DEFAULT_CACHE = Path("data/processed/prior/omnisvg-tokens")
DEFAULT_RENDER_CACHE = Path("data/processed/prior/omnisvg-renders-448")


def render_rows(
    rows: tuple[PilotRow, ...], pilot: Any, codec: Any, limits: RenderLimits, cache_root: Path
) -> dict[str, Path]:
    """Each row's program rendered at the model's image size, cached as PNG by hexcode."""

    cache_root.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for row in rows:
        path = cache_root / f"{row.hexcode}.png"
        if not path.is_file():
            image = render_program(
                _load_program(row, pilot, codec),
                codec,
                pilot.total_segment_slots,
                limits,
                IMAGE_SIZE,
            )
            image.save(path, format="PNG", optimize=True)
        paths[row.hexcode] = path
    return paths


def run_omnisvg_finetune(config_path: Path) -> dict[str, Any]:
    root = yaml.safe_load(config_path.read_text())
    if root.get("schema_version") != 1:
        raise OpenMojiPilotError("omnisvg finetune schema_version must be 1")
    version = str(root["study_version"])
    report_root = Path(str(root["report_root"]))
    checkpoint_root = Path(str(root.get("checkpoint_root", report_root)))
    pilot = load_openmoji_pilot_config(Path(str(root["pilot_config"])))
    by_split, _, _ = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    data = root["data"]
    generation = root["generation"]
    training = root["training"]
    size = int(root.get("render", {}).get("size", 72))
    limits = RenderLimits(max_paths=codec.max_paths, timeout_seconds=20)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cache_root = Path(str(training.get("token_cache_root", DEFAULT_CACHE)))
    annotations = captions(pilot.raw_root)

    icons = int(data["icons"])
    eval_rows = _select_rows(by_split["primary/validation"], icons, pilot.seed + 1)
    eval_hexcodes = {row.hexcode for row in eval_rows}
    held = int(training["held_out_icons"])
    select_rows = tuple(
        row
        for row in _select_rows(by_split["primary/validation"], held + icons, pilot.seed + 2)
        if row.hexcode not in eval_hexcodes
    )[:held]
    train_rows = by_split["primary/train"]
    if data.get("train_icons") is not None:
        train_rows = _select_rows(train_rows, int(data["train_icons"]), pilot.seed)

    started = time.perf_counter()
    model = OmniSVG.load(device, lora=training["lora"])
    load_seconds = time.perf_counter() - started

    split = training.get("split_max_dist")
    split_at = float(split) if split is not None else None
    train_tokens, train_failures = encode_rows(train_rows, pilot.raw_root, cache_root, split_at)
    select_tokens, select_failures = encode_rows(select_rows, pilot.raw_root, cache_root, split_at)
    max_tokens = int(training["max_tokens"])
    condition = str(training.get("condition", "caption"))
    renders: dict[str, Path] = {}
    if condition == "image":
        render_cache = Path(str(training.get("render_cache_root", DEFAULT_RENDER_CACHE)))
        renders = render_rows(train_rows + select_rows, pilot, codec, limits, render_cache)
    train, excluded_train = _examples(
        model, train_rows, train_tokens, annotations, max_tokens, condition, renders
    )
    select, excluded_select = _examples(
        model, select_rows, select_tokens, annotations, max_tokens, condition, renders
    )
    if not train or not select:
        raise OpenMojiPilotError("no examples fit under max_tokens")

    train_report, metrics = _train(model, train, select, training, device)
    train_report["training"].update(
        {
            "train_icons": len(train),
            "train_encode_failures": len(train_failures),
            "excluded_train_over_max_tokens": excluded_train,
            "selection_icons": len(select),
            "selection_encode_failures": len(select_failures),
            "excluded_selection_over_max_tokens": excluded_select,
            "max_tokens": max_tokens,
            "condition": condition,
            "split_max_dist": split_at,
            "load_seconds": load_seconds,
        }
    )
    _write_bytes_artifact(report_root / "metrics.jsonl", b"".join(dumps(row) for row in metrics))
    payload = adapter_bytes(model.model)
    _write_bytes_artifact(checkpoint_root / "adapter.zip", payload)
    train_report["training"]["adapter_sha256"] = sha256(payload)
    train_report["training"]["adapter_bytes"] = len(payload)

    inner = model._inner()
    inner.gradient_checkpointing_disable()
    model.model.eval()
    clip = Clip(device)
    train_sequences = {tuple(tokens[1:-1].tolist()) for tokens in train_tokens.values()}
    evaluation, rows, sheet = evaluate(
        model,
        eval_rows,
        pilot,
        codec,
        clip,
        samples=int(generation["samples"]),
        max_new_tokens=int(generation["max_new_tokens"]),
        seed=int(generation["seed"]),
        style=str(generation.get("prompt_style", "image" if condition == "image" else "training")),
        limits=limits,
        size=size,
        train_sequences=train_sequences,
        sampling={
            key: generation[key]
            for key in ("temperature", "top_p", "top_k", "repetition_penalty", "greedy")
            if key in generation
        },
    )
    rows_payload = b"".join(dumps(record) for record in rows)
    _write_bytes_artifact(report_root / "drawings.jsonl", rows_payload)
    _write_bytes_artifact(report_root / "samples.png", sheet)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "study_version": version,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "model": {
            "repo": OMNISVG_REPO,
            "revision": OMNISVG_REVISION,
            "checkpoint_sha256": model.checkpoint_sha256,
            "parameters": model.parameters,
            "trainable_parameters": model.trainable_parameters,
            "fine_tuned": True,
            "lora": training["lora"],
        },
        "clip": {"repo": CLIP_REPO, "revision": CLIP_REVISION},
        **train_report,
        **evaluation,
        "sheet_sha256": hashlib.sha256(sheet).hexdigest(),
        "rows_sha256": hashlib.sha256(rows_payload).hexdigest(),
    }
    if "control_report_root" in root:
        summary["control"] = compare_to_control(rows, Path(str(root["control_report_root"])))
    if "criteria" in root:
        summary["criteria"] = root["criteria"]
        summary["checks"] = _checks(summary, root["criteria"])
        summary["predeclared_outcome"] = (
            "passed" if all(summary["checks"].values()) else "falsified"
        )
    _write_bytes_artifact(report_root / "summary.json", dumps(summary))
    return summary


# ------------------------------------------------------------------------ examples


class Example:
    """One training sequence: the prompt, then the drawing; loss on the drawing.

    An image-conditioned example keeps its render's path and rebuilds the vision
    inputs when used, so a corpus of thousands does not hold its pixels in memory.
    """

    __slots__ = ("hexcode", "input_ids", "prompt_tokens", "render", "model")

    def __init__(
        self,
        hexcode: str,
        prompt_ids: torch.Tensor,
        drawing: torch.Tensor,
        render: Path | None = None,
        model: OmniSVG | None = None,
    ) -> None:
        self.hexcode = hexcode
        self.input_ids = torch.cat((prompt_ids.cpu(), drawing.cpu())).long()
        self.prompt_tokens = int(prompt_ids.numel())
        self.render = render
        self.model = model

    @property
    def length(self) -> int:
        return int(self.input_ids.numel())

    def tensors(self, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        ids = self.input_ids[None].to(device)
        labels = ids.clone()
        labels[:, : self.prompt_tokens] = -100
        return ids, labels

    def vision(self) -> dict[str, torch.Tensor] | None:
        if self.render is None or self.model is None:
            return None
        with Image.open(self.render) as image:
            inputs = self.model.prompt_ids("x", style="image", image=image.convert("RGB"))
        # Everything the processor made besides the token ids: the pixels, the grid and
        # the token-type ids. Without the token-type ids the model cannot build its 3-D
        # rotary positions and silently falls back to flat ones, which is not what
        # generation uses - the first image fine-tune trained under that mismatch.
        vision = {
            key: value
            for key, value in inputs.items()
            if key not in ("input_ids", "attention_mask")
        }
        if "mm_token_type_ids" in vision:
            # The processor typed the prompt's tokens; the drawing's are all text (0).
            types = vision["mm_token_type_ids"]
            tail = types.new_zeros((types.shape[0], self.length - types.shape[1]))
            vision["mm_token_type_ids"] = torch.cat((types, tail), dim=1)
        return vision


def _examples(
    model: OmniSVG,
    rows: tuple[PilotRow, ...],
    tokens: dict[str, torch.Tensor],
    annotations: dict[str, str],
    max_tokens: int,
    condition: str = "caption",
    renders: dict[str, Path] | None = None,
) -> tuple[list[Example], int]:
    examples, excluded = [], 0
    for row in rows:
        if row.hexcode not in tokens:
            continue
        if condition == "image":
            render = (renders or {})[row.hexcode]
            with Image.open(render) as image:
                prompt = model.prompt_ids("x", style="image", image=image.convert("RGB"))
            example = Example(
                row.hexcode, prompt["input_ids"][0], tokens[row.hexcode], render, model
            )
        else:
            prompt = model.prompt_ids(annotations.get(row.hexcode, row.hexcode), style="training")
            example = Example(row.hexcode, prompt["input_ids"][0], tokens[row.hexcode])
        if example.length > max_tokens:
            excluded += 1  # never train on a silently truncated icon
            continue
        examples.append(example)
    return examples, excluded


# ------------------------------------------------------------------------ training


def _train(
    model: OmniSVG,
    train: list[Example],
    select: list[Example],
    training: dict[str, Any],
    device: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    steps = int(training["steps"])
    accumulate = int(training.get("accumulate", 8))
    chunk = int(training.get("loss_chunk", 512))
    eval_every = int(training["eval_every"])
    patience = int(training.get("patience_evals", 8))
    trainable = [p for p in model.model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
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
        model.model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for example in select:
                ids, labels = example.tensors(device)
                loss, n = model.suffix_loss(ids, labels, chunk, vision=example.vision())
                total, count = total + float(loss), count + n
        model.model.train()
        return total / max(count, 1)

    probe_cfg = training.get("probe")
    probe_examples = select[: int(probe_cfg["icons"])] if probe_cfg else []
    probe_clip = Clip(device) if probe_cfg else None

    def probe() -> dict[str, Any]:
        """Free-running drawings for a few selection icons: what likelihood cannot see.

        Greedy, no repetition penalty, the drawing scored against the icon's own render;
        selection is on this when configured, because a model can fit the corpus under
        teacher forcing and still not draw.
        """

        assert probe_cfg is not None and probe_clip is not None
        inner = model._inner()
        inner.gradient_checkpointing_disable()
        model.model.eval()
        ended, lengths, similarities = 0, [], []
        with torch.inference_mode():
            for example in probe_examples:
                with Image.open(example.render) as image:  # type: ignore[arg-type]
                    condition = image.convert("RGB")
                drawing = model.generate(
                    "probe",
                    samples=1,
                    max_new_tokens=int(probe_cfg.get("max_new_tokens", 1024)),
                    seed=0,
                    style="image",
                    image=condition,
                    greedy=True,
                    repetition_penalty=1.0,
                )[0]
                svg, info = model.tokens_to_svg(drawing)
                ended += int(bool(info.get("ended")))
                lengths.append(int(info["tokens"]))
                if svg is not None:
                    try:
                        picture = render_raw(svg, 72)
                        reference = condition.resize((72, 72))
                        features = probe_clip.image([reference, picture])
                        similarities.append(float(features[0] @ features[1]))
                    except Exception:  # noqa: BLE001 - a drawing that does not render scores nothing
                        similarities.append(0.0)
                else:
                    similarities.append(0.0)
        model.model.train()
        inner.gradient_checkpointing_enable()
        return {
            "probe_ended_rate": ended / max(len(probe_examples), 1),
            "probe_median_tokens": float(np.median(lengths)) if lengths else None,
            "probe_clip_mean": float(np.mean(similarities)) if similarities else 0.0,
        }

    metrics: list[dict[str, Any]] = [{"step": 0, "held_out_nll": held_out_nll()}]
    if probe_cfg:
        metrics[0].update(probe())
    select_on = str(training.get("select_on", "held_out_nll"))

    def score(row: dict[str, Any]) -> float:
        # Lower is better: negative likelihood, or negative probe similarity.
        return (
            float(row["held_out_nll"])
            if select_on == "held_out_nll"
            else -float(row["probe_clip_mean"])
        )

    def snapshot() -> dict[str, torch.Tensor]:
        return {
            k: v.detach().cpu().clone()
            for k, v in model.model.state_dict().items()
            if "lora" in k.lower()
        }

    # The initial adapters are a candidate too: if no step beats them, they are restored.
    best_nll, best_step, best_state, stale = score(metrics[0]), 0, snapshot(), 0
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    model.model.train()
    order = rng.permutation(len(train))
    cursor = 0
    running, running_tokens = 0.0, 0
    for step in range(1, steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulate):
            if cursor >= len(order):
                order, cursor = rng.permutation(len(train)), 0
            example = train[order[cursor]]
            cursor += 1
            ids, labels = example.tensors(device)
            loss, n = model.suffix_loss(ids, labels, chunk, vision=example.vision())
            (loss / (accumulate * max(n, 1))).backward()  # type: ignore[no-untyped-call]
            running += float(loss.detach())
            running_tokens += n
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        schedule.step()
        if step % eval_every == 0 or step == steps:
            nll = held_out_nll()
            metrics.append(
                {
                    "step": step,
                    "train_loss": running / max(running_tokens, 1),
                    "held_out_nll": nll,
                    "seconds": time.perf_counter() - started,
                }
            )
            if probe_cfg and step % int(probe_cfg.get("every", eval_every)) == 0:
                metrics[-1].update(probe())
            running, running_tokens = 0.0, 0
            print(json.dumps(metrics[-1]), flush=True)
            if "probe_clip_mean" not in metrics[-1] and select_on != "held_out_nll":
                continue  # no probe at this step: nothing to select on
            if score(metrics[-1]) < best_nll:
                best_nll, best_step, stale = score(metrics[-1]), step, 0
                best_state = snapshot()
            else:
                stale += 1
                if stale >= patience:
                    break
    train_seconds = time.perf_counter() - started
    if best_state is not None:
        model.model.load_state_dict(best_state, strict=False)
    return {
        "training": {
            "steps_run": metrics[-1]["step"],
            "sequences_per_step": accumulate,
            "selected_step": best_step,
            "selected_on": select_on,
            "initial_held_out_nll": metrics[0]["held_out_nll"],
            "held_out_nll": min(float(row["held_out_nll"]) for row in metrics),
            "selected_probe": {k: v for k, v in metrics[-1].items() if k.startswith("probe_")}
            if probe_cfg
            else None,
            "train_seconds": train_seconds,
            "peak_vram_gib": float(torch.cuda.max_memory_allocated()) / 2**30
            if device == "cuda"
            else None,
        }
    }, metrics


# ------------------------------------------------------------- against the control


def compare_to_control(
    rows: list[dict[str, Any]], control_root: Path, *, resamples: int = 10_000, seed: int = 0
) -> dict[str, Any]:
    """Per-icon paired gain over the control's drawings, with a bootstrap interval.

    Both runs draw the same held-out icons, so each icon's mean similarity to its own
    render is paired with the control's; the interval is over icons.
    """

    control_rows = [
        json.loads(line)
        for line in (control_root / "drawings.jsonl").read_text().splitlines()
        if line.strip()
    ]
    control_summary = json.loads((control_root / "summary.json").read_text())
    ours, theirs = _per_icon(rows), _per_icon(control_rows)
    common = sorted(set(ours) & set(theirs))
    diffs = np.array([ours[hexcode] - theirs[hexcode] for hexcode in common])
    rng = np.random.default_rng(seed)
    means = np.array(
        [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(resamples)]
    )
    return {
        "report_root": str(control_root),
        "control_rows_sha256": control_summary.get("rows_sha256"),
        "paired_icons": len(common),
        "clip_to_reference_mean": float(np.mean([ours[h] for h in common])),
        "control_clip_to_reference_mean": float(np.mean([theirs[h] for h in common])),
        "mean_gain": float(diffs.mean()),
        "gain_ci95": [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))],
        "icons_improved": int((diffs > 0).sum()),
        "control_reference_top1_rate": (control_summary.get("reference_rank") or {}).get(
            "top1_rate"
        ),
        "control_codec_valid_rate": control_summary.get("codec_valid_rate"),
    }


def _per_icon(rows: list[dict[str, Any]]) -> dict[str, float]:
    scores: dict[str, list[float]] = {}
    for record in rows:
        if "clip_to_reference" in record:
            scores.setdefault(record["hexcode"], []).append(float(record["clip_to_reference"]))
    return {hexcode: float(np.mean(values)) for hexcode, values in scores.items()}


def _checks(summary: dict[str, Any], criteria: dict[str, Any]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    if "min_codec_valid_rate" in criteria:
        checks["codec_valid_rate"] = summary["codec_valid_rate"] >= float(
            criteria["min_codec_valid_rate"]
        )
    if criteria.get("clip_gain_over_control_ci_excludes_zero"):
        control: dict[str, Any] = summary.get("control") or {}
        checks["clip_gain_over_control"] = bool(control) and float(control["gain_ci95"][0]) > 0
    if "min_ended_rate" in criteria:
        checks["ended_rate"] = summary["ended_rate"] >= float(criteria["min_ended_rate"])
    if "min_clip_to_reference_mean" in criteria:
        value = summary["clip_to_reference"]["mean"]
        checks["clip_to_reference"] = value is not None and value >= float(
            criteria["min_clip_to_reference_mean"]
        )
    if "min_reference_top1_rate" in criteria:
        checks["reference_top1_rate"] = summary["reference_rank"]["top1_rate"] >= float(
            criteria["min_reference_top1_rate"]
        )
    if "max_memorised_exactly" in criteria:
        checks["memorisation"] = summary["memorised_exactly"] <= int(
            criteria["max_memorised_exactly"]
        )
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    summary = run_omnisvg_finetune(args.config)
    keys = (
        "drawings",
        "decoded_rate",
        "ended_rate",
        "codec_valid_rate",
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
        "timing",
    )
    print(json.dumps({key: summary[key] for key in keys if key in summary}, indent=1))


if __name__ == "__main__":
    main()
