"""Gate I at corpus scale: the cached autoregressive baseline, measured end to end.

PROJECT_PLAN section 8 asks for a comparison at matched parameter count, data, codec and
engineering effort, measured on one named GPU - not nominal step counts. This trains the
causal model on exactly the icons Gate G's v16 trained on, at 524,674 parameters against
its 525,152, for the same number of icon presentations, and then measures what a
generation result actually needs: held-out likelihood against a floor that is not
nothing, samples that are valid and not collapsed, and wall-clock cost.

The floor matters as much as the number above it. Gate G spent months reading model
scores that looked like progress until the identity policy was put beside them and beat
every one. The analogue here is a position-marginal policy: it knows the grammar and the
empirical token distribution at each position, and it has no parameters. A model that
cannot clear it by a wide margin has learned the shape of the sequence and nothing about
icons.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from mojidiff.learning.autoregressive import (
    CausalProgramModel,
    SequenceLayout,
    flatten_program,
    generate,
    legal_mask,
    teacher_forcing_inputs,
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
from mojidiff.learning.tiny_study import _decode_checkpoint, _save_checkpoint
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.packed import serialize_packed_svg, validate_packed_tensor_program
from mojidiff.representation.renderer import RenderLimits, render_typed_svg_isolated


@dataclass(frozen=True)
class ARCorpusConfig:
    version: str
    pilot_config: Path
    report_root: Path
    checkpoint_root: Path
    d_model: int
    heads: int
    layers: int
    feedforward: int
    metric_coordinates: int
    steps: int
    batch_size: int
    learning_rate: float
    seed: int
    eval_every: int
    patience_evals: int
    marginal_alpha: float
    samples: int
    sample_seed: int
    reference_icons: int
    render_timeout_seconds: int
    max_nll_ratio_to_marginal: float
    min_distinct_sample_rate: float
    require_all_valid: bool


def load_ar_corpus_config(path: Path) -> ARCorpusConfig:
    root = yaml.safe_load(path.read_bytes())
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise OpenMojiPilotError("ar corpus schema_version must be 1")
    model = root["model"]
    training = root["training"]
    sampling = root["sampling"]
    criteria = root["criteria"]
    return ARCorpusConfig(
        version=str(root["study_version"]),
        pilot_config=Path(str(root["pilot_config"])),
        report_root=Path(str(root["report_root"])),
        checkpoint_root=Path(str(root["checkpoint_root"])),
        d_model=int(model["d_model"]),
        heads=int(model["heads"]),
        layers=int(model["layers"]),
        feedforward=int(model["feedforward"]),
        metric_coordinates=int(model.get("metric_coordinates", 0)),
        steps=int(training["steps"]),
        batch_size=int(training["batch_size"]),
        learning_rate=float(training["learning_rate"]),
        seed=int(training["seed"]),
        eval_every=int(training["eval_every"]),
        patience_evals=int(training["patience_evals"]),
        marginal_alpha=float(training["marginal_alpha"]),
        samples=int(sampling["count"]),
        sample_seed=int(sampling["seed"]),
        reference_icons=int(sampling["reference_icons"]),
        render_timeout_seconds=int(sampling["render_timeout_seconds"]),
        max_nll_ratio_to_marginal=float(criteria["max_nll_ratio_to_marginal"]),
        min_distinct_sample_rate=float(criteria["min_distinct_sample_rate"]),
        require_all_valid=bool(criteria["require_all_valid"]),
    )


class _Split:
    """Flattened tokens plus their legal masks, packed to bits to fit in memory.

    Unpacked, the masks for the 3,020 icons in train and validation would be 1.7 GiB of
    booleans. Packed they are 217 MiB, and a batch unpacks in about a millisecond, so the
    grammar can constrain every training step rather than only decoding.
    """

    def __init__(self, rows: tuple[PilotRow, ...], layout: SequenceLayout) -> None:
        self.rows = rows
        self.layout = layout
        self.tokens = torch.zeros((len(rows), layout.length), dtype=torch.long)
        self.packed = np.zeros((len(rows), layout.length * layout.vocabulary // 8), dtype=np.uint8)
        self.shape = (layout.length, layout.vocabulary)

    def masks(self, indices: np.ndarray) -> torch.Tensor:
        bits = np.unpackbits(self.packed[indices], axis=1)
        return torch.from_numpy(bits.reshape(len(indices), *self.shape)).bool()


def _build_split(
    rows: tuple[PilotRow, ...], pilot: Any, codec: Any, layout: SequenceLayout
) -> _Split:
    split = _Split(rows, layout)
    for index, row in enumerate(rows):
        tokens = flatten_program(_load_program(row, pilot, codec), layout)
        split.tokens[index] = tokens
        mask = torch.stack([legal_mask(p, tokens, layout) for p in range(layout.length)])
        split.packed[index] = np.packbits(mask.numpy().reshape(-1))
    return split


def _free_mask(masks: torch.Tensor) -> torch.Tensor:
    """Positions the grammar does not already decide.

    About a third of the 1,376 positions have exactly one legal token - an inactive
    path's padding, the kind field of a segment that can only close. Scoring them would
    report a confident number for a model that had learned nothing, and it would flatter
    the floor and the model equally, hiding the gap that is the whole question.
    """

    return masks.sum(dim=-1) > 1


def _position_marginals(split: _Split, layout: SequenceLayout, alpha: float) -> torch.Tensor:
    counts = torch.zeros((layout.length, layout.vocabulary), dtype=torch.float64)
    counts.scatter_add_(
        1, split.tokens.t(), torch.ones((layout.length, len(split.rows)), dtype=torch.float64)
    )
    return counts + alpha


def _marginal_nll(
    marginals: torch.Tensor, tokens: torch.Tensor, masks: torch.Tensor, free: torch.Tensor
) -> tuple[float, int]:
    """Negative log likelihood of a zero-parameter policy that knows the grammar.

    It is given every advantage short of learning: the empirical token frequency at each
    position, and renormalisation over exactly the legal set the model is also restricted
    to. What it does not have is any knowledge of which icon it is writing.
    """

    scores = marginals[None].expand(len(tokens), -1, -1).clone()
    scores[~masks] = 0.0
    probabilities = scores / scores.sum(dim=-1, keepdim=True)
    chosen = probabilities.gather(2, tokens[:, :, None]).squeeze(2)
    selected = chosen[free]
    return float(-torch.log(selected).sum()), int(free.sum())


def run_ar_corpus(config: ARCorpusConfig, config_path: Path) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    pilot = load_openmoji_pilot_config(config.pilot_config)
    by_split, groups, subgroups = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)

    prepared = time.perf_counter()
    train = _build_split(by_split["primary/train"], pilot, codec, layout)
    validation = _build_split(by_split["primary/validation"], pilot, codec, layout)
    prepare_seconds = time.perf_counter() - prepared

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CausalProgramModel(
        layout,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        feedforward=config.feedforward,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
        metric_coordinates=config.metric_coordinates,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    condition = {
        split_name: {
            "group": torch.tensor(
                [groups[row.group] for row in split.rows], dtype=torch.long
            ),
            "subgroup": torch.tensor(
                [subgroups[row.subgroup] for row in split.rows], dtype=torch.long
            ),
        }
        for split_name, split in (("train", train), ("validation", validation))
    }

    marginals = _position_marginals(train, layout, config.marginal_alpha)
    baseline_total = 0.0
    baseline_count = 0
    for start in range(0, len(validation.rows), 32):
        window = np.arange(start, min(start + 32, len(validation.rows)))
        masks = validation.masks(window)
        tokens = validation.tokens[window]
        total, count = _marginal_nll(marginals, tokens, masks, _free_mask(masks))
        baseline_total += total
        baseline_count += count
    baseline_nll = baseline_total / baseline_count

    rng = np.random.default_rng(config.seed)
    metrics: list[dict[str, Any]] = []
    best_step = 0
    best_nll = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    for step in range(1, config.steps + 1):
        indices = rng.choice(len(train.rows), size=config.batch_size, replace=False)
        tokens = train.tokens[indices].to(device)
        masks = train.masks(indices).to(device)
        shifted, kinds = teacher_forcing_inputs(tokens, layout)
        batch_condition = {
            key: value[indices].to(device) for key, value in condition["train"].items()
        }
        free = _free_mask(masks)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(shifted, batch_condition, kinds=kinds)
        loss = torch.nn.functional.cross_entropy(
            logits.masked_fill(~masks, float("-inf"))[free], tokens[free]
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()

        if step % config.eval_every == 0 or step == config.steps:
            held_out = _evaluate(model, validation, condition["validation"], device)
            metrics.append(
                {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "held_out_nll": held_out,
                    "marginal_nll": baseline_nll,
                }
            )
            if held_out < best_nll:
                best_step = step
                best_nll = held_out
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                stale = 0
            else:
                stale += 1
                if stale >= config.patience_evals:
                    break
    train_seconds = time.perf_counter() - started
    peak_vram = (
        float(torch.cuda.max_memory_allocated()) / 2**30 if device.type == "cuda" else None
    )

    if best_state is None:
        raise OpenMojiPilotError("no evaluation ran, so no checkpoint was selected")
    model.load_state_dict(best_state)
    model.eval()

    # The metric trace and the checkpoint are the expensive part and they are complete
    # here; sampling and latency are diagnostics over a model that is already trained.
    # Writing at the end once cost a full run to a bug in the latency path, so the
    # training result is committed before anything that can still fail runs.
    _write_bytes_artifact(config.report_root / "metrics.jsonl", _jsonl(metrics))
    _write_bytes_artifact(
        config.checkpoint_root / "checkpoint.zip", _save_checkpoint(model, optimizer, best_step)
    )

    sample_report = _sample(model, config, pilot, codec, layout, by_split, groups, subgroups)
    latency = _latency(model, by_split, pilot, codec, layout, groups, subgroups, device)

    final_nll = best_nll
    ratio = final_nll / baseline_nll
    checks = {
        "nll_ratio_to_marginal": ratio <= config.max_nll_ratio_to_marginal,
        "distinct_sample_rate": (
            sample_report["distinct_rate"] >= config.min_distinct_sample_rate
        ),
        "all_valid": sample_report["all_valid"] or not config.require_all_valid,
    }

    checkpoint = (config.checkpoint_root / "checkpoint.zip").read_bytes()
    summary = {
        "schema_version": 1,
        "study_version": config.version,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch_version": str(torch.__version__),
        "deterministic_algorithms": True,
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "sequence_length": layout.length,
        "vocabulary": layout.vocabulary,
        "train_icons": len(train.rows),
        "validation_icons": len(validation.rows),
        "icon_presentations": config.steps * config.batch_size,
        "selected_step": best_step,
        "held_out_nll_per_free_token": final_nll,
        "marginal_nll_per_free_token": baseline_nll,
        "nll_ratio_to_marginal": ratio,
        "bits_per_free_token": final_nll / float(np.log(2.0)),
        "samples": sample_report,
        "timing": {
            "prepare_seconds": prepare_seconds,
            "train_seconds": train_seconds,
            "seconds_per_step": train_seconds / max(len(metrics) * config.eval_every, 1),
            "peak_vram_gib": peak_vram,
            **latency,
        },
        "criteria": {
            "max_nll_ratio_to_marginal": config.max_nll_ratio_to_marginal,
            "min_distinct_sample_rate": config.min_distinct_sample_rate,
            "require_all_valid": config.require_all_valid,
        },
        "checks": checks,
        "predeclared_outcome": "passed" if all(checks.values()) else "falsified",
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "checkpoint_round_trip": _decode_checkpoint(checkpoint)["step"] == best_step,
        "metrics_sha256": hashlib.sha256(_jsonl(metrics)).hexdigest(),
    }
    _write_bytes_artifact(config.report_root / "summary.json", _json(summary))
    return summary


@torch.no_grad()
def _evaluate(
    model: CausalProgramModel,
    split: _Split,
    condition: dict[str, torch.Tensor],
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for start in range(0, len(split.rows), 16):
        window = np.arange(start, min(start + 16, len(split.rows)))
        tokens = split.tokens[window].to(device)
        masks = split.masks(window).to(device)
        shifted, kinds = teacher_forcing_inputs(tokens, split.layout)
        logits, _ = model(
            shifted,
            {key: value[window].to(device) for key, value in condition.items()},
            kinds=kinds,
        )
        free = _free_mask(masks)
        loss = torch.nn.functional.cross_entropy(
            logits.masked_fill(~masks, float("-inf"))[free], tokens[free], reduction="sum"
        )
        total += float(loss)
        count += int(free.sum())
    model.train()
    return total / count


def _sample(
    model: CausalProgramModel,
    config: ARCorpusConfig,
    pilot: Any,
    codec: Any,
    layout: SequenceLayout,
    by_split: dict[str, tuple[PilotRow, ...]],
    groups: dict[str, int],
    subgroups: dict[str, int],
) -> dict[str, Any]:
    """Ancestral samples, judged on the two things a first generator can fail outright.

    Validity is free here - the masks make an invalid program unreachable - so it is a
    check on the masks rather than on the model. Distinctness is not free: the standard
    failure of a conditional model trained on a corpus this small is to emit one program
    per class, which scores well on likelihood and is not a generator.
    """

    device = next(model.parameters()).device
    reference = _select_rows(by_split["primary/train"], config.reference_icons, pilot.seed)
    limits = RenderLimits(max_paths=codec.max_paths, timeout_seconds=config.render_timeout_seconds)
    coverage = []
    for row in reference:
        program = _load_program(row, pilot, codec)
        svg = serialize_packed_svg(program, codec, pilot.total_segment_slots)
        _, raster = render_typed_svg_isolated(svg, 72, limits)
        coverage.append(float((raster[:, :, 3] > 0).mean()))
    low, high = float(np.quantile(coverage, 0.25)), float(np.quantile(coverage, 0.75))

    targets = _select_rows(by_split["primary/validation"], config.samples, config.sample_seed)
    digests: list[str] = []
    sample_coverage = []
    valid = 0
    for index, row in enumerate(targets):
        template = _load_program(row, pilot, codec)
        condition = {
            "group": torch.tensor([groups[row.group]], dtype=torch.long, device=device),
            "subgroup": torch.tensor([subgroups[row.subgroup]], dtype=torch.long, device=device),
        }
        program, _ = generate(
            model,
            template,
            condition,
            greedy=False,
            rng=np.random.default_rng(config.sample_seed + index),
        )
        validate_packed_tensor_program(program, codec, pilot.total_segment_slots)
        svg = serialize_packed_svg(program, codec, pilot.total_segment_slots)
        _, raster = render_typed_svg_isolated(svg, 72, limits)
        valid += 1
        tokens = flatten_program(program, layout).numpy().tobytes()
        digests.append(hashlib.sha256(tokens).hexdigest())
        sample_coverage.append(float((raster[:, :, 3] > 0).mean()))

    median = float(np.median(sample_coverage))
    return {
        "count": len(targets),
        "all_valid": valid == len(targets),
        "distinct": len(set(digests)),
        "distinct_rate": len(set(digests)) / len(targets),
        "median_ink_coverage": median,
        "corpus_ink_coverage_iqr": [low, high],
        "median_inside_corpus_iqr": low <= median <= high,
        "reference_icons": len(reference),
        "digests": digests,
    }


def _latency(
    model: CausalProgramModel,
    by_split: dict[str, tuple[PilotRow, ...]],
    pilot: Any,
    codec: Any,
    layout: SequenceLayout,
    groups: dict[str, int],
    subgroups: dict[str, int],
    device: torch.device,
) -> dict[str, Any]:
    """What the cache actually buys, in seconds rather than in asymptotics.

    The uncached comparison re-runs the whole prefix at every position, which is the
    honest alternative implementation and not a strawman: it is what the same model does
    without the per-layer key/value state.
    """

    row = by_split["primary/validation"][0]
    template = _load_program(row, pilot, codec)
    condition = {
        "group": torch.tensor([groups[row.group]], dtype=torch.long, device=device),
        "subgroup": torch.tensor([subgroups[row.subgroup]], dtype=torch.long, device=device),
    }
    started = time.perf_counter()
    cached_program, calls = generate(model, template, condition, greedy=True)
    cached = time.perf_counter() - started

    # The uncached path follows the cached path's tokens rather than its own argmax.
    # Left to diverge, a single near-tie early on sends the two down different programs
    # and the comparison stops being about the cache at all - which is what the first
    # version of this measured.
    reference = flatten_program(cached_program, layout)
    _, kinds_full = teacher_forcing_inputs(reference[None], layout)
    kinds_full = kinds_full.to(device)
    divergences = 0
    smallest_margin = float("inf")
    started = time.perf_counter()
    with torch.no_grad():
        for position in range(layout.length):
            prefix = torch.cat((torch.zeros(1, dtype=torch.long), reference[:position]))
            logits, _ = model(
                prefix[None].to(device), condition, kinds=kinds_full[:, : position + 1]
            )
            mask = legal_mask(position, reference, layout).to(device)
            scores = logits[0, -1].masked_fill(~mask, float("-inf"))
            token = int(scores.argmax())
            if token != int(reference[position]):
                divergences += 1
                ordered = scores.sort(descending=True).values
                smallest_margin = min(smallest_margin, float(ordered[0] - ordered[1]))
    uncached = time.perf_counter() - started

    return {
        "cached_decode_seconds": cached,
        "cached_forward_calls": calls,
        "uncached_decode_seconds": uncached,
        "cache_speedup": uncached / cached,
        # A disagreement is only alarming if the runner-up was not a near-tie. Both
        # paths are the same arithmetic in a different order, and the test suite already
        # pins cached logits to a full forward within 1e-5, so drift at a position where
        # two tokens score that close flips the argmax without anything being wrong. A
        # wide margin at a disagreement is a different matter and means a real bug.
        "argmax_divergences": divergences,
        "smallest_divergence_margin": None if divergences == 0 else smallest_margin,
    }


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_json(row) for row in rows)


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_ar_corpus_config(args.config)
    print(json.dumps(run_ar_corpus(config, args.config), sort_keys=True))


if __name__ == "__main__":
    main()
