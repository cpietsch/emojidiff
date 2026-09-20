"""Gate I's first evidence: can the causal model learn these programs at all?

Gate E did this for the denoiser before anything ran at corpus scale, and the same
discipline applies here. This overfits a handful of hash-pinned icons and asks three
things: that teacher-forced next-token accuracy over legal positions reaches
memorisation, that the loss actually falls rather than plateauing on a broken signal -
which Gate G spent a long time discovering - and that every greedy sample is still a
program the packed validator and the isolated renderer accept.

It is a learnability diagnostic, not a generation result. A model that memorises four
icons has shown its plumbing works and nothing else.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from mojidiff.learning.autoregressive import (
    CausalProgramModel,
    SequenceLayout,
    flatten_program,
    generate,
    legal_mask,
)
from mojidiff.learning.openmoji_pilot import (
    OpenMojiPilotError,
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.tiny_study import _decode_checkpoint, _model_hash, _save_checkpoint
from mojidiff.representation.codec_study import _write_bytes_artifact
from mojidiff.representation.packed import serialize_packed_svg, validate_packed_tensor_program
from mojidiff.representation.renderer import RenderLimits, render_typed_svg_isolated


@dataclass(frozen=True)
class AROverfitConfig:
    version: str
    pilot_config: Path
    report_root: Path
    checkpoint_root: Path
    icons: int
    d_model: int
    heads: int
    layers: int
    feedforward: int
    steps: int
    learning_rate: float
    seed: int
    device: str
    min_free_token_accuracy: float
    min_loss_reduction_factor: float
    min_exact_match_rate: float
    require_all_valid: bool


def load_ar_overfit_config(path: Path) -> AROverfitConfig:
    root = yaml.safe_load(path.read_bytes())
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise OpenMojiPilotError("ar overfit schema_version must be 1")
    model = root.get("model", {})
    training = root.get("training", {})
    criteria = root["criteria"]
    return AROverfitConfig(
        version=str(root["study_version"]),
        pilot_config=Path(str(root["pilot_config"])),
        report_root=Path(str(root["report_root"])),
        checkpoint_root=Path(str(root["checkpoint_root"])),
        icons=int(root["icons"]),
        d_model=int(model["d_model"]),
        heads=int(model["heads"]),
        layers=int(model["layers"]),
        feedforward=int(model["feedforward"]),
        steps=int(training["steps"]),
        learning_rate=float(training["learning_rate"]),
        seed=int(training["seed"]),
        device=str(training.get("device", "auto")),
        min_free_token_accuracy=float(criteria["min_free_token_accuracy"]),
        min_loss_reduction_factor=float(criteria["min_loss_reduction_factor"]),
        min_exact_match_rate=float(criteria["min_exact_match_rate"]),
        require_all_valid=bool(criteria["require_all_valid"]),
    )


def run_ar_overfit(config: AROverfitConfig, config_path: Path) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    pilot = load_openmoji_pilot_config(config.pilot_config)
    by_split, groups, subgroups = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)
    rows = _distinct_subgroup_rows(by_split["primary/train"], config.icons, pilot.seed)
    programs = [_load_program(row, pilot, codec) for row in rows]
    device = torch.device(
        "cuda" if config.device in {"auto", "cuda"} and torch.cuda.is_available() else "cpu"
    )
    tokens = torch.stack([flatten_program(p, layout) for p in programs]).to(device)
    condition = {
        "group": torch.tensor([groups[r.group] for r in rows], dtype=torch.long, device=device),
        "subgroup": torch.tensor(
            [subgroups[r.subgroup] for r in rows], dtype=torch.long, device=device
        ),
    }

    # Only positions with more than one legal token are learnable; the rest are forced
    # by the grammar and counting them would flatter the accuracy.
    masks = torch.stack(
        [
            torch.stack([legal_mask(p, tokens[i].cpu(), layout) for p in range(layout.length)])
            for i in range(len(programs))
        ]
    ).to(device)
    free = masks.sum(dim=-1) > 1

    model = CausalProgramModel(
        layout,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        feedforward=config.feedforward,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    shifted = torch.cat((torch.zeros_like(tokens[:, :1]), tokens[:, :-1]), dim=1)

    metrics: list[dict[str, Any]] = []
    initial_loss = None
    for step in range(1, config.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(shifted, condition)
        logits = logits.masked_fill(~masks, float("-inf"))
        loss = torch.nn.functional.cross_entropy(
            logits[free], tokens[free], reduction="mean"
        )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        if initial_loss is None:
            initial_loss = float(loss.detach())
        if step % max(config.steps // 20, 1) == 0 or step == config.steps:
            with torch.no_grad():
                correct = int((logits[free].argmax(dim=-1) == tokens[free]).sum())
            metrics.append(
                {
                    "step": step,
                    "loss": float(loss.detach()),
                    "free_token_accuracy": correct / int(free.sum()),
                }
            )

    samples = []
    model.eval()
    for index, program in enumerate(programs):
        single = {key: value[index : index + 1] for key, value in condition.items()}
        sampled, calls = generate(model, program, single, greedy=True)
        validate_packed_tensor_program(sampled, codec, pilot.total_segment_slots)
        svg = serialize_packed_svg(sampled, codec, pilot.total_segment_slots)
        _, raster = render_typed_svg_isolated(
            svg, 72, RenderLimits(max_paths=codec.max_paths, timeout_seconds=20)
        )
        samples.append(
            {
                "hexcode": rows[index].hexcode,
                "subgroup": rows[index].subgroup,
                "decode_calls": calls,
                "valid": True,
                "rendered": bool(raster.shape == (72, 72, 4)),
                # The conditioning identifies the icon inside this set because every
                # icon comes from a different subgroup, so an exact match is a real
                # memorisation result rather than the model emitting one average program.
                "exact_match": bool(
                    torch.equal(flatten_program(sampled, layout), tokens[index].cpu())
                ),
                "svg_sha256": hashlib.sha256(svg).hexdigest(),
            }
        )

    checkpoint = _save_checkpoint(model, optimizer, config.steps)
    restored = CausalProgramModel(
        layout,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        feedforward=config.feedforward,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
    ).to(device)
    restored.load_state_dict(_decode_checkpoint(checkpoint)["model"])
    round_trip = _model_hash(restored) == _model_hash(model)

    accuracy = float(metrics[-1]["free_token_accuracy"])
    reduction = (initial_loss or 0.0) / max(float(metrics[-1]["loss"]), 1e-12)
    exact_rate = sum(bool(sample["exact_match"]) for sample in samples) / len(samples)
    all_valid = all(sample["valid"] and sample["rendered"] for sample in samples)
    checks = {
        "free_token_accuracy": accuracy >= config.min_free_token_accuracy,
        "loss_reduction_factor": reduction >= config.min_loss_reduction_factor,
        "exact_match_rate": exact_rate >= config.min_exact_match_rate,
        "all_valid": all_valid or not config.require_all_valid,
    }

    summary = {
        "schema_version": 1,
        "study_version": config.version,
        "scope": "autoregressive learnability diagnostic; not a generation result",
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "deterministic_algorithms": True,
        "sequence_length": layout.length,
        "vocabulary": layout.vocabulary,
        "free_positions_per_icon": int(free.sum()) // len(programs),
        "icons": [row.source_path for row in rows],
        "model_parameters": sum(p.numel() for p in model.parameters()),
        "initial_loss": initial_loss,
        "final": metrics[-1],
        "loss_reduction_factor": reduction,
        "samples": samples,
        "exact_match_rate": exact_rate,
        "criteria": {
            "min_free_token_accuracy": config.min_free_token_accuracy,
            "min_loss_reduction_factor": config.min_loss_reduction_factor,
            "min_exact_match_rate": config.min_exact_match_rate,
            "require_all_valid": config.require_all_valid,
        },
        "checks": checks,
        "predeclared_outcome": "passed" if all(checks.values()) else "falsified",
        "checkpoint_sha256": hashlib.sha256(checkpoint).hexdigest(),
        "checkpoint_round_trip": round_trip,
        "metrics_sha256": hashlib.sha256(_jsonl(metrics)).hexdigest(),
    }
    _write_bytes_artifact(config.report_root / "metrics.jsonl", _jsonl(metrics))
    _write_bytes_artifact(config.report_root / "summary.json", _json(summary))
    _write_bytes_artifact(config.checkpoint_root / "checkpoint.zip", checkpoint)
    return summary


def _distinct_subgroup_rows(
    rows: tuple[Any, ...], count: int, seed: int
) -> tuple[Any, ...]:
    """One icon per subgroup, so the conditioning can tell the training icons apart.

    The model sees only group and subgroup, so two icons sharing a subgroup are the
    same prompt and greedy decoding cannot reproduce both. Drawing from distinct
    subgroups makes exact reproduction a meaningful criterion instead of an
    impossible one.
    """

    ordered = _select_rows(rows, len(rows), seed)
    seen: set[str] = set()
    chosen = []
    for row in ordered:
        if row.subgroup in seen:
            continue
        seen.add(row.subgroup)
        chosen.append(row)
        if len(chosen) == count:
            return tuple(chosen)
    raise OpenMojiPilotError(f"only {len(chosen)} distinct subgroups available")


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_json(row) for row in rows)


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_ar_overfit_config(args.config)
    print(json.dumps(run_ar_overfit(config, args.config), sort_keys=True))


if __name__ == "__main__":
    main()
