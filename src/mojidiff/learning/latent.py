"""A continuous latent space over emoji programs: interpolate and sample, as vectors.

The render-to-SVG decoder reads a memory through cross-attention; here the memory comes
from a 64-dimensional Gaussian latent instead of an image. A program encoder maps a
program to the latent's mean and variance (a variational autoencoder), so

* any point in the latent decodes, under the grammar, to a valid program;
* a straight line between two icons' latents is a vector interpolation between them;
* a draw from the prior is an unconditional sample.

Everything else - the decoder, metric coordinates, path-major order, the grammar masks,
the exact augmentation and composition stream, the graph decoder - is reused. Two
standard guards keep an autoregressive decoder from ignoring its latent: the KL term is
annealed in with a per-dimension floor ("free bits"), and decoder inputs are randomly
blanked so the next token cannot always be read off the prefix.

    python -m mojidiff.learning.latent --config configs/latent/<name>.yaml
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from mojidiff.learning.autoregressive import SequenceLayout
from mojidiff.learning.render2svg import (
    CACHE_ROOT,
    REPO_ROOT,
    DecodeStats,
    ModelConfig,
    RenderToProgram,
    TrainConfig,
    _append_registry,
    _DecoderBlock,
    _EncoderBlock,
    _now,
    _online_batches,
    _pilot_rows,
    _programs_to_renders,
    _schedule,
    _train_config,
    _write_yaml,
    bootstrap_mean_interval,
    contact_sheet,
    greedy_decode,
    load_corpus,
    nearest_training_icon,
    parameter_count,
    pixel_error,
    run_identity,
    split_orders,
)
from mojidiff.learning.telemetry import measure_latency, reset_peak_memory, resource_summary


@dataclass(frozen=True)
class LatentSettings:
    latent_dim: int = 64
    memory_tokens: int = 16
    encoder_layers: int = 3
    beta: float = 1.0
    beta_warmup_fraction: float = 0.3
    free_bits: float = 0.1
    """Per-dimension KL floor in nats; below it the KL term exerts no pull."""
    input_dropout: float = 0.25
    """Probability of blanking each decoder input token during training."""


@dataclass(frozen=True)
class LatentConfig:
    slug: str
    hypothesis: str
    pilot_config: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    latent: LatentSettings = field(default_factory=LatentSettings)
    training: TrainConfig = field(default_factory=TrainConfig)
    eval_icons: int = 339
    parent_run: str | None = None


def load_config(path: Path) -> LatentConfig:
    root = yaml.safe_load(path.read_text())
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise ValueError("latent config needs schema_version: 1")
    return LatentConfig(
        slug=str(root["slug"]),
        hypothesis=str(root["hypothesis"]).strip(),
        pilot_config=Path(str(root["pilot_config"])),
        model=ModelConfig(**root.get("model", {})),
        latent=LatentSettings(**root.get("latent", {})),
        training=_train_config(root.get("training", {})),
        eval_icons=int(root.get("eval_icons", 339)),
        parent_run=root.get("parent_run"),
    )


class LatentToProgram(RenderToProgram):
    """The render-to-SVG decoder, reading a latent instead of an image."""

    def __init__(
        self, layout: SequenceLayout, config: ModelConfig, settings: LatentSettings
    ) -> None:
        super().__init__(layout, config)
        # The image path is not used; drop it so the parameter count is honest.
        del self.stem
        del self.grid_embedding
        del self.encoder
        del self.encoder_norm
        if hasattr(self, "grid_projection"):
            del self.grid_projection
        width = config.d_model
        self.settings = settings
        self.program_embedding = nn.Embedding(layout.vocabulary, width)
        self.program_position = nn.Embedding(layout.length + 1, width)
        self.program_encoder = nn.ModuleList(
            _EncoderBlock(width, config.heads, config.feedforward, config.dropout)
            for _ in range(settings.encoder_layers)
        )
        self.program_norm = nn.LayerNorm(width)
        self.to_posterior = nn.Linear(width, 2 * settings.latent_dim)
        self.to_memory = nn.Linear(settings.latent_dim, settings.memory_tokens * width)
        self.memory_position = nn.Parameter(torch.zeros(1, settings.memory_tokens, width))
        nn.init.normal_(self.memory_position, std=0.02)
        self.memory_norm = nn.LayerNorm(width)

    @property
    def memory_length(self) -> int:
        return self.settings.memory_tokens

    def posterior(self, tokens: Tensor) -> tuple[Tensor, Tensor]:
        """Mean and log-variance of q(z | program), read through a summary token."""

        batch = tokens.shape[0]
        summary = torch.full((batch, 1), 0, dtype=torch.long, device=tokens.device)
        positions = torch.arange(tokens.shape[1] + 1, device=tokens.device)
        hidden = self.program_embedding(torch.cat((summary, tokens), dim=1))
        hidden = hidden + self.program_position(positions)[None]
        for block in self.program_encoder:
            hidden = block(hidden)
        mean, log_variance = self.to_posterior(self.program_norm(hidden[:, 0])).chunk(2, dim=-1)
        return mean, log_variance.clamp(-10.0, 10.0)

    def encode(self, latent: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Per-decoder-layer cross-attention keys and values for a batch of latents."""

        batch = latent.shape[0]
        memory = self.to_memory(latent.to(self.memory_position.dtype)).view(
            batch, self.settings.memory_tokens, -1
        )
        memory = self.memory_norm(memory + self.memory_position)
        return [
            cast(_DecoderBlock, block).cross_attention.keys_values(memory) for block in self.decoder
        ]

    def reconstruct_logits(
        self, tokens: Tensor, order: Tensor | None, latent: Tensor, input_dropout: float
    ) -> Tensor:
        """Teacher-forced logits from `latent`, at original positions."""

        if self.config.order == "layout":
            order = None
        inputs, extras = self.step_inputs(tokens, order, 0, tokens.shape[1])
        if input_dropout > 0.0 and self.training:
            keep = torch.rand(inputs.shape, device=inputs.device) >= input_dropout
            inputs = inputs * keep
        logits, _ = self.decode(inputs, self.encode(latent), extras=extras)
        if order is None:
            return logits
        inverse = torch.argsort(order, dim=1)
        return logits.gather(1, inverse[..., None].expand(-1, -1, logits.shape[-1]))


def kl_per_dimension(mean: Tensor, log_variance: Tensor) -> Tensor:
    return 0.5 * (mean.pow(2) + log_variance.exp() - 1.0 - log_variance)


def _interpolation_sheet(
    model: LatentToProgram,
    latents: Tensor,
    pairs: list[tuple[int, int]],
    template: Any,
    steps: int,
) -> tuple[bytes, list[list[np.ndarray | None]]]:
    rows: list[list[np.ndarray | None]] = []
    for first, second in pairs:
        weights = torch.linspace(0.0, 1.0, steps, device=latents.device)[:, None]
        path = (1.0 - weights) * latents[first] + weights * latents[second]
        tokens = greedy_decode(model, path)
        rows.append(_programs_to_renders(tokens, model.layout, template))
    return contact_sheet(rows), rows


def train_and_evaluate(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    torch.manual_seed(config.training.seed)
    device = torch.device("cuda")
    plain, _, layout, pilot, dataset_hash = load_corpus(
        config.pilot_config, config.model.image_size
    )
    identity = run_identity(config_path, dataset_hash, config.slug)
    run_id = identity["run_id"]
    run_dir = REPO_ROOT / "runs" / run_id
    checkpoint_dir = CACHE_ROOT / "runs" / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists: {run_id}")
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    from mojidiff.learning.openmoji_pilot import _load_program

    template = _load_program(_pilot_rows(pilot)["primary/train"][0], pilot, layout.codec)
    train = plain["primary/train"]
    validation = plain["primary/validation"].subset(config.eval_icons)
    model = LatentToProgram(layout, config.model, config.latent).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _schedule(step, config.training)
    )
    record: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "state": "running",
        "planned_at": _now(),
        "hypothesis": config.hypothesis,
        "parent_run": config.parent_run,
        **{k: identity[k] for k in ("git_commit", "dirty_patch_sha256", "config_sha256")},
        "config": str(config_path),
        "config_resolved": {
            "model": asdict(config.model),
            "latent": asdict(config.latent),
            "training": asdict(config.training),
        },
        "dataset": {
            "pilot_config": str(config.pilot_config),
            "cache_sha256": dataset_hash,
            "train_icons": len(train.tokens),
            "evaluated_on": "primary/validation",
        },
        "seed": config.training.seed,
        "determinism": "seeded; cuDNN and SDPA kernels not forced deterministic",
        "model_parameters": parameter_count(model),
        "command": f".venv/bin/python -m mojidiff.learning.latent --config {config_path}",
        "outputs": {"run_dir": f"runs/{run_id}", "checkpoints": str(checkpoint_dir)},
    }
    _write_yaml(run_dir / "run.yaml", record)
    _append_registry(run_id, "running", config=str(config_path), slug=config.slug)
    print(json.dumps({"run_id": run_id, "parameters": parameter_count(model)}), flush=True)

    batches = _online_batches(
        train, layout, template, config.model.image_size, config.training, render_images=False
    )
    validation_orders = split_orders(validation.tokens, layout)
    settings = config.latent
    metrics_path = run_dir / "metrics.jsonl"
    best_loss = float("inf")
    reset_peak_memory(device)
    started = time.perf_counter()
    try:
        for step in range(1, config.training.steps + 1):
            model.train()
            _, tokens, masks, orders, _ = next(batches)
            tokens, masks = tokens.to(device), masks.to(device)
            order = cast(Tensor, orders).to(device)
            beta = settings.beta * min(
                1.0, step / max(1.0, settings.beta_warmup_fraction * config.training.steps)
            )
            with torch.autocast("cuda", dtype=torch.bfloat16):
                mean, log_variance = model.posterior(tokens)
                latent = mean + torch.randn_like(mean) * (0.5 * log_variance).exp()
                logits = model.reconstruct_logits(tokens, order, latent, settings.input_dropout)
            free = masks.sum(dim=-1) > 1
            reconstruction = (
                F.cross_entropy(
                    logits.float().masked_fill(~masks, float("-inf"))[free],
                    tokens[free],
                    reduction="sum",
                )
                / tokens.shape[0]
            )
            kl = kl_per_dimension(mean.float(), log_variance.float())
            kl_term = kl.clamp_min(settings.free_bits).sum(dim=-1).mean()
            loss = reconstruction + beta * kl_term
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step % config.training.eval_every == 0 or step == config.training.steps:
                held_out = _held_out(model, validation, validation_orders, device)
                entry = {
                    "step": step,
                    "train_reconstruction_nats_per_icon": float(reconstruction.detach()),
                    "train_kl_nats": float(kl.sum(dim=-1).mean().detach()),
                    "beta": beta,
                    **held_out,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(entry) + "\n")
                print(json.dumps(entry), flush=True)
                total = held_out["held_out_reconstruction_nats"] + held_out["held_out_kl_nats"]
                if total < best_loss:
                    best_loss = total
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "step": step,
                            "config": asdict(config.model),
                            "latent": asdict(settings),
                        },
                        checkpoint_dir / "best.pt",
                    )
    except Exception as error:
        record.update(state="failed", failed_at=_now(), failure_reason=repr(error))
        _write_yaml(run_dir / "run.yaml", record)
        _append_registry(run_id, "failed", reason=repr(error))
        raise
    train_seconds = time.perf_counter() - started
    best = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    model.eval()
    result = _final_evaluation(model, plain, validation, template, device, run_dir)
    result["selected_step"] = int(best["step"])

    graph_latent = torch.zeros(1, settings.latent_dim, device=device)
    from mojidiff.learning.fast_decode import GraphDecoder

    graph = GraphDecoder(model, dtype=torch.float32)
    stats = DecodeStats()
    graph.decode(graph_latent[0], stats=stats)
    latency = measure_latency(lambda: graph.decode(graph_latent[0]), device=device, repeats=10)
    resource = resource_summary(device, train_seconds=train_seconds, latency=latency).as_record()
    resource.update(
        decoder="fast_decode.GraphDecoder, float32, batch 1, from a latent",
        decoder_calls=stats.model_calls,
        excludes="rasterising the output SVG",
        note="measured while other GPU work may be running; see gpu load at the time",
    )
    record.update(
        state="completed",
        completed_at=_now(),
        resource=resource,
        baselines={
            "nearest_training_icon_pixel_error": result["nearest_training_icon_pixel_error"][0],
            "note": "retrieval stores every training icon; the latent stores 64 numbers",
        },
        result=result,
    )
    _write_yaml(run_dir / "run.yaml", record)
    (run_dir / "result.md").write_text(_markdown(record))
    (run_dir / "artifacts.json").write_text(
        json.dumps(
            {
                "checkpoint_best": str(checkpoint_dir / "best.pt"),
                "reconstructions": "reconstructions.png (reference, reconstruction)",
                "interpolations": "interpolations.png (one pair per row)",
                "samples": "prior-samples.png",
            },
            indent=2,
        )
        + "\n"
    )
    _append_registry(run_id, "completed", result_file=f"runs/{run_id}/result.md")
    print(json.dumps({"completed": run_id, "result": result}), flush=True)
    return record


@torch.no_grad()
def _held_out(
    model: LatentToProgram, split: Any, orders: Tensor, device: torch.device, count: int = 64
) -> dict[str, float]:
    model.eval()
    reconstruction = 0.0
    kl_total = 0.0
    correct = 0
    seen = 0
    for start in range(0, count, 32):
        indices = np.arange(start, min(start + 32, count))
        tokens = split.tokens[indices].to(device)
        masks = split.mask_batch(indices).to(device)
        order = orders[indices].to(device)
        mean, log_variance = model.posterior(tokens)
        logits = model.reconstruct_logits(tokens, order, mean, 0.0).float()
        free = masks.sum(dim=-1) > 1
        scores = logits.masked_fill(~masks, float("-inf"))[free]
        reconstruction += float(F.cross_entropy(scores, tokens[free], reduction="sum"))
        kl_total += float(kl_per_dimension(mean, log_variance).sum())
        correct += int((scores.argmax(-1) == tokens[free]).sum())
        seen += int(free.sum())
    icons = min(count, len(split.tokens))
    return {
        "held_out_reconstruction_nats": reconstruction / icons,
        "held_out_kl_nats": kl_total / icons,
        "held_out_free_token_accuracy": correct / max(seen, 1),
    }


@torch.no_grad()
def _final_evaluation(
    model: LatentToProgram,
    plain: dict[str, Any],
    validation: Any,
    template: Any,
    device: torch.device,
    run_dir: Path,
) -> dict[str, Any]:
    count = len(validation.tokens)
    means = []
    for start in range(0, count, 32):
        mean, _ = model.posterior(validation.tokens[start : start + 32].to(device))
        means.append(mean)
    latents = torch.cat(means)
    decoded = torch.cat(
        [greedy_decode(model, latents[start : start + 64]) for start in range(0, count, 64)]
    )
    renders = _programs_to_renders(decoded, model.layout, template)
    errors = [pixel_error(r, validation.targets[i].numpy()) for i, r in enumerate(renders)]
    library = plain["primary/train"]
    _, nearest_errors = nearest_training_icon(validation.targets, library.targets, device)
    # The control for "the latent carries the icon": one decode of the prior mean,
    # scored against every icon. Per-icon latents must beat it, paired.
    zero = greedy_decode(model, torch.zeros(1, model.settings.latent_dim, device=device))
    zero_render = _programs_to_renders(zero, model.layout, template)[0]
    zero_errors = [pixel_error(zero_render, validation.targets[i].numpy()) for i in range(count)]
    carried = [z - e for e, z in zip(errors, zero_errors, strict=True)]
    (run_dir / "reconstructions.png").write_bytes(
        contact_sheet([[validation.targets[i].numpy(), renders[i]] for i in range(24)])
    )
    rng = np.random.default_rng(17)
    pairs = [tuple(int(v) for v in rng.choice(count, 2, replace=False)) for _ in range(8)]
    sheet, _ = _interpolation_sheet(model, latents, pairs, template, steps=7)  # type: ignore[arg-type]
    (run_dir / "interpolations.png").write_bytes(sheet)
    generator = torch.Generator(device=device).manual_seed(23)
    prior = torch.randn(32, model.settings.latent_dim, device=device, generator=generator)
    samples = greedy_decode(model, prior)
    sample_renders = _programs_to_renders(samples, model.layout, template)
    grid = [sample_renders[row * 8 : row * 8 + 8] for row in range(4)]
    (run_dir / "prior-samples.png").write_bytes(contact_sheet(grid))
    sample_targets = torch.from_numpy(
        np.stack(
            [
                r if r is not None else np.full((72, 72, 3), 255, dtype=np.uint8)
                for r in sample_renders
            ]
        )
    )
    _, sample_nearest = nearest_training_icon(sample_targets, library.targets, device)
    distinct = len({tuple(row.tolist()) for row in samples})
    return {
        "icons": count,
        "reconstruction_pixel_error": bootstrap_mean_interval(errors),
        "nearest_training_icon_pixel_error": bootstrap_mean_interval(
            [float(v) for v in nearest_errors]
        ),
        "prior_mean_decode_pixel_error": bootstrap_mean_interval(zero_errors),
        "latent_minus_prior_mean_error_reduction": bootstrap_mean_interval(carried),
        "exact_reconstruction_rate": float(
            np.mean([torch.equal(decoded[i], validation.tokens[i]) for i in range(count)])
        ),
        "prior_samples": {
            "count": len(samples),
            "distinct": distinct,
            "rendered": sum(1 for r in sample_renders if r is not None),
            "median_pixel_distance_to_nearest_training_icon": float(
                np.median(sample_nearest.numpy())
            ),
        },
    }


def _markdown(record: dict[str, Any]) -> str:
    result = record["result"]
    resource = record["resource"]
    reconstruction = result["reconstruction_pixel_error"]
    nearest = result["nearest_training_icon_pixel_error"]
    prior = result["prior_samples"]
    return "\n".join(
        [
            f"# {record['run_id']}",
            "",
            f"**Hypothesis.** {record['hypothesis']}",
            "",
            "| measure | value |",
            "| --- | --- |",
            f"| validation icons | {result['icons']} |",
            f"| reconstruction pixel error | {reconstruction[0]:.4f} "
            f"[{reconstruction[1]:.4f}, {reconstruction[2]:.4f}] |",
            f"| nearest training icon pixel error | {nearest[0]:.4f} |",
            "| prior-mean decode pixel error (control) | "
            f"{result['prior_mean_decode_pixel_error'][0]:.4f} |",
            "| error reduction, own latent vs prior mean | "
            f"{result['latent_minus_prior_mean_error_reduction'][0]:.4f} "
            f"[{result['latent_minus_prior_mean_error_reduction'][1]:.4f}, "
            f"{result['latent_minus_prior_mean_error_reduction'][2]:.4f}] |",
            f"| exact reconstructions | {result['exact_reconstruction_rate']:.3f} |",
            f"| prior samples distinct / rendered | {prior['distinct']} / {prior['rendered']} "
            f"of {prior['count']} |",
            "| prior samples, median pixel distance to nearest training icon | "
            f"{prior['median_pixel_distance_to_nearest_training_icon']:.4f} |",
            f"| parameters | {record['model_parameters']:,} |",
            f"| train seconds | {resource['train_seconds']:.0f} |",
            f"| decode ms per icon (graph, batch 1) | {resource['inference_ms_per_icon']:.1f} |",
            "",
            "Sheets: `reconstructions.png`, `interpolations.png`, `prior-samples.png`.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    train_and_evaluate(args.config)


if __name__ == "__main__":
    main()
