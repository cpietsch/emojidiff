"""The canvas latent, stage A: a variational v9 transcriber ("VT").

v9 reads a 144 px render into an 18 x 18 grid of 256-d cells and writes the program by
cross-attending to that grid; the grid is its record of where ink remains. VT keeps the
grid and makes it a stochastic bottleneck:

* **Posterior q(z | x).** v9's conv stem, then a per-cell Linear 256 -> 2c: a mean and a
  log-variance per cell, and z = mu + sigma * eps in R^{c x 18 x 18} (2,592 numbers at
  c = 8, one c-vector per 4 x 4 view-unit patch).
* **Decoder p(program | z).** A per-cell Linear c -> 256, v9's grid embedding and
  Fourier cell features, v9's two bidirectional encoder blocks and `encoder_norm` as a
  "reader", then v9's six-layer causal decoder. The decoder cross-attends to the same
  324 memory cells as before (`memory_length` is unchanged), with the same metric
  coordinates, path-major order and grammar masks.

`encode` dispatches on its input: a uint8 NHWC render encodes to its posterior *mean*;
a float (B, c, 18, 18) grid is read as a latent. `greedy_decode` and
`fast_decode.GraphDecoder` call nothing but `encode`, so both reconstruct from renders
and decode sampled or interpolated latents without a change.

Initialisation. Every weight shared with v9 comes from v9's best checkpoint. The down
and up projections come from the top-c principal directions of v9's stem features
over the training renders, whitened, so each latent channel has unit variance over
the training cells and step 0 is v9 with its stem projected to rank c. The
log-variance bias starts at -6 (sigma ~ 0.05), so training starts nearly deterministic
and compresses from above.

Loss. v9's per-free-token cross-entropy plus beta * KL / F, where KL is nats per icon
and F the batch's mean free-token count: exactly (sum CE + beta * sum KL) / free
tokens, a beta-weighted ELBO per free token. A multiplicative controller holds KL near
a declared budget of C nats per icon, log beta += rate * (KL_ema - C) / C with beta
clipped to [beta_min, beta_max]. The posterior statistics and the KL are computed in
float32 under bf16 autocast.

Naming. z is the latent grid, mu and sigma its posterior. This is a variational
autoencoder with an autoregressive program decoder; a later stage fits a flow prior
over z. Neither is a D3PM.

Precision. Evaluation, the pre-flight and the PCA run in IEEE float32 (TF32 off,
`precision.strict_float32`): with TF32 on, greedy decodes of the same latent differ with
the batch they sit in. Training keeps the environment's TF32 under bf16 autocast.

Provenance. A real run refuses to start while anything under src/, configs/ or tests/
is untracked or uncommitted (its record could not identify its code), or without the
pre-flight report its config names, made from the same parent and data, whose chosen
channel count is the config's.

    python -m mojidiff.learning.pixel_latent --config configs/latent/vt-v1.yaml
    python -m mojidiff.learning.pixel_latent --config ... --preflight [--out PATH]
    python -m mojidiff.learning.pixel_latent --config ... --smoke DIR [--steps 20 --batch-size 4]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from mojidiff.learning.autoregressive import SequenceLayout
from mojidiff.learning.latent import kl_per_dimension
from mojidiff.learning.precision import float32_flags, strict_float32, tf32_enabled
from mojidiff.learning.render2svg import (
    CACHE_ROOT,
    EVAL_SIZE,
    REPO_ROOT,
    DecodeStats,
    ModelConfig,
    RenderToProgram,
    SplitData,
    TrainConfig,
    _append_registry,
    _DecoderBlock,
    _git,
    _now,
    _online_batches,
    _pilot_rows,
    _programs_to_renders,
    _schedule,
    _train_config,
    _write_yaml,
    bootstrap_mean_interval,
    contact_sheet,
    decode_and_score,
    greedy_decode,
    load_corpus,
    nearest_training_icon,
    parameter_count,
    pixel_error,
    run_identity,
    split_orders,
    teacher_forced_loss,
    uncommitted_sources,
)
from mojidiff.learning.telemetry import (
    LatencyReport,
    measure_latency,
    reset_peak_memory,
    resource_summary,
)

NEW_WEIGHTS = frozenset(
    {"to_posterior.weight", "to_posterior.bias", "from_latent.weight", "from_latent.bias"}
)
"""VT's only parameters that v9 does not have."""

POSTERIOR_SAMPLE_SEED = 29
PRIOR_SEED = 23
PAIR_SEED = 17
INK_THRESHOLD = 250
"""A pixel is ink when its smallest channel is below this (the plan's definition)."""
PREFLIGHT_ICONS = 64
PREFLIGHT_THRESHOLD = 0.12
"""The declared pre-flight rule: rank 8 at or below this error keeps c = 8; above it,
c = 16 if rank 16 is at or below it; if both are above, no outcome was declared."""


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class CanvasSettings:
    channels: int = 8
    """c: latent channels per grid cell."""
    budget_nats: float = 2000.0
    """C: the KL per icon, in nats, that the controller holds."""
    beta_initial: float = 1e-4
    beta_min: float = 1e-4
    beta_max: float = 1.0
    beta_rate: float = 0.01
    """log beta moves by rate * (KL_ema - C) / C per step."""
    kl_ema_decay: float = 0.9
    """Smoothing of the per-batch KL the controller reads (time constant ~10 steps)."""
    band_low: float = 0.8
    band_high: float = 1.2
    """Checkpoints are selected among evaluations with held-out KL in [low C, high C]."""
    log_variance_bias: float = -6.0
    pca_icons: int | None = None
    """Training renders whose stem features set the projections; None uses all."""


@dataclass(frozen=True)
class CanvasConfig:
    slug: str
    hypothesis: str
    pilot_config: Path
    parent_run: str
    parent_checkpoint: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    canvas: CanvasSettings = field(default_factory=CanvasSettings)
    training: TrainConfig = field(default_factory=TrainConfig)
    eval_icons: int = 339
    prior_samples: int = 64
    interpolation_pairs: int = 8
    interpolation_frames: int = 9
    non_inferiority_margin: float = 0.015
    """A1: the upper 95% bound of (VT - v9) pixel error, paired, must not exceed this."""
    sample_cost_margin: float = 0.010
    """A3: decoding one posterior sample instead of mu may cost at most this, paired mean."""
    trace_every: int = 50
    """Training steps between controller rows (KL, beta) in metrics.jsonl."""
    preflight_report: Path | None = None
    """The `--preflight` report the declared channel rule was applied to; a real run
    refuses to start without it (`check_preflight`)."""
    notes: str = ""
    channel_fallback_from: str | None = None
    """The declared second chance (latent plan, arm 2): a run at c = 16 is allowed when
    this names a completed VT run that used c = 8 from the same pre-flight report and
    failed A1. A second failure stops the canvas line; `check_preflight` enforces it."""


FALLBACK_CHANNELS = 16


def load_config(path: Path) -> CanvasConfig:
    root = yaml.safe_load(path.read_text())
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise ValueError("canvas latent config needs schema_version: 1")
    known = {
        "schema_version", "slug", "hypothesis", "pilot_config", "parent_run",
        "parent_checkpoint", "model", "canvas", "training", "eval_icons", "prior_samples",
        "interpolation_pairs", "interpolation_frames", "non_inferiority_margin",
        "sample_cost_margin", "trace_every", "preflight_report", "notes",
        "channel_fallback_from",
    }  # fmt: skip
    unknown = sorted(set(root) - known)
    if unknown:
        raise ValueError(f"unknown canvas config fields: {', '.join(unknown)}")
    config = CanvasConfig(
        slug=str(root["slug"]),
        hypothesis=str(root["hypothesis"]).strip(),
        pilot_config=Path(str(root["pilot_config"])),
        parent_run=str(root["parent_run"]),
        parent_checkpoint=Path(str(root["parent_checkpoint"])),
        model=ModelConfig(**root.get("model", {})),
        canvas=CanvasSettings(**root.get("canvas", {})),
        training=_train_config(root.get("training", {})),
        eval_icons=int(root.get("eval_icons", 339)),
        prior_samples=int(root.get("prior_samples", 64)),
        interpolation_pairs=int(root.get("interpolation_pairs", 8)),
        interpolation_frames=int(root.get("interpolation_frames", 9)),
        non_inferiority_margin=float(root.get("non_inferiority_margin", 0.015)),
        sample_cost_margin=float(root.get("sample_cost_margin", 0.010)),
        trace_every=int(root.get("trace_every", 50)),
        preflight_report=(
            Path(str(root["preflight_report"])) if root.get("preflight_report") else None
        ),
        notes=str(root.get("notes", "")).strip(),
        channel_fallback_from=(
            str(root["channel_fallback_from"]) if root.get("channel_fallback_from") else None
        ),
    )
    settings = config.canvas
    if not settings.beta_min <= settings.beta_initial <= settings.beta_max:
        raise ValueError("beta_initial must lie in [beta_min, beta_max]")
    if not config.training.augment_online:
        raise ValueError("the canvas latent trains on v9's online stream: augment_online: true")
    return config


# --------------------------------------------------------------------------- model


def stem_features(model: RenderToProgram, images: Tensor) -> Tensor:
    """A transcriber's conv stem on uint8 NHWC renders: (B, cells, d_model), before any
    position is added."""

    pixels = images.permute(0, 3, 1, 2).to(model.grid_embedding.dtype) / 127.5 - 1.0
    return cast(Tensor, model.stem(pixels)).flatten(2).transpose(1, 2)


class VariationalTranscriber(RenderToProgram):
    """v9 with a stochastic per-cell bottleneck between its stem and its reader."""

    def __init__(
        self, layout: SequenceLayout, config: ModelConfig, settings: CanvasSettings
    ) -> None:
        super().__init__(layout, config)
        self.settings = settings
        self.grid = config.image_size // 8
        channels = settings.channels
        self.to_posterior = nn.Linear(config.d_model, 2 * channels)
        self.from_latent = nn.Linear(channels, config.d_model)
        with torch.no_grad():
            # The log-variance starts input-independent: sigma = exp(bias / 2) everywhere.
            self.to_posterior.weight[channels:].zero_()
            self.to_posterior.bias[channels:].fill_(settings.log_variance_bias)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.settings.channels, self.grid, self.grid)

    def posterior(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Mean and log-variance of q(z | render), each (B, c, grid, grid), in float32."""

        features = stem_features(self, images)
        with torch.autocast(features.device.type, enabled=False):
            stats = self.to_posterior(features.float())
        grids = stats.transpose(1, 2).reshape(len(images), -1, self.grid, self.grid)
        mean, log_variance = grids.chunk(2, dim=1)
        return mean, log_variance.clamp(-10.0, 10.0)

    def sample_posterior(self, images: Tensor, noise: Tensor | None = None) -> Tensor:
        """z = mu + sigma * eps; `noise` is eps when given (same shape as mu)."""

        mean, log_variance = self.posterior(images)
        if noise is None:
            noise = torch.randn_like(mean)
        return mean + noise.to(mean.device, mean.dtype) * (0.5 * log_variance).exp()

    def memory_from_latent(self, latent: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Per-decoder-layer cross-attention keys and values for (B, c, grid, grid) latents."""

        if tuple(latent.shape[1:]) != self.latent_shape:
            raise ValueError(f"expected latents (B, {self.latent_shape}), got {latent.shape}")
        cells = latent.flatten(2).transpose(1, 2).to(self.grid_embedding.dtype)
        features = self.from_latent(cells) + self.grid_embedding
        if self.config.metric:
            positions = cast(Tensor, self._cell_features).to(features.dtype)
            features = features + self.grid_projection(positions)[None]
        for block in self.encoder:
            features = block(features)
        memory = self.encoder_norm(features)
        return [
            cast(_DecoderBlock, block).cross_attention.keys_values(memory) for block in self.decoder
        ]

    def encode(self, inputs: Tensor) -> list[tuple[Tensor, Tensor]]:
        """uint8 NHWC renders -> their posterior means' memory; float latent grids -> theirs."""

        if inputs.dtype == torch.uint8:
            mean, _ = self.posterior(inputs)
            return self.memory_from_latent(mean)
        if inputs.is_floating_point() and tuple(inputs.shape[1:]) == self.latent_shape:
            return self.memory_from_latent(inputs)
        raise ValueError(
            "encode takes uint8 (B, H, W, 3) renders or float (B, "
            f"{', '.join(map(str, self.latent_shape))}) latents, got {inputs.dtype} "
            f"{tuple(inputs.shape)}"
        )


def gaussian_kl(mean: Tensor, log_variance: Tensor) -> Tensor:
    """KL(q || N(0, I)) per icon, in nats, summed over every latent dimension."""

    return kl_per_dimension(mean.float(), log_variance.float()).flatten(1).sum(dim=1)


# --------------------------------------------------------------------------- initialisation


@dataclass(frozen=True)
class StemPCA:
    """Principal directions of a transcriber's stem features over a set of renders."""

    mean: Tensor  # (d,)
    variances: Tensor  # (d,), descending
    directions: Tensor  # (d, d), column k is the k-th direction
    icons: int
    cells: int

    def explained(self, rank: int) -> float:
        total = float(self.variances.sum())
        return float(self.variances[:rank].sum()) / total if total > 0 else 0.0


@torch.no_grad()
def stem_pca(
    model: RenderToProgram, images: Tensor, device: torch.device, batch_size: int = 64
) -> StemPCA:
    """PCA of every grid cell's stem feature over `images` (uint8 NHWC), in float32
    features and float64 moments."""

    width = model.config.d_model
    total = torch.zeros(width, dtype=torch.float64, device=device)
    outer = torch.zeros(width, width, dtype=torch.float64, device=device)
    count = 0
    for start in range(0, len(images), batch_size):
        chunk = images[start : start + batch_size].to(device)
        with torch.autocast(device.type, enabled=False):
            features = stem_features(model, chunk).reshape(-1, width).double()
        total += features.sum(dim=0)
        outer += features.T @ features
        count += len(features)
    mean = total / count
    covariance = (outer / count - torch.outer(mean, mean)).cpu()
    covariance = 0.5 * (covariance + covariance.T)
    variances, directions = torch.linalg.eigh(covariance)
    return StemPCA(
        mean=mean.float().cpu(),
        variances=variances.flip(0).clamp_min(0.0).float(),
        directions=directions.flip(1).float(),
        icons=len(images),
        cells=count,
    )


@torch.no_grad()
def set_projections(model: VariationalTranscriber, pca: StemPCA) -> None:
    """Down and up projections from the top-c directions, whitened: mu_k = v_k . (x - m)
    / s_k and x_hat = m + sum_k s_k mu_k v_k, so from_latent(mu) is x projected on the
    top c directions and mu has unit variance per channel over the PCA's cells."""

    channels = model.settings.channels
    if channels > len(pca.variances):
        raise ValueError(f"{channels} channels from a {len(pca.variances)}-d PCA")
    top = pca.directions[:, :channels]
    floor = max(float(pca.variances[0]) * 1e-10, 1e-12)
    scale = pca.variances[:channels].clamp_min(floor).sqrt()
    down = (top / scale).T
    model.to_posterior.weight[:channels].copy_(down)
    model.to_posterior.bias[:channels].copy_(-(down @ pca.mean))
    model.to_posterior.weight[channels:].zero_()
    model.to_posterior.bias[channels:].fill_(model.settings.log_variance_bias)
    model.from_latent.weight.copy_(top * scale)
    model.from_latent.bias.copy_(pca.mean)


def load_transcriber_weights(model: VariationalTranscriber, state: Mapping[str, Tensor]) -> None:
    """Every weight a transcriber shares with VT; refuses anything but an exact fit."""

    missing, unexpected = model.load_state_dict(dict(state), strict=False)
    if set(missing) != NEW_WEIGHTS or unexpected:
        raise ValueError(
            f"transcriber weights do not fit: missing {sorted(set(missing) - NEW_WEIGHTS)}, "
            f"unexpected {sorted(unexpected)}"
        )


def from_transcriber(
    layout: SequenceLayout,
    parent: Mapping[str, Any],
    settings: CanvasSettings,
    pca: StemPCA,
) -> VariationalTranscriber:
    """VT from a transcriber checkpoint (`model`, `config`) and a PCA of its stem."""

    model = VariationalTranscriber(layout, ModelConfig(**parent["config"]), settings)
    load_transcriber_weights(model, parent["model"])
    set_projections(model, pca)
    return model


def checkpoint_state(
    model: VariationalTranscriber, step: int, controller: BetaController | None = None
) -> dict[str, Any]:
    """What a VT checkpoint holds; `load_checkpoint` and the gallery read it."""

    return {
        "model": model.state_dict(),
        "step": step,
        "config": asdict(model.config),
        "canvas": asdict(model.settings),
        "controller": asdict(controller) if controller is not None else None,
    }


def load_checkpoint(state: Mapping[str, Any], layout: SequenceLayout) -> VariationalTranscriber:
    """A VT with its checkpoint's weights, on the CPU, in eval mode."""

    model = VariationalTranscriber(
        layout, ModelConfig(**state["config"]), CanvasSettings(**state["canvas"])
    )
    model.load_state_dict(state["model"])
    return model.eval()


# --------------------------------------------------------------------------- training


@dataclass
class BetaController:
    """log beta += rate * (KL_ema - C) / C, clipped: holds KL near C nats per icon."""

    budget: float
    rate: float
    minimum: float
    maximum: float
    decay: float
    log_beta: float
    kl_ema: float | None = None

    @classmethod
    def from_settings(cls, settings: CanvasSettings) -> BetaController:
        return cls(
            budget=settings.budget_nats,
            rate=settings.beta_rate,
            minimum=settings.beta_min,
            maximum=settings.beta_max,
            decay=settings.kl_ema_decay,
            log_beta=math.log(settings.beta_initial),
        )

    @property
    def beta(self) -> float:
        return math.exp(self.log_beta)

    @property
    def at_floor(self) -> bool:
        return self.log_beta <= math.log(self.minimum) + 1e-9

    def update(self, kl: float) -> float:
        """Fold in one batch's mean KL per icon; the beta for the next step."""

        if self.kl_ema is None:
            self.kl_ema = kl
        else:
            self.kl_ema = self.decay * self.kl_ema + (1.0 - self.decay) * kl
        moved = self.log_beta + self.rate * (self.kl_ema - self.budget) / self.budget
        self.log_beta = min(max(moved, math.log(self.minimum)), math.log(self.maximum))
        return self.beta


@dataclass
class LossParts:
    loss: Tensor
    cross_entropy: Tensor
    """Mean over free tokens."""
    kl: Tensor
    """Mean KL per icon, nats."""
    free_tokens: Tensor
    """F: mean free-token count per icon in the batch."""


def variational_loss(
    model: VariationalTranscriber,
    images: Tensor,
    tokens: Tensor,
    masks: Tensor,
    order: Tensor | None,
    beta: float,
    *,
    bf16: bool,
    noise: Tensor | None = None,
) -> LossParts:
    """CE per free token + beta * KL / F, from one posterior sample per icon."""

    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        mean, log_variance = model.posterior(images)
        if noise is None:
            noise = torch.randn_like(mean)
        latent = mean + noise * (0.5 * log_variance).exp()
        logits = model(latent, tokens, order)
    free = masks.sum(dim=-1) > 1
    cross_entropy = F.cross_entropy(
        logits.float().masked_fill(~masks, float("-inf"))[free], tokens[free]
    )
    kl = gaussian_kl(mean, log_variance).mean()
    free_tokens = free.sum(dim=1).float().mean()
    return LossParts(cross_entropy + beta * kl / free_tokens, cross_entropy, kl, free_tokens)


@torch.no_grad()
def posterior_statistics(
    model: VariationalTranscriber, images: Tensor, device: torch.device, batch_size: int = 64
) -> tuple[Tensor, Tensor]:
    """Posterior means and log-variances of uint8 renders, float32, on `device`."""

    model.eval()
    means: list[Tensor] = []
    log_variances: list[Tensor] = []
    for start in range(0, len(images), batch_size):
        mean, log_variance = model.posterior(images[start : start + batch_size].to(device))
        means.append(mean)
        log_variances.append(log_variance)
    return torch.cat(means), torch.cat(log_variances)


@torch.no_grad()
def decode_latents(model: VariationalTranscriber, latents: Tensor, batch_size: int = 64) -> Tensor:
    """Float32 greedy programs for a batch of latent grids."""

    model.eval()
    return torch.cat(
        [
            greedy_decode(model, latents[start : start + batch_size])
            for start in range(0, len(latents), batch_size)
        ]
    )


def ink_coverage(render: np.ndarray | None) -> float:
    """Share of pixels whose smallest channel is below `INK_THRESHOLD`; 0 if unrendered."""

    if render is None:
        return 0.0
    return float((render.min(axis=-1) < INK_THRESHOLD).mean())


def precision_note() -> str:
    """What "float32" meant, from the flags in force: evaluation runs inside
    `strict_float32`, where TF32 is off; outside it this environment routes float32
    matmuls and convolutions through TF32, and greedy decodes depend on the batch."""

    flags = float32_flags()
    if not tf32_enabled():
        return "IEEE float32 greedy decoding (TF32 off: matmul and cuDNN); pixel metrics at 72 px"
    return (
        f"float32 greedy decoding with TF32 on (matmul {flags['matmul_allow_tf32']}, "
        f"cuDNN {flags['cudnn_allow_tf32']}, precision {flags['float32_matmul_precision']}); "
        "pixel metrics at 72 px"
    )


def preflight_choice(errors: Mapping[int, float]) -> tuple[int | None, str]:
    """The declared channel rule on the pre-flight's mean pixel error per rank: the
    channel count, or None where the rule declares no outcome; and why."""

    limit = PREFLIGHT_THRESHOLD
    if 8 not in errors:
        return None, "no rank-8 decode: the rule cannot be applied"
    if errors[8] <= limit:
        return 8, f"rank-8 error {errors[8]:.4f} <= {limit}: c = 8"
    if 16 in errors and errors[16] <= limit:
        return 16, f"rank-8 error {errors[8]:.4f} > {limit}, rank-16 {errors[16]:.4f} <= it: c = 16"
    rank_16 = f"{errors[16]:.4f}" if 16 in errors else "not measured"
    return None, (
        f"rank-8 error {errors[8]:.4f} and rank-16 error {rank_16} both above {limit}: the "
        "rule declares no outcome; the operator decides before any run"
    )


def hypothesis_channels(hypothesis: str) -> int | None:
    """The channel count a hypothesis names first ("c = 8"), if any."""

    found = re.search(r"\bc = (\d+)\b", hypothesis)
    return int(found.group(1)) if found else None


def check_preflight(config: CanvasConfig, dataset_hash: str, pca_icons: int) -> dict[str, Any]:
    """The pre-flight rule, enforced before a real run: the report the config names
    exists, was made from this parent, this data and `pca_icons` training renders over
    the declared icons, and chose the config's channel count, which the hypothesis names
    first. Returns what the run record keeps; raises SystemExit otherwise."""

    path = config.preflight_report
    if path is None or not path.is_file():
        raise SystemExit(
            f"no pre-flight report ({path}): run `python -m mojidiff.learning.pixel_latent "
            "--config ... --preflight --out <path>`, apply the channel rule and set "
            "`preflight_report` in the config"
        )
    report = json.loads(path.read_text())
    channels = config.canvas.channels
    problems = []
    if report.get("parent_checkpoint") != str(config.parent_checkpoint):
        problems.append(f"made from {report.get('parent_checkpoint')}, not the parent")
    elif report.get("parent_checkpoint_sha256") != _sha256(config.parent_checkpoint):
        problems.append("the parent checkpoint changed since the pre-flight")
    if report.get("dataset_sha256") != dataset_hash:
        problems.append("made from other data")
    if report.get("icons") != PREFLIGHT_ICONS:
        problems.append(f"{report.get('icons')} icons, not the declared {PREFLIGHT_ICONS}")
    if report.get("pca", {}).get("icons") != pca_icons:
        problems.append(f"a PCA over {report.get('pca', {}).get('icons')} renders, not {pca_icons}")
    chosen = report.get("chosen_channels")
    rule_outcome = report.get("rule_outcome")
    fallback: dict[str, Any] | None = None
    if config.channel_fallback_from is not None:
        fallback, failure = _check_fallback(config, path, chosen)
        if failure:
            problems.append(failure)
        else:
            chosen = FALLBACK_CHANNELS
            rule_outcome = (
                f"declared fallback: {config.channel_fallback_from} failed A1 at c = "
                f"{report.get('chosen_channels')}; rerun once at c = {FALLBACK_CHANNELS}"
            )
    if chosen is None:
        problems.append(f"no declared outcome: {rule_outcome}")
    elif chosen != channels:
        problems.append(f"the rule chose c = {chosen}; the config has channels: {channels}")
    if hypothesis_channels(config.hypothesis) != channels:
        problems.append(f"the hypothesis does not name c = {channels} first")
    if problems:
        raise SystemExit(f"pre-flight check failed ({path}): " + "; ".join(problems))
    return {
        "report": str(path),
        "report_sha256": _sha256(path),
        "chosen_channels": chosen,
        "rule_outcome": rule_outcome,
        "fallback": fallback,
        "rank_errors": {
            key: report[key]["pixel_error"][0] for key in report if key.startswith("rank_")
        },
    }


def _check_fallback(
    config: CanvasConfig, report_path: Path, report_choice: object, runs_root: Path | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """The fallback's conditions, or the first one that fails."""

    run_id = str(config.channel_fallback_from)
    record_path = (runs_root or REPO_ROOT / "runs") / run_id / "run.yaml"
    if not record_path.is_file():
        return None, f"fallback run {run_id} has no run record"
    record = yaml.safe_load(record_path.read_text())
    preflight = (record.get("initialisation") or {}).get("preflight") or {}
    channels = ((record.get("config_resolved") or {}).get("canvas") or {}).get("channels")
    a1 = (((record.get("result") or {}).get("criteria") or {}).get("A1") or {}).get("pass")
    if record.get("state") != "completed":
        return None, f"fallback run {run_id} is {record.get('state')}, not completed"
    if channels != report_choice or report_choice == FALLBACK_CHANNELS:
        return None, f"fallback run {run_id} used c = {channels}, not the pre-flight's first choice"
    if Path(str(preflight.get("report"))) != report_path:
        return None, f"fallback run {run_id} used pre-flight {preflight.get('report')}"
    if preflight.get("report_sha256") != _sha256(report_path):
        return None, "the pre-flight report changed since the fallback run"
    if a1 is not False:
        return None, f"fallback run {run_id} did not fail A1 (pass = {a1})"
    return {"run": run_id, "failed": "A1", "run_channels": channels}, None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_parent(
    config: CanvasConfig, layout: SequenceLayout
) -> tuple[dict[str, Any], RenderToProgram]:
    state = torch.load(config.parent_checkpoint, map_location="cpu")
    parent_config = ModelConfig(**state["config"])
    if parent_config != config.model:
        raise ValueError(
            f"the config's model block {config.model} is not the parent's {parent_config}"
        )
    parent = RenderToProgram(layout, parent_config)
    parent.load_state_dict(state["model"])
    return state, parent.eval()


def smoke_config(config: CanvasConfig, steps: int, batch_size: int) -> CanvasConfig:
    """The real config, shrunk to a mechanics check: few steps and icons, a small PCA."""

    training = replace(
        config.training,
        steps=steps,
        batch_size=batch_size,
        eval_every=max(1, steps // 2),
        eval_icons=4,
        loader_workers=min(config.training.loader_workers, 2),
    )
    return replace(
        config,
        training=training,
        canvas=replace(config.canvas, pca_icons=64),
        eval_icons=4,
        prior_samples=4,
        interpolation_pairs=1,
        interpolation_frames=3,
        trace_every=5,
    )


def train_and_evaluate(
    config_path: Path,
    *,
    smoke: Path | None = None,
    steps: int = 20,
    batch_size: int = 4,
) -> dict[str, Any]:
    """Train VT from v9, select, evaluate, record.

    With `smoke`, the config is shrunk by `smoke_config` and everything - run record,
    metrics, sheets, checkpoints - is written under `smoke`/<run_id>; nothing goes to
    `runs/`, the cache's run directories or the registry. Without it, the run refuses
    to start while src/, configs/ or tests/ hold untracked or uncommitted files, or
    without a matching pre-flight report (`check_preflight`).
    """

    config = load_config(config_path)
    uncommitted = uncommitted_sources()
    if smoke is None and uncommitted:
        raise SystemExit(
            "commit first: a run record names a commit and hashes `git diff HEAD`, which "
            "cannot identify untracked or uncommitted code; uncommitted under src/, "
            "configs/, tests/:\n" + "\n".join(uncommitted)
        )
    if smoke is not None:
        config = smoke_config(config, steps, batch_size)
    torch.manual_seed(config.training.seed)
    device = torch.device("cuda")
    plain, _, layout, pilot, dataset_hash = load_corpus(
        config.pilot_config, config.model.image_size
    )
    train = plain["primary/train"]
    settings = config.canvas
    pca_count = min(settings.pca_icons or len(train.images), len(train.images))
    preflight_record: dict[str, Any] | str = (
        check_preflight(config, dataset_hash, pca_count) if smoke is None else "not checked (smoke)"
    )
    identity = run_identity(config_path, dataset_hash, config.slug)
    run_id = identity["run_id"]
    if smoke is None:
        run_dir = REPO_ROOT / "runs" / run_id
        checkpoint_dir = CACHE_ROOT / "runs" / run_id
    else:
        run_id = f"{run_id}-smoke"
        run_dir = smoke / run_id
        checkpoint_dir = run_dir / "checkpoints"
    if run_dir.exists():
        raise SystemExit(f"run directory already exists; preserve it and change the id: {run_id}")
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    registry = smoke is None

    from mojidiff.learning.openmoji_pilot import _load_program

    template = _load_program(_pilot_rows(pilot)["primary/train"][0], pilot, layout.codec)
    validation = plain["primary/validation"].subset(config.eval_icons)
    evaluation = validation.subset(min(config.training.eval_icons, len(validation.tokens)))

    parent_state, parent = _load_parent(config, layout)
    with strict_float32():
        # As the pre-flight computes it, so step 0 is the rank-c model it measured.
        pca = stem_pca(parent.to(device), train.images[:pca_count], device)
    parent.cpu()
    model = from_transcriber(layout, parent_state, settings, pca).to(device)
    controller = BetaController.from_settings(settings)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: _schedule(step, config.training)
    )
    command = f".venv/bin/python -m mojidiff.learning.pixel_latent --config {config_path}"
    if smoke is not None:
        command += f" --smoke {smoke} --steps {steps} --batch-size {batch_size}"
    record: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "state": "running",
        "planned_at": _now(),
        "hypothesis": config.hypothesis,
        "parent_run": config.parent_run,
        **{k: identity[k] for k in ("git_commit", "dirty_patch_sha256", "config_sha256")},
        "uncommitted_sources": uncommitted,
        "config": str(config_path),
        "config_resolved": {
            "model": asdict(config.model),
            "canvas": asdict(settings),
            "training": asdict(config.training),
            "eval_icons": config.eval_icons,
            "prior_samples": config.prior_samples,
            "interpolation_pairs": config.interpolation_pairs,
            "interpolation_frames": config.interpolation_frames,
            "non_inferiority_margin": config.non_inferiority_margin,
            "sample_cost_margin": config.sample_cost_margin,
        },
        "initialisation": {
            "parent_checkpoint": str(config.parent_checkpoint),
            "parent_checkpoint_sha256": _sha256(config.parent_checkpoint),
            "parent_step": parent_state.get("step"),
            "pca_icons": pca.icons,
            "pca_cells": pca.cells,
            "pca_explained_variance": {
                str(rank): pca.explained(rank) for rank in (4, 8, 16, 32, settings.channels)
            },
            "log_variance_bias": settings.log_variance_bias,
            "pca_precision": "IEEE float32 features (TF32 off), float64 moments",
            "preflight": preflight_record,
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
        "command": command,
        "outputs": {
            "run_dir": str(run_dir.relative_to(REPO_ROOT)) if smoke is None else str(run_dir),
            "checkpoints": str(checkpoint_dir),
        },
        "notes": config.notes,
    }
    if smoke is not None:
        record["smoke"] = {"steps": steps, "batch_size": batch_size, "registry": False}
    _write_yaml(run_dir / "run.yaml", record)
    if registry:
        _append_registry(run_id, "running", config=str(config_path), slug=config.slug)
    print(json.dumps({"run_id": run_id, "parameters": parameter_count(model)}), flush=True)

    batches = _online_batches(
        train, layout, template, config.model.image_size, config.training, render_images=True
    )
    evaluation_orders = split_orders(evaluation.tokens, layout)
    metrics_path = run_dir / "metrics.jsonl"
    bf16 = config.training.bf16
    budget = settings.budget_nats
    band = (settings.band_low * budget, settings.band_high * budget)
    best_in_band = float("inf")
    best_any = float("inf")
    selected: dict[str, Any] | None = None
    fallback: dict[str, Any] | None = None

    def fail(error: Exception) -> None:
        record.update(state="failed", failed_at=_now(), failure_reason=repr(error))
        _write_yaml(run_dir / "run.yaml", record)
        if registry:
            _append_registry(run_id, "failed", reason=repr(error))

    def write(entry: dict[str, Any]) -> None:
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    def evaluate(step: int, last: LossParts | None) -> dict[str, Any]:
        model.eval()
        nll, accuracy = teacher_forced_loss(
            model, evaluation, device, len(evaluation.tokens), bf16, evaluation_orders
        )
        errors, _, _, _ = decode_and_score(
            model, evaluation, template, device, len(evaluation.tokens), bf16=bf16
        )
        means, log_variances = posterior_statistics(model, evaluation.images, device)
        held_out_kl = float(gaussian_kl(means, log_variances).mean())
        return {
            "kind": "eval",
            "step": step,
            "train_cross_entropy": float(last.cross_entropy) if last else None,
            "train_kl_nats": float(last.kl) if last else None,
            "kl_ema": controller.kl_ema,
            "beta": controller.beta,
            "beta_at_floor": controller.at_floor,
            "held_out_kl_nats": held_out_kl,
            "held_out_kl_in_band": band[0] <= held_out_kl <= band[1],
            "eval_nll_per_free_token": nll,
            "eval_free_token_accuracy": accuracy,
            "eval_pixel_error_mean": float(np.mean(errors)),
            "learning_rate": scheduler.get_last_lr()[0],
            "elapsed_seconds": time.perf_counter() - started,
        }

    reset_peak_memory(device)
    started = time.perf_counter()
    try:
        # Step 0 is v9 with a rank-c stem and sigma ~ 0.05: a reference row, never selected.
        write(evaluate(0, None))
        for step in range(1, config.training.steps + 1):
            model.train()
            images, tokens, masks, orders, _ = next(batches)
            images, tokens, masks = images.to(device), tokens.to(device), masks.to(device)
            order = cast(Tensor, orders).to(device)
            beta = controller.beta
            parts = variational_loss(model, images, tokens, masks, order, beta, bf16=bf16)
            optimizer.zero_grad(set_to_none=True)
            parts.loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            controller.update(float(parts.kl.detach()))
            if step % config.trace_every == 0:
                write(
                    {
                        "kind": "train",
                        "step": step,
                        "cross_entropy": float(parts.cross_entropy.detach()),
                        "kl_nats": float(parts.kl.detach()),
                        "free_tokens": float(parts.free_tokens),
                        "beta_used": beta,
                        "kl_ema": controller.kl_ema,
                        "beta_next": controller.beta,
                    }
                )
            if step % config.training.eval_every == 0 or step == config.training.steps:
                entry = evaluate(step, parts)
                write(entry)
                current = float(entry["eval_pixel_error_mean"])
                summary = {
                    key: entry[key]
                    for key in ("step", "eval_pixel_error_mean", "held_out_kl_nats", "beta")
                }
                if current < best_any:
                    best_any = current
                    fallback = summary
                    torch.save(
                        checkpoint_state(model, step, controller), checkpoint_dir / "best-any.pt"
                    )
                if entry["held_out_kl_in_band"] and current < best_in_band:
                    best_in_band = current
                    selected = summary
                    torch.save(
                        checkpoint_state(model, step, controller), checkpoint_dir / "best.pt"
                    )
        latest = checkpoint_state(model, config.training.steps, controller)
        latest["optimizer"] = optimizer.state_dict()
        torch.save(latest, checkpoint_dir / "latest.pt")
    except Exception as error:
        fail(error)
        raise
    train_seconds = time.perf_counter() - started
    try:
        return _select_and_evaluate(
            model,
            parent,
            plain,
            template,
            device,
            config,
            record,
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            selected=selected,
            fallback=fallback,
            train_seconds=train_seconds,
            registry=registry,
            latency_repeats=10 if smoke is None else 2,
        )
    except Exception as error:
        fail(error)
        raise


def _select_and_evaluate(
    model: VariationalTranscriber,
    parent: RenderToProgram,
    plain: dict[str, SplitData],
    template: Any,
    device: torch.device,
    config: CanvasConfig,
    record: dict[str, Any],
    *,
    run_dir: Path,
    checkpoint_dir: Path,
    selected: dict[str, Any] | None,
    fallback: dict[str, Any] | None,
    train_seconds: float,
    registry: bool,
    latency_repeats: int,
) -> dict[str, Any]:
    """Load the selected checkpoint, evaluate it, and complete the run record."""

    settings = config.canvas
    run_id = str(record["run_id"])
    band = (settings.band_low * settings.budget_nats, settings.band_high * settings.budget_nats)
    fell_back = selected is None
    if fell_back:
        # No evaluation held KL inside the band: select the best overall, and say so.
        shutil.copyfile(checkpoint_dir / "best-any.pt", checkpoint_dir / "best.pt")
        selected = fallback
    selection = {
        "rule": (
            f"lowest {config.training.eval_icons}-icon bf16 greedy pixel error among evaluations "
            f"with held-out KL in [{band[0]:.0f}, {band[1]:.0f}] nats; else lowest overall"
        ),
        "fallback_no_eval_in_band": fell_back,
        **cast(dict[str, Any], selected),
    }
    best = torch.load(checkpoint_dir / "best.pt", map_location=device)
    model.load_state_dict(best["model"])
    model.eval()
    selected_controller = best.get("controller") or {}
    selected_beta_floor = (
        float(selected_controller.get("log_beta", 0.0)) <= math.log(settings.beta_min) + 1e-9
    )

    parent = parent.to(device)
    result, latency, decoder_calls = final_evaluation(
        model,
        parent,
        plain,
        template,
        device,
        run_dir,
        config,
        beta_at_floor=selected_beta_floor,
        latency_repeats=latency_repeats,
    )
    parent.cpu()
    result["selection"] = selection
    result["selected_step"] = int(best["step"])
    resource = resource_summary(device, train_seconds=train_seconds, latency=latency).as_record()
    resource.update(
        decoder=(
            "fast_decode.GraphDecoder, IEEE float32 (TF32 off), batch 1, from a uint8 render "
            "(mu path)"
        ),
        decoder_calls=decoder_calls,
        excludes="rasterising the output SVG",
        note="measured while other GPU work may be running",
    )
    record.update(
        state="completed",
        completed_at=_now(),
        resource=resource,
        baselines={
            "parent_v9_float32_pixel_error": result["parent_pixel_error"][0],
            "nearest_training_icon_pixel_error": result["nearest_training_icon_pixel_error"][0],
            "nearest_training_icon_pixel_error_without_mirrors": result[
                "nearest_training_icon_pixel_error_without_mirrors"
            ][0],
            "prior_mean_decode_pixel_error": result["prior_mean_decode_pixel_error"][0],
            "blank_canvas_pixel_error": result["blank_pixel_error"],
            "note": (
                "v9 re-decoded here in IEEE float32 on the same icons; the nearest training "
                "icon searches training renders and their mirrors; all comparisons paired"
            ),
        },
        result=result,
    )
    _write_yaml(run_dir / "run.yaml", record)
    (run_dir / "result.md").write_text(result_markdown(record))
    artifacts = {
        "checkpoint_best": str(checkpoint_dir / "best.pt"),
        "checkpoint_best_any": str(checkpoint_dir / "best-any.pt"),
        "checkpoint_latest": str(checkpoint_dir / "latest.pt"),
        "reconstructions": (
            "reconstructions.png (reference, VT from mu, VT from one posterior sample, "
            "v9 float32, nearest training render or mirror)"
        ),
        "interpolations": "interpolations.png (reference A, lerp of posterior means, reference B)",
        "samples": "prior-samples.png (z ~ N(0, I): the prior-hole control)",
        "per_icon": "per_icon.jsonl",
    }
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n")
    if registry:
        _append_registry(run_id, "completed", result_file=f"runs/{run_id}/result.md")
    print(json.dumps({"completed": run_id, "result": result}), flush=True)
    return record


# --------------------------------------------------------------------------- evaluation


def final_evaluation(
    model: VariationalTranscriber,
    parent: RenderToProgram,
    plain: dict[str, SplitData],
    template: Any,
    device: torch.device,
    run_dir: Path,
    config: CanvasConfig,
    *,
    beta_at_floor: bool,
    latency_repeats: int = 10,
) -> tuple[dict[str, Any], LatencyReport, int]:
    """Everything on the first `eval_icons` validation icons in IEEE float32 (TF32 off:
    with it on, a latent's greedy decode often depends on its batch), paired; with the
    graph decoder's latency and model calls for one render."""

    with strict_float32() as flags:
        result, latency, calls = _final_evaluation(
            model,
            parent,
            plain,
            template,
            device,
            run_dir,
            config,
            beta_at_floor=beta_at_floor,
            latency_repeats=latency_repeats,
        )
        result["float32_flags"] = flags
    return result, latency, calls


@torch.no_grad()
def _final_evaluation(
    model: VariationalTranscriber,
    parent: RenderToProgram,
    plain: dict[str, SplitData],
    template: Any,
    device: torch.device,
    run_dir: Path,
    config: CanvasConfig,
    *,
    beta_at_floor: bool,
    latency_repeats: int,
) -> tuple[dict[str, Any], LatencyReport, int]:
    model.eval()
    layout = model.layout
    validation = plain["primary/validation"].subset(config.eval_icons)
    count = len(validation.tokens)
    targets = [validation.targets[i].numpy() for i in range(count)]

    def score(tokens: Tensor) -> tuple[list[float], list[np.ndarray | None]]:
        renders = _programs_to_renders(tokens, layout, template)
        return [pixel_error(r, targets[i]) for i, r in enumerate(renders)], renders

    # Reconstruction from mu: the uint8 render goes through `encode`.
    errors, decoded, renders, _ = decode_and_score(
        model, validation, template, device, count, bf16=False
    )
    means, log_variances = posterior_statistics(model, validation.images, device)
    kl = gaussian_kl(means, log_variances)
    noise = torch.randn(means.shape, generator=torch.Generator().manual_seed(POSTERIOR_SAMPLE_SEED))
    sampled = means + noise.to(device) * (0.5 * log_variances).exp()
    sample_errors, sample_renders = score(decode_latents(model, sampled))
    zero = decode_latents(model, torch.zeros(1, *model.latent_shape, device=device))
    zero_render = _programs_to_renders(zero, layout, template)[0]
    zero_errors = [pixel_error(zero_render, target) for target in targets]
    parent_errors, _, parent_renders, _ = decode_and_score(
        parent, validation, template, device, count, bf16=False
    )
    # Every model trains with mirrors, so the baseline searches them too; the plain
    # search stays beside it, comparable with earlier run records.
    plain_library = plain["primary/train"].targets
    library = torch.cat((plain_library, plain_library.flip(2)))
    nearest, nearest_errors_tensor = nearest_training_icon(validation.targets, library, device)
    nearest_errors = [float(v) for v in nearest_errors_tensor]
    _, plain_errors = nearest_training_icon(validation.targets, plain_library, device)
    nearest_plain_errors = [float(v) for v in plain_errors]
    blank = np.full((EVAL_SIZE, EVAL_SIZE, 3), 255, dtype=np.uint8)
    blank_errors = [pixel_error(blank, target) for target in targets]

    vt_minus_parent = [e - p for e, p in zip(errors, parent_errors, strict=True)]
    reduction_vs_nearest = [n - e for e, n in zip(errors, nearest_errors, strict=True)]
    sample_cost = [s - e for e, s in zip(errors, sample_errors, strict=True)]
    carried = [z - e for e, z in zip(errors, zero_errors, strict=True)]

    # N(0, I) draws: the prior-hole control until a learned prior exists.
    prior = torch.randn(
        config.prior_samples,
        *model.latent_shape,
        generator=torch.Generator().manual_seed(PRIOR_SEED),
    ).to(device)
    prior_tokens = decode_latents(model, prior)
    prior_renders = _programs_to_renders(prior_tokens, layout, template)
    prior_ink = [ink_coverage(r) for r in prior_renders]
    validation_ink = [ink_coverage(t) for t in targets]
    columns = 8
    (run_dir / "prior-samples.png").write_bytes(
        contact_sheet(
            [prior_renders[i : i + columns] for i in range(0, len(prior_renders), columns)]
        )
    )

    # Lerp strips between posterior means: a canvas dissolve.
    rng = np.random.default_rng(PAIR_SEED)
    pairs = [
        tuple(int(v) for v in rng.choice(count, 2, replace=False))
        for _ in range(config.interpolation_pairs)
    ]
    strips: list[list[np.ndarray | None]] = []
    weights = torch.linspace(0.0, 1.0, config.interpolation_frames, device=device)
    weights = weights.view(-1, 1, 1, 1)
    for first, second in pairs:
        path = (1.0 - weights) * means[first] + weights * means[second]
        frames = _programs_to_renders(decode_latents(model, path), layout, template)
        strips.append([targets[first], *frames, targets[second]])
    (run_dir / "interpolations.png").write_bytes(contact_sheet(strips))

    sheet = [
        [
            targets[i],
            renders[i],
            sample_renders[i],
            parent_renders[i],
            library[int(nearest[i])].numpy(),
        ]
        for i in range(min(24, count))
    ]
    (run_dir / "reconstructions.png").write_bytes(contact_sheet(sheet))
    per_icon = [
        {
            "hexcode": validation.hexcodes[i],
            "vt_mu_pixel_error": errors[i],
            "vt_sample_pixel_error": sample_errors[i],
            "parent_pixel_error": parent_errors[i],
            "nearest_training_icon_pixel_error": nearest_errors[i],
            "nearest_training_icon_is_mirror": int(nearest[i]) >= len(plain_library),
            "nearest_training_icon_pixel_error_without_mirrors": nearest_plain_errors[i],
            "kl_nats": float(kl[i]),
            "exact_program": bool(torch.equal(decoded[i], validation.tokens[i])),
        }
        for i in range(count)
    ]
    (run_dir / "per_icon.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_icon))

    # Latency: one render end to end through the float32 graph decoder, as the demo would.
    image = validation.images[0]
    graph_tokens, graph_prior, latency, calls = _graph_latency(
        model, image, prior[0], device, latency_repeats
    )
    torch.cuda.empty_cache()

    settings = config.canvas
    budget = settings.budget_nats
    difference = bootstrap_mean_interval(vt_minus_parent)
    reduction = bootstrap_mean_interval(reduction_vs_nearest)
    cost = bootstrap_mean_interval(sample_cost)
    held_out_kl = bootstrap_mean_interval([float(v) for v in kl])
    criteria = {
        "A1": {
            "claim": (
                "reconstruction from mu is non-inferior to v9 (upper 95% bound of VT - v9 "
                f"<= +{config.non_inferiority_margin}) and beats the nearest training render "
                "or its mirror (paired interval on the reduction above zero)"
            ),
            "vt_minus_v9": difference,
            "reduction_vs_nearest_training_icon": reduction,
            "pass": difference[2] <= config.non_inferiority_margin and reduction[1] > 0.0,
        },
        "A2": {
            "claim": (
                f"held-out KL within [{settings.band_low * budget:.0f}, "
                f"{settings.band_high * budget:.0f}] nats with beta above its floor "
                "(detects collapse only; the controller enforces the band)"
            ),
            "held_out_kl_nats": held_out_kl,
            "beta_at_floor": beta_at_floor,
            "pass": settings.band_low * budget <= held_out_kl[0] <= settings.band_high * budget
            and not beta_at_floor,
        },
        "A3": {
            "claim": (
                "decoding one posterior sample instead of mu costs at most "
                f"{config.sample_cost_margin} pixel error (paired mean)"
            ),
            "sample_minus_mu": cost,
            "pass": cost[0] <= config.sample_cost_margin,
        },
    }
    result: dict[str, Any] = {
        "icons": count,
        "precision": precision_note(),
        "reconstruction_pixel_error": bootstrap_mean_interval(errors),
        "reconstruction_pixel_error_median": float(np.median(errors)),
        "posterior_sample_pixel_error": bootstrap_mean_interval(sample_errors),
        "parent_pixel_error": bootstrap_mean_interval(parent_errors),
        "nearest_training_icon_pixel_error": bootstrap_mean_interval(nearest_errors),
        "nearest_training_icon_pixel_error_without_mirrors": bootstrap_mean_interval(
            nearest_plain_errors
        ),
        "prior_mean_decode_pixel_error": bootstrap_mean_interval(zero_errors),
        "latent_minus_prior_mean_error_reduction": bootstrap_mean_interval(carried),
        "blank_pixel_error": float(np.mean(blank_errors)),
        "icons_vt_beats_nearest": sum(1 for d in reduction_vs_nearest if d > 0),
        "icons_vt_beats_v9": sum(1 for d in vt_minus_parent if d < 0),
        "exact_reconstruction_rate": float(np.mean([r["exact_program"] for r in per_icon])),
        "rendered_rate": sum(1 for r in renders if r is not None) / count,
        "held_out_kl_nats": held_out_kl,
        "held_out_kl_nats_per_dimension": held_out_kl[0] / math.prod(model.latent_shape),
        "prior_samples": {
            "count": len(prior_tokens),
            "distinct": len({tuple(row.tolist()) for row in prior_tokens}),
            "rendered": sum(1 for r in prior_renders if r is not None),
            "ink_coverage_median": float(np.median(prior_ink)),
            "fragment_rate_ink_below_0.10": float(np.mean([v < 0.10 for v in prior_ink])),
            "near_blank_rate_ink_below_0.02": float(np.mean([v < 0.02 for v in prior_ink])),
            "validation_fragment_rate_ink_below_0.10": float(
                np.mean([v < 0.10 for v in validation_ink])
            ),
            "role": "N(0, I) prior-hole control, not a generator claim",
        },
        "interpolation": {"pairs": [list(p) for p in pairs], "frames": config.interpolation_frames},
        "graph_decoder_agrees_with_greedy": {
            "render": bool(torch.equal(graph_tokens[0], decoded[0])),
            "prior_latent": bool(torch.equal(graph_prior[0], prior_tokens[0])),
        },
        "criteria": criteria,
    }
    return result, latency, calls


def _graph_latency(
    model: VariationalTranscriber,
    image: Tensor,
    latent: Tensor,
    device: torch.device,
    repeats: int,
) -> tuple[Tensor, Tensor, LatencyReport, int]:
    """Float32 graph decodes of one render and one latent, and the render's latency."""

    from mojidiff.learning.fast_decode import GraphDecoder

    graph = GraphDecoder(model, dtype=torch.float32)
    stats = DecodeStats()
    from_render = graph.decode(image, stats=stats)
    from_latent = graph.decode(latent)
    latency = measure_latency(lambda: graph.decode(image), device=device, warmup=2, repeats=repeats)
    return from_render, from_latent, latency, stats.model_calls


# --------------------------------------------------------------------------- preflight


def preflight(
    config_path: Path,
    *,
    icons: int = PREFLIGHT_ICONS,
    ranks: Sequence[int] = (8, 16),
    pca_icons: int | None = None,
    batch_size: int = 64,
) -> dict[str, Any]:
    """No training: PCA of v9's stem, and v9 decoding with its stem projected to rank c,
    all in IEEE float32 (TF32 off).

    The declared rule (`preflight_choice`): if the rank-8 pixel error on the first 64
    validation icons exceeds 0.12, the run uses c = 16; if rank 16 exceeds it too, no
    outcome was declared and `chosen_channels` is None, so no run can start from it.
    """

    with strict_float32() as flags:
        report = _preflight(
            config_path, icons=icons, ranks=ranks, pca_icons=pca_icons, batch_size=batch_size
        )
    report["float32_flags"] = flags
    errors = {
        int(key.removeprefix("rank_")): float(value["pixel_error"][0])
        for key, value in report.items()
        if key.startswith("rank_")
    }
    report["rule"] = (
        f"rank-8 error <= {PREFLIGHT_THRESHOLD}: c = 8; else rank-16 error <= "
        f"{PREFLIGHT_THRESHOLD}: c = 16; else no declared outcome (the operator decides)"
    )
    report["chosen_channels"], report["rule_outcome"] = preflight_choice(errors)
    if icons != PREFLIGHT_ICONS:
        report["chosen_channels"] = None
        report["rule_outcome"] = f"{icons} icons, not the declared {PREFLIGHT_ICONS}: no outcome"
    return report


@torch.no_grad()
def _preflight(
    config_path: Path,
    *,
    icons: int,
    ranks: Sequence[int],
    pca_icons: int | None,
    batch_size: int,
) -> dict[str, Any]:
    config = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plain, _, layout, pilot, dataset_hash = load_corpus(
        config.pilot_config, config.model.image_size
    )
    from mojidiff.learning.openmoji_pilot import _load_program

    template = _load_program(_pilot_rows(pilot)["primary/train"][0], pilot, layout.codec)
    parent_state, parent = _load_parent(config, layout)
    parent = parent.to(device)
    train_images = plain["primary/train"].images
    limit = pca_icons if pca_icons is not None else config.canvas.pca_icons
    pca = stem_pca(parent, train_images[:limit] if limit else train_images, device)
    validation = plain["primary/validation"].subset(icons)
    started = time.perf_counter()
    parent_errors, _, _, _ = decode_and_score(
        parent, validation, template, device, icons, batch_size=batch_size, bf16=False
    )
    parent.cpu()
    report: dict[str, Any] = {
        "config": str(config_path),
        "git_commit": _git("rev-parse", "HEAD").strip(),
        "uncommitted_sources": uncommitted_sources(),
        "parent_checkpoint": str(config.parent_checkpoint),
        "parent_checkpoint_sha256": _sha256(config.parent_checkpoint),
        "dataset_sha256": dataset_hash,
        "icons": icons,
        "precision": precision_note(),
        "pca": {
            "icons": pca.icons,
            "cells": pca.cells,
            "explained_variance": {
                str(rank): pca.explained(rank) for rank in (1, 2, 4, 8, 16, 32, 64)
            },
        },
        "v9_pixel_error": bootstrap_mean_interval(parent_errors),
    }
    for rank in ranks:
        settings = replace(config.canvas, channels=rank)
        model = from_transcriber(layout, parent_state, settings, pca).to(device).eval()
        errors, _, _, _ = decode_and_score(
            model, validation, template, device, icons, batch_size=batch_size, bf16=False
        )
        means, log_variances = posterior_statistics(model, validation.images, device)
        report[f"rank_{rank}"] = {
            "explained_variance": pca.explained(rank),
            "pixel_error": bootstrap_mean_interval(errors),
            "minus_v9": bootstrap_mean_interval(
                [e - p for e, p in zip(errors, parent_errors, strict=True)]
            ),
            "initial_kl_nats_per_icon": float(gaussian_kl(means, log_variances).mean()),
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report["seconds"] = time.perf_counter() - started
    return report


# --------------------------------------------------------------------------- report


def _interval(value: Sequence[float]) -> str:
    return f"{value[0]:.4f} [{value[1]:.4f}, {value[2]:.4f}]"


def result_markdown(record: dict[str, Any]) -> str:
    result = record["result"]
    resource = record["resource"]
    prior = result["prior_samples"]
    criteria = result["criteria"]
    selection = result["selection"]
    lines = [
        f"# {record['run_id']}",
        "",
        f"**Hypothesis.** {record['hypothesis']}",
        "",
        "## Criteria",
        "",
        "| criterion | measure | pass |",
        "| --- | --- | --- |",
        f"| A1 | VT - v9 {_interval(criteria['A1']['vt_minus_v9'])}; reduction vs nearest "
        f"{_interval(criteria['A1']['reduction_vs_nearest_training_icon'])} | "
        f"{criteria['A1']['pass']} |",
        f"| A2 | held-out KL {_interval(criteria['A2']['held_out_kl_nats'])} nats; beta at "
        f"floor {criteria['A2']['beta_at_floor']} | {criteria['A2']['pass']} |",
        f"| A3 | sample - mu {_interval(criteria['A3']['sample_minus_mu'])} | "
        f"{criteria['A3']['pass']} |",
        "",
        f"## Result ({result['icons']} validation icons, {result['precision']})",
        "",
        "| measure | value |",
        "| --- | --- |",
        f"| VT from mu, pixel error | {_interval(result['reconstruction_pixel_error'])} |",
        f"| VT from one posterior sample | {_interval(result['posterior_sample_pixel_error'])} |",
        f"| v9 (parent), same icons, float32 | {_interval(result['parent_pixel_error'])} |",
        f"| nearest training render or mirror | "
        f"{_interval(result['nearest_training_icon_pixel_error'])} |",
        f"| nearest training render, no mirrors | "
        f"{_interval(result['nearest_training_icon_pixel_error_without_mirrors'])} |",
        f"| z = 0 decode (control) | {_interval(result['prior_mean_decode_pixel_error'])} |",
        f"| blank canvas | {result['blank_pixel_error']:.4f} |",
        f"| icons VT beats nearest / beats v9 | {result['icons_vt_beats_nearest']} / "
        f"{result['icons_vt_beats_v9']} |",
        f"| exact reconstructions | {result['exact_reconstruction_rate']:.3f} |",
        f"| held-out KL per dimension | {result['held_out_kl_nats_per_dimension']:.3f} |",
        f"| N(0, I) samples distinct / rendered | {prior['distinct']} / {prior['rendered']} "
        f"of {prior['count']} |",
        f"| N(0, I) samples with ink < 0.10 | {prior['fragment_rate_ink_below_0.10']:.3f} "
        f"(validation icons {prior['validation_fragment_rate_ink_below_0.10']:.3f}) |",
        f"| selected step | {result['selected_step']} "
        f"(fallback: {selection['fallback_no_eval_in_band']}) |",
        "",
        "## Resources",
        "",
        "| measure | value |",
        "| --- | --- |",
        f"| device | {resource['device']} |",
        f"| parameters | {record['model_parameters']:,} |",
        f"| train seconds | {resource['train_seconds']:.0f} |",
        f"| peak VRAM GiB | {resource['peak_vram_gib']} |",
        f"| graph decode ms per icon, batch 1, IEEE float32 | "
        f"{resource['inference_ms_per_icon']:.1f} |",
        f"| torch / CUDA | {resource['torch_version']} / {resource['cuda_version']} |",
        "",
        "Sheets: `reconstructions.png` (reference, VT from mu, VT from a posterior sample,",
        "v9, nearest training render or mirror), `interpolations.png` (lerp of posterior",
        "means, a canvas dissolve), `prior-samples.png` (N(0, I), the prior-hole control).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--preflight", action="store_true", help="PCA and rank-c decodes of v9; no training"
    )
    parser.add_argument("--preflight-icons", type=int, default=PREFLIGHT_ICONS)
    parser.add_argument("--pca-icons", type=int, help="preflight only: PCA over the first N")
    parser.add_argument(
        "--out",
        type=Path,
        help="preflight only: write the report here (default: the config's preflight_report); "
        "an existing report is never overwritten",
    )
    parser.add_argument(
        "--smoke",
        type=Path,
        help="a shrunk mechanics run written under this directory, with no registry row",
    )
    parser.add_argument("--steps", type=int, help="smoke only (default 20)")
    parser.add_argument("--batch-size", type=int, help="smoke only (default 4)")
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    if args.preflight:
        out = args.out if args.out is not None else load_config(args.config).preflight_report
        if out is not None and out.exists():
            parser.error(f"{out} exists; preserve it and write elsewhere with --out")
        report = preflight(args.config, icons=args.preflight_icons, pca_icons=args.pca_icons)
        text = json.dumps(report, indent=2)
        if out is not None:
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text + "\n")
        print(text)
        return 0
    if args.smoke is None and (args.steps is not None or args.batch_size is not None):
        parser.error("--steps and --batch-size change a run's identity; use them with --smoke")
    smoke = args.smoke.resolve() if args.smoke is not None else None
    train_and_evaluate(
        args.config,
        smoke=smoke,
        steps=args.steps if args.steps is not None else 20,
        batch_size=args.batch_size if args.batch_size is not None else 4,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
