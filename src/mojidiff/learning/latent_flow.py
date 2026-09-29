"""The canvas latent, stage B: a rectified-flow prior ("LFP") over the frozen VT's grid.

Stage A (`pixel_latent`) gave a variational transcriber (VT): q(z | render) is a
(c, 18, 18) Gaussian grid, and its decoder writes a program from any such grid. Its own
prior, N(0, I), has holes - most N(0, I) grids look like no posterior. This module fits
the prior to the aggregate posterior instead, as a two-stage VAE or latent diffusion
does, and leaves the VT frozen.

* **Latents.** A frozen VT checkpoint encodes K exact variants of every training icon
  (default 32: the icon itself, then variants drawn by `render2svg`'s exact
  augmentation - mirror 0.5, translation up to 12 view units, palette permutation 0.1
  - rendered by its cached-variant worker `_augment_chunk`; no compositions, because the
  target is the emoji distribution) and the validation icons, in IEEE float32. The
  posterior means and sigmas are cached in float16 under
  `CACHE_ROOT/latent-flow/<vt run id>-<settings hash>.pt`; the hash covers the VT
  checkpoint, the corpus and every setting. Test icons are never encoded.
* **Standardisation.** Every channel is standardised by the aggregate posterior's mean
  and standard deviation over the training latents (variance = var(mu) + E[sigma^2]);
  the flow lives in that space, x = (z - m_c) / s_c, and the statistics travel with
  every checkpoint.
* **Network.** A DiT: the (c, 18, 18) grid is cut into 2 x 2 patches (81 tokens of 4c
  numbers), embedded to d = 256 with fixed 2-D sin-cos positions, 8 blocks with
  adaLN-zero conditioning on t (sinusoidal features through an MLP), and an adaLN output
  layer initialised to zero, so the untrained network predicts v = 0.
* **Rectified flow.** t in [0, 1], t = 0 is data and t = 1 is noise:

      x_t = (1 - t) x_0 + t eps,   eps ~ N(0, I),   v = dx_t / dt = eps - x_0,

  and the network v_theta(x_t, t) regresses v by mean squared error, with t
  logit-normal (sigmoid of N(m, s^2)). Each step draws z ~ q(z | x) from a cached mean
  and sigma. AdamW, bf16 autocast, gradient clipping and an EMA of the weights (0.999),
  which is what is evaluated, selected and served. The checkpoint is selected by the
  EMA's flow loss on the validation icons' latents under fixed draws of t and eps: the
  guard against memorising 2,681 training icons.
* **Sampling.** Euler steps from x_1 = scale * eps at t = 1 down to t = 0, each
  x_{t - dt} = x_t - dt v_theta(x_t, t). At any t the predicted clean latent is
  x_hat_0 = x_t - t v_theta(x_t, t); un-standardised it is z_hat_0, which the VT decodes
  greedily under the grammar in IEEE float32. "Watch it draw" decodes z_hat_0 at 8 times
  of one trajectory (t = 1 down to 2 / steps; at the last step z_hat_0 is the sample),
  then the sample. Inversion integrates the same ODE from t = 0 to 1, taking a
  posterior mean to its noise; "slerp interpolation" inverts two means, slerps between
  their noises and integrates every frame forward, so each frame is a prior sample.
  "Lerp" is the straight line between the two posterior means, for comparison. The
  metrics harness scores the slerp only when run with `--interpolation backend`.

Naming. x_t is the raw noised (standardised) latent and x_hat_0 the network's predicted
clean latent; z_hat_0 is x_hat_0 un-standardised, and what the VT decodes (constrained
greedy decoding). This is continuous latent flow plus an autoregressive program decoder:
not categorical diffusion over programs, and not a D3PM.

Precision. Latents are encoded, and samples integrated and decoded, in IEEE float32
(`precision.strict_float32`, TF32 off); training runs under bf16 autocast with the
environment's TF32; selection-time validation losses run in float32 with it.

Provenance. A real run refuses to start while anything under src/, configs/ or tests/ is
untracked or uncommitted, and unless the parent VT run's record is completed, names the
config's parent checkpoint and passed every criterion the config's `parent_gate` lists.
`--smoke DIR` shrinks everything and writes only under DIR, the latents included, with
no registry row.

    python -m mojidiff.learning.latent_flow --config configs/latent/lfp-v1.yaml
    python -m mojidiff.learning.latent_flow --config ... --smoke DIR [--steps 50
        --batch-size 8 --layers 2 --variants 2 --train-icons 32 --parent-checkpoint VT.pt]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from mojidiff.learning.autoregressive import SequenceLayout
from mojidiff.learning.pixel_latent import (
    PAIR_SEED,
    PRIOR_SEED,
    VariationalTranscriber,
    _sha256,
    decode_latents,
    ink_coverage,
    load_checkpoint,
    precision_note,
)
from mojidiff.learning.precision import strict_float32
from mojidiff.learning.render2svg import (
    CACHE_ROOT,
    REPO_ROOT,
    SplitData,
    TrainConfig,
    _append_registry,
    _augment_chunk,
    _now,
    _pilot_rows,
    _programs_to_renders,
    _write_yaml,
    bootstrap_mean_interval,
    contact_sheet,
    load_corpus,
    parameter_count,
    pixel_error,
    run_identity,
    uncommitted_sources,
)
from mojidiff.learning.telemetry import (
    LatencyReport,
    measure_latency,
    reset_peak_memory,
    resource_summary,
)
from mojidiff.representation.packed import PackedTensorProgram

LATENT_FORMAT = 1
"""Bump when the precompute changes what a cache file holds for the same settings."""
CHECKPOINT_KIND = "latent-flow"
TRAJECTORY_SEED = 31
VALIDATION_DRAW_SEED = 41
POSTERIOR_DRAW_SEED = 43
FRAGMENT_INK = 0.10
"""The in-run fragment rule, as `pixel_latent` reports it (ink coverage under 0.10). The
harness (`latent_metrics`) calibrates its own threshold on the validation icons."""
NEAR_BLANK_INK = 0.02

Velocity = Callable[[Tensor, Tensor], Tensor]
"""(x_t, t) -> v: a batch of standardised latents and their times, (B,), to velocities."""


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class PrecomputeSettings:
    variants: int = 32
    """K: latents per training icon."""
    originals: bool = True
    """The first of an icon's K latents is the icon itself, unmodified; the rest are
    exact variants. False draws all K as variants."""
    mirror: float = 0.5
    max_shift: int = 48
    """Largest translation per axis, in quarter units (48 = 12 view units)."""
    colour: float = 0.1
    """Per-variant probability of a palette permutation."""
    seed: int = 9101
    chunk_icons: int = 8
    """Icons per rendering job; each job's generator is seeded from (seed, job)."""
    train_icons: int | None = None
    """The first N training icons (smoke runs); None encodes all."""
    validation_icons: int = 339
    encode_batch: int = 64
    workers: int = 16
    """Rendering processes; not part of the settings hash (it changes no result)."""


@dataclass(frozen=True)
class FlowNetworkConfig:
    channels: int = 8
    grid: int = 18
    """Latent shape (channels, grid, grid): always the parent VT's, never the config's."""
    patch: int = 2
    d_model: int = 256
    layers: int = 8
    heads: int = 8
    mlp_ratio: float = 4.0
    frequency_dim: int = 256
    """Sinusoidal features of t before the time MLP."""


@dataclass(frozen=True)
class FlowTrainConfig:
    steps: int = 40_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 0.0
    warmup_steps: int = 1000
    schedule: str = "constant"
    """"constant" after warmup (with the EMA doing the averaging), or "cosine"."""
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    time_mean: float = 0.0
    time_std: float = 1.0
    """t = sigmoid(time_mean + time_std * n), n ~ N(0, 1)."""
    posterior_sampling: bool = True
    """Train on z = mu + sigma * eps drawn afresh each step; False trains on the means."""
    seed: int = 7101
    eval_every: int = 1000
    validation_draws: int = 16
    """Fixed (t, eps) draws per validation latent in the validation flow loss."""
    trace_every: int = 100
    bf16: bool = True


@dataclass(frozen=True)
class SamplingConfig:
    steps: int = 50
    """Euler steps from noise to z_hat_0, and for inversion."""
    trajectory_frames: int = 8
    """z_hat_0 frames decoded along one trajectory ("watch it draw")."""


@dataclass(frozen=True)
class FlowConfig:
    slug: str
    hypothesis: str
    pilot_config: Path
    parent_run: str
    parent_checkpoint: Path
    parent_gate: tuple[str, ...] = ("A1", "A3")
    """Criteria the parent VT run must have passed before a real run starts."""
    latents: PrecomputeSettings = field(default_factory=PrecomputeSettings)
    network: FlowNetworkConfig = field(default_factory=FlowNetworkConfig)
    training: FlowTrainConfig = field(default_factory=FlowTrainConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    eval_icons: int = 339
    """Validation icons the interpolation pairs are drawn from (as `pixel_latent`)."""
    prior_samples: int = 64
    interpolation_pairs: int = 8
    interpolation_frames: int = 9
    trajectories: int = 4
    latency_repeats: int = 10
    notes: str = ""


_KNOWN = {
    "schema_version", "slug", "hypothesis", "pilot_config", "parent_run",
    "parent_checkpoint", "parent_gate", "latents", "network", "training", "sampling",
    "eval_icons", "prior_samples", "interpolation_pairs", "interpolation_frames",
    "trajectories", "latency_repeats", "notes",
}  # fmt: skip


def load_config(path: Path) -> FlowConfig:
    root = yaml.safe_load(path.read_text())
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise ValueError("latent flow config needs schema_version: 1")
    unknown = sorted(set(root) - _KNOWN)
    if unknown:
        raise ValueError(f"unknown latent flow config fields: {', '.join(unknown)}")
    network = dict(root.get("network") or {})
    if {"channels", "grid"} & set(network):
        raise ValueError("network.channels and network.grid come from the parent VT")
    config = FlowConfig(
        slug=str(root["slug"]),
        hypothesis=str(root["hypothesis"]).strip(),
        pilot_config=Path(str(root["pilot_config"])),
        parent_run=str(root["parent_run"]),
        parent_checkpoint=Path(str(root["parent_checkpoint"])),
        parent_gate=tuple(str(name) for name in root.get("parent_gate", ("A1", "A3"))),
        latents=PrecomputeSettings(**(root.get("latents") or {})),
        network=FlowNetworkConfig(**network),
        training=FlowTrainConfig(**(root.get("training") or {})),
        sampling=SamplingConfig(**(root.get("sampling") or {})),
        eval_icons=int(root.get("eval_icons", 339)),
        prior_samples=int(root.get("prior_samples", 64)),
        interpolation_pairs=int(root.get("interpolation_pairs", 8)),
        interpolation_frames=int(root.get("interpolation_frames", 9)),
        trajectories=int(root.get("trajectories", 4)),
        latency_repeats=int(root.get("latency_repeats", 10)),
        notes=str(root.get("notes", "")).strip(),
    )
    check_config(config)
    return config


def check_config(config: FlowConfig) -> None:
    """Refuse settings the code cannot honour, before anything runs."""

    latents, network, training = config.latents, config.network, config.training
    problems = []
    if latents.variants < 1:
        problems.append("latents.variants must be at least 1")
    if not (0.0 <= latents.mirror <= 1.0 and 0.0 <= latents.colour <= 1.0):
        problems.append("latents.mirror and latents.colour are probabilities")
    if network.d_model % 4 or network.d_model % network.heads:
        problems.append("network.d_model must divide by 4 (2-D sin-cos) and by heads")
    if network.frequency_dim % 2:
        problems.append("network.frequency_dim must be even")
    if training.schedule not in ("constant", "cosine"):
        problems.append(f"training.schedule {training.schedule!r} is not constant or cosine")
    if not 0.0 < training.ema_decay < 1.0:
        problems.append("training.ema_decay must lie in (0, 1)")
    if config.interpolation_frames < 3 or config.sampling.steps < 1:
        problems.append("interpolation_frames >= 3 and sampling.steps >= 1")
    if config.sampling.trajectory_frames > config.sampling.steps:
        problems.append("sampling.trajectory_frames cannot exceed sampling.steps")
    if problems:
        raise ValueError("; ".join(problems))


# --------------------------------------------------------------------------- network


def patchify(x: Tensor, patch: int) -> Tensor:
    """(B, c, H, W) -> (B, H/p * W/p, c p p), row-major over patches."""

    batch, channels, height, width = x.shape
    grid = x.reshape(batch, channels, height // patch, patch, width // patch, patch)
    return grid.permute(0, 2, 4, 1, 3, 5).reshape(
        batch, (height // patch) * (width // patch), channels * patch * patch
    )


def unpatchify(tokens: Tensor, channels: int, patch: int, side: int) -> Tensor:
    """The inverse of `patchify` for a square grid of `side` x `side` patches."""

    grid = tokens.reshape(len(tokens), side, side, channels, patch, patch)
    return grid.permute(0, 3, 1, 4, 2, 5).reshape(len(tokens), channels, side * patch, side * patch)


def sincos_2d(width: int, side: int) -> Tensor:
    """Fixed 2-D sin-cos positions, (side * side, width): a quarter of the width each
    for sin and cos of the row and of the column, row-major as `patchify` orders."""

    quarter = width // 4
    omega = 1.0 / 10_000.0 ** (torch.arange(quarter, dtype=torch.float64) / quarter)
    rows, columns = torch.meshgrid(
        torch.arange(side, dtype=torch.float64),
        torch.arange(side, dtype=torch.float64),
        indexing="ij",
    )
    row_angles = rows.reshape(-1, 1) * omega[None]
    column_angles = columns.reshape(-1, 1) * omega[None]
    return torch.cat(
        [row_angles.sin(), row_angles.cos(), column_angles.sin(), column_angles.cos()], dim=1
    ).float()


def timestep_features(t: Tensor, dim: int, max_period: float = 10_000.0) -> Tensor:
    """Sinusoidal features of 1000 t, (B, dim), in float32 (as DiT embeds its steps)."""

    half = dim // 2
    frequencies = torch.exp(
        -math.log(max_period) * torch.arange(half, device=t.device, dtype=torch.float32) / half
    )
    angles = 1000.0 * t.float()[:, None] * frequencies[None]
    return torch.cat([angles.cos(), angles.sin()], dim=-1)


class _Block(nn.Module):
    """Self-attention and MLP, each under a shift, scale and gate from t (adaLN-zero)."""

    def __init__(self, width: int, heads: int, hidden: int) -> None:
        super().__init__()
        self.heads = heads
        self.attention_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.mlp_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(width, hidden), nn.GELU(approximate="tanh"), nn.Linear(hidden, width)
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))

    def forward(self, x: Tensor, condition: Tensor) -> Tensor:
        modulation = cast(Tensor, self.modulation(condition))[:, None]
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = modulation.chunk(6, dim=-1)
        batch, length, width = x.shape
        h = self.attention_norm(x) * (1.0 + scale_a) + shift_a
        qkv = cast(Tensor, self.qkv(h)).view(batch, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4)
        attended = F.scaled_dot_product_attention(query, key, value)
        x = x + gate_a * self.attention_out(attended.transpose(1, 2).reshape(batch, length, width))
        h = self.mlp_norm(x) * (1.0 + scale_m) + shift_m
        return x + gate_m * cast(Tensor, self.mlp(h))


class FlowTransformer(nn.Module):
    """A DiT velocity network over (c, grid, grid) latents: v_theta(x_t, t)."""

    def __init__(self, config: FlowNetworkConfig) -> None:
        super().__init__()
        if config.grid % config.patch:
            raise ValueError(f"grid {config.grid} does not divide into {config.patch} patches")
        if config.d_model % 4 or config.d_model % config.heads:
            raise ValueError("d_model must divide by 4 and by heads")
        self.config = config
        self.side = config.grid // config.patch
        width = config.d_model
        patch_dim = config.channels * config.patch**2
        self.embed = nn.Linear(patch_dim, width)
        self.register_buffer("positions", sincos_2d(width, self.side), persistent=False)
        self.time = nn.Sequential(
            nn.Linear(config.frequency_dim, width), nn.SiLU(), nn.Linear(width, width)
        )
        hidden = int(round(width * config.mlp_ratio))
        self.blocks = nn.ModuleList(
            [_Block(width, config.heads, hidden) for _ in range(config.layers)]
        )
        self.final_norm = nn.LayerNorm(width, elementwise_affine=False, eps=1e-6)
        self.final_modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 2 * width))
        self.head = nn.Linear(width, patch_dim)
        self._initialise()

    def _initialise(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        for index in (0, 2):
            nn.init.normal_(cast(nn.Linear, self.time[index]).weight, std=0.02)
        zeroed = [cast(nn.Linear, cast(_Block, b).modulation[1]) for b in self.blocks]
        zeroed += [cast(nn.Linear, self.final_modulation[1]), self.head]
        for layer in zeroed:
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return (self.config.channels, self.config.grid, self.config.grid)

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        config = self.config
        tokens = self.embed(patchify(x, config.patch)) + cast(Tensor, self.positions)
        condition = cast(Tensor, self.time(timestep_features(t, config.frequency_dim)))
        for block in self.blocks:
            tokens = block(tokens, condition)
        shift, scale = cast(Tensor, self.final_modulation(condition))[:, None].chunk(2, dim=-1)
        out = self.head(self.final_norm(tokens) * (1.0 + scale) + shift)
        return unpatchify(out, config.channels, config.patch, self.side)


# --------------------------------------------------------------------------- flow


def logit_normal_times(
    count: int,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    generator: torch.Generator | None = None,
    device: torch.device | None = None,
) -> Tensor:
    """t = sigmoid(mean + std * n), n ~ N(0, 1): (count,) times in (0, 1)."""

    return torch.sigmoid(mean + std * torch.randn(count, generator=generator, device=device))


def _per_item(t: Tensor, like: Tensor) -> Tensor:
    return t.to(like.dtype).view(-1, *([1] * (like.dim() - 1)))


def noised(x0: Tensor, noise: Tensor, t: Tensor) -> Tensor:
    """x_t = (1 - t) x_0 + t eps, per item."""

    view = _per_item(t, x0)
    return (1.0 - view) * x0 + view * noise


def flow_matching_loss(velocity: Velocity, x0: Tensor, t: Tensor, noise: Tensor) -> Tensor:
    """Mean squared error of v_theta(x_t, t) against eps - x_0, over every number."""

    prediction = velocity(noised(x0, noise, t), t)
    return F.mse_loss(prediction.float(), (noise - x0).float())


def gaussian_velocity(x_t: Tensor, t: Tensor) -> Tensor:
    """The exact velocity E[eps - x_0 | x_t] if x_0 were N(0, I): (2t - 1) x_t / ((1 - t)^2
    + t^2). The zero-parameter baseline for the validation flow loss."""

    view = _per_item(t, x_t)
    return (2.0 * view - 1.0) * x_t / ((1.0 - view) ** 2 + view**2)


@torch.no_grad()
def euler_generate(
    velocity: Velocity, noise: Tensor, steps: int, record_at: Iterable[int] = ()
) -> tuple[Tensor, list[tuple[float, Tensor]]]:
    """Integrate from x_1 = `noise` at t = 1 to t = 0 in `steps` uniform Euler steps.

    Returns x_0 and, for every step index in `record_at` (0 is t = 1), the time and the
    predicted clean latent there, x_hat_0 = x_t - t v(x_t, t)."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    times = torch.linspace(1.0, 0.0, steps + 1, dtype=torch.float64).tolist()
    wanted = set(record_at)
    x = noise
    frames: list[tuple[float, Tensor]] = []
    for index in range(steps):
        now, following = times[index], times[index + 1]
        v = velocity(x, torch.full((len(x),), now, device=x.device, dtype=x.dtype))
        if index in wanted:
            frames.append((now, x - now * v))
        x = x + (following - now) * v
    return x, frames


@torch.no_grad()
def euler_invert(velocity: Velocity, x0: Tensor, steps: int) -> Tensor:
    """Integrate the same ODE from t = 0 to t = 1: a clean latent to its noise."""

    if steps < 1:
        raise ValueError("steps must be at least 1")
    times = torch.linspace(0.0, 1.0, steps + 1, dtype=torch.float64).tolist()
    x = x0
    for index in range(steps):
        now, following = times[index], times[index + 1]
        v = velocity(x, torch.full((len(x),), now, device=x.device, dtype=x.dtype))
        x = x + (following - now) * v
    return x


def slerp(first: Tensor, second: Tensor, weights: Tensor) -> Tensor:
    """Spherical interpolation between two tensors of one shape, one frame per weight:
    (len(weights), *shape). Weight 0 is `first`, 1 is `second`; nearly parallel inputs
    fall back to the straight line."""

    a, b = first.reshape(-1).double(), second.reshape(-1).double()
    w = weights.to(a.device, torch.float64).view(-1, 1)
    cosine = (a @ b / (a.norm() * b.norm()).clamp_min(1e-12)).clamp(-1.0, 1.0)
    omega = torch.arccos(cosine)
    if float(torch.sin(omega)) < 1e-6:
        frames = (1.0 - w) * a + w * b
    else:
        frames = (torch.sin((1.0 - w) * omega) * a + torch.sin(w * omega) * b) / torch.sin(omega)
    return frames.to(first.dtype).reshape(len(w), *first.shape)


def trajectory_indices(steps: int, frames: int) -> list[int]:
    """`frames` Euler step indices spread over one trajectory, the first at t = 1 and the
    last at t = 2 / steps. Not the final step: at index steps - 1 (t = 1 / steps),
    x_hat_0 = x_t - t v is exactly the last Euler update, so that frame would be the
    sample itself, which "watch it draw" shows after the recorded frames."""

    return sorted({int(round(v)) for v in np.linspace(0, max(steps - 2, 0), frames)})


class LatentFlowPrior(nn.Module):
    """A velocity network over standardised latents, with the maps to and from the VT's
    latents z. Every method takes and returns z (un-standardised) unless it says so."""

    def __init__(
        self, network: FlowTransformer, mean: Tensor, std: Tensor, steps: int = 50
    ) -> None:
        super().__init__()
        self.network = network
        self.steps = steps
        self.register_buffer("channel_mean", mean.detach().float().clone())
        self.register_buffer("channel_std", std.detach().float().clone())

    @property
    def latent_shape(self) -> tuple[int, int, int]:
        return self.network.latent_shape

    def _view(self, name: str, like: Tensor) -> Tensor:
        value = cast(Tensor, getattr(self, name))
        return value.to(like.device, like.dtype).view(1, -1, *([1] * (like.dim() - 2)))

    def standardise(self, z: Tensor) -> Tensor:
        return (z - self._view("channel_mean", z)) / self._view("channel_std", z)

    def unstandardise(self, x: Tensor) -> Tensor:
        return x * self._view("channel_std", x) + self._view("channel_mean", x)

    def velocity(self, x: Tensor, t: Tensor) -> Tensor:
        """v_theta(x_t, t) in float32, on standardised latents."""

        return cast(Tensor, self.network(x, t)).float()

    @torch.no_grad()
    def generate(
        self, noise: Tensor, steps: int | None = None, record_at: Iterable[int] = ()
    ) -> tuple[Tensor, list[tuple[float, Tensor]]]:
        """z_0 from standardised noise (scale it first for a temperature), and z_hat_0 at
        the recorded step indices."""

        self.network.eval()
        x, frames = euler_generate(self.velocity, noise.float(), steps or self.steps, record_at)
        return self.unstandardise(x), [(t, self.unstandardise(f)) for t, f in frames]

    @torch.no_grad()
    def invert(self, z: Tensor, steps: int | None = None) -> Tensor:
        """The standardised noise the ODE maps to `z` (reverse-time Euler)."""

        self.network.eval()
        return euler_invert(self.velocity, self.standardise(z.float()), steps or self.steps)

    @torch.no_grad()
    def slerp_path(
        self, first: Tensor, second: Tensor, frames: int, steps: int | None = None
    ) -> Tensor:
        """Invert both latents, slerp between their noises, integrate each frame forward:
        (frames, *latent_shape)."""

        noises = self.invert(torch.stack([first, second]), steps)
        weights = torch.linspace(0.0, 1.0, frames, device=noises.device)
        path, _ = self.generate(slerp(noises[0], noises[1], weights), steps)
        return path


# --------------------------------------------------------------------------- training


def flow_schedule(step: int, config: FlowTrainConfig) -> float:
    """Learning-rate multiplier: linear warmup, then constant or cosine to zero."""

    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    if config.schedule == "constant":
        return 1.0
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


class FlowTrainer:
    """One rectified-flow step at a time: logit-normal t, bf16 autocast on CUDA, AdamW,
    gradient clipping, the schedule, and an EMA of the weights."""

    def __init__(self, network: FlowTransformer, config: FlowTrainConfig) -> None:
        self.network = network
        self.config = config
        self.ema = copy.deepcopy(network).eval().requires_grad_(False)
        self.optimizer = torch.optim.AdamW(
            network.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda step: flow_schedule(step, config)
        )
        self.steps = 0

    def step(self, x0: Tensor, generator: torch.Generator | None = None) -> tuple[Tensor, Tensor]:
        """Train on one batch of clean standardised latents; (loss, gradient norm)."""

        config = self.config
        self.network.train()
        t = logit_normal_times(
            len(x0),
            mean=config.time_mean,
            std=config.time_std,
            generator=generator,
            device=x0.device,
        )
        noise = torch.randn(x0.shape, generator=generator, device=x0.device, dtype=x0.dtype)
        bf16 = config.bf16 and x0.device.type == "cuda"
        with torch.autocast(x0.device.type, dtype=torch.bfloat16, enabled=bf16):
            loss = flow_matching_loss(self.network, x0, t, noise)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()  # type: ignore[no-untyped-call]
        norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), config.grad_clip)
        self.optimizer.step()
        self.scheduler.step()
        self.update_ema()
        self.steps += 1
        return loss.detach(), norm.detach()

    @torch.no_grad()
    def update_ema(self) -> None:
        weight = 1.0 - self.config.ema_decay
        for average, current in zip(self.ema.parameters(), self.network.parameters(), strict=True):
            average.lerp_(current.detach(), weight)


@dataclass
class FlowValidation:
    """Fixed (t, eps) draws over held-out clean latents: a deterministic flow loss."""

    x_t: Tensor
    t: Tensor
    target: Tensor

    @classmethod
    def draw(
        cls, x0: Tensor, draws: int, seed: int, *, time_mean: float, time_std: float
    ) -> FlowValidation:
        generator = torch.Generator().manual_seed(seed)
        clean = x0.float().cpu().repeat(draws, *([1] * (x0.dim() - 1)))
        t = logit_normal_times(len(clean), mean=time_mean, std=time_std, generator=generator)
        noise = torch.randn(clean.shape, generator=generator)
        device = x0.device
        return cls(noised(clean, noise, t).to(device), t.to(device), (noise - clean).to(device))

    @torch.no_grad()
    def loss(self, velocity: Velocity, batch: int = 1024) -> float:
        total = 0.0
        for start in range(0, len(self.t), batch):
            end = start + batch
            prediction = velocity(self.x_t[start:end], self.t[start:end]).float()
            total += float(((prediction - self.target[start:end]) ** 2).double().sum())
        return total / self.target.numel()


# --------------------------------------------------------------------------- latents


class ChannelMoments:
    """Per-channel mean and standard deviation of z ~ q(z | x) over many latents,
    accumulated in float64: var(z) = var(mu) + E[sigma^2]."""

    def __init__(self, channels: int) -> None:
        self.count = 0
        self.mean_sum = torch.zeros(channels, dtype=torch.float64)
        self.square_sum = torch.zeros(channels, dtype=torch.float64)
        self.variance_sum = torch.zeros(channels, dtype=torch.float64)

    def add(self, mean: Tensor, sigma: Tensor) -> None:
        mu = mean.detach().double().cpu().transpose(0, 1).reshape(len(self.mean_sum), -1)
        spread = sigma.detach().double().cpu().transpose(0, 1).reshape(len(self.mean_sum), -1)
        self.count += mu.shape[1]
        self.mean_sum += mu.sum(dim=1)
        self.square_sum += (mu**2).sum(dim=1)
        self.variance_sum += (spread**2).sum(dim=1)

    def result(self) -> tuple[Tensor, Tensor]:
        mean = self.mean_sum / self.count
        variance = self.square_sum / self.count - mean**2 + self.variance_sum / self.count
        return mean.float(), variance.clamp_min(1e-12).sqrt().float()


@dataclass
class LatentSet:
    """Posterior statistics of the training variants and validation icons (float16)."""

    train_mean: Tensor  # (N, c, g, g)
    train_sigma: Tensor
    train_icon: Tensor  # (N,) int32: training-split index of each latent's icon
    train_variant: Tensor  # (N,) int16: 0 the icon itself when `originals`, else 1..
    validation_mean: Tensor  # (V, c, g, g)
    validation_sigma: Tensor
    validation_hexcodes: list[str]
    channel_mean: Tensor  # (c,) float32
    channel_std: Tensor
    settings: dict[str, Any]
    dropped_variants: int
    seconds: float

    def state(self) -> dict[str, Any]:
        return {
            **{key: getattr(self, key) for key in self.__dataclass_fields__},
            "format": LATENT_FORMAT,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any]) -> LatentSet:
        return cls(**{key: state[key] for key in cls.__dataclass_fields__})


def latent_settings_record(
    settings: PrecomputeSettings,
    *,
    vt_sha256: str,
    dataset_sha256: str,
    latent_shape: Sequence[int],
    image_size: int,
) -> dict[str, Any]:
    """Everything that decides a latent cache's contents; its hash names the file."""

    values = asdict(settings)
    del values["workers"]
    return {
        "format": LATENT_FORMAT,
        "vt_checkpoint_sha256": vt_sha256,
        "dataset_sha256": dataset_sha256,
        "latent_shape": list(latent_shape),
        "image_size": image_size,
        "precision": "IEEE float32 encoding (TF32 off), stored float16",
        **values,
    }


def latent_cache_path(root: Path, vt_run_id: str, record: Mapping[str, Any]) -> Path:
    digest = hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()[:12]
    return root / "latent-flow" / f"{vt_run_id}-{digest}.pt"


def _variant_numbers(icons: np.ndarray, first: int) -> np.ndarray:
    """first, first + 1, ... within each run of equal icon indices."""

    numbers = np.zeros(len(icons), dtype=np.int16)
    for position in range(len(icons)):
        same = position > 0 and icons[position] == icons[position - 1]
        numbers[position] = numbers[position - 1] + 1 if same else first
    return numbers


@torch.no_grad()
def precompute_latents(
    vt: VariationalTranscriber,
    train: SplitData,
    validation: SplitData,
    layout: SequenceLayout,
    template: PackedTensorProgram,
    settings: PrecomputeSettings,
    device: torch.device,
    record: Mapping[str, Any],
    log: Callable[[str], None] = print,
) -> LatentSet:
    """Posterior means and sigmas of K latents per training icon and of the validation
    icons, encoded by the frozen VT in IEEE float32 (TF32 off)."""

    started = time.perf_counter()
    vt.eval()
    count = min(settings.train_icons or len(train.tokens), len(train.tokens))
    moments = ChannelMoments(vt.settings.channels)
    means: list[Tensor] = []
    sigmas: list[Tensor] = []
    icons: list[np.ndarray] = []
    variants: list[np.ndarray] = []

    def encode(images: Tensor, accumulate: bool) -> tuple[Tensor, Tensor]:
        parts: list[tuple[Tensor, Tensor]] = []
        with strict_float32():
            for start in range(0, len(images), settings.encode_batch):
                mean, log_variance = vt.posterior(
                    images[start : start + settings.encode_batch].to(device)
                )
                sigma = (0.5 * log_variance).exp()
                if accumulate:
                    moments.add(mean, sigma)
                parts.append((mean.half().cpu(), sigma.half().cpu()))
        return torch.cat([p[0] for p in parts]), torch.cat([p[1] for p in parts])

    if settings.originals:
        mean, sigma = encode(train.images[:count], True)
        means.append(mean)
        sigmas.append(sigma)
        icons.append(np.arange(count, dtype=np.int64))
        variants.append(np.zeros(count, dtype=np.int16))
    extra = settings.variants - int(settings.originals)
    dropped = 0
    if extra > 0:
        augment = TrainConfig(
            augment_variants=extra,
            augment_seed=settings.seed,
            augment_mirror=settings.mirror,
            augment_max_shift=settings.max_shift,
            augment_colour=settings.colour,
        )
        tokens = train.tokens[:count].numpy().astype(np.int16)
        chunks = np.array_split(np.arange(count), max(1, math.ceil(count / settings.chunk_icons)))
        jobs = [
            (
                chunk,
                tokens[chunk],
                layout,
                template,
                vt.config.image_size,
                augment,
                settings.seed * 100_003 + number,
            )
            for number, chunk in enumerate(chunks)
        ]
        pool = ProcessPoolExecutor(max_workers=settings.workers) if settings.workers > 1 else None
        try:
            results: Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]] = (
                pool.map(_augment_chunk, jobs) if pool is not None else map(_augment_chunk, jobs)
            )
            for number, (index, _, images) in enumerate(results):
                dropped += len(chunks[number]) * extra - len(index)
                if len(index):
                    mean, sigma = encode(torch.from_numpy(images), True)
                    means.append(mean)
                    sigmas.append(sigma)
                    icons.append(index)
                    variants.append(_variant_numbers(index, 1))
                if number % 50 == 0 or number == len(jobs) - 1:
                    log(json.dumps({"latents": "variants", "jobs": number + 1, "of": len(jobs)}))
        finally:
            if pool is not None:
                pool.shutdown()
    kept = min(settings.validation_icons, len(validation.tokens))
    validation_mean, validation_sigma = encode(validation.images[:kept], False)
    channel_mean, channel_std = moments.result()
    return LatentSet(
        train_mean=torch.cat(means),
        train_sigma=torch.cat(sigmas),
        train_icon=torch.from_numpy(np.concatenate(icons).astype(np.int32)),
        train_variant=torch.from_numpy(np.concatenate(variants)),
        validation_mean=validation_mean,
        validation_sigma=validation_sigma,
        validation_hexcodes=list(validation.hexcodes[:kept]),
        channel_mean=channel_mean,
        channel_std=channel_std,
        settings=dict(record),
        dropped_variants=dropped,
        seconds=time.perf_counter() - started,
    )


def load_or_precompute(
    vt: VariationalTranscriber,
    train: SplitData,
    validation: SplitData,
    layout: SequenceLayout,
    template: PackedTensorProgram,
    settings: PrecomputeSettings,
    device: torch.device,
    *,
    cache_root: Path,
    vt_run_id: str,
    vt_sha256: str,
    dataset_sha256: str,
) -> tuple[LatentSet, Path, bool]:
    """The cached latents for these settings, computed and written once: (latents, the
    cache file, whether this call made it). A file whose settings differ is refused."""

    record = latent_settings_record(
        settings,
        vt_sha256=vt_sha256,
        dataset_sha256=dataset_sha256,
        latent_shape=vt.latent_shape,
        image_size=vt.config.image_size,
    )
    path = latent_cache_path(cache_root, vt_run_id, record)
    if path.exists():
        state = torch.load(path, map_location="cpu")
        if state.get("format") != LATENT_FORMAT or state.get("settings") != record:
            raise ValueError(f"{path} holds other settings; preserve it and investigate")
        return LatentSet.from_state(state), path, False
    latents = precompute_latents(vt, train, validation, layout, template, settings, device, record)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(latents.state(), temporary)
    temporary.rename(path)
    return latents, path, True


# --------------------------------------------------------------------------- checkpoints


def checkpoint_state(
    network: FlowTransformer,
    prior: LatentFlowPrior,
    *,
    step: int,
    parent: Mapping[str, str],
    validation_flow_loss: float | None,
) -> dict[str, Any]:
    """What a flow checkpoint holds: `network`'s weights (the EMA's, when served), the
    standardisation, the sampler's steps and the frozen parent VT's path and sha256."""

    return {
        "kind": CHECKPOINT_KIND,
        "model": network.state_dict(),
        "network": asdict(network.config),
        "standardisation": {
            "mean": cast(Tensor, prior.channel_mean).tolist(),
            "std": cast(Tensor, prior.channel_std).tolist(),
        },
        "sampling_steps": prior.steps,
        "parent_run": parent["run"],
        "parent_checkpoint": parent["checkpoint"],
        "parent_checkpoint_sha256": parent["sha256"],
        "step": step,
        "validation_flow_loss": validation_flow_loss,
    }


def load_flow_checkpoint(state: Mapping[str, Any]) -> LatentFlowPrior:
    """The prior a flow checkpoint holds, on the CPU, in eval mode."""

    if state.get("kind") != CHECKPOINT_KIND:
        raise ValueError(f"not a latent flow checkpoint (kind {state.get('kind')!r})")
    network = FlowTransformer(FlowNetworkConfig(**state["network"]))
    network.load_state_dict(state["model"])
    standard = state["standardisation"]
    prior = LatentFlowPrior(
        network,
        torch.tensor(standard["mean"], dtype=torch.float32),
        torch.tensor(standard["std"], dtype=torch.float32),
        steps=int(state.get("sampling_steps", 50)),
    )
    return prior.eval()


def load_flow_parent(state: Mapping[str, Any], layout: SequenceLayout) -> VariationalTranscriber:
    """The frozen VT a flow checkpoint names, on the CPU; refused if the file changed."""

    path = Path(str(state["parent_checkpoint"]))
    recorded = state.get("parent_checkpoint_sha256")
    if recorded and _sha256(path) != recorded:
        raise ValueError(f"{path} changed since the flow was trained on it")
    model = load_checkpoint(torch.load(path, map_location="cpu"), layout)
    if model.latent_shape != tuple(state["network"][k] for k in ("channels", "grid", "grid")):
        raise ValueError(f"{path}'s latent shape {model.latent_shape} is not the flow's")
    return model


# --------------------------------------------------------------------------- run


@dataclass(frozen=True)
class SmokeOptions:
    steps: int = 50
    batch_size: int = 8
    layers: int = 2
    variants: int = 2
    train_icons: int = 32
    parent_checkpoint: Path | None = None
    """Another VT checkpoint (a smoke VT's) in place of the config's parent."""


def smoke_config(config: FlowConfig, options: SmokeOptions) -> FlowConfig:
    """The real config shrunk to a mechanics check."""

    steps = options.steps
    return replace(
        config,
        parent_checkpoint=options.parent_checkpoint or config.parent_checkpoint,
        latents=replace(
            config.latents,
            variants=options.variants,
            train_icons=options.train_icons,
            validation_icons=8,
            workers=min(config.latents.workers, 4),
        ),
        network=replace(config.network, layers=options.layers),
        training=replace(
            config.training,
            steps=steps,
            batch_size=options.batch_size,
            warmup_steps=max(1, steps // 5),
            eval_every=max(1, steps // 2),
            trace_every=max(1, steps // 10),
            validation_draws=2,
        ),
        eval_icons=8,
        prior_samples=4,
        interpolation_pairs=1,
        interpolation_frames=3,
        trajectories=1,
        latency_repeats=2,
    )


def checkpoint_run(path: Path) -> str:
    """The run a checkpoint belongs to, by its folder: runs/<id>/best.pt in the cache, or
    <id>/checkpoints/best.pt for a smoke run."""

    folder = path.parent
    return folder.parent.name if folder.name == "checkpoints" else folder.name


def check_parent(config: FlowConfig, runs_root: Path | None = None) -> dict[str, Any]:
    """The declared gate: the parent VT run completed, its record names this checkpoint,
    and it passed every criterion in `parent_gate`. Raises SystemExit otherwise."""

    path = (runs_root or REPO_ROOT / "runs") / config.parent_run / "run.yaml"
    if not path.is_file():
        raise SystemExit(f"no parent run record: {path}")
    record = yaml.safe_load(path.read_text()) or {}
    problems = []
    if record.get("state") != "completed":
        problems.append(f"the parent run is {record.get('state')!r}, not completed")
    checkpoints = (record.get("outputs") or {}).get("checkpoints")
    if checkpoints is None or Path(str(checkpoints)) / "best.pt" != config.parent_checkpoint:
        problems.append(f"the parent's checkpoints are {checkpoints}, not the config's")
    criteria = (record.get("result") or {}).get("criteria") or {}
    outcomes = {name: (criteria.get(name) or {}).get("pass") for name in config.parent_gate}
    problems += [
        f"{name}: pass is {value}" for name, value in outcomes.items() if value is not True
    ]
    if problems:
        raise SystemExit(f"parent gate failed ({path}): " + "; ".join(problems))
    return {"record": str(path), "state": "completed", "gate": outcomes}


def train_and_evaluate(
    config_path: Path, *, smoke: Path | None = None, options: SmokeOptions | None = None
) -> dict[str, Any]:
    """Precompute latents, train the flow, select by EMA validation flow loss, evaluate,
    record. With `smoke`, everything - latents, record, sheets, checkpoints - goes under
    `smoke`, nothing to `runs/`, the cache's run directories or the registry."""

    config = load_config(config_path)
    uncommitted = uncommitted_sources()
    if smoke is None and uncommitted:
        raise SystemExit(
            "commit first: a run record names a commit and hashes `git diff HEAD`, which "
            "cannot identify untracked or uncommitted code; uncommitted under src/, "
            "configs/, tests/:\n" + "\n".join(uncommitted)
        )
    if smoke is not None:
        config = smoke_config(config, options or SmokeOptions())
    gate: dict[str, Any] | str = check_parent(config) if smoke is None else "not checked (smoke)"
    torch.manual_seed(config.training.seed)
    device = torch.device("cuda")
    parent_sha = _sha256(config.parent_checkpoint)
    parent_state = torch.load(config.parent_checkpoint, map_location="cpu")
    image_size = int(parent_state["config"]["image_size"])
    plain, _, layout, pilot, dataset_hash = load_corpus(config.pilot_config, image_size)
    vt = load_checkpoint(parent_state, layout).to(device).requires_grad_(False)
    del parent_state

    identity = run_identity(config_path, dataset_hash, config.slug)
    run_id = identity["run_id"]
    if smoke is None:
        run_dir = REPO_ROOT / "runs" / run_id
        checkpoint_dir = CACHE_ROOT / "runs" / run_id
        parent_run = config.parent_run
    else:
        run_id = f"{run_id}-smoke"
        run_dir = smoke / run_id
        checkpoint_dir = run_dir / "checkpoints"
        parent_run = checkpoint_run(config.parent_checkpoint)
    if run_dir.exists():
        raise SystemExit(f"run directory already exists; preserve it and change the id: {run_id}")

    from mojidiff.learning.openmoji_pilot import _load_program

    template = _load_program(_pilot_rows(pilot)["primary/train"][0], pilot, layout.codec)
    latents, latent_path, created = load_or_precompute(
        vt,
        plain["primary/train"],
        plain["primary/validation"],
        layout,
        template,
        config.latents,
        device,
        cache_root=CACHE_ROOT if smoke is None else smoke,
        vt_run_id=parent_run,
        vt_sha256=parent_sha,
        dataset_sha256=dataset_hash,
    )
    channels, grid, _ = vt.latent_shape
    network_config = replace(config.network, channels=channels, grid=grid)
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    registry = smoke is None

    network = FlowTransformer(network_config).to(device)
    trainer = FlowTrainer(network, config.training)
    parameters = {
        "flow": parameter_count(network),
        "parent_vt_frozen": parameter_count(vt),
        "total": parameter_count(network) + parameter_count(vt),
        "note": (
            "model_parameters is the served sampler: the flow and the frozen VT that "
            "decodes its samples, as the gallery's canvas-flow backend counts it; only the "
            "flow trains"
        ),
    }
    mean, std = latents.channel_mean.to(device), latents.channel_std.to(device)
    prior = LatentFlowPrior(network, mean, std, config.sampling.steps)
    ema_prior = LatentFlowPrior(trainer.ema, mean, std, config.sampling.steps)
    parent = {"run": parent_run, "checkpoint": str(config.parent_checkpoint), "sha256": parent_sha}
    command = f".venv/bin/python -m mojidiff.learning.latent_flow --config {config_path}"
    if smoke is not None:
        chosen = options or SmokeOptions()
        command += (
            f" --smoke {smoke} --steps {chosen.steps} --batch-size {chosen.batch_size} "
            f"--layers {chosen.layers} --variants {chosen.variants} "
            f"--train-icons {chosen.train_icons}"
        )
        if chosen.parent_checkpoint is not None:
            command += f" --parent-checkpoint {chosen.parent_checkpoint}"
    record: dict[str, Any] = {
        "schema_version": 2,
        "run_id": run_id,
        "state": "running",
        "planned_at": _now(),
        "hypothesis": config.hypothesis,
        "parent_run": parent_run,
        **{k: identity[k] for k in ("git_commit", "dirty_patch_sha256", "config_sha256")},
        "uncommitted_sources": uncommitted,
        "config": str(config_path),
        "config_resolved": {
            "parent_gate": list(config.parent_gate),
            "latents": asdict(config.latents),
            "network": asdict(network_config),
            "training": asdict(config.training),
            "sampling": asdict(config.sampling),
            "eval_icons": config.eval_icons,
            "prior_samples": config.prior_samples,
            "interpolation_pairs": config.interpolation_pairs,
            "interpolation_frames": config.interpolation_frames,
            "trajectories": config.trajectories,
        },
        "parent": {
            "run": parent_run,
            "checkpoint": str(config.parent_checkpoint),
            "checkpoint_sha256": parent_sha,
            "latent_shape": list(vt.latent_shape),
            "gate": gate,
            "frozen": True,
        },
        "latents": {
            "cache": str(latent_path),
            "cache_sha256": _sha256(latent_path),
            "created_by_this_run": created,
            "settings": latents.settings,
            "train_latents": len(latents.train_mean),
            "train_icons": int(latents.train_icon.max()) + 1 if len(latents.train_icon) else 0,
            "dropped_variants": latents.dropped_variants,
            "validation_latents": len(latents.validation_mean),
            "precompute_seconds": latents.seconds,
            "channel_mean": latents.channel_mean.tolist(),
            "channel_std": latents.channel_std.tolist(),
            "note": "test icons are never encoded; validation latents select the checkpoint",
        },
        "dataset": {
            "pilot_config": str(config.pilot_config),
            "cache_sha256": dataset_hash,
            "train_icons": len(plain["primary/train"].tokens),
            "evaluated_on": "primary/validation",
        },
        "seed": config.training.seed,
        "determinism": (
            "seeded (torch generator on the device for batches, t and eps; fixed CPU draws "
            "for validation); cuDNN and SDPA kernels not forced deterministic"
        ),
        "model_parameters": parameters["total"],
        "parameters": parameters,
        "command": command,
        "outputs": {
            "run_dir": str(run_dir.relative_to(REPO_ROOT)) if smoke is None else str(run_dir),
            "checkpoints": str(checkpoint_dir),
        },
        "notes": config.notes,
    }
    if smoke is not None:
        record["smoke"] = {**asdict(options or SmokeOptions()), "registry": False}
    _write_yaml(run_dir / "run.yaml", record)
    if registry:
        _append_registry(run_id, "running", config=str(config_path), slug=config.slug)
    print(
        json.dumps(
            {
                "run_id": run_id,
                "flow_parameters": parameters["flow"],
                "model_parameters": parameters["total"],
            }
        ),
        flush=True,
    )

    metrics_path = run_dir / "metrics.jsonl"

    def fail(error: BaseException) -> None:
        record.update(state="failed", failed_at=_now(), failure_reason=repr(error))
        _write_yaml(run_dir / "run.yaml", record)
        if registry:
            _append_registry(run_id, "failed", reason=repr(error))

    def write(entry: dict[str, Any]) -> None:
        with metrics_path.open("a") as handle:
            handle.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    try:
        training = config.training
        # On the device for the training loop only; cleared before the evaluation.
        data = {"mean": latents.train_mean.to(device), "sigma": latents.train_sigma.to(device)}
        generator = torch.Generator(device=device).manual_seed(training.seed)
        validation_clean = latents.validation_mean.float()
        if training.posterior_sampling:
            draws = torch.randn(
                validation_clean.shape, generator=torch.Generator().manual_seed(POSTERIOR_DRAW_SEED)
            )
            validation_clean = validation_clean + draws * latents.validation_sigma.float()
        validation = FlowValidation.draw(
            prior.standardise(validation_clean.to(device)),
            training.validation_draws,
            VALIDATION_DRAW_SEED,
            time_mean=training.time_mean,
            time_std=training.time_std,
        )
        baseline_loss = validation.loss(gaussian_velocity)

        def batch() -> Tensor:
            index = torch.randint(
                len(data["mean"]), (training.batch_size,), device=device, generator=generator
            )
            z = data["mean"][index].float()
            if training.posterior_sampling:
                eps = torch.randn(z.shape, device=device, generator=generator)
                z = z + eps * data["sigma"][index].float()
            return prior.standardise(z)

        best = float("inf")
        selected: dict[str, Any] | None = None
        window: list[Tensor] = []
        reset_peak_memory(device)
        started = time.perf_counter()
        write(
            {
                "kind": "eval",
                "step": 0,
                "validation_flow_loss_ema": validation.loss(ema_prior.velocity),
                "validation_flow_loss_gaussian_baseline": baseline_loss,
            }
        )
        for step in range(1, training.steps + 1):
            loss, norm = trainer.step(batch(), generator)
            window.append(loss)
            if step % training.trace_every == 0:
                write(
                    {
                        "kind": "train",
                        "step": step,
                        "flow_loss": float(torch.stack(window).mean()),
                        "grad_norm": float(norm),
                        "learning_rate": trainer.scheduler.get_last_lr()[0],
                    }
                )
                window = []
            if step % training.eval_every == 0 or step == training.steps:
                ema_loss = validation.loss(ema_prior.velocity)
                entry = {
                    "kind": "eval",
                    "step": step,
                    "validation_flow_loss_ema": ema_loss,
                    "validation_flow_loss_raw": validation.loss(prior.velocity),
                    "validation_flow_loss_gaussian_baseline": baseline_loss,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                write(entry)
                if ema_loss < best:
                    best = ema_loss
                    selected = {"step": step, "validation_flow_loss_ema": ema_loss}
                    torch.save(
                        checkpoint_state(
                            trainer.ema,
                            ema_prior,
                            step=step,
                            parent=parent,
                            validation_flow_loss=ema_loss,
                        ),
                        checkpoint_dir / "best.pt",
                    )
        latest = checkpoint_state(
            trainer.ema, ema_prior, step=training.steps, parent=parent, validation_flow_loss=None
        )
        latest.update(
            raw_model=network.state_dict(),
            optimizer=trainer.optimizer.state_dict(),
            scheduler=trainer.scheduler.state_dict(),
        )
        torch.save(latest, checkpoint_dir / "latest.pt")
    except BaseException as error:
        fail(error)
        raise
    train_seconds = time.perf_counter() - started
    data.clear()
    del validation
    torch.cuda.empty_cache()
    try:
        best_state = torch.load(checkpoint_dir / "best.pt", map_location="cpu")
        served = load_flow_checkpoint(best_state).to(device)
        result, latency = final_evaluation(
            served, vt, plain, template, device, run_dir, config, latents
        )
        result["selection"] = {
            "rule": (
                "lowest EMA validation flow loss (fixed t and eps draws, "
                f"{training.validation_draws} per validation latent) over evaluations every "
                f"{training.eval_every} steps"
            ),
            **cast(dict[str, Any], selected),
        }
        result["selected_step"] = int(best_state["step"])
        result["validation_flow_loss"]["gaussian_velocity_baseline_at_selection_time"] = (
            baseline_loss
        )
        resource = resource_summary(
            device, train_seconds=train_seconds, latency=latency
        ).as_record()
        resource.update(
            inference=(
                f"{config.sampling.steps} Euler steps of the EMA flow at batch 1 plus the VT's "
                "float32 CUDA-graph greedy decode, IEEE float32 (TF32 off)"
            ),
            excludes="rasterising the output SVG",
            note="measured while other GPU work may be running",
        )
        prior_block = result["prior_samples"]
        record.update(
            state="completed",
            completed_at=_now(),
            resource=resource,
            baselines={
                "gaussian_velocity_validation_flow_loss": result["validation_flow_loss"][
                    "gaussian_velocity_baseline"
                ],
                "vt_standard_normal_prior_fragment_rate": prior_block["vt_standard_normal"][
                    "fragment_rate_ink_below_0.10"
                ],
                "validation_icons_fragment_rate": prior_block[
                    "validation_fragment_rate_ink_below_0.10"
                ],
                "note": (
                    "zero-parameter controls on the same draws: the exact velocity if the "
                    "standardised latents were N(0, I); the parent VT decoding the same "
                    "N(0, I) seeds as latents (its own prior); the validation icons"
                ),
            },
            result=result,
        )
        _write_yaml(run_dir / "run.yaml", record)
        (run_dir / "result.md").write_text(result_markdown(record))
        artifacts = {
            "checkpoint_best": str(checkpoint_dir / "best.pt"),
            "checkpoint_latest": str(checkpoint_dir / "latest.pt"),
            "latents": str(latent_path),
            "samples": "prior-samples.png (flow samples, noise scale 1, Euler, greedy decode)",
            "vt_normal_samples": "vt-normal-samples.png (the parent VT on the same N(0, I) seeds)",
            "slerp": "slerp-interpolations.png (A, frames through the prior's noise, B)",
            "lerp": "lerp-interpolations.png (A, lerp of posterior means, B)",
            "watch_it_draw": "watch-it-draw.png (z_hat_0 at 8 flow times, then the sample)",
        }
        (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n")
        if registry:
            _append_registry(run_id, "completed", result_file=f"runs/{run_id}/result.md")
    except BaseException as error:
        fail(error)
        raise
    print(json.dumps({"completed": run_id}), flush=True)
    return record


# --------------------------------------------------------------------------- evaluation


def fixed_pairs(count: int, pairs: int, seed: int = PAIR_SEED) -> list[tuple[int, int]]:
    """The validation pairs `pixel_latent` and `latent_metrics` draw, in their order."""

    rng = np.random.default_rng(seed)
    return [
        (int(a), int(b)) for a, b in (rng.choice(count, 2, replace=False) for _ in range(pairs))
    ]


def _ink_block(renders: Sequence[np.ndarray | None], tokens: Tensor) -> dict[str, Any]:
    ink = [ink_coverage(r) for r in renders]
    return {
        "count": len(renders),
        "distinct": len({tuple(row.tolist()) for row in tokens}),
        "rendered": sum(1 for r in renders if r is not None),
        "ink_coverage_median": float(np.median(ink)) if ink else None,
        "fragment_rate_ink_below_0.10": float(np.mean([v < FRAGMENT_INK for v in ink])),
        "near_blank_rate_ink_below_0.02": float(np.mean([v < NEAR_BLANK_INK for v in ink])),
    }


def final_evaluation(
    prior: LatentFlowPrior,
    vt: VariationalTranscriber,
    plain: dict[str, SplitData],
    template: PackedTensorProgram,
    device: torch.device,
    run_dir: Path,
    config: FlowConfig,
    latents: LatentSet,
) -> tuple[dict[str, Any], LatencyReport]:
    """Samples, interpolation and trajectory sheets, the validation flow loss and
    latency, all in IEEE float32 (TF32 off)."""

    with strict_float32() as flags:
        result, latency = _final_evaluation(
            prior, vt, plain, template, device, run_dir, config, latents
        )
        result["float32_flags"] = flags
    return result, latency


@torch.no_grad()
def _final_evaluation(
    prior: LatentFlowPrior,
    vt: VariationalTranscriber,
    plain: dict[str, SplitData],
    template: PackedTensorProgram,
    device: torch.device,
    run_dir: Path,
    config: FlowConfig,
    latents: LatentSet,
) -> tuple[dict[str, Any], LatencyReport]:
    layout = vt.layout
    steps = config.sampling.steps
    shape = vt.latent_shape
    columns = 8

    def renders_of(tokens: Tensor) -> list[np.ndarray | None]:
        return _programs_to_renders(tokens, layout, template)

    # Validation flow loss of the served (EMA) weights and of the Gaussian velocity.
    training = config.training
    clean = latents.validation_mean.float()
    if training.posterior_sampling:
        eps = torch.randn(clean.shape, generator=torch.Generator().manual_seed(POSTERIOR_DRAW_SEED))
        clean = clean + eps * latents.validation_sigma.float()
    validation = FlowValidation.draw(
        prior.standardise(clean.to(device)),
        training.validation_draws,
        VALIDATION_DRAW_SEED,
        time_mean=training.time_mean,
        time_std=training.time_std,
    )
    flow_loss = {
        "ema_served": validation.loss(prior.velocity),
        "gaussian_velocity_baseline": validation.loss(gaussian_velocity),
        "icons": len(clean),
        "draws_per_icon": training.validation_draws,
    }
    del validation

    # Prior samples, and the parent VT decoding the same N(0, I) seeds as latents.
    noise = torch.randn(
        config.prior_samples, *shape, generator=torch.Generator().manual_seed(PRIOR_SEED)
    ).to(device)
    started = time.perf_counter()
    sampled, _ = prior.generate(noise, steps)
    sample_tokens = decode_latents(vt, sampled)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    batched_ms = (time.perf_counter() - started) * 1000.0 / len(noise)
    sample_renders = renders_of(sample_tokens)
    (run_dir / "prior-samples.png").write_bytes(
        contact_sheet(
            [sample_renders[i : i + columns] for i in range(0, len(sample_renders), columns)]
        )
    )
    normal_tokens = decode_latents(vt, noise)
    normal_renders = renders_of(normal_tokens)
    (run_dir / "vt-normal-samples.png").write_bytes(
        contact_sheet(
            [normal_renders[i : i + columns] for i in range(0, len(normal_renders), columns)]
        )
    )
    validation_split = plain["primary/validation"].subset(config.eval_icons)
    targets = [t.numpy() for t in validation_split.targets]
    validation_ink = [ink_coverage(t) for t in targets]
    prior_block = {
        **_ink_block(sample_renders, sample_tokens),
        "noise_scale": 1.0,
        "seed": PRIOR_SEED,
        "euler_steps": steps,
        "vt_standard_normal": _ink_block(normal_renders, normal_tokens),
        "validation_fragment_rate_ink_below_0.10": float(
            np.mean([v < FRAGMENT_INK for v in validation_ink])
        ),
    }

    # Interpolation: slerp through the prior's noise, and lerp of posterior means.
    pairs = fixed_pairs(len(validation_split.tokens), config.interpolation_pairs)
    frames = config.interpolation_frames
    ends = [v for pair in pairs for v in pair]
    means, _ = vt.posterior(validation_split.images[ends].to(device))
    weights = torch.linspace(0.0, 1.0, frames, device=device).view(-1, 1, 1, 1)
    slerp_rows: list[list[np.ndarray | None]] = []
    lerp_rows: list[list[np.ndarray | None]] = []
    measures: dict[str, dict[str, list[float]]] = {
        name: {"interior_ink": [], "endpoint_error": [], "distinct": []}
        for name in ("slerp", "lerp")
    }
    for number, (first, second) in enumerate(pairs):
        a, b = means[2 * number], means[2 * number + 1]
        paths = {
            "slerp": prior.slerp_path(a, b, frames, steps),
            "lerp": (1.0 - weights) * a + weights * b,
        }
        for name, path in paths.items():
            tokens = decode_latents(vt, path)
            strip = renders_of(tokens)
            row = [targets[first], *strip, targets[second]]
            (slerp_rows if name == "slerp" else lerp_rows).append(row)
            measures[name]["interior_ink"] += [ink_coverage(r) for r in strip[1:-1]]
            measures[name]["endpoint_error"] += [
                pixel_error(strip[0], targets[first]),
                pixel_error(strip[-1], targets[second]),
            ]
            measures[name]["distinct"].append(len({tuple(r.tolist()) for r in tokens}))
    (run_dir / "slerp-interpolations.png").write_bytes(contact_sheet(slerp_rows))
    (run_dir / "lerp-interpolations.png").write_bytes(contact_sheet(lerp_rows))
    interpolation: dict[str, Any] = {
        "pairs": [list(p) for p in pairs],
        "frames": frames,
        "inversion_steps": steps,
    }
    for name, values in measures.items():
        interpolation[name] = {
            "interior_fragment_rate_ink_below_0.10": float(
                np.mean([v < FRAGMENT_INK for v in values["interior_ink"]])
            ),
            "endpoint_pixel_error": bootstrap_mean_interval(values["endpoint_error"]),
            "distinct_programs_per_strip": float(np.mean(values["distinct"])),
        }

    # Watch it draw: z_hat_0 decoded at spread-out times of single trajectories.
    indices = trajectory_indices(steps, config.sampling.trajectory_frames)
    start_noise = torch.randn(
        config.trajectories, *shape, generator=torch.Generator().manual_seed(TRAJECTORY_SEED)
    ).to(device)
    final, recorded = prior.generate(start_noise, steps, indices)
    stacked = torch.stack([frame for _, frame in recorded] + [final], dim=1)
    trajectory_renders = renders_of(decode_latents(vt, stacked.reshape(-1, *shape)))
    width = len(recorded) + 1
    (run_dir / "watch-it-draw.png").write_bytes(
        contact_sheet(
            [trajectory_renders[i : i + width] for i in range(0, len(trajectory_renders), width)]
        )
    )
    watch = {
        "trajectories": config.trajectories,
        "seed": TRAJECTORY_SEED,
        "times": [t for t, _ in recorded],
        "step_indices": indices,
        "columns": "z_hat_0 at each time, then the final sample",
    }

    latency, flow_only = _latency(prior, vt, device, steps, config.latency_repeats)
    criteria = {
        name: {"claim": claim, "status": CRITERIA_SCORED_BY[name]}
        for name, claim in CRITERIA.items()
    }
    result: dict[str, Any] = {
        "precision": precision_note(),
        "validation_flow_loss": flow_loss,
        "prior_samples": prior_block,
        "interpolation": interpolation,
        "watch_it_draw": watch,
        "latency": {
            "end_to_end_ms_batch_1": latency.median_ms_per_icon,
            "end_to_end_p95_ms_batch_1": latency.p95_ms_per_icon,
            "flow_only_ms_batch_1": flow_only.median_ms_per_icon,
            "batched_ms_per_sample": batched_ms,
            "batched_note": (
                f"{config.prior_samples} samples: flow then greedy_decode in batches of 64"
            ),
        },
        "criteria": criteria,
    }
    return result, latency


CRITERIA = {
    "B1": "fragment rate at most 10%",
    "B2": (
        "CLIP precision at least 2x the best latent-v2 prior row, and recall above it, "
        "both with intervals excluding zero"
    ),
    "B3": "copy rate at most 10% under copy rule without-twins, and all samples distinct",
    "B4": (
        "slerp interior fragments at most 15%, jump share below latent-v2's (paired), "
        "detours at most 20%"
    ),
    "B5": "slerp interior CLIP precision above v9 on crossfades, paired",
}

_SAMPLES_REPORT = (
    "scored after the run by latent_metrics through the gallery backend canvas-flow, copy "
    "rule without-twins, default settings: its samples block"
)
_SLERP_REPORT = (
    "scored after the run by latent_metrics through the gallery backend canvas-flow, copy "
    "rule without-twins, with --interpolation backend: its interpolation block, the "
    "backend's slerp through the prior's noise (the report records interpolation.path "
    "slerp-through-prior-noise). The default report's interpolation block is the frozen "
    "VT's lerp of posterior means and scores neither B4 nor B5"
)
CRITERIA_SCORED_BY = {
    "B1": _SAMPLES_REPORT,
    "B2": _SAMPLES_REPORT,
    "B3": _SAMPLES_REPORT,
    "B4": _SLERP_REPORT,
    "B5": _SLERP_REPORT,
}
"""Which harness report scores each criterion, as the run record states it."""


def _latency(
    prior: LatentFlowPrior,
    vt: VariationalTranscriber,
    device: torch.device,
    steps: int,
    repeats: int,
) -> tuple[LatencyReport, LatencyReport]:
    """Batch-1 latency: flow plus the VT's float32 CUDA-graph decode, and the flow alone."""

    from mojidiff.learning.fast_decode import GraphDecoder

    graph = GraphDecoder(vt, dtype=torch.float32)
    noise = torch.randn(1, *vt.latent_shape, generator=torch.Generator().manual_seed(PRIOR_SEED))
    noise = noise.to(device)

    def end_to_end() -> Tensor:
        z, _ = prior.generate(noise, steps)
        return graph.decode(z[0])

    both = measure_latency(end_to_end, device=device, warmup=2, repeats=repeats)
    flow = measure_latency(
        lambda: prior.generate(noise, steps), device=device, warmup=2, repeats=repeats
    )
    return both, flow


# --------------------------------------------------------------------------- report


def _interval(value: Sequence[float]) -> str:
    return f"{value[0]:.4f} [{value[1]:.4f}, {value[2]:.4f}]"


def result_markdown(record: Mapping[str, Any]) -> str:
    result = record["result"]
    resource = record["resource"]
    prior = result["prior_samples"]
    normal = prior["vt_standard_normal"]
    flow = result["validation_flow_loss"]
    interpolation = result["interpolation"]
    latency = result["latency"]
    lines = [
        f"# {record['run_id']}",
        "",
        f"**Hypothesis.** {record['hypothesis']}",
        "",
        "## Criteria",
        "",
        "B1-B5 are scored after this run by `latent_metrics` through the gallery backend",
        "`canvas-flow` (339 samples, 32 pairs, CLIP, copy rule without-twins): B1-B3 on its",
        "default report, B4 and B5 on its `--interpolation backend` report, whose",
        "interpolation block is the slerp through the prior's noise (the default report's is",
        "the frozen VT's lerp). The numbers below are this run's own checks, not the criteria.",
        "",
        "| criterion | claim |",
        "| --- | --- |",
        *[f"| {name} | {value['claim']} |" for name, value in result["criteria"].items()],
        "",
        f"## Result ({result['precision']})",
        "",
        "| measure | value |",
        "| --- | --- |",
        f"| validation flow loss, EMA (served) | {flow['ema_served']:.4f} |",
        f"| validation flow loss, Gaussian velocity (zero parameters) | "
        f"{flow['gaussian_velocity_baseline']:.4f} |",
        f"| selected step | {result['selected_step']} |",
        f"| flow samples distinct / rendered | {prior['distinct']} / {prior['rendered']} of "
        f"{prior['count']} |",
        f"| flow samples with ink < 0.10 | {prior['fragment_rate_ink_below_0.10']:.3f} |",
        f"| parent VT, same N(0, I) seeds, ink < 0.10 | "
        f"{normal['fragment_rate_ink_below_0.10']:.3f} |",
        f"| validation icons with ink < 0.10 | "
        f"{prior['validation_fragment_rate_ink_below_0.10']:.3f} |",
        f"| slerp interior frames with ink < 0.10 | "
        f"{interpolation['slerp']['interior_fragment_rate_ink_below_0.10']:.3f} |",
        f"| lerp interior frames with ink < 0.10 | "
        f"{interpolation['lerp']['interior_fragment_rate_ink_below_0.10']:.3f} |",
        f"| slerp endpoint pixel error (inversion round trip) | "
        f"{_interval(interpolation['slerp']['endpoint_pixel_error'])} |",
        f"| lerp endpoint pixel error (VT from mu) | "
        f"{_interval(interpolation['lerp']['endpoint_pixel_error'])} |",
        "",
        "## Resources",
        "",
        "| measure | value |",
        "| --- | --- |",
        f"| device | {resource['device']} |",
        f"| flow parameters (trained) | {record['parameters']['flow']:,} |",
        f"| served parameters (flow + frozen VT) | {record['model_parameters']:,} |",
        f"| train seconds | {resource['train_seconds']:.0f} |",
        f"| peak VRAM GiB | {resource['peak_vram_gib']} |",
        f"| flow + graph decode, ms per sample, batch 1 | {latency['end_to_end_ms_batch_1']:.1f} |",
        f"| flow alone, ms per sample, batch 1 | {latency['flow_only_ms_batch_1']:.1f} |",
        f"| batched, ms per sample | {latency['batched_ms_per_sample']:.1f} |",
        f"| torch / CUDA | {resource['torch_version']} / {resource['cuda_version']} |",
        "",
        "Sheets: `prior-samples.png`, `vt-normal-samples.png` (the parent's own prior on the",
        "same seeds), `slerp-interpolations.png` and `lerp-interpolations.png` (A, 9 frames,",
        "B), `watch-it-draw.png` (z_hat_0 at 8 flow times, then the sample).",
        "",
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--smoke", type=Path, help="a shrunk mechanics run under this directory, no registry"
    )
    defaults = SmokeOptions()
    parser.add_argument("--steps", type=int, help=f"smoke only (default {defaults.steps})")
    parser.add_argument("--batch-size", type=int, help=f"smoke only ({defaults.batch_size})")
    parser.add_argument("--layers", type=int, help=f"smoke only ({defaults.layers})")
    parser.add_argument("--variants", type=int, help=f"smoke only ({defaults.variants})")
    parser.add_argument("--train-icons", type=int, help=f"smoke only ({defaults.train_icons})")
    parser.add_argument(
        "--parent-checkpoint", type=Path, help="smoke only: another VT checkpoint (a smoke VT)"
    )
    args = parser.parse_args(argv)
    os.chdir(REPO_ROOT)
    given = {
        "steps": args.steps,
        "batch_size": args.batch_size,
        "layers": args.layers,
        "variants": args.variants,
        "train_icons": args.train_icons,
        "parent_checkpoint": (
            args.parent_checkpoint.resolve() if args.parent_checkpoint is not None else None
        ),
    }
    chosen = {key: value for key, value in given.items() if value is not None}
    if args.smoke is None:
        if chosen:
            parser.error("--steps, --batch-size, ... change a run's identity; use --smoke")
        train_and_evaluate(args.config)
        return 0
    train_and_evaluate(
        args.config, smoke=args.smoke.resolve(), options=replace(SmokeOptions(), **chosen)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
