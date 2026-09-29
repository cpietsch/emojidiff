"""What the gallery needs from a latent model, and how each kind of latent model loads.

The gallery shows every latent model three ways - a held-out icon reconstructed from
its posterior mean, the straight line between two icons' posterior means, and draws from
the prior - and asks the same five things of each, whatever its latent is (one vector,
a spatial grid, one vector per path):

* `model` - the RenderToProgram-compatible decoder module. The gallery reads its
  `layout` to serialize programs; its graph and greedy decoding go through the backend.
* `latent_shape` - the shape of one latent, without the batch dimension.
* `encode(tokens, images)` - the posterior mean, (B, *latent_shape), of a batch of
  held-out icons. It is given both the programs, (B, 1376) long, and their 144 px
  renders, (B, 144, 144, 3) uint8 - the renders the models trained on - both on the
  device; a backend reads whichever it needs.
* `decode(latents)` - greedy programs, (B, 1376) long, for a batch of latents.
* `sample(count, generator, scale)` - `count` draws from the backend's prior,
  (count, *latent_shape), on the CPU. `generator` is a seeded CPU generator, so one seed
  always draws the same latents. A Gaussian prior returns `scale` * N(0, I); a learned
  prior (a flow, say) applies `scale` to the noise it starts from.

and optionally

* `decode_one(latent)` - one latent, (*latent_shape), to one program, (1376,), by a
  CUDA graph when the backend captured one. Reconstruction uses it when it exists and
  `decode` otherwise.
* `interpolate(first, second, steps)` - `steps` latents, (steps, *latent_shape), on the
  device, from `first` to `second` (two posterior means on the device), with
  `interpolation_path`, a short name for that path that the gallery's answers and the
  harness's reports carry. Without it the gallery interpolates linearly in latent space
  ("lerp"). The gallery takes the backend's path unless a request asks for `path:
  "lerp"`; either way all frames decode in one batch. The metrics harness
  (`latent_metrics`) scores the linear path unless it runs with `--interpolation
  backend`, and records which path it scored.
* `parts` - every module the backend runs, when that is more than `model` (a prior over
  a frozen decoder, say): the gallery counts their parameters against the run record's
  `model_parameters`.

The gallery runs `encode`, `decode`, `decode_one`, `interpolate` and `sample` under its
global GPU lock, so a backend may use the device in any of them.

Precision. Both backends encode and decode in IEEE float32 (`precision.strict_float32`,
TF32 off) and capture their CUDA graphs that way: with TF32 on, a latent's greedy decode
often depends on the batch it sits in, so the same latent would decode differently in a
sample grid, an interpolation strip, a single reconstruction and the metrics harness.
IEEE float32 rounding can still differ between batch shapes, far more rarely. A new
backend should do the same.

Adding a backend
----------------
1. Write a class with the members above.
2. Write a factory `(checkpoint_state, layout, device) -> backend`. `checkpoint_state`
   is the model's `best.pt` loaded on the CPU (`torch.load(..., map_location="cpu")`);
   build the modules from it, load their weights, move them to `device` once, and
   capture any CUDA graphs there. A checkpoint that needs another run's weights (a prior
   over a frozen model) records their path in its own state and loads them here.
3. Add the factory to `BACKENDS` under a new name.
4. Add the model to `configs/gallery/models.yaml` with `kind: latent` and
   `backend: <name>`, and restart the gallery. Nothing in the server changes.

An unknown backend name, or a missing checkpoint, stops the gallery at startup.

Backends
--------
* `vae` - `LatentToProgram` (latent v1, v2): a program encoder's posterior mean, one
  64-number vector, the N(0, I) prior.
* `canvas` - `pixel_latent.VariationalTranscriber` (vt): the posterior mean of the
  icon's render, a (c, 18, 18) grid, and N(0, I) - the prior-hole control until a
  learned prior over the grid exists (a flow prior is a backend of its own).
* `canvas-flow` - `latent_flow.LatentFlowPrior` (lfp) over a frozen VT, whose path the
  flow checkpoint records: the VT's encoder and decoder, the flow as the prior (`scale`
  multiplies the noise it starts from), `interpolate` as slerp through the prior's
  noise ("slerp-through-prior-noise"), and `parts` the VT and the flow.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast, runtime_checkable

import torch
from torch import Tensor, nn

from mojidiff.learning.autoregressive import SequenceLayout
from mojidiff.learning.latent import LatentSettings, LatentToProgram
from mojidiff.learning.latent_flow import (
    LatentFlowPrior,
    load_flow_checkpoint,
    load_flow_parent,
    trajectory_indices,
)
from mojidiff.learning.pixel_latent import VariationalTranscriber, load_checkpoint
from mojidiff.learning.precision import strict_float32
from mojidiff.learning.render2svg import ModelConfig, RenderToProgram, greedy_decode


@runtime_checkable
class LatentBackend(Protocol):
    """A latent model as the gallery serves it; see the module docstring."""

    @property
    def model(self) -> RenderToProgram: ...

    @property
    def latent_shape(self) -> tuple[int, ...]: ...

    def encode(self, tokens: Tensor, images: Tensor) -> Tensor: ...

    def decode(self, latents: Tensor) -> Tensor: ...

    def sample(self, count: int, generator: torch.Generator, scale: float) -> Tensor: ...


BackendFactory = Callable[[Mapping[str, Any], SequenceLayout, torch.device], LatentBackend]


def decode_one(backend: LatentBackend, latent: Tensor) -> Tensor:
    """One latent to one program: the backend's own `decode_one` (a CUDA graph) when it
    has one, else its batched `decode` on a batch of one."""

    single = getattr(backend, "decode_one", None)
    if single is None:
        return backend.decode(latent[None])[0]
    return cast(Tensor, single(latent))


# --------------------------------------------------------------------------- vae


def vae_model(state: Mapping[str, Any], layout: SequenceLayout) -> LatentToProgram:
    """A `LatentToProgram` with its checkpoint's weights, on the CPU."""

    model = LatentToProgram(
        layout, ModelConfig(**state["config"]), LatentSettings(**state["latent"])
    )
    model.load_state_dict(state["model"])
    return model.eval()


class VaeBackend:
    """`LatentToProgram` (latent v1, v2): a program encoder's posterior mean, a
    vector latent, the N(0, I) prior, and an IEEE float32 CUDA graph for single
    decodes; everything in IEEE float32 (TF32 off)."""

    def __init__(self, model: LatentToProgram, *, graphs: bool = True) -> None:
        self.model = model.eval()
        self.latent_shape: tuple[int, ...] = (model.settings.latent_dim,)
        self.graph: Any = None
        if graphs and next(model.parameters()).device.type == "cuda":
            from mojidiff.learning.fast_decode import GraphDecoder

            # Captured with TF32 off, as the batched decoder runs (see Precision above).
            with strict_float32():
                self.graph = GraphDecoder(model, dtype=torch.float32)

    def encode(self, tokens: Tensor, images: Tensor) -> Tensor:
        with strict_float32():
            mean, _ = self.model.posterior(tokens)
        return mean

    def decode(self, latents: Tensor) -> Tensor:
        with strict_float32():
            return greedy_decode(self.model, latents)

    def decode_one(self, latent: Tensor) -> Tensor:
        with strict_float32():
            if self.graph is None:
                return greedy_decode(self.model, latent[None])[0]
            return cast(Tensor, self.graph.decode(latent))[0]

    def sample(self, count: int, generator: torch.Generator, scale: float) -> Tensor:
        return scale * torch.randn(count, *self.latent_shape, generator=generator)


def _vae(state: Mapping[str, Any], layout: SequenceLayout, device: torch.device) -> VaeBackend:
    # Loaded on the CPU and moved once, so no second copy of the weights sits on the
    # device while the next model loads.
    return VaeBackend(vae_model(state, layout).to(device))


# --------------------------------------------------------------------------- canvas


class CanvasBackend:
    """`VariationalTranscriber` (vt): an icon's render encodes to its posterior mean, a
    (c, 18, 18) latent grid; the prior is N(0, I), the prior-hole control until a
    learned prior exists; single decodes go through an IEEE float32 CUDA graph;
    everything in IEEE float32 (TF32 off), as the run's own evaluation decodes."""

    def __init__(self, model: VariationalTranscriber, *, graphs: bool = True) -> None:
        self.model = model.eval()
        self.latent_shape: tuple[int, ...] = model.latent_shape
        self.graph: Any = None
        if graphs and next(model.parameters()).device.type == "cuda":
            from mojidiff.learning.fast_decode import GraphDecoder

            # Captured with TF32 off, as the batched decoder runs (see Precision above).
            with strict_float32():
                self.graph = GraphDecoder(model, dtype=torch.float32)

    def encode(self, tokens: Tensor, images: Tensor) -> Tensor:
        with strict_float32():
            mean, _ = self.model.posterior(images)
        return mean

    def decode(self, latents: Tensor) -> Tensor:
        with strict_float32():
            return greedy_decode(self.model, latents)

    def decode_one(self, latent: Tensor) -> Tensor:
        with strict_float32():
            if self.graph is None:
                return greedy_decode(self.model, latent[None])[0]
            return cast(Tensor, self.graph.decode(latent))[0]

    def sample(self, count: int, generator: torch.Generator, scale: float) -> Tensor:
        return scale * torch.randn(count, *self.latent_shape, generator=generator)


def _canvas(
    state: Mapping[str, Any], layout: SequenceLayout, device: torch.device
) -> CanvasBackend:
    return CanvasBackend(load_checkpoint(state, layout).to(device))


# --------------------------------------------------------------------------- canvas-flow


class CanvasFlowBackend:
    """A rectified-flow prior (`latent_flow`, lfp) over a frozen VT. Encoding, decoding
    and single decodes are the VT's (`CanvasBackend`); the prior is the flow, Euler
    steps from `scale` * N(0, I) in its standardised space to z_hat_0. `interpolate`
    goes through the prior's noise: both posterior means are inverted by the reverse
    ODE, their noises slerped, and every frame integrated forward, so each frame is a
    prior sample. Everything in IEEE float32 (TF32 off)."""

    interpolation_path = "slerp-through-prior-noise"

    def __init__(
        self, vt: VariationalTranscriber, prior: LatentFlowPrior, *, graphs: bool = True
    ) -> None:
        self.canvas = CanvasBackend(vt, graphs=graphs)
        self.model = self.canvas.model
        self.latent_shape: tuple[int, ...] = self.canvas.latent_shape
        if tuple(prior.latent_shape) != self.latent_shape:
            raise ValueError(f"a flow over {prior.latent_shape} for a VT of {self.latent_shape}")
        self.prior = prior.eval()
        self.device = next(vt.parameters()).device

    @property
    def parts(self) -> tuple[nn.Module, ...]:
        """The VT and the flow: the sampler as served, parameters counted together."""

        return (self.canvas.model, self.prior)

    def encode(self, tokens: Tensor, images: Tensor) -> Tensor:
        return self.canvas.encode(tokens, images)

    def decode(self, latents: Tensor) -> Tensor:
        return self.canvas.decode(latents)

    def decode_one(self, latent: Tensor) -> Tensor:
        return self.canvas.decode_one(latent)

    def sample(self, count: int, generator: torch.Generator, scale: float) -> Tensor:
        noise = scale * torch.randn(count, *self.latent_shape, generator=generator)
        with strict_float32():
            latents, _ = self.prior.generate(noise.to(self.device))
        return latents.cpu()

    def interpolate(self, first: Tensor, second: Tensor, steps: int) -> Tensor:
        with strict_float32():
            return self.prior.slerp_path(first.to(self.device), second.to(self.device), steps)

    def trajectory(self, generator: torch.Generator, scale: float, frames: int = 8) -> Tensor:
        """z_hat_0 at `frames` times of one trajectory (`trajectory_indices`: t = 1 down to
        2 / steps), then the sample ("watch it draw"), (frames + 1, *latent_shape) on the
        CPU, fewer frames when the sampler has fewer steps. The page does not show it
        yet."""

        noise = scale * torch.randn(1, *self.latent_shape, generator=generator)
        indices = trajectory_indices(self.prior.steps, frames)
        with strict_float32():
            final, recorded = self.prior.generate(noise.to(self.device), record_at=indices)
        return torch.cat([frame for _, frame in recorded] + [final]).cpu()


def _canvas_flow(
    state: Mapping[str, Any], layout: SequenceLayout, device: torch.device
) -> CanvasFlowBackend:
    # The VT the flow was trained over, from the path and sha256 its checkpoint records.
    vt = load_flow_parent(state, layout).to(device)
    return CanvasFlowBackend(vt, load_flow_checkpoint(state).to(device))


BACKENDS: dict[str, BackendFactory] = {
    "vae": _vae,
    "canvas": _canvas,
    "canvas-flow": _canvas_flow,
}
"""Backend name, as `backend:` in the registry file, to its factory."""
