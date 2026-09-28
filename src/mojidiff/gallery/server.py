"""Every model of this phase behind one page, to compare them by eye on the same input.

Transcribers (render-to-SVG) take a canvas - a held-out icon, painted on or not - and
return its program, greedily or best of 8 by render-and-compare. Latent models (VAEs
over programs) reconstruct a held-out icon from its posterior mean, interpolate between
two icons' posterior means, and decode draws from the prior. Every output is a decode
under the grammar; nothing is retrieved. Only held-out icons are offered: validation
and test, which no model trained on.

The pure parts - the registry, the run-record statistics, the `Gallery` that owns the
loaded models - are separate from the thin HTTP layer, so tests can build a gallery
from tiny CPU models. Bound to one explicit address, the Tailscale IP by default, like
the other tools.
"""

from __future__ import annotations

import base64
import http.server
import io
import json
import sys
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeVar, cast
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch
import yaml
from torch import Tensor

from mojidiff.learning.latent import LatentToProgram
from mojidiff.learning.render2svg import CACHE_ROOT, REPO_ROOT, RenderToProgram
from mojidiff.representation.packed import PackedTensorProgram
from mojidiff.vectorise.server import (
    CANDIDATES,
    ICON_CACHE,
    PILOT_CONFIG,
    _Server,
    canvas_to_rgb,
    held_out_icons,
)

DEFAULT_HOST = "100.69.189.78"
DEFAULT_PORT = 8791
MAX_BODY = 4 << 20
RENDER_SIZE = 144
"""Size of the held-out icon PNGs; every model of the phase reads 144 px renders."""
RUNS_ROOT = REPO_ROOT / "runs"
PAGE = Path(__file__).with_name("index.html")
BASELINE_PIXEL_ERROR = 0.090
"""Nearest training icon, pixel error on validation; lower is better."""
HELD_OUT_SPLITS = ("primary/validation", "primary/test")
ICON_FIELDS = ("hexcode", "annotation", "split", "group", "subgroup")
ICON_LIMIT = 150
INTERPOLATION_STEPS = (3, 11)
SAMPLE_COUNT = (1, 16)
SAMPLE_SCALE = (0.2, 2.0)
RERANK_TEMPERATURE = 0.7
"""Best of 8 is greedy plus seven samples at this temperature, as in the vectorise demo."""
RERANK_SEED = 0
"""The sampling seed of best of 8, `fast_decode.rerank`'s default: the same canvas always
gets the same eight candidates."""

Kind = Literal["transcriber", "latent"]
T = TypeVar("T")


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class ModelSpec:
    id: str
    run_id: str
    kind: Kind
    label: str
    description: str

    @property
    def checkpoint(self) -> Path:
        return CACHE_ROOT / "runs" / self.run_id / "best.pt"


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "v9",
        "r2s-full-v9-colour-621bc8e-b26ad95e-47646604",
        "transcriber",
        "v9 colour",
        "v7 with palette permutation 0.1 - colours fixed; best on validation",
    ),
    ModelSpec(
        "v7",
        "r2s-full-v7-systems-42762c2-e9842024-47646604",
        "transcriber",
        "v7 systems",
        "metric coordinates + path order + online variants + compositions, 60k steps; "
        "scored on test",
    ),
    ModelSpec(
        "v8",
        "r2s-full-v8-twemoji-f529d3f-5f240ad1-47646604",
        "transcriber",
        "v8 + Twemoji",
        "v7 plus 2,211 Twemoji icons - strokes and fills clash",
    ),
    ModelSpec(
        "v6",
        "r2s-full-v6-compose-7505e5b-6472dcf0-47646604",
        "transcriber",
        "v6 compositions",
        "online variants + collages of real parts; no metric, no path order",
    ),
    ModelSpec(
        "v5",
        "r2s-full-v5-path-7a72de9-12766043-47646604",
        "transcriber",
        "v5 path order",
        "metric coordinates + path-major order, 16 cached variants",
    ),
    ModelSpec(
        "v4",
        "r2s-full-v4-online-c0fe6d3-a854be2e-47646604",
        "transcriber",
        "v4 online",
        "fresh exact variant every step; no metric",
    ),
    ModelSpec(
        "v3",
        "r2s-full-v3-metric-5dba9d7-ffcc2ffb-47646604",
        "transcriber",
        "v3 metric",
        "Fourier position features shared by image and coordinates",
    ),
    ModelSpec(
        "v2",
        "r2s-full-v2-aug-4032fbd-1e601024-47646604",
        "transcriber",
        "v2 augmented",
        "16 cached exact variants: mirror, shift, recolour",
    ),
    ModelSpec(
        "v1",
        "r2s-full-v1-940f5d3-1974cf82-47646604",
        "transcriber",
        "v1 plain",
        "no augmentation - memorises, recalls near neighbours",
    ),
    ModelSpec(
        "o4",
        "r2s-overfit4-940f5d3-f7306d2b-47646604",
        "transcriber",
        "overfit fixture",
        "trained on 4 icons only - draws one of them whatever you show it",
    ),
    ModelSpec(
        "l2",
        "latent-v2-kl-efbd2e2-451268e6-47646604",
        "latent",
        "latent v2",
        "VAE, beta 0.1 - the latent carries the icon, weakly",
    ),
    ModelSpec(
        "l1",
        "latent-v1-8b0d3f9-6e874509-47646604",
        "latent",
        "latent v1",
        "VAE, beta 1 - samples, cannot reconstruct",
    ),
)


def run_record(spec: ModelSpec, runs_root: Path = RUNS_ROOT) -> dict[str, Any]:
    """The run's registry record, `runs/<run_id>/run.yaml`."""

    loaded = yaml.safe_load((runs_root / spec.run_id / "run.yaml").read_text())
    if not isinstance(loaded, dict):
        raise ValueError(f"run record is not a mapping: {spec.run_id}")
    return loaded


def _dig(value: Any, *keys: str) -> Any:
    for key in keys:
        if not isinstance(value, Mapping):
            return None
        value = value.get(key)
    return value


def _point(value: Any) -> float | None:
    """The point estimate of a `[mean, low, high]` interval, or a bare number."""

    if isinstance(value, list | tuple):
        value = value[0] if value else None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value)


def _count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def model_stats(spec: ModelSpec, record: Mapping[str, Any]) -> dict[str, Any]:
    """Headline measurements from a run record; null where the record has none.

    Scores measured on training icons (the overfit fixture) are withheld: next to
    held-out scores they would read as the best model. `evaluated_on` says which split
    the record scored, so the page can say why a value is missing.
    """

    result = record.get("result")
    result = result if isinstance(result, Mapping) else {}
    evaluated_on = _dig(record, "dataset", "evaluated_on")
    held_out = evaluated_on in HELD_OUT_SPLITS
    if spec.kind == "transcriber":
        stats: dict[str, Any] = {
            "pixel_error": _point(result.get("model_pixel_error")),
            "beats_nearest": _count(result.get("icons_model_beats_baseline")),
            "clip_top1": _point(_dig(result, "clip_retrieval", "model_greedy", "top1_rate")),
            "icons": _count(result.get("icons")),
        }
    else:
        stats = {
            "reconstruction_pixel_error": _point(result.get("reconstruction_pixel_error")),
            "prior_mean_pixel_error": _point(result.get("prior_mean_decode_pixel_error")),
            "samples_distinct": _count(_dig(result, "prior_samples", "distinct")),
        }
    if not held_out:
        stats = dict.fromkeys(stats)
    stats["evaluated_on"] = evaluated_on if isinstance(evaluated_on, str) else None
    return stats


def model_entry(spec: ModelSpec, record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "id": spec.id,
        "label": spec.label,
        "kind": spec.kind,
        "run_id": spec.run_id,
        "description": spec.description,
        "parameters": _count(record.get("model_parameters")),
        "stats": model_stats(spec, record),
    }


# --------------------------------------------------------------------------- gallery


class GalleryError(Exception):
    """A request the gallery refuses before decoding: unknown model, wrong kind of
    model, unknown icon, or an argument out of range. The HTTP layer answers 400."""


@dataclass
class _Loaded:
    spec: ModelSpec
    model: RenderToProgram
    entry: dict[str, Any]
    lock: threading.Lock
    graph: Any = None
    """Batch-1 CUDA graph decoder: greedy transcription, latent reconstruction."""
    graph_many: Any = None
    """Batch-8 CUDA graph decoder for best of 8 (transcribers only)."""


class Gallery:
    """Loaded models by id, the held-out icons and their programs, and the locks.

    Each model has its own lock, and all device work also takes one global GPU lock:
    graph replays on one device must not interleave across request threads.
    """

    def __init__(
        self,
        models: Sequence[tuple[ModelSpec, RenderToProgram]],
        template: PackedTensorProgram,
        programs: Mapping[str, Tensor],
        device: torch.device,
        *,
        icons: Sequence[Mapping[str, Any]] | None = None,
        runs_root: Path = RUNS_ROOT,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.template = template
        self.programs = dict(programs)
        self.device = device
        self.gpu_lock = threading.Lock()
        source = held_out_icons() if icons is None else icons
        self.icons = [{key: icon.get(key) for key in ICON_FIELDS} for icon in source]
        self._icon_codes = {str(icon["hexcode"]) for icon in self.icons}
        self._pngs: dict[str, bytes] = {}
        self._models: dict[str, _Loaded] = {}
        for index, (spec, model) in enumerate(models, 1):
            if spec.id in self._models:
                raise ValueError(f"duplicate model id: {spec.id}")
            if (spec.kind == "latent") != isinstance(model, LatentToProgram):
                raise ValueError(f"model {spec.id} does not match its kind {spec.kind!r}")
            try:
                record = run_record(spec, runs_root)
            except FileNotFoundError:
                record = {}
                if log is not None:
                    log(f"warning: no run record for {spec.run_id}; stats are null")
            loaded = _Loaded(spec, model.eval(), model_entry(spec, record), threading.Lock())
            if device.type == "cuda":
                from mojidiff.learning.fast_decode import GraphDecoder

                started = time.perf_counter()
                # Float32 graphs decode exactly what the evaluated batched decoder decodes.
                loaded.graph = GraphDecoder(model, dtype=torch.float32)
                if spec.kind == "transcriber":
                    loaded.graph_many = GraphDecoder(model, dtype=torch.float32, batch=CANDIDATES)
                if log is not None:
                    log(
                        f"[{index}/{len(models)}] {spec.id}: graphs captured in "
                        f"{time.perf_counter() - started:.1f} s, "
                        f"{torch.cuda.memory_allocated(device) / 2**20:.0f} MiB allocated"
                    )
            self._models[spec.id] = loaded

    @classmethod
    def from_checkpoints(
        cls,
        device: torch.device,
        specs: Sequence[ModelSpec] = MODELS,
        *,
        runs_root: Path = RUNS_ROOT,
        log: Callable[[str], None] | None = None,
    ) -> Gallery:
        """Load every listed model's best checkpoint; refuse to start if one is missing."""

        from mojidiff.learning.latent import LatentSettings
        from mojidiff.learning.openmoji_pilot import _load_program, load_pilot_index
        from mojidiff.learning.render2svg import ModelConfig, load_corpus, parameter_count

        missing = [str(spec.checkpoint) for spec in specs if not spec.checkpoint.is_file()]
        if missing:
            raise FileNotFoundError("missing checkpoints: " + ", ".join(missing))
        say = log if log is not None else (lambda _: None)
        started = time.perf_counter()
        plain, _, layout, pilot, dataset_hash = load_corpus(PILOT_CONFIG, RENDER_SIZE)
        programs = {
            hexcode: plain[split].tokens[row]
            for split in HELD_OUT_SPLITS
            for row, hexcode in enumerate(plain[split].hexcodes)
        }
        del plain
        by_split, _, _ = load_pilot_index(pilot)
        template = _load_program(by_split["primary/train"][0], pilot, layout.codec)
        say(
            f"corpus {dataset_hash[:10]}: {len(programs)} held-out programs "
            f"in {time.perf_counter() - started:.1f} s"
        )
        models: list[tuple[ModelSpec, RenderToProgram]] = []
        for index, spec in enumerate(specs, 1):
            started = time.perf_counter()
            # Loaded on the CPU and moved once, so no second copy of the weights sits on
            # the device while the next model loads.
            state = torch.load(spec.checkpoint, map_location="cpu")
            config = ModelConfig(**state["config"])
            model: RenderToProgram
            if spec.kind == "latent":
                model = LatentToProgram(layout, config, LatentSettings(**state["latent"]))
            else:
                model = RenderToProgram(layout, config)
            model.load_state_dict(state["model"])
            model = model.to(device)
            count = parameter_count(model)
            recorded = run_record(spec, runs_root).get("model_parameters")
            if recorded != count:
                say(f"warning: {spec.id} has {count:,} parameters, its run record {recorded}")
            say(
                f"[{index}/{len(specs)}] {spec.id} {spec.run_id}: step {state['step']}, "
                f"{count:,} parameters, {time.perf_counter() - started:.1f} s"
            )
            models.append((spec, model))
        gallery = cls(models, template, programs, device, runs_root=runs_root, log=log)
        unmatched = gallery._icon_codes - set(programs)
        if unmatched:
            say(f"warning: {len(unmatched)} held-out icons have no program for latent models")
        return gallery

    def __len__(self) -> int:
        return len(self._models)

    # ------------------------------------------------------------------ lookups

    def _get(self, model_id: str, kind: Kind) -> _Loaded:
        loaded = self._models.get(model_id)
        if loaded is None:
            raise GalleryError(f"unknown model: {model_id!r}")
        if loaded.spec.kind != kind:
            raise GalleryError(f"model {model_id!r} is a {loaded.spec.kind}, not a {kind}")
        return loaded

    def _program(self, hexcode: str) -> Tensor:
        tokens = self.programs.get(hexcode)
        if tokens is None:
            raise GalleryError(f"unknown held-out icon: {hexcode!r}")
        return tokens

    def models_payload(self) -> dict[str, Any]:
        return {
            "models": [dict(loaded.entry) for loaded in self._models.values()],
            "baseline_pixel_error": BASELINE_PIXEL_ERROR,
        }

    def icons_matching(self, query: str, limit: int = ICON_LIMIT) -> dict[str, Any]:
        """Held-out icons whose annotation, hexcode, group or subgroup contain `query`:
        the first `limit` of them (clamped to 1..all icons) and how many match in all."""

        limit = max(1, min(limit, len(self.icons)))
        needle = query.strip().lower()
        found = [
            icon
            for icon in self.icons
            if not needle
            or any(needle in str(icon[key] or "").lower() for key in ICON_FIELDS if key != "split")
        ]
        return {"icons": found[:limit], "total": len(found)}

    def icon_png(self, hexcode: str) -> bytes:
        """A held-out icon at `RENDER_SIZE`, rendered as the models saw icons in training."""

        from PIL import Image

        from mojidiff.learning.render2svg import render_trusted_rgb

        if hexcode not in self._icon_codes:
            raise GalleryError(f"not a held-out icon: {hexcode!r}")
        cached = self._pngs.get(hexcode)
        if cached is None:
            svg = (ICON_CACHE / f"{hexcode}.svg").read_bytes()
            buffer = io.BytesIO()
            Image.fromarray(render_trusted_rgb(svg, RENDER_SIZE)).save(buffer, format="PNG")
            cached = self._pngs[hexcode] = buffer.getvalue()
        return cached

    def input_size(self, model_id: str) -> int:
        """The square input size of a transcriber, for decoding a canvas PNG."""

        return int(self._get(model_id, "transcriber").model.config.image_size)

    # ------------------------------------------------------------------ decoding

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _run(self, loaded: _Loaded, work: Callable[[], T]) -> tuple[T, float]:
        """Run `work` under the model's lock and the GPU lock; model time in ms."""

        with loaded.lock, self.gpu_lock, torch.no_grad():
            self._synchronize()
            started = time.perf_counter()
            value = work()
            self._synchronize()
            elapsed = (time.perf_counter() - started) * 1000.0
        return value, round(elapsed, 1)

    def _svg(self, loaded: _Loaded, tokens: Tensor) -> tuple[str, PackedTensorProgram]:
        from mojidiff.learning.autoregressive import unflatten_program
        from mojidiff.representation.packed import serialize_packed_svg

        layout = loaded.model.layout
        program = unflatten_program(tokens.cpu(), self.template, layout)
        svg = serialize_packed_svg(program, layout.codec, layout.total_segment_slots)
        return svg.decode(), program

    def _choose(self, loaded: _Loaded, tokens: Tensor, target: np.ndarray) -> int:
        """The candidate that redraws `target` best: render-and-compare on the CPU, as
        `fast_decode.rerank` scores it. Runs outside every lock; it needs no device."""

        from concurrent.futures import ThreadPoolExecutor

        from mojidiff.learning.autoregressive import unflatten_program
        from mojidiff.learning.render2svg import pixel_error, render_trusted_rgb
        from mojidiff.representation.packed import serialize_packed_svg
        from mojidiff.representation.renderer import IsolatedRenderError

        layout = loaded.model.layout
        size = int(target.shape[0])

        def score(row: Tensor) -> float:
            program = unflatten_program(row, self.template, layout)
            try:
                svg = serialize_packed_svg(program, layout.codec, layout.total_segment_slots)
                return pixel_error(render_trusted_rgb(svg, size), target)
            except (ValueError, IsolatedRenderError):
                return 1.0

        with ThreadPoolExecutor(max_workers=len(tokens)) as pool:
            errors = list(pool.map(score, tokens))
        return int(np.argmin(errors))

    def transcribe(self, model_id: str, rgb: np.ndarray, candidates: int = 1) -> dict[str, Any]:
        """Greedy transcription, or best of `CANDIDATES` when `candidates` > 1.

        `ms` is model time only: for best of 8, decoding the eight candidates. Choosing
        among them renders each on the CPU after the locks are released, untimed, so it
        neither inflates the model's time nor holds the GPU from other requests.
        """

        from mojidiff.learning.render2svg import DecodeStats, greedy_decode

        loaded = self._get(model_id, "transcriber")
        size = int(loaded.model.config.image_size)
        if rgb.shape != (size, size, 3) or rgb.dtype != np.uint8:
            raise GalleryError(f"expected a {size}x{size} RGB uint8 image, got {rgb.shape}")
        if candidates < 1:
            raise GalleryError("candidates must be 1 (greedy) or more (best of 8)")
        best = candidates > 1
        image = torch.from_numpy(np.array(rgb, copy=True))  # canvas arrays are read-only
        stats = DecodeStats()
        model = loaded.model

        def work() -> Tensor:
            on_device = image.to(self.device)
            if best and loaded.graph_many is not None:
                # Row 0 greedy, the rest sampled: `fast_decode.rerank`'s candidates.
                return cast(
                    Tensor,
                    loaded.graph_many.decode_many(
                        on_device,
                        temperature=RERANK_TEMPERATURE,
                        generator=torch.Generator().manual_seed(RERANK_SEED),
                        stats=stats,
                    ),
                )
            if best:
                # `render2svg.rerank_decode`'s candidates, without its scoring.
                images = on_device[None]
                greedy = greedy_decode(model, images, stats=stats)
                generator = torch.Generator(device=on_device.device).manual_seed(RERANK_SEED)
                sampled = greedy_decode(
                    model,
                    images.expand(CANDIDATES - 1, -1, -1, -1).contiguous(),
                    temperature=RERANK_TEMPERATURE,
                    generator=generator,
                    stats=stats,
                )
                return torch.cat((greedy, sampled))
            if loaded.graph is not None:
                return cast(Tensor, loaded.graph.decode(on_device, stats=stats))
            return greedy_decode(model, on_device[None], stats=stats)

        tokens, ms = self._run(loaded, work)
        chosen = self._choose(loaded, tokens, rgb) if best else 0
        svg, program = self._svg(loaded, tokens[chosen])
        lengths = [int(v) for v in program.path_length if int(v) > 0]
        return {
            "ok": True,
            "model": model_id,
            "svg": svg,
            "ms": ms,
            "decoder_calls": stats.model_calls,
            "paths": len(lengths),
            "segments": sum(lengths),
            "candidates": CANDIDATES if best else 1,
        }

    def _latent(self, model_id: str) -> tuple[_Loaded, LatentToProgram]:
        loaded = self._get(model_id, "latent")
        return loaded, cast(LatentToProgram, loaded.model)

    def reconstruct(self, model_id: str, hexcode: str) -> dict[str, Any]:
        """The posterior mean of a held-out icon's program, decoded greedily."""

        from mojidiff.learning.render2svg import greedy_decode

        loaded, model = self._latent(model_id)
        tokens = self._program(hexcode)[None]

        def work() -> Tensor:
            mean, _ = model.posterior(tokens.to(self.device))
            if loaded.graph is not None:
                return cast(Tensor, loaded.graph.decode(mean[0]))
            return greedy_decode(model, mean)

        decoded, ms = self._run(loaded, work)
        return {"ok": True, "svg": self._svg(loaded, decoded[0])[0], "ms": ms}

    def interpolate(self, model_id: str, a: str, b: str, steps: int) -> dict[str, Any]:
        """`steps` frames on the straight line between two posterior means, one batch."""

        from mojidiff.learning.render2svg import greedy_decode

        loaded, model = self._latent(model_id)
        low, high = INTERPOLATION_STEPS
        if not low <= steps <= high:
            raise GalleryError(f"steps must be {low}..{high}, got {steps}")
        tokens = torch.stack((self._program(a), self._program(b)))

        def work() -> Tensor:
            means, _ = model.posterior(tokens.to(self.device))
            weights = torch.linspace(0.0, 1.0, steps, device=self.device)[:, None]
            return greedy_decode(model, (1.0 - weights) * means[0] + weights * means[1])

        decoded, ms = self._run(loaded, work)
        frames = [self._svg(loaded, row)[0] for row in decoded]
        return {"ok": True, "frames": frames, "ms": ms}

    def sample(self, model_id: str, count: int, seed: int, scale: float) -> dict[str, Any]:
        """`count` prior draws z = scale * N(0, I), seeded on the CPU, one batch."""

        from mojidiff.learning.render2svg import greedy_decode

        loaded, model = self._latent(model_id)
        low, high = SAMPLE_COUNT
        if not low <= count <= high:
            raise GalleryError(f"count must be {low}..{high}, got {count}")
        if not SAMPLE_SCALE[0] <= scale <= SAMPLE_SCALE[1]:
            raise GalleryError(f"scale must be {SAMPLE_SCALE[0]}..{SAMPLE_SCALE[1]}, got {scale}")
        if not 0 <= seed < 2**63:
            raise GalleryError(f"seed must be a non-negative 63-bit integer, got {seed}")
        generator = torch.Generator().manual_seed(seed)
        latents = scale * torch.randn(count, model.settings.latent_dim, generator=generator)

        def work() -> Tensor:
            return greedy_decode(model, latents.to(self.device))

        decoded, ms = self._run(loaded, work)
        samples = [self._svg(loaded, row)[0] for row in decoded]
        return {"ok": True, "samples": samples, "ms": ms}


# --------------------------------------------------------------------------- HTTP


def _text(payload: Mapping[str, Any], key: str) -> str:
    value = payload[key]
    if not isinstance(value, str):
        raise TypeError(f"{key} must be a string")
    return value


def _integer(payload: Mapping[str, Any], key: str, default: int | None = None) -> int:
    value = payload.get(key, default) if default is not None else payload[key]
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{key} must be an integer")
    return value


def _number(payload: Mapping[str, Any], key: str) -> float:
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{key} must be a number")
    return float(value)


Work = Callable[[], dict[str, Any]]


def _transcribe(gallery: Gallery, payload: Mapping[str, Any]) -> Work:
    model_id = _text(payload, "model")
    candidates = _integer(payload, "candidates", 1)
    size = gallery.input_size(model_id)
    data = _text(payload, "png")
    png = base64.b64decode(data.split(",", 1)[1] if "," in data else data)
    rgb = canvas_to_rgb(png, size)
    return lambda: gallery.transcribe(model_id, rgb, candidates)


def _reconstruct(gallery: Gallery, payload: Mapping[str, Any]) -> Work:
    model_id, hexcode = _text(payload, "model"), _text(payload, "hexcode")
    return lambda: gallery.reconstruct(model_id, hexcode)


def _interpolate(gallery: Gallery, payload: Mapping[str, Any]) -> Work:
    model_id, a, b = _text(payload, "model"), _text(payload, "a"), _text(payload, "b")
    steps = _integer(payload, "steps")
    return lambda: gallery.interpolate(model_id, a, b, steps)


def _sample(gallery: Gallery, payload: Mapping[str, Any]) -> Work:
    model_id = _text(payload, "model")
    count, seed = _integer(payload, "count"), _integer(payload, "seed")
    scale = _number(payload, "scale")
    return lambda: gallery.sample(model_id, count, seed, scale)


ROUTES: dict[str, Callable[[Gallery, Mapping[str, Any]], Work]] = {
    "/transcribe": _transcribe,
    "/latent/reconstruct": _reconstruct,
    "/latent/interpolate": _interpolate,
    "/latent/sample": _sample,
}


def _refusal(error: Exception) -> str:
    if isinstance(error, KeyError):
        return f"missing field {error}"
    return str(error)[:200] or type(error).__name__


class _Handler(http.server.BaseHTTPRequestHandler):
    gallery: Gallery
    timeout = 60

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        sys.stderr.write(f"{self.address_string()} {format % args}\n")

    def _send(self, status: int, body: bytes, content_type: str, cache: str = "no-store") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, payload: object) -> None:
        self._send(status, json.dumps(payload).encode(), "application/json")

    def do_GET(self) -> None:  # noqa: N802
        url = urlparse(self.path)
        if url.path in ("/", "/index.html"):
            if not PAGE.is_file():
                self._json(503, {"ok": False, "error": "the gallery page is missing"})
                return
            self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        elif url.path == "/health":
            self._json(200, {"ok": True, "models": len(self.gallery)})
        elif url.path == "/models":
            self._json(200, self.gallery.models_payload())
        elif url.path == "/icons":
            fields = parse_qs(url.query)
            query = fields.get("q", [""])[0]
            try:
                limit = int(fields.get("limit", [str(ICON_LIMIT)])[0])
            except ValueError:
                self._json(400, {"ok": False, "error": "limit must be an integer"})
                return
            self._json(200, self.gallery.icons_matching(query, limit))
        elif url.path.startswith("/render/") and url.path.endswith(".png"):
            hexcode = url.path[len("/render/") : -len(".png")]
            try:
                png = self.gallery.icon_png(hexcode)
            except GalleryError as error:
                self._json(404, {"ok": False, "error": str(error)})
                return
            self._send(200, png, "image/png", "max-age=86400")
        else:
            self._json(404, {"ok": False, "error": "not found"})

    def do_POST(self) -> None:  # noqa: N802
        route = ROUTES.get(urlparse(self.path).path)
        if route is None:
            self._json(404, {"ok": False, "error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._json(400, {"ok": False, "error": "bad Content-Length"})
            return
        if length > MAX_BODY:
            self._json(413, {"ok": False, "error": "request too large"})
            return
        if length <= 0:
            self._json(400, {"ok": False, "error": "empty request"})
            return
        try:
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise TypeError("the request must be a JSON object")
            work = route(self.gallery, payload)
        except Exception as error:  # noqa: BLE001 - any unreadable request is the client's
            self._json(400, {"ok": False, "error": _refusal(error)})
            return
        try:
            result = work()
        except GalleryError as error:
            self._json(400, {"ok": False, "error": _refusal(error)})
            return
        except Exception as error:  # noqa: BLE001 - report, never crash the server
            self._json(200, {"ok": False, "error": f"{type(error).__name__}: {str(error)[:200]}"})
            return
        self._json(200, result)


def make_server(gallery: Gallery, host: str, port: int) -> _Server:
    """A threading HTTP server over `gallery`; the caller runs and closes it."""

    handler = type("Handler", (_Handler,), {"gallery": gallery})
    return _Server((host, port), handler)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def serve(host: str, port: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    gallery = Gallery.from_checkpoints(device, log=_log)
    with make_server(gallery, host, port) as server:
        _log(
            f"gallery on http://{host}:{port} - {len(gallery)} models, "
            f"{len(gallery.icons)} held-out icons, ready in {time.perf_counter() - started:.0f} s"
        )
        server.serve_forever()
