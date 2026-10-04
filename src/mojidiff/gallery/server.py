"""Every model of this phase behind one page, to compare them by eye on the same input.

Transcribers (render-to-SVG) take a canvas - a held-out icon, painted on or not - and
return its program, greedily or best of 8 by render-and-compare. Latent models
reconstruct a held-out icon from its posterior mean, interpolate between two icons'
posterior means, and decode draws from the prior. Every output is a decode under the
grammar; nothing is retrieved. Only held-out icons are offered: validation and test,
which no model trained on.

Which models load is the registry file, `configs/gallery/models.yaml` (`MODELS` below
is its fallback when the file is absent). A latent model names a backend there - how
its checkpoint loads, encodes, decodes and samples (`latent_backends`) - so a new kind
of latent model joins the gallery by one backend entry and one registry entry, with no
change here. The operator's ratings of latent models are appended, one JSON line each,
to `reports/gallery/ratings.jsonl`, which is never rewritten.

The pure parts - the registry, the run-record statistics, the `Gallery` that owns the
loaded models, the rating log - are separate from the thin HTTP layer, so tests can
build a gallery from tiny CPU models. Bound to one explicit address, the Tailscale IP by
default, like the other tools.
"""

from __future__ import annotations

import base64
import http.server
import io
import json
import math
import os
import re
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

from mojidiff.gallery.latent_backends import BACKENDS, LatentBackend, VaeBackend, decode_one
from mojidiff.learning.autoregressive import SequenceLayout
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
REGISTRY = REPO_ROOT / "configs" / "gallery" / "models.yaml"
"""The models the gallery loads; `serve_gallery.py --registry` names another file."""
RATINGS = REPO_ROOT / "reports" / "gallery" / "ratings.jsonl"
"""The operator's ratings, appended one JSON line each; `--ratings` names another file."""
BASELINE_PIXEL_ERROR = 0.090
"""Nearest training icon, pixel error on validation; lower is better."""
HELD_OUT_SPLITS = ("primary/validation", "primary/test")
ICON_FIELDS = ("hexcode", "annotation", "split", "group", "subgroup")
ICON_LIMIT = 150
INTERPOLATION_STEPS = (3, 11)
INTERPOLATION_PATHS = ("backend", "lerp")
"""`backend`: the backend's own path when it has one (canvas-flow: slerp through its
prior's noise), else the straight line; `lerp`: always the straight line between the two
posterior means."""
SAMPLE_COUNT = (1, 16)
SAMPLE_SCALE = (0.2, 2.0)
RERANK_TEMPERATURE = 0.7
"""Best of 8 is greedy plus seven samples at this temperature, as in the vectorise demo."""
RERANK_SEED = 0
"""The sampling seed of best of 8, `fast_decode.rerank`'s default: the same canvas always
gets the same eight candidates."""
RATING_VIEWS = ("samples", "interpolation", "reconstruction")
RATING_SCORES = (1, 5)
RATING_NOTE_LIMIT = 500
RATING_FIELDS = ("model", "view", "score", "blind", "seed", "scale", "a", "b", "steps", "note")
"""A rating's fields as posted; the server adds `at` (UTC) and the model's `run_id`."""

Kind = Literal["transcriber", "latent"]
KINDS: tuple[Kind, ...] = ("transcriber", "latent")
MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,31}")
T = TypeVar("T")


# --------------------------------------------------------------------------- registry


@dataclass(frozen=True)
class ModelSpec:
    id: str
    run_id: str
    kind: Kind
    label: str
    description: str
    backend: str | None = None
    """A latent model's backend, a key of `latent_backends.BACKENDS`; None for a
    transcriber."""

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"model {self.id!r}: kind must be one of {KINDS}, got {self.kind!r}")
        if self.kind == "latent" and not self.backend:
            raise ValueError(f"latent model {self.id!r} names no backend")
        if self.kind == "transcriber" and self.backend is not None:
            raise ValueError(f"transcriber {self.id!r} takes no backend")

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
        "vae",
    ),
    ModelSpec(
        "l1",
        "latent-v1-8b0d3f9-6e874509-47646604",
        "latent",
        "latent v1",
        "VAE, beta 1 - samples, cannot reconstruct",
        "vae",
    ),
)
"""The twelve models of the registry file as first written: its fallback when absent."""

_ENTRY_FIELDS = {"id", "run_id", "kind", "label", "description", "backend"}


def load_registry(path: Path = REGISTRY) -> tuple[ModelSpec, ...]:
    """The models a registry file lists, in display order; refuse a malformed one.

    The file is `{schema_version: 1, models: [...]}`; each entry has `id`, `run_id`,
    `kind` (transcriber or latent), `label`, `description`, and for a latent model
    `backend`, a name in `latent_backends.BACKENDS`. Unknown or missing fields,
    duplicate ids and unknown backends are errors: the gallery does not start.
    """

    root = yaml.safe_load(path.read_text())
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise ValueError(f"{path}: a gallery registry needs schema_version: 1")
    entries = root.get("models")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{path}: `models` must be a non-empty list")
    specs: list[ModelSpec] = []
    for number, entry in enumerate(entries, 1):
        where = f"{path}: model {number}"
        if not isinstance(entry, dict):
            raise ValueError(f"{where} is not a mapping")
        unknown = sorted(str(key) for key in set(entry) - _ENTRY_FIELDS)
        if unknown:
            raise ValueError(f"{where}: unknown field(s) {', '.join(map(str, unknown))}")
        missing = sorted(_ENTRY_FIELDS - {"backend"} - set(entry))
        if missing:
            raise ValueError(f"{where}: missing field(s) {', '.join(missing)}")
        values = {key: entry[key] for key in _ENTRY_FIELDS - {"backend"}}
        for key, value in values.items():
            if not isinstance(value, str) or (key != "description" and not value.strip()):
                raise ValueError(f"{where}: {key} must be a non-empty string")
        if not MODEL_ID.fullmatch(values["id"]):
            raise ValueError(f"{where}: id {values['id']!r} must match {MODEL_ID.pattern}")
        backend = entry.get("backend")
        if backend is not None and (not isinstance(backend, str) or backend not in BACKENDS):
            known = ", ".join(sorted(BACKENDS))
            raise ValueError(f"{where}: unknown backend {backend!r} (known: {known})")
        try:
            spec = ModelSpec(
                id=values["id"],
                run_id=values["run_id"],
                kind=cast(Kind, values["kind"]),
                label=values["label"],
                description=values["description"],
                backend=backend,
            )
        except ValueError as error:
            raise ValueError(f"{where}: {error}") from None
        if spec.id in {earlier.id for earlier in specs}:
            raise ValueError(f"{where}: duplicate id {spec.id!r}")
        specs.append(spec)
    return tuple(specs)


def registry_models(
    path: Path | None = None, log: Callable[[str], None] | None = None
) -> tuple[ModelSpec, ...]:
    """The registry file's models: `path`, or the default file, or `MODELS` when the
    default file is absent. A path given explicitly must exist."""

    if path is None:
        if not REGISTRY.is_file():
            if log is not None:
                log(f"warning: no registry at {REGISTRY}; serving the built-in MODELS")
            return MODELS
        path = REGISTRY
    return load_registry(path)


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


# --------------------------------------------------------------------------- ratings


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class RatingLog:
    """The operator's ratings: one JSON object per line, only ever appended.

    Appends take a lock, so concurrent requests never interleave within a line; a file
    left without a final newline (a write cut short) gets one before the next row, and
    rows that do not parse are skipped when read, never repaired in place.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.lock = threading.Lock()

    def append(self, row: Mapping[str, Any]) -> int:
        """Append `row`; the number of rows the file holds afterwards."""

        line = (json.dumps(row) + "\n").encode()
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a+b") as handle:
                size = handle.seek(0, os.SEEK_END)
                if size:
                    handle.seek(size - 1)
                    if handle.read(1) != b"\n":
                        line = b"\n" + line
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
            return len(self._read())

    def rows(self) -> list[dict[str, Any]]:
        with self.lock:
            return self._read()

    def _read(self) -> list[dict[str, Any]]:
        if not self.path.is_file():
            return []
        rows = []
        for line in self.path.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict):
                rows.append(row)
        return rows


def _is_score(value: Any) -> bool:
    low, high = RATING_SCORES
    return isinstance(value, int) and not isinstance(value, bool) and low <= value <= high


def rating_summary(
    rows: Sequence[Mapping[str, Any]], run_ids: Mapping[str, str]
) -> dict[str, dict[str, dict[str, Any]]]:
    """Per model id and view, the number of ratings and their mean score.

    A row counts toward a loaded model only if it rated that model's current run: an
    id the registry has since given to another run does not inherit the old ratings.
    Rows of models not loaded now count under their id.
    """

    scores: dict[str, dict[str, list[int]]] = {}
    for row in rows:
        model, view, score = row.get("model"), row.get("view"), row.get("score")
        if not isinstance(model, str) or view not in RATING_VIEWS or not _is_score(score):
            continue
        current = run_ids.get(model)
        if current is not None and row.get("run_id") != current:
            continue
        scores.setdefault(model, {}).setdefault(str(view), []).append(cast(int, score))
    return {
        model: {
            view: {"n": len(values), "mean": round(sum(values) / len(values), 3)}
            for view, values in views.items()
        }
        for model, views in scores.items()
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
    """Batch-1 CUDA graph decoder for greedy transcription (transcribers only)."""
    graph_many: Any = None
    """Batch-8 CUDA graph decoder for best of 8 (transcribers only)."""
    backend: LatentBackend | None = None
    """How a latent model encodes, decodes and samples (latent models only)."""


def _as_backend(spec: ModelSpec, given: RenderToProgram | LatentBackend) -> LatentBackend:
    """A latent model's backend: as given, or a bare `LatentToProgram` served as a vae."""

    if isinstance(given, LatentToProgram):
        return VaeBackend(given)
    if isinstance(given, LatentBackend):
        return given
    raise ValueError(f"model {spec.id} does not match its kind 'latent'")


def load_model(
    spec: ModelSpec,
    layout: SequenceLayout,
    device: torch.device,
    *,
    checkpoint: Path | None = None,
) -> tuple[RenderToProgram | LatentBackend, dict[str, Any]]:
    """One model from its checkpoint (`spec.checkpoint` unless given), on `device`: a
    transcriber, or a latent model's backend built by the factory its spec names. With
    it, the checkpoint's other entries (`step`, `config`, ...) without the weights."""

    from mojidiff.learning.render2svg import ModelConfig

    if spec.kind == "latent" and spec.backend not in BACKENDS:
        raise ValueError(f"model {spec.id}: unknown backend {spec.backend!r}")
    # Loaded on the CPU and moved once, so no second copy of the weights sits on the
    # device while the next model loads.
    state = torch.load(checkpoint or spec.checkpoint, map_location="cpu")
    loaded: RenderToProgram | LatentBackend
    if spec.kind == "latent":
        loaded = BACKENDS[cast(str, spec.backend)](state, layout, device)
    else:
        model = RenderToProgram(layout, ModelConfig(**state["config"]))
        model.load_state_dict(state["model"])
        loaded = model.to(device)
    return loaded, {key: value for key, value in state.items() if key != "model"}


class Gallery:
    """Loaded models by id, the held-out icons and their programs, and the locks.

    Each model has its own lock, and all device work also takes one global GPU lock:
    graph replays on one device must not interleave across request threads.

    A latent model is given as its backend, or as a bare `LatentToProgram`, which the
    `vae` backend serves. `images` are the held-out programs' renders at `RENDER_SIZE`,
    the latent encoders' second input; an icon without one is rendered on first use.
    Ratings are appended to `ratings`; without it the gallery records none.
    """

    def __init__(
        self,
        models: Sequence[tuple[ModelSpec, RenderToProgram | LatentBackend]],
        template: PackedTensorProgram,
        programs: Mapping[str, Tensor],
        device: torch.device,
        *,
        icons: Sequence[Mapping[str, Any]] | None = None,
        images: Mapping[str, Tensor] | None = None,
        ratings: Path | None = None,
        runs_root: Path = RUNS_ROOT,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self.template = template
        self.programs = dict(programs)
        self.device = device
        self.gpu_lock = threading.Lock()
        self.ratings = None if ratings is None else RatingLog(ratings)
        source = held_out_icons() if icons is None else icons
        self.icons = [{key: icon.get(key) for key in ICON_FIELDS} for icon in source]
        self._icon_codes = {str(icon["hexcode"]) for icon in self.icons}
        self._pngs: dict[str, bytes] = {}
        self._images: dict[str, Tensor] = dict(images or {})
        self._models: dict[str, _Loaded] = {}
        for index, (spec, given) in enumerate(models, 1):
            if spec.id in self._models:
                raise ValueError(f"duplicate model id: {spec.id}")
            started = time.perf_counter()
            backend: LatentBackend | None = None
            if spec.kind == "latent":
                backend = _as_backend(spec, given)
                model = backend.model
            elif isinstance(given, RenderToProgram) and not isinstance(given, LatentToProgram):
                model = given
            else:
                raise ValueError(f"model {spec.id} does not match its kind {spec.kind!r}")
            try:
                record = run_record(spec, runs_root)
            except FileNotFoundError:
                record = {}
                if log is not None:
                    log(f"warning: no run record for {spec.run_id}; stats are null")
            entry = model_entry(spec, record)
            loaded = _Loaded(spec, model.eval(), entry, threading.Lock(), backend=backend)
            if device.type == "cuda":
                if spec.kind == "transcriber":
                    from mojidiff.learning.fast_decode import GraphDecoder

                    # Float32 graphs decode exactly what the evaluated batched decoder
                    # decodes. A latent backend captures its own when it is built.
                    loaded.graph = GraphDecoder(model, dtype=torch.float32)
                    loaded.graph_many = GraphDecoder(model, dtype=torch.float32, batch=CANDIDATES)
                if log is not None:
                    what = "graphs captured" if backend is None else f"{spec.backend} backend ready"
                    log(
                        f"[{index}/{len(models)}] {spec.id}: {what} in "
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
        ratings: Path | None = RATINGS,
        runs_root: Path = RUNS_ROOT,
        log: Callable[[str], None] | None = None,
    ) -> Gallery:
        """Load every listed model's best checkpoint; refuse to start if one is missing
        or names an unknown backend. A model without a run record is served with null
        stats, as `Gallery` serves it, and a warning."""

        from mojidiff.learning.openmoji_pilot import _load_program, load_pilot_index
        from mojidiff.learning.render2svg import load_corpus, parameter_count

        unknown = [
            f"{spec.id} ({spec.backend})"
            for spec in specs
            if spec.kind == "latent" and spec.backend not in BACKENDS
        ]
        if unknown:
            raise ValueError("unknown latent backends: " + ", ".join(unknown))
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
        # The renders every model trained on, at RENDER_SIZE: the latent encoders' input.
        images = {
            hexcode: plain[split].images[row]
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
        models: list[tuple[ModelSpec, RenderToProgram | LatentBackend]] = []
        for index, spec in enumerate(specs, 1):
            started = time.perf_counter()
            loaded, state = load_model(spec, layout, device)
            model = loaded if isinstance(loaded, RenderToProgram) else loaded.model
            # A backend that runs more than its decoder (a prior over a frozen VT) lists
            # every module in `parts`; its run record counts them all.
            parts = getattr(loaded, "parts", None) or (model,)
            count = sum(parameter_count(part) for part in parts)
            try:
                recorded = run_record(spec, runs_root).get("model_parameters")
            except FileNotFoundError:
                recorded = count  # nothing to check against; `Gallery` warns of the record
            if recorded != count:
                say(f"warning: {spec.id} has {count:,} parameters, its run record {recorded}")
            kind = spec.kind if spec.backend is None else f"{spec.kind}, {spec.backend}"
            say(
                f"[{index}/{len(specs)}] {spec.id} {spec.run_id} ({kind}): "
                f"step {state.get('step')}, {count:,} parameters, "
                f"{time.perf_counter() - started:.1f} s"
            )
            models.append((spec, loaded))
        gallery = cls(
            models,
            template,
            programs,
            device,
            images=images,
            ratings=ratings,
            runs_root=runs_root,
            log=log,
        )
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
            "ratings": self.ratings is not None,
            # The page names where its times come from: the GPU, or a CPU-only Space.
            "device": self.device.type,
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

    def _latent(self, model_id: str) -> tuple[_Loaded, LatentBackend]:
        loaded = self._get(model_id, "latent")
        return loaded, cast(LatentBackend, loaded.backend)

    def _image(self, loaded: _Loaded, hexcode: str) -> Tensor:
        """A held-out icon's program rendered at `RENDER_SIZE`, as the models trained on
        it: from the corpus when given, else rendered once, outside every lock."""

        from mojidiff.learning.render2svg import render_trusted_rgb

        image = self._images.get(hexcode)
        if image is None:
            svg = self._svg(loaded, self._program(hexcode))[0].encode()
            image = self._images[hexcode] = torch.from_numpy(render_trusted_rgb(svg, RENDER_SIZE))
        return image

    def _encoded(
        self, loaded: _Loaded, backend: LatentBackend, hexcodes: Sequence[str]
    ) -> Callable[[], Tensor]:
        """Work that encodes held-out icons to their posterior means on the device."""

        tokens = torch.stack([self._program(hexcode) for hexcode in hexcodes])
        images = torch.stack([self._image(loaded, hexcode) for hexcode in hexcodes])

        def work() -> Tensor:
            means = backend.encode(tokens.to(self.device), images.to(self.device))
            return _shaped(loaded, "encode", means, (len(hexcodes), *backend.latent_shape))

        return work

    def reconstruct(self, model_id: str, hexcode: str) -> dict[str, Any]:
        """The posterior mean of a held-out icon, decoded greedily (by the backend's
        single decoder, a CUDA graph, when it has one)."""

        loaded, backend = self._latent(model_id)
        encoded = self._encoded(loaded, backend, [hexcode])
        length = loaded.model.layout.length

        def work() -> Tensor:
            return _shaped(loaded, "decode_one", decode_one(backend, encoded()[0]), (length,))

        decoded, ms = self._run(loaded, work)
        return {"ok": True, "svg": self._svg(loaded, decoded)[0], "ms": ms}

    def interpolate(
        self, model_id: str, a: str, b: str, steps: int, path: str = "backend"
    ) -> dict[str, Any]:
        """`steps` frames between two posterior means, decoded in one batch. With `path`
        "backend", the backend's own path when it has an `interpolate` (canvas-flow:
        slerp through its prior's noise), else the straight line; with "lerp", the
        straight line always. The answer's `path` names the path taken."""

        loaded, backend = self._latent(model_id)
        low, high = INTERPOLATION_STEPS
        if not low <= steps <= high:
            raise GalleryError(f"steps must be {low}..{high}, got {steps}")
        if path not in INTERPOLATION_PATHS:
            raise GalleryError(f"path must be one of {', '.join(INTERPOLATION_PATHS)}")
        own = getattr(backend, "interpolate", None) if path == "backend" else None
        taken = "lerp" if own is None else str(getattr(backend, "interpolation_path", "backend"))
        encoded = self._encoded(loaded, backend, [a, b])
        length = loaded.model.layout.length

        def work() -> Tensor:
            means = encoded()
            if own is not None:
                shape = (steps, *backend.latent_shape)
                latents = _shaped(loaded, "interpolate", own(means[0], means[1], steps), shape)
            else:
                weights = torch.linspace(0.0, 1.0, steps, device=self.device)
                weights = weights.view(steps, *(1,) * (means.dim() - 1))
                latents = (1.0 - weights) * means[0] + weights * means[1]
            return _shaped(loaded, "decode", backend.decode(latents), (steps, length))

        decoded, ms = self._run(loaded, work)
        frames = [self._svg(loaded, row)[0] for row in decoded]
        return {"ok": True, "frames": frames, "path": taken, "ms": ms}

    def sample(self, model_id: str, count: int, seed: int, scale: float) -> dict[str, Any]:
        """`count` draws from the backend's prior at `scale` (for a vae z = scale *
        N(0, I)), seeded by a CPU generator, decoded in one batch."""

        loaded, backend = self._latent(model_id)
        low, high = SAMPLE_COUNT
        if not low <= count <= high:
            raise GalleryError(f"count must be {low}..{high}, got {count}")
        if not SAMPLE_SCALE[0] <= scale <= SAMPLE_SCALE[1]:
            raise GalleryError(f"scale must be {SAMPLE_SCALE[0]}..{SAMPLE_SCALE[1]}, got {scale}")
        if not 0 <= seed < 2**63:
            raise GalleryError(f"seed must be a non-negative 63-bit integer, got {seed}")
        generator = torch.Generator().manual_seed(seed)
        length = loaded.model.layout.length

        def work() -> Tensor:
            # Under the GPU lock: a learned prior may run on the device.
            drawn = backend.sample(count, generator, scale)
            latents = _shaped(loaded, "sample", drawn, (count, *backend.latent_shape))
            return _shaped(
                loaded, "decode", backend.decode(latents.to(self.device)), (count, length)
            )

        decoded, ms = self._run(loaded, work)
        samples = [self._svg(loaded, row)[0] for row in decoded]
        return {"ok": True, "samples": samples, "ms": ms}

    # ------------------------------------------------------------------ ratings

    def _rating_log(self) -> RatingLog:
        if self.ratings is None:
            raise GalleryError("this gallery records no ratings")
        return self.ratings

    def rating_row(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """A posted rating, checked, as the line it appends: `at` (UTC) and the model's
        `run_id` added. Every refusal is a `GalleryError`."""

        unknown = sorted(set(payload) - set(RATING_FIELDS))
        if unknown:
            raise GalleryError(f"unknown rating field(s): {', '.join(map(str, unknown))}")
        for key in ("model", "view", "score", "blind"):
            if key not in payload:
                raise GalleryError(f"missing field {key!r}")
        model_id = payload["model"]
        if not isinstance(model_id, str):
            raise GalleryError("model must be a string")
        loaded = self._get(model_id, "latent")
        view = payload["view"]
        if view not in RATING_VIEWS:
            raise GalleryError(f"view must be one of {', '.join(RATING_VIEWS)}")
        score = payload["score"]
        if not _is_score(score):
            raise GalleryError(f"score must be an integer {RATING_SCORES[0]}..{RATING_SCORES[1]}")
        blind = payload["blind"]
        if not isinstance(blind, bool):
            raise GalleryError("blind must be true or false")
        seed = payload.get("seed")
        if seed is not None and not (
            isinstance(seed, int) and not isinstance(seed, bool) and 0 <= seed < 2**63
        ):
            raise GalleryError("seed must be null or a non-negative 63-bit integer")
        scale = payload.get("scale")
        if scale is not None:
            low, high = SAMPLE_SCALE
            if isinstance(scale, bool) or not isinstance(scale, int | float):
                raise GalleryError("scale must be null or a number")
            if not (math.isfinite(scale) and low <= scale <= high):
                raise GalleryError(f"scale must be null or {low}..{high}")
            scale = float(scale)
        icons: dict[str, str | None] = {}
        for key in ("a", "b"):
            hexcode = payload.get(key)
            if hexcode is not None and not (isinstance(hexcode, str) and hexcode in self.programs):
                raise GalleryError(f"{key} must be null or a held-out icon's hexcode")
            icons[key] = hexcode
        steps = payload.get("steps")
        if steps is not None:
            low, high = INTERPOLATION_STEPS
            if isinstance(steps, bool) or not isinstance(steps, int) or not low <= steps <= high:
                raise GalleryError(f"steps must be null or an integer {low}..{high}")
        note = payload.get("note", "")
        if not isinstance(note, str) or len(note) > RATING_NOTE_LIMIT:
            raise GalleryError(f"note must be a string of at most {RATING_NOTE_LIMIT} characters")
        return {
            "at": _utc_now(),
            "model": model_id,
            "run_id": loaded.spec.run_id,
            "view": view,
            "score": score,
            "blind": blind,
            "seed": seed,
            "scale": scale,
            "a": icons["a"],
            "b": icons["b"],
            "steps": steps,
            "note": note,
        }

    def rate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Append one rating; the file's rating count afterwards."""

        log = self._rating_log()
        return {"ok": True, "count": log.append(self.rating_row(payload))}

    def ratings_payload(self) -> dict[str, Any]:
        """Every rating in the file, and per model and view the count and mean score."""

        rows = self._rating_log().rows()
        run_ids = {model_id: loaded.spec.run_id for model_id, loaded in self._models.items()}
        return {"ratings": rows, "summary": rating_summary(rows, run_ids)}


def _shaped(loaded: _Loaded, method: str, value: Tensor, shape: tuple[int, ...]) -> Tensor:
    """`value` if it has `shape`; else a backend broke its contract, said by name."""

    if tuple(value.shape) != shape:
        raise RuntimeError(
            f"backend {loaded.spec.backend!r} of {loaded.spec.id!r}: {method} returned "
            f"shape {tuple(value.shape)}, expected {shape}"
        )
    return value


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
    path = _text(payload, "path") if "path" in payload else "backend"
    return lambda: gallery.interpolate(model_id, a, b, steps, path)


def _sample(gallery: Gallery, payload: Mapping[str, Any]) -> Work:
    model_id = _text(payload, "model")
    count, seed = _integer(payload, "count"), _integer(payload, "seed")
    scale = _number(payload, "scale")
    return lambda: gallery.sample(model_id, count, seed, scale)


def _rate(gallery: Gallery, payload: Mapping[str, Any]) -> Work:
    # Checked before anything is written: every refusal is a GalleryError, a 400.
    return lambda: gallery.rate(payload)


ROUTES: dict[str, Callable[[Gallery, Mapping[str, Any]], Work]] = {
    "/transcribe": _transcribe,
    "/latent/reconstruct": _reconstruct,
    "/latent/interpolate": _interpolate,
    "/latent/sample": _sample,
    "/rating": _rate,
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
        elif url.path == "/ratings":
            if self.gallery.ratings is None:
                self._json(503, {"ok": False, "error": "this gallery records no ratings"})
                return
            try:
                self._json(200, self.gallery.ratings_payload())
            except OSError as error:
                self._json(500, {"ok": False, "error": f"{type(error).__name__}: {error}"[:200]})
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
        path = urlparse(self.path).path
        route = ROUTES.get(path)
        if route is None:
            self._json(404, {"ok": False, "error": "not found"})
            return
        if path == "/rating" and self.gallery.ratings is None:
            self._json(503, {"ok": False, "error": "this gallery records no ratings"})
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


def serve(
    host: str, port: int, registry: Path | None = None, ratings: Path | None = RATINGS
) -> None:
    """Load the registry's models (`registry`, else the default file, else `MODELS`) and
    serve them; ratings append to `ratings`, or with None (a public copy) are refused and
    the page hides its rating controls."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    started = time.perf_counter()
    specs = registry_models(registry, log=_log)
    where = "ratings off" if ratings is None else f"ratings to {ratings}"
    _log(f"registry {registry or REGISTRY}: {len(specs)} models on {device.type}; {where}")
    gallery = Gallery.from_checkpoints(device, specs, ratings=ratings, log=_log)
    with make_server(gallery, host, port) as server:
        _log(
            f"gallery on http://{host}:{port} - {len(gallery)} models, "
            f"{len(gallery.icons)} held-out icons, ready in {time.perf_counter() - started:.0f} s"
        )
        server.serve_forever()
