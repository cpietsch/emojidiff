"""One scorer for every latent model: prior samples, reconstructions, interpolations.

Every latent model reaches this code through its gallery backend
(`gallery.latent_backends.LatentBackend`), so every one is scored on the same fixed sets
by the same measures (the staged latent plan, section 1):

* Sets: 339 prior draws (seed 23); the first 339 validation icons, reconstructed from
  their posterior means; 32 validation pairs (rng 17, latent.py's pairs) interpolated
  linearly over 9 frames. Decoding is greedy in IEEE float32 (TF32 off,
  `precision.strict_float32`; with TF32 on, a decode depends on the batch it sits in);
  renders are scored at 72 px. The command line runs everything, CLIP included, that
  way and records the flags.
* Ink: the share of pixels whose darkest channel is below 250. A fragment has less ink
  than the validation icons' 5th percentile; a near-blank render has less than 0.02.
* Collapse: the share of decoded paths spanning under half a view unit, and per icon the
  median decoded path extent over the true one.
* Emoji-likeness, in CLIP ViT-B/32 space (`omnisvg_study.Clip`, pinned) against the
  validation renders: k-NN precision and recall (k=3), density and coverage (k=5), and
  FD_CLIP (reported, no criterion). Reference rows score training renders (the ceiling),
  `compose_program` collages (a sampler with no parameters) and a blank canvas alike.
* Novelty: the nearest training render or mirror by CLIP cosine, and the smallest pixel
  error to a shortlisted one with the render moved up to 12 units in steps of 4. A copy
  is at or above a cosine threshold and at or below an aligned-error threshold, both
  calibrated on the validation icons (95th and 5th percentiles) under a named rule,
  which the command line requires (`--copy-rule`):
  - `declared`, the plan's rule as written, over every validation icon. About 7% of
    validation icons are near-exact twins of a training render or its mirror (aligned
    error under 0.002: walking left and right, say, are different families), so the
    5th percentile falls inside that cluster and a copy means a pixel-identical render.
  - `without-twins`, the same percentiles over the validation icons that are not twins.
  The twins are listed in the calibration; per-item cosines and aligned errors are kept,
  so any other rule can be applied afterwards.
* Reconstruction baseline: the nearest training render or its mirror (every model
  trains with mirrors), with the mirror-free search beside it for older records.
* Interpolation: endpoint fidelity, jump share (largest consecutive-frame CLIP distance
  over their sum; 1/(frames-1) is even, 1 a single cut), interior fragments, near-blanks
  and precision, and detours (interior copies of a training icon that neither endpoint's
  nearest training icon is). A static strip (no frame moves: a decoder that ignores
  the latent) has no jump share: it is counted in `static_pairs`, left out of the jump
  share, and scored as a single cut in `jump_share_static_as_cut`.

`--prior refit` draws from a full-covariance Gaussian fitted to the training icons'
posterior means instead of the backend's prior. A transcriber's registry id scores the
pixel-crossfade reference instead: each frame alpha-blends the pair's renders, the
transcriber writes it down greedily, and only the interpolation block is filled.

Reports are named by run, prior and a hash of the settings, record the sha256 of the
scoring code and any uncommitted source, and are never overwritten without
`--overwrite`.

    python -m mojidiff.learning.latent_metrics --registry-id l2 --copy-rule RULE [--prior refit]
    python -m mojidiff.learning.latent_metrics --registry-id v9 --copy-rule RULE  # crossfade
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from mojidiff.gallery.latent_backends import LatentBackend, decode_one
from mojidiff.learning.autoregressive import (
    PATH_FIELDS,
    PATH_STRIDE,
    SEGMENT_STRIDE,
    SequenceLayout,
    flatten_program,
    unflatten_program,
)
from mojidiff.learning.precision import precision_tag, strict_float32
from mojidiff.learning.render2svg import (
    _CONTROL_SLOTS,
    CACHE_ROOT,
    REPO_ROOT,
    RenderToProgram,
    SplitData,
    _git,
    _programs_to_renders,
    bootstrap_mean_interval,
    compose_program,
    contact_sheet,
    greedy_decode,
    nearest_training_icon,
    pixel_error,
    uncommitted_sources,
)
from mojidiff.learning.telemetry import (
    LatencyReport,
    measure_latency,
    reset_peak_memory,
    resource_summary,
)
from mojidiff.representation.packed import PackedTensorProgram

Embed = Callable[[Sequence[np.ndarray]], Tensor]
"""72 px uint8 renders to unit-length features, (N, D) on the CPU."""
Renders = list[np.ndarray | None]

INK_LEVEL = 250
NEAR_BLANK_INK = 0.02
COLLAPSED_UNITS = 0.5
TWIN_ERROR = 0.002
"""A validation icon within this aligned error of a training render or its mirror is a
twin of it: a near-exact duplicate across the family-disjoint split."""
COPY_RULES = ("declared", "without-twins")
SHIFTS = tuple(range(-12, 13, 4))
"""Offsets per axis, in 72 px pixels (= view units), tried when aligning two renders."""
BLANK = np.full((72, 72, 3), 255, dtype=np.uint8)
CPU = torch.device("cpu")
OUT_DIR = REPO_ROOT / "reports" / "latent"


@dataclass(frozen=True)
class Settings:
    samples: int = 339
    sample_seed: int = 23
    pairs: int = 32
    frames: int = 9
    pair_seed: int = 17
    icons: int = 339
    """Validation icons reconstructed, from the first."""
    prior: str = "normal"
    """"normal": the backend's own prior; "refit": `RefitPrior` over training means."""
    copy_rule: str = "declared"
    """Which validation icons calibrate the copy thresholds (one of `COPY_RULES`)."""
    batch: int = 32
    shortlist: int = 16
    """Nearest library renders by CLIP and by raw pixels searched for the aligned error."""
    latency_repeats: int = 10


# --------------------------------------------------------------------------- measures


def ink(render: np.ndarray | None) -> float:
    """The share of pixels whose darkest channel is below 250; none for a failed render."""

    if render is None:
        return 0.0
    return float((render.min(axis=-1) < INK_LEVEL).mean())


def path_extents(program: Tensor, layout: SequenceLayout) -> list[float]:
    """Each active path's extent in view units: the larger of its x and y ranges over its
    start point and every segment's endpoint (the pair after the kind's control slots)."""

    row = [int(v) for v in program.tolist()]
    start = len(PATH_FIELDS)
    extents: list[float] = []
    offset = 0
    for path in range(layout.codec.max_paths):
        base = path * PATH_STRIDE
        length = row[base]
        if length <= 0:
            break
        xs, ys = [row[base + start]], [row[base + start + 1]]
        for slot in range(offset, min(offset + length, layout.total_segment_slots)):
            at = layout.path_positions + slot * SEGMENT_STRIDE
            if 1 <= row[at] <= 3:
                end = at + 1 + _CONTROL_SLOTS[row[at]]
                xs.append(row[end])
                ys.append(row[end + 1])
        offset += length
        extents.append(0.25 * max(max(xs) - min(xs), max(ys) - min(ys)))
    return extents


def collapsed_share(programs: Tensor, layout: SequenceLayout) -> float:
    """The share of all paths in `programs` spanning under half a view unit."""

    extents = [e for row in programs for e in path_extents(row, layout)]
    return float(np.mean(np.array(extents) < COLLAPSED_UNITS)) if extents else 0.0


def extent_ratios(decoded: Tensor, truth: Tensor, layout: SequenceLayout) -> list[float]:
    """Per icon, the median decoded path extent over the median true one; 0 without paths."""

    ratios = []
    for mine, true in zip(decoded, truth, strict=True):
        a, b = path_extents(mine, layout), path_extents(true, layout)
        ratios.append(float(np.median(a)) / max(float(np.median(b)), 0.25) if a and b else 0.0)
    return ratios


def _distance(a: Tensor, b: Tensor) -> Tensor:
    return (1.0 - a @ b.T).clamp_min(0.0)


def knn_radii(features: Tensor, k: int) -> Tensor:
    """Cosine distance from each point to its k-th nearest other point (k capped by size)."""

    if len(features) < 2:
        return torch.zeros(len(features))
    distance = _distance(features, features).fill_diagonal_(float("inf"))
    return distance.kthvalue(min(k, len(features) - 1), dim=1).values


def precision_recall(real: Tensor, fake: Tensor, k: int = 3) -> tuple[list[float], list[float]]:
    """Improved precision and recall (Kynkaanniemi et al. 2019), per point: whether each
    fake point lies in some real point's k-NN ball, and each real point in a fake one's."""

    distance = _distance(fake, real)
    precision = (distance <= knn_radii(real, k)[None]).any(dim=1)
    recall = (distance.T <= knn_radii(fake, k)[None]).any(dim=1)
    return precision.float().tolist(), recall.float().tolist()


def density_coverage(real: Tensor, fake: Tensor, k: int = 5) -> tuple[list[float], list[float]]:
    """Density and coverage (Naeem et al. 2020), per point: the real k-NN balls holding
    each fake point, over k; and whether each real point's ball holds any fake point."""

    inside = _distance(fake, real) <= knn_radii(real, k)[None]
    k = max(1, min(k, len(real) - 1))
    return (inside.sum(dim=1) / k).tolist(), inside.any(dim=0).float().tolist()


def frechet_distance(real: Tensor, fake: Tensor) -> float | None:
    """Frechet distance between Gaussian fits of two feature sets; None below two points."""

    if min(len(real), len(fake)) < 2:
        return None
    a, b = real.double(), fake.double()
    shift = a.mean(dim=0) - b.mean(dim=0)
    size = a.shape[1]
    first, second = torch.cov(a.T).reshape(size, size), torch.cov(b.T).reshape(size, size)
    values, vectors = torch.linalg.eigh(first)
    root = vectors @ torch.diag(values.clamp_min(0.0).sqrt()) @ vectors.T
    cross = torch.linalg.eigvalsh(root @ second @ root).clamp_min(0.0).sqrt().sum()
    return float(shift @ shift + first.trace() + second.trace() - 2.0 * cross)


def mean_pairwise_distance(features: Tensor) -> float:
    if len(features) < 2:
        return 0.0
    upper = torch.triu_indices(len(features), len(features), offset=1)
    return float(_distance(features, features)[upper[0], upper[1]].mean())


def _steps(strip: Tensor) -> Tensor:
    return (1.0 - (strip[1:] * strip[:-1]).sum(dim=-1)).clamp_min(0.0)


def jump_share(strip: Tensor) -> float | None:
    """The largest consecutive-frame distance over their sum: 1/(frames-1) when every step
    moves equally, 1 for a single cut; None for a static strip, where no frame moves (a
    decoder that ignores its latent), which must not score as perfectly even."""

    steps = _steps(strip)
    total = float(steps.sum())
    return float(steps.max()) / total if total > 1e-6 else None


def aligned_error(
    query: np.ndarray, candidates: np.ndarray, device: torch.device = CPU
) -> tuple[float, int]:
    """The smallest mean absolute error, in [0, 1], between `query` moved by any of
    `SHIFTS` on each axis (white filling in) and any candidate; and that candidate."""

    size, pad = query.shape[0], max(SHIFTS)
    padded = np.pad(query, ((pad, pad), (pad, pad), (0, 0)), constant_values=255)
    moved = np.stack(
        [padded[pad - y : pad - y + size, pad - x : pad - x + size] for y in SHIFTS for x in SHIFTS]
    )
    shifted = torch.from_numpy(moved).to(device=device, dtype=torch.float32).flatten(1)
    options = torch.from_numpy(candidates).to(device=device, dtype=torch.float32).flatten(1)
    errors = torch.cdist(shifted, options, p=1).min(dim=0).values / (shifted.shape[1] * 255.0)
    best = int(errors.argmin())
    return float(errors[best]), best


def crossfade_frames(first: Tensor, second: Tensor, frames: int) -> Tensor:
    """`frames` alpha blends of two uint8 renders, from exactly `first` to exactly `second`."""

    weights = torch.linspace(0.0, 1.0, frames).view(-1, 1, 1, 1)
    return ((1.0 - weights) * first.float() + weights * second.float()).round().to(torch.uint8)


def fixed_pairs(count: int, pairs: int, seed: int) -> list[tuple[int, int]]:
    """The fixed validation pairs, drawn as latent.py draws its interpolation sheet's."""

    rng = np.random.default_rng(seed)
    return [
        (int(a), int(b)) for a, b in (rng.choice(count, 2, replace=False) for _ in range(pairs))
    ]


class RefitPrior:
    """A full-covariance Gaussian fitted to posterior means: the aggregate posterior's
    first two moments, a prior control that needs no training."""

    def __init__(self, latents: Tensor, ridge: float = 1e-6) -> None:
        flat = latents.reshape(len(latents), -1).double()
        self.shape = tuple(latents.shape[1:])
        self.count = len(flat)
        self.mean: Tensor = flat.mean(dim=0)
        size = flat.shape[1]
        self.covariance: Tensor = torch.cov(flat.T).reshape(size, size)
        # A grid latent can have more dimensions than icons: widen the ridge until the
        # covariance factors, and record the ridge used.
        eye = torch.eye(size, dtype=torch.float64)
        scale = float(self.covariance.diagonal().mean()) + 1e-12
        self.ridge = ridge
        factor, info = torch.linalg.cholesky_ex(self.covariance + ridge * scale * eye)
        while int(info) != 0 and self.ridge < 1.0:
            self.ridge *= 10.0
            factor, info = torch.linalg.cholesky_ex(self.covariance + self.ridge * scale * eye)
        self.factor: Tensor = factor

    def sample(self, count: int, generator: torch.Generator, scale: float = 1.0) -> Tensor:
        noise = torch.randn(count, len(self.mean), generator=generator, dtype=torch.float64)
        return (self.mean + scale * noise @ self.factor.T).float().reshape(count, *self.shape)


# --------------------------------------------------------------------------- references


@dataclass
class Reference:
    """The fixed sets every model is scored against, and thresholds calibrated on them."""

    layout: SequenceLayout
    template: PackedTensorProgram
    validation: SplitData
    train: SplitData
    embed: Embed
    device: torch.device
    validation_features: Tensor
    library_features: Tensor
    """Training renders, then their mirror images."""
    library_renders: np.ndarray
    library_pixels: Tensor
    """The library as 24 px thumbnails, to shortlist neighbours by raw pixels."""
    training_programs: set[bytes]
    shortlist: int = 16
    calibration: dict[str, Any] = field(default_factory=dict)
    validation_neighbour: list[int] = field(default_factory=list)
    """Each validation icon's nearest training icon by CLIP (mirrors folded in)."""
    validation_nearest_error: list[float] = field(default_factory=list)
    """Each validation icon's pixel error to the closest training render or mirror: no
    model at all."""
    validation_nearest_error_without_mirrors: list[float] = field(default_factory=list)
    """The same over training renders alone, as earlier run records measured it."""

    def features(self, renders: Sequence[np.ndarray | None]) -> Tensor:
        return self.embed([r if r is not None else BLANK for r in renders])

    def nearest(
        self, renders: Sequence[np.ndarray], features: Tensor
    ) -> tuple[np.ndarray, list[int], np.ndarray, list[int]]:
        """Per render: cosine to the nearest library render, that training icon, the
        aligned error to the closest of its shortlisted neighbours (by CLIP and pixels),
        and that neighbour's library index (a mirror from `len(train.tokens)` on)."""

        similarity = features @ self.library_features.T
        cosine, index = similarity.max(dim=1)
        count = min(self.shortlist, similarity.shape[1])
        by_clip = similarity.topk(count, dim=1).indices
        aligned: list[float] = []
        closest: list[int] = []
        for start in range(0, len(renders), 64):
            pixels = _thumbnails(np.stack(renders[start : start + 64]), self.device)
            by_pixel = torch.cdist(pixels, self.library_pixels, p=1)
            for offset, row in enumerate(by_pixel.topk(count, dim=1, largest=False).indices):
                i = start + offset
                options = sorted(set(by_clip[i].tolist()) | set(row.tolist()))
                error, best = aligned_error(renders[i], self.library_renders[options], self.device)
                aligned.append(error)
                closest.append(options[best])
        folded = (index % len(self.train.tokens)).tolist()
        return cosine.numpy(), folded, np.array(aligned), closest

    def copies(self, cosine: np.ndarray, aligned: np.ndarray) -> np.ndarray:
        near = (cosine >= float(self.calibration["copy_cosine"])) & (
            aligned <= float(self.calibration["copy_error"])
        )
        return near.astype(np.float64)


def _thumbnails(renders: np.ndarray, device: torch.device) -> Tensor:
    """24 px thumbnails, pooled on the CPU in chunks so the device only holds the result."""

    chunks = [
        F.avg_pool2d(torch.from_numpy(renders[s : s + 512]).permute(0, 3, 1, 2) / 255.0, 3)
        for s in range(0, len(renders), 512)
    ]
    return torch.cat(chunks).flatten(1).to(device)


def build_reference(
    layout: SequenceLayout,
    template: PackedTensorProgram,
    validation: SplitData,
    train: SplitData,
    embed: Embed,
    device: torch.device,
    *,
    extra_programs: Tensor | None = None,
    cache: Path | None = None,
    shortlist: int = 16,
    copy_rule: str = "declared",
) -> Reference:
    """Embed the validation and training renders (and mirrors), calibrate the thresholds
    (the copy thresholds under `copy_rule`, see the module docstring; both rules' are
    recorded). `extra_programs` join the training programs for exact-copy checks (their
    mirrors)."""

    if copy_rule not in COPY_RULES:
        raise ValueError(f"unknown copy rule {copy_rule!r}; one of {', '.join(COPY_RULES)}")
    renders = train.targets.numpy()
    library = np.concatenate((renders, np.ascontiguousarray(renders[:, :, ::-1])))
    targets = list(validation.targets.numpy())
    features: dict[str, Tensor] | None = None
    if cache is not None and cache.is_file():
        features = cast(dict[str, Tensor], torch.load(cache))
        if [len(features["validation"]), len(features["library"])] != [len(targets), len(library)]:
            features = None
    if features is None:
        features = {"validation": embed(targets), "library": embed(list(library))}
        if cache is not None:
            cache.parent.mkdir(parents=True, exist_ok=True)
            torch.save(features, cache)
    programs = train.tokens if extra_programs is None else torch.cat((train.tokens, extra_programs))
    reference = Reference(
        layout,
        template,
        validation,
        train,
        embed,
        device,
        features["validation"],
        features["library"],
        library,
        _thumbnails(library, device),
        {row.numpy().tobytes() for row in programs},
        shortlist,
    )
    cosine, neighbour, aligned, closest = reference.nearest(targets, reference.validation_features)
    reference.validation_neighbour = neighbour
    reference.validation_nearest_error = nearest_training_icon(
        validation.targets, torch.from_numpy(library), device
    )[1].tolist()
    reference.validation_nearest_error_without_mirrors = nearest_training_icon(
        validation.targets, train.targets, device
    )[1].tolist()
    count = len(train.tokens)
    twin = aligned < TWIN_ERROR
    thresholds = {
        "declared": _copy_thresholds(cosine, aligned),
        "without-twins": _copy_thresholds(cosine[~twin], aligned[~twin]),
    }
    if thresholds[copy_rule]["copy_cosine"] is None:
        raise ValueError(f"no validation icons left to calibrate the {copy_rule!r} copy rule")
    reference.calibration = {
        "fragment_ink": float(np.quantile([ink(t) for t in targets], 0.05)),
        "copy_rule": copy_rule,
        "copy_cosine": thresholds[copy_rule]["copy_cosine"],
        "copy_error": thresholds[copy_rule]["copy_error"],
        "copy_rules": thresholds,
        "twin_error": TWIN_ERROR,
        # Held-out icons within TWIN_ERROR of a training render or its mirror: they pin
        # the declared rule's aligned-error threshold at a pixel-identical render.
        "twins": [
            {
                "validation": validation.hexcodes[i],
                "training": train.hexcodes[closest[i] % count],
                "mirror": closest[i] >= count,
                "aligned_error": float(aligned[i]),
            }
            for i in np.flatnonzero(twin)
        ],
        "validation_icons": len(targets),
        "library_renders": len(library),
        "validation_mean_pairwise_clip_distance": mean_pairwise_distance(
            reference.validation_features
        ),
        # Kept whole so a stricter or looser rule can be applied to the per-item values
        # afterwards.
        "validation_cosine_quantiles": dict(
            zip(
                ("p50", "p90", "p95", "p99"),
                np.quantile(cosine, [0.5, 0.9, 0.95, 0.99]).tolist(),
                strict=True,
            )
        ),
        "validation_aligned_error_quantiles": dict(
            zip(
                ("p1", "p5", "p10", "p50"),
                np.quantile(aligned, [0.01, 0.05, 0.1, 0.5]).tolist(),
                strict=True,
            )
        ),
    }
    return reference


def _copy_thresholds(cosine: np.ndarray, aligned: np.ndarray) -> dict[str, float | int | None]:
    """A copy is at or above the 95th percentile of these cosines and at or below the 5th
    percentile of these aligned errors."""

    if not len(cosine):
        return {"copy_cosine": None, "copy_error": None, "calibration_icons": 0}
    return {
        "copy_cosine": float(np.quantile(cosine, 0.95)),
        "copy_error": float(np.quantile(aligned, 0.05)),
        "calibration_icons": len(cosine),
    }


def _interval(values: Sequence[float]) -> list[float] | None:
    return list(bootstrap_mean_interval(values)) if len(values) else None


def score_set(
    reference: Reference, renders: Renders, features: Tensor, programs: Tensor | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Every sample measure of one set of renders: (summary, per-item values)."""

    calibration = reference.calibration
    inks = [ink(r) for r in renders]
    precision, recall = precision_recall(reference.validation_features, features, 3)
    density, coverage = density_coverage(reference.validation_features, features, 5)
    drawn = [r if r is not None else BLANK for r in renders]
    cosine, neighbour, aligned, _ = reference.nearest(drawn, features)
    copy = reference.copies(cosine, aligned).tolist()
    summary: dict[str, Any] = {
        "count": len(renders),
        "rendered": sum(r is not None for r in renders),
        "ink_mean": float(np.mean(inks)),
        "fragment_rate": _interval([float(v < calibration["fragment_ink"]) for v in inks]),
        "near_blank_rate": _interval([float(v < NEAR_BLANK_INK) for v in inks]),
        "clip_precision": _interval(precision),
        "clip_recall": _interval(recall),
        "clip_density": _interval(density),
        "clip_coverage": _interval(coverage),
        "fd_clip": frechet_distance(reference.validation_features, features),
        "nearest_training_cosine_median": float(np.median(cosine)),
        "copy_rate_cosine_only": float(np.mean(cosine >= calibration["copy_cosine"])),
        "copy_rate": _interval(copy),
        "mean_pairwise_clip_distance": mean_pairwise_distance(features),
    }
    if programs is not None:
        keys = [row.numpy().tobytes() for row in programs]
        summary["distinct_programs"] = len(set(keys))
        summary["exact_token_copies"] = sum(key in reference.training_programs for key in keys)
        summary["collapsed_path_share"] = collapsed_share(programs, reference.layout)
    per_item = {
        "ink": inks,
        "clip_precision": precision,
        "copy": copy,
        "nearest_training_cosine": cosine.tolist(),
        "aligned_error": aligned.tolist(),
        "nearest_training_icon": [reference.train.hexcodes[i] for i in neighbour],
    }
    return summary, per_item


def score_interpolation(
    reference: Reference, programs: Tensor, pairs: list[tuple[int, int]], frames: int
) -> tuple[dict[str, Any], dict[str, Any], list[Renders]]:
    """Strips of `frames` programs per pair, stacked: (summary, per pair, sheet rows)."""

    renders = _programs_to_renders(programs, reference.layout, reference.template)
    features = reference.features(renders)
    targets = reference.validation.targets.numpy()
    interior = [p * frames + t for p in range(len(pairs)) for t in range(1, frames - 1)]
    inner = [renders[i] for i in interior]
    precision, _ = precision_recall(reference.validation_features, features[interior], 3)
    drawn = [r if r is not None else BLANK for r in inner]
    cosine, neighbour, aligned, _ = reference.nearest(drawn, features[interior])
    copy = reference.copies(cosine, aligned)
    keys = ("jump_share", "static", "endpoint", "fragment", "near_blank", "precision", "detour")
    per_pair: dict[str, list[Any]] = {key: [] for key in keys}
    rows: list[Renders] = []
    width = frames - 2
    for p, (a, b) in enumerate(pairs):
        strip = renders[p * frames : (p + 1) * frames]
        steps = features[p * frames : (p + 1) * frames]
        inks = np.array([ink(r) for r in inner[p * width : (p + 1) * width]])
        ends = {reference.validation_neighbour[a], reference.validation_neighbour[b]}
        window = range(p * width, (p + 1) * width)
        share = jump_share(steps)
        # A static strip has no jump share (None): it is not perfectly even.
        per_pair["jump_share"].append(share)
        per_pair["static"].append(share is None)
        per_pair["endpoint"].append(
            0.5 * (pixel_error(strip[0], targets[a]) + pixel_error(strip[-1], targets[b]))
        )
        per_pair["fragment"].append(float(np.mean(inks < reference.calibration["fragment_ink"])))
        per_pair["near_blank"].append(float(np.mean(inks < NEAR_BLANK_INK)))
        per_pair["precision"].append(float(np.mean([precision[j] for j in window])))
        per_pair["detour"].append(
            float(np.mean([copy[j] > 0 and neighbour[j] not in ends for j in window]))
        )
        rows.append([targets[a], *strip, targets[b]])
    moving = [share for share in per_pair["jump_share"] if share is not None]
    summary = {
        "pairs": len(pairs),
        "frames": frames,
        "endpoint_pixel_error": _interval(per_pair["endpoint"]),
        "jump_share": _interval(moving),
        "jump_share_pairs": len(moving),
        "jump_share_even": 1.0 / (frames - 1),
        "static_pairs": sum(per_pair["static"]),
        "jump_share_static_as_cut": _interval(
            [1.0 if share is None else share for share in per_pair["jump_share"]]
        ),
        "interior_fragment_rate": _interval(per_pair["fragment"]),
        "interior_near_blank_rate": _interval(per_pair["near_blank"]),
        "interior_precision": _interval(per_pair["precision"]),
        "detour_rate": _interval(per_pair["detour"]),
        "interior_collapsed_path_share": collapsed_share(programs[interior], reference.layout),
    }
    return summary, {"pairs": [list(pair) for pair in pairs], **per_pair}, rows


def reference_rows(reference: Reference, settings: Settings) -> dict[str, Any]:
    """Training renders (the ceiling), collages of training parts (no parameters) and a
    blank canvas, as many as the samples, through the same sample measures."""

    train, layout = reference.train, reference.layout
    rng = np.random.default_rng(settings.sample_seed)
    count = min(settings.samples, len(train.tokens))
    picks = rng.choice(len(train.tokens), count, replace=False)
    collages: list[Tensor] = []
    for _ in range(50 * settings.samples):
        if len(collages) == settings.samples:
            break
        sources = rng.integers(len(train.tokens), size=int(rng.integers(2, 5)))
        parts = [
            unflatten_program(train.tokens[int(i)], reference.template, layout) for i in sources
        ]
        composed = compose_program(parts, layout, rng)
        if composed is not None:
            collages.append(flatten_program(composed, layout))
    programs = torch.stack(collages)
    renders = _programs_to_renders(programs, layout, reference.template)
    blank = reference.features([BLANK]).expand(settings.samples, -1)
    return {
        "training_renders": score_set(
            reference,
            list(train.targets.numpy()[picks]),
            reference.library_features[picks],
            train.tokens[picks],
        )[0],
        "collages": score_set(reference, renders, reference.features(renders), programs)[0],
        "blank_canvas": score_set(reference, [BLANK] * settings.samples, blank, None)[0],
    }


# --------------------------------------------------------------------------- scoring


def _batched(function: Callable[[Tensor], Tensor], inputs: Tensor, batch: int) -> Tensor:
    return torch.cat([function(inputs[s : s + batch]).cpu() for s in range(0, len(inputs), batch)])


def _resources(
    device: torch.device, latency: LatencyReport, batched_ms: float, resident: float | None
) -> dict[str, Any]:
    record = resource_summary(device, train_seconds=None, latency=latency).as_record()
    record.update(
        train_seconds_null_reason="evaluation only",
        batched_ms_per_decode=batched_ms,
        resident_before_decoding_gib=resident,
        note="peak VRAM over decoding, resident weights included (the CLIP judge's when "
        "it runs on the GPU); "
        "latency is one greedy float32 decode at batch 1 (decode_one for a latent model), "
        "rendering excluded; "
        "other GPU work may have been running",
    )
    return record


def _resident(device: torch.device) -> float | None:
    return torch.cuda.memory_allocated(device) / 2**30 if device.type == "cuda" else None


def _sheet(samples: Renders, strips: list[Renders]) -> bytes:
    rows = [samples[s : s + 12] for s in range(0, min(len(samples), 48), 12)]
    return contact_sheet(rows + strips[:8])


@torch.no_grad()
def score_backend(
    backend: LatentBackend, reference: Reference, settings: Settings
) -> tuple[dict[str, Any], bytes]:
    """Every measure of one latent model: (report, contact sheet PNG). The backend
    encodes and decodes in IEEE float32 (TF32 off); CLIP runs as the caller set it."""

    if reference.calibration["copy_rule"] != settings.copy_rule:
        raise ValueError(
            f"the reference was calibrated under {reference.calibration['copy_rule']!r}, "
            f"the settings name {settings.copy_rule!r}"
        )
    device, shape, batch = reference.device, tuple(backend.latent_shape), settings.batch
    validation = reference.validation
    icons = min(settings.icons, len(validation.tokens))
    pairs = fixed_pairs(len(validation.tokens), settings.pairs, settings.pair_seed)
    ends = [v for pair in pairs for v in pair]

    def encode(tokens: Tensor, images: Tensor) -> Tensor:
        with strict_float32():
            return torch.cat(
                [
                    backend.encode(
                        tokens[s : s + batch].to(device), images[s : s + batch].to(device)
                    )
                    .float()
                    .cpu()
                    for s in range(0, len(tokens), batch)
                ]
            )

    def decode(latents: Tensor) -> Tensor:
        with strict_float32():
            return backend.decode(latents.to(device))

    def decode_single(latent: Tensor) -> Tensor:
        with strict_float32():
            return decode_one(backend, latent)

    report: dict[str, Any] = {"prior": settings.prior}
    generator = torch.Generator().manual_seed(settings.sample_seed)
    if settings.prior == "refit":
        prior = RefitPrior(encode(reference.train.tokens, reference.train.images))
        latents = prior.sample(settings.samples, generator)
        report["refit"] = {
            "fitted_on_training_icons": prior.count,
            "dimensions": len(prior.mean),
            "mean_norm": float(prior.mean.norm()),
            "covariance_trace": float(prior.covariance.trace()),
            "ridge": prior.ridge,
        }
    elif settings.prior == "normal":
        latents = backend.sample(settings.samples, generator, 1.0)
    else:
        raise ValueError(f"unknown prior {settings.prior!r}")
    means = encode(validation.tokens[:icons], validation.images[:icons])
    endpoints = encode(validation.tokens[ends], validation.images[ends])
    weights = torch.linspace(0.0, 1.0, settings.frames).view(-1, *([1] * len(shape)))
    path = torch.cat(
        [
            (1 - weights) * endpoints[2 * p] + weights * endpoints[2 * p + 1]
            for p in range(len(pairs))
        ]
    )

    resident = _resident(device)
    reset_peak_memory(device)
    started = time.perf_counter()
    samples = _batched(decode, latents, batch)
    batched_ms = (time.perf_counter() - started) * 1000.0 / len(samples)
    decoded = _batched(decode, means, batch)
    zero = decode(torch.zeros(1, *shape)).cpu()
    strips = _batched(decode, path, batch)
    first = latents[0].to(device)
    latency = measure_latency(
        lambda: decode_single(first),
        device=device,
        warmup=1,
        repeats=settings.latency_repeats,
    )
    report["resources"] = _resources(device, latency, batched_ms, resident)

    layout, template = reference.layout, reference.template
    sample_renders = _programs_to_renders(samples, layout, template)
    report["samples"], per_sample = score_set(
        reference, sample_renders, reference.features(sample_renders), samples
    )
    renders = _programs_to_renders(decoded, layout, template)
    zero_render = _programs_to_renders(zero, layout, template)[0]
    targets = validation.targets.numpy()
    errors = [pixel_error(r, targets[i]) for i, r in enumerate(renders)]
    zero_errors = [pixel_error(zero_render, targets[i]) for i in range(icons)]
    nearest = reference.validation_nearest_error[:icons]
    nearest_plain = reference.validation_nearest_error_without_mirrors[:icons]
    truth = validation.tokens[:icons]
    ratios = extent_ratios(decoded, truth, layout)
    report["reconstruction"] = {
        "icons": icons,
        "pixel_error": _interval(errors),
        "zero_latent_pixel_error": _interval(zero_errors),
        "reduction_vs_zero_latent": _interval(
            [z - e for z, e in zip(zero_errors, errors, strict=True)]
        ),
        "nearest_training_icon_pixel_error": _interval(nearest),
        "reduction_vs_nearest_training_icon": _interval(
            [n - e for n, e in zip(nearest, errors, strict=True)]
        ),
        "nearest_training_icon_note": "training renders and their mirrors",
        "nearest_training_icon_pixel_error_without_mirrors": _interval(nearest_plain),
        "reduction_vs_nearest_training_icon_without_mirrors": _interval(
            [n - e for n, e in zip(nearest_plain, errors, strict=True)]
        ),
        "exact_rate": float(
            np.mean([torch.equal(d, t) for d, t in zip(decoded, truth, strict=True)])
        ),
        "fragment_rate": _interval(
            [float(ink(r) < reference.calibration["fragment_ink"]) for r in renders]
        ),
        "collapsed_path_share": collapsed_share(decoded, layout),
        "true_collapsed_path_share": collapsed_share(truth, layout),
        "extent_ratio_median": float(np.median(ratios)),
    }
    report["interpolation"], per_pair, rows = score_interpolation(
        reference, strips, pairs, settings.frames
    )
    report["references"] = reference_rows(reference, settings)
    report["per_item"] = {
        "samples": per_sample,
        "reconstruction": {
            "pixel_error": errors,
            "zero_latent": zero_errors,
            "extent_ratio": ratios,
        },
        "interpolation": per_pair,
    }
    return report, _sheet(sample_renders, rows)


@torch.no_grad()
def score_crossfade(
    transcriber: RenderToProgram, reference: Reference, settings: Settings
) -> tuple[dict[str, Any], bytes]:
    """The pixel-crossfade reference: each frame an alpha blend of the pair's two renders,
    written down greedily by the transcriber in IEEE float32 (TF32 off); the
    interpolation measures only."""

    if reference.calibration["copy_rule"] != settings.copy_rule:
        raise ValueError(
            f"the reference was calibrated under {reference.calibration['copy_rule']!r}, "
            f"the settings name {settings.copy_rule!r}"
        )
    device, images = reference.device, reference.validation.images
    pairs = fixed_pairs(len(images), settings.pairs, settings.pair_seed)
    inputs = torch.cat([crossfade_frames(images[a], images[b], settings.frames) for a, b in pairs])

    def decode(frames: Tensor) -> Tensor:
        with strict_float32():
            return greedy_decode(transcriber, frames.to(device))

    resident = _resident(device)
    reset_peak_memory(device)
    started = time.perf_counter()
    programs = _batched(decode, inputs, settings.batch)
    batched_ms = (time.perf_counter() - started) * 1000.0 / len(programs)
    latency = measure_latency(
        lambda: decode(inputs[:1]), device=device, warmup=1, repeats=settings.latency_repeats
    )
    summary, per_pair, rows = score_interpolation(reference, programs, pairs, settings.frames)
    report = {
        "prior": "crossfade",
        "resources": _resources(device, latency, batched_ms, resident),
        "interpolation": summary,
        "per_item": {"interpolation": per_pair},
    }
    return report, contact_sheet(rows[:8])


# --------------------------------------------------------------------------- command line


def clip_embedder(device: torch.device) -> Embed:
    from PIL import Image

    from mojidiff.learning.omnisvg_study import Clip

    clip = Clip(str(device))

    def embed(renders: Sequence[np.ndarray]) -> Tensor:
        chunks = [
            clip.image([Image.fromarray(r) for r in renders[s : s + 64]])
            for s in range(0, len(renders), 64)
        ]
        return torch.cat(chunks) if chunks else torch.zeros(0, 512)

    return embed


def headline(report: dict[str, Any]) -> dict[str, Any]:
    """The report's main numbers, means only, for a terminal."""

    picks = (
        "samples.fragment_rate samples.near_blank_rate samples.clip_precision "
        "samples.clip_recall samples.copy_rate samples.distinct_programs "
        "samples.collapsed_path_share reconstruction.pixel_error "
        "reconstruction.reduction_vs_zero_latent reconstruction.collapsed_path_share "
        "reconstruction.extent_ratio_median interpolation.endpoint_pixel_error "
        "interpolation.jump_share interpolation.static_pairs "
        "interpolation.interior_fragment_rate "
        "interpolation.interior_near_blank_rate interpolation.interior_precision "
        "interpolation.detour_rate resources.batched_ms_per_decode "
        "resources.inference_ms_per_icon resources.peak_vram_gib"
    ).split()
    out: dict[str, Any] = {"run_id": report.get("run_id"), "prior": report.get("prior")}
    for pick in picks:
        block, key = pick.split(".")
        if block in report:
            value = report[block][key]
            out[pick] = value[0] if isinstance(value, list) else value
    return out


SCORING_SOURCES = (
    "src/mojidiff/learning/latent_metrics.py",
    "src/mojidiff/learning/precision.py",
    "src/mojidiff/gallery/latent_backends.py",
    "src/mojidiff/gallery/server.py",
    "src/mojidiff/learning/pixel_latent.py",
    "src/mojidiff/learning/latent.py",
    "src/mojidiff/learning/render2svg.py",
    "src/mojidiff/learning/fast_decode.py",
    "src/mojidiff/learning/autoregressive.py",
    "src/mojidiff/learning/omnisvg_study.py",
)
"""The code a report's numbers come from, hashed into every report: `git diff HEAD`
leaves untracked files out, so the commit alone may not identify it."""


def _file_sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(REPO_ROOT)) if resolved.is_relative_to(REPO_ROOT) else str(path)


def settings_hash(settings: Settings) -> str:
    """Eight hex digits naming a report's settings in its file name."""

    text = json.dumps(asdict(settings), sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:8]


def report_name(run_id: str, prior: str, settings: Settings) -> str:
    return f"metrics-{run_id}-{prior}-{settings.copy_rule}-{settings_hash(settings)}"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--registry-id", required=True, help="a model id in the gallery registry")
    parser.add_argument("--registry", type=Path, default=None, help="another registry file")
    parser.add_argument("--samples", type=int, default=Settings.samples)
    parser.add_argument("--pairs", type=int, default=Settings.pairs)
    parser.add_argument("--frames", type=int, default=Settings.frames)
    parser.add_argument("--icons", type=int, default=Settings.icons, help="reconstructions")
    parser.add_argument("--prior", choices=("normal", "refit"), default="normal")
    parser.add_argument(
        "--copy-rule",
        choices=COPY_RULES,
        required=True,
        help="which validation icons calibrate the copy thresholds (module docstring); "
        "choose and record it before any copy-based criterion is scored",
    )
    parser.add_argument("--latency-repeats", type=int, default=Settings.latency_repeats)
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument(
        "--overwrite", action="store_true", help="replace an existing report of the same name"
    )
    parser.add_argument(
        "--clip-device", choices=("cuda", "cpu"), default=None, help="default: the model's"
    )
    args = parser.parse_args(argv)
    if args.samples < 2 or args.pairs < 1 or args.frames < 3 or args.icons < 1:
        parser.error("need --samples >= 2, --pairs >= 1, --frames >= 3, --icons >= 1")
    settings = Settings(
        samples=args.samples,
        pairs=args.pairs,
        frames=args.frames,
        icons=args.icons,
        prior=args.prior,
        copy_rule=args.copy_rule,
        latency_repeats=args.latency_repeats,
    )

    from mojidiff.gallery.server import REGISTRY, load_model, load_registry
    from mojidiff.learning.omnisvg_study import CLIP_REPO, CLIP_REVISION
    from mojidiff.learning.openmoji_pilot import _load_program
    from mojidiff.learning.render2svg import _pilot_rows, load_corpus
    from mojidiff.vectorise.server import PILOT_CONFIG

    registry = args.registry or REGISTRY
    specs = {spec.id: spec for spec in load_registry(registry)}
    if args.registry_id not in specs:
        parser.error(f"no model {args.registry_id!r} in the registry ({', '.join(specs)})")
    spec = specs[args.registry_id]
    if spec.kind == "transcriber" and args.prior != "normal":
        parser.error("a transcriber scores the crossfade reference; --prior does not apply")
    name = report_name(
        spec.run_id, "crossfade" if spec.kind == "transcriber" else settings.prior, settings
    )
    written = [args.out / f"{name}.json", args.out / f"{name}.png"]
    if not args.overwrite and any(path.exists() for path in written):
        parser.error(f"{written[0]} exists; preserve it, or pass --overwrite")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    judge = torch.device(args.clip_device or device.type)
    started = time.perf_counter()
    # Everything - decoding, CLIP and the pixel measures - in IEEE float32.
    with strict_float32() as flags:
        plain, mirrored, layout, pilot, dataset_hash = load_corpus(PILOT_CONFIG, 144)
        template = _load_program(_pilot_rows(pilot)["primary/train"][0], pilot, layout.codec)
        # CPU CLIP never uses TF32; on the GPU the features depend on it.
        mode = judge.type if judge.type == "cpu" else f"{judge.type}-{precision_tag()}"
        reference = build_reference(
            layout,
            template,
            plain["primary/validation"],
            plain["primary/train"],
            clip_embedder(judge),
            device,
            extra_programs=mirrored["primary/train"].tokens,
            cache=CACHE_ROOT
            / "latent-metrics"
            / f"clip-{dataset_hash[:12]}-{CLIP_REVISION[:8]}-{mode}.pt",
            shortlist=settings.shortlist,
            copy_rule=settings.copy_rule,
        )
        loaded, _ = load_model(spec, layout, device)
        if spec.kind == "latent":
            body, sheet = score_backend(cast(LatentBackend, loaded), reference, settings)
        else:
            body, sheet = score_crossfade(cast(RenderToProgram, loaded), reference, settings)
    dirty = _git("diff", "HEAD")
    report = {
        "schema_version": 2,
        "registry_id": spec.id,
        "run_id": spec.run_id,
        "kind": spec.kind,
        "backend": spec.backend,
        "scored_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": _git("rev-parse", "HEAD").strip(),
        "dirty_patch_sha256": hashlib.sha256(dirty.encode()).hexdigest() if dirty else "",
        "uncommitted_sources": uncommitted_sources(),
        "source_sha256": {
            _repo_relative(path): _file_sha256(path)
            for path in (*(REPO_ROOT / source for source in SCORING_SOURCES), Path(registry))
        },
        "dataset_sha256": dataset_hash,
        "clip": {
            "repo": CLIP_REPO,
            "revision": CLIP_REVISION,
            "device": judge.type,
            "features": mode,
        },
        "decoding": (
            "greedy, IEEE float32 (TF32 off for matmul and cuDNN), batched greedy_decode; "
            "renders scored at 72 px"
        ),
        "float32_flags": flags,
        "settings": asdict(settings),
        "settings_hash": settings_hash(settings),
        "calibration": reference.calibration,
        **body,
        "wall_seconds": time.perf_counter() - started,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    written[0].write_text(json.dumps(report, indent=1, allow_nan=False) + "\n")
    written[1].write_bytes(sheet)
    print(
        json.dumps(
            {**headline(report), "report": str(written[0]), "wall_seconds": report["wall_seconds"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
