"""A small render-to-SVG model: read an icon's raster, write its codec program.

The first phase showed that 2,681 icons are too few for any generator to learn what an
emoji looks like from a label. A transcriber does not have to: the render carries the
shape, and the model only has to learn how this codec writes a shape down. That is the
bet this module makes.

The model is an encoder-decoder:

* **Encoder.** A three-stage strided convolution turns a 144 px render into an 18 x 18
  grid of features, followed by a few bidirectional transformer layers.
* **Decoder.** A causal transformer over the same flattened, padded program sequence
  Gate I used (1,376 positions, 418-way shared vocabulary), with cross-attention to the
  encoder grid and a key/value cache.

Decoding is constrained by `legal_mask`, so every output is a program the packed
validator accepts. Most positions are forced by the grammar - a median of about 1,260
of 1,376 on the training split - so the decoder feeds forced tokens as one chunk and
only calls the model where a choice exists. That is exact, and it is where the speed
comes from.

Evaluation is in pixels, on the fixed family-disjoint validation split, against a
zero-parameter baseline that has every advantage short of learning: the training icon
whose render is closest to the input render. A transcriber that cannot beat retrieval
has learned nothing a lookup table does not already know.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from mojidiff.learning.autoregressive import (
    SequenceLayout,
    flatten_program,
    legal_mask,
    teacher_forcing_inputs,
    unflatten_program,
)
from mojidiff.learning.telemetry import (
    measure_latency,
    reset_peak_memory,
    resource_summary,
)
from mojidiff.representation.packed import (
    PackedTensorProgram,
    serialize_packed_svg,
    validate_packed_tensor_program,
)
from mojidiff.representation.renderer import (
    IsolatedRenderError,
    RenderLimits,
    render_typed_svg_isolated,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = Path("/home/dev/.cache/mojidiff")
EVAL_SIZE = 72
"""Pixel metrics are read at the native 72-unit box, as in every earlier gate."""


# --------------------------------------------------------------------------- config


@dataclass(frozen=True)
class ModelConfig:
    image_size: int = 144
    d_model: int = 256
    heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 6
    feedforward: int = 1024
    dropout: float = 0.0


@dataclass(frozen=True)
class TrainConfig:
    steps: int = 2000
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    seed: int = 7001
    eval_every: int = 250
    eval_icons: int = 64
    train_icons: int | None = None
    """Restrict training to the first N training icons; the overfit fixture uses 4."""
    evaluate_on_train: bool = False
    """Score decodes on the training icons themselves, for the overfit fixture."""
    augment_mirror: float = 0.0
    """Probability of presenting an icon left-right mirrored, in pixels and tokens."""
    bf16: bool = True


@dataclass(frozen=True)
class Render2SvgConfig:
    slug: str
    hypothesis: str
    pilot_config: Path
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    final_eval_icons: int = 339
    clip_icons: int = 32
    parent_run: str | None = None
    notes: str = ""


def load_config(path: Path) -> Render2SvgConfig:
    root = yaml.safe_load(path.read_text())
    if not isinstance(root, dict) or root.get("schema_version") != 1:
        raise ValueError("render2svg config needs schema_version: 1")
    return Render2SvgConfig(
        slug=str(root["slug"]),
        hypothesis=str(root["hypothesis"]).strip(),
        pilot_config=Path(str(root["pilot_config"])),
        model=ModelConfig(**root.get("model", {})),
        training=TrainConfig(**root.get("training", {})),
        final_eval_icons=int(root.get("final_eval_icons", 339)),
        clip_icons=int(root.get("clip_icons", 32)),
        parent_run=root.get("parent_run"),
        notes=str(root.get("notes", "")).strip(),
    )


# --------------------------------------------------------------------------- data


def _composite_on_white(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[..., 3:4].astype(np.float32) / 255.0
    rgb = rgba[..., :3].astype(np.float32) * alpha + 255.0 * (1.0 - alpha)
    return cast(np.ndarray, np.clip(np.rint(rgb), 0, 255).astype(np.uint8))


def render_program_rgb(
    program: PackedTensorProgram, layout: SequenceLayout, size: int
) -> np.ndarray | None:
    """A program's render on white, or None when it does not serialize or render."""

    try:
        svg = serialize_packed_svg(program, layout.codec, layout.total_segment_slots)
        limits = RenderLimits(max_paths=layout.codec.max_paths, timeout_seconds=20)
        _, rgba = render_typed_svg_isolated(svg, size, limits)
    except (IsolatedRenderError, ValueError):
        return None
    return _composite_on_white(rgba)


def mirror_program_tokens(tokens: Tensor, layout: SequenceLayout) -> Tensor | None:
    """Mirror a program left to right in token space, or None if it cannot be exact.

    The quarter-unit endpoint lattice is closed under x -> 72 - x: token t maps to
    290 - t. The control lattice spans -8 to 96 in quarter units, so its token t maps
    to 354 - t and stays in range only while the mirrored value does; a program that
    would leave it is not mirrored rather than clamped. Within a segment the slots are
    (x, y) pairs, control pairs first, so only the even slots of the pairs the kind
    uses change. Painter order and every style field are untouched.
    """

    from mojidiff.learning.autoregressive import PATH_FIELDS, PATH_STRIDE, SEGMENT_STRIDE
    from mojidiff.representation.program import SegmentType

    codec = layout.codec
    if (
        codec.coordinate_bins != 289
        or codec.effective_control_coordinate_bins != 417
        or (codec.control_coordinate_min, codec.control_coordinate_max) != (-8.0, 96.0)
    ):
        return None
    out = tokens.clone()
    for path in range(layout.codec.max_paths):
        position = path * PATH_STRIDE + len(PATH_FIELDS)
        token = int(tokens[position])
        if token >= 1:
            out[position] = 290 - token
    controls = {int(SegmentType.LINE): 0, int(SegmentType.QUAD): 1, int(SegmentType.CUBIC): 2}
    for segment in range(layout.total_segment_slots):
        base = layout.path_positions + segment * SEGMENT_STRIDE
        kind = int(tokens[base])
        if kind not in controls:
            continue
        pairs = controls[kind] + 1
        for pair in range(pairs):
            position = base + 1 + 2 * pair
            token = int(tokens[position])
            if token < 1:
                return None
            if pair < controls[kind]:
                mirrored = 354 - token
                if not 1 <= mirrored <= 417:
                    return None
                out[position] = mirrored
            else:
                out[position] = 290 - token
    return out


@dataclass
class SplitData:
    hexcodes: list[str]
    tokens: Tensor  # (N, L) long
    masks: np.ndarray  # (N, L*V/8) packed bits
    images: Tensor  # (N, S, S, 3) uint8, model input
    targets: Tensor  # (N, 72, 72, 3) uint8, evaluation reference
    shape: tuple[int, int]

    def mask_batch(self, indices: np.ndarray) -> Tensor:
        bits = np.unpackbits(self.masks[indices], axis=1)
        return torch.from_numpy(bits.reshape(len(indices), *self.shape)).bool()

    def subset(self, count: int) -> SplitData:
        return SplitData(
            self.hexcodes[:count],
            self.tokens[:count],
            self.masks[:count],
            self.images[:count],
            self.targets[:count],
            self.shape,
        )


def _prepare_row(args: tuple[Any, Any, Any, SequenceLayout, int, bool]) -> dict[str, Any]:
    from mojidiff.learning.openmoji_pilot import _load_program

    row, pilot, codec, layout, size, mirror = args
    program = _load_program(row, pilot, codec)
    tokens = flatten_program(program, layout)
    record: dict[str, Any] = {"hexcode": row.hexcode}
    variants = [("plain", tokens)]
    if mirror:
        mirrored = mirror_program_tokens(tokens, layout)
        if mirrored is not None:
            variants.append(("mirror", mirrored))
    for name, sequence in variants:
        candidate = unflatten_program(sequence, program, layout)
        try:
            validate_packed_tensor_program(candidate, codec, layout.total_segment_slots)
        except ValueError:
            continue
        image = render_program_rgb(candidate, layout, size)
        target = render_program_rgb(candidate, layout, EVAL_SIZE)
        if image is None or target is None:
            if name == "plain":
                raise RuntimeError(f"reference icon failed to render: {row.hexcode}")
            continue
        mask = torch.stack([legal_mask(p, sequence, layout) for p in range(layout.length)])
        record[name] = {
            "tokens": sequence.numpy().astype(np.int16),
            "mask": np.packbits(mask.numpy().reshape(-1)),
            "image": image,
            "target": target,
        }
    return record


def _cache_key(pilot_config: Path, image_size: int, mirror: bool) -> str:
    digest = hashlib.sha256()
    digest.update(pilot_config.read_bytes())
    digest.update(f"size={image_size};mirror={mirror};v=1".encode())
    for name in ("render2svg.py", "autoregressive.py", "openmoji_pilot.py"):
        digest.update((Path(__file__).parent / name).read_bytes())
    return digest.hexdigest()[:12]


def load_corpus(
    pilot_config: Path, image_size: int, *, mirror: bool = True, workers: int = 20
) -> tuple[dict[str, SplitData], dict[str, SplitData], SequenceLayout, Any, str]:
    """Plain and mirrored splits, built once and cached under `CACHE_ROOT`.

    Returns (plain splits, mirrored splits, layout, pilot config, dataset hash). The
    dataset hash covers every cached array, so it identifies exactly what trained.
    """

    from mojidiff.learning.openmoji_pilot import (
        _selected_codec,
        load_openmoji_pilot_config,
        load_pilot_index,
    )

    pilot = load_openmoji_pilot_config(pilot_config)
    by_split, _, _ = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)
    key = _cache_key(pilot_config, image_size, mirror)
    path = CACHE_ROOT / f"render2svg-corpus-{key}.npz"
    shape = (layout.length, layout.vocabulary)
    names = ("primary/train", "primary/validation", "primary/test")
    if not path.exists():
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for name in names:
                rows = by_split[name]
                jobs = [(row, pilot, codec, layout, image_size, mirror) for row in rows]
                records = list(pool.map(_prepare_row, jobs, chunksize=8))
                for variant in ("plain", "mirror"):
                    kept = [r for r in records if variant in r]
                    prefix = f"{variant}:{name}"
                    arrays[f"{prefix}:hexcodes"] = np.array([r["hexcode"] for r in kept])
                    for part in ("tokens", "mask", "image", "target"):
                        arrays[f"{prefix}:{part}"] = np.stack([r[variant][part] for r in kept])
                print(json.dumps({"prepared": name, "icons": len(records)}), flush=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez(temporary, **arrays)  # type: ignore[arg-type]
        temporary.rename(path)
    data = np.load(path)
    dataset_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    def split(variant: str, name: str) -> SplitData:
        prefix = f"{variant}:{name}"
        return SplitData(
            hexcodes=[str(h) for h in data[f"{prefix}:hexcodes"]],
            tokens=torch.from_numpy(data[f"{prefix}:tokens"].astype(np.int64)),
            masks=data[f"{prefix}:mask"],
            images=torch.from_numpy(data[f"{prefix}:image"]),
            targets=torch.from_numpy(data[f"{prefix}:target"]),
            shape=shape,
        )

    plain = {name: split("plain", name) for name in names}
    mirrored = {name: split("mirror", name) for name in names} if mirror else {}
    return plain, mirrored, layout, pilot, dataset_hash


# --------------------------------------------------------------------------- model


class _Attention(nn.Module):
    def __init__(self, d_model: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.query = nn.Linear(d_model, d_model)
        self.key_value = nn.Linear(d_model, 2 * d_model)
        self.project = nn.Linear(d_model, d_model)

    def split(self, value: Tensor) -> Tensor:
        batch, length, width = value.shape
        return value.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2)

    def keys_values(self, source: Tensor) -> tuple[Tensor, Tensor]:
        key, value = self.key_value(source).chunk(2, dim=-1)
        return self.split(key), self.split(value)

    def attend(self, hidden: Tensor, key: Tensor, value: Tensor, causal: bool) -> Tensor:
        query = self.split(self.query(hidden))
        length, total = query.shape[2], key.shape[2]
        if not causal or length == 1:
            mask = None
            is_causal = False
        elif length == total:
            mask = None
            is_causal = True
        else:
            mask = torch.ones(length, total, dtype=torch.bool, device=hidden.device).tril(
                diagonal=total - length
            )
            is_causal = False
        attended = F.scaled_dot_product_attention(
            query, key, value, attn_mask=mask, is_causal=is_causal
        )
        batch = hidden.shape[0]
        return cast(Tensor, self.project(attended.transpose(1, 2).reshape(batch, length, -1)))


class _EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = _Attention(d_model, heads)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward), nn.GELU(), nn.Linear(feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor) -> Tensor:
        normed = self.norm_attention(hidden)
        key, value = self.attention.keys_values(normed)
        hidden = hidden + self.dropout(self.attention.attend(normed, key, value, causal=False))
        forwarded: Tensor = self.feedforward(self.norm_feedforward(hidden))
        return hidden + cast(Tensor, self.dropout(forwarded))


class _DecoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        self.norm_self = nn.LayerNorm(d_model)
        self.self_attention = _Attention(d_model, heads)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attention = _Attention(d_model, heads)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward), nn.GELU(), nn.Linear(feedforward, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden: Tensor,
        memory: tuple[Tensor, Tensor],
        past: tuple[Tensor, Tensor] | None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        normed = self.norm_self(hidden)
        key, value = self.self_attention.keys_values(normed)
        if past is not None:
            key = torch.cat((past[0], key), dim=2)
            value = torch.cat((past[1], value), dim=2)
        hidden = hidden + self.dropout(self.self_attention.attend(normed, key, value, causal=True))
        hidden = hidden + self.dropout(
            self.cross_attention.attend(self.norm_cross(hidden), *memory, causal=False)
        )
        hidden = hidden + self.dropout(self.feedforward(self.norm_feedforward(hidden)))
        return hidden, (key, value)


class RenderToProgram(nn.Module):
    """Encode a render, decode its program left to right with a key/value cache."""

    def __init__(self, layout: SequenceLayout, config: ModelConfig) -> None:
        super().__init__()
        if config.image_size % 8:
            raise ValueError("image_size must be divisible by 8")
        self.layout = layout
        self.config = config
        width = config.d_model
        self.stem = nn.Sequential(
            nn.Conv2d(3, width // 4, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 4, width // 2, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(width // 2, width, 3, stride=2, padding=1),
        )
        grid = config.image_size // 8
        self.grid_embedding = nn.Parameter(torch.zeros(1, grid * grid, width))
        nn.init.normal_(self.grid_embedding, std=0.02)
        self.encoder = nn.ModuleList(
            _EncoderBlock(width, config.heads, config.feedforward, config.dropout)
            for _ in range(config.encoder_layers)
        )
        self.encoder_norm = nn.LayerNorm(width)
        self.token_embedding = nn.Embedding(layout.vocabulary, width)
        self.position_embedding = nn.Embedding(layout.length, width)
        self.decoder = nn.ModuleList(
            _DecoderBlock(width, config.heads, config.feedforward, config.dropout)
            for _ in range(config.decoder_layers)
        )
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, layout.vocabulary)

    def encode(self, images: Tensor) -> list[tuple[Tensor, Tensor]]:
        """Per-decoder-layer cross-attention keys and values for uint8 NHWC renders."""

        pixels = images.permute(0, 3, 1, 2).to(self.grid_embedding.dtype) / 127.5 - 1.0
        features = self.stem(pixels).flatten(2).transpose(1, 2) + self.grid_embedding
        for block in self.encoder:
            features = block(features)
        memory = self.encoder_norm(features)
        return [
            cast(_DecoderBlock, block).cross_attention.keys_values(memory) for block in self.decoder
        ]

    def decode(
        self,
        inputs: Tensor,
        memory: list[tuple[Tensor, Tensor]],
        cache: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        positions = torch.arange(offset, offset + inputs.shape[1], device=inputs.device)
        hidden = self.token_embedding(inputs) + self.position_embedding(positions)[None]
        updated: list[tuple[Tensor, Tensor]] = []
        for index, block in enumerate(self.decoder):
            past = cache[index] if cache is not None else None
            hidden, present = block(hidden, memory[index], past)
            updated.append(present)
        return self.head(self.norm(hidden)), updated

    def forward(self, images: Tensor, tokens: Tensor) -> Tensor:
        """Teacher-forced logits for every position."""

        shifted, _ = teacher_forcing_inputs(tokens, self.layout)
        logits, _ = self.decode(shifted, self.encode(images))
        return logits


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


# --------------------------------------------------------------------------- decoding


@dataclass
class DecodeStats:
    model_calls: int = 0
    positions: int = 0


@torch.no_grad()
def greedy_decode(
    model: RenderToProgram,
    images: Tensor,
    *,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
    stats: DecodeStats | None = None,
) -> Tensor:
    """Decode a batch of programs under the grammar, skipping forced positions.

    Tokens the grammar forces for every icon in the batch are not predicted; they are
    queued and fed to the cache in one chunk the next time any icon has a choice. The
    logits at a free position are therefore exactly those of a position-by-position
    decode, at a fraction of the calls.
    """

    layout = model.layout
    device = images.device
    batch = images.shape[0]
    memory = model.encode(images)
    decoded = torch.zeros((batch, layout.length), dtype=torch.long)
    cache: list[tuple[Tensor, Tensor]] | None = None
    fed = 0
    calls = 0
    for position in range(layout.length):
        masks = torch.stack([legal_mask(position, decoded[b], layout) for b in range(batch)])
        counts = masks.sum(dim=1)
        if bool((counts == 0).any()):
            raise ValueError(f"no legal token at position {position}")
        if bool((counts == 1).all()):
            decoded[:, position] = masks.to(torch.long).argmax(dim=1)
            continue
        shifted = torch.zeros((batch, position + 1 - fed), dtype=torch.long)
        for index, absolute in enumerate(range(fed, position + 1)):
            shifted[:, index] = decoded[:, absolute - 1] if absolute > 0 else 0
        logits, cache = model.decode(shifted.to(device), memory, cache, offset=fed)
        calls += 1
        fed = position + 1
        scores = logits[:, -1].float().masked_fill(~masks.to(device), float("-inf"))
        if temperature > 0.0:
            probabilities = torch.softmax(scores / temperature, dim=-1)
            choice = torch.multinomial(probabilities, 1, generator=generator)[:, 0]
        else:
            choice = scores.argmax(dim=-1)
        decoded[:, position] = choice.cpu()
    if stats is not None:
        stats.model_calls += calls
        stats.positions += layout.length
    return decoded


# --------------------------------------------------------------------------- evaluation


def _programs_to_renders(
    tokens: Tensor, layout: SequenceLayout, template: PackedTensorProgram
) -> list[np.ndarray | None]:
    programs = [unflatten_program(row, template, layout) for row in tokens]
    with ThreadPoolExecutor(max_workers=16) as pool:
        return list(pool.map(lambda p: render_program_rgb(p, layout, EVAL_SIZE), programs))


def pixel_error(render: np.ndarray | None, target: np.ndarray) -> float:
    """Mean absolute RGB error in [0, 1] at 72 px; a missing render scores as 1."""

    if render is None:
        return 1.0
    return float(np.abs(render.astype(np.float32) - target.astype(np.float32)).mean() / 255.0)


def nearest_training_icon(
    queries: Tensor, library: Tensor, device: torch.device
) -> tuple[Tensor, Tensor]:
    """For each query render, the index and pixel error of the closest library render."""

    library_flat = library.reshape(len(library), -1).to(device, torch.float32) / 255.0
    best_index = torch.zeros(len(queries), dtype=torch.long)
    best_error = torch.zeros(len(queries))
    for start in range(0, len(queries), 32):
        chunk = queries[start : start + 32].reshape(-1, library_flat.shape[1])
        chunk = chunk.to(device, torch.float32) / 255.0
        errors = torch.cdist(chunk, library_flat, p=1) / library_flat.shape[1]
        values, indices = errors.min(dim=1)
        best_index[start : start + 32] = indices.cpu()
        best_error[start : start + 32] = values.cpu()
    return best_index, best_error


def bootstrap_mean_interval(
    values: Sequence[float], seed: int = 0, resamples: int = 4000
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = rng.choice(array, size=(resamples, len(array)), replace=True).mean(axis=1)
    return float(array.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


@torch.no_grad()
def teacher_forced_loss(
    model: RenderToProgram, split: SplitData, device: torch.device, count: int, bf16: bool
) -> tuple[float, float]:
    """Mean NLL per free token and free-token accuracy on the first `count` icons."""

    total = 0.0
    correct = 0
    tokens_seen = 0
    for start in range(0, min(count, len(split.tokens)), 32):
        indices = np.arange(start, min(start + 32, count, len(split.tokens)))
        tokens = split.tokens[indices].to(device)
        masks = split.mask_batch(indices).to(device)
        images = split.images[indices].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            logits = model(images, tokens)
        free = masks.sum(dim=-1) > 1
        scores = logits.float().masked_fill(~masks, float("-inf"))[free]
        targets = tokens[free]
        total += float(F.cross_entropy(scores, targets, reduction="sum"))
        correct += int((scores.argmax(dim=-1) == targets).sum())
        tokens_seen += int(free.sum())
    return total / max(tokens_seen, 1), correct / max(tokens_seen, 1)


@torch.no_grad()
def decode_and_score(
    model: RenderToProgram,
    split: SplitData,
    template: PackedTensorProgram,
    device: torch.device,
    count: int,
    *,
    batch_size: int = 64,
    bf16: bool = True,
) -> tuple[list[float], Tensor, list[np.ndarray | None], DecodeStats]:
    """Greedy-decode the first `count` icons; per-icon pixel errors against their renders."""

    model.eval()
    stats = DecodeStats()
    outputs: list[Tensor] = []
    for start in range(0, count, batch_size):
        images = split.images[start : min(start + batch_size, count)].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16 and device.type == "cuda"):
            outputs.append(greedy_decode(model, images, stats=stats))
    tokens = torch.cat(outputs)
    renders = _programs_to_renders(tokens, model.layout, template)
    errors = [
        pixel_error(render, split.targets[index].numpy()) for index, render in enumerate(renders)
    ]
    return errors, tokens, renders, stats


def clip_retrieval(
    renders: Sequence[np.ndarray | None], references: Sequence[np.ndarray], device: str
) -> dict[str, float]:
    """Gate N's metric: is each drawing closest, in CLIP space, to its own reference?"""

    from PIL import Image

    from mojidiff.learning.omnisvg_study import Clip

    clip = Clip(device)
    reference_features = clip.image([Image.fromarray(r) for r in references])
    ranks: list[int] = []
    similarity: list[float] = []
    for index, render in enumerate(renders):
        if render is None:
            ranks.append(len(references))
            continue
        feature = clip.image([Image.fromarray(render)])
        scores = (feature @ reference_features.T)[0]
        own = float(scores[index])
        similarity.append(own)
        ranks.append(1 + int((scores > own).sum()))
    return {
        "top1_rate": sum(1 for rank in ranks if rank == 1) / len(ranks),
        "top5_rate": sum(1 for rank in ranks if rank <= 5) / len(ranks),
        "mean_rank": float(np.mean(ranks)),
        "clip_to_reference_mean": float(np.mean(similarity)) if similarity else 0.0,
        "chance_top1": 1.0 / len(references),
        "icons": len(references),
    }


def contact_sheet(rows: Sequence[Sequence[np.ndarray | None]], size: int = EVAL_SIZE) -> bytes:
    """A PNG grid, one row per icon: reference, model, baseline."""

    import io

    from PIL import Image

    columns = max(len(row) for row in rows)
    sheet = Image.new("RGB", (columns * (size + 4), len(rows) * (size + 4)), "white")
    for y, row in enumerate(rows):
        for x, tile in enumerate(row):
            image = (
                Image.fromarray(tile)
                if tile is not None
                else Image.new("RGB", (size, size), (255, 200, 200))
            )
            sheet.paste(image, (x * (size + 4) + 2, y * (size + 4) + 2))
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return buffer.getvalue()


# --------------------------------------------------------------------------- training


def _schedule(step: int, config: TrainConfig) -> float:
    if step < config.warmup_steps:
        return (step + 1) / config.warmup_steps
    progress = (step - config.warmup_steps) / max(config.steps - config.warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def _batches(
    train: SplitData, mirrored: SplitData | None, config: TrainConfig, rng: np.random.Generator
) -> Iterator[tuple[Tensor, Tensor, Tensor]]:
    """(images, tokens, masks) on CPU; mirrored icons stand in with the set probability."""

    mirror_index: dict[str, int] = (
        {h: i for i, h in enumerate(mirrored.hexcodes)} if mirrored is not None else {}
    )
    count = len(train.tokens)
    while True:
        indices = rng.choice(count, size=config.batch_size, replace=count < config.batch_size)
        images = train.images[indices].clone()
        tokens = train.tokens[indices].clone()
        masks = train.mask_batch(indices)
        if mirrored is not None and config.augment_mirror > 0.0:
            for slot, index in enumerate(indices):
                other = mirror_index.get(train.hexcodes[index])
                if other is not None and rng.random() < config.augment_mirror:
                    images[slot] = mirrored.images[other]
                    tokens[slot] = mirrored.tokens[other]
                    masks[slot] = mirrored.mask_batch(np.array([other]))[0]
        yield images, tokens, masks


def _git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout


def run_identity(config_path: Path, dataset_hash: str, slug: str) -> dict[str, str]:
    commit = _git("rev-parse", "HEAD").strip()
    dirty = _git("diff", "HEAD")
    config_hash = hashlib.sha256(config_path.read_bytes()).hexdigest()
    run_id = f"{slug}-{commit[:7]}-{config_hash[:8]}-{dataset_hash[:8]}"
    return {
        "run_id": run_id,
        "git_commit": commit,
        "dirty_patch_sha256": hashlib.sha256(dirty.encode()).hexdigest() if dirty else "",
        "config_sha256": config_hash,
        "dataset_sha256": dataset_hash,
    }


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _append_registry(run_id: str, state: str, **extra: Any) -> None:
    from mojidiff.orchestration.registry import append_event

    append_event(
        REPO_ROOT / "state" / "runs.jsonl",
        {"run_id": run_id, "state": state, "timestamp": _now(), **extra},
    )


def _write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(value, sort_keys=False, width=88, allow_unicode=True))


def train_and_evaluate(config_path: Path) -> dict[str, Any]:
    config = load_config(config_path)
    torch.manual_seed(config.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plain, mirrored, layout, pilot, dataset_hash = load_corpus(
        config.pilot_config, config.model.image_size
    )
    identity = run_identity(config_path, dataset_hash, config.slug)
    run_id = identity["run_id"]
    run_dir = REPO_ROOT / "runs" / run_id
    checkpoint_dir = CACHE_ROOT / "runs" / run_id
    if run_dir.exists():
        raise SystemExit(f"run directory already exists; preserve it and change the id: {run_id}")
    run_dir.mkdir(parents=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train = plain["primary/train"]
    if config.training.train_icons is not None:
        train = train.subset(config.training.train_icons)
    validation = plain["primary/validation"]
    evaluation = train if config.training.evaluate_on_train else validation
    mirror_train = mirrored.get("primary/train") if config.training.augment_mirror else None

    from mojidiff.learning.openmoji_pilot import _load_program, _select_rows

    by_split_rows = _pilot_rows(pilot)
    template = _load_program(by_split_rows["primary/train"][0], pilot, layout.codec)

    model = RenderToProgram(layout, config.model).to(device)
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
        "git_commit": identity["git_commit"],
        "dirty_patch_sha256": identity["dirty_patch_sha256"],
        "config": str(config_path),
        "config_sha256": identity["config_sha256"],
        "config_resolved": {
            "model": asdict(config.model),
            "training": asdict(config.training),
            "final_eval_icons": config.final_eval_icons,
            "clip_icons": config.clip_icons,
        },
        "dataset": {
            "pilot_config": str(config.pilot_config),
            "cache_sha256": dataset_hash,
            "train_icons": len(train.tokens),
            "evaluated_on": "primary/train"
            if config.training.evaluate_on_train
            else "primary/validation",
        },
        "seed": config.training.seed,
        "determinism": "seeded; cuDNN and SDPA kernels not forced deterministic",
        "model_parameters": parameter_count(model),
        "command": f".venv/bin/python -m mojidiff.learning.render2svg --config {config_path}",
        "outputs": {
            "run_dir": str(run_dir.relative_to(REPO_ROOT)),
            "checkpoints": str(checkpoint_dir),
        },
        "notes": config.notes,
    }
    _write_yaml(run_dir / "run.yaml", record)
    _append_registry(run_id, "running", config=str(config_path), slug=config.slug)
    print(json.dumps({"run_id": run_id, "parameters": parameter_count(model)}), flush=True)

    metrics_path = run_dir / "metrics.jsonl"
    rng = np.random.default_rng(config.training.seed)
    batches = _batches(train, mirror_train, config.training, rng)
    bf16 = config.training.bf16 and device.type == "cuda"
    best_error = float("inf")
    best_step = 0
    reset_peak_memory(device)
    started = time.perf_counter()
    try:
        for step in range(1, config.training.steps + 1):
            model.train()
            images, tokens, masks = next(batches)
            images, tokens, masks = images.to(device), tokens.to(device), masks.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
                logits = model(images, tokens)
            free = masks.sum(dim=-1) > 1
            loss = F.cross_entropy(
                logits.float().masked_fill(~masks, float("-inf"))[free], tokens[free]
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()  # type: ignore[no-untyped-call]
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if step % config.training.eval_every == 0 or step == config.training.steps:
                nll, accuracy = teacher_forced_loss(
                    model, evaluation, device, config.training.eval_icons, bf16
                )
                errors, _, _, _ = decode_and_score(
                    model,
                    evaluation,
                    template,
                    device,
                    min(config.training.eval_icons, len(evaluation.tokens)),
                    bf16=bf16,
                )
                entry = {
                    "step": step,
                    "train_loss": float(loss.detach()),
                    "eval_nll_per_free_token": nll,
                    "eval_free_token_accuracy": accuracy,
                    "eval_pixel_error_mean": float(np.mean(errors)),
                    "elapsed_seconds": time.perf_counter() - started,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                with metrics_path.open("a") as handle:
                    handle.write(json.dumps(entry) + "\n")
                print(json.dumps(entry), flush=True)
                current = float(entry["eval_pixel_error_mean"])
                if current < best_error:
                    best_error = current
                    best_step = step
                    torch.save(
                        {"model": model.state_dict(), "step": step, "config": asdict(config.model)},
                        checkpoint_dir / "best.pt",
                    )
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": config.training.steps,
                "config": asdict(config.model),
            },
            checkpoint_dir / "latest.pt",
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

    # ---- final evaluation on the selected checkpoint
    final_count = min(config.final_eval_icons, len(evaluation.tokens))
    final = evaluation.subset(final_count)
    errors, tokens, renders, stats = decode_and_score(
        model, final, template, device, final_count, bf16=bf16
    )
    exact = [bool(torch.equal(tokens[i], final.tokens[i])) for i in range(final_count)]
    library = plain["primary/train"]
    nearest, nearest_errors = nearest_training_icon(final.targets, library.targets, device)
    if config.training.evaluate_on_train:
        # Retrieval against a library that contains the query is exact by construction.
        baseline_errors = [float("nan")] * final_count
    else:
        baseline_errors = [float(v) for v in nearest_errors]
    blank = np.full((EVAL_SIZE, EVAL_SIZE, 3), 255, dtype=np.uint8)
    blank_errors = [pixel_error(blank, final.targets[i].numpy()) for i in range(final_count)]
    differences = [b - m for m, b in zip(errors, baseline_errors, strict=True)]
    wins = sum(1 for d in differences if d > 0)

    summary: dict[str, Any] = {
        "icons": final_count,
        "model_pixel_error": bootstrap_mean_interval(errors),
        "model_pixel_error_median": float(np.median(errors)),
        "exact_program_rate": sum(exact) / final_count,
        "rendered_rate": sum(1 for r in renders if r is not None) / final_count,
        "blank_pixel_error": float(np.mean(blank_errors)),
        "selected_step": best_step,
    }
    if not config.training.evaluate_on_train:
        summary["nearest_training_icon_pixel_error"] = bootstrap_mean_interval(baseline_errors)
        summary["model_minus_baseline_error_reduction"] = bootstrap_mean_interval(differences)
        summary["icons_model_beats_baseline"] = wins

    # ---- Gate N's metric on Gate N's 32 icons
    clip_block: dict[str, Any] | None = None
    if not config.training.evaluate_on_train and config.clip_icons:
        selected = _select_rows(
            by_split_rows["primary/validation"], config.clip_icons, pilot.seed + 1
        )
        positions = {h: i for i, h in enumerate(validation.hexcodes)}
        chosen = [positions[row.hexcode] for row in selected]
        subset_images = validation.images[chosen].to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            clip_tokens = greedy_decode(model, subset_images)
        clip_renders = _programs_to_renders(clip_tokens, layout, template)
        references = [validation.targets[i].numpy() for i in chosen]
        clip_nearest, _ = nearest_training_icon(validation.targets[chosen], library.targets, device)
        baseline_renders = [library.targets[int(i)].numpy() for i in clip_nearest]
        clip_block = {
            "model_greedy": clip_retrieval(clip_renders, references, str(device)),
            "nearest_training_icon": clip_retrieval(baseline_renders, references, str(device)),
            "gate_n_omnisvg_zero_shot_top1": 39 / 64,
            "gate_n_omnisvg_best_of_12_top1": 53 / 64,
        }
        summary["clip_retrieval"] = clip_block

    # ---- latency, one icon at a time, the way the demo would call it
    sample = final.images[:1].to(device)

    def one_icon() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            greedy_decode(model, sample)

    latency = measure_latency(one_icon, device=device, warmup=2, repeats=10)
    single = DecodeStats()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
        greedy_decode(model, sample, stats=single)
    summary["model_calls_per_icon"] = single.model_calls
    batch_images = final.images[: min(64, final_count)].to(device)

    def batch_of_icons() -> None:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=bf16):
            greedy_decode(model, batch_images)

    batched = measure_latency(
        batch_of_icons, device=device, icons_per_call=len(batch_images), warmup=1, repeats=3
    )
    resource = resource_summary(device, train_seconds=train_seconds, latency=latency).as_record()
    resource["batched_ms_per_icon"] = batched.median_ms_per_icon
    resource["batched_icons_per_call"] = batched.icons_per_call
    resource["excludes"] = "rendering the output SVG to pixels"

    # ---- artifacts
    sheet_rows: list[list[np.ndarray | None]] = []
    for index in range(min(24, final_count)):
        row: list[np.ndarray | None] = [final.targets[index].numpy(), renders[index]]
        if not config.training.evaluate_on_train:
            row.append(library.targets[int(nearest[index])].numpy())
        sheet_rows.append(row)
    (run_dir / "samples.png").write_bytes(contact_sheet(sheet_rows))
    per_icon = [
        {
            "hexcode": final.hexcodes[i],
            "model_pixel_error": errors[i],
            "baseline_pixel_error": baseline_errors[i],
            "exact_program": exact[i],
        }
        for i in range(final_count)
    ]
    (run_dir / "per_icon.jsonl").write_text("".join(json.dumps(r) + "\n" for r in per_icon))
    artifacts = {
        "checkpoint_best": str(checkpoint_dir / "best.pt"),
        "checkpoint_latest": str(checkpoint_dir / "latest.pt"),
        "samples": "samples.png (rows: reference, model, nearest training icon)",
        "per_icon": "per_icon.jsonl",
    }
    (run_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2) + "\n")

    baselines: dict[str, Any] = {"blank_canvas_pixel_error": summary["blank_pixel_error"]}
    if not config.training.evaluate_on_train:
        baselines["nearest_training_icon_pixel_error"] = summary[
            "nearest_training_icon_pixel_error"
        ][0]
    if clip_block is not None:
        baselines["gate_n_omnisvg_zero_shot_top1"] = clip_block["gate_n_omnisvg_zero_shot_top1"]
        baselines["nearest_training_icon_clip_top1"] = clip_block["nearest_training_icon"][
            "top1_rate"
        ]
    record.update(
        state="completed",
        completed_at=_now(),
        resource=resource,
        baselines=baselines,
        result=summary,
    )
    _write_yaml(run_dir / "run.yaml", record)
    (run_dir / "result.md").write_text(_result_markdown(record))
    _append_registry(run_id, "completed", result_file=f"runs/{run_id}/result.md")
    print(json.dumps({"completed": run_id, "result": summary, "resource": resource}), flush=True)
    return record


def _pilot_rows(pilot: Any) -> dict[str, Any]:
    from mojidiff.learning.openmoji_pilot import load_pilot_index

    by_split, _, _ = load_pilot_index(pilot)
    return by_split


def _format_interval(value: Any) -> str:
    if isinstance(value, list | tuple) and len(value) == 3:
        return f"{value[0]:.4f} [{value[1]:.4f}, {value[2]:.4f}]"
    return str(value)


def _result_markdown(record: dict[str, Any]) -> str:
    result = record["result"]
    resource = record["resource"]
    lines = [
        f"# {record['run_id']}",
        "",
        f"**Hypothesis.** {record['hypothesis']}",
        "",
        "## Result",
        "",
        "| measure | value |",
        "| --- | --- |",
        f"| icons evaluated | {result['icons']} ({record['dataset']['evaluated_on']}) |",
        f"| model pixel error, mean [95% CI] | {_format_interval(result['model_pixel_error'])} |",
        f"| model pixel error, median | {result['model_pixel_error_median']:.4f} |",
        f"| exact program rate | {result['exact_program_rate']:.3f} |",
        f"| rendered rate | {result['rendered_rate']:.3f} |",
        f"| blank canvas pixel error | {result['blank_pixel_error']:.4f} |",
    ]
    if "nearest_training_icon_pixel_error" in result:
        lines += [
            "| nearest training icon pixel error | "
            f"{_format_interval(result['nearest_training_icon_pixel_error'])} |",
            "| error reduction vs nearest icon | "
            f"{_format_interval(result['model_minus_baseline_error_reduction'])} |",
            f"| icons where model beats nearest icon | {result['icons_model_beats_baseline']} |",
        ]
    clip = result.get("clip_retrieval")
    if clip:
        lines += [
            f"| CLIP top-1, model greedy ({clip['model_greedy']['icons']} Gate N icons) | "
            f"{clip['model_greedy']['top1_rate']:.3f} |",
            "| CLIP top-1, nearest training icon | "
            f"{clip['nearest_training_icon']['top1_rate']:.3f} |",
            f"| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | {39 / 64:.3f} |",
        ]
    lines += [
        "",
        "## Resources",
        "",
        "| measure | value |",
        "| --- | --- |",
        f"| device | {resource['device']} |",
        f"| parameters | {record['model_parameters']:,} |",
        f"| train seconds | {resource['train_seconds']:.0f} |",
        f"| peak VRAM GiB | {resource['peak_vram_gib']} |",
        f"| latency ms/icon, batch 1, median | {resource['inference_ms_per_icon']:.1f} |",
        f"| latency ms/icon, batch 1, p95 | {resource['inference_p95_ms_per_icon']:.1f} |",
        f"| throughput ms/icon, batch {resource['batched_icons_per_call']} | "
        f"{resource['batched_ms_per_icon']:.1f} |",
        f"| model calls per icon (decoding) | {result['model_calls_per_icon']:.1f} |",
        f"| torch / CUDA | {resource['torch_version']} / {resource['cuda_version']} |",
        "",
        "Latency covers encoding and decoding to a validated token program; it excludes",
        "rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest",
        "training icon.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--prepare-only", action="store_true", help="build the cached corpus and exit"
    )
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    if args.prepare_only:
        config = load_config(args.config)
        _, _, _, _, dataset_hash = load_corpus(config.pilot_config, config.model.image_size)
        print(json.dumps({"dataset_sha256": dataset_hash}))
        return 0
    train_and_evaluate(args.config)
    return 0


if __name__ == "__main__":
    sys.exit(main())
