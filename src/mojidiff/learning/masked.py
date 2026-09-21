"""Gate L: a masked any-order model over the same typed SVG codec sequence.

Gate I showed that a left-to-right model over this codec learns the corpus's colours and
ink and nothing about icons, and Gate G showed that a denoiser given a mostly intact
icon can repair it. The lesson is about conditioning, not architecture: on 2,681
programs a model can complete an icon it is shown, and cannot invent one it is not. This
model is built for the first task. It sees the flattened program with some positions
replaced by a MASK token and predicts what was there, and the sets of positions it is
trained to fill are the editing operations an editor exposes:

* **path** - one whole path, header and segments, with its length kept so that the
  packed segment layout does not move under the model;
* **span** - a contiguous run of segments inside one path;
* **style** - the style fields of a set of paths, geometry and topology kept;
* **geometry** - the coordinates of a set of paths, styles and segment kinds kept;
* **random** - a uniform token mask at a random rate, so that everything-masked
  generation is the same model rather than a different one.

Decoding commits tokens in grammatical dependency order - path lengths, then the rest
of each header in field order, then segment kinds, then coordinates - so that
`legal_mask` from the causal model is exact at every commit and a completed program is
valid by construction, exactly as Gate I's samples were. Within the kind and
coordinate tiers the positions are independent given what is committed, and they are
filled in a few confidence-ordered parallel passes rather than one at a time.

Everything Gates G and I verified on this codec is built in rather than discovered
again: coordinate tokens enter as Fourier features of their decoded view-unit value,
structural padding is excluded from attention, the coordinate loss spreads its target
by lattice distance so that being close earns gradient, and dropout is a first-class
option rather than an afterthought.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from mojidiff.learning.autoregressive import (
    PAD,
    PATH_FIELDS,
    PATH_STRIDE,
    SEGMENT_STRIDE,
    SequenceLayout,
    _kind_position,
    _slot_index,
    legal_mask,
)
from mojidiff.representation.packed import (
    PackedTensorProgram,
    pack_tensor_program,
    unpack_tensor_program,
)
from mojidiff.representation.program import CodecConfig, TensorProgram

MASK_FAMILIES = ("random", "path", "span", "style", "geometry")

_STYLE_OFFSETS = tuple(range(2, len(PATH_FIELDS)))
"""Header offsets of the style fields: opacity through fill_rule."""

_START_OFFSETS = (len(PATH_FIELDS), len(PATH_FIELDS) + 1)
"""Header offsets of the two start coordinates."""


def mask_token(layout: SequenceLayout) -> int:
    """The MASK id sits just past the shared vocabulary, so no legal token collides."""

    return layout.vocabulary


@dataclass(frozen=True)
class PathBlock:
    """Where one active path's tokens live in the flattened sequence."""

    path: int
    length: int
    offset: int
    header: tuple[int, ...]
    """Every header position except `path_length`, in field order."""
    segments: tuple[int, ...]
    """Every position of the path's segment blocks, kinds and coordinates alike."""

    def kinds(self, layout: SequenceLayout) -> tuple[int, ...]:
        return tuple(
            layout.segment_type_position(slot)
            for slot in range(self.offset, self.offset + self.length)
        )

    def coordinates(self, layout: SequenceLayout) -> tuple[int, ...]:
        return tuple(
            position for position in self.segments if position not in set(self.kinds(layout))
        )


def path_blocks(tokens: Tensor, layout: SequenceLayout) -> tuple[PathBlock, ...]:
    """The active paths of a flattened program, in painter order.

    Active paths form a packed prefix - the grammar forbids a gap - so the first zero
    length ends the list, and each path's segments occupy the next `length` slots.
    """

    blocks: list[PathBlock] = []
    offset = 0
    for path in range(layout.codec.max_paths):
        length = int(tokens[path * PATH_STRIDE])
        if length == 0:
            break
        base = path * PATH_STRIDE
        header = tuple(base + within for within in range(1, PATH_STRIDE))
        segments = tuple(
            layout.path_positions + slot * SEGMENT_STRIDE + within
            for slot in range(offset, offset + length)
            for within in range(SEGMENT_STRIDE)
        )
        blocks.append(PathBlock(path, length, offset, header, segments))
        offset += length
    return tuple(blocks)


def coordinate_positions(layout: SequenceLayout) -> Tensor:
    """Positions whose free tokens are lattice coordinates: start points and segment slots."""

    flags = torch.zeros(layout.length, dtype=torch.bool)
    for path in range(layout.codec.max_paths):
        for within in _START_OFFSETS:
            flags[path * PATH_STRIDE + within] = True
    flags |= _slot_index(layout) >= 0
    return flags


@dataclass(frozen=True)
class MaskMixture:
    """How training masks are drawn: which family, how often, and how much."""

    weights: dict[str, float]
    random_rate_min: float = 0.05
    random_rate_max: float = 0.95
    max_paths: int = 2
    span_max: int | None = None
    """Longest run of segments a span may hide; None lets it reach the path's end."""

    def __post_init__(self) -> None:
        unknown = set(self.weights) - set(MASK_FAMILIES)
        if unknown:
            raise ValueError(f"unknown mask families: {sorted(unknown)}")
        if not self.weights or sum(self.weights.values()) <= 0:
            raise ValueError("mask mixture needs at least one positive weight")
        if not 0.0 <= self.random_rate_min <= self.random_rate_max <= 1.0:
            raise ValueError("random mask rate bounds must satisfy 0 <= min <= max <= 1")
        if self.max_paths < 1:
            raise ValueError("max_paths must be at least 1")
        if self.span_max is not None and self.span_max < 1:
            raise ValueError("span_max must be at least 1")

    @property
    def families(self) -> tuple[str, ...]:
        return tuple(family for family in MASK_FAMILIES if self.weights.get(family, 0.0) > 0)

    @property
    def probabilities(self) -> np.ndarray:
        raw = np.array([self.weights[family] for family in self.families], dtype=np.float64)
        return raw / raw.sum()


def sample_mask(
    tokens: Tensor, layout: SequenceLayout, mixture: MaskMixture, rng: np.random.Generator
) -> tuple[str, Tensor]:
    """Draw one family from the mixture and the positions it hides.

    A hidden position is one the model must predict; the loss is later restricted to the
    hidden positions the grammar leaves free, so hiding a forced position costs nothing
    and matches what the model will be handed at edit time, where a whole block is
    masked without anyone checking which of its fields were forced.
    """

    family = str(rng.choice(mixture.families, p=mixture.probabilities))
    return family, family_mask(family, tokens, layout, mixture, rng)


def family_mask(
    family: str,
    tokens: Tensor,
    layout: SequenceLayout,
    mixture: MaskMixture,
    rng: np.random.Generator,
) -> Tensor:
    hide = torch.zeros(layout.length, dtype=torch.bool)
    blocks = path_blocks(tokens, layout)
    if family == "random":
        rate = float(rng.uniform(mixture.random_rate_min, mixture.random_rate_max))
        return torch.from_numpy(rng.random(layout.length) < rate)
    if not blocks:
        raise ValueError("a program with no active path cannot be masked by path")
    if family == "path":
        count = int(rng.integers(1, min(mixture.max_paths, len(blocks)) + 1))
        for index in rng.choice(len(blocks), size=count, replace=False):
            block = blocks[int(index)]
            hide[list(block.header)] = True
            hide[list(block.segments)] = True
        return hide
    if family == "span":
        candidates = [block for block in blocks if block.length >= 2] or list(blocks)
        block = candidates[int(rng.integers(len(candidates)))]
        first = int(rng.integers(0, block.length))
        longest = block.length - first
        if mixture.span_max is not None:
            longest = min(longest, mixture.span_max)
        run = int(rng.integers(1, longest + 1))
        for slot in range(block.offset + first, block.offset + first + run):
            base = layout.path_positions + slot * SEGMENT_STRIDE
            hide[base : base + SEGMENT_STRIDE] = True
        return hide
    chosen = [block for block in blocks if rng.random() < 0.5]
    if not chosen:
        chosen = [blocks[int(rng.integers(len(blocks)))]]
    if family == "style":
        for block in chosen:
            hide[[block.path * PATH_STRIDE + within for within in _STYLE_OFFSETS]] = True
        return hide
    if family == "geometry":
        for block in chosen:
            hide[[block.path * PATH_STRIDE + within for within in _START_OFFSETS]] = True
            hide[list(block.coordinates(layout))] = True
        return hide
    raise ValueError(f"unknown mask family: {family}")


def whole_path_mask(tokens: Tensor, layout: SequenceLayout, path: int) -> Tensor:
    """The editing operation this gate is scored on: redraw one path, length kept."""

    for block in path_blocks(tokens, layout):
        if block.path == path:
            hide = torch.zeros(layout.length, dtype=torch.bool)
            hide[list(block.header)] = True
            hide[list(block.segments)] = True
            return hide
    raise ValueError(f"path {path} is not active")


def apply_mask(tokens: Tensor, hide: Tensor, layout: SequenceLayout) -> Tensor:
    return tokens.masked_fill(hide, mask_token(layout))


# ------------------------------------------------------------------------------ model


class MaskedProgramModel(nn.Module):
    """A bidirectional transformer that predicts masked tokens of the flattened program."""

    def __init__(
        self,
        layout: SequenceLayout,
        *,
        d_model: int,
        heads: int,
        layers: int,
        feedforward: int,
        group_vocab_size: int = 0,
        subgroup_vocab_size: int = 0,
        metric_coordinates: int = 0,
        dropout: float = 0.0,
        path_binding: bool = False,
        metric_head: bool = False,
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        self.layout = layout
        self.mask_id = mask_token(layout)
        self.metric_coordinates = metric_coordinates
        # The packed layout never tells the encoder which path a segment slot belongs to
        # or where in that path it sits: a hole at slot 57 could be any path's, and the
        # model has to count path_length tokens to find out. The corpus run learned
        # style co-occurrence and nothing about coordinates, on its own training icons.
        # With binding, every position carries its owning path's index and every segment
        # its index within that path, both derived from the visible lengths, so a path's
        # header and its geometry share one identity the way slot binding gave the
        # denoiser's coordinates theirs.
        self.path_binding = path_binding
        self.path_identity = (
            nn.Embedding(layout.codec.max_paths + 1, d_model) if path_binding else None
        )
        self.segment_identity = (
            nn.Embedding(layout.codec.max_segments + 1, d_model) if path_binding else None
        )
        self.token_embedding = nn.Embedding(layout.vocabulary + 1, d_model)
        self.position_embedding = nn.Embedding(layout.length, d_model)
        # Gate G verified on this codec that a categorical table carries no order over
        # a quarter-unit lattice, and that replacing it with Fourier features of the
        # decoded value took render recovery from 0.0447 to 0.2511 while shrinking the
        # model. Here it applies to every known coordinate token, start points included.
        self.coordinate_projection = (
            nn.Linear(2 * metric_coordinates + 2, d_model) if metric_coordinates > 0 else None
        )
        self.register_buffer("_slot_index", _slot_index(layout), persistent=False)
        self.register_buffer("_kind_position", _kind_position(layout), persistent=False)
        self.register_buffer("_is_start", _start_positions(layout), persistent=False)
        self.register_buffer(
            "_control_counts", torch.tensor([0, 0, 2, 4, 0], dtype=torch.long), persistent=False
        )
        self.group_embedding = (
            nn.Embedding(group_vocab_size, d_model) if group_vocab_size > 0 else None
        )
        self.subgroup_embedding = (
            nn.Embedding(subgroup_vocab_size, d_model) if subgroup_vocab_size > 0 else None
        )
        self.input_dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, layout.vocabulary)
        # The categorical head is a linear map onto 289 unordered bins, so to place a
        # bump wherever a neighbour's value says it must first discover an ordering over
        # them. The metric head skips that: at a coordinate position whose role is known
        # the logit for a bin is the inner product of a projection of the state with the
        # same Fourier features of that bin's decoded value that the input side uses,
        # plus the categorical bias, so "near v" is one vector away rather than 289.
        self.metric_head = metric_head
        if metric_head and metric_coordinates <= 0:
            raise ValueError("the metric head needs metric coordinates")
        self.head_projection = (
            nn.Linear(d_model, 2 * metric_coordinates + 2) if metric_head else None
        )
        if metric_head:
            codec = layout.codec
            self.register_buffer(
                "_endpoint_basis",
                _bin_features(codec.coordinate_bins, 0.0, 72.0, metric_coordinates),
                persistent=False,
            )
            self.register_buffer(
                "_control_basis",
                _bin_features(
                    codec.effective_control_coordinate_bins,
                    codec.control_coordinate_min,
                    codec.control_coordinate_max,
                    metric_coordinates,
                ),
                persistent=False,
            )

    def forward(self, inputs: Tensor, condition: dict[str, Tensor] | None = None) -> Tensor:
        """Logits over the shared vocabulary at every position.

        `inputs` carries the MASK id at the positions to predict. Positions holding
        typed padding are structurally inert - an inactive path's fields, an unused
        segment slot - and are excluded from attention as keys; Gate G measured 29.4%
        of its sequence being attended as content before that was fixed. Masked
        positions are attended: a hole is information.
        """

        if inputs.dim() != 2 or inputs.shape[1] != self.layout.length:
            raise ValueError("inputs must be a batch of complete flattened programs")
        positions = torch.arange(self.layout.length, device=inputs.device)
        hidden = self.token_embedding(inputs) + self.position_embedding(positions)[None]
        if self.coordinate_projection is not None:
            hidden = self._apply_metric_coordinates(hidden, inputs)
        if self.path_identity is not None and self.segment_identity is not None:
            owner, within = segment_ownership(inputs, self.layout)
            hidden = hidden + self.path_identity(owner) + self.segment_identity(within)
        if self.group_embedding is not None or self.subgroup_embedding is not None:
            if condition is None:
                raise ValueError("this model was built with structured conditioning")
            extra = torch.zeros_like(hidden[:, 0])
            if self.group_embedding is not None:
                extra = extra + self.group_embedding(condition["group"])
            if self.subgroup_embedding is not None:
                extra = extra + self.subgroup_embedding(condition["subgroup"])
            hidden = hidden + extra[:, None, :]
        if self.input_dropout is not None:
            hidden = self.input_dropout(hidden)
        encoded: Tensor = self.encoder(hidden, src_key_padding_mask=inputs == PAD)
        normed = self.norm(encoded)
        logits: Tensor = self.head(normed)
        if self.head_projection is not None:
            logits = self._apply_metric_head(logits, normed, inputs)
        return logits

    def _coordinate_roles(self, inputs: Tensor) -> tuple[Tensor, Tensor]:
        """Which positions are endpoint coordinates and which control handles, by role.

        Decided from the segment kind governing each position, which is visible at
        training time and committed before the coordinate tier at decode time; a
        position whose kind is masked has no role and keeps the categorical head.
        """

        batch = inputs.shape[0]
        slot = cast(Tensor, self._slot_index)[None].expand(batch, -1)
        kind_at = cast(Tensor, self._kind_position)
        kinds = inputs.gather(1, kind_at.clamp_min(0)[None].expand(batch, -1))
        known = (kind_at >= 0)[None] & (kinds >= 1) & (kinds <= 3)
        counts = cast(Tensor, self._control_counts)[kinds.clamp(0, 4)]
        coordinate_count = counts + 2
        is_segment = (slot >= 0) & known & (slot < coordinate_count)
        is_control = is_segment & (slot < counts)
        is_endpoint = (is_segment & ~is_control) | cast(Tensor, self._is_start)[None]
        return is_endpoint, is_control

    def _apply_metric_head(self, logits: Tensor, normed: Tensor, inputs: Tensor) -> Tensor:
        projection = cast(nn.Linear, self.head_projection)
        query: Tensor = projection(normed)
        is_endpoint, is_control = self._coordinate_roles(inputs)
        bias = self.head.bias
        endpoint = query @ cast(Tensor, self._endpoint_basis).t()
        control = query @ cast(Tensor, self._control_basis).t()
        vocabulary = logits.shape[-1]
        metric = torch.full_like(logits, float("-inf"))
        # Token 0 is padding at every coordinate position and is never legal there; the
        # legal mask removes it, so its logit here does not matter.
        endpoint_full = (
            torch.cat((metric[..., :1], endpoint, metric[..., 1 + endpoint.shape[-1] :]), dim=-1)
            + bias
        )
        control_full = (
            torch.cat((metric[..., :1], control, metric[..., 1 + control.shape[-1] :]), dim=-1)
            + bias
        )
        assert endpoint_full.shape[-1] == vocabulary and control_full.shape[-1] == vocabulary
        logits = torch.where(is_endpoint[..., None], endpoint_full, logits)
        return torch.where(is_control[..., None], control_full, logits)

    def _apply_metric_coordinates(self, hidden: Tensor, inputs: Tensor) -> Tensor:
        codec = self.layout.codec
        batch = inputs.shape[0]
        slot = cast(Tensor, self._slot_index)[None].expand(batch, -1)
        kind_at = cast(Tensor, self._kind_position)
        kinds = inputs.gather(1, kind_at.clamp_min(0)[None].expand(batch, -1))
        # A coordinate whose governing kind is itself masked has no known role, so it
        # stays categorical; the model sees the token and the hole beside it.
        known_kind = (kind_at >= 0)[None] & (kinds <= 4)
        visible = (inputs > 0) & (inputs < self.mask_id)
        is_segment = (slot >= 0) & known_kind & visible
        is_start = cast(Tensor, self._is_start)[None] & visible
        counts = cast(Tensor, self._control_counts)[kinds.clamp(0, 4)]
        is_control = is_segment & (slot < counts)
        is_coordinate = is_segment | is_start

        endpoint_step = 72.0 / (codec.coordinate_bins - 1)
        control_span = codec.control_coordinate_max - codec.control_coordinate_min
        control_step = control_span / (codec.effective_control_coordinate_bins - 1)
        index = (inputs.to(torch.float32) - 1.0).clamp_min(0.0)
        value = torch.where(
            is_control,
            codec.control_coordinate_min + index * control_step,
            index * endpoint_step,
        ) * is_coordinate.to(torch.float32)
        frequencies = (
            torch.pow(
                2.0,
                torch.arange(self.metric_coordinates, dtype=torch.float32, device=inputs.device),
            )
            * torch.pi
            / 72.0
        )
        scaled = value[..., None] * frequencies
        features = torch.cat(
            (
                (value / 72.0)[..., None],
                is_coordinate.to(torch.float32)[..., None],
                torch.sin(scaled),
                torch.cos(scaled),
            ),
            dim=-1,
        )
        projection = cast(nn.Linear, self.coordinate_projection)
        replacement: Tensor = projection(features)
        categorical: Tensor = self.token_embedding(inputs)
        keep = is_coordinate[..., None].to(hidden.dtype)
        return hidden + keep * (replacement - categorical)


def segment_ownership(inputs: Tensor, layout: SequenceLayout) -> tuple[Tensor, Tensor]:
    """Which path each position belongs to, and each segment's index within its path.

    Header positions belong to their own path and carry no within-path index (the
    sentinel `max_segments`). Segment slots are assigned by walking the visible path
    lengths in painter order; a slot past every declared length, or one whose owner
    cannot be determined because a length before it is masked, gets the sentinel
    `max_paths` for its owner and `max_segments` for its index. Everything is derived
    from `inputs`, so it is available at decode time exactly as in training.
    """

    batch = inputs.shape[0]
    device = inputs.device
    max_paths = layout.codec.max_paths
    max_segments = layout.codec.max_segments
    mask_id = mask_token(layout)
    owner = torch.full((batch, layout.length), max_paths, dtype=torch.long, device=device)
    within = torch.full((batch, layout.length), max_segments, dtype=torch.long, device=device)
    for path in range(max_paths):
        owner[:, path * PATH_STRIDE : (path + 1) * PATH_STRIDE] = path
    lengths = inputs[:, [path * PATH_STRIDE for path in range(max_paths)]]
    known = lengths != mask_id
    # Once a length is masked, every later offset is unknown.
    determinable = torch.cumprod(known.to(torch.long), dim=1).bool()
    counted = torch.where(known, lengths, torch.zeros_like(lengths))
    starts = torch.cumsum(counted, dim=1) - counted
    slots = torch.arange(layout.total_segment_slots, device=device)
    slot_owner = torch.full(
        (batch, layout.total_segment_slots), max_paths, dtype=torch.long, device=device
    )
    slot_index = torch.full(
        (batch, layout.total_segment_slots), max_segments, dtype=torch.long, device=device
    )
    for path in range(max_paths):
        begin = starts[:, path : path + 1]
        end = begin + counted[:, path : path + 1]
        mine = (slots[None] >= begin) & (slots[None] < end) & determinable[:, path : path + 1]
        slot_owner = torch.where(mine, torch.full_like(slot_owner, path), slot_owner)
        slot_index = torch.where(mine, slots[None] - begin, slot_index)
    segment_owner = slot_owner.repeat_interleave(SEGMENT_STRIDE, dim=1)
    segment_index = slot_index.repeat_interleave(SEGMENT_STRIDE, dim=1)
    owner[:, layout.path_positions :] = segment_owner
    within[:, layout.path_positions :] = segment_index
    return owner, within


def _bin_features(bins: int, low: float, high: float, frequencies: int) -> Tensor:
    """The input side's Fourier features, evaluated at every bin of one role."""

    value = low + torch.arange(bins, dtype=torch.float32) * ((high - low) / (bins - 1))
    scale = torch.pow(2.0, torch.arange(frequencies, dtype=torch.float32)) * torch.pi / 72.0
    scaled = value[:, None] * scale[None]
    return torch.cat(
        ((value / 72.0)[:, None], torch.ones(bins, 1), torch.sin(scaled), torch.cos(scaled)),
        dim=-1,
    )


def _start_positions(layout: SequenceLayout) -> Tensor:
    flags = torch.zeros(layout.length, dtype=torch.bool)
    for path in range(layout.codec.max_paths):
        for within in _START_OFFSETS:
            flags[path * PATH_STRIDE + within] = True
    return flags


# ------------------------------------------------------------------------------- loss


def masked_loss(
    logits: Tensor,
    tokens: Tensor,
    legal: Tensor,
    targets: Tensor,
    coordinates: Tensor,
    tau: float | tuple[float, ...],
    step: float = 0.25,
) -> Tensor:
    """Cross-entropy at the target positions, pooled over fields, soft on coordinates.

    The target at a coordinate position spreads as `exp(-|b - t| * step / tau)` over the
    legal bins, `tau` in view units, so a prediction one bin out earns most of the
    credit rather than none - v15's correction, which cut view-unit error 27%. Every
    other position keeps an exact one-hot target. The mean is over fields, never over
    per-field groups; Gate G's loss weighted four rare fields as a third of the total
    that way.

    `tau` may be several widths, in which case the target is their equal mixture: a
    sharp peak that rewards precision and a wide basin that rewards being in the right
    region. With one narrow kernel a prediction four view units away earns nothing, so
    coarse localisation gets no gradient at all - which is how the first corpus arms
    learned where things are on average and never that a segment starts where the last
    one ended.
    """

    picked = logits[targets].float()
    if picked.shape[0] == 0:
        return picked.sum()
    truth = tokens[targets]
    allowed = legal[targets]
    soft = coordinates[None].expand_as(targets)[targets]
    bins = torch.arange(logits.shape[-1], device=logits.device)
    exact = (bins[None] == truth[:, None]).to(torch.float32)
    widths = tuple(width for width in (tau if isinstance(tau, tuple) else (tau,)) if width > 0.0)
    if widths:
        # Logits here index tokens, not lattice bins, so the kernel is centred on the
        # token itself. The denoiser's distance kernel subtracts one because its heads
        # index bins; copying that here centred every coordinate target one token low,
        # and the overfit test caught it as every coordinate exactly one bin off.
        distance = (bins[None] - truth[:, None]).abs().to(torch.float32)
        spread = torch.zeros_like(distance)
        for width in widths:
            kernel = torch.exp(-distance * step / width) * allowed
            spread = spread + kernel / kernel.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = torch.where(soft[:, None], spread, exact)
    else:
        weights = exact
    weights = weights * allowed
    weights = weights / weights.sum(dim=-1, keepdim=True)
    log_probs = torch.log_softmax(picked.masked_fill(~allowed, float("-inf")), dim=-1)
    log_probs = log_probs.masked_fill(~allowed, 0.0)
    return -(weights * log_probs).sum(dim=-1).mean()


def masked_nll(
    logits: Tensor, tokens: Tensor, legal: Tensor, targets: Tensor
) -> tuple[float, int, int]:
    """Exact-token negative log likelihood and hits at the target positions, summed."""

    picked = logits[targets].float().masked_fill(~legal[targets], float("-inf"))
    truth = tokens[targets]
    if truth.shape[0] == 0:
        return 0.0, 0, 0
    nll = torch.nn.functional.cross_entropy(picked, truth, reduction="sum")
    hits = int((picked.argmax(dim=-1) == truth).sum())
    return float(nll), hits, int(truth.shape[0])


# ---------------------------------------------------------------------------- decoding


LogitsFn = Callable[[Tensor], Tensor]
"""Maps one flattened program with MASK ids, shape `[length]`, to logits `[length, V]`."""


@torch.no_grad()
def complete(
    logits: LogitsFn,
    inputs: Tensor,
    layout: SequenceLayout,
    *,
    greedy: bool = True,
    rng: np.random.Generator | None = None,
    iterations: int = 8,
    temperature: float = 1.0,
) -> tuple[Tensor, int]:
    """Fill every MASK in `inputs`, committing in grammatical dependency order.

    Tier one walks the path headers in position order, one forward per path with a
    masked header, so that every `legal_mask` it consults reads only committed tokens:
    a path's length depends on the lengths before it, its layer on the layers before
    it, its style fields on its own fill and stroke and on the previous path's style
    when the layer repeats. Tier two fills the segment kinds, whose legality depends
    only on the lengths, and tier three the coordinates, whose legality depends only
    on the kinds; both are filled in `iterations` confidence-ordered parallel passes.

    The masked set must be dependency-closed - when a field is masked, every field
    whose legality depends on it is masked too - which the editing families guarantee.
    A random token mask need not be, and the caller validates the result regardless.

    Returns the completed sequence and the number of forward calls.
    """

    mask_id = mask_token(layout)
    decoded = inputs.clone()
    calls = 0
    generator = rng if rng is not None else np.random.default_rng(0)

    for path in range(layout.codec.max_paths):
        base = path * PATH_STRIDE
        block = [
            position
            for position in range(base, base + PATH_STRIDE)
            if int(decoded[position]) == mask_id
        ]
        if not block:
            continue
        scores = logits(decoded)
        calls += 1
        for position in block:
            mask = legal_mask(position, decoded, layout) & _later_path_bound(
                position, decoded, layout, mask_id
            )
            decoded[position], _ = _choose(scores[position], mask, greedy, generator, temperature)

    kinds = [
        layout.segment_type_position(slot)
        for slot in range(layout.total_segment_slots)
        if int(decoded[layout.segment_type_position(slot)]) == mask_id
    ]
    calls += _fill_tier(logits, decoded, kinds, layout, iterations, greedy, generator, temperature)
    coordinates = [
        position
        for position in range(layout.path_positions, layout.length)
        if int(decoded[position]) == mask_id
    ]
    calls += _fill_tier(
        logits, decoded, coordinates, layout, iterations, greedy, generator, temperature
    )
    if bool((decoded == mask_id).any()):
        raise RuntimeError("completion left a MASK in place")
    return decoded, calls


def _later_path_bound(
    position: int, decoded: Tensor, layout: SequenceLayout, mask_id: int
) -> Tensor:
    """Constraints that paths further on impose, which `legal_mask` cannot see.

    `legal_mask` was written for left-to-right decoding and reads only earlier tokens.
    When a masked path sits among known ones, three of the grammar's rules also bind
    from the other side: a length is bounded by the capacity the known later paths
    already take, and cannot be zero when a later path is known active, because active
    paths form a packed prefix; a layer cannot exceed the next known layer, because
    layers are nondecreasing; and when the chosen layer equals that next known layer,
    every style field is pinned to the later path's, because contours in one layer
    share a style.
    """

    bound = torch.ones(layout.vocabulary, dtype=torch.bool)
    path, within = divmod(position, PATH_STRIDE)
    if within == 0:
        later = [
            int(decoded[other * PATH_STRIDE])
            for other in range(path + 1, layout.codec.max_paths)
            if int(decoded[other * PATH_STRIDE]) != mask_id
        ]
        earlier = sum(int(decoded[other * PATH_STRIDE]) for other in range(path))
        spare = layout.total_segment_slots - earlier - sum(later)
        bound[max(spare, 0) + 1 :] = False
        if any(value > 0 for value in later):
            bound[0] = False
        return bound
    if within >= len(PATH_FIELDS) or int(decoded[path * PATH_STRIDE]) == 0:
        return bound
    following = _next_known_layer(path, decoded, layout, mask_id)
    if following is None:
        return bound
    other, layer = following
    if within == 1:
        bound[layer + 1 :] = False
        return bound
    if int(decoded[path * PATH_STRIDE + 1]) == layer:
        pinned = int(decoded[other * PATH_STRIDE + within])
        if pinned != mask_id:
            bound[:] = False
            bound[pinned] = True
    return bound


def _next_known_layer(
    path: int, decoded: Tensor, layout: SequenceLayout, mask_id: int
) -> tuple[int, int] | None:
    """The first later path whose layer is already known, with that layer."""

    for other in range(path + 1, layout.codec.max_paths):
        length = int(decoded[other * PATH_STRIDE])
        if length == mask_id or length == 0:
            return None
        layer = int(decoded[other * PATH_STRIDE + 1])
        if layer != mask_id:
            return other, layer
    return None


def _fill_tier(
    logits: LogitsFn,
    decoded: Tensor,
    positions: list[int],
    layout: SequenceLayout,
    iterations: int,
    greedy: bool,
    rng: np.random.Generator,
    temperature: float,
) -> int:
    if not positions:
        return 0
    # Every dependency of this tier is committed, so the legal sets are fixed and are
    # computed once rather than once per pass.
    masks = {position: legal_mask(position, decoded, layout) for position in positions}
    remaining = list(positions)
    calls = 0
    for pass_index in range(iterations):
        if not remaining:
            break
        scores = logits(decoded)
        calls += 1
        choices: list[tuple[float, int, int]] = []
        for position in remaining:
            token, confidence = _choose(scores[position], masks[position], greedy, rng, temperature)
            choices.append((-confidence, position, token))
        choices.sort()
        left = iterations - pass_index
        commit = len(remaining) if left == 1 else math.ceil(len(remaining) / left)
        for _, position, token in choices[:commit]:
            decoded[position] = token
        committed = {position for _, position, _ in choices[:commit]}
        remaining = [position for position in remaining if position not in committed]
    return calls


def _choose(
    scores: Tensor, mask: Tensor, greedy: bool, rng: np.random.Generator, temperature: float
) -> tuple[int, float]:
    if not bool(mask.any()):
        raise ValueError("no legal token at a position being completed")
    masked = scores.detach().float().cpu().masked_fill(~mask, float("-inf"))
    probabilities = (
        torch.softmax(masked / max(temperature, 1e-6), dim=-1).numpy().astype(np.float64)
    )
    probabilities = probabilities / probabilities.sum()
    if greedy:
        token = int(np.argmax(probabilities))
    else:
        token = int(rng.choice(len(probabilities), p=probabilities))
    return token, float(probabilities[token])


def marginal_logits(marginals: Tensor) -> LogitsFn:
    """The zero-parameter policy of Gate I as a completion strategy.

    Knows the grammar and the empirical token frequency at every position, nothing about
    the icon it is completing. Decoded through the same tiers as the model, it is what
    "corpus statistics alone" produces for an edit, and the model has to beat it.
    """

    scores = torch.log(marginals.to(torch.float32))

    def logits(_: Tensor) -> Tensor:
        return scores

    return logits


def model_logits(model: MaskedProgramModel, condition: dict[str, Tensor] | None) -> LogitsFn:
    device = next(model.parameters()).device

    def logits(inputs: Tensor) -> Tensor:
        output: Tensor = model(inputs[None].to(device), condition)
        return output[0].cpu()

    return logits


# ---------------------------------------------------------------------------- baseline


def drop_path(
    program: PackedTensorProgram, path: int, codec: CodecConfig, total_segment_slots: int
) -> PackedTensorProgram:
    """The program without one of its paths: what an editor shows before the fill.

    This is the identity policy of an inpainting task - the state a user is looking at
    when they ask for the path to be drawn - and every trained completion is scored
    against it.
    """

    dense = unpack_tensor_program(program, codec, total_segment_slots)
    if int(dense.path_length[path]) == 0:
        raise ValueError(f"path {path} is not active")
    fields: dict[str, Any] = {}
    for name in TensorProgram.__dataclass_fields__:
        array = np.asarray(getattr(dense, name))
        kept = np.delete(array, path, axis=0)
        padding = np.zeros((1, *array.shape[1:]), dtype=array.dtype)
        fields[name] = np.concatenate((kept, padding), axis=0)
    return pack_tensor_program(TensorProgram(**fields), codec, total_segment_slots)
