"""Gate I: a cached autoregressive baseline over the same typed SVG codec.

Gate G settled every training-side factor it identified without producing a generator,
because denoising is not generation. This is PROJECT_PLAN.md section 12 branch 4 - the
same codec, the same splits, the same renderer and evaluation, decoded left to right
instead of denoised in place.

The plan's terms are built in rather than bolted on:

* **Legal-token masks.** Every position has a legal vocabulary, and a coordinate's
  vocabulary depends on the segment kind decoded earlier in the same sequence, so the
  mask is dynamic. Sampling can only ever produce a program the packed validator accepts.
* **KV caching.** Attention keys and values are cached across decode steps, so producing
  token `t` costs one position rather than a re-read of the prefix. The plan is explicit
  that nominal step counts are not a speed result, which makes real incremental decoding
  a requirement rather than an optimisation.

The sequence is the packed program in painter order: each path's metadata and start
point, then each segment's kind and coordinates. It is a fixed-length grid including
typed padding, so a program and its token sequence are exact inverses.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from mojidiff.learning.geometry import _coordinate_counts, _path_field_sizes
from mojidiff.representation.packed import PackedTensorProgram
from mojidiff.representation.program import CodecConfig, SegmentType

PATH_FIELDS = (
    "path_length",
    "layer",
    "opacity",
    "fill",
    "fill_opacity",
    "stroke",
    "stroke_opacity",
    "stroke_width",
    "linecap",
    "linejoin",
    "miter_limit",
    "dash_pattern",
    "fill_rule",
)

PATH_STRIDE = len(PATH_FIELDS) + 2
"""Metadata fields plus the two start coordinates."""

SEGMENT_STRIDE = 7
"""The segment kind plus six coordinate slots."""


@dataclass(frozen=True)
class SequenceLayout:
    """Where each token lives and what it is allowed to be."""

    codec: CodecConfig
    total_segment_slots: int

    @property
    def path_positions(self) -> int:
        return self.codec.max_paths * PATH_STRIDE

    @property
    def length(self) -> int:
        return self.path_positions + self.total_segment_slots * SEGMENT_STRIDE

    @property
    def vocabulary(self) -> int:
        """One shared vocabulary; positions are narrowed by their legal mask."""

        sizes = _path_field_sizes(self.codec)
        return max(
            *sizes.values(),
            5,
            self.codec.coordinate_bins + 1,
            self.codec.effective_control_coordinate_bins + 1,
        )

    def static_limits(self) -> Tensor:
        """Per-position legal vocabulary size, ignoring coordinate role.

        Coordinate positions get the widest legal size here; `legal_mask` narrows them
        once the segment kind is known.
        """

        sizes = _path_field_sizes(self.codec)
        limits = torch.zeros(self.length, dtype=torch.long)
        for path in range(self.codec.max_paths):
            base = path * PATH_STRIDE
            for offset, name in enumerate(PATH_FIELDS):
                limits[base + offset] = sizes[name]
            limits[base + len(PATH_FIELDS)] = self.codec.coordinate_bins + 1
            limits[base + len(PATH_FIELDS) + 1] = self.codec.coordinate_bins + 1
        for segment in range(self.total_segment_slots):
            base = self.path_positions + segment * SEGMENT_STRIDE
            limits[base] = 5
            for slot in range(6):
                limits[base + 1 + slot] = self.codec.effective_control_coordinate_bins + 1
        return limits

    def coordinate_slot_of(self, position: int) -> tuple[int, int] | None:
        """`(segment index, slot)` when this position is a segment coordinate."""

        if position < self.path_positions:
            return None
        offset = position - self.path_positions
        segment, within = divmod(offset, SEGMENT_STRIDE)
        if within == 0:
            return None
        return segment, within - 1

    def segment_type_position(self, segment: int) -> int:
        return self.path_positions + segment * SEGMENT_STRIDE

    def path_length_index(self, position: int) -> int | None:
        """The path whose length this position carries, if it carries one."""

        if position >= self.path_positions:
            return None
        path, within = divmod(position, PATH_STRIDE)
        return path if within == 0 else None

    def remaining_capacity(self, position: int, decoded: Tensor) -> int | None:
        """Segment slots still free once the earlier paths have taken theirs."""

        path = self.path_length_index(position)
        if path is None:
            return None
        used = sum(int(decoded[earlier * PATH_STRIDE]) for earlier in range(path))
        return max(self.total_segment_slots - used, 0)


def _slot_index(layout: SequenceLayout) -> Tensor:
    """The coordinate slot 0-5 each absolute position carries, or -1 if it carries none."""

    index = torch.full((layout.length,), -1, dtype=torch.long)
    for position in range(layout.length):
        found = layout.coordinate_slot_of(position)
        if found is not None:
            index[position] = found[1]
    return index


def _kind_position(layout: SequenceLayout) -> Tensor:
    """Where the segment-kind token governing each coordinate position lives.

    A slot decodes over a different range depending on the kind - slot 0 is an endpoint
    for a LINE and a control handle for a CUBIC - so the value of a coordinate token is
    not knowable from the token alone. The kind sits at the head of its segment's block,
    already decoded by the time any of its coordinates are read.
    """

    index = torch.full((layout.length,), -1, dtype=torch.long)
    for position in range(layout.length):
        found = layout.coordinate_slot_of(position)
        if found is not None:
            index[position] = layout.segment_type_position(found[0])
    return index


def coordinate_kind_tokens(tokens: Tensor, layout: SequenceLayout) -> Tensor:
    """For each absolute position, the segment-kind token that governs it, else 0."""

    where = _kind_position(layout).to(tokens.device)
    gathered = tokens.gather(1, where.clamp_min(0)[None].expand(tokens.shape[0], -1))
    return gathered * (where >= 0)[None]


def teacher_forcing_inputs(tokens: Tensor, layout: SequenceLayout) -> tuple[Tensor, Tensor]:
    """The shifted input sequence and the segment kinds aligned to it.

    Both shift by one, and getting only one of them right is an off-by-one that changes
    which affine map decodes a coordinate without changing any shape - so it trains, and
    quietly. Doing the shift once, here, is the point of this function.
    """

    zero = torch.zeros_like(tokens[:, :1])
    kinds = coordinate_kind_tokens(tokens, layout)
    return (
        torch.cat((zero, tokens[:, :-1]), dim=1),
        torch.cat((zero, kinds[:, :-1]), dim=1),
    )


def flatten_program(program: PackedTensorProgram, layout: SequenceLayout) -> Tensor:
    """Pack a program into its token sequence, padding included."""

    tokens = torch.zeros(layout.length, dtype=torch.long)
    for path in range(layout.codec.max_paths):
        base = path * PATH_STRIDE
        for offset, name in enumerate(PATH_FIELDS):
            tokens[base + offset] = int(getattr(program, name)[path])
        tokens[base + len(PATH_FIELDS)] = int(program.start[path, 0])
        tokens[base + len(PATH_FIELDS) + 1] = int(program.start[path, 1])
    for segment in range(layout.total_segment_slots):
        base = layout.segment_type_position(segment)
        tokens[base] = int(program.segment_type[segment])
        for slot in range(6):
            tokens[base + 1 + slot] = int(program.coordinates[segment, slot])
    return tokens


def unflatten_program(
    tokens: Tensor, template: PackedTensorProgram, layout: SequenceLayout
) -> PackedTensorProgram:
    """Rebuild a program from its token sequence, using `template` only for dtypes."""

    import copy

    result = copy.deepcopy(template)
    for path in range(layout.codec.max_paths):
        base = path * PATH_STRIDE
        for offset, name in enumerate(PATH_FIELDS):
            getattr(result, name)[path] = int(tokens[base + offset])
        result.start[path, 0] = int(tokens[base + len(PATH_FIELDS)])
        result.start[path, 1] = int(tokens[base + len(PATH_FIELDS) + 1])
    for segment in range(layout.total_segment_slots):
        base = layout.segment_type_position(segment)
        result.segment_type[segment] = int(tokens[base])
        for slot in range(6):
            result.coordinates[segment, slot] = int(tokens[base + 1 + slot])
    return result


PAD, NONE = 0, 1
"""Typed padding and the explicit absence token, as the codec defines them."""


def _field_offset(name: str) -> int:
    return PATH_FIELDS.index(name)


def legal_mask(position: int, decoded: Tensor, layout: SequenceLayout) -> Tensor:
    """Boolean mask over the shared vocabulary for one position.

    This encodes the codec's typed grammar, not just each field's vocabulary size, so a
    sampler cannot produce a program the packed validator rejects: inactive paths must
    be entirely PAD and must form a suffix, a path must paint a fill or a stroke, the
    fields that depend on a NONE fill or stroke must themselves be NONE, segments must
    fill exactly their path's declared length, and CLOSE may only end one.

    Layers may repeat, as the grammar allows, and when a path reuses the previous
    layer every style field is then pinned to that path's value - which is exactly the
    "contours in one layer share a style" rule, expressed as a mask. Forcing layers
    strictly increasing instead would be simpler but can exhaust the layer vocabulary
    and leave a position with no legal token at all.
    """

    codec = layout.codec
    mask = torch.zeros(layout.vocabulary, dtype=torch.bool)

    slot = layout.coordinate_slot_of(position)
    if slot is not None:
        segment, index = slot
        return _segment_coordinate_mask(mask, segment, index, decoded, layout)
    if position >= layout.path_positions:
        return _segment_kind_mask(mask, decoded, layout, position)

    path, within = divmod(position, PATH_STRIDE)
    if within == _field_offset("path_length"):
        earlier = [int(decoded[p * PATH_STRIDE]) for p in range(path)]
        capacity = max(layout.total_segment_slots - sum(earlier), 0)
        mask[0] = True  # a path may always be absent
        if all(value > 0 for value in earlier):  # active paths form a packed prefix
            mask[1 : min(codec.max_segments, capacity) + 1] = True
        return mask

    active = int(decoded[path * PATH_STRIDE]) > 0
    if not active:
        mask[PAD] = True  # every field of an inactive path is PAD
        return mask

    base = path * PATH_STRIDE
    fill = int(decoded[base + _field_offset("fill")])
    stroke = int(decoded[base + _field_offset("stroke")])

    # Contours sharing a layer must share a style, so a repeated layer pins every style
    # field to the previous path's value. Only path_length, layer and the start point
    # stay free.
    if path > 0 and within > _field_offset("layer"):
        layer_here = int(decoded[base + _field_offset("layer")])
        layer_before = int(decoded[(path - 1) * PATH_STRIDE + _field_offset("layer")])
        if layer_here and layer_here == layer_before and within < len(PATH_FIELDS):
            mask[int(decoded[(path - 1) * PATH_STRIDE + within])] = True
            return mask

    if within == _field_offset("layer"):
        previous = max(
            (int(decoded[p * PATH_STRIDE + _field_offset("layer")]) for p in range(path)),
            default=0,
        )
        if previous >= 1:
            mask[previous] = True  # reuse the layer, which pins this path's style below
        mask[previous + 1 : codec.max_paths + 1] = True
        return mask


    if within == _field_offset("opacity"):
        mask[2 : len(codec.opacities) + 2] = True
        return mask
    if within == _field_offset("fill"):
        mask[NONE] = True
        mask[2 : len(codec.palette) + 2] = True
        return mask
    if within == _field_offset("stroke"):
        if fill != NONE:
            mask[NONE] = True  # a fill-painting path may omit its stroke
        mask[2 : len(codec.palette) + 2] = True
        return mask
    if within in (_field_offset("fill_opacity"), _field_offset("fill_rule")):
        if fill == NONE:
            mask[NONE] = True
        elif within == _field_offset("fill_opacity"):
            mask[2 : len(codec.opacities) + 2] = True
        else:
            mask[2 : len(_FILL_RULE_NAMES) + 2] = True
        return mask

    stroke_fields = {
        "stroke_opacity": len(codec.opacities),
        "stroke_width": len(codec.stroke_widths),
        "linecap": len(_CAP_NAMES),
        "linejoin": len(_JOIN_NAMES),
        "miter_limit": len(codec.miter_limits),
        "dash_pattern": len(codec.dash_patterns),
    }
    for name, size in stroke_fields.items():
        if within != _field_offset(name):
            continue
        if stroke == NONE:
            mask[NONE] = True
        else:
            if name == "dash_pattern":
                mask[NONE] = True  # a stroked path may simply not dash
            mask[2 : size + 2] = True
        return mask

    # The two start coordinates, which are ordinary endpoint tokens.
    mask[1 : codec.coordinate_bins + 1] = True
    return mask


_FILL_RULE_NAMES = ("nonzero", "evenodd")
_CAP_NAMES = ("butt", "round", "square")
_JOIN_NAMES = ("miter", "round", "bevel")


def _segment_owner(segment: int, decoded: Tensor, layout: SequenceLayout) -> tuple[int, int] | None:
    """The path a segment slot belongs to, and its index within that path."""

    start = 0
    for path in range(layout.codec.max_paths):
        length = int(decoded[path * PATH_STRIDE])
        if length and start <= segment < start + length:
            return path, segment - start
        start += length
    return None


def _segment_kind_mask(
    mask: Tensor, decoded: Tensor, layout: SequenceLayout, position: int
) -> Tensor:
    segment = (position - layout.path_positions) // SEGMENT_STRIDE
    owner = _segment_owner(segment, decoded, layout)
    if owner is None:
        mask[PAD] = True  # beyond every declared path length
        return mask
    path, within = owner
    length = int(decoded[path * PATH_STRIDE])
    mask[int(SegmentType.LINE)] = True
    mask[int(SegmentType.QUAD)] = True
    mask[int(SegmentType.CUBIC)] = True
    if within == length - 1:
        mask[int(SegmentType.CLOSE)] = True  # close may only end a path
    return mask


def _segment_coordinate_mask(
    mask: Tensor, segment: int, index: int, decoded: Tensor, layout: SequenceLayout
) -> Tensor:
    kind_token = int(decoded[layout.segment_type_position(segment)])
    if kind_token not in (int(SegmentType.LINE), int(SegmentType.QUAD), int(SegmentType.CUBIC)):
        mask[PAD] = True  # padding and CLOSE carry no coordinates
        return mask
    control_count, coordinate_count = _coordinate_counts(SegmentType(kind_token))
    if index >= coordinate_count:
        mask[PAD] = True  # unused slots of a shorter segment kind
        return mask
    bins = (
        layout.codec.effective_control_coordinate_bins
        if index < control_count
        else layout.codec.coordinate_bins
    )
    mask[1 : bins + 1] = True
    return mask


class CausalProgramModel(nn.Module):
    """A decoder-only transformer over the flattened program, with KV caching."""

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
    ) -> None:
        super().__init__()
        self.layout = layout
        self.d_model = d_model
        self.heads = heads
        self.metric_coordinates = metric_coordinates
        self.token_embedding = nn.Embedding(layout.vocabulary, d_model)
        # Gate G verified this on the same codec: a coordinate token is a point on a
        # quarter-unit lattice, and a categorical table has no way to say two of them
        # are adjacent. Its trained table scored Spearman -0.136 against bin distance -
        # not weakly ordered, faintly anti-ordered - and replacing it with Fourier
        # features of the decoded view-unit value took render recovery from 0.0447 to
        # 0.2511 while shrinking the model. The causal model inherits the same defect
        # verbatim, so it gets the same correction, behind a flag so earlier runs stay
        # reproducible.
        self.coordinate_projection = (
            nn.Linear(2 * metric_coordinates + 2, d_model) if metric_coordinates > 0 else None
        )
        if metric_coordinates > 0:
            self.register_buffer("_slot_index", _slot_index(layout), persistent=False)
            self.register_buffer("_kind_position", _kind_position(layout), persistent=False)
            self.register_buffer(
                "_control_counts", torch.tensor([0, 0, 2, 4, 0], dtype=torch.long),
                persistent=False,
            )
        self.position_embedding = nn.Embedding(layout.length + 1, d_model)
        self.group_embedding = (
            nn.Embedding(group_vocab_size, d_model) if group_vocab_size > 0 else None
        )
        self.subgroup_embedding = (
            nn.Embedding(subgroup_vocab_size, d_model) if subgroup_vocab_size > 0 else None
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.blocks = nn.ModuleList(
            _CausalBlock(d_model, heads, feedforward, dropout) for _ in range(layers)
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, layout.vocabulary)

    def forward(
        self,
        tokens: Tensor,
        condition: dict[str, Tensor] | None = None,
        cache: list[tuple[Tensor, Tensor]] | None = None,
        offset: int = 0,
        kinds: Tensor | None = None,
    ) -> tuple[Tensor, list[tuple[Tensor, Tensor]]]:
        """Logits for each supplied position, plus the updated key/value cache.

        `offset` is the absolute position of the first supplied token, so a cached decode
        step passes one token and its true position rather than re-reading the prefix.

        `kinds` is required only under metric coordinates, and carries the segment-kind
        token governing each supplied token's own slot. The caller supplies it because
        a one-token cached step cannot see the kind, which sits up to six positions back.
        """

        positions = torch.arange(
            offset, offset + tokens.shape[1], device=tokens.device
        ).clamp_max(self.layout.length)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)[None]
        if self.coordinate_projection is not None:
            if kinds is None:
                raise ValueError("metric coordinates need the governing segment kinds")
            hidden = self._apply_metric_coordinates(hidden, tokens, positions, kinds)
        if self.group_embedding is not None or self.subgroup_embedding is not None:
            if condition is None:
                raise ValueError("this model was built with structured conditioning")
            extra = torch.zeros_like(hidden[:, 0])
            if self.group_embedding is not None:
                extra = extra + self.group_embedding(condition["group"])
            if self.subgroup_embedding is not None:
                extra = extra + self.subgroup_embedding(condition["subgroup"])
            hidden = hidden + extra[:, None, :]
        if self.dropout is not None:
            hidden = self.dropout(hidden)
        updated: list[tuple[Tensor, Tensor]] = []
        for index, block in enumerate(self.blocks):
            past = cache[index] if cache is not None else None
            hidden, present = block(hidden, past)
            updated.append(present)
        return self.head(self.norm(hidden)), updated

    def _apply_metric_coordinates(
        self, hidden: Tensor, tokens: Tensor, positions: Tensor, kinds: Tensor
    ) -> Tensor:
        """Replace the categorical embedding with Fourier features of the decoded value.

        The supplied token at index i occupies absolute slot `positions[i] - 1`: the
        model reads the token before the one it predicts. Position 0 has no predecessor
        and is left categorical, which costs nothing - it is the start token.
        """

        codec = self.layout.codec
        slot_index = cast(Tensor, self._slot_index)
        control_counts = cast(Tensor, self._control_counts)
        absolute = (positions - 1).clamp_min(0)
        slot = slot_index.to(tokens.device)[absolute][None].expand_as(tokens)
        is_coordinate = (slot >= 0) & (positions > 0)[None] & (tokens > 0)

        endpoint_step = 72.0 / (codec.coordinate_bins - 1)
        control_span = codec.control_coordinate_max - codec.control_coordinate_min
        control_step = control_span / (codec.effective_control_coordinate_bins - 1)
        index = (tokens.to(torch.float32) - 1.0).clamp_min(0.0)
        counts = control_counts.to(tokens.device)[kinds.clamp(0, 4)]
        is_control = slot < counts
        value = torch.where(
            is_control,
            codec.control_coordinate_min + index * control_step,
            index * endpoint_step,
        ) * is_coordinate.to(torch.float32)

        frequencies = torch.pow(
            2.0,
            torch.arange(self.metric_coordinates, dtype=torch.float32, device=tokens.device),
        ) * torch.pi / 72.0
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
        keep = is_coordinate[..., None].to(hidden.dtype)
        categorical: Tensor = self.token_embedding(tokens)
        # Only the token term is replaced; position and conditioning still apply.
        return hidden + keep * (replacement - categorical)


class _CausalBlock(nn.Module):
    def __init__(
        self, d_model: int, heads: int, feedforward: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by heads")
        # Default zero so every arm run before this reproduces byte-identically. A
        # dropout layer that drops nothing is still a layer, so it is omitted entirely
        # rather than constructed with p=0.
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None
        self.heads = heads
        self.head_dim = d_model // heads
        self.norm_attention = nn.LayerNorm(d_model)
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.project = nn.Linear(d_model, d_model)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward), nn.GELU(), nn.Linear(feedforward, d_model)
        )

    def forward(
        self, hidden: Tensor, past: tuple[Tensor, Tensor] | None
    ) -> tuple[Tensor, tuple[Tensor, Tensor]]:
        batch, length, _ = hidden.shape
        normed = self.norm_attention(hidden)

        def split(value: Tensor) -> Tensor:
            return value.reshape(batch, -1, self.heads, self.head_dim).transpose(1, 2)

        q = split(self.query(normed))
        k = split(self.key(normed))
        v = split(self.value(normed))
        if past is not None:
            k = torch.cat((past[0], k), dim=2)
            v = torch.cat((past[1], v), dim=2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        total = k.shape[2]
        # Each query attends to everything already cached plus itself and earlier tokens
        # in this call, and nothing after - identical to a full-sequence causal mask.
        causal = torch.ones(length, total, dtype=torch.bool, device=hidden.device).tril(
            diagonal=total - length
        )
        scores = scores.masked_fill(~causal[None, None], float("-inf"))
        attended = (scores.softmax(dim=-1) @ v).transpose(1, 2).reshape(batch, length, -1)
        projected = self.project(attended)
        if self.dropout is not None:
            projected = self.dropout(projected)
        hidden = hidden + projected
        forwarded = self.feedforward(self.norm_feedforward(hidden))
        if self.dropout is not None:
            forwarded = self.dropout(forwarded)
        return hidden + forwarded, (k, v)


@torch.no_grad()
def generate(
    model: CausalProgramModel,
    template: PackedTensorProgram,
    condition: dict[str, Tensor] | None,
    *,
    greedy: bool = True,
    rng: np.random.Generator | None = None,
) -> tuple[PackedTensorProgram, int]:
    """Decode one program left to right with KV caching and legal-token masks.

    Returns the program and the number of forward calls, which is one per position -
    the point of the cache is that each costs a single token rather than a prefix.
    """

    layout = model.layout
    device = next(model.parameters()).device
    decoded = torch.zeros(layout.length, dtype=torch.long)
    cache: list[tuple[Tensor, Tensor]] | None = None
    current = torch.zeros((1, 1), dtype=torch.long, device=device)
    calls = 0
    kind_at = _kind_position(layout)
    for position in range(layout.length):
        # The token being fed in occupies slot `position - 1`; its governing kind was
        # decoded earlier in this same loop, so it is always available here.
        kinds = torch.zeros((1, 1), dtype=torch.long, device=device)
        if position > 0 and int(kind_at[position - 1]) >= 0:
            kinds[0, 0] = int(decoded[int(kind_at[position - 1])])
        logits, cache = model(current, condition, cache, offset=position, kinds=kinds)
        calls += 1
        mask = legal_mask(position, decoded, layout).to(device)
        if not bool(mask.any()):
            # A position with no legal token means the grammar encoding is wrong, and it
            # would otherwise surface as NaN probabilities several steps later.
            raise ValueError(f"no legal token at position {position}")
        masked = logits[0, -1].masked_fill(~mask, float("-inf"))
        if greedy:
            token = int(masked.argmax())
        else:
            probabilities = torch.softmax(masked, dim=-1).cpu().numpy().astype(np.float64)
            probabilities = probabilities / probabilities.sum()
            generator = rng if rng is not None else np.random.default_rng(0)
            token = int(generator.choice(len(probabilities), p=probabilities))
        decoded[position] = token
        current = torch.tensor([[token]], dtype=torch.long, device=device)
    return unflatten_program(decoded, template, layout), calls
