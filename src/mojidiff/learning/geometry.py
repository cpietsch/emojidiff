"""Typed, fixed-topology corruption and denoising utilities.

The Gate F operators here deliberately keep path lengths, segment kinds, styles, and
typed padding fixed.  They therefore isolate geometry corruption correlation; they are
not a claim to model topology or a mathematically specified D3PM transition.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import fields
from typing import cast

import numpy as np
import torch
from torch import Tensor, nn

from mojidiff.representation.packed import PackedTensorProgram, validate_packed_tensor_program
from mojidiff.representation.program import CodecConfig, SegmentType

_PATH_FIELDS = (
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


class GeometryDenoiser(nn.Module):
    """Bidirectional record transformer that predicts clean geometry only."""

    def __init__(
        self,
        codec: CodecConfig,
        total_segment_slots: int,
        *,
        d_model: int,
        heads: int,
        layers: int,
        feedforward: int,
        group_vocab_size: int = 0,
        subgroup_vocab_size: int = 0,
        noise_level_features: int = 0,
        metric_coordinates: int = 0,
        slot_binding: bool = False,
        edit_mask: bool = False,
        mask_padding: bool = False,
    ) -> None:
        super().__init__()
        self.max_paths = codec.max_paths
        self.total_segment_slots = total_segment_slots
        self.endpoint_bins = codec.coordinate_bins
        self.control_bins = codec.effective_control_coordinate_bins
        self.control_minimum = codec.control_coordinate_min
        self.control_maximum = codec.control_coordinate_max
        sizes = _path_field_sizes(codec)
        self.path_embeddings = nn.ModuleDict(
            {name: nn.Embedding(size, d_model, padding_idx=0) for name, size in sizes.items()}
        )
        # Coordinates live on a quarter-unit lattice - a metric space - but a token
        # table encodes them as unordered categories, so the local-difference the
        # continuity statistic computes in closed form has no operand the encoder can
        # subtract. Measured on a trained checkpoint: Spearman between embedding
        # distance and bin distance is -0.136 and adjacent bins sit 0.989 of the
        # all-pairs mean apart, i.e. no order at all. With `metric_coordinates` the
        # input side decodes each token to its view-unit value instead, using the
        # role-correct affine map, and projects Fourier features of it. The output
        # heads stay categorical. Replacement, not augmentation: a numeric code added
        # on top of the embeddings is swamped by them.
        # The categorical tables are not allocated at all under metric encoding, so the
        # change is a strict capacity REDUCTION and cannot be confounded with the
        # capacity arm already eliminated.
        self.metric_coordinates = metric_coordinates
        self.endpoint_embedding = (
            None
            if metric_coordinates
            else nn.Embedding(self.endpoint_bins + 1, d_model, padding_idx=0)
        )
        self.segment_type_embedding = nn.Embedding(5, d_model, padding_idx=0)
        self.coordinate_embedding = (
            None
            if metric_coordinates
            else nn.Embedding(self.control_bins + 1, d_model, padding_idx=0)
        )
        self.position_embedding = nn.Embedding(codec.max_paths + total_segment_slots, d_model)
        self.group_embedding = (
            nn.Embedding(group_vocab_size, d_model) if group_vocab_size > 0 else None
        )
        self.subgroup_embedding = (
            nn.Embedding(subgroup_vocab_size, d_model) if subgroup_vocab_size > 0 else None
        )
        # Continuous conditioning on how corrupted the input is. Without it the model
        # has no way to know whether to trust `x_t` or to overwrite it, and a model
        # trained at one fixed corruption level never needs to ask. Sinusoidal features
        # rather than a bucket embedding, so a level never seen in training still lands
        # somewhere sensible between the levels that were.
        self.noise_level_features = noise_level_features
        self.noise_level_projection = (
            nn.Linear(2 * noise_level_features, d_model) if noise_level_features > 0 else None
        )
        # Bind each coordinate value to the slot it occupies. Without this the six
        # coordinate lookups - all from one shared table - are summed into a single
        # vector, so a segment is represented by an unordered bag of its values and the
        # model provably cannot tell which coordinate held which. It still has to
        # predict all six separately, which makes copying its own input impossible.
        # Adding a per-slot vector would not help: a sum of sums is still symmetric.
        # Multiplying binds value to slot while costing 8 * d_model parameters, so the
        # fix is structural rather than a capacity change.
        if metric_coordinates:
            width = 2 + 2 * metric_coordinates
            self.coordinate_projection: nn.Linear | None = nn.Linear(6 * width, d_model)
            self.start_projection: nn.Linear | None = nn.Linear(2 * width, d_model)
            self.register_buffer(
                "control_counts", torch.tensor([0, 0, 2, 4, 0], dtype=torch.long), persistent=False
            )
        else:
            self.coordinate_projection = None
            self.start_projection = None
        self.slot_binding = slot_binding
        self.coordinate_slot: nn.Parameter | None = None
        self.start_slot: nn.Parameter | None = None
        if slot_binding:
            self.coordinate_slot = nn.Parameter(torch.normal(1.0, 0.02, (6, d_model)))
            self.start_slot = nn.Parameter(torch.normal(1.0, 0.02, (2, d_model)))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=feedforward,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=layers,
            enable_nested_tensor=False,
        )
        self.start_head = nn.Linear(d_model, 2 * (self.endpoint_bins + 1))
        self.coordinate_head = nn.Linear(d_model, 6 * (self.control_bins + 1))
        # Per-field keep-or-change decision. Without it, representing "leave this field
        # alone" means reconstructing its exact token through a 289- or 417-way softmax,
        # so the identity policy - which beats every trained model so far - costs the
        # model as much as inventing new values. With it, identity is predict-keep
        # everywhere, and the value heads only have to handle fields marked changed.
        self.mask_padding = mask_padding
        self.edit_mask = edit_mask
        self.start_keep_head = nn.Linear(d_model, 2 * 2) if edit_mask else None
        self.coordinate_keep_head = nn.Linear(d_model, 6 * 2) if edit_mask else None

    def _metric_features(
        self, tokens: Tensor, segment_type: Tensor, *, endpoints_only: bool = False
    ) -> Tensor:
        """Decode tokens to view units and return Fourier features, flattened per slot.

        A slot's role depends on the segment kind - slot 0 is an endpoint for a LINE and
        a control handle for a CUBIC - and the two decode with different affine maps
        over different ranges, so the kind selects the map per field.
        """

        endpoint_step = 72.0 / (self.endpoint_bins - 1)
        control_span = self.control_maximum - self.control_minimum
        control_step = control_span / (self.control_bins - 1)
        index = (tokens.to(torch.float32) - 1.0).clamp_min(0.0)
        endpoint_value = index * endpoint_step
        if endpoints_only:
            value = endpoint_value
        else:
            control_value = self.control_minimum + index * control_step
            slots = torch.arange(tokens.shape[2], device=tokens.device)[None, None, :]
            counts = cast(Tensor, self.control_counts)
            is_control = slots < counts[segment_type][:, :, None]
            value = torch.where(is_control, control_value, endpoint_value)
        legal = (tokens > 0).to(torch.float32)
        value = value * legal
        scaled = value[..., None] * (
            torch.pow(2.0, torch.arange(self.metric_coordinates, dtype=torch.float32,
                                        device=tokens.device))[None, None, None, :]
            * torch.pi / 72.0
        )
        features = torch.cat(
            (
                (value / 72.0)[..., None],
                legal[..., None],
                torch.sin(scaled),
                torch.cos(scaled),
            ),
            dim=-1,
        )
        return features.reshape(*tokens.shape[:2], -1)

    def forward(
        self,
        batch: dict[str, Tensor],
        condition: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Value logits only, so every existing caller is unchanged."""

        start, coordinates, _ = self.forward_with_edits(batch, condition)
        return start, coordinates

    def forward_with_edits(
        self,
        batch: dict[str, Tensor],
        condition: dict[str, Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, Tensor] | None]:
        # Validate the condition contract before doing any work, so a config and a
        # checkpoint that disagree about conditioning fail immediately and loudly
        # rather than part-way through a forward pass.
        expected: set[str] = set()
        if self.group_embedding is not None or self.subgroup_embedding is not None:
            expected |= {"group", "subgroup"}
        if self.noise_level_projection is not None:
            expected.add("noise_level")
        if expected or condition is not None:
            if condition is None or set(condition) != expected:
                raise ValueError(f"condition must contain exactly {sorted(expected)}")
            batch_size = batch["path_length"].shape[:1]
            for name in sorted(expected):
                if condition[name].shape != batch_size:
                    raise ValueError(f"{name} condition must contain one value per batch item")

        path_hidden = torch.zeros(
            (*batch["path_length"].shape, self.position_embedding.embedding_dim),
            dtype=torch.float32,
            device=batch["path_length"].device,
        )
        for name in _PATH_FIELDS:
            path_hidden = path_hidden + self.path_embeddings[name](batch[name])
        if self.start_projection is not None:
            path_hidden = path_hidden + self.start_projection(
                self._metric_features(
                    batch["start"], torch.zeros_like(batch["start"][:, :, 0]), endpoints_only=True
                )
            )
        elif self.endpoint_embedding is not None:
            for start_index in range(2):
                embedded = self.endpoint_embedding(batch["start"][:, :, start_index])
                if self.start_slot is not None:
                    embedded = embedded * self.start_slot[start_index]
                path_hidden = path_hidden + embedded

        segment_hidden = self.segment_type_embedding(batch["segment_type"])
        if self.coordinate_projection is not None:
            segment_hidden = segment_hidden + self.coordinate_projection(
                self._metric_features(batch["coordinates"], batch["segment_type"])
            )
        elif self.coordinate_embedding is not None:
            for coordinate_index in range(6):
                embedded = self.coordinate_embedding(batch["coordinates"][:, :, coordinate_index])
                if self.coordinate_slot is not None:
                    embedded = embedded * self.coordinate_slot[coordinate_index]
                segment_hidden = segment_hidden + embedded
        hidden = torch.cat((path_hidden, segment_hidden), dim=1)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        if expected:
            assert condition is not None
            conditioning = torch.zeros_like(hidden[:, 0])
            if self.group_embedding is not None:
                conditioning = conditioning + self.group_embedding(condition["group"])
            if self.subgroup_embedding is not None:
                conditioning = conditioning + self.subgroup_embedding(condition["subgroup"])
            if self.noise_level_projection is not None:
                conditioning = conditioning + self.noise_level_projection(
                    _noise_level_features(
                        condition["noise_level"],
                        self.noise_level_features,
                        self.noise_level_projection.weight.dtype,
                    )
                )
            hidden = hidden + conditioning[:, None, :]
        # Mask typed padding. Without this the encoder attends to empty path and
        # segment slots, which for a sparsely packed bucket is a large share of the
        # sequence. Opt-in for the same reason as the loss pooling: runs v1 through v7
        # must stay reproducible under the behaviour they were run with.
        padding_mask = None
        if self.mask_padding:
            padding_mask = torch.cat(
                (batch["path_length"] == 0, batch["segment_type"] == 0), dim=1
            )
            # A row that is entirely padding would make softmax produce NaN, so keep at
            # least the first position live; its output is discarded by the heads anyway.
            padding_mask[:, 0] = False
        encoded = self.encoder(hidden, src_key_padding_mask=padding_mask)
        path_encoded = encoded[:, : self.max_paths]
        segment_encoded = encoded[:, self.max_paths :]
        start = self.start_head(path_encoded).reshape(
            *path_encoded.shape[:2], 2, self.endpoint_bins + 1
        )
        coordinates = self.coordinate_head(segment_encoded).reshape(
            *segment_encoded.shape[:2], 6, self.control_bins + 1
        )
        keep: tuple[Tensor, Tensor] | None = None
        if self.start_keep_head is not None and self.coordinate_keep_head is not None:
            keep = (
                self.start_keep_head(path_encoded).reshape(*path_encoded.shape[:2], 2, 2),
                self.coordinate_keep_head(segment_encoded).reshape(
                    *segment_encoded.shape[:2], 6, 2
                ),
            )
        return start, coordinates, keep


def _noise_level_features(level: Tensor, count: int, dtype: torch.dtype) -> Tensor:
    """Sinusoidal features of a corruption level in 0..1, as [sin | cos] pairs.

    `dtype` follows the projection's parameters rather than being hardcoded, so the
    module works when cast to another precision.
    """

    frequencies = (
        torch.pow(2.0, torch.arange(count, dtype=dtype, device=level.device)) * torch.pi
    )
    scaled = level.to(dtype)[:, None] * frequencies[None, :]
    return torch.cat((torch.sin(scaled), torch.cos(scaled)), dim=1)


def corrupt_geometry(
    clean: PackedTensorProgram,
    codec: CodecConfig,
    probability: float,
    rng: np.random.Generator,
) -> PackedTensorProgram:
    """Backward-compatible name for factorized fixed-topology corruption."""

    return corrupt_factorized_geometry(clean, codec, probability, rng)


def corrupt_factorized_geometry(
    clean: PackedTensorProgram,
    codec: CodecConfig,
    probability: float,
    rng: np.random.Generator,
    *,
    locked_paths: np.ndarray | None = None,
) -> PackedTensorProgram:
    """Independently replace each legal geometry token with a different legal token."""

    if not 0 <= probability <= 1:
        raise ValueError("corruption probability must be within 0..1")
    locks = _locked_path_mask(locked_paths, codec.max_paths)
    noisy = copy.deepcopy(clean)
    for path_index, raw_length in enumerate(clean.path_length):
        if int(raw_length) == 0:
            break
        if locks[path_index]:
            continue
        for coordinate_index in range(2):
            if rng.random() < probability:
                noisy.start[path_index, coordinate_index] = _different_token(
                    int(clean.start[path_index, coordinate_index]), codec.coordinate_bins, rng
                )
    offset = 0
    for path_index, raw_length in enumerate(clean.path_length):
        length = int(raw_length)
        if not length:
            break
        if locks[path_index]:
            offset += length
            continue
        for segment_index in range(offset, offset + length):
            kind = SegmentType(int(clean.segment_type[segment_index]))
            control_count, coordinate_count = _coordinate_counts(kind)
            for coordinate_index in range(coordinate_count):
                if rng.random() >= probability:
                    continue
                bins = (
                    codec.effective_control_coordinate_bins
                    if coordinate_index < control_count
                    else codec.coordinate_bins
                )
                noisy.coordinates[segment_index, coordinate_index] = _different_token(
                    int(clean.coordinates[segment_index, coordinate_index]), bins, rng
                )
        offset += length
    return noisy


def role_token_marginals(
    programs: Sequence[PackedTensorProgram], codec: CodecConfig
) -> np.ndarray:
    """Token frequencies per (segment kind, coordinate slot), as a [5, 6, vocab] table.

    Keyed by kind as well as slot because the role of a slot depends on the kind: slot 0
    is an endpoint for a LINE but a control handle for a QUAD or CUBIC, and the two have
    different legal vocabularies - 289 endpoint bins against 417 control bins. A table
    keyed by slot alone would put mass on tokens that are illegal for that role, which
    the packed validator rejects.

    Within each legal support a token never observed keeps a Laplace floor so it stays
    reachable; everything outside the support is exactly zero.
    """

    vocabulary = codec.effective_control_coordinate_bins + 1
    counts = np.zeros((5, 6, vocabulary), dtype=np.float64)
    for kind in (SegmentType.LINE, SegmentType.QUAD, SegmentType.CUBIC):
        control_count, coordinate_count = _coordinate_counts(kind)
        for slot in range(coordinate_count):
            bins = (
                codec.effective_control_coordinate_bins
                if slot < control_count
                else codec.coordinate_bins
            )
            counts[int(kind), slot, 1 : bins + 1] = 1.0
    countable = {int(SegmentType.LINE), int(SegmentType.QUAD), int(SegmentType.CUBIC)}
    for program in programs:
        for index, raw_kind in enumerate(program.segment_type):
            observed = int(raw_kind)
            if observed not in countable:
                continue
            _, coordinate_count = _coordinate_counts(SegmentType(observed))
            for slot in range(coordinate_count):
                token = int(program.coordinates[index, slot])
                if token > 0 and counts[observed, slot, token] > 0:
                    counts[observed, slot, token] += 1.0
    totals = counts.sum(axis=2, keepdims=True)
    return np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)


def corrupt_marginal_geometry(
    clean: PackedTensorProgram,
    codec: CodecConfig,
    probability: float,
    rng: np.random.Generator,
    marginal: np.ndarray,
) -> PackedTensorProgram:
    """Replace fields with a different token drawn from that role's corpus marginal.

    Factorized corruption draws uniformly from the full legal vocabulary, so a large
    share of replacements land on tokens real icons essentially never use. A detector
    that only knows the corpus marginal then reaches 2.831 precision lift with no
    context at all, which makes detection under that process largely a density test
    rather than a geometric one. Drawing from the marginal instead makes every corrupted
    token in-distribution by construction, so identifying it requires the surrounding
    geometry.

    The marginal is keyed by segment kind as well as slot, because a slot's role - and
    therefore its legal vocabulary - depends on the kind.
    """

    if not 0 <= probability <= 1:
        raise ValueError("corruption probability must be within 0..1")
    if marginal.shape != (5, 6, codec.effective_control_coordinate_bins + 1):
        raise ValueError("marginal must be a [5, 6, vocabulary] table for this codec")
    noisy = copy.deepcopy(clean)
    offset = 0
    for path_index, raw_length in enumerate(clean.path_length):
        length = int(raw_length)
        if not length:
            break
        for coordinate_index in range(2):
            if rng.random() < probability:
                noisy.start[path_index, coordinate_index] = _different_token(
                    int(clean.start[path_index, coordinate_index]), codec.coordinate_bins, rng
                )
        for segment_index in range(offset, offset + length):
            kind = SegmentType(int(clean.segment_type[segment_index]))
            _, coordinate_count = _coordinate_counts(kind)
            for slot in range(coordinate_count):
                if rng.random() >= probability:
                    continue
                current = int(clean.coordinates[segment_index, slot])
                weights = marginal[int(kind), slot].copy()
                weights[current] = 0.0
                total = weights.sum()
                if total <= 0:
                    continue
                noisy.coordinates[segment_index, slot] = int(
                    rng.choice(len(weights), p=weights / total)
                )
        offset += length
    validate_packed_tensor_program(noisy, codec, clean.coordinates.shape[0])
    return noisy


def corrupt_path_correlated_geometry(
    clean: PackedTensorProgram,
    codec: CodecConfig,
    probability: float,
    rng: np.random.Generator,
) -> PackedTensorProgram:
    """Corrupt complete active geometry blocks behind one independently sampled path gate.

    Once a path gate opens, every legal start/coordinate token in that path is replaced
    by a different value in its role-specific vocabulary.  This preserves the packed
    grammar exactly while making corruption correlation, rather than token rate, the
    treatment factor against :func:`corrupt_factorized_geometry`.
    """

    _require_probability(probability)
    noisy = copy.deepcopy(clean)
    offset = 0
    for path_index, raw_length in enumerate(clean.path_length):
        length = int(raw_length)
        if not length:
            break
        if rng.random() < probability:
            for coordinate_index in range(2):
                noisy.start[path_index, coordinate_index] = _different_token(
                    int(clean.start[path_index, coordinate_index]), codec.coordinate_bins, rng
                )
            for segment_index in range(offset, offset + length):
                kind = SegmentType(int(clean.segment_type[segment_index]))
                control_count, coordinate_count = _coordinate_counts(kind)
                for coordinate_index in range(coordinate_count):
                    bins = (
                        codec.effective_control_coordinate_bins
                        if coordinate_index < control_count
                        else codec.coordinate_bins
                    )
                    noisy.coordinates[segment_index, coordinate_index] = _different_token(
                        int(clean.coordinates[segment_index, coordinate_index]), bins, rng
                    )
        offset += length
    return noisy


def corrupt_whole_path_geometry_replacement(
    clean: PackedTensorProgram,
    donor: PackedTensorProgram,
    codec: CodecConfig,
    probability: float,
    rng: np.random.Generator,
) -> PackedTensorProgram:
    """Replace compatible whole geometry paths from a donor under locked topology.

    A path is eligible only when its length and segment-kind sequence match the donor.
    This is the safe fixed-topology form of whole-path replacement: it copies every
    legal geometry token, preserves destination styles/topology, and never synthesizes
    an invalid field.  Incompatible paths are retained and are reported by callers via
    token differences rather than silently projected.
    """

    _require_probability(probability)
    noisy = copy.deepcopy(clean)
    clean_offset = 0
    donor_offset = 0
    for path_index, raw_length in enumerate(clean.path_length):
        length = int(raw_length)
        donor_length = int(donor.path_length[path_index])
        if not length:
            break
        compatible = (
            length == donor_length
            and np.array_equal(
                clean.segment_type[clean_offset : clean_offset + length],
                donor.segment_type[donor_offset : donor_offset + donor_length],
            )
        )
        if compatible and rng.random() < probability:
            noisy.start[path_index] = donor.start[path_index]
            noisy.coordinates[clean_offset : clean_offset + length] = donor.coordinates[
                donor_offset : donor_offset + donor_length
            ]
        clean_offset += length
        donor_offset += donor_length
    validate_packed_tensor_program(noisy, codec, len(noisy.segment_type))
    return noisy


def corrupt_whole_path_geometry_from_pool(
    clean: PackedTensorProgram,
    donor_pool: Sequence[PackedTensorProgram],
    codec: CodecConfig,
    probability: float,
    rng: np.random.Generator,
) -> PackedTensorProgram:
    """Replace each eligible path from a compatible path in another pooled program.

    Compatibility is exact segment-kind sequence equality, irrespective of the donor's
    path slot.  Programs that are object-identical to ``clean`` are excluded, so a
    whole-path replacement always draws its geometry from another training example.
    A path without a compatible external donor remains untouched; callers must audit
    that support before comparing this operator at a claimed marginal noise level.
    """

    _require_probability(probability)
    candidates = _path_pool(donor_pool, exclude=clean)
    noisy = copy.deepcopy(clean)
    offset = 0
    for path_index, raw_length in enumerate(clean.path_length):
        length = int(raw_length)
        if not length:
            break
        signature = tuple(int(value) for value in clean.segment_type[offset : offset + length])
        compatible = candidates.get(signature, ())
        if compatible and rng.random() < probability:
            donor_start, donor_coordinates = compatible[int(rng.integers(0, len(compatible)))]
            noisy.start[path_index] = donor_start
            noisy.coordinates[offset : offset + length] = donor_coordinates
        offset += length
    validate_packed_tensor_program(noisy, codec, len(noisy.segment_type))
    return noisy


def whole_path_pool_support(
    programs: Sequence[PackedTensorProgram],
) -> dict[str, int]:
    """Return exact-path donor support without sampling or changing any program."""

    active_paths = 0
    eligible_paths = 0
    geometry_fields = 0
    eligible_geometry_fields = 0
    for clean in programs:
        candidates = _path_pool(programs, exclude=clean)
        offset = 0
        for raw_length in clean.path_length:
            length = int(raw_length)
            if not length:
                break
            signature = tuple(int(value) for value in clean.segment_type[offset : offset + length])
            fields = 2 + sum(_coordinate_counts(SegmentType(kind))[1] for kind in signature)
            active_paths += 1
            geometry_fields += fields
            if candidates.get(signature):
                eligible_paths += 1
                eligible_geometry_fields += fields
            offset += length
    return {
        "active_paths": active_paths,
        "eligible_paths": eligible_paths,
        "geometry_fields": geometry_fields,
        "eligible_geometry_fields": eligible_geometry_fields,
    }


def _path_pool(
    programs: Sequence[PackedTensorProgram], *, exclude: PackedTensorProgram
) -> dict[tuple[int, ...], list[tuple[np.ndarray, np.ndarray]]]:
    result: dict[tuple[int, ...], list[tuple[np.ndarray, np.ndarray]]] = {}
    for program in programs:
        if program is exclude:
            continue
        offset = 0
        for path_index, raw_length in enumerate(program.path_length):
            length = int(raw_length)
            if not length:
                break
            segment_types = program.segment_type[offset : offset + length]
            signature = tuple(int(value) for value in segment_types)
            result.setdefault(signature, []).append(
                (program.start[path_index], program.coordinates[offset : offset + length])
            )
            offset += length
    return result


def _different_token(original: int, bins: int, rng: np.random.Generator) -> int:
    """Sample uniformly from 1..bins excluding ``original``."""

    if bins < 2 or not 1 <= original <= bins:
        raise ValueError("token replacement requires a valid vocabulary of at least two values")
    replacement = int(rng.integers(1, bins))
    return replacement + 1 if replacement >= original else replacement


def _require_probability(probability: float) -> None:
    if not 0 <= probability <= 1:
        raise ValueError("corruption probability must be within 0..1")


def packed_batch(programs: list[PackedTensorProgram], device: torch.device) -> dict[str, Tensor]:
    """Stack identically shaped packed programs as integer model inputs."""

    if not programs:
        raise ValueError("packed batch cannot be empty")
    return {
        field.name: torch.as_tensor(
            np.stack([getattr(program, field.name) for program in programs]),
            dtype=torch.long,
            device=device,
        )
        for field in fields(PackedTensorProgram)
    }


def geometry_loss_and_accuracy(
    logits: tuple[Tensor, Tensor],
    clean: dict[str, Tensor],
    codec: CodecConfig,
    *,
    pool_over_fields: bool = False,
) -> tuple[Tensor, dict[str, int]]:
    """Cross-entropy over active legal coordinate fields only.

    The default averages over per-(segment kind, coordinate slot) GROUPS, which is a
    defect: QUAD segments are 0.93% of the corpus, so on the 128-icon held-out draw the
    four QUAD groups hold one field each and carry 4/13 of the loss against 40,004 other
    fields - a 5,287x per-field weight ratio. It also renormalises between 9 and 13
    groups depending on whether a batch happens to contain a QUAD segment. Because
    `_evaluate` uses this same function, held-out loss - the checkpoint-selection and
    early-stopping signal - was ~31% four individual fields.

    `pool_over_fields` weights every field equally instead. It is opt-in so that runs
    v1 through v7 stay exactly reproducible under the behaviour they were run with.
    """

    start_logits, coordinate_logits = logits
    active_paths = clean["path_length"] > 0
    start_mask = active_paths[:, :, None].expand(-1, -1, 2)
    start_targets = clean["start"][start_mask]
    selected_start = start_logits[start_mask]
    selected_start = selected_start[:, 1 : codec.coordinate_bins + 1]
    start_loss = nn.functional.cross_entropy(
        selected_start, start_targets - 1, reduction="none" if pool_over_fields else "mean"
    )
    start_correct = int((selected_start.argmax(dim=-1) + 1 == start_targets).sum().item())

    coordinate_losses: list[Tensor] = []
    coordinate_correct = 0
    coordinate_total = 0
    segment_types = clean["segment_type"]
    for kind in (SegmentType.LINE, SegmentType.QUAD, SegmentType.CUBIC):
        control_count, coordinate_count = _coordinate_counts(kind)
        kind_mask = segment_types == int(kind)
        for coordinate_index in range(coordinate_count):
            targets = clean["coordinates"][:, :, coordinate_index][kind_mask]
            if not targets.numel():
                continue
            field_logits = coordinate_logits[:, :, coordinate_index][kind_mask]
            bins = (
                codec.effective_control_coordinate_bins
                if coordinate_index < control_count
                else codec.coordinate_bins
            )
            field_logits = field_logits[:, 1 : bins + 1]
            coordinate_losses.append(
                nn.functional.cross_entropy(
                    field_logits, targets - 1, reduction="none" if pool_over_fields else "mean"
                )
            )
            coordinate_correct += int(
                (field_logits.argmax(dim=-1) + 1 == targets).sum().item()
            )
            coordinate_total += int(targets.numel())
    if pool_over_fields:
        loss = torch.cat([start_loss, *coordinate_losses]).mean()
    else:
        loss = start_loss + torch.stack(coordinate_losses).mean()
    return loss, {
        "correct": start_correct + coordinate_correct,
        "total": int(start_targets.numel()) + coordinate_total,
    }


def geometry_accuracy_by_corruption(
    logits: tuple[Tensor, Tensor],
    noisy: dict[str, Tensor],
    clean: dict[str, Tensor],
    codec: CodecConfig,
) -> dict[str, dict[str, int]]:
    """Split legal-field accuracy by whether corruption actually changed the token."""

    result = {
        "changed": {"correct": 0, "total": 0},
        "retained": {"correct": 0, "total": 0},
    }

    def accumulate(predictions: Tensor, targets: Tensor, changed: Tensor) -> None:
        matches = predictions == targets
        for name, mask in (("changed", changed), ("retained", ~changed)):
            result[name]["correct"] += int((matches & mask).sum().item())
            result[name]["total"] += int(mask.sum().item())

    start_logits, coordinate_logits = logits
    active_paths = clean["path_length"] > 0
    start_mask = active_paths[:, :, None].expand(-1, -1, 2)
    start_targets = clean["start"][start_mask]
    start_predictions = (
        start_logits[start_mask][:, 1 : codec.coordinate_bins + 1].argmax(dim=-1) + 1
    )
    accumulate(
        start_predictions,
        start_targets,
        noisy["start"][start_mask] != start_targets,
    )

    segment_types = clean["segment_type"]
    for kind in (SegmentType.LINE, SegmentType.QUAD, SegmentType.CUBIC):
        control_count, coordinate_count = _coordinate_counts(kind)
        kind_mask = segment_types == int(kind)
        for coordinate_index in range(coordinate_count):
            targets = clean["coordinates"][:, :, coordinate_index][kind_mask]
            if not targets.numel():
                continue
            bins = (
                codec.effective_control_coordinate_bins
                if coordinate_index < control_count
                else codec.coordinate_bins
            )
            predictions = (
                coordinate_logits[:, :, coordinate_index][kind_mask][:, 1 : bins + 1].argmax(
                    dim=-1
                )
                + 1
            )
            accumulate(
                predictions,
                targets,
                noisy["coordinates"][:, :, coordinate_index][kind_mask] != targets,
            )
    return result


def _legal_fields(
    logits: tuple[Tensor, Tensor],
    keep: tuple[Tensor, Tensor] | None,
    noisy: dict[str, Tensor],
    clean: dict[str, Tensor],
    codec: CodecConfig,
) -> list[tuple[Tensor, Tensor | None, Tensor, Tensor]]:
    """Flatten the active legal geometry fields into comparable groups.

    Returns `(value_logits, keep_logits, targets, noisy_tokens)` per group, with tokens
    still 1-based as the codec stores them and `value_logits` already narrowed to that
    group's legal bins.
    """

    start_logits, coordinate_logits = logits
    groups: list[tuple[Tensor, Tensor | None, Tensor, Tensor]] = []

    active_paths = clean["path_length"] > 0
    start_mask = active_paths[:, :, None].expand(-1, -1, 2)
    groups.append(
        (
            start_logits[start_mask][:, 1 : codec.coordinate_bins + 1],
            keep[0][start_mask] if keep is not None else None,
            clean["start"][start_mask],
            noisy["start"][start_mask],
        )
    )

    segment_types = clean["segment_type"]
    for kind in (SegmentType.LINE, SegmentType.QUAD, SegmentType.CUBIC):
        control_count, coordinate_count = _coordinate_counts(kind)
        kind_mask = segment_types == int(kind)
        for coordinate_index in range(coordinate_count):
            targets = clean["coordinates"][:, :, coordinate_index][kind_mask]
            if not targets.numel():
                continue
            bins = (
                codec.effective_control_coordinate_bins
                if coordinate_index < control_count
                else codec.coordinate_bins
            )
            groups.append(
                (
                    coordinate_logits[:, :, coordinate_index][kind_mask][:, 1 : bins + 1],
                    keep[1][:, :, coordinate_index][kind_mask] if keep is not None else None,
                    targets,
                    noisy["coordinates"][:, :, coordinate_index][kind_mask],
                )
            )
    return groups


def distance_kernel_loss(
    logits: Tensor, targets: Tensor, bins: int, tau: float, step: float = 0.25
) -> Tensor:
    """Cross-entropy against a soft target that decays with distance on the lattice.

    Coordinates live on a quarter-unit lattice, but an exact-token softmax scores being
    one bin out exactly as badly as being a hundred out, so a model that has localised a
    coordinate to within a view unit earns nothing for it. This spreads the target mass
    as `exp(-|b - t| * step / tau)`, normalised over the legal bins, so proximity earns
    gradient. `tau` is in view units.

    Returns per-field losses; the caller reduces them.
    """

    positions = torch.arange(bins, device=logits.device, dtype=torch.float32)
    distance = (positions[None, :] - (targets - 1).to(torch.float32)[:, None]).abs()
    weights = torch.exp(-distance * step / tau)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    return -(weights * torch.log_softmax(logits.float(), dim=-1)).sum(dim=-1)


def token_view_values(tokens: Tensor, segment_type: Tensor, codec: CodecConfig) -> Tensor:
    """Decode coordinate tokens to view units, selecting the map by segment kind.

    Slot 0 is an endpoint for a LINE and a control handle for a CUBIC, and the two decode
    over different ranges, so the kind picks the affine map per field. Illegal or padded
    fields decode to zero and should be masked by the caller.
    """

    endpoint_step = 72.0 / (codec.coordinate_bins - 1)
    control_span = codec.control_coordinate_max - codec.control_coordinate_min
    control_step = control_span / (codec.effective_control_coordinate_bins - 1)
    control_counts = torch.tensor([0, 0, 2, 4, 0], dtype=torch.long, device=tokens.device)
    index = (tokens.to(torch.float32) - 1.0).clamp_min(0.0)
    slots = torch.arange(tokens.shape[-1], device=tokens.device)
    is_control = slots[None, None, :] < control_counts[segment_type][:, :, None]
    value = torch.where(
        is_control, codec.control_coordinate_min + index * control_step, index * endpoint_step
    )
    return value * (tokens > 0).to(torch.float32)


def geometric_gate_threshold(
    change_probability: Tensor,
    keep_error: Tensor,
    edit_error: Tensor,
    *,
    steps: int = 200,
) -> tuple[float, float]:
    """Pick the decode threshold that most reduces absolute view-unit error.

    The break-even rule `p > 1/(1+q)` counts a field as recovered only when the predicted
    token is exact, so a value head that is closer but not exacter raises its own
    threshold and edits less of what it could improve - measured, v15 localised 27%
    better than v14 and gated itself more tightly. This instead asks what editing
    actually does to the geometry: for each candidate threshold, the total error saved by
    editing every field above it, and the threshold that saves the most wins.

    Returns the threshold and the error saved per field at it. Computed on withheld
    calibration data, never on what is reported.
    """

    gain = keep_error - edit_error
    best_threshold, best_gain = 1.0, 0.0
    for index in range(steps + 1):
        threshold = index / steps
        flagged = change_probability >= threshold
        total = float(gain[flagged].sum()) if bool(flagged.any()) else 0.0
        if total > best_gain:
            best_gain, best_threshold = total, threshold
    per_field = best_gain / max(int(change_probability.numel()), 1)
    return best_threshold, per_field


def break_even_threshold(value_accuracy: float) -> float:
    """Confidence above which changing a field beats keeping it.

    Keeping a retained field is always right; changing one is right only if the value
    head happens to re-predict the same token, which is negligible. So with `q` the
    value head's accuracy on genuinely corrupted fields, the expected gain from changing
    at confidence `p` is `p*q - (1 - p)`, which is positive exactly when

        p > 1 / (1 + q)

    At q = 0.12 that is 0.893: the model must be nearly certain before editing pays.
    This is derivable from a training-split estimate of `q`, so the decode threshold
    never has to be fitted on the data it is reported on.
    """

    if not 0.0 <= value_accuracy <= 1.0:
        raise ValueError("value accuracy must be within 0..1")
    return 1.0 / (1.0 + value_accuracy)


def edit_mask_predictions(
    value_logits: Tensor, keep_logits: Tensor, noisy_tokens: Tensor, threshold: float = 0.5
) -> Tensor:
    """Copy the input unless the model is confident enough that changing it pays.

    `threshold` is the probability of "change" required to act. The default 0.5 is plain
    argmax, which is what every run before v13 used.
    """

    predicted = value_logits.argmax(dim=-1) + 1
    change_probability = torch.softmax(keep_logits.float(), dim=-1)[..., 0]
    return torch.where(change_probability >= threshold, predicted, noisy_tokens)


def edit_mask_loss_and_accuracy(
    logits: tuple[Tensor, Tensor],
    keep: tuple[Tensor, Tensor],
    noisy: dict[str, Tensor],
    clean: dict[str, Tensor],
    codec: CodecConfig,
    *,
    detection_only: bool = False,
    pool_over_fields: bool = False,
    value_loss_weight: float = 1.0,
    threshold: float = 0.5,
    value_distance_tau: float = 0.0,
) -> tuple[Tensor, dict[str, int]]:
    """Keep-or-change cross-entropy plus value cross-entropy on changed fields only.

    Accuracy is measured on the gated prediction, so it stays directly comparable with
    `geometry_loss_and_accuracy` and with the identity baseline.

    `value_loss_weight` scales the value term. The two objectives share an encoder and
    are wildly unequal at chance - log 2 nats for a 2-class keep decision against log
    417 for the value head, a factor of 8.7 - so an unweighted sum lets the value task,
    which the model performs at 0.1087, dominate the representation and cost the
    detector 1.064 of measured lift. With `detection_only` the value term is dropped
    entirely. The two terms have very
    different scales - a 2-class decision against a 289- or 417-class one - and they
    share an encoder, so the value objective dominates what the encoder learns to
    represent. Dropping it asks whether this model can learn the detection signal at
    all when nothing competes for the representation.
    """

    keep_losses: list[Tensor] = []
    value_losses: list[Tensor] = []
    correct = 0
    total = 0
    for value_logits, keep_logits, targets, noisy_tokens in _legal_fields(
        logits, keep, noisy, clean, codec
    ):
        if keep_logits is None or not targets.numel():
            continue
        keep_targets = (noisy_tokens == targets).long()
        reduction = "none" if pool_over_fields else "mean"
        keep_losses.append(
            nn.functional.cross_entropy(keep_logits, keep_targets, reduction=reduction)
        )
        changed = keep_targets == 0
        if bool(changed.any()):
            if value_distance_tau > 0:
                per_field = distance_kernel_loss(
                    value_logits[changed], targets[changed], value_logits.shape[-1],
                    value_distance_tau,
                )
                value_losses.append(per_field if pool_over_fields else per_field.mean())
            else:
                value_losses.append(
                    nn.functional.cross_entropy(
                        value_logits[changed], targets[changed] - 1, reduction=reduction
                    )
                )
        predictions = edit_mask_predictions(value_logits, keep_logits, noisy_tokens, threshold)
        correct += int((predictions == targets).sum().item())
        total += int(targets.numel())
    if pool_over_fields:
        loss = torch.cat(keep_losses).mean()
        if value_losses and not detection_only:
            loss = loss + value_loss_weight * torch.cat(value_losses).mean()
    else:
        loss = torch.stack(keep_losses).mean()
        if value_losses and not detection_only:
            loss = loss + value_loss_weight * torch.stack(value_losses).mean()
    return loss, {"correct": correct, "total": total}


def edit_mask_accuracy_by_corruption(
    logits: tuple[Tensor, Tensor],
    keep: tuple[Tensor, Tensor],
    noisy: dict[str, Tensor],
    clean: dict[str, Tensor],
    codec: CodecConfig,
    *,
    threshold: float = 0.5,
) -> dict[str, dict[str, int]]:
    """Split gated-prediction accuracy by whether corruption changed the token."""

    result = {
        "changed": {"correct": 0, "total": 0},
        "retained": {"correct": 0, "total": 0},
    }
    for value_logits, keep_logits, targets, noisy_tokens in _legal_fields(
        logits, keep, noisy, clean, codec
    ):
        if keep_logits is None or not targets.numel():
            continue
        predictions = edit_mask_predictions(value_logits, keep_logits, noisy_tokens, threshold)
        matches = predictions == targets
        changed = noisy_tokens != targets
        for name, mask in (("changed", changed), ("retained", ~changed)):
            result[name]["correct"] += int((matches & mask).sum().item())
            result[name]["total"] += int(mask.sum().item())
    return result


def predict_clean_geometry(
    noisy: PackedTensorProgram,
    logits: tuple[Tensor, Tensor],
    codec: CodecConfig,
    *,
    locked_paths: np.ndarray | None = None,
    keep: tuple[Tensor, Tensor] | None = None,
    threshold: float = 0.5,
) -> PackedTensorProgram:
    """Project logits through known topology and legal coordinate vocabularies.

    With `keep`, a field the model marks as unchanged is left exactly as the input had
    it rather than being reconstructed from the value head, so the identity policy is
    reachable by predicting keep everywhere.
    """

    result = copy.deepcopy(noisy)
    locks = _locked_path_mask(locked_paths, codec.max_paths)
    start_logits, coordinate_logits = logits
    start_keep = keep[0] if keep is not None else None
    coordinate_keep = keep[1] if keep is not None else None
    for path_index, raw_length in enumerate(result.path_length):
        if not int(raw_length):
            break
        if locks[path_index]:
            continue
        for coordinate_index in range(2):
            if start_keep is not None and float(
                torch.softmax(start_keep[0, path_index, coordinate_index].float(), dim=-1)[0]
            ) < threshold:
                continue
            result.start[path_index, coordinate_index] = int(
                start_logits[0, path_index, coordinate_index, 1 : codec.coordinate_bins + 1]
                .argmax()
                .item()
                + 1
            )
    offset = 0
    for path_index, raw_length in enumerate(result.path_length):
        length = int(raw_length)
        if not length:
            break
        if locks[path_index]:
            offset += length
            continue
        for segment_index in range(offset, offset + length):
            kind = SegmentType(int(result.segment_type[segment_index]))
            control_count, coordinate_count = _coordinate_counts(kind)
            for coordinate_index in range(coordinate_count):
                if coordinate_keep is not None and float(
                    torch.softmax(
                        coordinate_keep[0, segment_index, coordinate_index].float(), dim=-1
                    )[0]
                ) < threshold:
                    continue
                bins = (
                    codec.effective_control_coordinate_bins
                    if coordinate_index < control_count
                    else codec.coordinate_bins
                )
                result.coordinates[segment_index, coordinate_index] = int(
                    coordinate_logits[0, segment_index, coordinate_index, 1 : bins + 1]
                    .argmax()
                    .item()
                    + 1
                )
        offset += length
    return result


def _locked_path_mask(value: np.ndarray | None, max_paths: int) -> np.ndarray:
    if value is None:
        return np.zeros((max_paths,), dtype=np.bool_)
    if not isinstance(value, np.ndarray) or value.shape != (max_paths,) or value.dtype != np.bool_:
        raise ValueError("locked_paths must be a boolean vector with one entry per path slot")
    return value


def _coordinate_counts(kind: SegmentType) -> tuple[int, int]:
    return {
        SegmentType.LINE: (0, 2),
        SegmentType.QUAD: (2, 4),
        SegmentType.CUBIC: (4, 6),
        SegmentType.CLOSE: (0, 0),
    }[kind]


def _path_field_sizes(codec: CodecConfig) -> dict[str, int]:
    return {
        "path_length": codec.max_segments + 1,
        "layer": codec.max_paths + 1,
        "opacity": len(codec.opacities) + 2,
        "fill": len(codec.palette) + 2,
        "fill_opacity": len(codec.opacities) + 2,
        "stroke": len(codec.palette) + 2,
        "stroke_opacity": len(codec.opacities) + 2,
        "stroke_width": len(codec.stroke_widths) + 2,
        "linecap": 5,
        "linejoin": 5,
        "miter_limit": len(codec.miter_limits) + 2,
        "dash_pattern": len(codec.dash_patterns) + 2,
        "fill_rule": 4,
    }
