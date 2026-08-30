"""Small fixed-topology geometry denoiser for the Gate E learning proof."""

from __future__ import annotations

import copy
from dataclasses import fields

import numpy as np
import torch
from torch import Tensor, nn

from mojidiff.representation.packed import PackedTensorProgram
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
    ) -> None:
        super().__init__()
        self.max_paths = codec.max_paths
        self.total_segment_slots = total_segment_slots
        self.endpoint_bins = codec.coordinate_bins
        self.control_bins = codec.effective_control_coordinate_bins
        sizes = _path_field_sizes(codec)
        self.path_embeddings = nn.ModuleDict(
            {name: nn.Embedding(size, d_model, padding_idx=0) for name, size in sizes.items()}
        )
        self.endpoint_embedding = nn.Embedding(self.endpoint_bins + 1, d_model, padding_idx=0)
        self.segment_type_embedding = nn.Embedding(5, d_model, padding_idx=0)
        self.coordinate_embedding = nn.Embedding(self.control_bins + 1, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(codec.max_paths + total_segment_slots, d_model)
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

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        path_hidden = torch.zeros(
            (*batch["path_length"].shape, self.position_embedding.embedding_dim),
            dtype=torch.float32,
            device=batch["path_length"].device,
        )
        for name in _PATH_FIELDS:
            path_hidden = path_hidden + self.path_embeddings[name](batch[name])
        path_hidden = path_hidden + self.endpoint_embedding(batch["start"][:, :, 0])
        path_hidden = path_hidden + self.endpoint_embedding(batch["start"][:, :, 1])

        segment_hidden = self.segment_type_embedding(batch["segment_type"])
        for coordinate_index in range(6):
            segment_hidden = segment_hidden + self.coordinate_embedding(
                batch["coordinates"][:, :, coordinate_index]
            )
        hidden = torch.cat((path_hidden, segment_hidden), dim=1)
        positions = torch.arange(hidden.shape[1], device=hidden.device)
        hidden = hidden + self.position_embedding(positions)[None, :, :]
        encoded = self.encoder(hidden)
        path_encoded = encoded[:, : self.max_paths]
        segment_encoded = encoded[:, self.max_paths :]
        start = self.start_head(path_encoded).reshape(
            *path_encoded.shape[:2], 2, self.endpoint_bins + 1
        )
        coordinates = self.coordinate_head(segment_encoded).reshape(
            *segment_encoded.shape[:2], 6, self.control_bins + 1
        )
        return start, coordinates


def corrupt_geometry(
    clean: PackedTensorProgram,
    codec: CodecConfig,
    probability: float,
    rng: np.random.Generator,
) -> PackedTensorProgram:
    """Apply fixed-topology, role-aware uniform coordinate corruption."""

    if not 0 <= probability <= 1:
        raise ValueError("corruption probability must be within 0..1")
    noisy = copy.deepcopy(clean)
    for path_index, raw_length in enumerate(clean.path_length):
        if int(raw_length) == 0:
            break
        for coordinate_index in range(2):
            if rng.random() < probability:
                noisy.start[path_index, coordinate_index] = int(
                    rng.integers(1, codec.coordinate_bins + 1)
                )
    offset = 0
    for raw_length in clean.path_length:
        length = int(raw_length)
        if not length:
            break
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
                noisy.coordinates[segment_index, coordinate_index] = int(
                    rng.integers(1, bins + 1)
                )
        offset += length
    return noisy


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
    logits: tuple[Tensor, Tensor], clean: dict[str, Tensor], codec: CodecConfig
) -> tuple[Tensor, dict[str, int]]:
    """Cross-entropy over active legal coordinate fields only."""

    start_logits, coordinate_logits = logits
    active_paths = clean["path_length"] > 0
    start_mask = active_paths[:, :, None].expand(-1, -1, 2)
    start_targets = clean["start"][start_mask]
    selected_start = start_logits[start_mask]
    selected_start = selected_start[:, 1 : codec.coordinate_bins + 1]
    start_loss = nn.functional.cross_entropy(selected_start, start_targets - 1)
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
            coordinate_losses.append(nn.functional.cross_entropy(field_logits, targets - 1))
            coordinate_correct += int(
                (field_logits.argmax(dim=-1) + 1 == targets).sum().item()
            )
            coordinate_total += int(targets.numel())
    loss = start_loss + torch.stack(coordinate_losses).mean()
    return loss, {
        "correct": start_correct + coordinate_correct,
        "total": int(start_targets.numel()) + coordinate_total,
    }


def predict_clean_geometry(
    noisy: PackedTensorProgram,
    logits: tuple[Tensor, Tensor],
    codec: CodecConfig,
) -> PackedTensorProgram:
    """Project logits through known topology and legal coordinate vocabularies."""

    result = copy.deepcopy(noisy)
    start_logits, coordinate_logits = logits
    for path_index, raw_length in enumerate(result.path_length):
        if not int(raw_length):
            break
        for coordinate_index in range(2):
            result.start[path_index, coordinate_index] = int(
                start_logits[0, path_index, coordinate_index, 1 : codec.coordinate_bins + 1]
                .argmax()
                .item()
                + 1
            )
    offset = 0
    for raw_length in result.path_length:
        length = int(raw_length)
        if not length:
            break
        for segment_index in range(offset, offset + length):
            kind = SegmentType(int(result.segment_type[segment_index]))
            control_count, coordinate_count = _coordinate_counts(kind)
            for coordinate_index in range(coordinate_count):
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
