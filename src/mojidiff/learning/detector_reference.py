"""How well can corrupted geometry fields be detected from cheap local features?

The zero-parameter continuity statistic reaches 2.13x precision lift, and the trained
denoiser reaches 1.365x. Neither number says how much detection headroom exists. This
fits a logistic regression on a handful of hand-computed local features and reports its
lift, which bounds the useful range from a direction the transformer is not using: if a
linear model on cheap features goes far past 2.13x, the network is leaving a great deal
on the table; if it barely improves, then even a perfect detector would not rescue the
task and the ceiling itself is the finding.

Features are all local and all computed in view units, from the corrupted program alone
- nothing here sees the clean program, so the detector is usable at inference time.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from mojidiff.learning.detectability import field_values
from mojidiff.representation.packed import PackedTensorProgram
from mojidiff.representation.program import CodecConfig, SegmentType

FEATURE_NAMES = (
    "abs_delta_previous",
    "abs_delta_next",
    "abs_deviation_from_neighbour_midpoint",
    "abs_delta_other_slot_same_segment",
    "value",
    "distance_from_viewbox_centre",
    "is_control_handle",
    "has_both_neighbours",
)

_COORDINATE_COUNTS = {
    SegmentType.LINE: (0, 2),
    SegmentType.QUAD: (2, 4),
    SegmentType.CUBIC: (4, 6),
    SegmentType.CLOSE: (0, 0),
}


def features_and_labels(
    clean: PackedTensorProgram, noisy: PackedTensorProgram, codec: CodecConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Per-field local features from `noisy`, and whether that field was corrupted."""

    values = field_values(noisy, codec)
    rows: list[list[float]] = []
    labels: list[int] = []
    offset = 0
    for raw_length in noisy.path_length:
        length = int(raw_length)
        if not length:
            break
        for segment_index in range(offset, offset + length):
            kind = SegmentType(int(noisy.segment_type[segment_index]))
            control_count, coordinate_count = _COORDINATE_COUNTS[kind]
            for slot in range(coordinate_count):
                key = (segment_index, slot)
                if key not in values:
                    continue
                value = values[key]
                previous = (
                    values.get((segment_index - 1, slot))
                    if segment_index - 1 >= offset
                    else None
                )
                following = (
                    values.get((segment_index + 1, slot))
                    if segment_index + 1 < offset + length
                    else None
                )
                # The paired coordinate of the same point: x and y alternate, so the
                # partner slot is slot^1. A corrupted x leaves its y untouched, which
                # makes the pair's joint displacement informative.
                partner = values.get((segment_index, slot ^ 1))
                delta_previous = abs(value - previous) if previous is not None else 0.0
                delta_next = abs(value - following) if following is not None else 0.0
                if previous is not None and following is not None:
                    midpoint = abs(value - 0.5 * (previous + following))
                else:
                    midpoint = 0.0
                rows.append(
                    [
                        delta_previous,
                        delta_next,
                        midpoint,
                        abs(value - partner) if partner is not None else 0.0,
                        value,
                        abs(value - 36.0),
                        1.0 if slot < control_count else 0.0,
                        1.0 if previous is not None and following is not None else 0.0,
                    ]
                )
                labels.append(
                    int(
                        int(noisy.coordinates[segment_index, slot])
                        != int(clean.coordinates[segment_index, slot])
                    )
                )
        offset += length
    if not rows:
        return np.zeros((0, len(FEATURE_NAMES))), np.zeros((0,), dtype=np.int64)
    return np.asarray(rows, dtype=np.float64), np.asarray(labels, dtype=np.int64)


def fit_logistic(
    features: np.ndarray, labels: np.ndarray, *, steps: int = 600, seed: int = 3101
) -> tuple[Tensor, Tensor, Tensor]:
    """Fit a standardised logistic regression; returns weights, bias and feature means."""

    torch.manual_seed(seed)
    x = torch.from_numpy(features)
    y = torch.from_numpy(labels).double()
    mean = x.mean(dim=0)
    scale = x.std(dim=0).clamp_min(1e-6)
    z = (x - mean) / scale
    weight = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
    bias = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weight, bias], max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> Tensor:
        optimizer.zero_grad()
        logits = z @ weight + bias
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, y)
        loss.backward()  # type: ignore[no-untyped-call]
        return loss

    optimizer.step(closure)  # type: ignore[no-untyped-call]
    return weight.detach(), bias.detach(), torch.stack([mean, scale])


def score(features: np.ndarray, weight: Tensor, bias: Tensor, norm: Tensor) -> np.ndarray:
    z = (torch.from_numpy(features) - norm[0]) / norm[1]
    return (z @ weight + bias).numpy()
