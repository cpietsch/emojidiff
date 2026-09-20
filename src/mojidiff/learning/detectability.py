"""Is a corrupted geometry field distinguishable from a legitimate one at all?

Gate G closed on the finding that a trained corrupted-field detector reaches only
0.1058 recall at 0.4312 precision against a 0.3494 base rate under factorized
role-uniform corruption.  Two readings fit that: the corruption is genuinely not
identifiable, or it is identifiable and the model failed to learn it.  The two have
very different consequences, and telling them apart needs no training.

This probe scores every legal coordinate field with a fixed local-continuity statistic
and measures how well that statistic alone separates corrupted fields from retained
ones.  Contours are locally smooth, so a coordinate drawn uniformly from its legal
vocabulary should usually break continuity; if even a trivial statistic separates them,
the corruption is identifiable and the trained detector simply did not find it.

The statistic is the absolute distance, in view units, between a field and the value in
the same slot of the adjacent segment of the same path.  It uses no learned parameters,
so it is a floor on detectability rather than a ceiling.
"""

from __future__ import annotations

import numpy as np

from mojidiff.representation.packed import PackedTensorProgram
from mojidiff.representation.program import CodecConfig, SegmentType

_COORDINATE_COUNTS = {
    SegmentType.LINE: (0, 2),
    SegmentType.QUAD: (2, 4),
    SegmentType.CUBIC: (4, 6),
    SegmentType.CLOSE: (0, 0),
}


def field_values(
    program: PackedTensorProgram, codec: CodecConfig
) -> dict[tuple[int, int], float]:
    """Decode every legal coordinate field to view units, keyed by (segment, slot)."""

    endpoint_step = 72.0 / (codec.coordinate_bins - 1)
    control_span = codec.control_coordinate_max - codec.control_coordinate_min
    control_step = control_span / (codec.effective_control_coordinate_bins - 1)
    values: dict[tuple[int, int], float] = {}
    offset = 0
    for raw_length in program.path_length:
        length = int(raw_length)
        if not length:
            break
        for segment_index in range(offset, offset + length):
            kind = SegmentType(int(program.segment_type[segment_index]))
            control_count, coordinate_count = _COORDINATE_COUNTS[kind]
            for slot in range(coordinate_count):
                token = int(program.coordinates[segment_index, slot])
                if token <= 0:
                    continue
                if slot < control_count:
                    values[(segment_index, slot)] = (
                        codec.control_coordinate_min + (token - 1) * control_step
                    )
                else:
                    values[(segment_index, slot)] = (token - 1) * endpoint_step
        offset += length
    return values


def continuity_scores(
    program: PackedTensorProgram, codec: CodecConfig
) -> dict[tuple[int, int], float]:
    """Local-continuity score per legal field: distance to the same slot next door."""

    values = field_values(program, codec)
    scores: dict[tuple[int, int], float] = {}
    offset = 0
    for raw_length in program.path_length:
        length = int(raw_length)
        if not length:
            break
        for segment_index in range(offset, offset + length):
            for slot in range(6):
                key = (segment_index, slot)
                if key not in values:
                    continue
                neighbours = [
                    values[(other, slot)]
                    for other in (segment_index - 1, segment_index + 1)
                    if offset <= other < offset + length and (other, slot) in values
                ]
                if neighbours:
                    # Mean, not min. At a 35% corruption rate adjacent fields are often
                    # both corrupted, and `min` would then score a corrupted field low
                    # whenever it happened to land near its equally corrupted neighbour.
                    scores[key] = float(
                        np.mean([abs(values[key] - value) for value in neighbours])
                    )
        offset += length
    return scores


def roc_auc(positive: list[float], negative: list[float]) -> float:
    """Rank-based AUC: the chance a random positive outscores a random negative."""

    if not positive or not negative:
        return float("nan")
    combined = np.concatenate([np.asarray(positive), np.asarray(negative)])
    order = combined.argsort(kind="stable")
    ranks = np.empty(len(combined), dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1, dtype=np.float64)
    # Average ranks within ties, so a statistic that cannot separate scores 0.5.
    unique, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros(len(unique), dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    ranks = (sums / counts)[inverse]
    positive_rank_sum = float(ranks[: len(positive)].sum())
    count_positive = len(positive)
    count_negative = len(negative)
    return (
        positive_rank_sum - count_positive * (count_positive + 1) / 2
    ) / (count_positive * count_negative)


def separation(
    clean: PackedTensorProgram, noisy: PackedTensorProgram, codec: CodecConfig
) -> tuple[list[float], list[float]]:
    """Continuity scores of corrupted fields and of retained fields, from `noisy`."""

    scores = continuity_scores(noisy, codec)
    corrupted: list[float] = []
    retained: list[float] = []
    for key, score in scores.items():
        segment_index, slot = key
        changed = int(noisy.coordinates[segment_index, slot]) != int(
            clean.coordinates[segment_index, slot]
        )
        (corrupted if changed else retained).append(score)
    return corrupted, retained
