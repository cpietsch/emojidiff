"""Packed segment storage for the selected typed SVG representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mojidiff.representation.program import (
    PAD,
    BudgetExceeded,
    CodecConfig,
    IntArray,
    ProgramValidationError,
    TensorProgram,
    serialize_svg,
    validate_tensor_program,
)


@dataclass(frozen=True)
class PackedTensorProgram:
    """Path metadata plus one contiguous painter-ordered segment array."""

    path_length: IntArray
    layer: IntArray
    opacity: IntArray
    fill: IntArray
    fill_opacity: IntArray
    stroke: IntArray
    stroke_opacity: IntArray
    stroke_width: IntArray
    linecap: IntArray
    linejoin: IntArray
    miter_limit: IntArray
    dash_pattern: IntArray
    fill_rule: IntArray
    start: IntArray
    segment_type: IntArray
    coordinates: IntArray


_PATH_VECTOR_FIELDS = (
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


def pack_tensor_program(
    program: TensorProgram, config: CodecConfig, total_segment_slots: int
) -> PackedTensorProgram:
    """Pack active segment prefixes without changing path or painter order."""

    _validate_total_slots(total_segment_slots)
    validate_tensor_program(program, config)
    active_segments = sum(int(value) for value in program.path_length)
    if active_segments > total_segment_slots:
        raise BudgetExceeded(
            f"program needs {active_segments} packed segments but capacity is "
            f"{total_segment_slots}"
        )
    segment_type = np.full((total_segment_slots,), PAD, dtype=np.int64)
    coordinates = np.full((total_segment_slots, 6), PAD, dtype=np.int64)
    offset = 0
    for path_index, raw_length in enumerate(program.path_length):
        length = int(raw_length)
        if not length:
            break
        segment_type[offset : offset + length] = program.segment_type[path_index, :length]
        coordinates[offset : offset + length] = program.coordinates[path_index, :length]
        offset += length
    packed = PackedTensorProgram(
        **{name: getattr(program, name).copy() for name in _PATH_VECTOR_FIELDS},
        start=program.start.copy(),
        segment_type=segment_type,
        coordinates=coordinates,
    )
    validate_packed_tensor_program(packed, config, total_segment_slots)
    return packed


def unpack_tensor_program(
    program: PackedTensorProgram, config: CodecConfig, total_segment_slots: int
) -> TensorProgram:
    """Expand packed segments into the canonical dense validation representation."""

    _validate_packed_shapes(program, config, total_segment_slots)
    active_segments = sum(int(value) for value in program.path_length)
    if active_segments > total_segment_slots:
        raise ProgramValidationError("path lengths exceed packed segment capacity")
    if np.any(program.segment_type[active_segments:] != PAD):
        raise ProgramValidationError("inactive packed segment types must be PAD")
    if np.any(program.coordinates[active_segments:] != PAD):
        raise ProgramValidationError("inactive packed coordinates must be PAD")

    segment_type = np.full(
        (config.max_paths, config.max_segments), PAD, dtype=np.int64
    )
    coordinates = np.full(
        (config.max_paths, config.max_segments, 6), PAD, dtype=np.int64
    )
    offset = 0
    for path_index, raw_length in enumerate(program.path_length):
        length = int(raw_length)
        if length < 0 or length > config.max_segments:
            raise ProgramValidationError("path_length is outside the segment budget")
        if not length:
            continue
        segment_type[path_index, :length] = program.segment_type[offset : offset + length]
        coordinates[path_index, :length] = program.coordinates[offset : offset + length]
        offset += length
    dense = TensorProgram(
        **{name: getattr(program, name).copy() for name in _PATH_VECTOR_FIELDS},
        start=program.start.copy(),
        segment_type=segment_type,
        coordinates=coordinates,
    )
    validate_tensor_program(dense, config)
    return dense


def validate_packed_tensor_program(
    program: PackedTensorProgram, config: CodecConfig, total_segment_slots: int
) -> None:
    """Validate packed shapes, padding, path grammar, styles, and typed tokens."""

    unpack_tensor_program(program, config, total_segment_slots)


def serialize_packed_svg(
    program: PackedTensorProgram, config: CodecConfig, total_segment_slots: int
) -> bytes:
    """Serialize only after expanding through the canonical grammar validator."""

    return serialize_svg(unpack_tensor_program(program, config, total_segment_slots), config)


def _validate_packed_shapes(
    program: PackedTensorProgram, config: CodecConfig, total_segment_slots: int
) -> None:
    _validate_total_slots(total_segment_slots)
    for name in _PATH_VECTOR_FIELDS:
        _require_int_array(getattr(program, name), (config.max_paths,), name)
    _require_int_array(program.start, (config.max_paths, 2), "start")
    _require_int_array(program.segment_type, (total_segment_slots,), "segment_type")
    _require_int_array(program.coordinates, (total_segment_slots, 6), "coordinates")


def _require_int_array(array: IntArray, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(array, np.ndarray) or array.shape != shape:
        raise ProgramValidationError(f"{name} has the wrong packed shape")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ProgramValidationError(f"{name} must use an integer dtype")


def _validate_total_slots(value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ProgramValidationError("total_segment_slots must be a positive integer")
