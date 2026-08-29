"""Typed, bounded drawing programs and their deterministic safe serializer."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

VIEWBOX_SIZE = 72.0
PAD = 0
NONE = 1
_HEX_COLOR = re.compile(r"#[0-9a-f]{6}\Z")
_CAPS = ("butt", "round", "square")
_JOINS = ("miter", "round", "bevel")
_FILL_RULES = ("nonzero", "evenodd")
OPACITY_VOCABULARY = (0.25, 0.4, 0.5, 0.502, 0.6, 0.9969, 0.997, 0.999, 1.0)


class CodecError(ValueError):
    """A drawing program cannot be represented by the configured codec."""


class BudgetExceeded(CodecError):
    """A program exceeds a fixed path or segment budget in strict mode."""


class ProgramValidationError(CodecError):
    """A tensor program violates one or more hard grammar invariants."""


class SegmentType(IntEnum):
    """Legal segment tokens; zero is typed padding."""

    PAD = 0
    LINE = 1
    QUAD = 2
    CUBIC = 3
    CLOSE = 4


_COORDS_PER_SEGMENT = {
    SegmentType.LINE: 2,
    SegmentType.QUAD: 4,
    SegmentType.CUBIC: 6,
    SegmentType.CLOSE: 0,
}


@dataclass(frozen=True)
class CodecConfig:
    """Fixed vocabularies and tensor ceilings for one codec identity."""

    max_paths: int
    max_segments: int
    coordinate_bins: int
    palette: tuple[str, ...]
    stroke_widths: tuple[float, ...]
    dash_patterns: tuple[tuple[float, ...], ...]
    miter_limits: tuple[float, ...]
    opacities: tuple[float, ...]
    max_serialized_bytes: int = 2_000_000

    def __post_init__(self) -> None:
        for name, value in (
            ("max_paths", self.max_paths),
            ("max_segments", self.max_segments),
            ("max_serialized_bytes", self.max_serialized_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise CodecError(f"{name} must be a positive integer")
        if self.max_paths > 512 or self.max_segments > 1024:
            raise CodecError("path and segment ceilings exceed renderer-safe bounds")
        if self.max_paths * self.max_segments > 100_000:
            raise CodecError("combined path/segment ceiling exceeds renderer-safe bounds")
        if self.max_serialized_bytes > 4_000_000:
            raise CodecError("serialized byte ceiling exceeds the renderer-safe bound")
        if (
            isinstance(self.coordinate_bins, bool)
            or not isinstance(self.coordinate_bins, int)
            or self.coordinate_bins < 2
            or self.coordinate_bins > 65_536
        ):
            raise CodecError("coordinate_bins must be an integer of at least 2")
        if not self.palette or len(set(self.palette)) != len(self.palette):
            raise CodecError("palette must contain unique colors")
        if any(_HEX_COLOR.fullmatch(color) is None for color in self.palette):
            raise CodecError("palette entries must be lowercase #rrggbb colors")
        _validate_positive_vocabulary("stroke_widths", self.stroke_widths)
        _validate_positive_vocabulary("miter_limits", self.miter_limits)
        if (
            not self.opacities
            or len(set(self.opacities)) != len(self.opacities)
            or tuple(sorted(self.opacities)) != self.opacities
        ):
            raise CodecError("opacities must contain unique increasing values")
        if not all(math.isfinite(value) and 0 < value <= 1 for value in self.opacities):
            raise CodecError("opacities must contain finite values within (0, 1]")
        if 1.0 not in self.opacities:
            raise CodecError("opacities must contain the fully opaque value 1.0")
        if len(set(self.dash_patterns)) != len(self.dash_patterns):
            raise CodecError("dash_patterns must be unique")
        for pattern in self.dash_patterns:
            if not pattern or not all(math.isfinite(value) and value >= 0 for value in pattern):
                raise CodecError("dash patterns must contain finite nonnegative values")
            if not any(value > 0 for value in pattern):
                raise CodecError("dash patterns cannot be entirely zero")
            if len(pattern) % 2 or len(pattern) > 32:
                raise CodecError("dash patterns must be canonical even tuples of at most 32 values")


def _validate_positive_vocabulary(name: str, values: tuple[float, ...]) -> None:
    if not values or len(set(values)) != len(values):
        raise CodecError(f"{name} must contain unique values")
    if not all(math.isfinite(value) and 0 < value <= VIEWBOX_SIZE for value in values):
        raise CodecError(f"{name} must contain finite values within (0, 72]")


@dataclass(frozen=True)
class FloatSegment:
    """An absolute segment in the fixed 72×72 coordinate system."""

    kind: SegmentType
    coords: tuple[float, ...]


@dataclass(frozen=True)
class FloatContour:
    """One contour; equal adjacent layers form one compound SVG paint operation."""

    layer: int
    fill: str | None
    stroke: str | None
    stroke_width: float
    linecap: str
    linejoin: str
    miter_limit: float
    dash_pattern: tuple[float, ...]
    fill_rule: str
    opacity: float
    fill_opacity: float | None
    stroke_opacity: float | None
    start: tuple[float, float]
    segments: tuple[FloatSegment, ...]


@dataclass(frozen=True)
class FloatProgram:
    """Ordered float contours before categorical quantization."""

    contours: tuple[FloatContour, ...]


IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class TensorProgram:
    """Fixed-shape categorical program arrays."""

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


@dataclass(frozen=True)
class EncodingReport:
    """Explicit record of every lossy safety projection during encoding."""

    input_contours: int
    encoded_contours: int
    dropped_contours: int
    dropped_segments: int
    partial_layers: tuple[int, ...]
    clamped_coordinates: int
    approximated_stroke_widths: int
    max_stroke_width_error: float
    approximated_miter_limits: int
    max_miter_limit_error: float

    @property
    def lossless(self) -> bool:
        return not any(
            (
                self.dropped_contours,
                self.dropped_segments,
                self.clamped_coordinates,
                self.approximated_stroke_widths,
                self.approximated_miter_limits,
            )
        )


def quantize_coordinate(value: float, bins: int) -> int:
    """Quantize an in-range coordinate using a fixed round-half-up rule."""

    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise CodecError("coordinate bins must be an integer of at least 2")
    if not math.isfinite(value) or value < 0 or value > VIEWBOX_SIZE:
        raise CodecError("coordinate must be finite and within 0..72")
    return int(math.floor(value * (bins - 1) / VIEWBOX_SIZE + 0.5))


def dequantize_coordinate(token: int, bins: int) -> float:
    """Map a zero-based coordinate category back into the fixed viewBox."""

    if isinstance(bins, bool) or not isinstance(bins, int) or bins < 2:
        raise CodecError("coordinate bins must be an integer of at least 2")
    if isinstance(token, bool) or not isinstance(token, (int, np.integer)):
        raise CodecError("coordinate token must be an integer")
    if token < 0 or token >= bins:
        raise CodecError("coordinate token is outside the configured vocabulary")
    return VIEWBOX_SIZE * int(token) / (bins - 1)


def encode_program(
    program: FloatProgram,
    config: CodecConfig,
    *,
    allow_truncation: bool = False,
    allow_clamping: bool = False,
) -> tuple[TensorProgram, EncodingReport]:
    """Encode a float program, making every lossy operation explicit in the report."""

    _validate_float_program(program)
    arrays = _empty_tensor(config)
    retained = program.contours[: config.max_paths]
    dropped = program.contours[config.max_paths :]
    dropped_segments = sum(len(contour.segments) for contour in dropped)
    dropped_segments += sum(
        max(0, len(contour.segments) - config.max_segments) for contour in retained
    )
    if (dropped or dropped_segments) and not allow_truncation:
        raise BudgetExceeded(
            f"program needs {len(program.contours)} contours and up to "
            f"{max((len(item.segments) for item in program.contours), default=0)} segments"
        )

    kept_layers = {contour.layer for contour in retained}
    dropped_layers = {contour.layer for contour in dropped}
    partial_layers = tuple(sorted(kept_layers & dropped_layers))
    clamp_count = 0
    width_count = 0
    width_error = 0.0
    miter_count = 0
    miter_error = 0.0

    for path_index, contour in enumerate(retained):
        segments = contour.segments[: config.max_segments]
        arrays.path_length[path_index] = len(segments)
        arrays.layer[path_index] = contour.layer
        arrays.opacity[path_index] = _categorical(contour.opacity, config.opacities, "opacity")
        arrays.fill[path_index] = _categorical(contour.fill, config.palette, "fill")
        arrays.fill_opacity[path_index] = _categorical(
            contour.fill_opacity, config.opacities, "fill_opacity"
        )
        arrays.stroke[path_index] = _categorical(contour.stroke, config.palette, "stroke")
        arrays.stroke_opacity[path_index] = _categorical(
            contour.stroke_opacity, config.opacities, "stroke_opacity"
        )
        arrays.fill_rule[path_index] = (
            NONE if contour.fill is None else _FILL_RULES.index(contour.fill_rule) + 2
        )
        start_tokens, count = _encode_coords(contour.start, config, allow_clamping)
        arrays.start[path_index] = start_tokens
        clamp_count += count

        if contour.stroke is None:
            for field in (
                arrays.stroke_width,
                arrays.linecap,
                arrays.linejoin,
                arrays.miter_limit,
                arrays.dash_pattern,
            ):
                field[path_index] = NONE
        else:
            width_token, error = _nearest_token(contour.stroke_width, config.stroke_widths)
            arrays.stroke_width[path_index] = width_token
            width_count += int(error != 0)
            width_error = max(width_error, error)
            arrays.linecap[path_index] = _CAPS.index(contour.linecap) + 2
            arrays.linejoin[path_index] = _JOINS.index(contour.linejoin) + 2
            miter_token, error = _nearest_token(contour.miter_limit, config.miter_limits)
            arrays.miter_limit[path_index] = miter_token
            miter_count += int(error != 0)
            miter_error = max(miter_error, error)
            arrays.dash_pattern[path_index] = _categorical(
                contour.dash_pattern or None, config.dash_patterns, "dash_pattern"
            )

        for segment_index, segment in enumerate(segments):
            arrays.segment_type[path_index, segment_index] = int(segment.kind)
            encoded, count = _encode_coords(segment.coords, config, allow_clamping)
            arrays.coordinates[path_index, segment_index, : len(encoded)] = encoded
            clamp_count += count

    tensor = TensorProgram(**arrays.as_dict())
    validate_tensor_program(tensor, config)
    report = EncodingReport(
        input_contours=len(program.contours),
        encoded_contours=len(retained),
        dropped_contours=len(dropped),
        dropped_segments=dropped_segments,
        partial_layers=partial_layers,
        clamped_coordinates=clamp_count,
        approximated_stroke_widths=width_count,
        max_stroke_width_error=width_error,
        approximated_miter_limits=miter_count,
        max_miter_limit_error=miter_error,
    )
    return tensor, report


@dataclass
class _MutableTensor:
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

    def as_dict(self) -> dict[str, IntArray]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}


def _empty_tensor(config: CodecConfig) -> _MutableTensor:
    def vector() -> IntArray:
        return np.zeros(config.max_paths, dtype=np.int64)

    return _MutableTensor(
        path_length=vector(),
        layer=vector(),
        opacity=vector(),
        fill=vector(),
        fill_opacity=vector(),
        stroke=vector(),
        stroke_opacity=vector(),
        stroke_width=vector(),
        linecap=vector(),
        linejoin=vector(),
        miter_limit=vector(),
        dash_pattern=vector(),
        fill_rule=vector(),
        start=np.zeros((config.max_paths, 2), dtype=np.int64),
        segment_type=np.zeros((config.max_paths, config.max_segments), dtype=np.int64),
        coordinates=np.zeros((config.max_paths, config.max_segments, 6), dtype=np.int64),
    )


def _categorical(value: object | None, vocabulary: tuple[object, ...], field: str) -> int:
    if value is None:
        return NONE
    try:
        return vocabulary.index(value) + 2
    except ValueError as exc:
        raise CodecError(f"{field} value is outside the configured vocabulary: {value!r}") from exc


def _nearest_token(value: float, vocabulary: tuple[float, ...]) -> tuple[int, float]:
    index = min(range(len(vocabulary)), key=lambda item: (abs(vocabulary[item] - value), item))
    error = abs(vocabulary[index] - value)
    return index + 2, error


def _encode_coords(
    values: tuple[float, ...], config: CodecConfig, allow_clamping: bool
) -> tuple[IntArray, int]:
    encoded = np.zeros(len(values), dtype=np.int64)
    clamped = 0
    for index, raw in enumerate(values):
        if not math.isfinite(raw):
            raise CodecError("coordinate must be finite")
        value = raw
        if value < 0 or value > VIEWBOX_SIZE:
            if not allow_clamping:
                raise CodecError("coordinate is outside 0..72 and clamping is disabled")
            value = min(VIEWBOX_SIZE, max(0.0, value))
            clamped += 1
        encoded[index] = quantize_coordinate(value, config.coordinate_bins) + 1
    return encoded, clamped


def _validate_float_program(program: FloatProgram) -> None:
    previous_layer = 0
    for contour in program.contours:
        if contour.layer <= 0 or contour.layer < previous_layer:
            raise CodecError("float contour layers must be positive and nondecreasing")
        previous_layer = contour.layer
        if contour.fill is None and contour.stroke is None:
            raise CodecError("active contours must have a fill or stroke")
        for field, color in (("fill", contour.fill), ("stroke", contour.stroke)):
            if color is not None and _HEX_COLOR.fullmatch(color) is None:
                raise CodecError(f"{field} must be a lowercase #rrggbb color")
        _validate_opacity(contour.opacity, "opacity")
        if (contour.fill is None) != (contour.fill_opacity is None):
            raise CodecError("fill_opacity must be present exactly when fill is painted")
        if contour.fill_opacity is not None:
            _validate_opacity(contour.fill_opacity, "fill_opacity")
        if (contour.stroke is None) != (contour.stroke_opacity is None):
            raise CodecError("stroke_opacity must be present exactly when stroke is painted")
        if contour.stroke_opacity is not None:
            _validate_opacity(contour.stroke_opacity, "stroke_opacity")
        if contour.linecap not in _CAPS or contour.linejoin not in _JOINS:
            raise CodecError("unsupported line cap or join")
        if contour.fill_rule not in _FILL_RULES:
            raise CodecError("unsupported fill rule")
        if contour.stroke is not None:
            if not math.isfinite(contour.stroke_width) or contour.stroke_width <= 0:
                raise CodecError("painted stroke width must be finite and positive")
            if not math.isfinite(contour.miter_limit) or contour.miter_limit <= 0:
                raise CodecError("painted miter limit must be finite and positive")
            if not all(math.isfinite(value) and value >= 0 for value in contour.dash_pattern):
                raise CodecError("dash values must be finite and nonnegative")
        if len(contour.start) != 2 or not all(math.isfinite(value) for value in contour.start):
            raise CodecError("contour start must contain two finite coordinates")
        if not contour.segments:
            raise CodecError("active contours require at least one segment")
        for index, segment in enumerate(contour.segments):
            expected = _COORDS_PER_SEGMENT.get(segment.kind)
            if expected is None or len(segment.coords) != expected:
                raise CodecError("segment has an illegal coordinate arity")
            if not all(math.isfinite(value) for value in segment.coords):
                raise CodecError("segment coordinates must be finite")
            if segment.kind == SegmentType.CLOSE and index != len(contour.segments) - 1:
                raise CodecError("close must be the final segment in a contour")


def _validate_opacity(value: float, field: str) -> None:
    if not math.isfinite(value) or value <= 0 or value > 1:
        raise CodecError(f"{field} must be finite within (0, 1]")


def validate_tensor_program(program: TensorProgram, config: CodecConfig) -> None:
    """Raise with the first hard grammar violation in a fixed tensor program."""

    vector_names = (
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
    for name in vector_names:
        _require_int_array(getattr(program, name), (config.max_paths,), name)
    _require_int_array(program.start, (config.max_paths, 2), "start")
    _require_int_array(
        program.segment_type, (config.max_paths, config.max_segments), "segment_type"
    )
    _require_int_array(
        program.coordinates,
        (config.max_paths, config.max_segments, 6),
        "coordinates",
    )

    previous_layer = 0
    inactive_seen = False
    layer_styles: dict[int, tuple[int, ...]] = {}
    for path_index in range(config.max_paths):
        length = int(program.path_length[path_index])
        if length < 0 or length > config.max_segments:
            raise ProgramValidationError("path_length is outside the segment budget")
        if length == 0:
            inactive_seen = True
            fields = (
                program.layer[path_index],
                program.opacity[path_index],
                program.fill[path_index],
                program.fill_opacity[path_index],
                program.stroke[path_index],
                program.stroke_opacity[path_index],
                program.stroke_width[path_index],
                program.linecap[path_index],
                program.linejoin[path_index],
                program.miter_limit[path_index],
                program.dash_pattern[path_index],
                program.fill_rule[path_index],
            )
            if any(int(value) != PAD for value in fields):
                raise ProgramValidationError("inactive path style fields must be PAD")
            if np.any(program.start[path_index]) or np.any(program.segment_type[path_index]):
                raise ProgramValidationError("inactive path geometry must be PAD")
            if np.any(program.coordinates[path_index]):
                raise ProgramValidationError("inactive path coordinates must be PAD")
            continue

        if inactive_seen:
            raise ProgramValidationError("active paths must form a packed prefix")

        layer = int(program.layer[path_index])
        if layer <= 0 or layer > config.max_paths or layer < previous_layer:
            raise ProgramValidationError("active layers must be in range and nondecreasing")
        previous_layer = layer
        _token_range(int(program.opacity[path_index]), len(config.opacities) + 1, "opacity")
        fill = int(program.fill[path_index])
        stroke = int(program.stroke[path_index])
        _token_range(fill, len(config.palette) + 1, "fill", allow_none=True)
        _token_range(stroke, len(config.palette) + 1, "stroke", allow_none=True)
        if fill == NONE and stroke == NONE:
            raise ProgramValidationError("active path must paint fill or stroke")
        fill_rule = int(program.fill_rule[path_index])
        if fill == NONE:
            if fill_rule != NONE or int(program.fill_opacity[path_index]) != NONE:
                raise ProgramValidationError(
                    "fill-none path must use NONE fill_rule and fill_opacity"
                )
        else:
            _token_range(fill_rule, len(_FILL_RULES) + 1, "fill_rule")
            _token_range(
                int(program.fill_opacity[path_index]),
                len(config.opacities) + 1,
                "fill_opacity",
            )
        _coordinate_tokens(program.start[path_index], config)

        if stroke == NONE:
            if any(
                int(field[path_index]) != NONE
                for field in (
                    program.stroke_opacity,
                    program.stroke_width,
                    program.linecap,
                    program.linejoin,
                    program.miter_limit,
                    program.dash_pattern,
                )
            ):
                raise ProgramValidationError("stroke-none style fields must be NONE")
        else:
            _token_range(
                int(program.stroke_opacity[path_index]),
                len(config.opacities) + 1,
                "stroke_opacity",
            )
            _token_range(
                int(program.stroke_width[path_index]),
                len(config.stroke_widths) + 1,
                "stroke_width",
            )
            _token_range(int(program.linecap[path_index]), len(_CAPS) + 1, "linecap")
            _token_range(int(program.linejoin[path_index]), len(_JOINS) + 1, "linejoin")
            _token_range(
                int(program.miter_limit[path_index]),
                len(config.miter_limits) + 1,
                "miter_limit",
            )
            _token_range(
                int(program.dash_pattern[path_index]),
                len(config.dash_patterns) + 1,
                "dash_pattern",
                allow_none=True,
            )

        style = tuple(
            int(field[path_index])
            for field in (
                program.opacity,
                program.fill,
                program.fill_opacity,
                program.stroke,
                program.stroke_opacity,
                program.stroke_width,
                program.linecap,
                program.linejoin,
                program.miter_limit,
                program.dash_pattern,
                program.fill_rule,
            )
        )
        if layer in layer_styles and layer_styles[layer] != style:
            raise ProgramValidationError("contours in one layer must share style")
        layer_styles[layer] = style

        for segment_index in range(config.max_segments):
            raw_kind = int(program.segment_type[path_index, segment_index])
            coords = program.coordinates[path_index, segment_index]
            if segment_index >= length:
                if raw_kind != PAD or np.any(coords):
                    raise ProgramValidationError("segments after path_length must be PAD")
                continue
            try:
                kind = SegmentType(raw_kind)
            except ValueError as exc:
                raise ProgramValidationError("unknown segment type") from exc
            if kind == SegmentType.PAD:
                raise ProgramValidationError("active segment cannot be PAD")
            if kind == SegmentType.CLOSE and segment_index != length - 1:
                raise ProgramValidationError("close cannot precede active geometry")
            active_coords = _COORDS_PER_SEGMENT[kind]
            _coordinate_tokens(coords[:active_coords], config)
            if np.any(coords[active_coords:]):
                raise ProgramValidationError("unused segment coordinates must be PAD")


def _require_int_array(array: IntArray, shape: tuple[int, ...], name: str) -> None:
    if not isinstance(array, np.ndarray) or array.shape != shape:
        raise ProgramValidationError(f"{name} has the wrong shape")
    if not np.issubdtype(array.dtype, np.integer) or np.issubdtype(array.dtype, np.bool_):
        raise ProgramValidationError(f"{name} must use an integer dtype")


def _token_range(token: int, maximum: int, field: str, allow_none: bool = False) -> None:
    minimum = NONE if allow_none else 2
    if token < minimum or token > maximum:
        raise ProgramValidationError(f"{field} token is outside its vocabulary")


def _coordinate_tokens(tokens: IntArray, config: CodecConfig) -> None:
    if np.any(tokens < 1) or np.any(tokens > config.coordinate_bins):
        raise ProgramValidationError("active coordinate token is outside its vocabulary")


def decode_program(program: TensorProgram, config: CodecConfig) -> FloatProgram:
    """Decode validated categorical arrays to quantized float contours."""

    validate_tensor_program(program, config)
    contours: list[FloatContour] = []
    for path_index in range(config.max_paths):
        length = int(program.path_length[path_index])
        if length == 0:
            continue
        stroke_token = int(program.stroke[path_index])
        stroke = _decode_categorical(stroke_token, config.palette)
        segments: list[FloatSegment] = []
        for segment_index in range(length):
            kind = SegmentType(int(program.segment_type[path_index, segment_index]))
            count = _COORDS_PER_SEGMENT[kind]
            coords = tuple(
                dequantize_coordinate(int(token) - 1, config.coordinate_bins)
                for token in program.coordinates[path_index, segment_index, :count]
            )
            segments.append(FloatSegment(kind, coords))
        contours.append(
            FloatContour(
                layer=int(program.layer[path_index]),
                opacity=config.opacities[int(program.opacity[path_index]) - 2],
                fill=_decode_categorical(int(program.fill[path_index]), config.palette),
                fill_opacity=(
                    None
                    if int(program.fill_opacity[path_index]) == NONE
                    else config.opacities[int(program.fill_opacity[path_index]) - 2]
                ),
                stroke=stroke,
                stroke_opacity=(
                    None
                    if int(program.stroke_opacity[path_index]) == NONE
                    else config.opacities[int(program.stroke_opacity[path_index]) - 2]
                ),
                stroke_width=(
                    0.0
                    if stroke is None
                    else config.stroke_widths[int(program.stroke_width[path_index]) - 2]
                ),
                linecap=("butt" if stroke is None else _CAPS[int(program.linecap[path_index]) - 2]),
                linejoin=(
                    "miter" if stroke is None else _JOINS[int(program.linejoin[path_index]) - 2]
                ),
                miter_limit=(
                    4.0
                    if stroke is None
                    else config.miter_limits[int(program.miter_limit[path_index]) - 2]
                ),
                dash_pattern=(
                    ()
                    if stroke is None or int(program.dash_pattern[path_index]) == NONE
                    else config.dash_patterns[int(program.dash_pattern[path_index]) - 2]
                ),
                fill_rule=(
                    "nonzero"
                    if int(program.fill[path_index]) == NONE
                    else _FILL_RULES[int(program.fill_rule[path_index]) - 2]
                ),
                start=(
                    dequantize_coordinate(
                        int(program.start[path_index, 0]) - 1, config.coordinate_bins
                    ),
                    dequantize_coordinate(
                        int(program.start[path_index, 1]) - 1, config.coordinate_bins
                    ),
                ),
                segments=tuple(segments),
            )
        )
    return FloatProgram(tuple(contours))


def _decode_categorical(token: int, vocabulary: tuple[str, ...]) -> str | None:
    return None if token == NONE else vocabulary[token - 2]


def serialize_svg(program: TensorProgram, config: CodecConfig) -> bytes:
    """Serialize only fixed safe SVG syntax from a validated tensor program."""

    decoded = decode_program(program, config)
    return serialize_float_svg(decoded, max_serialized_bytes=config.max_serialized_bytes)


def serialize_float_svg(program: FloatProgram, *, max_serialized_bytes: int) -> bytes:
    """Serialize validated float geometry for controlled codec analyses.

    Unlike ``serialize_svg``, this analysis-only path does not impose the categorical
    coordinate vocabulary. It is useful for rendering a counterfactual with finite
    Bezier control handles outside the viewBox; model-exposed states must still use the
    validated tensor serializer above.
    """

    if (
        isinstance(max_serialized_bytes, bool)
        or not isinstance(max_serialized_bytes, int)
        or max_serialized_bytes <= 0
        or max_serialized_bytes > 4_000_000
    ):
        raise CodecError("serialized byte ceiling must be within 1..4000000")
    _validate_float_program(program)
    layers: list[list[FloatContour]] = []
    for contour in program.contours:
        if not layers or layers[-1][0].layer != contour.layer:
            layers.append([contour])
        else:
            layers[-1].append(contour)

    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">']
    for contours in layers:
        first = contours[0]
        path_data = " ".join(_contour_path_data(contour) for contour in contours)
        attributes = [f'd="{path_data}"', f'fill="{first.fill or "none"}"']
        if first.fill_opacity is not None and first.fill_opacity != 1.0:
            attributes.append(f'fill-opacity="{_number(first.fill_opacity)}"')
        if first.fill_rule != "nonzero":
            attributes.append(f'fill-rule="{first.fill_rule}"')
        attributes.append(f'stroke="{first.stroke or "none"}"')
        if first.stroke is not None:
            if first.stroke_opacity is not None and first.stroke_opacity != 1.0:
                attributes.append(f'stroke-opacity="{_number(first.stroke_opacity)}"')
            attributes.extend(
                (
                    f'stroke-width="{_number(first.stroke_width)}"',
                    f'stroke-linecap="{first.linecap}"',
                    f'stroke-linejoin="{first.linejoin}"',
                    f'stroke-miterlimit="{_number(first.miter_limit)}"',
                )
            )
            if first.dash_pattern:
                dash = " ".join(_number(value) for value in first.dash_pattern)
                attributes.append(f'stroke-dasharray="{dash}"')
        if first.opacity != 1.0:
            attributes.append(f'opacity="{_number(first.opacity)}"')
        parts.append(f"<path {' '.join(attributes)}/>")
    parts.append("</svg>")
    output = "".join(parts).encode("utf-8")
    if len(output) > max_serialized_bytes:
        raise CodecError("serialized SVG exceeds the configured byte ceiling")
    return output


def _contour_path_data(contour: FloatContour) -> str:
    parts = ["M", _number(contour.start[0]), _number(contour.start[1])]
    for segment in contour.segments:
        command = {
            SegmentType.LINE: "L",
            SegmentType.QUAD: "Q",
            SegmentType.CUBIC: "C",
            SegmentType.CLOSE: "Z",
        }[segment.kind]
        parts.append(command)
        parts.extend(_number(value) for value in segment.coords)
    return " ".join(parts)


def _number(value: float) -> str:
    if value == 0:
        return "0"
    return repr(float(value))
