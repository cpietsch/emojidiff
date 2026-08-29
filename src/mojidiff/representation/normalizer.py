"""Deterministic semantic-stroke normalization for bounded OpenMoji SVGs."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from defusedxml import ElementTree
from picosvg.svg import SVG
from picosvg.svg_transform import Affine2D
from picosvg.svg_types import SVGShape

from mojidiff.representation.program import (
    VIEWBOX_SIZE,
    FloatContour,
    FloatProgram,
    FloatSegment,
    SegmentType,
)

_SAFE_TAGS = {
    "svg",
    "g",
    "defs",
    "title",
    "desc",
    "metadata",
    "path",
    "rect",
    "circle",
    "ellipse",
    "line",
    "polyline",
    "polygon",
}
_RESOURCE_TAGS = {
    "script",
    "image",
    "use",
    "foreignObject",
    "style",
    "filter",
    "mask",
    "pattern",
    "clipPath",
    "marker",
    "linearGradient",
    "radialGradient",
}
_RESOURCE_ATTRIBUTES = {
    "href",
    "src",
    "filter",
    "mask",
    "clip-path",
    "marker",
    "marker-start",
    "marker-mid",
    "marker-end",
}
_UNSUPPORTED_PRESENTATION = {
    "color",
    "paint-order",
    "shape-rendering",
    "stroke-alignment",
    "visibility",
}
_NUMBER_SPLIT = re.compile(r"[\s,]+")


class NormalizationError(ValueError):
    """A source SVG uses semantics not represented by the semantic codec."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class NormalizationReport:
    """Structural facts observed while normalizing one source SVG."""

    source_shapes: int
    painted_layers: int
    contours: int
    segments: int
    transformed_shapes: int
    compound_layers: int
    dashed_layers: int
    partially_opaque_layers: int
    out_of_bounds_coordinates: int


@dataclass(frozen=True)
class NormalizedProgram:
    """A normalized program and its deterministic feature report."""

    program: FloatProgram
    report: NormalizationReport


def normalize_svg(svg_bytes: bytes) -> NormalizedProgram:
    """Flatten safe source geometry while retaining categorical paint semantics."""

    _screen_source(svg_bytes)
    try:
        svg = SVG.fromstring(svg_bytes)  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise NormalizationError("parse_error", str(exc)[:300]) from exc

    contours: list[FloatContour] = []
    source_shapes = 0
    transformed_shapes = 0
    dashed_layers = 0
    partially_opaque_layers = 0
    layer = 1
    try:
        contexts = svg.depth_first(resolve_clip_paths=False)
        for context in contexts:
            if not context.is_shape():
                continue
            source_shapes += 1
            shape = context.shape().apply_style_attribute()
            _validate_shape_style(shape)
            if shape.style:
                raise NormalizationError(
                    "unsupported_style", "style contains unsupported CSS declarations"
                )
            if shape.display == "none" or not shape.might_paint():
                continue

            fill = _canonical_color(shape.fill, "fill")
            stroke = _canonical_color(shape.stroke, "stroke")
            opacity = _opacity(float(shape.opacity), "opacity")
            fill_opacity = _opacity(float(shape.fill_opacity), "fill-opacity")
            stroke_opacity = _opacity(float(shape.stroke_opacity), "stroke-opacity")
            if opacity == 0:
                continue
            if fill_opacity == 0:
                fill = None
            encoded_fill_opacity = None if fill is None else fill_opacity
            if stroke_opacity == 0 or float(shape.stroke_width) == 0:
                stroke = None
            encoded_stroke_opacity = None if stroke is None else stroke_opacity
            if fill is None and stroke is None:
                continue

            transform = context.transform
            scale = _transform_scale(transform, stroke is not None)
            if not transform.almost_equals(Affine2D.identity(), 1e-12):
                transformed_shapes += 1
            path = shape.apply_transform(transform)
            commands = tuple(path.as_cmd_seq())
            dash = _dash_pattern(shape.stroke_dasharray)
            if dash and stroke is not None:
                dashed_layers += 1
            if stroke is not None and float(shape.stroke_dashoffset) != 0:
                raise NormalizationError(
                    "unsupported_dash_offset", "nonzero stroke-dashoffset is not encoded"
                )
            shape_contours = _commands_to_contours(
                commands,
                layer=layer,
                fill=fill,
                stroke=stroke,
                stroke_width=float(shape.stroke_width) * scale,
                linecap=shape.stroke_linecap,
                linejoin=shape.stroke_linejoin,
                miter_limit=float(shape.stroke_miterlimit),
                dash_pattern=tuple(value * scale for value in dash),
                fill_rule=shape.fill_rule,
                opacity=opacity,
                fill_opacity=encoded_fill_opacity,
                stroke_opacity=encoded_stroke_opacity,
            )
            if shape_contours:
                contours.extend(shape_contours)
                if opacity < 1 or any(
                    value is not None and value < 1
                    for value in (encoded_fill_opacity, encoded_stroke_opacity)
                ):
                    partially_opaque_layers += 1
                layer += 1
    except NormalizationError:
        raise
    except Exception as exc:
        raise NormalizationError("geometry_error", str(exc)[:300]) from exc

    compound = len({item.layer for item in contours if _layer_count(contours, item.layer) > 1})
    coordinates = [
        value
        for contour in contours
        for value in (*contour.start, *(value for seg in contour.segments for value in seg.coords))
    ]
    report = NormalizationReport(
        source_shapes=source_shapes,
        painted_layers=layer - 1,
        contours=len(contours),
        segments=sum(len(contour.segments) for contour in contours),
        transformed_shapes=transformed_shapes,
        compound_layers=compound,
        dashed_layers=dashed_layers,
        partially_opaque_layers=partially_opaque_layers,
        out_of_bounds_coordinates=sum(value < 0 or value > VIEWBOX_SIZE for value in coordinates),
    )
    return NormalizedProgram(FloatProgram(tuple(contours)), report)


def _screen_source(svg_bytes: bytes) -> None:
    try:
        root = ElementTree.fromstring(svg_bytes)
    except Exception as exc:
        raise NormalizationError("parse_error", str(exc)[:300]) from exc
    if _tag(root.tag) != "svg":
        raise NormalizationError("invalid_root", "document root must be svg")
    view_box = root.attrib.get("viewBox", "").replace(",", " ").split()
    try:
        parsed_view_box = tuple(float(value) for value in view_box)
    except ValueError as exc:
        raise NormalizationError("invalid_viewbox", "viewBox contains non-numeric data") from exc
    if parsed_view_box != (0.0, 0.0, VIEWBOX_SIZE, VIEWBOX_SIZE):
        raise NormalizationError("invalid_viewbox", "viewBox must be exactly 0 0 72 72")

    for element in root.iter():
        tag = _tag(element.tag)
        if tag == "svg" and element is not root:
            raise NormalizationError(
                "nested_svg", "nested SVG viewports are outside the normalizer"
            )
        if tag in _RESOURCE_TAGS:
            raise NormalizationError("unsupported_resource", f"resource tag {tag} is forbidden")
        if tag not in _SAFE_TAGS:
            raise NormalizationError("unsupported_tag", f"tag {tag} is outside the allow-list")
        for raw_name, value in element.attrib.items():
            name = _tag(raw_name)
            lowered = value.lower()
            if name == "style":
                raise NormalizationError(
                    "unsupported_style", "inline CSS cascade is outside the normalizer"
                )
            if name in _UNSUPPORTED_PRESENTATION:
                raise NormalizationError(
                    "unsupported_presentation", f"attribute {name} is outside the codec"
                )
            if name.startswith("on") or name in _RESOURCE_ATTRIBUTES or "url(" in lowered:
                if name == "clip-path":
                    code = "unsupported_clip_path"
                else:
                    code = "unsupported_resource"
                raise NormalizationError(code, f"attribute {name} is outside the codec")
            if tag in {"svg", "g"} and name == "opacity":
                try:
                    group_opacity = float(value)
                except ValueError as exc:
                    raise NormalizationError(
                        "invalid_opacity", "group/root opacity must be numeric"
                    ) from exc
                if not math.isfinite(group_opacity) or group_opacity != 1:
                    raise NormalizationError(
                        "group_opacity", "group/root opacity cannot be flattened safely"
                    )
            if element is root and name == "transform" and value.strip():
                raise NormalizationError(
                    "root_transform", "root SVG transforms are outside the normalizer"
                )
            if name == "vector-effect" and value != "none":
                raise NormalizationError(
                    "unsupported_vector_effect", "non-scaling strokes are not encoded"
                )


def _tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _canonical_color(value: str, field: str) -> str | None:
    color = value.strip().lower()
    if color == "none":
        return None
    aliases = {"black": "#000000", "white": "#ffffff"}
    color = aliases.get(color, color)
    if re.fullmatch(r"#[0-9a-f]{3}", color):
        color = "#" + "".join(character * 2 for character in color[1:])
    if re.fullmatch(r"#[0-9a-f]{6}", color) is None:
        raise NormalizationError("unsupported_color", f"{field} color {value!r} is unsupported")
    return color


def _opacity(value: float, field: str) -> float:
    if not math.isfinite(value) or value < 0 or value > 1:
        raise NormalizationError("invalid_opacity", f"{field} must be finite within 0..1")
    return value


def _validate_shape_style(shape: SVGShape) -> None:
    width = float(shape.stroke_width)
    miter = float(shape.stroke_miterlimit)
    dash_offset = float(shape.stroke_dashoffset)
    if not math.isfinite(width) or width < 0:
        raise NormalizationError("invalid_stroke_width", "stroke width must be nonnegative")
    if not math.isfinite(miter) or miter <= 0:
        raise NormalizationError("invalid_miter_limit", "miter limit must be positive")
    if not math.isfinite(dash_offset):
        raise NormalizationError("invalid_dash_offset", "dash offset must be finite")
    if shape.stroke_linecap not in {"butt", "round", "square"}:
        raise NormalizationError("unsupported_linecap", "stroke linecap is unsupported")
    if shape.stroke_linejoin not in {"miter", "round", "bevel"}:
        raise NormalizationError("unsupported_linejoin", "stroke linejoin is unsupported")
    if shape.fill_rule not in {"nonzero", "evenodd"}:
        raise NormalizationError("unsupported_fill_rule", "fill rule is unsupported")


def _transform_scale(transform: Affine2D, painted_stroke: bool) -> float:
    values = tuple(float(value) for value in transform)
    if not all(math.isfinite(value) for value in values):
        raise NormalizationError("invalid_transform", "transform contains non-finite values")
    determinant = transform.determinant()
    if not math.isfinite(determinant) or abs(determinant) <= 1e-12:
        raise NormalizationError("singular_transform", "transform is singular")
    if not painted_stroke:
        return 1.0
    sx = math.hypot(transform.a, transform.b)
    sy = math.hypot(transform.c, transform.d)
    dot = transform.a * transform.c + transform.b * transform.d
    tolerance = 1e-7 * max(1.0, sx, sy, sx * sy)
    if abs(sx - sy) > tolerance or abs(dot) > tolerance:
        raise NormalizationError(
            "anisotropic_stroke_transform",
            "a scalar stroke width cannot preserve a non-similarity transform",
        )
    return (sx + sy) / 2


def _dash_pattern(value: str) -> tuple[float, ...]:
    if value.strip().lower() == "none":
        return ()
    try:
        pattern = tuple(float(item) for item in _NUMBER_SPLIT.split(value.strip()) if item)
    except ValueError as exc:
        raise NormalizationError("unsupported_dash", "dash array must be numeric") from exc
    if not pattern or not all(math.isfinite(item) and item >= 0 for item in pattern):
        raise NormalizationError("unsupported_dash", "dash array must be finite and nonnegative")
    if len(pattern) % 2:
        pattern += pattern
    if not any(pattern):
        return ()
    return pattern


def _commands_to_contours(
    commands: tuple[tuple[str, tuple[float, ...]], ...],
    *,
    layer: int,
    fill: str | None,
    stroke: str | None,
    stroke_width: float,
    linecap: str,
    linejoin: str,
    miter_limit: float,
    dash_pattern: tuple[float, ...],
    fill_rule: str,
    opacity: float,
    fill_opacity: float | None,
    stroke_opacity: float | None,
) -> list[FloatContour]:
    contours: list[FloatContour] = []
    start: tuple[float, float] | None = None
    subpath_start: tuple[float, float] | None = None
    segments: list[FloatSegment] = []

    def flush() -> None:
        nonlocal start, segments
        if start is not None and segments:
            contours.append(
                FloatContour(
                    layer=layer,
                    fill=fill,
                    stroke=stroke,
                    stroke_width=stroke_width,
                    linecap=linecap,
                    linejoin=linejoin,
                    miter_limit=miter_limit,
                    dash_pattern=dash_pattern,
                    fill_rule=fill_rule,
                    opacity=opacity,
                    fill_opacity=fill_opacity,
                    stroke_opacity=stroke_opacity,
                    start=start,
                    segments=tuple(segments),
                )
            )
        start = None
        segments = []

    for raw_command, raw_args in commands:
        command = raw_command.upper()
        args = tuple(float(value) for value in raw_args)
        if command == "M":
            if len(args) != 2:
                raise NormalizationError("invalid_geometry", "move must have two coordinates")
            flush()
            start = (args[0], args[1])
            subpath_start = start
            continue
        if start is None:
            raise NormalizationError("invalid_geometry", "geometry appears before a move")
        if segments and segments[-1].kind == SegmentType.CLOSE:
            flush()
            if subpath_start is None:
                raise NormalizationError("invalid_geometry", "close has no subpath start")
            start = subpath_start
        kind = {
            "L": SegmentType.LINE,
            "Q": SegmentType.QUAD,
            "C": SegmentType.CUBIC,
            "Z": SegmentType.CLOSE,
        }.get(command)
        if kind is None:
            raise NormalizationError("unsupported_command", f"command {raw_command} remains")
        expected = {
            SegmentType.LINE: 2,
            SegmentType.QUAD: 4,
            SegmentType.CUBIC: 6,
            SegmentType.CLOSE: 0,
        }[kind]
        if len(args) != expected:
            raise NormalizationError("invalid_geometry", f"{command} has wrong coordinate arity")
        segments.append(FloatSegment(kind, args))
    flush()
    return contours


def _layer_count(contours: list[FloatContour], layer: int) -> int:
    return sum(contour.layer == layer for contour in contours)
