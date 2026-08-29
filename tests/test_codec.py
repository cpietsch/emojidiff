from __future__ import annotations

import copy
from collections.abc import Callable
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from mojidiff.representation.codec_study import load_codec_study_config
from mojidiff.representation.normalizer import NormalizationError, normalize_svg
from mojidiff.representation.program import (
    NONE,
    OPACITY_VOCABULARY,
    PAD,
    VIEWBOX_SIZE,
    BudgetExceeded,
    CodecConfig,
    CodecError,
    FloatContour,
    FloatProgram,
    FloatSegment,
    ProgramValidationError,
    SegmentType,
    TensorProgram,
    decode_program,
    dequantize_coordinate,
    encode_program,
    quantize_coordinate,
    serialize_svg,
    validate_tensor_program,
)


def _config(
    *,
    max_paths: int = 4,
    max_segments: int = 8,
    coordinate_bins: int = 128,
) -> CodecConfig:
    return CodecConfig(
        max_paths=max_paths,
        max_segments=max_segments,
        coordinate_bins=coordinate_bins,
        palette=("#000000", "#ffffff", "#ff0000"),
        stroke_widths=(1.0, 2.0),
        dash_patterns=((2.0, 2.0), (2.0, 3.0, 4.0, 2.0, 3.0, 4.0)),
        miter_limits=(4.0, 10.0),
        opacities=OPACITY_VOCABULARY,
        max_serialized_bytes=20_000,
    )


def test_aligned_coordinate_probe_has_exact_half_and_quarter_unit_lattices() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_codec_study_config(root / "configs/codec/aligned-coordinate-probe-v1.yaml")

    assert config.coordinate_bins == (145, 289)
    assert dequantize_coordinate(8, 145) == 4.0
    assert dequantize_coordinate(16, 289) == 4.0


def _contour(
    *,
    layer: int = 1,
    start: tuple[float, float] = (1.0, 2.0),
    segments: tuple[FloatSegment, ...] = (
        FloatSegment(SegmentType.LINE, (10.0, 20.0)),
        FloatSegment(SegmentType.CLOSE, ()),
    ),
    fill: str | None = "#ffffff",
    stroke: str | None = None,
) -> FloatContour:
    return FloatContour(
        layer=layer,
        fill=fill,
        stroke=stroke,
        stroke_width=2.0,
        linecap="round",
        linejoin="round",
        miter_limit=4.0,
        dash_pattern=(),
        fill_rule="nonzero",
        opacity=1.0,
        fill_opacity=None if fill is None else 1.0,
        stroke_opacity=None if stroke is None else 1.0,
        start=start,
        segments=segments,
    )


@pytest.mark.parametrize("bins", (128, 256))
def test_coordinate_quantizer_roundtrips_every_token_and_bounds_error(bins: int) -> None:
    step = VIEWBOX_SIZE / (bins - 1)

    assert quantize_coordinate(0.0, bins) == 0
    assert quantize_coordinate(VIEWBOX_SIZE, bins) == bins - 1
    for token in range(bins):
        assert quantize_coordinate(dequantize_coordinate(token, bins), bins) == token
    for token in range(bins - 1):
        midpoint = (token + 0.5) * VIEWBOX_SIZE / (bins - 1)
        assert quantize_coordinate(midpoint, bins) == token + 1

    samples = (index * step / 4 for index in range(4 * (bins - 1) + 1))
    max_error = max(
        abs(dequantize_coordinate(quantize_coordinate(value, bins), bins) - value)
        for value in samples
    )
    assert max_error <= step / 2 + 1e-12


def test_encode_validate_and_inactive_padding_are_canonical() -> None:
    config = _config()
    tensor, report = encode_program(FloatProgram((_contour(),)), config)

    assert report.lossless
    assert tensor.path_length.tolist() == [2, 0, 0, 0]
    assert tensor.segment_type[0, :2].tolist() == [SegmentType.LINE, SegmentType.CLOSE]
    assert np.all(tensor.segment_type[0, 2:] == PAD)
    assert np.all(tensor.coordinates[0, 0, :2] > PAD)
    assert np.all(tensor.coordinates[0, 0, 2:] == PAD)
    assert np.all(tensor.coordinates[0, 1:] == PAD)
    assert tensor.stroke[0] == NONE
    for field in (
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
    ):
        assert np.all(getattr(tensor, field)[1:] == PAD), field
    assert np.all(tensor.start[1:] == PAD)
    assert np.all(tensor.segment_type[1:] == PAD)
    assert np.all(tensor.coordinates[1:] == PAD)
    validate_tensor_program(tensor, config)


def _activate_unused_line_coordinate(tensor: TensorProgram) -> None:
    tensor.coordinates[0, 0, 2] = 1


def _set_unknown_segment_type(tensor: TensorProgram) -> None:
    tensor.segment_type[0, 0] = 99


@pytest.mark.parametrize(
    "mutate",
    (_activate_unused_line_coordinate, _set_unknown_segment_type),
)
def test_validator_rejects_mutated_coordinate_or_segment_type(
    mutate: Callable[[TensorProgram], None],
) -> None:
    config = _config()
    encoded, _report = encode_program(FloatProgram((_contour(),)), config)
    tensor = copy.deepcopy(encoded)
    mutate(tensor)

    with pytest.raises(ProgramValidationError):
        validate_tensor_program(tensor, config)


def test_validator_rejects_noncanonical_inactive_hole() -> None:
    config = _config()
    encoded, _report = encode_program(FloatProgram((_contour(),)), config)
    tensor = copy.deepcopy(encoded)
    for field in tensor.__dataclass_fields__:
        array = getattr(tensor, field)
        array[1] = array[0]
        array[0] = 0

    with pytest.raises(ProgramValidationError, match="packed prefix"):
        validate_tensor_program(tensor, config)


def test_strict_overflow_fails_and_truncation_report_is_explicit() -> None:
    config = _config(max_paths=2, max_segments=1)
    program = FloatProgram((_contour(), _contour(), _contour()))

    with pytest.raises(BudgetExceeded):
        encode_program(program, config)

    tensor, report = encode_program(program, config, allow_truncation=True)

    assert tensor.path_length.tolist() == [1, 1]
    assert report.input_contours == 3
    assert report.encoded_contours == 2
    assert report.dropped_contours == 1
    assert report.dropped_segments == 4
    assert report.partial_layers == (1,)
    assert not report.lossless
    validate_tensor_program(tensor, config)


def test_multi_move_contours_recombine_into_one_compound_path() -> None:
    source = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">
    <path fill="#fff" d="M1 1L10 1L10 10Z M20 20L30 20L30 30Z"/>
    </svg>"""
    normalized = normalize_svg(source)

    assert normalized.report.contours == 2
    assert normalized.report.compound_layers == 1
    assert {contour.layer for contour in normalized.program.contours} == {1}
    config = _config()
    tensor, report = encode_program(normalized.program, config)
    assert report.lossless

    root = ElementTree.fromstring(serialize_svg(tensor, config))
    paths = list(root)
    assert len(paths) == 1
    assert paths[0].attrib["d"].count("M") == 2


def test_serializer_is_deterministic_and_uses_only_allowed_xml() -> None:
    source = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">
    <path id="operator-id" class="operator-class" data-note="operator-note"
      fill="#fff" fill-rule="evenodd" d="M1 1L10 1L10 10Z"/>
    </svg>"""
    config = _config()
    tensor, _report = encode_program(normalize_svg(source).program, config)

    first = serialize_svg(tensor, config)
    second = serialize_svg(tensor, config)

    assert first == second
    assert len(first) <= config.max_serialized_bytes
    root = ElementTree.fromstring(first)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert root.attrib == {"viewBox": "0 0 72 72"}
    allowed = {
        "d",
        "fill",
        "fill-opacity",
        "fill-rule",
        "opacity",
        "stroke",
        "stroke-opacity",
        "stroke-width",
        "stroke-linecap",
        "stroke-linejoin",
        "stroke-miterlimit",
        "stroke-dasharray",
    }
    assert list(root)
    assert all(child.tag == "{http://www.w3.org/2000/svg}path" for child in root)
    assert all(set(child.attrib) <= allowed for child in root)
    lowered = first.lower()
    for forbidden in (
        b"<script",
        b"<style",
        b"href=",
        b"src=",
        b"onload=",
        b"url(",
        b"id=",
        b"class=",
        b"data-note=",
        b"transform=",
    ):
        assert forbidden not in lowered


def test_normalizer_flattens_primitives_and_inherited_styles() -> None:
    source = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">
    <g fill="#FFF" fill-rule="evenodd" stroke="black" stroke-width="2"
       stroke-linecap="round" stroke-linejoin="bevel" stroke-miterlimit="4"
       stroke-dasharray="2 3 4">
      <rect x="1" y="2" width="10" height="8" fill="#f00"/>
      <circle cx="30" cy="20" r="5" fill="none"/>
      <line x1="40" y1="10" x2="50" y2="20" fill="none"/>
    </g></svg>"""
    normalized = normalize_svg(source)

    assert normalized.report.source_shapes == 3
    assert normalized.report.painted_layers == 3
    assert normalized.report.dashed_layers == 3
    assert len(normalized.program.contours) == 3
    assert normalized.program.contours[0].fill == "#ff0000"
    assert [contour.stroke for contour in normalized.program.contours] == ["#000000"] * 3
    for contour in normalized.program.contours:
        assert contour.stroke_width == 2.0
        assert contour.linecap == "round"
        assert contour.linejoin == "bevel"
        assert contour.fill_rule == "evenodd"
        assert contour.opacity == 1.0
        assert contour.fill_opacity == (None if contour.fill is None else 1.0)
        assert contour.stroke_opacity == 1.0
        assert contour.dash_pattern == (2.0, 3.0, 4.0, 2.0, 3.0, 4.0)
        assert all(segment.kind != SegmentType.PAD for segment in contour.segments)
        if SegmentType.CLOSE in {segment.kind for segment in contour.segments}:
            assert contour.segments[-1].kind == SegmentType.CLOSE


def test_normalizer_flattens_nested_transforms() -> None:
    source = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">
    <g transform="translate(10 5)">
      <path transform="scale(2)" fill="#fff" d="M1 2L3 4L1 4Z"/>
    </g></svg>"""
    normalized = normalize_svg(source)

    assert normalized.report.transformed_shapes == 1
    contour = normalized.program.contours[0]
    assert contour.start == pytest.approx((12.0, 9.0))
    assert contour.segments[0] == FloatSegment(SegmentType.LINE, (16.0, 13.0))
    assert contour.segments[1] == FloatSegment(SegmentType.LINE, (12.0, 13.0))
    assert contour.segments[-1].kind == SegmentType.CLOSE


def test_normalizer_converts_arcs_to_supported_cubics() -> None:
    source = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">
    <path d="M10 36A26 26 0 0 1 62 36" fill="none" stroke="#000" stroke-width="2"/>
    </svg>"""
    normalized = normalize_svg(source)

    contour = normalized.program.contours[0]
    assert contour.start == (10.0, 36.0)
    assert len(contour.segments) == 2
    assert {segment.kind for segment in contour.segments} == {SegmentType.CUBIC}
    assert contour.segments[-1].coords[-2:] == (62.0, 36.0)
    tensor, _report = encode_program(normalized.program, _config())
    assert tensor.fill[0] == NONE
    assert tensor.fill_rule[0] == NONE
    validate_tensor_program(tensor, _config())


@pytest.mark.parametrize(
    ("shape", "code"),
    (
        (
            '<path transform="scale(2 1)" d="M1 1L5 5" fill="none" '
            'stroke="#000" stroke-width="2"/>',
            "anisotropic_stroke_transform",
        ),
    ),
)
def test_normalizer_classifies_unrepresentable_semantics(shape: str, code: str) -> None:
    source = (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">{shape}</svg>').encode()

    with pytest.raises(NormalizationError) as raised:
        normalize_svg(source)

    assert raised.value.code == code


@pytest.mark.parametrize(
    ("attribute", "code"),
    (
        ('transform="translate(1 1)"', "root_transform"),
        ('opacity=".5"', "group_opacity"),
        ('visibility="hidden"', "unsupported_presentation"),
        ('style="fill:#fff"', "unsupported_style"),
    ),
)
def test_normalizer_rejects_silently_lossy_root_semantics(attribute: str, code: str) -> None:
    source = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72" {attribute}>'
        '<path fill="#fff" d="M1 1L2 2Z"/></svg>'
    ).encode()

    with pytest.raises(NormalizationError) as raised:
        normalize_svg(source)

    assert raised.value.code == code


def test_partial_opacity_roundtrips_as_typed_compound_path_style() -> None:
    source = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">
    <path fill="#fff" fill-opacity=".6" stroke="#000" stroke-opacity=".4"
      stroke-width="2" opacity=".5"
      d="M1 1L10 1L10 10Z M20 20L30 20L30 30Z"/>
    </svg>"""
    normalized = normalize_svg(source)

    assert normalized.report.partially_opaque_layers == 1
    assert len(normalized.program.contours) == 2
    for contour in normalized.program.contours:
        assert contour.opacity == 0.5
        assert contour.fill_opacity == 0.6
        assert contour.stroke_opacity == 0.4

    config = _config()
    tensor, report = encode_program(normalized.program, config)
    assert report.lossless
    assert np.all(tensor.opacity[:2] == OPACITY_VOCABULARY.index(0.5) + 2)
    assert np.all(tensor.fill_opacity[:2] == OPACITY_VOCABULARY.index(0.6) + 2)
    assert np.all(tensor.stroke_opacity[:2] == OPACITY_VOCABULARY.index(0.4) + 2)
    decoded = decode_program(tensor, config)
    assert decoded.contours[0].opacity == 0.5
    assert decoded.contours[0].fill_opacity == 0.6
    assert decoded.contours[0].stroke_opacity == 0.4

    root = ElementTree.fromstring(serialize_svg(tensor, config))
    assert len(root) == 1
    assert root[0].attrib["opacity"] == "0.5"
    assert root[0].attrib["fill-opacity"] == "0.6"
    assert root[0].attrib["stroke-opacity"] == "0.4"
    assert root[0].attrib["d"].count("M") == 2


def test_zero_paint_opacity_removes_only_that_paint() -> None:
    source = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">
    <rect x="1" y="1" width="10" height="10" fill="#fff" fill-opacity="0"
      stroke="#000" stroke-width="2"/>
    </svg>"""
    contour = normalize_svg(source).program.contours[0]

    assert contour.fill is None
    assert contour.fill_opacity is None
    assert contour.stroke == "#000000"
    assert contour.stroke_opacity == 1.0
    tensor, report = encode_program(FloatProgram((contour,)), _config())
    assert report.lossless
    assert tensor.fill[0] == NONE
    assert tensor.fill_opacity[0] == NONE


def test_encoder_rejects_opacity_outside_configured_vocabulary() -> None:
    source = b"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">
    <rect x="1" y="1" width="10" height="10" fill="#fff" fill-opacity=".3"/>
    </svg>"""
    program = normalize_svg(source).program

    with pytest.raises(CodecError, match="fill_opacity value is outside"):
        encode_program(program, _config())


def test_validator_rejects_noncanonical_opacity_none_token() -> None:
    config = _config()
    encoded, _report = encode_program(FloatProgram((_contour(),)), config)
    tensor = copy.deepcopy(encoded)
    tensor.fill_opacity[0] = NONE

    with pytest.raises(ProgramValidationError, match="fill_opacity"):
        validate_tensor_program(tensor, config)
