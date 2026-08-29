from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest

from mojidiff.representation.capacity_study import (
    DenseCapacity,
    PackedCapacity,
    dense_retention,
    load_capacity_study_config,
    packed_retention,
)
from mojidiff.representation.codec_study import (
    CodecStudyError,
    _coordinate_excursions,
    _quantize_unclamped,
    _write_bytes_artifact,
    load_codec_study_config,
)
from mojidiff.representation.normalizer import NormalizationError, normalize_svg
from mojidiff.representation.packed import (
    pack_tensor_program,
    serialize_packed_svg,
    unpack_tensor_program,
    validate_packed_tensor_program,
)
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
    dequantize_bounded_coordinate,
    dequantize_coordinate,
    encode_program,
    quantize_bounded_coordinate,
    quantize_coordinate,
    serialize_float_svg,
    serialize_svg,
    validate_tensor_program,
)
from mojidiff.representation.renderer import (
    IsolatedRenderError,
    RenderLimits,
    render_typed_svg_isolated,
    validate_typed_svg,
)
from mojidiff.representation.style_study import (
    load_style_study_config,
    optimal_relative_l1_vocabulary,
)


def _config(
    *,
    max_paths: int = 4,
    max_segments: int = 8,
    coordinate_bins: int = 128,
    control_coordinate_bins: int | None = None,
    control_coordinate_min: float = 0.0,
    control_coordinate_max: float = VIEWBOX_SIZE,
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
        control_coordinate_bins=control_coordinate_bins,
        control_coordinate_min=control_coordinate_min,
        control_coordinate_max=control_coordinate_max,
    )


def test_aligned_coordinate_probe_has_exact_half_and_quarter_unit_lattices() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_codec_study_config(root / "configs/codec/aligned-coordinate-probe-v1.yaml")

    assert config.coordinate_bins == (145, 289)
    assert config.opacities == (1.0,)
    assert config.allow_truncation
    assert config.allow_clamping
    assert dequantize_coordinate(8, 145) == 4.0
    assert dequantize_coordinate(16, 289) == 4.0


def test_opacity_recovery_probe_has_explicit_schema_v2_projection() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_codec_study_config(root / "configs/codec/opacity-recovery-v1.yaml")

    assert config.schema_version == 2
    assert config.coordinate_bins == (289,)
    assert config.opacities == OPACITY_VOCABULARY
    assert not config.allow_truncation
    assert config.allow_clamping


def test_oob_control_probe_is_pinned_and_analysis_only() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_codec_study_config(root / "configs/codec/oob-control-probe-v1.yaml")

    assert config.schema_version == 2
    assert config.coordinate_bins == (289,)
    assert config.budgets[0].path_slots == 48
    assert config.budgets[0].segments_per_path == 64
    assert not config.allow_truncation
    assert config.allow_clamping
    assert config.compare_unclamped_coordinates

    fixture = json.loads(config.fixture_manifest.read_text(encoding="utf-8"))
    selected = {
        row["source_path"]: row["parent_out_of_bounds_coordinates"]
        for row in fixture["rows"]
    }
    parent = [
        json.loads(line)
        for line in (root / fixture["selection"]["parent_hybrid"]).read_text().splitlines()
    ]
    expected = {
        row["source_path"]: row["out_of_bounds_coordinates"]
        for row in parent
        if row["out_of_bounds_coordinates"] > 0
    }
    assert selected == expected
    assert len(selected) == 24
    assert sum(selected.values()) == 37


def test_control_coordinate_vocabulary_probe_separates_endpoints() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_codec_study_config(
        root / "configs/codec/control-coordinate-vocabulary-v1.yaml"
    )

    assert config.schema_version == 3
    assert config.coordinate_bins == (289,)
    assert config.control_coordinate_bins == (417,)
    assert config.control_coordinate_min == -8.0
    assert config.control_coordinate_max == 96.0
    assert not config.allow_clamping


def test_style_vocabulary_probe_is_pinned_and_bounded() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_style_study_config(root / "configs/codec/style-vocabulary-v1.yaml")

    assert [(item.name, item.tokens) for item in config.width_candidates] == [
        ("compact-k32", 32),
        ("leading-k48", 48),
    ]
    assert config.coordinate_bins == 289
    assert config.control_coordinate_bins == 417
    assert (config.max_paths, config.max_segments) == (96, 384)
    assert len(config.dash_patterns) == 6
    assert config.max_fixture_icons == 96

    sentinel = load_style_study_config(
        root / "configs/codec/style-vocabulary-v2-render-sentinel.yaml"
    )
    assert sentinel.fixture_manifest == Path("reports/codec/style-vocabulary-v1/fixture.json")
    assert sentinel.width_candidates[-1].supplemental_values == (4.1,)
    assert sentinel.max_fixture_icons == 35


def test_relative_l1_vocabulary_uses_deterministic_observed_weighted_medians() -> None:
    counts = {1.0: 1, 2.0: 1, 10.0: 1}

    assert optimal_relative_l1_vocabulary(counts, 1) == (1.0,)
    assert optimal_relative_l1_vocabulary(counts, 2) == (1.0, 10.0)
    assert optimal_relative_l1_vocabulary(counts, 3) == (1.0, 2.0, 10.0)


def test_capacity_study_uses_whole_contour_packed_prefixes() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_capacity_study_config(root / "configs/codec/capacity-layout-v1.yaml")

    assert config.packed[-1] == PackedCapacity("packed-p80-t1216", 80, 1216)
    assert dense_retention((3, 8, 2), DenseCapacity("dense", 2, 4)) == (2, 7)
    assert packed_retention((3, 8, 2), PackedCapacity("packed", 3, 10)) == (1, 3)
    assert packed_retention((3, 8, 2), PackedCapacity("packed", 2, 11)) == (2, 11)


def test_packed_tensor_round_trip_and_serializer_match_dense() -> None:
    config = _config(max_paths=4, max_segments=8)
    program = FloatProgram(
        (
            _contour(layer=1),
            _contour(
                layer=2,
                start=(3.0, 4.0),
                segments=(
                    FloatSegment(SegmentType.LINE, (5.0, 6.0)),
                    FloatSegment(SegmentType.LINE, (7.0, 8.0)),
                ),
            ),
        )
    )
    dense, _ = encode_program(program, config)

    packed = pack_tensor_program(dense, config, total_segment_slots=4)
    recovered = unpack_tensor_program(packed, config, total_segment_slots=4)

    for field in dense.__dataclass_fields__:
        assert np.array_equal(getattr(dense, field), getattr(recovered, field))
    assert packed.path_length.tolist() == [2, 2, 0, 0]
    assert packed.segment_type.tolist() == [
        int(SegmentType.LINE),
        int(SegmentType.CLOSE),
        int(SegmentType.LINE),
        int(SegmentType.LINE),
    ]
    assert serialize_packed_svg(packed, config, 4) == serialize_svg(dense, config)


def test_packed_tensor_rejects_capacity_padding_and_grammar_violations() -> None:
    config = _config(max_paths=4, max_segments=8)
    dense, _ = encode_program(FloatProgram((_contour(),)), config)

    with pytest.raises(BudgetExceeded, match="needs 2 packed segments"):
        pack_tensor_program(dense, config, total_segment_slots=1)

    packed = pack_tensor_program(dense, config, total_segment_slots=4)
    bad_tail = copy.deepcopy(packed)
    bad_tail.segment_type[-1] = int(SegmentType.LINE)
    with pytest.raises(ProgramValidationError, match="inactive packed segment types"):
        validate_packed_tensor_program(bad_tail, config, 4)

    bad_lengths = copy.deepcopy(packed)
    bad_lengths.path_length[0] = 5
    with pytest.raises(ProgramValidationError, match="path lengths exceed"):
        validate_packed_tensor_program(bad_lengths, config, 4)

    wrong_shape = replace(packed, coordinates=np.zeros((3, 6), dtype=np.int64))
    with pytest.raises(ProgramValidationError, match="wrong packed shape"):
        validate_packed_tensor_program(wrong_shape, config, 4)


@pytest.mark.parametrize("seed", range(20))
def test_random_valid_programs_pack_reversibly(seed: int) -> None:
    rng = np.random.default_rng(seed)
    config = _config(
        max_paths=5,
        max_segments=7,
        coordinate_bins=289,
        control_coordinate_bins=417,
        control_coordinate_min=-8.0,
        control_coordinate_max=96.0,
    )
    contours: list[FloatContour] = []
    for path_index in range(int(rng.integers(1, config.max_paths + 1))):
        segments: list[FloatSegment] = []
        geometry_count = int(rng.integers(1, config.max_segments + 1))
        for _ in range(geometry_count):
            kind = (SegmentType.LINE, SegmentType.QUAD, SegmentType.CUBIC)[
                int(rng.integers(0, 3))
            ]
            control_count = {SegmentType.LINE: 0, SegmentType.QUAD: 2, SegmentType.CUBIC: 4}[
                kind
            ]
            control = tuple(float(rng.integers(-32, 385)) / 4 for _ in range(control_count))
            endpoint = tuple(float(rng.integers(0, 289)) / 4 for _ in range(2))
            segments.append(FloatSegment(kind, control + endpoint))
        contours.append(
            _contour(
                layer=path_index + 1,
                start=tuple(float(rng.integers(0, 289)) / 4 for _ in range(2)),  # type: ignore[arg-type]
                segments=tuple(segments),
            )
        )
    dense, _ = encode_program(FloatProgram(tuple(contours)), config)
    total = sum(int(value) for value in dense.path_length)

    packed = pack_tensor_program(dense, config, total)
    recovered = unpack_tensor_program(packed, config, total)

    for field in dense.__dataclass_fields__:
        assert np.array_equal(getattr(dense, field), getattr(recovered, field))
    assert serialize_packed_svg(packed, config, total) == serialize_svg(dense, config)


def test_typed_svg_renders_in_resource_limited_subprocess() -> None:
    config = _config()
    dense, _ = encode_program(
        FloatProgram(
            (
                _contour(
                    segments=(
                        FloatSegment(SegmentType.LINE, (20.0, 2.0)),
                        FloatSegment(SegmentType.LINE, (10.0, 20.0)),
                        FloatSegment(SegmentType.CLOSE, ()),
                    )
                ),
            )
        ),
        config,
    )
    svg = serialize_svg(dense, config)

    png, rgba = render_typed_svg_isolated(svg, 18, RenderLimits(timeout_seconds=3))

    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    assert rgba.shape == (18, 18, 4)
    assert np.count_nonzero(rgba[:, :, 3]) > 0


@pytest.mark.parametrize(
    ("source", "code"),
    (
        (b"<not-svg/>", "INVALID_TYPED_ROOT"),
        (
            b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">'
            b'<script d="M0 0"/></svg>',
            "INVALID_TYPED_CHILD",
        ),
        (
            b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">'
            b'<path d="M0 0" href="https://example.invalid/x"/></svg>',
            "INVALID_TYPED_ATTRIBUTE",
        ),
        (
            b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">'
            b'<path d="M0 0" fill="url(https://example.invalid/x)"/></svg>',
            "UNSAFE_TYPED_VALUE",
        ),
    ),
)
def test_typed_renderer_rejects_non_serializer_xml(source: bytes, code: str) -> None:
    with pytest.raises(IsolatedRenderError) as captured:
        validate_typed_svg(source, RenderLimits())
    assert captured.value.code == code


def test_typed_renderer_enforces_input_and_path_bounds() -> None:
    with pytest.raises(IsolatedRenderError) as captured:
        validate_typed_svg(b"x" * 9, RenderLimits(max_svg_bytes=8))
    assert captured.value.code == "SVG_BYTES_LIMIT"

    source = (
        b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">'
        b'<path d="M0 0"/><path d="M1 1"/></svg>'
    )
    with pytest.raises(IsolatedRenderError) as captured:
        validate_typed_svg(source, RenderLimits(max_paths=1))
    assert captured.value.code == "PATH_LIMIT"


def test_codec_study_artifacts_are_create_or_identical(tmp_path: Path) -> None:
    artifact = tmp_path / "report" / "evidence.json"
    _write_bytes_artifact(artifact, b"first")
    _write_bytes_artifact(artifact, b"first")

    with pytest.raises(CodecStudyError, match="differing artifact"):
        _write_bytes_artifact(artifact, b"second")

    assert artifact.read_bytes() == b"first"
    symlink = tmp_path / "report" / "linked.json"
    symlink.symlink_to(artifact)
    with pytest.raises(CodecStudyError, match="symlink"):
        _write_bytes_artifact(symlink, b"first")


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


def test_extended_control_lattice_is_quarter_unit_and_keeps_endpoints_bounded() -> None:
    config = _config(
        coordinate_bins=289,
        control_coordinate_bins=417,
        control_coordinate_min=-8.0,
        control_coordinate_max=96.0,
    )
    assert quantize_bounded_coordinate(-8.0, 417, -8.0, 96.0) == 0
    assert quantize_bounded_coordinate(0.0, 417, -8.0, 96.0) == 32
    assert quantize_bounded_coordinate(72.0, 417, -8.0, 96.0) == 320
    assert quantize_bounded_coordinate(96.0, 417, -8.0, 96.0) == 416
    assert dequantize_bounded_coordinate(412, 417, -8.0, 96.0) == 95.0

    program = FloatProgram(
        (
            _contour(
                segments=(
                    FloatSegment(
                        SegmentType.CUBIC,
                        (-6.6875, 95.0224, 20.0, 30.0, 72.0, 50.0),
                    ),
                )
            ),
        )
    )
    tensor, report = encode_program(program, config)
    assert report.lossless
    assert report.clamped_coordinates == 0
    assert report.clamped_endpoint_coordinates == 0
    assert report.clamped_control_coordinates == 0
    decoded = decode_program(tensor, config)
    assert decoded.contours[0].segments[0].coords == (-6.75, 95.0, 20.0, 30.0, 72.0, 50.0)

    mutated = copy.deepcopy(tensor)
    mutated.coordinates[0, 0, 0] = 417
    validate_tensor_program(mutated, config)
    mutated.coordinates[0, 0, 4] = 417
    with pytest.raises(ProgramValidationError, match="endpoint coordinate"):
        validate_tensor_program(mutated, config)

    endpoint_oob = FloatProgram(
        (
            _contour(
                segments=(FloatSegment(SegmentType.LINE, (73.0, 50.0)),),
            ),
        )
    )
    with pytest.raises(CodecError, match="endpoint coordinate"):
        encode_program(endpoint_oob, config)


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


def test_oob_analysis_classifies_control_handles_and_preserves_counterfactual() -> None:
    program = FloatProgram(
        (
            _contour(
                start=(-1.0, 2.0),
                segments=(
                    FloatSegment(SegmentType.CUBIC, (10.0, 73.0, 20.0, 30.0, 40.0, 50.0)),
                ),
            ),
        )
    )

    excursions = _coordinate_excursions(program)
    assert [(item["segment_type"], item["role"], item["axis"]) for item in excursions] == [
        ("move", "endpoint", "x"),
        ("cubic", "control", "y"),
    ]
    quantized = _quantize_unclamped(program, 289)
    assert quantized.contours[0].start == (-1.0, 2.0)
    assert quantized.contours[0].segments[0].coords[1] == 73.0
    counterfactual = serialize_float_svg(quantized, max_serialized_bytes=20_000)
    assert b"M -1.0 2.0" in counterfactual
    assert b"10.0 73.0" in counterfactual

    tensor, report = encode_program(program, _config(coordinate_bins=289), allow_clamping=True)
    assert report.clamped_coordinates == 2
    assert report.clamped_endpoint_coordinates == 1
    assert report.clamped_control_coordinates == 1
    assert counterfactual != serialize_svg(tensor, _config(coordinate_bins=289))


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
