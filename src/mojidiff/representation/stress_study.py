"""Deterministic Gate D stress study for packed grammar and renderer safety."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import yaml

from mojidiff.representation.codec_study import _palette, _write_bytes_artifact
from mojidiff.representation.packed import (
    PackedTensorProgram,
    pack_tensor_program,
    serialize_packed_svg,
    unpack_tensor_program,
    validate_packed_tensor_program,
)
from mojidiff.representation.program import (
    PAD,
    CodecConfig,
    FloatContour,
    FloatProgram,
    FloatSegment,
    ProgramValidationError,
    SegmentType,
    encode_program,
    serialize_svg,
)
from mojidiff.representation.renderer import (
    IsolatedRenderError,
    RenderLimits,
    render_typed_svg_isolated,
    validate_typed_svg,
)


class StressStudyError(RuntimeError):
    """The invariant stress run failed or cannot be reproduced."""


@dataclass(frozen=True)
class StressStudyConfig:
    version: str
    palette_path: Path
    palette_sha256: str
    style_summary: Path
    style_summary_sha256: str
    style_candidate: str
    report_root: Path
    random_seed: int
    valid_programs: int
    mutation_trials: int
    render_programs: int
    render_sizes: tuple[int, ...]
    max_paths: int
    max_segments: int
    total_segment_slots: int
    outer_max_paths: int
    outer_max_segments: int
    outer_total_segment_slots: int
    timeout_seconds: int


def load_stress_study_config(path: Path) -> StressStudyConfig:
    root = _mapping(yaml.safe_load(path.read_bytes()), "root")
    if root.get("schema_version") != 1:
        raise StressStudyError("stress study schema_version must be 1")
    inputs = _mapping(root.get("inputs"), "inputs")
    random = _mapping(root.get("random"), "random")
    fixture = _mapping(root.get("fixture_codec"), "fixture_codec")
    outer = _mapping(root.get("outer_bound"), "outer_bound")
    render = _mapping(root.get("render"), "render")
    sizes = tuple(
        _positive_int(value, "render size")
        for value in _sequence(render.get("sizes"), "sizes")
    )
    if not sizes or any(size > 512 for size in sizes):
        raise StressStudyError("render sizes must be within 1..512")
    seed = random.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise StressStudyError("random seed must be a nonnegative integer")
    return StressStudyConfig(
        version=_string(root, "study_version"),
        palette_path=Path(_string(inputs, "palette")),
        palette_sha256=_sha256(_string(inputs, "palette_sha256")),
        style_summary=Path(_string(inputs, "style_summary")),
        style_summary_sha256=_sha256(_string(inputs, "style_summary_sha256")),
        style_candidate=_string(inputs, "style_candidate"),
        report_root=Path(_string(root, "report_root")),
        random_seed=seed,
        valid_programs=_positive_int(random.get("valid_programs"), "valid_programs"),
        mutation_trials=_positive_int(random.get("mutation_trials"), "mutation_trials"),
        render_programs=_positive_int(random.get("render_programs"), "render_programs"),
        render_sizes=sizes,
        max_paths=_positive_int(fixture.get("max_paths"), "max_paths"),
        max_segments=_positive_int(fixture.get("max_segments"), "max_segments"),
        total_segment_slots=_positive_int(
            fixture.get("total_segment_slots"), "total_segment_slots"
        ),
        outer_max_paths=_positive_int(outer.get("max_paths"), "outer max_paths"),
        outer_max_segments=_positive_int(outer.get("max_segments"), "outer max_segments"),
        outer_total_segment_slots=_positive_int(
            outer.get("total_segment_slots"), "outer total_segment_slots"
        ),
        timeout_seconds=_positive_int(render.get("timeout_seconds"), "timeout_seconds"),
    )


def run_stress_study(config: StressStudyConfig, config_path: Path) -> dict[str, Any]:
    _verify(config.palette_path, config.palette_sha256, "palette")
    _verify(config.style_summary, config.style_summary_sha256, "style summary")
    palette = _palette(config.palette_path)
    style = _mapping(json.loads(config.style_summary.read_bytes()), "style summary")
    analytics = _mapping(style.get("width_vocabulary_analytics"), "width analytics")
    candidate = _mapping(analytics.get(config.style_candidate), "style candidate")
    widths = tuple(float(value) for value in _sequence(candidate.get("values"), "widths"))
    codec = _codec(config.max_paths, config.max_segments, palette, widths)
    rng = np.random.default_rng(config.random_seed)
    programs = [_random_program(rng, codec) for _ in range(config.valid_programs)]
    valid_rows: list[dict[str, Any]] = []
    packed_programs: list[PackedTensorProgram] = []
    for index, program in enumerate(programs):
        dense, report = encode_program(program, codec)
        if not report.lossless:
            raise StressStudyError("random valid program unexpectedly projected")
        active_segments = sum(int(value) for value in dense.path_length)
        if active_segments > config.total_segment_slots - 8:
            raise StressStudyError("random generator exceeded reserved packed slack")
        packed = pack_tensor_program(dense, codec, config.total_segment_slots)
        recovered = unpack_tensor_program(packed, codec, config.total_segment_slots)
        if serialize_packed_svg(packed, codec, config.total_segment_slots) != serialize_svg(
            recovered, codec
        ):
            raise StressStudyError("packed serializer disagrees with dense serializer")
        packed_programs.append(packed)
        valid_rows.append(
            {
                "index": index,
                "paths": sum(int(value) > 0 for value in dense.path_length),
                "segments": active_segments,
                "svg_sha256": hashlib.sha256(serialize_svg(dense, codec)).hexdigest(),
            }
        )

    mutation_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    for trial in range(config.mutation_trials):
        mutation = _MUTATIONS[trial % len(_MUTATIONS)]
        source = packed_programs[trial % len(packed_programs)]
        mutant = _mutate(source, mutation, codec, config.total_segment_slots)
        mutation_counts[mutation] += 1
        try:
            validate_packed_tensor_program(mutant, codec, config.total_segment_slots)
        except ProgramValidationError as exc:
            rejection_counts[str(exc)] += 1
        else:
            raise StressStudyError(f"invalid mutation was accepted: {mutation}")

    render_rows: list[dict[str, Any]] = []
    limits = RenderLimits(
        max_paths=config.max_paths,
        timeout_seconds=config.timeout_seconds,
    )
    for index, packed in enumerate(packed_programs[: config.render_programs]):
        svg = serialize_packed_svg(packed, codec, config.total_segment_slots)
        for size in config.render_sizes:
            png, rgba = render_typed_svg_isolated(svg, size, limits)
            render_rows.append(
                {
                    "index": index,
                    "size": size,
                    "png_sha256": hashlib.sha256(png).hexdigest(),
                    "visible_pixels": int(np.count_nonzero(rgba[:, :, 3])),
                }
            )

    boundary = _boundary_program(config)
    outer_codec = _codec(config.outer_max_paths, config.outer_max_segments, palette, widths)
    boundary_dense, boundary_report = encode_program(boundary, outer_codec)
    if not boundary_report.lossless:
        raise StressStudyError("outer-bound fixture unexpectedly projected")
    boundary_packed = pack_tensor_program(
        boundary_dense, outer_codec, config.outer_total_segment_slots
    )
    boundary_svg = serialize_packed_svg(
        boundary_packed, outer_codec, config.outer_total_segment_slots
    )
    boundary_png, _ = render_typed_svg_isolated(
        boundary_svg,
        min(config.render_sizes),
        RenderLimits(
            max_paths=config.outer_max_paths,
            timeout_seconds=config.timeout_seconds,
        ),
    )

    malformed_codes = _malformed_cases(limits)
    valid_payload = _jsonl(valid_rows)
    render_payload = _jsonl(render_rows)
    summary = {
        "schema_version": 1,
        "study_version": config.version,
        "code_identity": _code_identity(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "random_seed": config.random_seed,
        "valid_programs": len(valid_rows),
        "stable_packed_round_trips": len(valid_rows),
        "mutation_trials": config.mutation_trials,
        "mutation_counts": dict(sorted(mutation_counts.items())),
        "rejected_mutations": sum(rejection_counts.values()),
        "rejection_classes": dict(sorted(rejection_counts.items())),
        "isolated_renders": len(render_rows) + 1,
        "malformed_input_classes": malformed_codes,
        "outer_boundary": {
            "paths": sum(int(value) > 0 for value in boundary_dense.path_length),
            "segments": sum(int(value) for value in boundary_dense.path_length),
            "packed_slots": config.outer_total_segment_slots,
            "svg_bytes": len(boundary_svg),
            "svg_sha256": hashlib.sha256(boundary_svg).hexdigest(),
            "png_sha256": hashlib.sha256(boundary_png).hexdigest(),
        },
        "valid_programs_sha256": hashlib.sha256(valid_payload).hexdigest(),
        "render_metrics_sha256": hashlib.sha256(render_payload).hexdigest(),
    }
    _write_bytes_artifact(config.report_root / "valid-programs.jsonl", valid_payload)
    _write_bytes_artifact(config.report_root / "render-metrics.jsonl", render_payload)
    _write_bytes_artifact(config.report_root / "summary.json", _json(summary))
    _write_bytes_artifact(config.report_root / "README.md", _markdown(summary).encode())
    return summary


_MUTATIONS = (
    "unknown-segment",
    "unused-coordinate",
    "endpoint-token",
    "path-style-token",
    "inactive-segment-tail",
    "inactive-coordinate-tail",
    "invalid-layer",
    "inactive-path-style",
    "path-length-overflow",
    "wrong-coordinate-shape",
)


def _mutate(
    source: PackedTensorProgram,
    mutation: str,
    codec: CodecConfig,
    total_slots: int,
) -> PackedTensorProgram:
    value = copy.deepcopy(source)
    active = sum(int(item) for item in value.path_length)
    active_paths = sum(int(item) > 0 for item in value.path_length)
    if mutation == "unknown-segment":
        value.segment_type[0] = 99
    elif mutation == "unused-coordinate":
        kind = SegmentType(int(value.segment_type[0]))
        used = {SegmentType.LINE: 2, SegmentType.QUAD: 4, SegmentType.CUBIC: 6}[kind]
        if used == 6:
            value.segment_type[0] = int(SegmentType.LINE)
            used = 2
        value.coordinates[0, used] = 1
    elif mutation == "endpoint-token":
        value.segment_type[0] = int(SegmentType.LINE)
        value.coordinates[0, 0] = codec.coordinate_bins
        value.coordinates[0, 2:] = PAD
    elif mutation == "path-style-token":
        value.fill[0] = len(codec.palette) + 2
    elif mutation == "inactive-segment-tail":
        value.segment_type[active] = int(SegmentType.LINE)
    elif mutation == "inactive-coordinate-tail":
        value.coordinates[active, 0] = 1
    elif mutation == "invalid-layer":
        value.layer[0] = 0
    elif mutation == "inactive-path-style":
        if active_paths < codec.max_paths:
            value.layer[active_paths] = 1
        else:
            value.path_length[-1] = 0
    elif mutation == "path-length-overflow":
        value.path_length[0] = total_slots + 1
    elif mutation == "wrong-coordinate-shape":
        value = replace(value, coordinates=value.coordinates[:-1])
    else:
        raise AssertionError(mutation)
    return value


def _random_program(rng: np.random.Generator, codec: CodecConfig) -> FloatProgram:
    contours: list[FloatContour] = []
    path_count = int(rng.integers(1, codec.max_paths + 1))
    for path_index in range(path_count):
        max_length = min(codec.max_segments, 12)
        length = int(rng.integers(1, max_length + 1))
        segments = tuple(_random_segment(rng) for _ in range(length))
        contours.append(_contour(path_index + 1, _endpoint(rng), segments))
    return FloatProgram(tuple(contours))


def _random_segment(rng: np.random.Generator) -> FloatSegment:
    kind = (SegmentType.LINE, SegmentType.QUAD, SegmentType.CUBIC)[int(rng.integers(0, 3))]
    control_count = {SegmentType.LINE: 0, SegmentType.QUAD: 2, SegmentType.CUBIC: 4}[kind]
    controls = tuple(float(rng.integers(-32, 385)) / 4 for _ in range(control_count))
    return FloatSegment(kind, controls + _endpoint(rng))


def _endpoint(rng: np.random.Generator) -> tuple[float, float]:
    return (float(rng.integers(0, 289)) / 4, float(rng.integers(0, 289)) / 4)


def _contour(
    layer: int, start: tuple[float, float], segments: tuple[FloatSegment, ...]
) -> FloatContour:
    return FloatContour(
        layer=layer,
        fill="#000000",
        stroke=None,
        stroke_width=1.0,
        linecap="butt",
        linejoin="miter",
        miter_limit=4.0,
        dash_pattern=(),
        fill_rule="nonzero",
        opacity=1.0,
        fill_opacity=1.0,
        stroke_opacity=None,
        start=start,
        segments=segments,
    )


def _boundary_program(config: StressStudyConfig) -> FloatProgram:
    lengths = [300, 300, 300, *([4] * 76), 12]
    if len(lengths) != config.outer_max_paths or sum(lengths) != config.outer_total_segment_slots:
        raise StressStudyError("outer-bound config does not match the pinned boundary fixture")
    segment = FloatSegment(SegmentType.LINE, (1.0, 1.0))
    return FloatProgram(
        tuple(
            _contour(index + 1, (0.0, 0.0), (segment,) * length)
            for index, length in enumerate(lengths)
        )
    )


def _malformed_cases(limits: RenderLimits) -> dict[str, str]:
    cases = {
        "parse": b"<svg",
        "unsafe": (
            b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">'
            b'<path d="M0 0" fill="url(https://example.invalid/x)"/></svg>'
        ),
        "wrong-root": b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1 1"/>',
    }
    result: dict[str, str] = {}
    for name, source in cases.items():
        try:
            validate_typed_svg(source, limits)
        except IsolatedRenderError as exc:
            result[name] = exc.code
        else:
            raise StressStudyError(f"malformed typed SVG accepted: {name}")
    return result


def _codec(
    max_paths: int, max_segments: int, palette: tuple[str, ...], widths: tuple[float, ...]
) -> CodecConfig:
    return CodecConfig(
        max_paths=max_paths,
        max_segments=max_segments,
        coordinate_bins=289,
        control_coordinate_bins=417,
        control_coordinate_min=-8.0,
        control_coordinate_max=96.0,
        palette=palette,
        stroke_widths=widths,
        dash_patterns=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.1598, 5.1598),
            (0.0, 0.0, 3.9396, 3.9396),
            (0.0, 6.7346, 0.0, 0.0, 0.0, 0.0),
            (2.0, 4.0),
            (5.2132, 5.2132),
            (6.1156, 4.5867),
        ),
        miter_limits=(1.5, 2.0, 4.0, 7.0, 10.0),
        opacities=(0.25, 0.4, 0.5, 0.502, 0.6, 0.9969, 0.997, 0.999, 1.0),
        max_serialized_bytes=2_000_000,
    )


def _markdown(summary: dict[str, Any]) -> str:
    boundary = summary["outer_boundary"]
    return "\n".join(
        (
            "# Packed invariant and isolated-render stress study",
            "",
            f"Validated {summary['valid_programs']} deterministic random programs and "
            f"rejected {summary['rejected_mutations']}/{summary['mutation_trials']} "
            "deliberately invalid packed tensors.",
            "",
            f"Completed {summary['isolated_renders']} subprocess renders, including the "
            f"P{boundary['paths']}/T{boundary['segments']} outer-bound fixture "
            f"({boundary['svg_bytes']} serialized bytes).",
            "",
            "Malformed typed XML, unsafe values, grammar violations, padding errors, "
            "token-range failures, shape failures, and resource bounds remain explicitly "
            "classified in `summary.json`.",
            "",
        )
    )


def _code_identity() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise StressStudyError(f"cannot identify Git commit: {exc}") from exc
    sources = (
        "representation/stress_study.py",
        "representation/packed.py",
        "representation/program.py",
        "representation/renderer.py",
        "representation/render_worker.py",
    )
    return {
        "git_commit": commit,
        **{
            Path(source).name.replace(".py", "_sha256"): _file_sha256(
                repository / "src/mojidiff" / source
            )
            for source in sources
        },
    }


def _verify(path: Path, expected: str, label: str) -> None:
    if _file_sha256(path) != expected:
        raise StressStudyError(f"{label} hash mismatch")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise StressStudyError(f"{field} must be a mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise StressStudyError(f"{field} must be a list")
    return value


def _string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise StressStudyError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise StressStudyError(f"{field} must be a positive integer")
    return value


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise StressStudyError("hash must be lowercase SHA-256")
    return value


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows).encode()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mojidiff-stress-study")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_stress_study(load_stress_study_config(args.config), args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
