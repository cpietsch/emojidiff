"""Deterministic SVG parsing, bounded rendering, and raw corpus metrics."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import re
import signal
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cairosvg
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from defusedxml import ElementTree
from PIL import Image
from svg.path import Close, Move, parse_path

from mojidiff.curation.source import OpenMojiSource, load_source_config, sha256_file


class AuditError(RuntimeError):
    """Audit configuration or execution failed without altering raw source."""


class RenderTimeout(TimeoutError):
    """A bounded raster render exceeded its wall-clock limit."""


@dataclass(frozen=True)
class AuditConfig:
    version: str
    source: OpenMojiSource
    output_root: Path
    sizes: tuple[int, ...]
    timeout_seconds: int
    max_svg_bytes: int
    max_elements: int
    decisions: dict[str, Any]


def load_audit_config(path: Path) -> AuditConfig:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise AuditError(f"cannot load audit config {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise AuditError("audit config schema_version must be 1")
    render = document.get("render")
    decisions = document.get("decisions")
    if not isinstance(render, dict) or not isinstance(decisions, dict):
        raise AuditError("render and decisions must be mappings")
    sizes = render.get("sizes")
    if not isinstance(sizes, list) or sizes != [72, 18]:
        raise AuditError("audit render sizes must be exactly [72, 18]")
    source_config = document.get("source_config")
    if not isinstance(source_config, str):
        raise AuditError("source_config must be a path")
    version = document.get("audit_version")
    output_root = document.get("output_root")
    if not isinstance(version, str) or not isinstance(output_root, str):
        raise AuditError("audit_version and output_root must be strings")
    return AuditConfig(
        version=version,
        source=load_source_config(Path(source_config)),
        output_root=Path(output_root),
        sizes=(72, 18),
        timeout_seconds=_positive_int(render.get("timeout_seconds"), "timeout_seconds"),
        max_svg_bytes=_positive_int(render.get("max_svg_bytes"), "max_svg_bytes"),
        max_elements=_positive_int(render.get("max_elements"), "max_elements"),
        decisions=decisions,
    )


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise AuditError(f"{field} must be a positive integer")
    return value


@contextmanager
def _deadline(seconds: int) -> Iterator[None]:
    def expired(_signum: int, _frame: object) -> None:
        raise RenderTimeout(f"render exceeded {seconds} seconds")

    previous = signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _safe_parse(svg_bytes: bytes, config: AuditConfig) -> tuple[Any, list[Any]]:
    if len(svg_bytes) > config.max_svg_bytes:
        raise AuditError("SVG_BYTES_LIMIT")
    try:
        root = ElementTree.fromstring(svg_bytes)
    except Exception as exc:
        raise AuditError(f"PARSE_ERROR:{type(exc).__name__}") from exc
    elements = list(root.iter())
    if len(elements) > config.max_elements:
        raise AuditError("ELEMENT_LIMIT")
    denied_tags = {"script", "image", "foreignObject", "iframe", "audio", "video"}
    for element in elements:
        if _local_name(element.tag) in denied_tags:
            raise AuditError(f"UNSAFE_TAG:{_local_name(element.tag)}")
        for key, value in element.attrib.items():
            key_name = _local_name(key).lower()
            lowered = value.lower()
            if key_name.startswith("on"):
                raise AuditError("UNSAFE_EVENT_ATTRIBUTE")
            if key_name in {"href", "src"} and not re.fullmatch(
                r"#[A-Za-z_][A-Za-z0-9_.:-]*", value.strip()
            ):
                raise AuditError("EXTERNAL_REFERENCE")
            without_internal_urls = re.sub(r"url\(\s*#[A-Za-z_][A-Za-z0-9_.:-]*\s*\)", "", lowered)
            if "url(" in without_internal_urls or "javascript:" in lowered or "data:" in lowered:
                raise AuditError("EXTERNAL_OR_EMBEDDED_RESOURCE")
    return root, elements


def _render(svg_bytes: bytes, size: int, timeout_seconds: int) -> tuple[bytes, np.ndarray]:
    with _deadline(timeout_seconds):
        png = cairosvg.svg2png(
            bytestring=svg_bytes,
            output_width=size,
            output_height=size,
            unsafe=False,
        )
    image = Image.open(io.BytesIO(png)).convert("RGBA")
    return png, np.asarray(image)


def _raster_metrics(array: np.ndarray) -> dict[str, Any]:
    height, width, _channels = array.shape
    alpha = array[:, :, 3]
    visible = alpha > 0
    count = int(np.count_nonzero(visible))
    mass = float(alpha.astype(np.float64).sum() / 255.0)
    center_x: float | None
    center_y: float | None
    if count:
        ys, xs = np.nonzero(visible)
        left, right = int(xs.min()), int(xs.max())
        top, bottom = int(ys.min()), int(ys.max())
        bbox_width = right - left + 1
        bbox_height = bottom - top + 1
        center_x = float((left + right) / 2)
        center_y = float((top + bottom) / 2)
        colors = np.unique(array[visible][:, :3], axis=0)
    else:
        left = top = right = bottom = -1
        bbox_width = bbox_height = 0
        center_x = center_y = None
        colors = np.empty((0, 3), dtype=np.uint8)
    components = _components(visible)
    return {
        "alpha_pixel_count": count,
        "alpha_fraction": count / (width * height),
        "ink_mass": mass,
        "ink_fraction": mass / (width * height),
        "bbox_left": left,
        "bbox_top": top,
        "bbox_right": right,
        "bbox_bottom": bottom,
        "bbox_width": bbox_width,
        "bbox_height": bbox_height,
        "bbox_area": bbox_width * bbox_height,
        "bbox_center_x": center_x,
        "bbox_center_y": center_y,
        "touches_boundary": bool(
            count and (left == 0 or top == 0 or right == width - 1 or bottom == height - 1)
        ),
        "visible_color_count": int(len(colors)),
        "component_count": len(components),
        "tiny_component_count": sum(size <= 2 for size in components),
        "smallest_component_pixels": min(components, default=0),
    }


def _components(mask: np.ndarray) -> list[int]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    sizes: list[int] = []
    for start_y, start_x in zip(*np.nonzero(mask), strict=True):
        if seen[start_y, start_x]:
            continue
        seen[start_y, start_x] = True
        queue = deque([(int(start_y), int(start_x))])
        size = 0
        while queue:
            y, x = queue.popleft()
            size += 1
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    ny, nx = y + dy, x + dx
                    if (
                        (dx or dy)
                        and 0 <= ny < height
                        and 0 <= nx < width
                        and mask[ny, nx]
                        and not seen[ny, nx]
                    ):
                        seen[ny, nx] = True
                        queue.append((ny, nx))
        sizes.append(size)
    return sorted(sizes)


def _dhash(array: np.ndarray) -> str:
    image = (
        Image.fromarray(array, mode="RGBA").convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    )
    pixels = np.asarray(image)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = sum(int(bit) << index for index, bit in enumerate(bits.flat))
    return f"{value:016x}"


def _style_and_geometry(elements: list[Any]) -> dict[str, Any]:
    element_counts: dict[str, int] = {}
    fill_count = stroke_count = opacity_count = 0
    path_count = subpath_count = segment_count = close_count = 0
    open_path_count = 0
    path_length = 0.0
    geometry_points: list[complex] = []
    transform_count = 0
    parse_failures = 0
    for element in elements:
        name = _local_name(element.tag)
        element_counts[name] = element_counts.get(name, 0) + 1
        attrs = element.attrib
        fill = attrs.get("fill")
        stroke = attrs.get("stroke")
        if fill and fill.lower() != "none":
            fill_count += 1
        if stroke and stroke.lower() != "none":
            stroke_count += 1
        if "opacity" in attrs or "fill-opacity" in attrs or "stroke-opacity" in attrs:
            opacity_count += 1
        if "transform" in attrs:
            transform_count += 1
        if name != "path":
            geometry_points.extend(_basic_shape_points(name, attrs))
            continue
        path_count += 1
        data = attrs.get("d", "")
        try:
            parsed = parse_path(data)
        except Exception:
            parse_failures += 1
            continue
        moves = sum(isinstance(segment, Move) for segment in parsed)
        closes = sum(isinstance(segment, Close) for segment in parsed)
        subpaths = max(1, moves)
        subpath_count += subpaths
        close_count += closes
        open_path_count += max(0, subpaths - closes)
        drawable = [segment for segment in parsed if not isinstance(segment, Move)]
        segment_count += len(drawable)
        for segment in drawable:
            try:
                path_length += float(segment.length(error=1e-4))
                geometry_points.extend(segment.point(step / 8) for step in range(9))
            except (AttributeError, ValueError, ZeroDivisionError):
                parse_failures += 1
    inside = sum(0 <= point.real <= 72 and 0 <= point.imag <= 72 for point in geometry_points)
    outside = len(geometry_points) - inside
    return {
        "element_count": len(elements),
        "element_counts_json": json.dumps(element_counts, sort_keys=True, separators=(",", ":")),
        "path_count": path_count,
        "subpath_count": subpath_count,
        "segment_count": segment_count,
        "open_path_count": open_path_count,
        "closed_path_count": close_count,
        "total_path_length_estimate": path_length,
        "fill_element_count": fill_count,
        "stroke_element_count": stroke_count,
        "opacity_attribute_count": opacity_count,
        "transform_count": transform_count,
        "path_parse_failure_count": parse_failures,
        "geometry_sample_count": len(geometry_points),
        "geometry_inside_sample_count": inside,
        "geometry_outside_sample_count": outside,
        "geometry_inside_fraction": inside / len(geometry_points) if geometry_points else None,
        "geometry_estimate_has_unapplied_transforms": transform_count > 0,
    }


def _basic_shape_points(name: str, attrs: dict[str, str]) -> list[complex]:
    def number(field: str, default: float = 0.0) -> float:
        try:
            return float(attrs.get(field, default))
        except ValueError:
            return math.nan

    if name == "circle":
        cx, cy, radius = number("cx"), number("cy"), number("r")
        return [
            complex(cx + radius * math.cos(t), cy + radius * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)
        ]
    if name == "ellipse":
        cx, cy, rx, ry = number("cx"), number("cy"), number("rx"), number("ry")
        return [
            complex(cx + rx * math.cos(t), cy + ry * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, 16, endpoint=False)
        ]
    if name == "line":
        return [complex(number("x1"), number("y1")), complex(number("x2"), number("y2"))]
    if name == "rect":
        x, y, width, height = number("x"), number("y"), number("width"), number("height")
        return [
            complex(x, y),
            complex(x + width, y),
            complex(x + width, y + height),
            complex(x, y + height),
        ]
    if name in {"polygon", "polyline"}:
        values = attrs.get("points", "").replace(",", " ").split()
        try:
            return [
                complex(float(values[i]), float(values[i + 1]))
                for i in range(0, len(values) - 1, 2)
            ]
        except ValueError:
            return []
    return []


def _family_id(metadata: dict[str, str]) -> str:
    base = metadata.get("skintone_base_hexcode", "").strip() or metadata.get("hexcode", "")
    excluded = {"FE0F", "200D", "2640", "2642", "1F3FB", "1F3FC", "1F3FD", "1F3FE", "1F3FF"}
    normalized = "-".join(code for code in base.upper().split("-") if code not in excluded)
    return hashlib.sha256(f"unicode-family:{normalized}".encode()).hexdigest()[:16]


def audit_one(
    svg_path: Path,
    metadata: dict[str, str],
    config: AuditConfig,
    renders_root: Path,
) -> dict[str, Any]:
    source_path = svg_path.relative_to(config.source.raw_root).as_posix()
    row: dict[str, Any] = {
        "audit_version": config.version,
        "source_revision": config.source.revision,
        "source_path": source_path,
        "source_svg_sha256": sha256_file(svg_path),
        "source_svg_bytes": svg_path.stat().st_size,
        "hexcode": metadata.get("hexcode", svg_path.stem),
        "unicode_sequence": metadata.get("emoji", ""),
        "annotation": metadata.get("annotation", ""),
        "group": metadata.get("group", ""),
        "subgroup": metadata.get("subgroups", ""),
        "tags": metadata.get("tags", ""),
        "openmoji_tags": metadata.get("openmoji_tags", ""),
        "variant_family_id": _family_id(metadata),
        "parse_ok": False,
        "parse_error": None,
        "render_error": None,
    }
    try:
        svg_bytes = svg_path.read_bytes()
        root, elements = _safe_parse(svg_bytes, config)
        row["parse_ok"] = True
        row["viewbox"] = root.attrib.get("viewBox", "")
        row.update(_style_and_geometry(elements))
    except Exception as exc:
        row["parse_error"] = str(exc)[:500]
        for size in config.sizes:
            row[f"render_{size}_ok"] = False
        return row

    for size in config.sizes:
        try:
            png, array = _render(svg_bytes, size, config.timeout_seconds)
            render_path = renders_root / str(size) / f"{svg_path.stem}.png"
            render_path.parent.mkdir(parents=True, exist_ok=True)
            render_path.write_bytes(png)
            row[f"render_{size}_ok"] = True
            row[f"render_{size}_rgba_sha256"] = hashlib.sha256(array.tobytes()).hexdigest()
            row[f"render_{size}_dhash"] = _dhash(array)
            for key, value in _raster_metrics(array).items():
                row[f"{key}_{size}"] = value
        except Exception as exc:
            row[f"render_{size}_ok"] = False
            row["render_error"] = f"{size}px:{type(exc).__name__}:{exc}"[:500]
    return row


def run_audit(config: AuditConfig, limit: int | None = None) -> list[dict[str, Any]]:
    if not config.source.raw_root.is_dir():
        raise AuditError(f"raw source is absent: {config.source.raw_root}")
    metadata_path = config.source.raw_root / config.source.metadata_path
    with metadata_path.open(encoding="utf-8", newline="") as handle:
        metadata_rows = list(csv.DictReader(handle))
    metadata = {row["hexcode"].upper(): row for row in metadata_rows}
    svg_paths = sorted(config.source.raw_root.glob(config.source.svg_glob))
    if limit is not None:
        svg_paths = svg_paths[:limit]
    rows = [
        audit_one(path, metadata.get(path.stem.upper(), {}), config, config.output_root / "renders")
        for path in svg_paths
    ]
    _assign_duplicate_clusters(rows)
    config.output_root.mkdir(parents=True, exist_ok=True)
    parquet_path = config.output_root / "audit.parquet"
    pq.write_table(pa.Table.from_pylist(rows), parquet_path, compression="zstd")
    jsonl_path = config.output_root / "audit.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    return rows


def _assign_duplicate_clusters(rows: list[dict[str, Any]]) -> None:
    exact: dict[str, list[dict[str, Any]]] = {}
    perceptual: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        raster_hash = row.get("render_72_rgba_sha256")
        dhash = row.get("render_72_dhash")
        if isinstance(raster_hash, str):
            exact.setdefault(raster_hash, []).append(row)
        if isinstance(dhash, str):
            perceptual.setdefault(dhash, []).append(row)
    for groups, field, prefix in (
        (exact, "exact_raster_cluster", "exact"),
        (perceptual, "perceptual_hash_cluster", "dhash"),
    ):
        for digest, members in groups.items():
            cluster = f"{prefix}-{digest[:16]}" if len(members) > 1 else None
            for row in members:
                row[field] = cluster
