"""Strict typed-SVG validation and resource-limited subprocess rendering."""

from __future__ import annotations

import io
import subprocess
import sys
from dataclasses import dataclass

import numpy as np
from defusedxml import ElementTree
from numpy.typing import NDArray
from PIL import Image

_SVG_TAG = "{http://www.w3.org/2000/svg}svg"
_PATH_TAG = "{http://www.w3.org/2000/svg}path"
_PATH_ATTRIBUTES = {
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
_DEFAULT_LIMITS: RenderLimits


@dataclass(frozen=True)
class RenderLimits:
    max_svg_bytes: int = 2_000_000
    max_png_bytes: int = 16_000_000
    max_paths: int = 80
    timeout_seconds: int = 5

    def __post_init__(self) -> None:
        values = (
            ("max_svg_bytes", self.max_svg_bytes, 4_000_000),
            ("max_png_bytes", self.max_png_bytes, 64_000_000),
            ("max_paths", self.max_paths, 512),
            ("timeout_seconds", self.timeout_seconds, 60),
        )
        for name, value, maximum in values:
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer within 1..{maximum}")


class IsolatedRenderError(RuntimeError):
    """A typed SVG failed a classified validation or isolated-render boundary."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


_DEFAULT_LIMITS = RenderLimits()


def validate_typed_svg(svg_bytes: bytes, limits: RenderLimits) -> None:
    """Accept only the exact XML surface emitted by the typed serializer."""

    if len(svg_bytes) > limits.max_svg_bytes:
        raise IsolatedRenderError("SVG_BYTES_LIMIT")
    try:
        root = ElementTree.fromstring(svg_bytes)
    except Exception as exc:
        raise IsolatedRenderError(f"PARSE_ERROR:{type(exc).__name__}") from exc
    if root.tag != _SVG_TAG or root.attrib != {"viewBox": "0 0 72 72"}:
        raise IsolatedRenderError("INVALID_TYPED_ROOT")
    children = list(root)
    if len(children) > limits.max_paths:
        raise IsolatedRenderError("PATH_LIMIT")
    for child in children:
        if child.tag != _PATH_TAG or list(child):
            raise IsolatedRenderError("INVALID_TYPED_CHILD")
        if not child.attrib.keys() <= _PATH_ATTRIBUTES or "d" not in child.attrib:
            raise IsolatedRenderError("INVALID_TYPED_ATTRIBUTE")
        lowered = " ".join(child.attrib.values()).lower()
        if any(value in lowered for value in ("url(", "javascript:", "data:", "href=")):
            raise IsolatedRenderError("UNSAFE_TYPED_VALUE")


def render_typed_svg_isolated(
    svg_bytes: bytes, size: int, limits: RenderLimits = _DEFAULT_LIMITS
) -> tuple[bytes, NDArray[np.uint8]]:
    """Render validated typed output in a bounded child process."""

    if isinstance(size, bool) or not isinstance(size, int) or not 1 <= size <= 512:
        raise IsolatedRenderError("INVALID_RENDER_SIZE")
    validate_typed_svg(svg_bytes, limits)
    argv = (
        sys.executable,
        "-I",
        "-B",
        "-m",
        "mojidiff.representation.render_worker",
        "--size",
        str(size),
        "--max-svg-bytes",
        str(limits.max_svg_bytes),
        "--max-png-bytes",
        str(limits.max_png_bytes),
        "--cpu-seconds",
        str(limits.timeout_seconds),
    )
    try:
        completed = subprocess.run(
            argv,
            input=svg_bytes,
            capture_output=True,
            check=False,
            timeout=limits.timeout_seconds + 1,
            env={"LC_ALL": "C.UTF-8", "PYTHONHASHSEED": "0"},
        )
    except subprocess.TimeoutExpired as exc:
        raise IsolatedRenderError("RENDER_TIMEOUT") from exc
    if completed.returncode:
        code = {
            2: "WORKER_CONFIG_ERROR",
            3: "WORKER_SVG_BYTES_LIMIT",
            4: "RENDER_FAILURE",
            5: "PNG_BYTES_LIMIT",
        }.get(completed.returncode, "WORKER_RESOURCE_OR_PROCESS_FAILURE")
        raise IsolatedRenderError(code)
    if len(completed.stdout) > limits.max_png_bytes:
        raise IsolatedRenderError("PNG_BYTES_LIMIT")
    try:
        with Image.open(io.BytesIO(completed.stdout)) as image:
            if image.format != "PNG" or image.size != (size, size):
                raise IsolatedRenderError("INVALID_RENDER_OUTPUT")
            rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except IsolatedRenderError:
        raise
    except Exception as exc:
        raise IsolatedRenderError("INVALID_RENDER_OUTPUT") from exc
    return completed.stdout, rgba
