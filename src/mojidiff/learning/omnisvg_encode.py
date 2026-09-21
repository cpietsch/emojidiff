"""OpenMoji icons in OmniSVG's own token language, for fine-tuning it.

OmniSVG's training repository encodes an SVG by loading it with its vendored deepsvg,
normalising it into a 200-unit box, and emitting one token per command, one per grid
point and one per 12-bit fill colour. Its inference repository decodes with a second,
slightly different vendored deepsvg, and the two do not share a process. This module
uses only the inference copy - the one the released checkpoint is decoded with - and
re-implements the encoder's arithmetic from the training repository's tokenizer and
colour scheme, so that an OpenMoji icon encoded here decodes back through OmniSVG's own
path to the same drawing. A test pins that round trip.

OpenMoji's strokes are outlined first with the project's picosvg fallback, because
OmniSVG draws fills only.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mojidiff.learning.omnisvg import EXTERNAL, _stub_moviepy
from mojidiff.learning.openmoji_pilot import OpenMojiPilotError, PilotRow, _outline

BASE_VOCABULARY = 151_936
# The training repository's YAML lists these one lower; its config adds `num_svg_end`
# (1) to each, and the released decoder reads them at these values: a move at 151,938
# decodes as nothing and a close at 151,942 decodes as an arc, which is how the first
# round trip produced arcs from icons that had none.
CMD_MOVE, CMD_LINE, CMD_CURVE, CMD_ARC, CMD_CLOSE = 151_939, 151_940, 151_941, 151_942, 151_943
PIX_PAD = 151_944
COLOR_START = BASE_VOCABULARY + 40_011  # 191_947: none, then currentColor, then 4,096 colours
BBOX = 200
BOS, EOS = 196_998, 196_999


def _deepsvg() -> Any:
    _stub_moviepy()
    if str(EXTERNAL) not in sys.path:
        sys.path.insert(0, str(EXTERNAL))
    from deepsvg.svglib.geom import Bbox  # type: ignore[import-not-found]
    from deepsvg.svglib.svg import SVG  # type: ignore[import-not-found]

    return SVG, Bbox


def openmoji_to_deepsvg(source: bytes) -> tuple[Any, list[str]]:
    """Outline the icon's strokes, then load and normalise it into OmniSVG's 200 box.

    Returns the deepsvg document and each path's fill colour in document order, read
    from the outlined text because the inference deepsvg does not keep fill colours.
    """

    SVG, Bbox = _deepsvg()
    outlined = _outline(source).decode()
    fills = _fills(outlined)
    svg = SVG.from_str(outlined)
    if len(svg.svg_path_groups) != len(fills):
        raise ValueError(
            f"{len(svg.svg_path_groups)} path groups against {len(fills)} fills in the outline"
        )
    svg.normalize(Bbox(BBOX, BBOX))
    svg.simplify_arcs()
    return svg, fills


_PATH = re.compile(r"<path\b[^>]*>", re.IGNORECASE)
_FILL = re.compile(r'\bfill="([^"]*)"')


def _fills(svg_text: str) -> list[str]:
    fills = []
    for element in _PATH.findall(svg_text):
        match = _FILL.search(element)
        fills.append(match.group(1) if match else "#000000")  # SVG's default fill is black
    return fills


def color_token(fill: Any) -> int:
    """The training repository's colour scheme: 12-bit RGB, two special values first."""

    if fill is None:
        return COLOR_START
    text = str(fill).strip().lower()
    if text in {"", "none"}:
        return COLOR_START
    if text == "currentcolor":
        return COLOR_START + 1
    if text.startswith("url("):
        return COLOR_START + 4_097
    if text.startswith("#") and len(text) == 4:
        text = "#" + "".join(c * 2 for c in text[1:])
    if text.startswith("#") and len(text) == 7:
        r, g, b = int(text[1:3], 16), int(text[3:5], 16), int(text[5:7], 16)
    else:
        r, g, b = 0, 0, 0
    return COLOR_START + 2 + (((r >> 4) << 8) | ((g >> 4) << 4) | (b >> 4))


def _point(coordinate: np.ndarray) -> int:
    x, y = int(coordinate[0]), int(coordinate[1])
    return x + y * BBOX + PIX_PAD


def encode(svg: Any, fills: list[str]) -> np.ndarray:
    """Drawing tokens for a deepsvg SVG, path by path, each path ending in its colour."""

    tokens: list[int] = []
    for group, fill in zip(svg.svg_path_groups, fills, strict=True):
        tensor = group.to_tensor(PAD_VAL=0).round().int().clip(0, BBOX - 1)
        for index, row in enumerate(tensor):
            command = int(row[0])
            start, control1, control2, end = (
                row[6:8].numpy(),
                row[8:10].numpy(),
                row[10:12].numpy(),
                row[12:14].numpy(),
            )
            if command == 0:
                tokens.append(CMD_MOVE)
                tokens.append(_point(end if index == 0 else start))
                tokens.append(_point(end))
            elif command == 1:
                tokens.extend((CMD_LINE, _point(end)))
            elif command == 2:
                tokens.extend((CMD_CURVE, _point(control1), _point(control2), _point(end)))
            elif command == 6:
                tokens.extend((CMD_CLOSE, _point(end)))
            else:
                raise ValueError(f"unsupported deepsvg command {command}; arcs must be simplified")
        tokens.append(color_token(fill))
    return np.array(tokens, dtype=np.int64)


def sequence(svg: Any, fills: list[str]) -> torch.Tensor:
    """BOS, the drawing tokens, EOS - the labels of one training example."""

    return torch.tensor(np.concatenate(([BOS], encode(svg, fills), [EOS])), dtype=torch.long)


def encode_icon(source: bytes) -> torch.Tensor:
    """One OpenMoji SVG as OmniSVG's drawing sequence, BOS and EOS included."""

    svg, fills = openmoji_to_deepsvg(source)
    return sequence(svg, fills)


def encode_rows(
    rows: tuple[PilotRow, ...], raw_root: Path, cache_root: Path
) -> tuple[dict[str, torch.Tensor], dict[str, str]]:
    """Every row's drawing sequence, by hexcode, with the reason for each icon that fails.

    Outlining and re-encoding a few thousand icons takes minutes, so the result is
    cached under a key made of the rows' source hashes and this module's constants;
    a change to either produces a fresh encoding rather than a stale one.
    """

    key = hashlib.sha256(
        json.dumps(
            {
                "rows": [row.source_svg_sha256 for row in rows],
                "tokens": [CMD_MOVE, CMD_LINE, CMD_CURVE, CMD_ARC, CMD_CLOSE, PIX_PAD, COLOR_START],
                "box": BBOX,
                "version": 1,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]
    cache = cache_root / f"omnisvg-tokens-{key}.pt"
    if cache.is_file():
        payload = torch.load(cache, weights_only=True)
        return dict(payload["tokens"]), dict(payload["failures"])
    tokens: dict[str, torch.Tensor] = {}
    failures: dict[str, str] = {}
    for row in rows:
        source = (raw_root / row.source_path).read_bytes()
        if hashlib.sha256(source).hexdigest() != row.source_svg_sha256:
            raise OpenMojiPilotError(f"source hash mismatch: {row.source_path}")
        try:
            tokens[row.hexcode] = encode_icon(source)
        except Exception as error:  # noqa: BLE001 - counted, never silently dropped
            failures[row.hexcode] = f"{type(error).__name__}: {str(error)[:80]}"
    cache_root.mkdir(parents=True, exist_ok=True)
    torch.save({"tokens": tokens, "failures": failures}, cache)
    return tokens, failures
