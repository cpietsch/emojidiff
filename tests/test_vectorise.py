"""The re-vectorise demo's contract: a canvas PNG in, a valid codec SVG out."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from mojidiff.learning.autoregressive import SequenceLayout
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.render2svg import ModelConfig, RenderToProgram
from mojidiff.representation.renderer import RenderLimits, validate_typed_svg
from mojidiff.vectorise.server import Vectoriser, canvas_to_rgb, held_out_icons

_CONFIG = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")


def _png(rgba: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buffer, format="PNG")
    return buffer.getvalue()


def test_canvas_png_becomes_rgb_on_white_at_model_size() -> None:
    transparent = np.zeros((64, 64, 4), dtype=np.uint8)
    rgb = canvas_to_rgb(_png(transparent), 32)
    assert rgb.shape == (32, 32, 3)
    assert int(rgb.min()) == 255


def test_vectorise_returns_a_valid_typed_svg() -> None:
    pilot = load_openmoji_pilot_config(_CONFIG)
    codec = _selected_codec(pilot)
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)
    by_split, _, _ = load_pilot_index(pilot)
    template = _load_program(by_split["primary/train"][0], pilot, codec)
    torch.manual_seed(0)
    model = RenderToProgram(
        layout,
        ModelConfig(
            image_size=32, d_model=32, heads=4, encoder_layers=1, decoder_layers=1, feedforward=64
        ),
    )
    vectoriser = Vectoriser(model, template, torch.device("cpu"))
    result = vectoriser.vectorise(np.full((32, 32, 3), 255, dtype=np.uint8))
    assert result["ok"] and result["decoder_calls"] > 0 and result["candidates"] == 1
    validate_typed_svg(result["svg"].encode(), RenderLimits(max_paths=80))
    best = vectoriser.vectorise(np.full((32, 32, 3), 255, dtype=np.uint8), candidates=8)
    assert best["ok"] and best["candidates"] == 8
    validate_typed_svg(best["svg"].encode(), RenderLimits(max_paths=80))


def test_the_demo_offers_only_held_out_icons() -> None:
    icons = held_out_icons()
    assert icons and all(icon["split"] != "primary/train" for icon in icons)
