"""The text prior's contract: compact SVG round-trips, the chunked loss equals the whole loss."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from mojidiff.learning.omnisvg import parse_into_codec
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.prior import (
    SVG_CLOSE,
    chunked_cross_entropy,
    compact_svg,
    compact_text,
    control_prefix,
    extract_svg,
    training_text,
)

_PILOT = Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")


def test_compact_svg_round_trips_through_the_codec() -> None:
    pilot = load_openmoji_pilot_config(_PILOT)
    codec = _selected_codec(pilot)
    by_split, _, _ = load_pilot_index(pilot)
    for row in _select_rows(by_split["primary/validation"], 3, 5):
        program = _load_program(row, pilot, codec)
        text = compact_svg(program, codec, pilot.total_segment_slots)
        assert ".0 " not in text and "  " not in text and text.endswith(SVG_CLOSE)
        restored, info = parse_into_codec(text.encode(), codec, pilot.total_segment_slots)
        assert restored is not None, info
        assert np.array_equal(restored.path_length, program.path_length)
        assert np.array_equal(restored.coordinates, program.coordinates)
        assert np.array_equal(restored.fill, program.fill) and np.array_equal(
            restored.stroke_width, program.stroke_width
        )


def test_compact_text_keeps_fractional_coordinates() -> None:
    assert compact_text('<path d="M 17.0 59.5 L 3.25 4.0"  />') == '<path d="M 17 59.5 L 3.25 4"/>'


def test_prompt_pieces_and_extraction() -> None:
    assert training_text("fish", "<svg/>").startswith("<!-- fish -->\n<svg")
    prefix = control_prefix("fish")
    assert prefix.endswith('viewBox="0 0 72 72">')
    body = '<path d="M 1 1 L 2 2"/></svg> trailing text <svg>again'
    assert extract_svg(prefix + body) == prefix.split("\n", 1)[1] + '<path d="M 1 1 L 2 2"/></svg>'
    assert extract_svg("no tag here") is None
    assert extract_svg("<svg>never closed") is None


def test_chunked_cross_entropy_matches_the_whole_sequence() -> None:
    torch.manual_seed(0)
    width, vocabulary, length = 16, 50, 37
    head = torch.nn.Linear(width, vocabulary)
    hidden = torch.randn(1, length, width, requires_grad=True)
    labels = torch.randint(0, vocabulary, (1, length))
    labels[0, :5] = -100  # a caption prefix carries no loss
    logits = head(hidden[:, :-1]).float()
    whole = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocabulary),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
        reduction="sum",
    )
    for chunk in (7, 16, 100):
        total, count = chunked_cross_entropy(hidden, head, labels, chunk)
        assert count == length - 5
        assert torch.isclose(total, whole, rtol=1e-5), chunk
    total, _ = chunked_cross_entropy(hidden, head, labels, 8)
    total.backward()
    assert hidden.grad is not None and bool(hidden.grad.abs().sum() > 0)
