"""The render-to-SVG model: exact fast decoding, exact mirroring, a working step.

Pinned here: decoding that skips grammar-forced positions produces exactly the tokens a
position-by-position full-prefix decode would; every decoded program validates; the
token-space mirror is a true pixel mirror; and one training step runs and lowers loss.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from mojidiff.learning.autoregressive import (
    SequenceLayout,
    flatten_program,
    legal_mask,
    unflatten_program,
)
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.render2svg import (
    DecodeStats,
    ModelConfig,
    RenderToProgram,
    greedy_decode,
    mirror_program_tokens,
    pixel_error,
    render_program_rgb,
)
from mojidiff.representation.packed import PackedTensorProgram, validate_packed_tensor_program

_CONFIG = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")
_TINY = ModelConfig(
    image_size=32, d_model=32, heads=4, encoder_layers=1, decoder_layers=2, feedforward=64
)

Pieces = tuple[SequenceLayout, list[PackedTensorProgram]]


@pytest.fixture(scope="module")
def pieces() -> Pieces:
    config = load_openmoji_pilot_config(_CONFIG)
    codec = _selected_codec(config)
    by_split, _, _ = load_pilot_index(config)
    rows = _select_rows(by_split["primary/validation"], 3, config.seed + 1)
    programs = [_load_program(row, config, codec) for row in rows]
    layout = SequenceLayout(codec=codec, total_segment_slots=config.total_segment_slots)
    return layout, programs


def _reference_decode(model: RenderToProgram, images: torch.Tensor) -> torch.Tensor:
    """Position by position, re-reading the whole prefix, no cache and no skipping."""

    layout = model.layout
    memory = model.encode(images)
    decoded = torch.zeros((1, layout.length), dtype=torch.long)
    for position in range(layout.length):
        mask = legal_mask(position, decoded[0], layout)
        if int(mask.sum()) == 1:
            decoded[0, position] = int(mask.to(torch.long).argmax())
            continue
        shifted = torch.cat((torch.zeros((1, 1), dtype=torch.long), decoded[:, :position]), 1)
        logits, _ = model.decode(shifted, memory)
        decoded[0, position] = int(logits[0, -1].masked_fill(~mask, float("-inf")).argmax())
    return decoded


def test_skipping_forced_positions_is_exact(pieces: Pieces) -> None:
    layout, programs = pieces
    torch.manual_seed(0)
    model = RenderToProgram(layout, _TINY).eval()
    images = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.uint8)
    stats = DecodeStats()
    fast = greedy_decode(model, images, stats=stats)
    assert torch.equal(fast, _reference_decode(model, images))
    assert stats.model_calls < layout.length
    validate_packed_tensor_program(
        unflatten_program(fast[0], programs[0], layout), layout.codec, layout.total_segment_slots
    )


def test_batched_decode_matches_single_decodes(pieces: Pieces) -> None:
    layout, _ = pieces
    torch.manual_seed(1)
    model = RenderToProgram(layout, _TINY).eval()
    images = torch.randint(0, 256, (3, 32, 32, 3), dtype=torch.uint8)
    batched = greedy_decode(model, images)
    for index in range(3):
        assert torch.equal(batched[index], greedy_decode(model, images[index : index + 1])[0])


def test_mirror_is_an_exact_pixel_mirror(pieces: Pieces) -> None:
    layout, programs = pieces
    for program in programs:
        tokens = flatten_program(program, layout)
        mirrored = mirror_program_tokens(tokens, layout)
        assert mirrored is not None
        back = mirror_program_tokens(mirrored, layout)
        assert back is not None and torch.equal(back, tokens)
        candidate = unflatten_program(mirrored, program, layout)
        validate_packed_tensor_program(candidate, layout.codec, layout.total_segment_slots)
        original = render_program_rgb(program, layout, 72)
        flipped = render_program_rgb(candidate, layout, 72)
        assert original is not None and flipped is not None
        # Antialiasing is symmetric on a pixel grid aligned to the box, so the mirror
        # is exact up to rasteriser rounding.
        assert pixel_error(flipped, np.ascontiguousarray(original[:, ::-1])) < 0.002
        assert pixel_error(flipped, original) > pixel_error(
            flipped, np.ascontiguousarray(original[:, ::-1])
        )


def test_a_training_step_lowers_the_loss(pieces: Pieces) -> None:
    layout, programs = pieces
    torch.manual_seed(2)
    model = RenderToProgram(layout, _TINY)
    tokens = flatten_program(programs[0], layout)[None]
    masks = torch.stack([legal_mask(p, tokens[0], layout) for p in range(layout.length)])[None]
    images = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.uint8)
    free = masks.sum(dim=-1) > 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(5):
        logits = model(images, tokens)
        loss = F.cross_entropy(logits.masked_fill(~masks, float("-inf"))[free], tokens[free])
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
