"""The latent model: trains, and every point of its latent space decodes to a program."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from mojidiff.learning.autoregressive import SequenceLayout, flatten_program, legal_mask
from mojidiff.learning.latent import LatentSettings, LatentToProgram, kl_per_dimension
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.render2svg import ModelConfig, greedy_decode, path_major_order
from mojidiff.representation.packed import validate_packed_tensor_program

_CONFIG = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")


def test_latent_model_trains_and_decodes_valid_programs() -> None:
    from mojidiff.learning.autoregressive import unflatten_program

    pilot = load_openmoji_pilot_config(_CONFIG)
    codec = _selected_codec(pilot)
    by_split, _, _ = load_pilot_index(pilot)
    rows = _select_rows(by_split["primary/validation"], 2, pilot.seed + 1)
    programs = [_load_program(row, pilot, codec) for row in rows]
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)
    torch.manual_seed(0)
    config = ModelConfig(
        image_size=32,
        d_model=32,
        heads=4,
        encoder_layers=1,
        decoder_layers=2,
        feedforward=64,
        metric=True,
        fourier=6,
        order="path",
    )
    settings = LatentSettings(latent_dim=8, memory_tokens=4, encoder_layers=1)
    model = LatentToProgram(layout, config, settings)
    assert not hasattr(model, "stem")
    tokens = torch.stack([flatten_program(p, layout) for p in programs])
    order = torch.stack([path_major_order(row, layout) for row in tokens])
    masks = torch.stack(
        [torch.stack([legal_mask(p, row, layout) for p in range(layout.length)]) for row in tokens]
    )
    free = masks.sum(dim=-1) > 1
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    for _ in range(4):
        mean, log_variance = model.posterior(tokens)
        latent = mean + torch.randn_like(mean) * (0.5 * log_variance).exp()
        logits = model.reconstruct_logits(tokens, order, latent, 0.25)
        loss = (
            F.cross_entropy(logits.masked_fill(~masks, float("-inf"))[free], tokens[free])
            + 0.01 * kl_per_dimension(mean, log_variance).sum(-1).mean()
        )
        optimizer.zero_grad()
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(loss.detach()))
    assert losses[-1] < losses[0]
    model.eval()
    decoded = greedy_decode(model, torch.randn(2, settings.latent_dim))
    for row in decoded:
        validate_packed_tensor_program(
            unflatten_program(row, programs[0], layout), codec, layout.total_segment_slots
        )
