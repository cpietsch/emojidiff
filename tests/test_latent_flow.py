"""The canvas latent, stage B (the rectified-flow prior) on tiny CPU models: the DiT's
shapes and zero start, the flow's convention and Euler integration against exact
velocity fields, a toy 2-D mixture learnt and inverted, standardisation, the latent
precompute and its cache, checkpoints, the config's declared run, the refusals, and the
gallery's canvas-flow backend."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml

import mojidiff.learning.latent_flow as latent_flow
from mojidiff.gallery.latent_backends import BACKENDS, CanvasFlowBackend, LatentBackend
from mojidiff.gallery.server import Gallery, GalleryError, load_model, load_registry
from mojidiff.learning.autoregressive import SequenceLayout, flatten_program, unflatten_program
from mojidiff.learning.latent_flow import (
    ChannelMoments,
    FlowNetworkConfig,
    FlowTrainConfig,
    FlowTrainer,
    FlowTransformer,
    FlowValidation,
    LatentFlowPrior,
    PrecomputeSettings,
    SmokeOptions,
    check_parent,
    checkpoint_state,
    euler_generate,
    euler_invert,
    fixed_pairs,
    flow_matching_loss,
    gaussian_velocity,
    load_config,
    load_flow_checkpoint,
    load_flow_parent,
    load_or_precompute,
    noised,
    patchify,
    sincos_2d,
    slerp,
    smoke_config,
    train_and_evaluate,
    trajectory_indices,
    unpatchify,
)
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.pixel_latent import (
    CanvasSettings,
    VariationalTranscriber,
    _sha256,
)
from mojidiff.learning.pixel_latent import checkpoint_state as vt_checkpoint_state
from mojidiff.learning.render2svg import (
    ModelConfig,
    SplitData,
    TrainConfig,
    _augment_chunk,
    greedy_decode,
    render_trusted_rgb,
)
from mojidiff.representation.packed import serialize_packed_svg

_PILOT = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")
_CONFIG = Path("configs/latent/lfp-v1.yaml")
CENTRES = torch.tensor([[1.5, 1.5], [1.5, -1.5], [-1.5, 1.5], [-1.5, -1.5]])


def _toy_network(**changes: Any) -> FlowTransformer:
    """A DiT over (2, 1, 1) latents: one token holding a 2-D point."""

    config = FlowNetworkConfig(
        channels=2, grid=1, patch=1, d_model=64, layers=2, heads=4, frequency_dim=64
    )
    return FlowTransformer(replace(config, **changes))


def _mixture(count: int, generator: torch.Generator) -> torch.Tensor:
    which = torch.randint(len(CENTRES), (count,), generator=generator)
    points = CENTRES[which] + 0.2 * torch.randn(count, 2, generator=generator)
    return points.view(count, 2, 1, 1)


@pytest.fixture(scope="module")
def toy_prior() -> LatentFlowPrior:
    """The flow trained on a 4-component 2-D Gaussian mixture, through the real trainer."""

    torch.manual_seed(0)
    network = _toy_network()
    config = FlowTrainConfig(
        steps=1000, batch_size=256, learning_rate=1e-3, warmup_steps=50, ema_decay=0.99, bf16=False
    )
    trainer = FlowTrainer(network, config)
    generator = torch.Generator().manual_seed(1)
    for _ in range(config.steps):
        trainer.step(_mixture(config.batch_size, generator), generator)
    return LatentFlowPrior(trainer.ema, torch.zeros(2), torch.ones(2), steps=100)


# --------------------------------------------------------------------------- network


def test_patches_round_trip_and_the_untrained_network_predicts_zero_velocity() -> None:
    x = torch.randn(3, 8, 18, 18, generator=torch.Generator().manual_seed(0))
    tokens = patchify(x, 2)
    assert tokens.shape == (3, 81, 32)
    # Token 0 is the top-left 2 x 2 patch, channel-major.
    assert torch.equal(tokens[0, 0], x[0, :, :2, :2].reshape(-1))
    assert torch.equal(unpatchify(tokens, 8, 2, 9), x)
    positions = sincos_2d(256, 9)
    assert positions.shape == (81, 256) and len({tuple(row.tolist()) for row in positions}) == 81

    network = FlowTransformer(FlowNetworkConfig())
    assert network.latent_shape == (8, 18, 18)
    # The declared size (configs/latent/lfp-v1.yaml): DiT, d 256, 8 blocks, MLP ratio 4.
    assert sum(p.numel() for p in network.parameters()) == 9_747_744
    with torch.no_grad():
        out = network(x, torch.rand(3))
    assert out.shape == x.shape and torch.equal(out, torch.zeros_like(x))  # adaLN-zero
    with pytest.raises(ValueError, match="does not divide"):
        FlowTransformer(FlowNetworkConfig(grid=17))


# --------------------------------------------------------------------------- flow


def test_the_interpolant_and_loss_follow_the_declared_convention() -> None:
    """x_t = (1 - t) x_0 + t eps: t = 0 is data, t = 1 noise; the target is eps - x_0."""

    x0 = torch.randn(4, 2, 1, 1, generator=torch.Generator().manual_seed(0))
    eps = torch.randn(4, 2, 1, 1, generator=torch.Generator().manual_seed(1))
    assert torch.equal(noised(x0, eps, torch.zeros(4)), x0)
    assert torch.equal(noised(x0, eps, torch.ones(4)), eps)
    t = torch.tensor([0.1, 0.4, 0.6, 0.9])
    exact = flow_matching_loss(lambda x, _: eps - x0, x0, t, eps)
    assert float(exact) == 0.0
    assert math.isclose(float(flow_matching_loss(lambda x, _: torch.zeros_like(x), x0, t, eps)),
                        float(((eps - x0) ** 2).mean()), rel_tol=1e-6)  # fmt: skip


def _gaussian_field(mean: float, spread: float) -> latent_flow.Velocity:
    """The exact rectified-flow velocity for data N(mean, spread^2) per dimension."""

    def velocity(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        s = t.view(-1, *([1] * (x.dim() - 1)))
        sigma = ((1 - s) ** 2 * spread**2 + s**2).sqrt()
        derivative = (s - (1 - s) * spread**2) / sigma
        return -mean + derivative * (x - (1 - s) * mean) / sigma

    return velocity


def test_euler_integration_of_exact_velocity_fields_recovers_their_targets() -> None:
    noise = torch.randn(
        256, 3, 2, 2, generator=torch.Generator().manual_seed(0), dtype=torch.float64
    )
    # A point mass at c: v = (x - c) / t; the paths are straight, so Euler is exact.
    centre = torch.full((1, 3, 2, 2), 0.7, dtype=torch.float64)
    for steps in (1, 7, 50):
        landed, _ = euler_generate(lambda x, t: (x - centre) / t.view(-1, 1, 1, 1), noise, steps)
        assert torch.allclose(landed, centre.expand_as(landed), atol=1e-9)
    # N(m, s^2): the exact map is x_0 = m + s x_1; Euler converges to it at first order.
    field = _gaussian_field(0.5, 0.3)
    errors = []
    for steps in (10, 40, 160, 640):
        landed, _ = euler_generate(field, noise, steps)
        errors.append(float((landed - (0.5 + 0.3 * noise)).abs().max()))
    ratios = [coarse / fine for coarse, fine in zip(errors, errors[1:], strict=False)]
    assert all(3.5 < ratio < 4.5 for ratio in ratios) and errors[-1] < 5e-3, errors
    # Inversion runs the same ODE the other way: the data point back to its noise.
    back = euler_invert(field, 0.5 + 0.3 * noise, 640)
    assert float((back - noise).abs().max()) < 0.02
    # For N(0, I) data the exact flow is the identity map, and `gaussian_velocity` is it.
    t = torch.rand(256, dtype=torch.float64)
    assert torch.allclose(gaussian_velocity(noise, t), _gaussian_field(0.0, 1.0)(noise, t))
    landed, recorded = euler_generate(gaussian_velocity, noise, 50, record_at=[0, 49])
    # Linear, so every Euler step scales x by one number: x_0 = k x_1 with k near 1.
    scale = landed / noise
    assert torch.allclose(scale, scale.flatten()[0].expand_as(scale), atol=1e-9)
    assert abs(float(scale.flatten()[0]) - 1.0) < 0.05
    # x_hat_0 = x_t - t v: at t = 1 the prediction of a zero-mean Gaussian is its mean.
    assert [round(t, 6) for t, _ in recorded] == [1.0, 0.02]
    assert torch.allclose(recorded[0][1], torch.zeros_like(noise), atol=1e-12)


def test_rectified_flow_learns_to_sample_a_two_dimensional_gaussian_mixture(
    toy_prior: LatentFlowPrior,
) -> None:
    noise = torch.randn(2000, 2, 1, 1, generator=torch.Generator().manual_seed(2))
    samples, _ = toy_prior.generate(noise)
    points = samples.view(-1, 2)
    distance, which = torch.cdist(points, CENTRES).min(dim=1)
    assert float((distance < 0.6).float().mean()) > 0.9
    shares = torch.bincount(which, minlength=4).float() / len(points)
    assert bool(((shares > 0.18) & (shares < 0.32)).all()), shares
    for component in range(4):
        members = points[(which == component) & (distance < 0.6)]
        assert float((members.mean(dim=0) - CENTRES[component]).abs().max()) < 0.1
        assert bool(((members.std(dim=0) > 0.12) & (members.std(dim=0) < 0.3)).all())
    # The same check fails for the untrained network (v = 0 leaves the noise as it is).
    untrained = LatentFlowPrior(_toy_network(), torch.zeros(2), torch.ones(2), steps=100)
    raw, _ = untrained.generate(noise)
    assert float((torch.cdist(raw.view(-1, 2), CENTRES).min(dim=1)[0] < 0.6).float().mean()) < 0.2


def test_inversion_then_forward_integration_returns_close_to_the_start(
    toy_prior: LatentFlowPrior,
) -> None:
    points = _mixture(64, torch.Generator().manual_seed(5))
    noise = toy_prior.invert(points)
    # The inverted noise looks like noise, not like the data.
    assert abs(float(noise.mean())) < 0.3 and 0.6 < float(noise.std()) < 1.4
    back, _ = toy_prior.generate(noise)
    error = (back - points).flatten(1).norm(dim=1)
    assert float(error.mean()) < 0.03 and float(error.max()) < 0.2
    # Slerp through the noise: the endpoints come back, and every frame is a prior sample.
    path = toy_prior.slerp_path(points[0], points[1], 7)
    assert path.shape == (7, 2, 1, 1)
    assert float((path[0] - points[0]).norm()) < 0.2 and float((path[-1] - points[1]).norm()) < 0.2
    distance = torch.cdist(path.view(-1, 2), CENTRES).min(dim=1)[0]
    assert float((distance < 0.6).float().mean()) >= 5 / 7


def test_slerp_keeps_its_endpoints_and_the_norm_of_equal_norm_inputs() -> None:
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(8, 18, 18, generator=generator)
    b = torch.randn(8, 18, 18, generator=generator)
    b = b * a.norm() / b.norm()
    frames = slerp(a, b, torch.linspace(0, 1, 9))
    assert frames.shape == (9, 8, 18, 18)
    assert torch.allclose(frames[0], a, atol=1e-5) and torch.allclose(frames[-1], b, atol=1e-5)
    norms = frames.flatten(1).norm(dim=1)
    assert torch.allclose(norms, torch.full_like(norms, float(a.norm())), rtol=1e-5)
    # Parallel inputs fall back to the straight line.
    line = slerp(a, 2 * a, torch.tensor([0.0, 0.5, 1.0]))
    assert torch.allclose(line[1], 1.5 * a, atol=1e-5)


def test_the_trainer_keeps_an_exponential_moving_average_of_the_weights() -> None:
    torch.manual_seed(0)
    network = _toy_network()
    trainer = FlowTrainer(network, FlowTrainConfig(warmup_steps=1, ema_decay=0.9, bf16=False))
    before = [p.detach().clone() for p in network.parameters()]
    loss, norm = trainer.step(_mixture(32, torch.Generator().manual_seed(0)))
    assert loss.shape == () and float(norm) > 0.0 and trainer.steps == 1
    for average, old, new in zip(
        trainer.ema.parameters(), before, network.parameters(), strict=True
    ):
        assert torch.allclose(average, 0.9 * old + 0.1 * new.detach(), atol=1e-7)
        assert not average.requires_grad


def test_the_validation_flow_loss_uses_fixed_draws_and_the_gaussian_velocity_beats_zero() -> None:
    x0 = torch.randn(512, 2, 3, 3, generator=torch.Generator().manual_seed(0))
    first = FlowValidation.draw(x0, 4, 41, time_mean=0.0, time_std=1.0)
    second = FlowValidation.draw(x0, 4, 41, time_mean=0.0, time_std=1.0)
    assert len(first.t) == 2048 and torch.equal(first.x_t, second.x_t)
    gaussian = first.loss(gaussian_velocity, batch=100)
    assert gaussian == second.loss(gaussian_velocity, batch=100)
    assert math.isclose(gaussian, second.loss(gaussian_velocity), rel_tol=1e-9)  # any batch
    zero = first.loss(lambda x, _: torch.zeros_like(x))
    assert math.isclose(zero, 2.0, rel_tol=0.03)  # E|eps - x_0|^2 per number
    # For N(0, I) data the Gaussian velocity is the conditional mean, so its loss is the
    # irreducible 2 - (2t - 1)^2 / ((1 - t)^2 + t^2), averaged over the drawn t.
    t = first.t.double()
    floor = float((2.0 - (2 * t - 1) ** 2 / ((1 - t) ** 2 + t**2)).mean())
    assert gaussian < zero and math.isclose(gaussian, floor, rel_tol=0.03)


def test_trajectory_indices_spread_over_one_trajectory_before_its_last_step(
    toy_prior: LatentFlowPrior,
) -> None:
    assert trajectory_indices(50, 8) == [0, 7, 14, 21, 27, 34, 41, 48]
    assert trajectory_indices(3, 8) == [0, 1]
    # At the last step (t = 1 / steps), x_hat_0 = x_t - t v is the last Euler update
    # exactly: recording it would show the sample twice.
    noise = torch.randn(4, 2, 1, 1, generator=torch.Generator().manual_seed(3))
    final, frames = euler_generate(toy_prior.velocity, noise, 10, record_at=[9])
    assert torch.equal(frames[0][1], final)
    final, frames = euler_generate(toy_prior.velocity, noise, 10, trajectory_indices(10, 8))
    assert [round(t, 6) for t, _ in frames][-1] == 0.2
    assert not torch.equal(frames[-1][1], final)


# --------------------------------------------------------------------------- standardisation


def test_standardisation_round_trips_and_the_moments_are_the_aggregate_posterior() -> None:
    generator = torch.Generator().manual_seed(0)
    means = 3.0 + 2.0 * torch.randn(40, 4, 5, 5, generator=generator)
    sigmas = 0.5 * torch.rand(40, 4, 5, 5, generator=generator)
    moments = ChannelMoments(4)
    for chunk in (slice(0, 7), slice(7, 30), slice(30, 40)):
        moments.add(means[chunk], sigmas[chunk])
    mean, std = moments.result()
    per_channel = means.transpose(0, 1).reshape(4, -1).double()
    spread = sigmas.transpose(0, 1).reshape(4, -1).double()
    assert torch.allclose(mean.double(), per_channel.mean(dim=1), atol=1e-6)
    variance = per_channel.var(dim=1, unbiased=False) + (spread**2).mean(dim=1)
    assert torch.allclose(std.double(), variance.sqrt(), atol=1e-5)

    network = FlowTransformer(
        FlowNetworkConfig(channels=4, grid=4, d_model=32, layers=1, heads=4, frequency_dim=32)
    )
    prior = LatentFlowPrior(network, mean, std)
    z = means[:, :, :4, :4]
    x = prior.standardise(z)
    assert torch.allclose(prior.unstandardise(x), z, atol=1e-5)
    assert torch.allclose(prior.standardise(prior.unstandardise(x)), x, atol=1e-5)
    # Standardised posterior draws have zero mean and unit variance per channel.
    draws = prior.standardise(means + sigmas * torch.randn(means.shape, generator=generator))
    assert float(draws.mean(dim=(0, 2, 3)).abs().max()) < 0.1
    assert float((draws.transpose(0, 1).reshape(4, -1).std(dim=1) - 1).abs().max()) < 0.1


# --------------------------------------------------------------------------- latents


def _tiny(**changes: Any) -> ModelConfig:
    config = ModelConfig(
        image_size=32,
        d_model=32,
        heads=4,
        encoder_layers=1,
        decoder_layers=1,
        feedforward=64,
        metric=True,
        fourier=6,
        order="path",
    )
    return replace(config, **changes)


def _split(rows: Sequence[Any], pilot: Any, layout: SequenceLayout, size: int) -> SplitData:
    codec = layout.codec
    programs = [_load_program(row, pilot, codec) for row in rows]
    tokens = torch.stack([flatten_program(p, layout) for p in programs])

    def render(program: Any, pixels: int) -> torch.Tensor:
        svg = serialize_packed_svg(program, codec, layout.total_segment_slots)
        return torch.from_numpy(render_trusted_rgb(svg, pixels).copy())

    return SplitData(
        hexcodes=[row.hexcode for row in rows],
        tokens=tokens,
        masks=np.zeros((len(rows), 1), dtype=np.uint8),
        images=torch.stack([render(p, size) for p in programs]),
        targets=torch.stack([render(p, 72) for p in programs]),
        shape=(layout.length, layout.vocabulary),
    )


@pytest.fixture(scope="module")
def corpus() -> tuple[SequenceLayout, Any, SplitData, SplitData]:
    """(layout, template, 3 training icons, 2 validation icons), rendered at 32 px."""

    pilot = load_openmoji_pilot_config(_PILOT)
    codec = _selected_codec(pilot)
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)
    by_split, _, _ = load_pilot_index(pilot)
    template = _load_program(by_split["primary/train"][0], pilot, codec)
    train = _split(_select_rows(by_split["primary/train"], 3, pilot.seed + 1), pilot, layout, 32)
    validation = _split(
        _select_rows(by_split["primary/validation"], 2, pilot.seed + 1), pilot, layout, 32
    )
    return layout, template, train, validation


def test_the_precompute_encodes_originals_and_exact_variants_and_caches_them(
    corpus: tuple[SequenceLayout, Any, SplitData, SplitData], tmp_path: Path
) -> None:
    layout, template, train, validation = corpus
    torch.manual_seed(0)
    vt = VariationalTranscriber(layout, _tiny(), CanvasSettings(channels=2)).eval()
    settings = PrecomputeSettings(variants=3, seed=5, validation_icons=2, workers=1)
    cpu = torch.device("cpu")

    def load(given: PrecomputeSettings) -> tuple[latent_flow.LatentSet, Path, bool]:
        return load_or_precompute(
            vt, train, validation, layout, template, given, cpu,
            cache_root=tmp_path, vt_run_id="vt-toy", vt_sha256="ab" * 32, dataset_sha256="cd" * 32,
        )  # fmt: skip

    latents, path, created = load(settings)
    assert created and path.parent == tmp_path / "latent-flow"
    assert path.name.startswith("vt-toy-") and path.suffix == ".pt"
    count = len(latents.train_mean)
    assert count == 9 - latents.dropped_variants and latents.dropped_variants == 0
    assert latents.train_mean.shape == latents.train_sigma.shape == (9, 2, 4, 4)
    assert latents.train_mean.dtype == torch.float16
    assert sorted(latents.train_icon.tolist()) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert sorted(latents.train_variant.tolist()) == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert latents.validation_mean.shape == (2, 2, 4, 4)
    assert latents.validation_hexcodes == validation.hexcodes

    # Variant 0 of each icon is the icon itself; the others are render2svg's exact
    # variants under the same seed, re-rendered and encoded here.
    with torch.no_grad():
        original, original_log_variance = vt.posterior(train.images)
        augment = TrainConfig(
            augment_variants=2, augment_mirror=0.5, augment_max_shift=48, augment_colour=0.1
        )
        tokens = train.tokens.numpy().astype(np.int16)
        index, _, images = _augment_chunk(
            (np.arange(3), tokens, layout, template, 32, augment, 5 * 100_003)
        )
        variant, log_variance = vt.posterior(torch.from_numpy(images))
        held_out, _ = vt.posterior(validation.images)
    firsts = latents.train_variant == 0
    assert torch.allclose(latents.train_mean[firsts].float(), original, atol=2e-3, rtol=1e-3)
    assert torch.allclose(latents.train_mean[~firsts].float(), variant, atol=2e-3, rtol=1e-3)
    assert latents.train_icon[~firsts].tolist() == index.tolist()
    sigma = (0.5 * log_variance).exp()
    assert torch.allclose(latents.train_sigma[~firsts].float(), sigma, atol=2e-3, rtol=1e-3)
    assert torch.allclose(latents.validation_mean.float(), held_out, atol=2e-3, rtol=1e-3)
    moments = ChannelMoments(2)
    original_sigma = (0.5 * original_log_variance).exp()
    moments.add(torch.cat([original, variant]), torch.cat([original_sigma, sigma]))
    expected_mean, expected_std = moments.result()
    assert torch.allclose(latents.channel_mean, expected_mean, atol=1e-5)
    assert torch.allclose(latents.channel_std, expected_std, atol=1e-5)

    # A second call reads the cache; the rendering processes do not change the result;
    # other settings get another file; a file holding other settings is refused.
    again, same_path, created_again = load(settings)
    assert same_path == path and not created_again
    assert torch.equal(again.train_mean, latents.train_mean)
    pooled, pooled_path, _ = load(replace(settings, workers=2))
    assert pooled_path == path
    other, other_path, _ = load(replace(settings, variants=2, originals=False))
    assert (
        other_path != path and len(other.train_mean) == 6 and bool((other.train_variant > 0).all())
    )
    state = torch.load(other_path, map_location="cpu")
    state["settings"]["seed"] = 6
    torch.save(state, other_path)
    with pytest.raises(ValueError, match="holds other settings"):
        load(replace(settings, variants=2, originals=False))


def test_fixed_pairs_are_the_harness_pairs() -> None:
    from mojidiff.learning.latent_metrics import fixed_pairs as harness_pairs

    assert fixed_pairs(339, 32) == harness_pairs(339, 32, 17)


# --------------------------------------------------------------------------- checkpoints


def _vt_file(layout: SequenceLayout, path: Path, image_size: int = 144) -> VariationalTranscriber:
    torch.manual_seed(0)
    vt = VariationalTranscriber(layout, _tiny(image_size=image_size), CanvasSettings(channels=2))
    torch.save(vt_checkpoint_state(vt.eval(), 5), path)
    return vt


def _flow_file(vt_path: Path, path: Path, seed: int = 1) -> LatentFlowPrior:
    """A tiny DiT over (2, 18, 18) with every weight perturbed off its zero start."""

    torch.manual_seed(seed)
    network = FlowTransformer(
        FlowNetworkConfig(channels=2, grid=18, d_model=32, layers=1, heads=4, frequency_dim=32)
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
    prior = LatentFlowPrior(network, torch.tensor([0.5, -0.2]), torch.tensor([1.5, 0.8]), steps=6)
    parent = {"run": "vt-toy", "checkpoint": str(vt_path), "sha256": _sha256(vt_path)}
    torch.save(
        checkpoint_state(network, prior, step=3, parent=parent, validation_flow_loss=0.5), path
    )
    return prior.eval()


def test_a_flow_checkpoint_round_trips_and_names_its_frozen_parent(
    corpus: tuple[SequenceLayout, Any, SplitData, SplitData], tmp_path: Path
) -> None:
    layout = corpus[0]
    vt = _vt_file(layout, tmp_path / "vt.pt")
    prior = _flow_file(tmp_path / "vt.pt", tmp_path / "best.pt")
    state = torch.load(tmp_path / "best.pt", map_location="cpu")
    loaded = load_flow_checkpoint(state)
    assert loaded.steps == 6 and loaded.latent_shape == (2, 18, 18)
    noise = torch.randn(2, 2, 18, 18, generator=torch.Generator().manual_seed(0))
    assert torch.equal(loaded.generate(noise)[0], prior.generate(noise)[0])
    assert not torch.equal(loaded.generate(noise)[0], prior.unstandardise(noise))
    parent = load_flow_parent(state, layout)
    for name, value in vt.state_dict().items():
        assert torch.equal(parent.state_dict()[name], value)
    with pytest.raises(ValueError, match="not a latent flow checkpoint"):
        load_flow_checkpoint({**state, "kind": "vae"})
    wrong = {**state, "network": {**state["network"], "channels": 4}}
    with pytest.raises(ValueError, match="latent shape"):
        load_flow_parent(wrong, layout)
    _vt_file(layout, tmp_path / "vt.pt", image_size=32)  # the parent file changes
    with pytest.raises(ValueError, match="changed since"):
        load_flow_parent(state, layout)


# --------------------------------------------------------------------------- config and run


def test_lfp_v1_declares_the_planned_run() -> None:
    config = load_config(_CONFIG)
    assert config.slug == "lfp-v1"
    assert config.parent_run == "vt-v1-74dfbd2-46380052-47646604"
    assert config.parent_checkpoint == Path(
        "/home/dev/.cache/mojidiff/runs/vt-v1-74dfbd2-46380052-47646604/best.pt"
    )
    assert config.parent_gate == ("A1", "A3")
    training = config.training
    assert (training.steps, training.batch_size, training.learning_rate, training.seed) == (
        40_000, 256, 3e-4, 7101,
    )  # fmt: skip
    assert training.ema_decay == 0.999 and training.bf16 and training.posterior_sampling
    assert (training.time_mean, training.time_std) == (0.0, 1.0)
    latents = config.latents
    assert (latents.variants, latents.mirror, latents.max_shift, latents.colour) == (
        32,
        0.5,
        48,
        0.1,
    )
    assert latents.train_icons is None and latents.validation_icons == 339
    network = config.network
    assert (network.patch, network.d_model, network.layers, network.heads) == (2, 256, 8, 8)
    assert config.sampling.steps == 50 and config.sampling.trajectory_frames == 8
    assert (config.prior_samples, config.interpolation_pairs, config.trajectories) == (64, 8, 4)
    for name in ("(B1)", "(B2)", "(B3)", "(B4)", "(B5)"):
        assert name in config.hypothesis
    for claim in ("at most 10%", "at least 2x", "all 339 samples distinct", "at most 15%",
                  "at most 20%", "without-twins"):  # fmt: skip
        assert claim in config.hypothesis
    # B4 and B5 name the harness report that scores the slerp, not the default (lerp) one.
    assert "--interpolation backend" in config.hypothesis
    assert "slerp-through-prior-noise" in config.hypothesis
    for name in ("B4", "B5"):
        assert "--interpolation backend" in latent_flow.CRITERIA_SCORED_BY[name]
    for name in ("B1", "B2", "B3"):
        assert "default settings" in latent_flow.CRITERIA_SCORED_BY[name]
    assert set(latent_flow.CRITERIA_SCORED_BY) == set(latent_flow.CRITERIA)


def test_the_config_refuses_unknown_fields_and_a_latent_shape(tmp_path: Path) -> None:
    root = yaml.safe_load(_CONFIG.read_text())
    path = tmp_path / "flow.yaml"
    path.write_text(yaml.safe_dump({**root, "compose_probability": 0.5}))
    with pytest.raises(ValueError, match="unknown latent flow config fields: compose"):
        load_config(path)
    path.write_text(yaml.safe_dump({**root, "network": {**root["network"], "channels": 16}}))
    with pytest.raises(ValueError, match="come from the parent VT"):
        load_config(path)
    path.write_text(yaml.safe_dump({**root, "training": {**root["training"], "schedule": "step"}}))
    with pytest.raises(ValueError, match="not constant or cosine"):
        load_config(path)


def test_the_smoke_config_shrinks_the_run() -> None:
    options = SmokeOptions(parent_checkpoint=Path("/tmp/vt.pt"))
    config = smoke_config(load_config(_CONFIG), options)
    assert config.parent_checkpoint == Path("/tmp/vt.pt")
    assert (config.latents.variants, config.latents.train_icons) == (2, 32)
    assert config.network.layers == 2 and config.network.d_model == 256
    assert (config.training.steps, config.training.batch_size) == (50, 8)
    assert (config.prior_samples, config.interpolation_pairs, config.trajectories) == (4, 1, 1)


def test_a_real_run_refuses_to_start_with_uncommitted_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    status = ["?? src/mojidiff/learning/latent_flow.py", " M configs/latent/lfp-v1.yaml"]
    monkeypatch.setattr(latent_flow, "uncommitted_sources", lambda: status)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("nothing may load")

    monkeypatch.setattr(latent_flow, "load_corpus", refuse)
    monkeypatch.setattr(latent_flow, "check_parent", refuse)
    with pytest.raises(SystemExit, match="commit first") as stopped:
        train_and_evaluate(_CONFIG)
    assert "latent_flow.py" in str(stopped.value) and "lfp-v1.yaml" in str(stopped.value)
    with pytest.raises(SystemExit):
        latent_flow.main(["--config", str(_CONFIG), "--steps", "5"])  # smoke-only flag


def test_a_real_run_needs_the_parent_to_have_passed_its_gate(tmp_path: Path) -> None:
    config = load_config(_CONFIG)
    record_path = tmp_path / config.parent_run / "run.yaml"
    record_path.parent.mkdir()

    def write(state: str, a1: bool, a3: bool, checkpoints: str) -> None:
        criteria = {"A1": {"pass": a1}, "A2": {"pass": False}, "A3": {"pass": a3}}
        record = {
            "state": state,
            "outputs": {"checkpoints": checkpoints},
            "result": {"criteria": criteria},
        }
        record_path.write_text(yaml.safe_dump(record))

    with pytest.raises(SystemExit, match="no parent run record"):
        check_parent(config, tmp_path / "elsewhere")
    right = str(config.parent_checkpoint.parent)
    write("running", True, True, right)
    with pytest.raises(SystemExit, match="'running', not completed"):
        check_parent(config, tmp_path)
    write("completed", True, False, right)
    with pytest.raises(SystemExit, match="A3: pass is False"):
        check_parent(config, tmp_path)
    write("completed", True, True, "/somewhere/else")
    with pytest.raises(SystemExit, match="not the config's"):
        check_parent(config, tmp_path)
    write("completed", True, True, right)
    gate = check_parent(config, tmp_path)
    assert gate["gate"] == {"A1": True, "A3": True}  # A2 is not in the gate
    assert asdict(config.latents)["originals"] is True


# --------------------------------------------------------------------------- gallery


def test_the_canvas_flow_backend_serves_the_flow_prior_through_the_gallery(
    corpus: tuple[SequenceLayout, Any, SplitData, SplitData],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flow checkpoint loads through `canvas-flow` by a registry entry alone, with the
    VT its checkpoint names: samples are the flow's from seeded noise (the same seed, the
    same drawings), interpolation is the backend's slerp through noise, reconstruction
    is the VT's, and every answer is what the models' own greedy decoder answers."""

    layout, template, _, validation = corpus
    cpu = torch.device("cpu")
    vt = _vt_file(layout, tmp_path / "vt.pt").eval()
    prior = _flow_file(tmp_path / "vt.pt", tmp_path / "best.pt")
    entry = {
        "id": "f1",
        "run_id": "toy-flow",
        "kind": "latent",
        "backend": "canvas-flow",
        "label": "toy flow",
        "description": "a toy flow prior",
    }
    registry = tmp_path / "models.yaml"
    registry.write_text(yaml.safe_dump({"schema_version": 1, "models": [entry]}))
    (spec,) = load_registry(registry)
    backend, extras = load_model(spec, layout, cpu, checkpoint=tmp_path / "best.pt")
    assert isinstance(backend, CanvasFlowBackend) and isinstance(backend, LatentBackend)
    assert backend.latent_shape == (2, 18, 18) and backend.canvas.graph is None
    assert extras["step"] == 3 and extras["parent_run"] == "vt-toy" and "model" not in extras
    assert "canvas-flow" in BACKENDS
    programs = dict(zip(validation.hexcodes, validation.tokens, strict=True))
    served = Gallery(
        [(spec, cast(LatentBackend, backend))],
        template,
        programs,
        cpu,
        icons=[{"hexcode": code} for code in programs],
        runs_root=tmp_path,
    )
    first, second = sorted(programs)

    def svg(tokens: torch.Tensor) -> str:
        program = unflatten_program(tokens, template, layout)
        return serialize_packed_svg(program, layout.codec, layout.total_segment_slots).decode()

    # Samples: the flow from 0.5 * seeded noise on the CPU; the same seed, the same draws.
    drawn = backend.sample(3, torch.Generator().manual_seed(5), 0.5)
    assert drawn.shape == (3, 2, 18, 18) and drawn.device.type == "cpu"
    assert torch.equal(drawn, backend.sample(3, torch.Generator().manual_seed(5), 0.5))
    assert not torch.equal(drawn, backend.sample(3, torch.Generator().manual_seed(6), 0.5))
    noise = 0.5 * torch.randn(3, 2, 18, 18, generator=torch.Generator().manual_seed(5))
    assert torch.equal(drawn, prior.generate(noise)[0])
    with torch.no_grad():
        expected = [svg(row) for row in greedy_decode(vt, drawn)]
    answered = served.sample("f1", 3, seed=5, scale=0.5)["samples"]
    assert answered == expected == served.sample("f1", 3, seed=5, scale=0.5)["samples"]

    # Interpolation: the server takes the backend's path, slerp through the noise.
    images = torch.stack(
        [
            torch.from_numpy(render_trusted_rgb(svg(programs[code]).encode(), 144).copy())
            for code in (first, second)
        ]
    )
    with torch.no_grad():
        means, _ = vt.posterior(images)
        path = prior.slerp_path(means[0], means[1], 4)
        frames = [svg(row) for row in greedy_decode(vt, path)]
        reconstruction = svg(greedy_decode(vt, images[:1])[0])
    assert torch.equal(backend.interpolate(means[0], means[1], 4), path)
    calls: list[int] = []
    own = backend.interpolate

    def spy(a: torch.Tensor, b: torch.Tensor, steps: int) -> torch.Tensor:
        calls.append(steps)
        return own(a, b, steps)

    monkeypatch.setattr(backend, "interpolate", spy)
    answer = served.interpolate("f1", first, second, 4)
    assert answer["frames"] == frames and calls == [4]
    assert answer["path"] == "slerp-through-prior-noise"
    # The straight line between the same posterior means, on request: the VT's lerp.
    with torch.no_grad():
        weights = torch.linspace(0.0, 1.0, 4).view(-1, 1, 1, 1)
        lerp = [
            svg(row) for row in greedy_decode(vt, (1 - weights) * means[0] + weights * means[1])
        ]
    straight = served.interpolate("f1", first, second, 4, path="lerp")
    assert straight["frames"] == lerp and straight["path"] == "lerp" and calls == [4]
    with pytest.raises(GalleryError, match="path must be one of backend, lerp"):
        served.interpolate("f1", first, second, 4, path="slerp")
    monkeypatch.setattr(backend, "interpolate", lambda a, b, steps: torch.zeros(steps, 2))
    with pytest.raises(RuntimeError, match=r"'canvas-flow' of 'f1': interpolate returned"):
        served.interpolate("f1", first, second, 4)
    # Reconstruction is the VT's own, from the posterior mean.
    assert served.reconstruct("f1", first)["svg"] == reconstruction
    # Watch it draw: z_hat_0 at spread times, then the sample of the same noise.
    trajectory = backend.trajectory(torch.Generator().manual_seed(7), 1.0, frames=4)
    start = torch.randn(1, 2, 18, 18, generator=torch.Generator().manual_seed(7))
    final, recorded = prior.generate(start, record_at=trajectory_indices(6, 4))
    assert trajectory.shape == (5, 2, 18, 18) and torch.equal(trajectory[-1], final[0])
    assert torch.equal(trajectory[0], recorded[0][1][0])
    assert not torch.equal(trajectory[-2], trajectory[-1])  # the sample is shown once
    # The served sampler is the VT and the flow: the gallery counts both.
    assert backend.parts == (backend.model, backend.prior)
    counted = sum(p.numel() for part in backend.parts for p in part.parameters())
    assert counted == sum(p.numel() for p in vt.parameters()) + sum(
        p.numel() for p in prior.parameters()
    )
