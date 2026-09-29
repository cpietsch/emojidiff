"""The canvas latent (stage A, the variational transcriber) on tiny CPU models: encode
dispatch, the bottleneck's wiring and PCA initialisation, the KL and the beta
controller, one training step, checkpoints, the config's fixed factors; and, on a GPU,
graph decoding against the reference decoder from renders and from latent grids."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, replace
from pathlib import Path

import pytest
import torch
from torch.distributions import Normal, kl_divergence

import mojidiff.learning.render2svg as render2svg
from mojidiff.learning.autoregressive import SequenceLayout, flatten_program, legal_mask
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.pixel_latent import (
    NEW_WEIGHTS,
    PREFLIGHT_ICONS,
    BetaController,
    CanvasSettings,
    VariationalTranscriber,
    _sha256,
    check_preflight,
    checkpoint_state,
    gaussian_kl,
    hypothesis_channels,
    load_checkpoint,
    load_config,
    load_transcriber_weights,
    preflight_choice,
    set_projections,
    stem_features,
    stem_pca,
    train_and_evaluate,
    variational_loss,
)
from mojidiff.learning.render2svg import (
    ModelConfig,
    RenderToProgram,
    greedy_decode,
    path_major_order,
    render_trusted_rgb,
)
from mojidiff.representation.packed import serialize_packed_svg

_CONFIG = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
Fixture = tuple[SequenceLayout, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]


def _tiny(**changes: object) -> ModelConfig:
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
    return replace(config, **changes)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def icons() -> Fixture:
    """(layout, tokens, 32 px renders, legal masks, path-major orders) of 2 icons."""

    pilot = load_openmoji_pilot_config(_CONFIG)
    codec = _selected_codec(pilot)
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)
    by_split, _, _ = load_pilot_index(pilot)
    rows = _select_rows(by_split["primary/validation"], 2, pilot.seed + 1)
    programs = [_load_program(row, pilot, codec) for row in rows]
    tokens = torch.stack([flatten_program(p, layout) for p in programs])
    images = torch.stack(
        [
            torch.from_numpy(
                render_trusted_rgb(
                    serialize_packed_svg(p, codec, layout.total_segment_slots), 32
                ).copy()
            )
            for p in programs
        ]
    )
    masks = torch.stack(
        [torch.stack([legal_mask(p, row, layout) for p in range(layout.length)]) for row in tokens]
    )
    orders = torch.stack([path_major_order(row, layout) for row in tokens])
    return layout, tokens, images, masks, orders


def _noise_images(count: int, size: int = 32, seed: int = 3) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, (count, size, size, 3), dtype=torch.uint8, generator=generator)


def test_encode_dispatches_on_its_input_and_renders_read_the_posterior_mean(
    icons: Fixture,
) -> None:
    layout, _, images, _, _ = icons
    torch.manual_seed(0)
    model = VariationalTranscriber(layout, _tiny(), CanvasSettings(channels=4)).eval()
    assert model.latent_shape == (4, 4, 4) and model.memory_length == 16
    with torch.no_grad():
        mean, log_variance = model.posterior(images)
        assert mean.shape == log_variance.shape == (2, 4, 4, 4)
        assert mean.dtype == torch.float32
        # The log-variance head starts input-independent, at its bias.
        assert torch.equal(log_variance, torch.full_like(log_variance, -6.0))
        from_images = model.encode(images)
        from_mean = model.encode(mean)
        direct = model.memory_from_latent(mean)
    assert len(from_images) == len(from_mean) == len(model.decoder)
    for first, second, third in zip(from_images, from_mean, direct, strict=True):
        for a, b, c in zip(first, second, third, strict=True):
            assert torch.equal(a, b) and torch.equal(a, c)
    # A float NHWC render is neither a render (uint8) nor a latent grid.
    with pytest.raises(ValueError, match="encode takes"):
        model.encode(images.float())
    with pytest.raises(ValueError, match="encode takes"):
        model.encode(torch.zeros(2, 3, 4, 4))
    with pytest.raises(ValueError, match="expected latents"):
        model.memory_from_latent(torch.zeros(2, 4, 3, 3))


def test_identity_projections_give_exactly_the_transcriber(icons: Fixture) -> None:
    """With c = d_model and identity projections, VT is its parent transcriber: the
    bottleneck sits between the stem and the grid embedding and nowhere else."""

    layout, tokens, images, _, orders = icons
    config = _tiny()
    torch.manual_seed(0)
    parent = RenderToProgram(layout, config).eval()
    model = VariationalTranscriber(layout, config, CanvasSettings(channels=config.d_model)).eval()
    load_transcriber_weights(model, parent.state_dict())
    width = config.d_model
    with torch.no_grad():
        model.to_posterior.weight[:width].copy_(torch.eye(width))
        model.to_posterior.bias[:width].zero_()
        model.from_latent.weight.copy_(torch.eye(width))
        model.from_latent.bias.zero_()
        expected = parent(images, tokens, orders)
        got = model(images, tokens, orders)
        again = model(images, tokens, orders)
    torch.testing.assert_close(got, expected, rtol=0.0, atol=1e-5)
    assert torch.equal(got, again)  # the render path reads mu: deterministic at eval
    # Weights that do not fit are refused, not silently skipped.
    state = parent.state_dict()
    del state["stem.0.weight"]
    with pytest.raises(ValueError, match="stem.0.weight"):
        load_transcriber_weights(model, state)
    extra = {**parent.state_dict(), "surplus": torch.zeros(1)}
    with pytest.raises(ValueError, match="surplus"):
        load_transcriber_weights(model, extra)
    assert NEW_WEIGHTS == {name for name in model.state_dict() if name not in parent.state_dict()}


def test_pca_projections_are_the_whitened_rank_c_projection(icons: Fixture) -> None:
    layout, tokens, images, _, orders = icons
    config = _tiny()
    torch.manual_seed(0)
    parent = RenderToProgram(layout, config).eval()
    library = _noise_images(8)
    pca = stem_pca(parent, library, torch.device("cpu"), batch_size=3)
    assert pca.cells == 8 * 16 and pca.icons == 8
    assert bool((pca.variances[:-1] >= pca.variances[1:]).all())
    assert pca.explained(4) < pca.explained(8) < pca.explained(32) == pytest.approx(1.0)

    # Full rank: the projection is the identity, so VT is the parent again.
    full = VariationalTranscriber(layout, config, CanvasSettings(channels=config.d_model)).eval()
    load_transcriber_weights(full, parent.state_dict())
    set_projections(full, pca)
    with torch.no_grad():
        torch.testing.assert_close(
            full(images, tokens, orders), parent(images, tokens, orders), rtol=0.0, atol=1e-3
        )

    # Rank c: from_latent(mu) is the feature projected on the top c directions, and mu
    # is whitened over the cells the PCA saw.
    rank = 4
    model = VariationalTranscriber(layout, config, CanvasSettings(channels=rank)).eval()
    load_transcriber_weights(model, parent.state_dict())
    set_projections(model, pca)
    with torch.no_grad():
        features = stem_features(parent, library).reshape(-1, config.d_model)
        top = pca.directions[:, :rank]
        projected = pca.mean + (features - pca.mean) @ top @ top.T
        mean, log_variance = model.posterior(library)
        cells = mean.flatten(2).transpose(1, 2).reshape(-1, rank)
        torch.testing.assert_close(model.from_latent(cells), projected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(cells.mean(dim=0), torch.zeros(rank), rtol=0.0, atol=1e-4)
    torch.testing.assert_close(
        cells.var(dim=0, unbiased=False), torch.ones(rank), rtol=1e-3, atol=0
    )
    assert torch.equal(log_variance, torch.full_like(log_variance, -6.0))


def test_kl_is_the_closed_form_gaussian_kl_summed_per_icon() -> None:
    generator = torch.Generator().manual_seed(1)
    mean = torch.randn(3, 4, 5, 5, generator=generator)
    log_variance = torch.randn(3, 4, 5, 5, generator=generator)
    reference = kl_divergence(
        Normal(mean, (0.5 * log_variance).exp()), Normal(torch.zeros_like(mean), 1.0)
    )
    torch.testing.assert_close(gaussian_kl(mean, log_variance), reference.flatten(1).sum(dim=1))
    zeros = torch.zeros(2, 4, 3, 3)
    assert torch.equal(gaussian_kl(zeros, zeros), torch.zeros(2))
    # At initialisation (mu = 0, log-variance -6) every dimension costs 0.5 (e^-6 + 5).
    initial = gaussian_kl(zeros, torch.full_like(zeros, -6.0))
    torch.testing.assert_close(initial, torch.full((2,), 36 * 0.5 * (math.exp(-6.0) + 5.0)))


def test_the_beta_controller_moves_beta_toward_the_budget_and_stays_clipped() -> None:
    settings = CanvasSettings(budget_nats=100.0, beta_initial=0.01, kl_ema_decay=0.0)
    controller = BetaController.from_settings(settings)
    assert controller.beta == pytest.approx(0.01)
    # KL above budget: beta rises by exp(rate * (KL - C) / C).
    assert controller.update(300.0) == pytest.approx(0.01 * math.exp(0.01 * 2.0))
    before = controller.beta
    assert controller.update(50.0) < before  # below budget: falls
    before = controller.beta
    assert controller.update(100.0) == pytest.approx(before)  # at budget: holds
    for _ in range(5000):
        controller.update(1e6)
    assert controller.beta == pytest.approx(settings.beta_max) and not controller.at_floor
    for _ in range(5000):
        controller.update(0.0)
    assert controller.beta == pytest.approx(settings.beta_min) and controller.at_floor
    # The controller reads an exponential moving average of the batch KL.
    smoothed = BetaController.from_settings(replace(settings, kl_ema_decay=0.5))
    smoothed.update(300.0)
    smoothed.update(100.0)
    assert smoothed.kl_ema == pytest.approx(200.0)


def test_training_steps_lower_the_loss_and_the_loss_is_ce_plus_beta_kl_per_token(
    icons: Fixture,
) -> None:
    layout, tokens, images, masks, orders = icons
    torch.manual_seed(0)
    model = VariationalTranscriber(layout, _tiny(), CanvasSettings(channels=4))
    noise = torch.randn(2, *model.latent_shape, generator=torch.Generator().manual_seed(2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    beta = 0.1
    first = variational_loss(model, images, tokens, masks, orders, beta, bf16=False, noise=noise)
    free = masks.sum(dim=-1) > 1
    assert float(first.free_tokens) == pytest.approx(float(free.sum()) / 2)
    torch.testing.assert_close(
        first.loss, first.cross_entropy + beta * first.kl / first.free_tokens
    )
    losses = []
    for _ in range(3):
        parts = variational_loss(
            model, images, tokens, masks, orders, beta, bf16=False, noise=noise
        )
        optimizer.zero_grad()
        parts.loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(parts.loss.detach()))
    with torch.no_grad():
        last = variational_loss(model, images, tokens, masks, orders, beta, bf16=False, noise=noise)
    assert float(last.loss) < losses[0]
    # Gradients reached both new projections and the shared weights.
    assert model.to_posterior.weight.grad is not None
    assert float(model.to_posterior.weight.grad[4:].abs().sum()) > 0.0  # log-variance rows
    assert model.from_latent.weight.grad is not None
    assert next(model.stem.parameters()).grad is not None


def test_a_checkpoint_round_trips_to_the_same_model(icons: Fixture, tmp_path: Path) -> None:
    layout, tokens, images, _, orders = icons
    torch.manual_seed(0)
    model = VariationalTranscriber(layout, _tiny(), CanvasSettings(channels=4)).eval()
    controller = BetaController.from_settings(model.settings)
    controller.update(1234.0)
    torch.save(checkpoint_state(model, 7, controller), tmp_path / "best.pt")
    state = torch.load(tmp_path / "best.pt", map_location="cpu")
    assert state["step"] == 7 and state["canvas"] == asdict(model.settings)
    assert state["controller"]["kl_ema"] == pytest.approx(1234.0)
    loaded = load_checkpoint(state, layout)
    with torch.no_grad():
        assert torch.equal(loaded(images, tokens, orders), model(images, tokens, orders))
        latent = torch.randn(1, *model.latent_shape, generator=torch.Generator().manual_seed(4))
        assert torch.equal(greedy_decode(loaded, latent), greedy_decode(model, latent))


def test_vt_v1_holds_v9s_model_and_data_stream_fixed() -> None:
    """One factor against v9: the model block and the online stream are v9's."""

    config = load_config(Path("configs/latent/vt-v1.yaml"))
    v9 = render2svg.load_config(Path("configs/render2svg/full-v9-colour.yaml"))
    assert config.model == v9.model
    stream = (
        "batch_size", "augment_online", "augment_original", "loader_workers",
        "compose_probability", "compose_parts", "augment_mirror", "augment_max_shift",
        "augment_colour", "augment_seed", "bf16", "eval_icons", "eval_every", "weight_decay",
        "extra_training",
    )  # fmt: skip
    for name in stream:
        assert getattr(config.training, name) == getattr(v9.training, name), name
    training = config.training
    assert (training.steps, training.learning_rate, training.warmup_steps, training.seed) == (
        30000,
        3e-4,
        500,
        7001,
    )
    canvas = config.canvas
    assert (canvas.channels, canvas.budget_nats, canvas.log_variance_bias) == (8, 2000.0, -6.0)
    assert (canvas.beta_min, canvas.beta_max, canvas.beta_rate) == (1e-4, 1.0, 0.01)
    assert (canvas.band_low, canvas.band_high) == (0.8, 1.2)
    assert config.parent_checkpoint.parent.name == config.parent_run
    assert config.parent_checkpoint.name == "best.pt" and config.eval_icons == 339
    assert (config.non_inferiority_margin, config.sample_cost_margin) == (0.015, 0.010)
    # The launch is tied to the pre-flight rule, and the hypothesis names the config's c.
    assert config.preflight_report == Path("reports/latent/vt-v1-preflight.json")
    assert hypothesis_channels(config.hypothesis) == canvas.channels


def test_the_preflight_rule_chooses_8_or_16_and_declares_nothing_past_16() -> None:
    assert preflight_choice({8: 0.11, 16: 0.10})[0] == 8
    assert preflight_choice({8: 0.12, 16: 0.10})[0] == 8  # "above 0.12" switches
    assert preflight_choice({8: 0.13, 16: 0.12})[0] == 16
    chosen, why = preflight_choice({8: 0.156, 16: 0.125})
    assert chosen is None and "operator decides" in why
    assert preflight_choice({8: 0.13})[0] is None
    assert preflight_choice({16: 0.05})[0] is None
    assert hypothesis_channels("a latent (c = 16 channels); else c = 8") == 16
    assert hypothesis_channels("C = 2,000 nats and z = 0") is None


def test_a_real_run_needs_a_matching_preflight_report(tmp_path: Path) -> None:
    parent = tmp_path / "best.pt"
    parent.write_bytes(b"weights")
    config = replace(
        load_config(Path("configs/latent/vt-v1.yaml")),
        parent_checkpoint=parent,
        preflight_report=tmp_path / "preflight.json",
    )
    with pytest.raises(SystemExit, match="no pre-flight report"):
        check_preflight(config, "data", 2681)
    report = {
        "parent_checkpoint": str(parent),
        "parent_checkpoint_sha256": _sha256(parent),
        "dataset_sha256": "data",
        "icons": PREFLIGHT_ICONS,
        "pca": {"icons": 2681},
        "rank_8": {"pixel_error": [0.11, 0.1, 0.12]},
        "rank_16": {"pixel_error": [0.09, 0.08, 0.1]},
        "chosen_channels": 8,
        "rule_outcome": "rank-8 error 0.1100 <= 0.12: c = 8",
    }

    def check(**changes: object) -> dict[str, object]:
        (tmp_path / "preflight.json").write_text(json.dumps({**report, **changes}))
        return check_preflight(config, "data", 2681)

    kept = check()
    assert kept["chosen_channels"] == 8 and kept["rank_errors"] == {"rank_8": 0.11, "rank_16": 0.09}
    with pytest.raises(SystemExit, match="chose c = 16"):
        check(chosen_channels=16)
    with pytest.raises(SystemExit, match="no declared outcome"):
        check(chosen_channels=None, rule_outcome="both above 0.12")
    with pytest.raises(SystemExit, match="other data"):
        check(dataset_sha256="other")
    with pytest.raises(SystemExit, match="renders, not 2681"):
        check(pca={"icons": 64})
    with pytest.raises(SystemExit, match="icons, not the declared"):
        check(icons=4)
    with pytest.raises(SystemExit, match="checkpoint changed"):
        check(parent_checkpoint_sha256="0" * 64)
    # Switching to c = 16 needs the hypothesis to say so too.
    wider = replace(config, canvas=replace(config.canvas, channels=16))
    (tmp_path / "preflight.json").write_text(json.dumps({**report, "chosen_channels": 16}))
    with pytest.raises(SystemExit, match="hypothesis does not name c = 16"):
        check_preflight(wider, "data", 2681)


def test_a_real_run_refuses_to_start_with_uncommitted_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The run record names a commit and hashes `git diff HEAD`; an untracked module
    would be in neither, so a real run stops before it loads anything."""

    status = "?? src/mojidiff/learning/pixel_latent.py\n M configs/latent/vt-v1.yaml\n"
    monkeypatch.setattr(render2svg, "_git", lambda *args: status)

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("the corpus must not load")

    monkeypatch.setattr("mojidiff.learning.pixel_latent.load_corpus", refuse)
    with pytest.raises(SystemExit, match="commit first") as stopped:
        train_and_evaluate(Path("configs/latent/vt-v1.yaml"))
    assert "pixel_latent.py" in str(stopped.value) and "vt-v1.yaml" in str(stopped.value)


@cuda
def test_graph_and_greedy_decoding_agree_on_renders_and_on_latent_grids(
    icons: Fixture,
) -> None:
    from mojidiff.learning.fast_decode import GraphDecoder

    layout, _, _, _, _ = icons
    torch.manual_seed(0)
    model = (
        VariationalTranscriber(
            layout, _tiny(d_model=64, feedforward=128, decoder_layers=2), CanvasSettings(channels=4)
        )
        .cuda()
        .eval()
    )
    decoder = GraphDecoder(model, dtype=torch.float32)
    try:
        for seed in range(2):
            image = _noise_images(1, seed=seed)[0]
            reference = greedy_decode(model, image[None].cuda())
            assert torch.equal(decoder.decode(image), reference)
            with torch.no_grad():
                mean, _ = model.posterior(image[None].cuda())
            assert torch.equal(greedy_decode(model, mean), reference)
            latent = torch.randn(4, 4, 4, generator=torch.Generator().manual_seed(seed)).cuda()
            assert torch.equal(decoder.decode(latent), greedy_decode(model, latent[None]))
    finally:
        del decoder, model
        torch.cuda.empty_cache()


@cuda
def test_the_canvas_backend_decodes_in_ieee_float32_and_restores_the_flags(
    icons: Fixture,
) -> None:
    from mojidiff.gallery.latent_backends import CanvasBackend
    from mojidiff.learning.precision import float32_flags

    layout, _, _, _, _ = icons
    torch.manual_seed(0)
    model = VariationalTranscriber(layout, _tiny(), CanvasSettings(channels=4)).cuda().eval()
    before = float32_flags()
    backend = CanvasBackend(model)
    try:
        latents = torch.randn(4, 4, 4, 4, generator=torch.Generator().manual_seed(1)).cuda()
        with torch.no_grad():
            batched = backend.decode(latents)
            single = torch.stack([backend.decode_one(latent) for latent in latents])
        assert torch.equal(batched, single)
        assert float32_flags() == before
    finally:
        del backend, model
        torch.cuda.empty_cache()


def test_the_c16_fallback_needs_a_completed_c8_run_that_failed_a1(tmp_path: Path) -> None:
    import yaml

    from mojidiff.learning.pixel_latent import _check_fallback

    report_path = tmp_path / "preflight.json"
    report_path.write_text(json.dumps({"chosen_channels": 8}))
    runs = tmp_path / "runs"
    config = replace(
        load_config(Path("configs/latent/vt-v1.yaml")),
        preflight_report=report_path,
        channel_fallback_from="vt-x",
    )

    def record(**changes: object) -> None:
        base: dict[str, object] = {
            "state": "completed",
            "config_resolved": {"canvas": {"channels": 8}},
            "initialisation": {
                "preflight": {"report": str(report_path), "report_sha256": _sha256(report_path)}
            },
            "result": {"criteria": {"A1": {"pass": False}}},
        }
        base.update(changes)
        (runs / "vt-x").mkdir(parents=True, exist_ok=True)
        (runs / "vt-x" / "run.yaml").write_text(yaml.safe_dump(base))

    assert _check_fallback(config, report_path, 8, runs)[1] == "fallback run vt-x has no run record"
    record()
    kept, failure = _check_fallback(config, report_path, 8, runs)
    assert failure is None and kept == {"run": "vt-x", "failed": "A1", "run_channels": 8}
    record(result={"criteria": {"A1": {"pass": True}}})
    assert "did not fail A1" in str(_check_fallback(config, report_path, 8, runs)[1])
    record(state="failed")
    assert "not completed" in str(_check_fallback(config, report_path, 8, runs)[1])
    record(config_resolved={"canvas": {"channels": 16}})
    assert "not the pre-flight's first choice" in str(
        _check_fallback(config, report_path, 8, runs)[1]
    )
    record()
    assert "not the pre-flight's first choice" in str(
        _check_fallback(config, report_path, 16, runs)[1]
    )
    record()
    report_path.write_text(json.dumps({"chosen_channels": 8, "edited": True}))
    assert "changed" in str(_check_fallback(config, report_path, 8, runs)[1])
