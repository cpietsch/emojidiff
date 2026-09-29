"""The latent metrics harness: each measure on inputs with known answers, and one run of
the whole scorer on a tiny CPU latent model and a tiny transcriber."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import mojidiff.learning.latent_metrics as latent_metrics
from mojidiff.gallery.latent_backends import CanvasBackend, CanvasFlowBackend, VaeBackend
from mojidiff.learning.autoregressive import PATH_STRIDE, SEGMENT_STRIDE, SequenceLayout
from mojidiff.learning.autoregressive import flatten_program as flatten
from mojidiff.learning.latent import LatentSettings, LatentToProgram
from mojidiff.learning.latent_flow import FlowNetworkConfig, FlowTransformer, LatentFlowPrior
from mojidiff.learning.latent_metrics import (
    BLANK,
    TWIN_ERROR,
    Reference,
    RefitPrior,
    Settings,
    aligned_error,
    build_reference,
    collapsed_share,
    crossfade_frames,
    density_coverage,
    extent_ratios,
    fixed_pairs,
    frechet_distance,
    ink,
    jump_share,
    mean_pairwise_distance,
    path_extents,
    precision_recall,
    report_name,
    score_backend,
    score_crossfade,
    score_interpolation,
    settings_hash,
)
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.pixel_latent import CanvasSettings, VariationalTranscriber
from mojidiff.learning.render2svg import (
    ModelConfig,
    RenderToProgram,
    SplitData,
    _programs_to_renders,
    greedy_decode,
    pixel_error,
    render_program_rgb,
)
from mojidiff.representation.packed import PackedTensorProgram

_CONFIG = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")
_TINY = ModelConfig(
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
_SETTINGS = Settings(samples=3, pairs=1, frames=3, icons=2, batch=4, shortlist=4, latency_repeats=1)


def _unit(rows: list[list[float]]) -> torch.Tensor:
    return F.normalize(torch.tensor(rows, dtype=torch.float32), dim=-1)


def _embed(renders: Sequence[np.ndarray]) -> torch.Tensor:
    """A cheap stand-in for CLIP: 6 px ink thumbnails plus a constant, unit length."""

    if not renders:
        return torch.zeros(0, 145)
    grey = torch.from_numpy(np.stack(renders)).float().mean(dim=-1)[:, None]
    darkness = 1.0 - F.avg_pool2d(grey, 6).flatten(1) / 255.0
    return F.normalize(torch.cat((darkness, torch.ones(len(renders), 1)), dim=1), dim=-1)


# --------------------------------------------------------------------------- measures


def test_ink_counts_pixels_whose_darkest_channel_is_below_250() -> None:
    assert ink(BLANK) == 0.0
    assert ink(None) == 0.0
    assert ink(np.zeros((72, 72, 3), dtype=np.uint8)) == 1.0
    half = BLANK.copy()
    half[:36, :, 2] = 249  # one dark channel is enough
    half[36:40, :, :] = 250  # exactly 250 is not ink
    assert ink(half) == 0.5


def test_path_extents_read_start_points_and_endpoints_only() -> None:
    layout = _layout()
    tokens = torch.zeros(layout.length, dtype=torch.long)
    start = PATH_STRIDE - 2

    def segment(slot: int, kind: int, coordinates: list[int]) -> None:
        at = layout.path_positions + slot * SEGMENT_STRIDE
        tokens[at] = kind
        tokens[at + 1 : at + 1 + len(coordinates)] = torch.tensor(coordinates)

    tokens[0], tokens[start], tokens[start + 1] = 2, 5, 5  # path 0: two segments from (5, 5)
    segment(0, 1, [9, 5])  # LINE to (9, 5): x spans 4 quarter units
    segment(1, 3, [400, 400, 1, 1, 5, 7])  # CUBIC: controls ignored, endpoint (5, 7)
    tokens[PATH_STRIDE] = 1  # path 1: one QUAD from (20, 20) to (21, 20)
    tokens[PATH_STRIDE + start], tokens[PATH_STRIDE + start + 1] = 20, 20
    segment(2, 2, [300, 300, 21, 20])
    assert path_extents(tokens, layout) == [1.0, 0.25]
    assert collapsed_share(tokens[None], layout) == 0.5
    assert extent_ratios(tokens[None], tokens[None], layout) == [1.0]
    assert extent_ratios(torch.zeros_like(tokens)[None], tokens[None], layout) == [0.0]


def test_knn_measures_on_identical_and_disjoint_sets() -> None:
    real = _unit([[1, 0, 0, 0], [0.9, 0.1, 0, 0], [0.8, 0.3, 0, 0], [0.7, 0.2, 0.1, 0]])
    precision, recall = precision_recall(real, real.clone(), k=3)
    assert precision == [1.0] * 4 and recall == [1.0] * 4
    _, coverage = density_coverage(real, real.clone(), k=5)
    assert coverage == [1.0] * 4
    far = _unit([[0, 0, 0, 1], [0, 0, 0.1, 1], [0, 0, 0.2, 1]])
    precision, recall = precision_recall(real, far, k=3)
    assert precision == [0.0] * 3 and recall == [0.0] * 4
    density, coverage = density_coverage(real, far, k=5)
    assert density == [0.0] * 3 and coverage == [0.0] * 4
    assert mean_pairwise_distance(real[:1].expand(5, -1)) == pytest.approx(0.0, abs=1e-6)


def test_frechet_distance_is_zero_to_itself_and_the_squared_shift_when_moved() -> None:
    features = torch.randn(64, 6, generator=torch.Generator().manual_seed(0))
    assert frechet_distance(features, features) == pytest.approx(0.0, abs=1e-6)
    shift = torch.tensor([3.0, 0.0, 4.0, 0.0, 0.0, 0.0])
    assert frechet_distance(features, features + shift) == pytest.approx(25.0, abs=1e-6)
    assert frechet_distance(features[:1], features) is None


def test_jump_share_of_even_static_and_cut_strips() -> None:
    angles = torch.linspace(0.0, 1.2, 5)
    even = torch.stack((angles.cos(), angles.sin()), dim=1)
    assert jump_share(even) == pytest.approx(0.25, abs=1e-4)
    # A strip that never moves (a decoder ignoring its latent) is not perfectly even.
    assert jump_share(even[:1].expand(5, -1)) is None
    cut = torch.cat((even[:1].expand(3, -1), even[-1:].expand(2, -1)))
    assert jump_share(cut) == pytest.approx(1.0)


def test_aligned_error_finds_a_moved_copy_and_mirrors_are_candidates() -> None:
    icon = BLANK.copy()
    icon[20:40, 10:30] = (200, 30, 30)
    moved = np.full_like(icon, 255)
    moved[28:48, 22:42] = icon[20:40, 10:30]  # 8 down, 12 right
    other = BLANK.copy()
    other[50:70, 50:70] = 0
    assert pixel_error(moved, icon) > 0.0
    error, index = aligned_error(moved, np.stack((other, icon)))
    assert error == 0.0 and index == 1
    error, _ = aligned_error(moved, np.stack((other,)))
    assert error > 0.0


def test_crossfade_frames_run_from_one_render_to_the_other_exactly() -> None:
    first = torch.zeros(4, 4, 3, dtype=torch.uint8)
    second = torch.full((4, 4, 3), 200, dtype=torch.uint8)
    frames = crossfade_frames(first, second, 5)
    assert torch.equal(frames[0], first) and torch.equal(frames[-1], second)
    assert int(frames[2, 0, 0, 0]) == 100


def test_fixed_pairs_are_latent_py_sheet_pairs_and_distinct() -> None:
    rng = np.random.default_rng(17)
    sheet = [tuple(int(v) for v in rng.choice(339, 2, replace=False)) for _ in range(8)]
    pairs = fixed_pairs(339, 32, 17)
    assert pairs[:8] == sheet and all(a != b for a, b in pairs)


def test_refit_prior_matches_the_moments_it_was_fitted_to() -> None:
    generator = torch.Generator().manual_seed(1)
    latents = torch.randn(500, 2, 3, generator=generator) * 2.0 + 1.0
    prior = RefitPrior(latents)
    flat = latents.reshape(500, -1).double()
    assert torch.allclose(prior.mean, flat.mean(dim=0))
    assert torch.allclose(prior.covariance, torch.from_numpy(np.cov(flat.numpy().T)))
    first = prior.sample(4000, torch.Generator().manual_seed(23))
    again = prior.sample(4000, torch.Generator().manual_seed(23))
    assert first.shape == (4000, 2, 3) and torch.equal(first, again)
    assert torch.allclose(first.reshape(4000, -1).mean(dim=0).double(), prior.mean, atol=0.15)
    wide = RefitPrior(torch.randn(5, 12, generator=generator))  # more dimensions than points
    assert wide.sample(3, torch.Generator().manual_seed(0)).isfinite().all()


# --------------------------------------------------------------------------- end to end


def _layout() -> SequenceLayout:
    pilot = load_openmoji_pilot_config(_CONFIG)
    return SequenceLayout(
        codec=_selected_codec(pilot), total_segment_slots=pilot.total_segment_slots
    )


def _split(name: str, count: int) -> tuple[SplitData, PackedTensorProgram]:
    pilot = load_openmoji_pilot_config(_CONFIG)
    codec = _selected_codec(pilot)
    layout = _layout()
    by_split, _, _ = load_pilot_index(pilot)
    rows = _select_rows(by_split[name], count, pilot.seed + 1)
    programs = [_load_program(row, pilot, codec) for row in rows]

    def render(size: int) -> torch.Tensor:
        images = [render_program_rgb(program, layout, size) for program in programs]
        return torch.from_numpy(np.stack([image for image in images if image is not None]))

    split = SplitData(
        [row.hexcode for row in rows],
        torch.stack([flatten(program, layout) for program in programs]),
        np.zeros((count, 1), dtype=np.uint8),
        render(32),
        render(72),
        (1, 1),
    )
    assert len(split.targets) == count
    return split, programs[0]


@pytest.fixture(scope="module")
def reference() -> Reference:
    validation, _ = _split("primary/validation", 4)
    train, template = _split("primary/train", 5)
    return build_reference(
        _layout(),
        template,
        validation,
        train,
        _embed,
        torch.device("cpu"),
        shortlist=4,
    )


def test_calibration_holds_the_validation_icons_to_about_five_percent_copies(
    reference: Reference,
) -> None:
    calibration = reference.calibration
    assert calibration["validation_icons"] == 4 and calibration["library_renders"] == 10
    targets = [ink(t) for t in reference.validation.targets.numpy()]
    assert min(targets) <= calibration["fragment_ink"] <= max(targets)
    assert 0.0 <= calibration["copy_error"] and calibration["copy_cosine"] <= 1.0 + 1e-6
    assert calibration["copy_rule"] == "declared" and calibration["twins"] == []
    rules = calibration["copy_rules"]
    assert rules["declared"] == rules["without-twins"]  # no twins: the rules agree
    assert rules["declared"]["copy_error"] == calibration["copy_error"]
    # The reconstruction baseline searches mirrors too, never worse than without them.
    for mirrored, plain in zip(
        reference.validation_nearest_error,
        reference.validation_nearest_error_without_mirrors,
        strict=True,
    ):
        assert mirrored <= plain + 1e-9


def test_a_validation_twin_of_a_mirrored_training_icon_is_listed_and_can_be_left_out(
    reference: Reference,
) -> None:
    """A held-out icon that is a training icon's mirror image pins the declared rule's
    aligned-error threshold near zero; `without-twins` calibrates without it."""

    train, validation = reference.train, reference.validation
    renders = train.targets.numpy()
    # The training icon least like its own mirror, so the twin is only a mirror twin.
    source = int(np.argmax([np.abs(r.astype(int) - r[:, ::-1]).sum() for r in renders]))
    targets = validation.targets.clone()
    targets[0] = train.targets[source].flip(1)
    twinned = SplitData(
        validation.hexcodes,
        validation.tokens,
        validation.masks,
        validation.images,
        targets,
        validation.shape,
    )
    built = {
        rule: build_reference(
            reference.layout,
            reference.template,
            twinned,
            train,
            _embed,
            torch.device("cpu"),
            shortlist=4,
            copy_rule=rule,
        )
        for rule in ("declared", "without-twins")
    }
    calibration = built["without-twins"].calibration
    assert calibration["twins"] == [
        {
            "validation": validation.hexcodes[0],
            "training": train.hexcodes[source],
            "mirror": True,
            "aligned_error": 0.0,
        }
    ]
    assert calibration["twin_error"] == TWIN_ERROR
    rules = calibration["copy_rules"]
    assert rules["declared"]["calibration_icons"] == 4
    assert rules["without-twins"]["calibration_icons"] == 3
    assert rules["declared"]["copy_error"] < rules["without-twins"]["copy_error"]
    assert calibration["copy_error"] == rules["without-twins"]["copy_error"]
    assert built["declared"].calibration["copy_error"] == rules["declared"]["copy_error"]
    # The mirror-free baseline misses the twin; the mirrored one finds it exactly.
    assert built["declared"].validation_nearest_error[0] == 0.0
    assert built["declared"].validation_nearest_error_without_mirrors[0] > 0.0
    with pytest.raises(ValueError, match="unknown copy rule"):
        build_reference(
            reference.layout,
            reference.template,
            twinned,
            train,
            _embed,
            torch.device("cpu"),
            copy_rule="loose",
        )


def test_a_static_strip_is_counted_apart_and_scored_as_a_cut(reference: Reference) -> None:
    """A decoder that ignores its latent draws the same program in every frame: its
    strip has no jump share, is counted static, and scores worst when forced in."""

    frames = 3
    moving = reference.train.tokens[:frames]
    still = reference.train.tokens[:1].expand(frames, -1)
    summary, per_pair, _ = score_interpolation(
        reference, torch.cat((moving, still)), [(0, 1), (2, 3)], frames
    )
    assert per_pair["static"] == [False, True] and per_pair["jump_share"][1] is None
    assert summary["static_pairs"] == 1 and summary["jump_share_pairs"] == 1
    moved = per_pair["jump_share"][0]
    assert summary["jump_share"][0] == pytest.approx(moved)
    assert summary["jump_share_static_as_cut"][0] == pytest.approx(0.5 * (moved + 1.0))
    json.dumps(summary, allow_nan=False)


def test_report_names_carry_the_rule_and_a_settings_hash() -> None:
    first = report_name("run", "normal", _SETTINGS)
    assert first.startswith("metrics-run-normal-declared-")
    assert report_name("run", "normal", _SETTINGS) == first
    assert report_name("run", "normal", replace(_SETTINGS, samples=4)) != first
    other = report_name("run", "normal", replace(_SETTINGS, copy_rule="without-twins"))
    assert other.startswith("metrics-run-normal-without-twins-")
    # A setting added later leaves the names of the reports written before it alone (the
    # committed baselines: ...-normal-without-twins-92b2165c, ...-refit-...-37d666a3).
    assert settings_hash(Settings(copy_rule="without-twins")) == "92b2165c"
    assert settings_hash(Settings(prior="refit", copy_rule="without-twins")) == "37d666a3"
    own = report_name("run", "normal", replace(_SETTINGS, interpolation="backend"))
    assert own.startswith("metrics-run-normal-backend-path-declared-")
    assert own.rsplit("-", 1)[1] != first.rsplit("-", 1)[1]


def test_a_tiny_vae_is_scored_end_to_end_under_both_priors(reference: Reference) -> None:
    torch.manual_seed(0)
    model = LatentToProgram(
        reference.layout, _TINY, LatentSettings(latent_dim=8, memory_tokens=4, encoder_layers=1)
    ).eval()
    backend = VaeBackend(model, graphs=False)
    report, sheet = score_backend(backend, reference, _SETTINGS)
    json.dumps(report, allow_nan=False)
    assert sheet.startswith(b"\x89PNG")
    samples, recon, strips = report["samples"], report["reconstruction"], report["interpolation"]
    assert samples["count"] == 3 and len(report["per_item"]["samples"]["ink"]) == 3
    assert 0.0 <= samples["clip_precision"][0] <= 1.0 and 0.0 <= samples["copy_rate"][0] <= 1.0
    assert strips["pairs"] == 1 and strips["frames"] == 3 and strips["jump_share_even"] == 0.5
    assert strips["path"] == "lerp"
    if strips["static_pairs"]:
        assert strips["jump_share"] is None and strips["jump_share_static_as_cut"][0] == 1.0
    else:
        assert 0.5 <= strips["jump_share"][0] <= 1.0
    # The scorer scores exactly what the backend decodes.
    with torch.no_grad():
        means = backend.encode(reference.validation.tokens[:2], reference.validation.images[:2])
        decoded = backend.decode(means)
    renders = _programs_to_renders(decoded, reference.layout, reference.template)
    targets = reference.validation.targets.numpy()
    errors = [pixel_error(r, targets[i]) for i, r in enumerate(renders)]
    assert report["per_item"]["reconstruction"]["pixel_error"] == errors
    assert recon["icons"] == 2 and recon["pixel_error"][0] == pytest.approx(np.mean(errors))
    assert recon["true_collapsed_path_share"] < 0.5
    nearest = reference.validation_nearest_error[:2]
    plain = reference.validation_nearest_error_without_mirrors[:2]
    assert recon["nearest_training_icon_pixel_error"][0] == pytest.approx(np.mean(nearest))
    assert recon["nearest_training_icon_pixel_error_without_mirrors"][0] == pytest.approx(
        np.mean(plain)
    )
    rows = report["references"]
    assert rows["training_renders"]["copy_rate"][0] == 1.0
    assert rows["training_renders"]["exact_token_copies"] == 3
    assert rows["blank_canvas"]["near_blank_rate"][0] == 1.0
    assert rows["blank_canvas"]["ink_mean"] == 0.0
    assert rows["collages"]["count"] == 3
    assert report["resources"]["torch_version"] == torch.__version__
    refit, _ = score_backend(backend, reference, replace(_SETTINGS, prior="refit"))
    json.dumps(refit, allow_nan=False)
    assert refit["refit"]["fitted_on_training_icons"] == 5 and refit["refit"]["dimensions"] == 8
    assert refit["reconstruction"] == report["reconstruction"]
    # Settings naming another copy rule than the reference's are refused.
    with pytest.raises(ValueError, match="calibrated under"):
        score_backend(backend, reference, replace(_SETTINGS, copy_rule="without-twins"))


def test_the_crossfade_reference_scores_a_transcriber_on_blended_renders(
    reference: Reference,
) -> None:
    torch.manual_seed(0)
    transcriber = RenderToProgram(reference.layout, _TINY).eval()
    report, sheet = score_crossfade(transcriber, reference, _SETTINGS)
    json.dumps(report, allow_nan=False)
    assert report["prior"] == "crossfade" and "samples" not in report
    assert sheet.startswith(b"\x89PNG")
    ((a, b),) = fixed_pairs(4, 1, _SETTINGS.pair_seed)
    images, targets = reference.validation.images, reference.validation.targets.numpy()
    decoded = greedy_decode(transcriber, crossfade_frames(images[a], images[b], 3))
    renders = _programs_to_renders(decoded, reference.layout, reference.template)
    endpoint = 0.5 * (pixel_error(renders[0], targets[a]) + pixel_error(renders[-1], targets[b]))
    assert report["per_item"]["interpolation"]["endpoint"] == [endpoint]
    assert report["per_item"]["interpolation"]["pairs"] == [[a, b]]


def test_the_backend_path_is_scored_only_when_asked_and_named_in_the_report(
    reference: Reference, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With interpolation "backend" the pairs are scored along canvas-flow's own path,
    slerp through its prior's noise, and the report says so; by default they are scored
    along the straight line between the posterior means - the frozen VT's lerp, exactly
    what the same VT scores under `canvas`. A backend without a path of its own is
    refused rather than silently scored along the line."""

    torch.manual_seed(0)
    vt = VariationalTranscriber(reference.layout, _TINY, CanvasSettings(channels=2)).eval()
    network = FlowTransformer(
        FlowNetworkConfig(channels=2, grid=4, d_model=32, layers=1, heads=4, frequency_dim=32)
    )
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for parameter in network.parameters():  # off the zero start: a non-zero velocity
            parameter.add_(0.05 * torch.randn(parameter.shape, generator=generator))
    prior = LatentFlowPrior(network, torch.tensor([0.5, -0.2]), torch.tensor([1.5, 0.8]), steps=4)
    flow, canvas = CanvasFlowBackend(vt, prior, graphs=False), CanvasBackend(vt, graphs=False)
    scored: list[torch.Tensor] = []
    real = latent_metrics.score_interpolation

    def spy(
        given: Reference, programs: torch.Tensor, pairs: list[tuple[int, int]], frames: int
    ) -> Any:
        scored.append(programs.clone())
        return real(given, programs, pairs, frames)

    monkeypatch.setattr(latent_metrics, "score_interpolation", spy)
    own = replace(_SETTINGS, interpolation="backend")
    report, _ = score_backend(flow, reference, own)
    json.dumps(report, allow_nan=False)
    assert report["interpolation"]["path"] == "slerp-through-prior-noise"
    ((a, b),) = fixed_pairs(4, 1, _SETTINGS.pair_seed)
    weights = torch.linspace(0.0, 1.0, 3).view(-1, 1, 1, 1)
    with torch.no_grad():
        means, _ = vt.posterior(reference.validation.images[[a, b]])
        slerp = prior.slerp_path(means[0], means[1], 3)
        line = (1 - weights) * means[0] + weights * means[1]
        assert torch.equal(scored[-1], greedy_decode(vt, slerp))
        assert not torch.equal(slerp, line)
        straight = greedy_decode(vt, line)
    resources = report["resources"]
    assert resources["prior_ms_per_draw"] > 0.0 and resources["end_to_end_ms_batch_1"] > 0.0
    # The default: the straight line, the same strips as the VT under `canvas`.
    lerp, _ = score_backend(flow, reference, _SETTINGS)
    plain, _ = score_backend(canvas, reference, _SETTINGS)
    assert lerp["interpolation"]["path"] == plain["interpolation"]["path"] == "lerp"
    assert torch.equal(scored[-2], straight) and torch.equal(scored[-1], straight)
    with pytest.raises(ValueError, match="has none"):
        score_backend(canvas, reference, own)
    with pytest.raises(ValueError, match="unknown interpolation"):
        score_backend(canvas, reference, replace(_SETTINGS, interpolation="curvy"))
