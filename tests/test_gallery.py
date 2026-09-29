"""The model gallery's contract: every registered run, every endpoint, the latent
backends and the ratings log, on tiny CPU models."""

from __future__ import annotations

import base64
import http.client
import io
import json
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
import yaml
from PIL import Image

import mojidiff.gallery.server as gallery_server
import mojidiff.learning.render2svg as render2svg
from mojidiff.gallery.latent_backends import (
    BACKENDS,
    CanvasBackend,
    LatentBackend,
    VaeBackend,
    vae_model,
)
from mojidiff.gallery.server import (
    BASELINE_PIXEL_ERROR,
    ICON_FIELDS,
    MAX_BODY,
    MODELS,
    RATING_FIELDS,
    RATING_NOTE_LIMIT,
    REGISTRY,
    Gallery,
    GalleryError,
    ModelSpec,
    RatingLog,
    load_model,
    load_registry,
    make_server,
    model_entry,
    registry_models,
    run_record,
)
from mojidiff.learning.autoregressive import SequenceLayout, flatten_program, unflatten_program
from mojidiff.learning.latent import LatentSettings, LatentToProgram
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.pixel_latent import CanvasSettings, VariationalTranscriber, checkpoint_state
from mojidiff.learning.render2svg import (
    ModelConfig,
    RenderToProgram,
    greedy_decode,
    render_trusted_rgb,
    rerank_decode,
)
from mojidiff.representation.packed import serialize_packed_svg
from mojidiff.representation.renderer import RenderLimits, validate_typed_svg
from mojidiff.vectorise.server import canvas_to_rgb

_CONFIG = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")
_SPECS = {spec.id: spec for spec in MODELS}
_ENTRY_KEYS = {"id", "label", "kind", "run_id", "description", "parameters", "stats"}
_TRANSCRIBER_STATS = {"pixel_error", "beats_nearest", "clip_top1", "icons", "evaluated_on"}
_LATENT_STATS = {
    "reconstruction_pixel_error",
    "prior_mean_pixel_error",
    "samples_distinct",
    "evaluated_on",
}
_TRANSCRIBE_KEYS = {"ok", "model", "svg", "ms", "decoder_calls", "paths", "segments", "candidates"}
cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need a GPU")
Parts = tuple[SequenceLayout, Any, dict[str, torch.Tensor]]


def _valid(svg: str) -> None:
    validate_typed_svg(svg.encode(), RenderLimits(max_paths=80))


def _tiny() -> ModelConfig:
    return ModelConfig(
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


@pytest.fixture(scope="module")
def parts() -> tuple[SequenceLayout, Any, dict[str, torch.Tensor]]:
    pilot = load_openmoji_pilot_config(_CONFIG)
    codec = _selected_codec(pilot)
    layout = SequenceLayout(codec=codec, total_segment_slots=pilot.total_segment_slots)
    by_split, _, _ = load_pilot_index(pilot)
    template = _load_program(by_split["primary/train"][0], pilot, codec)
    rows = _select_rows(by_split["primary/validation"], 2, pilot.seed + 1)
    programs = {
        row.hexcode: flatten_program(_load_program(row, pilot, codec), layout) for row in rows
    }
    return layout, template, programs


def _tiny_models(layout: SequenceLayout) -> tuple[RenderToProgram, LatentToProgram]:
    torch.manual_seed(0)
    transcriber = RenderToProgram(layout, _tiny())
    latent = LatentToProgram(
        layout, _tiny(), LatentSettings(latent_dim=8, memory_tokens=4, encoder_layers=1)
    )
    return transcriber.eval(), latent.eval()


@pytest.fixture(scope="module")
def tiny(parts: Parts) -> tuple[RenderToProgram, LatentToProgram]:
    return _tiny_models(parts[0])


@pytest.fixture(scope="module")
def gallery(
    parts: Parts,
    tiny: tuple[RenderToProgram, LatentToProgram],
    tmp_path_factory: pytest.TempPathFactory,
) -> Gallery:
    _, template, programs = parts
    transcriber, latent = tiny
    return Gallery(
        [(_SPECS["v9"], transcriber), (_SPECS["l2"], latent)],
        template,
        programs,
        torch.device("cpu"),
        ratings=tmp_path_factory.mktemp("ratings") / "ratings.jsonl",
    )


def _icon_rgb(gallery: Gallery, size: int, index: int = 0) -> np.ndarray:
    return canvas_to_rgb(gallery.icon_png(str(gallery.icons[index]["hexcode"])), size)


def _svg_of(tokens: torch.Tensor, template: Any, layout: SequenceLayout) -> str:
    program = unflatten_program(tokens.cpu(), template, layout)
    return serialize_packed_svg(program, layout.codec, layout.total_segment_slots).decode()


# --------------------------------------------------------------------------- registry


def test_registry_lists_every_model_once_in_display_order() -> None:
    assert [spec.id for spec in MODELS] == [
        "v9", "v7", "v8", "v6", "v5", "v4", "v3", "v2", "v1", "o4", "l2", "l1",
    ]  # fmt: skip
    assert len({spec.run_id for spec in MODELS}) == len(MODELS)
    assert {spec.kind for spec in MODELS[:10]} == {"transcriber"}
    assert {spec.kind for spec in MODELS[10:]} == {"latent"}
    assert {spec.backend for spec in MODELS[:10]} == {None}
    assert {spec.backend for spec in MODELS[10:]} == {"vae"}
    for spec in MODELS:
        assert spec.checkpoint.name == "best.pt" and spec.checkpoint.parent.name == spec.run_id


def test_registry_file_keeps_the_original_models_and_every_checkpoint_exists() -> None:
    """The registry grows as latent models are trained: it keeps the original twelve
    exactly (ids, labels, descriptions, backends, relative order), the default load reads
    it, ids are unique, and every listed checkpoint exists."""

    specs = load_registry(REGISTRY)
    assert registry_models() == specs
    kept = [spec for spec in specs if spec.id in {model.id for model in MODELS}]
    assert tuple(kept) == MODELS
    assert len({spec.id for spec in specs}) == len(specs)
    assert all(spec.checkpoint.is_file() for spec in specs)


def _entry(**changes: Any) -> dict[str, Any]:
    entry = {
        "id": "l9",
        "run_id": "toy-run",
        "kind": "latent",
        "backend": "vae",
        "label": "toy",
        "description": "a toy latent",
    }
    entry.update(changes)
    return {key: value for key, value in entry.items() if value is not None}


def _write_registry(path: Path, entries: list[dict[str, Any]], **root: Any) -> Path:
    path.write_text(yaml.safe_dump({"schema_version": 1, "models": entries, **root}))
    return path


def test_registry_refuses_malformed_entries_and_unknown_backends(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "models.yaml"
    assert load_registry(_write_registry(path, [_entry()]))[0].backend == "vae"
    transcriber = _entry(kind="transcriber", backend=None)
    assert load_registry(_write_registry(path, [transcriber]))[0].backend is None
    refused: list[tuple[list[dict[str, Any]], str]] = [
        ([_entry(backend="nope")], "unknown backend 'nope'"),
        ([_entry(backend=["vae"])], "unknown backend"),
        ([_entry(backend=None)], "names no backend"),
        ([_entry(kind="transcriber")], "takes no backend"),
        ([_entry(kind="diffusion")], "kind must be one of"),
        ([_entry(), _entry(run_id="other")], "duplicate id 'l9'"),
        ([_entry(checkpoint="best.pt")], "unknown field"),
        ([_entry(label=None)], "missing field"),
        ([_entry(label="  ")], "label must be a non-empty string"),
        ([_entry(run_id=7)], "run_id must be a non-empty string"),
        ([_entry(id="../l9")], "id '../l9' must match"),
        ([["l9"]], "not a mapping"),  # type: ignore[list-item]
        ([], "non-empty list"),
    ]
    for entries, message in refused:
        with pytest.raises(ValueError, match=re.escape(message)):
            load_registry(_write_registry(path, entries))
    path.write_text(yaml.safe_dump({"models": [_entry()]}))
    with pytest.raises(ValueError, match="schema_version"):
        load_registry(path)

    # An explicit file must exist; only the default falls back to the built-in list.
    with pytest.raises(FileNotFoundError):
        registry_models(tmp_path / "absent.yaml")
    said: list[str] = []
    monkeypatch.setattr(gallery_server, "REGISTRY", tmp_path / "absent.yaml")
    assert registry_models(log=said.append) == MODELS and "built-in" in said[0]


def test_startup_refuses_unknown_backends_and_missing_checkpoints() -> None:
    """Both before the corpus loads or anything binds."""

    cpu = torch.device("cpu")
    missing = ModelSpec("l9", "no-such-run-0000", "latent", "toy", "", "vae")
    with pytest.raises(FileNotFoundError, match="missing checkpoints: .*no-such-run-0000"):
        Gallery.from_checkpoints(cpu, [_SPECS["v9"], missing])
    unknown = ModelSpec("l9", MODELS[-1].run_id, "latent", "toy", "", "nope")
    with pytest.raises(ValueError, match=r"unknown latent backends: l9 \(nope\)"):
        Gallery.from_checkpoints(cpu, [unknown])


def test_startup_serves_a_model_without_a_run_record(tmp_path: Path) -> None:
    """A registered model whose `runs/<run_id>/run.yaml` is absent starts with null stats
    and a warning, as `Gallery` itself treats it, rather than failing after the corpus
    and the earlier models have loaded."""

    spec = _SPECS["l1"]
    if not spec.checkpoint.is_file():
        pytest.skip(f"no checkpoint for {spec.run_id}")
    said: list[str] = []
    served = Gallery.from_checkpoints(
        torch.device("cpu"), [spec], ratings=None, runs_root=tmp_path, log=said.append
    )
    (entry,) = served.models_payload()["models"]
    assert entry["id"] == "l1" and entry["parameters"] is None
    assert all(value is None for value in entry["stats"].values())
    assert any(f"no run record for {spec.run_id}" in line for line in said)
    assert not any("parameters, its run record" in line for line in said)


def test_registry_stats_parse_from_every_real_run_record() -> None:
    for spec in MODELS:
        entry = model_entry(spec, run_record(spec))
        assert set(entry) == _ENTRY_KEYS
        assert isinstance(entry["parameters"], int) and entry["parameters"] > 1_000_000
        stats = entry["stats"]
        if spec.kind == "latent":
            assert set(stats) == _LATENT_STATS
            assert stats["evaluated_on"] == "primary/validation"
            assert 0.0 < stats["reconstruction_pixel_error"] < 1.0
            assert 0.0 < stats["prior_mean_pixel_error"] < 1.0
            assert isinstance(stats["samples_distinct"], int)
        elif spec.id == "o4":
            # Scored on its own four training icons: withheld, and said why.
            assert set(stats) == _TRANSCRIBER_STATS
            assert stats["evaluated_on"] == "primary/train"
            assert all(stats[key] is None for key in _TRANSCRIBER_STATS - {"evaluated_on"})
        else:
            assert set(stats) == _TRANSCRIBER_STATS
            assert stats["evaluated_on"] == "primary/validation"
            assert 0.0 < stats["pixel_error"] < 1.0
            assert isinstance(stats["beats_nearest"], int) and stats["icons"] == 339
            assert 0.0 <= stats["clip_top1"] <= 1.0
    v9 = model_entry(_SPECS["v9"], run_record(_SPECS["v9"]))["stats"]
    assert v9["pixel_error"] < BASELINE_PIXEL_ERROR


def test_a_model_registered_under_the_wrong_kind_is_refused(
    parts: tuple[SequenceLayout, Any, dict[str, torch.Tensor]],
) -> None:
    layout, template, programs = parts
    latent = LatentToProgram(layout, _tiny(), LatentSettings(latent_dim=8, memory_tokens=4))
    with pytest.raises(ValueError, match="kind"):
        Gallery([(_SPECS["v1"], latent)], template, programs, torch.device("cpu"))


# --------------------------------------------------------------------------- gallery


def test_models_payload_shape_and_order(gallery: Gallery) -> None:
    payload = gallery.models_payload()
    assert payload["baseline_pixel_error"] == BASELINE_PIXEL_ERROR == 0.090
    assert [m["id"] for m in payload["models"]] == ["v9", "l2"]
    for model in payload["models"]:
        assert set(model) == _ENTRY_KEYS
    transcriber, latent = payload["models"]
    assert transcriber["kind"] == "transcriber" and set(transcriber["stats"]) == _TRANSCRIBER_STATS
    assert latent["kind"] == "latent" and set(latent["stats"]) == _LATENT_STATS
    assert len(gallery) == 2


def test_transcribe_returns_a_valid_typed_svg_greedy_and_best_of_8(gallery: Gallery) -> None:
    rgb = _icon_rgb(gallery, gallery.input_size("v9"))
    greedy = gallery.transcribe("v9", rgb, 1)
    assert set(greedy) == _TRANSCRIBE_KEYS
    assert greedy["ok"] and greedy["model"] == "v9" and greedy["candidates"] == 1
    assert greedy["decoder_calls"] > 0 and greedy["ms"] >= 0.0
    assert greedy["segments"] >= greedy["paths"] >= 0
    _valid(greedy["svg"])
    best = gallery.transcribe("v9", rgb, 8)
    assert set(best) == _TRANSCRIBE_KEYS and best["ok"] and best["candidates"] == 8
    _valid(best["svg"])


def test_best_of_8_chooses_as_rerank_decode_and_renders_outside_the_locks(
    gallery: Gallery,
    parts: Parts,
    tiny: tuple[RenderToProgram, LatentToProgram],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layout, template, _ = parts
    transcriber, _ = tiny
    rgb = _icon_rgb(gallery, gallery.input_size("v9"))
    reference, _ = rerank_decode(
        transcriber,
        torch.from_numpy(rgb.copy()),
        template,
        candidates=8,
        temperature=gallery_server.RERANK_TEMPERATURE,
        seed=gallery_server.RERANK_SEED,
    )
    held: list[bool] = []
    render = render2svg.render_trusted_rgb

    def watched(svg: bytes, size: int) -> np.ndarray:
        held.append(gallery.gpu_lock.locked())
        return render(svg, size)

    monkeypatch.setattr(render2svg, "render_trusted_rgb", watched)
    best = gallery.transcribe("v9", rgb, 8)
    assert best["svg"] == _svg_of(reference[0], template, layout)
    assert best["candidates"] == 8 and best["decoder_calls"] > 0
    # Every candidate is rendered to choose, and none while the GPU is held: `ms` is
    # decoding only, and other models' requests do not wait on CPU rendering.
    assert len(held) == 8 and not any(held)


def test_latent_reconstruct_interpolate_and_sample_decode_valid_programs(
    gallery: Gallery,
) -> None:
    first, second = sorted(gallery.programs)
    reconstruction = gallery.reconstruct("l2", first)
    assert set(reconstruction) == {"ok", "svg", "ms"} and reconstruction["ok"]
    _valid(reconstruction["svg"])
    path = gallery.interpolate("l2", first, second, 3)
    assert set(path) == {"ok", "frames", "path", "ms"} and len(path["frames"]) == 3
    # A vae has no path of its own: the backend's path is the straight line.
    assert path["path"] == "lerp"
    assert gallery.interpolate("l2", first, second, 3, path="lerp")["frames"] == path["frames"]
    for frame in path["frames"]:
        _valid(frame)
    samples = gallery.sample("l2", 3, seed=7, scale=1.0)
    assert set(samples) == {"ok", "samples", "ms"} and len(samples["samples"]) == 3
    for sample in samples["samples"]:
        _valid(sample)
    assert gallery.sample("l2", 3, seed=7, scale=1.0)["samples"] == samples["samples"]


# --------------------------------------------------------------------------- backends


def _latent_references(
    latent: LatentToProgram,
    parts: Parts,
    first: str,
    second: str,
    steps: int,
    samples: torch.Tensor,
) -> tuple[str, list[str], list[str]]:
    """What the gallery answered before it had backends, computed from the model:
    the posterior mean decoded, the straight line between two means decoded in one
    batch, and given prior draws decoded in one batch."""

    layout, template, programs = parts
    with torch.no_grad():
        mean, _ = latent.posterior(programs[first][None])
        reconstruction = _svg_of(greedy_decode(latent, mean)[0], template, layout)
        means, _ = latent.posterior(torch.stack((programs[first], programs[second])))
        weights = torch.linspace(0.0, 1.0, steps)[:, None]
        path = greedy_decode(latent, (1.0 - weights) * means[0] + weights * means[1])
        drawn = greedy_decode(latent, samples)
    frames = [_svg_of(row, template, layout) for row in path]
    return reconstruction, frames, [_svg_of(row, template, layout) for row in drawn]


def test_the_vae_backend_answers_exactly_what_the_latent_model_decodes(
    gallery: Gallery, parts: Parts, tiny: tuple[RenderToProgram, LatentToProgram]
) -> None:
    _, template, programs = parts
    _, latent = tiny
    first, second = sorted(programs)
    draws = 1.5 * torch.randn(4, 8, generator=torch.Generator().manual_seed(9))
    reconstruction, frames, samples = _latent_references(latent, parts, first, second, 5, draws)
    assert gallery.reconstruct("l2", first)["svg"] == reconstruction
    assert gallery.interpolate("l2", first, second, 5)["frames"] == frames
    assert gallery.sample("l2", 4, seed=9, scale=1.5)["samples"] == samples
    # A bare LatentToProgram is served by the vae backend: given explicitly, the same.
    explicit = Gallery(
        [(_SPECS["l2"], VaeBackend(latent))],
        template,
        programs,
        torch.device("cpu"),
        icons=gallery.icons,
    )
    assert explicit.reconstruct("l2", first)["svg"] == reconstruction
    assert explicit.sample("l2", 4, seed=9, scale=1.5)["samples"] == samples
    assert isinstance(VaeBackend(latent), LatentBackend)
    assert VaeBackend(latent).latent_shape == (8,)


class _GridPrior:
    """A toy second backend, registered by the tests alone: the tiny vae's 8-number
    latent seen as a 2x4 grid, under a prior twice as wide, with no single decoder."""

    def __init__(self, model: LatentToProgram) -> None:
        self.model = model
        self.latent_shape: tuple[int, ...] = (2, 4)
        self.vae = VaeBackend(model)
        self.images: list[torch.Tensor] = []

    def encode(self, tokens: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        self.images.append(images)
        return self.vae.encode(tokens, images).view(-1, *self.latent_shape)

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents.reshape(len(latents), -1))

    def sample(self, count: int, generator: torch.Generator, scale: float) -> torch.Tensor:
        return 2.0 * scale * torch.randn(count, *self.latent_shape, generator=generator)


def _grid(state: Mapping[str, Any], layout: SequenceLayout, device: torch.device) -> LatentBackend:
    return _GridPrior(vae_model(state, layout).to(device))


def test_a_second_backend_plugs_in_by_a_registry_entry_alone(
    gallery: Gallery,
    parts: Parts,
    tiny: tuple[RenderToProgram, LatentToProgram],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Register a factory, name it in a registry file, load its checkpoint: the gallery
    serves it - a latent of another shape, another prior - with no server change."""

    layout, template, programs = parts
    _, latent = tiny
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model": latent.state_dict(),
            "step": 3,
            "config": asdict(_tiny()),
            "latent": asdict(latent.settings),
        },
        checkpoint,
    )
    registry = _write_registry(tmp_path / "models.yaml", [_entry(id="g1", backend="grid")])
    with pytest.raises(ValueError, match="unknown backend 'grid'"):
        load_registry(registry)
    monkeypatch.setitem(BACKENDS, "grid", _grid)
    (spec,) = load_registry(registry)
    backend, extras = load_model(spec, layout, torch.device("cpu"), checkpoint=checkpoint)
    assert isinstance(backend, _GridPrior) and extras["step"] == 3 and "model" not in extras
    said: list[str] = []
    served = Gallery(
        [(spec, backend)],
        template,
        programs,
        torch.device("cpu"),
        icons=gallery.icons,
        runs_root=tmp_path,
        log=said.append,
    )
    assert "no run record for toy-run" in said[0]
    (entry,) = served.models_payload()["models"]
    assert entry["id"] == "g1" and entry["kind"] == "latent"

    first, second = sorted(programs)
    draws = 2.0 * 0.5 * torch.randn(3, 2, 4, generator=torch.Generator().manual_seed(5))
    reconstruction, frames, samples = _latent_references(
        latent, parts, first, second, 4, draws.reshape(3, 8)
    )
    assert served.reconstruct("g1", first)["svg"] == reconstruction
    assert served.interpolate("g1", first, second, 4)["frames"] == frames
    assert served.sample("g1", 3, seed=5, scale=0.5)["samples"] == samples
    # The encoder is given each icon's program render at 144 px, uint8, on the device.
    reference = render_trusted_rgb(_svg_of(programs[first], template, layout).encode(), 144)
    assert [tuple(images.shape) for images in backend.images] == [
        (1, 144, 144, 3),
        (2, 144, 144, 3),
    ]
    assert backend.images[0].dtype == torch.uint8
    assert np.array_equal(backend.images[0][0].numpy(), reference)
    assert np.array_equal(backend.images[1][0].numpy(), reference)

    # A backend that breaks its contract is named in the error, not decoded.
    monkeypatch.setattr(backend, "sample", lambda count, generator, scale: torch.zeros(count, 8))
    with pytest.raises(RuntimeError, match=r"'grid' of 'g1': sample returned shape \(2, 8\)"):
        served.sample("g1", 2, seed=0, scale=1.0)


def test_the_canvas_backend_serves_a_variational_transcriber(
    gallery: Gallery, parts: Parts, tmp_path: Path
) -> None:
    """A VT checkpoint loads through the `canvas` backend by a registry entry: held-out
    renders encode to posterior-mean grids, and reconstruction, interpolation and
    N(0, I) sampling answer what the model's own greedy decoder answers."""

    layout, template, programs = parts
    torch.manual_seed(0)
    # The gallery hands latent encoders 144 px renders, so the tiny model reads 144 px.
    config = ModelConfig(**{**asdict(_tiny()), "image_size": 144})
    model = VariationalTranscriber(layout, config, CanvasSettings(channels=2)).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save(checkpoint_state(model, 5), checkpoint)
    registry = _write_registry(tmp_path / "models.yaml", [_entry(id="c1", backend="canvas")])
    (spec,) = load_registry(registry)
    backend, extras = load_model(spec, layout, torch.device("cpu"), checkpoint=checkpoint)
    assert isinstance(backend, CanvasBackend) and isinstance(backend, LatentBackend)
    assert backend.latent_shape == (2, 18, 18) and backend.graph is None
    assert extras["step"] == 5 and "model" not in extras
    served = Gallery(
        [(spec, backend)],
        template,
        programs,
        torch.device("cpu"),
        icons=gallery.icons,
        runs_root=tmp_path,
    )
    first, second = sorted(programs)
    images = torch.stack(
        [
            torch.from_numpy(
                render_trusted_rgb(_svg_of(programs[code], template, layout).encode(), 144)
            )
            for code in (first, second)
        ]
    )
    with torch.no_grad():
        means, _ = model.posterior(images)
        # Reconstruction reads the posterior mean: through the render or the grid alike.
        reconstruction = greedy_decode(model, images[:1])
        assert torch.equal(greedy_decode(model, means[:1]), reconstruction)
        weights = torch.linspace(0.0, 1.0, 4).view(4, 1, 1, 1)
        frames = greedy_decode(model, (1.0 - weights) * means[0] + weights * means[1])
        draws = 0.5 * torch.randn(3, 2, 18, 18, generator=torch.Generator().manual_seed(5))
        samples = greedy_decode(model, draws)
    assert served.reconstruct("c1", first)["svg"] == _svg_of(reconstruction[0], template, layout)
    assert served.interpolate("c1", first, second, 4)["frames"] == [
        _svg_of(row, template, layout) for row in frames
    ]
    answered = served.sample("c1", 3, seed=5, scale=0.5)["samples"]
    assert answered == [_svg_of(row, template, layout) for row in samples]
    for svg in answered:
        _valid(svg)
    assert "canvas" in BACKENDS


# --------------------------------------------------------------------------- ratings


def _rating(**changes: Any) -> dict[str, Any]:
    rating: dict[str, Any] = {
        "model": "l2",
        "view": "samples",
        "score": 4,
        "blind": False,
        "seed": 7,
        "scale": 1.0,
        "a": None,
        "b": None,
        "steps": None,
        "note": "",
    }
    rating.update(changes)
    return rating


@pytest.fixture
def rated(
    gallery: Gallery, parts: Parts, tiny: tuple[RenderToProgram, LatentToProgram], tmp_path: Path
) -> Gallery:
    _, template, programs = parts
    transcriber, latent = tiny
    return Gallery(
        [(_SPECS["v9"], transcriber), (_SPECS["l2"], latent)],
        template,
        programs,
        torch.device("cpu"),
        icons=gallery.icons,
        ratings=tmp_path / "gallery" / "ratings.jsonl",
    )


def test_ratings_are_checked_appended_and_summarised(rated: Gallery) -> None:
    log = cast(RatingLog, rated.ratings)
    assert not log.path.exists() and rated.ratings_payload() == {"ratings": [], "summary": {}}
    assert rated.rate(_rating()) == {"ok": True, "count": 1}
    (row,) = log.rows()
    assert list(row) == ["at", "model", "run_id", *RATING_FIELDS[1:]]
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", row["at"])
    assert row["run_id"] == _SPECS["l2"].run_id
    assert {key: row[key] for key in RATING_FIELDS} == _rating()
    first, second = sorted(rated.programs)
    interpolation = _rating(
        view="interpolation", score=2, blind=True, a=first, b=second, steps=7, note="wavy"
    )
    assert rated.rate(interpolation)["count"] == 2
    bare = {key: _rating(score=5)[key] for key in ("model", "view", "score", "blind")}
    assert rated.rate(bare)["count"] == 3
    assert log.rows()[-1]["note"] == "" and log.rows()[-1]["seed"] is None
    assert rated.rate(_rating(score=3, note="x" * RATING_NOTE_LIMIT, scale=2))["count"] == 4
    assert log.rows()[-1]["scale"] == 2.0

    refused: list[tuple[dict[str, Any], str]] = [
        (_rating(score=0), "score"),
        (_rating(score=6), "score"),
        (_rating(score=3.5), "score"),
        (_rating(score="3"), "score"),
        (_rating(score=True), "score"),
        (_rating(view="sample"), "view must be one of"),
        (_rating(model="nope"), "unknown model"),
        (_rating(model="v9"), "is a transcriber, not a latent"),
        (_rating(model=3), "model must be a string"),
        (_rating(blind="yes"), "blind"),
        ({key: value for key, value in _rating().items() if key != "blind"}, "missing field"),
        (_rating(seed=-1), "seed"),
        (_rating(seed=1.5), "seed"),
        (_rating(scale=2.5), "scale"),
        (_rating(scale=float("nan")), "scale"),
        (_rating(scale="1"), "scale"),
        (_rating(a="NOT-AN-ICON"), "a must be"),
        (_rating(b=7), "b must be"),
        (_rating(steps=2), "steps"),
        (_rating(steps=12), "steps"),
        (_rating(note="x" * (RATING_NOTE_LIMIT + 1)), "note"),
        (_rating(note=5), "note"),
        (_rating(extra=1), "unknown rating field(s): extra"),
    ]
    for payload, message in refused:
        with pytest.raises(GalleryError, match=re.escape(message)):
            rated.rate(payload)
    assert len(log.rows()) == 4

    # History the registry has moved past: a row for an older run under the same id is
    # kept but not averaged; a model no longer loaded keeps its own summary.
    log.append({"model": "l2", "run_id": "older-run", "view": "samples", "score": 1})
    log.append({"model": "l0", "run_id": "gone-run", "view": "samples", "score": 2})
    payload = rated.ratings_payload()
    assert len(payload["ratings"]) == 6
    assert payload["summary"] == {
        "l2": {
            "samples": {"n": 3, "mean": 4.0},
            "interpolation": {"n": 1, "mean": 2.0},
        },
        "l0": {"samples": {"n": 1, "mean": 2.0}},
    }
    assert rated.rate(_rating(score=1))["count"] == 7
    assert rated.ratings_payload()["summary"]["l2"]["samples"] == {"n": 4, "mean": 3.25}


def test_the_ratings_file_is_only_ever_appended(tmp_path: Path) -> None:
    """Existing bytes stay as they are, even a line cut short; the file is never
    replaced; concurrent appends never interleave."""

    path = tmp_path / "ratings.jsonl"
    path.write_bytes(b'{"model": "l2", "view": "samples", "score": 5}\n{"model": "l2", "vi')
    before = path.read_bytes()
    inode = path.stat().st_ino
    log = RatingLog(path)
    assert log.append({"model": "l2", "view": "samples", "score": 3}) == 2
    after = path.read_bytes()
    assert after.startswith(before + b"\n") and after.endswith(b"\n")
    assert [row["score"] for row in log.rows()] == [5, 3]

    def append(index: int) -> int:
        return log.append({"model": "l2", "view": "samples", "score": 1 + index % 5})

    with ThreadPoolExecutor(max_workers=8) as pool:
        counts = list(pool.map(append, range(64)))
    assert sorted(counts) == list(range(3, 67))
    assert path.stat().st_ino == inode and path.read_bytes().startswith(after)
    lines = path.read_bytes().splitlines()
    assert len(lines) == 67 and lines[1] == b'{"model": "l2", "vi'
    assert all(json.loads(line)["model"] == "l2" for line in lines[2:])
    assert len(log.rows()) == 66


def test_unknown_models_wrong_kinds_icons_and_ranges_are_refused(gallery: Gallery) -> None:
    hexcode = sorted(gallery.programs)[0]
    rgb = np.full((32, 32, 3), 255, dtype=np.uint8)
    with pytest.raises(GalleryError, match="unknown model"):
        gallery.transcribe("nope", rgb)
    with pytest.raises(GalleryError, match="unknown model"):
        gallery.sample("nope", 1, 0, 1.0)
    with pytest.raises(GalleryError, match="is a latent, not a transcriber"):
        gallery.transcribe("l2", rgb)
    with pytest.raises(GalleryError, match="is a latent, not a transcriber"):
        gallery.input_size("l2")
    with pytest.raises(GalleryError, match="is a transcriber, not a latent"):
        gallery.reconstruct("v9", hexcode)
    with pytest.raises(GalleryError, match="is a transcriber, not a latent"):
        gallery.interpolate("v9", hexcode, hexcode, 3)
    with pytest.raises(GalleryError, match="is a transcriber, not a latent"):
        gallery.sample("v9", 1, 0, 1.0)
    with pytest.raises(GalleryError, match="unknown held-out icon"):
        gallery.reconstruct("l2", "NOT-AN-ICON")
    with pytest.raises(GalleryError, match="unknown held-out icon"):
        gallery.interpolate("l2", hexcode, "NOT-AN-ICON", 3)
    with pytest.raises(GalleryError, match="steps"):
        gallery.interpolate("l2", hexcode, hexcode, 12)
    with pytest.raises(GalleryError, match="count"):
        gallery.sample("l2", 17, 0, 1.0)
    with pytest.raises(GalleryError, match="scale"):
        gallery.sample("l2", 1, 0, 2.5)
    with pytest.raises(GalleryError, match="seed"):
        gallery.sample("l2", 1, -1, 1.0)
    with pytest.raises(GalleryError, match="32x32"):
        gallery.transcribe("v9", np.full((16, 16, 3), 255, dtype=np.uint8))
    with pytest.raises(GalleryError, match="held-out"):
        gallery.icon_png("../../etc/passwd")


def test_icons_are_held_out_and_searchable(gallery: Gallery) -> None:
    everything = gallery.icons_matching("")
    assert everything["total"] == len(gallery.icons) and len(everything["icons"]) == 150
    assert all(set(icon) == set(ICON_FIELDS) for icon in gallery.icons)
    assert all(icon["split"] != "primary/train" for icon in gallery.icons)
    joker = gallery.icons_matching("  JOKER ")
    assert joker["total"] >= 1 and all("joker" in i["annotation"] for i in joker["icons"])
    # A larger limit reaches every match (random picks use it); it clamps to 1..all.
    assert len(gallery.icons) > 150
    assert gallery.icons_matching("", limit=100_000)["icons"] == gallery.icons
    assert len(gallery.icons_matching("", limit=0)["icons"]) == 1
    assert len(gallery.icons_matching("", limit=7)["icons"]) == 7


# --------------------------------------------------------------------------- HTTP


@pytest.fixture(scope="module")
def address(gallery: Gallery) -> Iterator[tuple[str, int]]:
    server = make_server(gallery, "127.0.0.1", 0)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        host, port = cast(tuple[str, int], server.server_address[:2])
        yield host, port
    finally:
        server.shutdown()
        server.server_close()


def _call(
    address: tuple[str, int], method: str, path: str, body: bytes | None = None
) -> tuple[int, str, bytes]:
    connection = http.client.HTTPConnection(*address, timeout=120)
    try:
        headers = {"Content-Type": "application/json"} if body is not None else {}
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        return response.status, response.getheader("Content-Type") or "", response.read()
    finally:
        connection.close()


def _post(address: tuple[str, int], path: str, payload: object) -> tuple[int, Any]:
    status, kind, raw = _call(address, "POST", path, json.dumps(payload).encode())
    assert kind == "application/json"
    return status, json.loads(raw)


def _data_url(rgb: np.ndarray) -> str:
    buffer = io.BytesIO()
    Image.fromarray(rgb).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


def test_http_get_endpoints(
    gallery: Gallery,
    address: tuple[str, int],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    status, _, raw = _call(address, "GET", "/health")
    assert status == 200 and json.loads(raw) == {"ok": True, "models": 2}
    status, _, raw = _call(address, "GET", "/models")
    assert status == 200 and json.loads(raw) == gallery.models_payload()
    status, _, raw = _call(address, "GET", "/icons?q=joker")
    found = json.loads(raw)
    assert status == 200 and found["total"] >= 1 and set(found["icons"][0]) == set(ICON_FIELDS)
    status, _, raw = _call(address, "GET", "/icons?q=&limit=100000")
    found = json.loads(raw)
    assert status == 200 and len(found["icons"]) == found["total"] == len(gallery.icons)
    status, kind, raw = _call(address, "GET", "/icons?q=&limit=many")
    assert status == 400 and kind == "application/json" and "limit" in json.loads(raw)["error"]
    hexcode = str(gallery.icons[0]["hexcode"])
    status, kind, raw = _call(address, "GET", f"/render/{hexcode}.png")
    assert status == 200 and kind == "image/png"
    with Image.open(io.BytesIO(raw)) as image:
        assert image.size == (144, 144)
    status, kind, raw = _call(address, "GET", "/render/NOT-AN-ICON.png")
    assert status == 404 and kind == "application/json" and not json.loads(raw)["ok"]
    page = tmp_path / "index.html"
    page.write_text("<!doctype html><title>gallery</title>")
    monkeypatch.setattr(gallery_server, "PAGE", page)
    status, kind, raw = _call(address, "GET", "/")
    assert status == 200 and kind.startswith("text/html") and raw == page.read_bytes()
    status, _, _ = _call(address, "GET", "/nope")
    assert status == 404


def test_http_transcribe_and_latent_endpoints(gallery: Gallery, address: tuple[str, int]) -> None:
    png = _data_url(_icon_rgb(gallery, 144))
    status, body = _post(address, "/transcribe", {"model": "v9", "png": png, "candidates": 1})
    assert status == 200 and set(body) == _TRANSCRIBE_KEYS and body["ok"]
    _valid(body["svg"])
    first, second = sorted(gallery.programs)
    status, body = _post(address, "/latent/reconstruct", {"model": "l2", "hexcode": first})
    assert status == 200 and body["ok"]
    _valid(body["svg"])
    status, body = _post(
        address, "/latent/interpolate", {"model": "l2", "a": first, "b": second, "steps": 3}
    )
    assert status == 200 and body["ok"] and len(body["frames"]) == 3 and body["path"] == "lerp"
    status, straight = _post(
        address,
        "/latent/interpolate",
        {"model": "l2", "a": first, "b": second, "steps": 3, "path": "lerp"},
    )
    assert status == 200 and straight["frames"] == body["frames"]
    request = {"model": "l2", "count": 2, "seed": 3, "scale": 0.5}
    status, body = _post(address, "/latent/sample", request)
    assert status == 200 and body["ok"] and len(body["samples"]) == 2
    assert _post(address, "/latent/sample", request)[1]["samples"] == body["samples"]


def test_http_rating_endpoints(
    gallery: Gallery, address: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    log = cast(RatingLog, gallery.ratings)
    count = len(log.rows())
    status, body = _post(address, "/rating", _rating(score=5, note="clean shapes"))
    assert status == 200 and body == {"ok": True, "count": count + 1}
    status, kind, raw = _call(address, "GET", "/ratings")
    ratings = json.loads(raw)
    assert status == 200 and kind == "application/json" and set(ratings) == {"ratings", "summary"}
    assert ratings["ratings"][-1]["note"] == "clean shapes"
    assert ratings["ratings"][-1]["run_id"] == _SPECS["l2"].run_id
    assert ratings["summary"] == gallery.ratings_payload()["summary"]
    assert ratings["summary"]["l2"]["samples"]["n"] >= 1
    refused = [
        (_rating(score=9), "score"),
        (_rating(model="nope"), "unknown model"),
        ({"model": "l2", "view": "samples", "score": 3}, "missing field"),
        (_rating(view="everything"), "view"),
    ]
    for payload, message in refused:
        status, body = _post(address, "/rating", payload)
        assert status == 400 and body["ok"] is False and message in body["error"], body
    status, _, raw = _call(address, "POST", "/rating", b"[1]")
    assert status == 400 and not json.loads(raw)["ok"]
    assert len(log.rows()) == count + 1

    monkeypatch.setattr(gallery, "ratings", None)
    status, body = _post(address, "/rating", _rating())
    assert status == 503 and body["ok"] is False
    status, _, raw = _call(address, "GET", "/ratings")
    assert status == 503 and not json.loads(raw)["ok"]


def test_http_refusals_are_400_413_and_decoding_failures_are_200_not_ok(
    gallery: Gallery, address: tuple[str, int], monkeypatch: pytest.MonkeyPatch
) -> None:
    png = _data_url(np.full((32, 32, 3), 255, dtype=np.uint8))
    hexcode = sorted(gallery.programs)[0]
    refused = [
        ("/transcribe", {"model": "nope", "png": png}, "unknown model"),
        ("/transcribe", {"model": "l2", "png": png}, "not a transcriber"),
        ("/transcribe", {"model": "v9"}, "missing field"),
        ("/transcribe", {"model": "v9", "png": "data:image/png;base64,AAAA"}, ""),
        ("/transcribe", {"model": "v9", "png": png, "candidates": 0}, "candidates"),
        ("/latent/reconstruct", {"model": "v9", "hexcode": hexcode}, "not a latent"),
        ("/latent/reconstruct", {"model": "l2", "hexcode": "NOPE"}, "unknown held-out icon"),
        ("/latent/interpolate", {"model": "l2", "a": hexcode, "b": hexcode, "steps": 2}, "steps"),
        (
            "/latent/interpolate",
            {"model": "l2", "a": hexcode, "b": hexcode, "steps": 3, "path": "slerp"},
            "path must be one of",
        ),
        (
            "/latent/interpolate",
            {"model": "l2", "a": hexcode, "b": hexcode, "steps": 3, "path": 1},
            "path must be a string",
        ),
        ("/latent/sample", {"model": "l2", "count": 1, "seed": 0}, "missing field"),
        ("/latent/sample", {"model": "l2", "count": "2", "seed": 0, "scale": 1}, "integer"),
        ("/latent/sample", {"model": "nope", "count": 1, "seed": 0, "scale": 1}, "unknown model"),
    ]
    for path, payload, message in refused:
        status, body = _post(address, path, payload)
        assert status == 400 and body["ok"] is False and message in body["error"], (path, body)
    status, kind, raw = _call(address, "POST", "/transcribe", b"{not json")
    assert status == 400 and not json.loads(raw)["ok"]
    status, _, raw = _call(address, "POST", "/transcribe", b"[1, 2]")
    assert status == 400 and not json.loads(raw)["ok"]

    connection = http.client.HTTPConnection(*address, timeout=30)
    try:
        connection.putrequest("POST", "/transcribe")
        connection.putheader("Content-Length", str(MAX_BODY + 1))
        connection.endheaders()
        response = connection.getresponse()
        assert response.status == 413 and not json.loads(response.read())["ok"]
    finally:
        connection.close()

    def boom(*args: object) -> dict[str, Any]:
        raise RuntimeError("boom")

    monkeypatch.setattr(gallery, "sample", boom)
    status, body = _post(
        address, "/latent/sample", {"model": "l2", "count": 1, "seed": 0, "scale": 1}
    )
    assert status == 200 and body == {"ok": False, "error": "RuntimeError: boom"}
    status, _, _ = _call(address, "GET", "/health")
    assert status == 200


# --------------------------------------------------------------------------- CUDA graphs


@pytest.fixture(scope="module")
def cuda_setup(parts: Parts) -> tuple[Gallery, RenderToProgram, LatentToProgram]:
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need a GPU")
    layout, template, programs = parts
    transcriber, latent = _tiny_models(layout)
    transcriber, latent = transcriber.cuda(), latent.cuda()
    gallery = Gallery(
        [(_SPECS["v9"], transcriber), (_SPECS["l2"], latent)],
        template,
        programs,
        torch.device("cuda"),
    )
    return gallery, transcriber, latent


@cuda
def test_cuda_graph_paths_decode_what_the_reference_decoders_decode(
    cuda_setup: tuple[Gallery, RenderToProgram, LatentToProgram], parts: Parts
) -> None:
    """The paths the real demo runs: greedy and best of 8 by graph replay, and a graph
    decoded reconstruction, each equal to its reference decoder's output."""

    from mojidiff.learning.fast_decode import GraphDecoder, rerank

    gallery, transcriber, latent = cuda_setup
    layout, template, _ = parts
    backend = gallery._models["l2"].backend
    # The vae backend reconstructs by a captured graph, not by the reference decoder.
    assert isinstance(backend, VaeBackend) and isinstance(backend.graph, GraphDecoder)
    rgb = _icon_rgb(gallery, gallery.input_size("v9"))
    image = torch.from_numpy(rgb.copy()).cuda()

    greedy = gallery.transcribe("v9", rgb, 1)
    assert greedy["ok"] and greedy["candidates"] == 1 and greedy["decoder_calls"] > 0
    assert greedy["svg"] == _svg_of(greedy_decode(transcriber, image[None])[0], template, layout)

    best = gallery.transcribe("v9", rgb, 8)
    assert best["ok"] and best["candidates"] == 8 and best["decoder_calls"] > 0
    _valid(best["svg"])
    reference, _ = rerank(
        GraphDecoder(transcriber, dtype=torch.float32, batch=8),
        image,
        template,
        temperature=gallery_server.RERANK_TEMPERATURE,
        seed=gallery_server.RERANK_SEED,
    )
    assert best["svg"] == _svg_of(reference[0], template, layout)

    first, second = sorted(gallery.programs)
    with torch.no_grad():
        mean, _ = latent.posterior(gallery.programs[first][None].cuda())
        expected = _svg_of(greedy_decode(latent, mean)[0], template, layout)
    assert gallery.reconstruct("l2", first)["svg"] == expected
    frames = gallery.interpolate("l2", first, second, 3)["frames"]
    assert len(frames) == 3
    for frame in frames:
        _valid(frame)
    samples = gallery.sample("l2", 4, seed=11, scale=1.0)["samples"]
    assert len(samples) == 4 and gallery.sample("l2", 4, seed=11, scale=1.0)["samples"] == samples


@cuda
def test_cuda_concurrent_requests_equal_their_sequential_answers(
    cuda_setup: tuple[Gallery, RenderToProgram, LatentToProgram],
) -> None:
    """Graph replays share device buffers: under the locks, requests from many threads
    at once must answer exactly what they answer one at a time."""

    gallery, _, _ = cuda_setup
    size = gallery.input_size("v9")
    first, second = sorted(gallery.programs)
    calls: list[Callable[[], Any]] = [
        lambda: gallery.transcribe("v9", _icon_rgb(gallery, size, 0), 1)["svg"],
        lambda: gallery.transcribe("v9", _icon_rgb(gallery, size, 1), 1)["svg"],
        lambda: gallery.transcribe("v9", _icon_rgb(gallery, size, 2), 8)["svg"],
        lambda: gallery.reconstruct("l2", first)["svg"],
        lambda: gallery.reconstruct("l2", second)["svg"],
        lambda: gallery.interpolate("l2", first, second, 3)["frames"],
        lambda: gallery.sample("l2", 3, seed=5, scale=0.8)["samples"],
    ]
    sequential = [call() for call in calls]
    with ThreadPoolExecutor(max_workers=len(calls)) as pool:
        for _ in range(3):
            futures = [pool.submit(call) for call in calls + calls[::-1]]
            answers = [future.result() for future in futures]
            assert answers == sequential + sequential[::-1]
