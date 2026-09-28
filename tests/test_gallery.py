"""The model gallery's contract: every registered run, every endpoint, on tiny CPU models."""

from __future__ import annotations

import base64
import http.client
import io
import json
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch
from PIL import Image

import mojidiff.gallery.server as gallery_server
import mojidiff.learning.render2svg as render2svg
from mojidiff.gallery.server import (
    BASELINE_PIXEL_ERROR,
    ICON_FIELDS,
    MAX_BODY,
    MODELS,
    Gallery,
    GalleryError,
    make_server,
    model_entry,
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
from mojidiff.learning.render2svg import (
    ModelConfig,
    RenderToProgram,
    greedy_decode,
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
def gallery(parts: Parts, tiny: tuple[RenderToProgram, LatentToProgram]) -> Gallery:
    _, template, programs = parts
    transcriber, latent = tiny
    return Gallery(
        [(_SPECS["v9"], transcriber), (_SPECS["l2"], latent)],
        template,
        programs,
        torch.device("cpu"),
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
    for spec in MODELS:
        assert spec.checkpoint.name == "best.pt" and spec.checkpoint.parent.name == spec.run_id


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
    assert set(path) == {"ok", "frames", "ms"} and len(path["frames"]) == 3
    for frame in path["frames"]:
        _valid(frame)
    samples = gallery.sample("l2", 3, seed=7, scale=1.0)
    assert set(samples) == {"ok", "samples", "ms"} and len(samples["samples"]) == 3
    for sample in samples["samples"]:
        _valid(sample)
    assert gallery.sample("l2", 3, seed=7, scale=1.0)["samples"] == samples["samples"]


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
    assert status == 200 and body["ok"] and len(body["frames"]) == 3
    request = {"model": "l2", "count": 2, "seed": 3, "scale": 0.5}
    status, body = _post(address, "/latent/sample", request)
    assert status == 200 and body["ok"] and len(body["samples"]) == 2
    assert _post(address, "/latent/sample", request)[1]["samples"] == body["samples"]


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
