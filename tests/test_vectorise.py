"""The re-vectorise demo's contract: a canvas PNG in, a valid codec SVG out."""

from __future__ import annotations

import io
import json
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


def test_the_icon_cache_builder_reproduces_the_cache_the_demos_read(tmp_path: Path) -> None:
    """`icon_cache` is how a fresh checkout (the Hugging Face Space) gets
    `data/processed/kitbash`: the codec's canonical SVG of each source, sha256-checked
    against the ledger, and an index in split order."""

    from mojidiff.vectorise.icon_cache import INDEX, SPLITS, build_icon_cache
    from mojidiff.vectorise.server import ICON_CACHE

    written = build_icon_cache(tmp_path, _CONFIG, limit=2)
    assert written == 2 * len(SPLITS)
    index = json.loads((tmp_path / INDEX).read_text())
    assert [icon["split"] for icon in index] == [split for split in SPLITS for _ in range(2)]
    fields = {"hexcode", "annotation", "split", "group", "subgroup"}
    assert all(set(icon) == fields for icon in index)
    for icon in index:
        svg = (tmp_path / f"{icon['hexcode']}.svg").read_bytes()
        validate_typed_svg(svg, RenderLimits(max_paths=80))
        # Byte-identical to the cache the demos have served since it was first built.
        assert svg == (ICON_CACHE / f"{icon['hexcode']}.svg").read_bytes()
    full = {entry["hexcode"]: entry for entry in json.loads((ICON_CACHE / INDEX).read_text())}
    assert all(full[icon["hexcode"]] == icon for icon in index)
    assert not list(tmp_path.glob("*.tmp"))


def _tiny_vectoriser() -> Vectoriser:
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
    return Vectoriser(model, template, torch.device("cpu"))


def test_cpu_best_of_8_is_rerank_decode_and_counts_its_decoder_calls() -> None:
    """Without CUDA graphs (the CPU Space) best of 8 picks what `rerank_decode` picks, and
    reports the decoder calls of all eight candidates instead of 0."""

    from mojidiff.learning.render2svg import DecodeStats, rerank_decode

    vectoriser = _tiny_vectoriser()
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    image = torch.from_numpy(rgb)
    stats = DecodeStats()
    with torch.no_grad():
        chosen = vectoriser._rerank(image, stats)
        expected, _ = rerank_decode(
            vectoriser.model, image, vectoriser.template, candidates=8, temperature=0.7
        )
    assert torch.equal(chosen, expected) and stats.model_calls > 0
    greedy = vectoriser.vectorise(rgb)
    best = vectoriser.vectorise(rgb, candidates=8)
    assert best["ok"] and best["decoder_calls"] >= greedy["decoder_calls"] > 0


def test_config_names_the_decode_device_for_the_page() -> None:
    """`/config` says `cpu` on a GPU-less server, and the page then names CPU, not the
    RTX 4080, as where its times come from."""

    import functools
    import threading
    import urllib.request

    from mojidiff.vectorise.server import PAGE, _Handler, _Server

    vectoriser = _tiny_vectoriser()
    handler = type(
        "Handler", (_Handler,), {"vectoriser": vectoriser, "icons": [], "palette": ["#000000"]}
    )
    with _Server(("127.0.0.1", 0), functools.partial(handler)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            host, port = server.server_address[:2]
            with urllib.request.urlopen(f"http://{host!s}:{port}/config", timeout=30) as reply:
                config = json.loads(reply.read())
        finally:
            server.shutdown()
    assert config == {"size": 32, "palette": ["#000000"], "device": "cpu"}
    page = PAGE.read_text()
    assert '<span id="where">on the RTX 4080</span>' in page
    assert "config.device === 'cpu'" in page
