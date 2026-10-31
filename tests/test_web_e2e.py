"""End to end: the JavaScript decoder on ONNX Runtime returns the Python reference's programs.

`mojidiff.learning.web_e2e` decodes validation renders with the Python ONNX Runtime decoder
and writes a reference bundle; `web/e2e/node.mjs` decodes the same pixels with the
JavaScript grammar, greedy loop and codec on onnxruntime-node and must reproduce every
token, call count and SVG byte. Pinned here on a tiny random export (whenever Node and
web/node_modules are present) and on the final model's export (when it is on this machine).
`web/e2e/browser.mjs` runs the same check in headless Chrome when MOJIDIFF_BROWSER_TEST=1.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnxruntime")

from mojidiff.learning.autoregressive import SequenceLayout, flatten_program  # noqa: E402
from mojidiff.learning.onnx_decode import OnnxDecoder  # noqa: E402
from mojidiff.learning.onnx_export import (  # noqa: E402
    DECODER_FILE,
    DEFAULT_PILOT_CONFIG,
    ENCODER_FILE,
    MANIFEST_FILE,
    export_model,
)
from mojidiff.learning.openmoji_pilot import (  # noqa: E402
    _load_program,
    _select_rows,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.render2svg import (  # noqa: E402
    CACHE_ROOT,
    DecodeStats,
    ModelConfig,
    RenderToProgram,
    SplitData,
    _cache_key,
    render_trusted_rgb,
)
from mojidiff.learning.web_e2e import (  # noqa: E402
    IMAGES_FILE,
    REFERENCE_FILE,
    build_reference,
    main,
    reference_icons,
    spread_indices,
    write_bundle,
)
from mojidiff.learning.web_reference import program_svg  # noqa: E402
from mojidiff.representation.packed import PackedTensorProgram, serialize_packed_svg  # noqa: E402

_WEB = Path(__file__).resolve().parents[1] / "web"
_FINAL_RUN = "r2s-cur-d02-s120k-seed2-53b3108-46294452-47646604"
_TINY = ModelConfig(
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

Tiny = tuple[Path, SplitData, SequenceLayout, PackedTensorProgram]


def _node() -> str | None:
    node = os.environ.get("MOJIDIFF_NODE") or shutil.which("node")
    if node is None:
        return None
    version = subprocess.run([node, "--version"], capture_output=True, text=True, check=False)
    match = re.match(r"v(\d+)\.", version.stdout.strip())
    return node if match and int(match.group(1)) >= 20 else None


def _node_modules(*packages: str) -> Path | None:
    """The first package root holding every package, searched as web/e2e/modules.mjs does."""

    roots = [Path(p) for p in [os.environ.get("MOJIDIFF_NODE_MODULES")] if p]
    for root in [*roots, _WEB / "node_modules"]:
        if all((root / name / "package.json").exists() for name in packages):
            return root.resolve()
    return None


def _node_or_skip(*packages: str) -> tuple[str, dict[str, str]]:
    node = _node()
    if node is None:
        pytest.skip("Node.js 20 or newer is not available (set MOJIDIFF_NODE)")
    modules = _node_modules(*packages)
    if modules is None:
        pytest.skip(f"{', '.join(packages)} not installed (npm ci --ignore-scripts in web/)")
    return node, {**os.environ, "MOJIDIFF_NODE_MODULES": str(modules)}


def _run_node(
    node: str, env: dict[str, str], model: Path, reference: Path, out: Path, *extra: str
) -> tuple[int, dict[str, Any], str]:
    command = [node, "e2e/node.mjs", "--model", str(model), "--reference", str(reference)]
    command += ["--json", str(out), "--threads", "2", *extra]
    result = subprocess.run(
        command, cwd=_WEB, env=env, capture_output=True, text=True, timeout=900, check=False
    )
    data = json.loads(out.read_text()) if out.exists() else {}
    return result.returncode, data, result.stderr[-3000:]


@pytest.fixture(scope="module")
def tiny(tmp_path_factory: pytest.TempPathFactory) -> Tiny:
    """A tiny random export and three validation programs rendered at its input size."""

    config = load_openmoji_pilot_config(DEFAULT_PILOT_CONFIG)
    codec = _selected_codec(config)
    layout = SequenceLayout(codec=codec, total_segment_slots=config.total_segment_slots)
    by_split, _, _ = load_pilot_index(config)
    template = _load_program(by_split["primary/train"][0], config, codec)
    rows = _select_rows(by_split["primary/validation"], 3, config.seed + 1)
    programs = [_load_program(row, config, codec) for row in rows]
    renders = [
        render_trusted_rgb(
            serialize_packed_svg(p, codec, layout.total_segment_slots), _TINY.image_size
        )
        for p in programs
    ]
    images = torch.from_numpy(np.stack(renders))
    split = SplitData(
        hexcodes=[f"fixture-{i}" for i in range(len(programs))],
        tokens=torch.stack([flatten_program(p, layout) for p in programs]),
        masks=np.zeros((len(programs), 1), dtype=np.uint8),
        images=images,
        targets=images,
        shape=(layout.length, layout.vocabulary),
    )
    torch.manual_seed(11)
    model = RenderToProgram(layout, _TINY).eval()
    directory = tmp_path_factory.mktemp("tiny-export")
    export_model(model, directory, provenance={"run_id": "tiny-run"})
    return directory, split, layout, template


def test_spread_indices_cover_the_split_evenly() -> None:
    assert spread_indices(339, 32)[:4] == [0, 11, 22, 33]
    assert spread_indices(339, 32)[-1] == 338 and len(spread_indices(339, 32)) == 32
    assert spread_indices(339, 16) == [
        0, 23, 45, 68, 90, 113, 135, 158, 180, 203, 225, 248, 270, 293, 315, 338
    ]  # fmt: skip
    assert spread_indices(5, 10) == [0, 1, 2, 3, 4]
    assert spread_indices(10, 1) == [0]
    with pytest.raises(ValueError):
        spread_indices(0, 3)
    with pytest.raises(ValueError):
        spread_indices(3, 0)


def test_bundle_records_the_python_onnx_decodes(tiny: Tiny, tmp_path: Path) -> None:
    """The bundle holds exactly the corpus pixels and what OnnxDecoder decodes from them."""

    directory, split, layout, template = tiny
    decoder = OnnxDecoder(directory, layout, threads=2)
    indices = [0, 2]
    icons = reference_icons(decoder, split, indices, template, layout)
    record = write_bundle(tmp_path, directory, split, indices, icons, {"dataset_sha256": "x"})
    images = np.frombuffer((tmp_path / IMAGES_FILE).read_bytes(), dtype=np.uint8)
    assert np.array_equal(images.reshape(2, 32, 32, 3), split.images[indices].numpy())
    assert record == json.loads((tmp_path / REFERENCE_FILE).read_text())
    assert record["run_id"] == "tiny-run" and record["dataset_sha256"] == "x"
    assert record["images"]["shape"] == [2, 32, 32, 3]
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    assert record["model"]["decoder_step_sha256"] == manifest["files"]["decoder_step"]["sha256"]
    for index, icon in zip(indices, record["icons"], strict=True):
        stats = DecodeStats()
        tokens = decoder.decode(split.images[index].numpy(), stats=stats)
        assert icon["index"] == index and icon["hexcode"] == split.hexcodes[index]
        assert icon["tokens"] == tokens.tolist()
        assert icon["model_calls"] == stats.model_calls > 0
        assert icon["svg"] == program_svg(tokens, template, layout)


def test_node_decodes_the_tiny_export_as_python_does(tiny: Tiny, tmp_path: Path) -> None:
    """onnxruntime-node under the JS grammar gives Python's tokens, calls and SVG; one
    altered reference token makes the script report that icon and exit 1, and a bundle
    from other model files is refused."""

    node, env = _node_or_skip("onnxruntime-node")
    directory, split, layout, template = tiny
    decoder = OnnxDecoder(directory, layout, threads=2)
    indices = [0, 1, 2]
    icons = reference_icons(decoder, split, indices, template, layout)
    good = tmp_path / "good"
    write_bundle(good, directory, split, indices, icons, {})
    code, result, stderr = _run_node(node, env, directory, good, tmp_path / "good.json")
    assert code == 0, stderr
    summary = result["summary"]
    assert summary["icons"] == summary["identical_tokens"] == 3
    assert summary["identical_svg"] == summary["equal_calls"] == 3
    assert summary["torch_checked"] == 0  # the committed fixtures are another checkpoint's
    assert [row["calls"] for row in result["rows"]] == [icon["model_calls"] for icon in icons]
    assert summary["grammar_share"] < 1

    icons[1]["tokens"][5] += 1
    bad = tmp_path / "bad"
    write_bundle(bad, directory, split, indices, icons, {})
    code, result, _ = _run_node(node, env, directory, bad, tmp_path / "bad.json")
    assert code == 1
    assert [row["identical"] for row in result["rows"]] == [True, False, True]
    assert result["rows"][1]["first_difference"] == 5

    record = json.loads((good / REFERENCE_FILE).read_text())
    record["model"]["encoder_sha256"] = "0" * 64
    (bad / REFERENCE_FILE).write_text(json.dumps(record))
    code, _, stderr = _run_node(node, env, directory, bad, tmp_path / "other.json")
    assert code != 0 and "another encoder" in stderr


@pytest.mark.skipif(
    os.environ.get("MOJIDIFF_BROWSER_TEST") != "1",
    reason="headless Chrome: source web/e2e/gpubox-env.sh and set MOJIDIFF_BROWSER_TEST=1",
)
def test_browser_harness_decodes_the_tiny_export_as_python_does(tiny: Tiny, tmp_path: Path) -> None:
    """web/e2e/browser.mjs: page.html in headless Chrome, WASM on one thread."""

    node, env = _node_or_skip("onnxruntime-web", "playwright")
    directory, split, layout, template = tiny
    decoder = OnnxDecoder(directory, layout, threads=2)
    icons = reference_icons(decoder, split, [0, 1], template, layout)
    write_bundle(tmp_path / "reference", directory, split, [0, 1], icons, {})
    command = [node, "e2e/browser.mjs", "--model", str(directory), "--configs", "wasm1"]
    command += ["--reference", str(tmp_path / "reference"), "--out", str(tmp_path / "out")]
    result = subprocess.run(
        command, cwd=_WEB, env=env, capture_output=True, text=True, timeout=600, check=False
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    run = json.loads((tmp_path / "out" / "wasm1.json").read_text())
    assert run["crossOriginIsolated"] is True and "error" not in run
    assert run["summary"]["icons"] == run["summary"]["identical_tokens"] == 2
    assert run["summary"]["identical_svg"] == run["summary"]["equal_calls"] == 2
    assert [row["calls"] for row in run["icons"]] == [icon["model_calls"] for icon in icons]


def _final_export() -> Path | None:
    directory = CACHE_ROOT / "onnx" / _FINAL_RUN
    corpus = CACHE_ROOT / f"render2svg-corpus-{_cache_key(DEFAULT_PILOT_CONFIG, 144, True)}.npz"
    files = (MANIFEST_FILE, ENCODER_FILE, DECODER_FILE)
    ready = corpus.exists() and all((directory / name).exists() for name in files)
    return directory if ready else None


@pytest.mark.skipif(_final_export() is None, reason="final model's export or corpus missing")
def test_final_model_decodes_in_node_as_in_python(tmp_path: Path) -> None:
    """The 9.0M final model, 16 validation icons spread over the split: the JS decoder on
    onnxruntime-node reproduces the Python ORT tokens, calls and SVG bytes, and torch
    greedy_decode's programs in the committed fixtures."""

    node, env = _node_or_skip("onnxruntime-node")
    directory = _final_export()
    assert directory is not None
    reference = tmp_path / "reference"
    assert main(["--run", _FINAL_RUN, "--icons", "16", "--out", str(reference)]) == 0
    record = json.loads((reference / REFERENCE_FILE).read_text())
    assert [icon["index"] for icon in record["icons"]] == spread_indices(339, 16)
    assert np.median([icon["model_calls"] for icon in record["icons"]]) > 400  # not light
    code, result, stderr = _run_node(node, env, directory, reference, tmp_path / "node.json")
    assert code == 0, stderr
    summary = result["summary"]
    assert summary["icons"] == summary["identical_tokens"] == summary["identical_svg"] == 16
    assert summary["equal_calls"] == summary["torch_checked"] == summary["torch_identical"] == 16


def test_build_reference_reads_the_pilot_config_from_the_manifest(
    tiny: Tiny, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """build_reference on a tiny export, with load_corpus replaced by the fixture split."""

    from mojidiff.learning import render2svg

    directory, split, layout, _ = tiny
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    manifest["pilot_config"] = str(DEFAULT_PILOT_CONFIG)
    model_dir = tmp_path / "model"
    shutil.copytree(directory, model_dir)
    (model_dir / MANIFEST_FILE).write_text(json.dumps(manifest))
    seen: list[tuple[Path, int]] = []

    def fake_corpus(
        pilot_config: Path, image_size: int, **_: object
    ) -> tuple[dict[str, SplitData], dict[str, SplitData], SequenceLayout, Any, str]:
        seen.append((pilot_config, image_size))
        pilot = load_openmoji_pilot_config(pilot_config)
        return {"primary/validation": split}, {}, layout, pilot, "fixture-corpus"

    monkeypatch.setattr(render2svg, "load_corpus", fake_corpus)
    record = build_reference(model_dir, tmp_path / "out", icons=2, threads=1)
    assert seen == [(DEFAULT_PILOT_CONFIG, 32)]
    assert [icon["index"] for icon in record["icons"]] == [0, 2]
    assert record["dataset_sha256"] == "fixture-corpus"
    assert record["split_size"] == 3
