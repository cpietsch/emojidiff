"""The browser export: ONNX Runtime must compute what the torch model computes.

Pinned here, on a tiny random model with v9's layout (metric coordinates, path order):
the exported decoder step's logits follow the model's own teacher-forced logits over a
whole program, chunk by chunk through the growing cache; greedy decoding through ONNX
Runtime under the Python grammar returns exactly `greedy_decode`'s program with the same
number of model calls; the step rules equal the batched tensor rules; and the graphs keep
milestone 0's browser contract (float32 decoder inputs, no Range, And or Einsum) and carry
none of the exporter's path-bearing debug metadata. When the final checkpoint and its
corpus are on this machine, its export decodes four validation icons exactly as torch does.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from itertools import cycle
from pathlib import Path

import numpy as np
import pytest
import torch

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxscript")
pytest.importorskip("onnxruntime")

from mojidiff.learning import render2svg  # noqa: E402
from mojidiff.learning.autoregressive import (  # noqa: E402
    SequenceLayout,
    flatten_program,
    unflatten_program,
)
from mojidiff.learning.onnx_decode import OnnxDecoder, StepRules  # noqa: E402
from mojidiff.learning.onnx_export import (  # noqa: E402
    DECODER_FILE,
    DECODER_INPUTS,
    DECODER_OUTPUTS,
    DEFAULT_PILOT_CONFIG,
    ENCODER_FILE,
    ENCODER_INPUTS,
    ENCODER_OUTPUTS,
    MANIFEST_FILE,
    BrowserDecoderStep,
    EncoderRows,
    export_encoder,
    export_model,
    export_run,
    load_layout,
    main,
    run_pilot_config,
    step_parity,
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
    _static_tables,
    coordinate_roles,
    greedy_decode,
    path_major_order,
    position_labels,
    render_trusted_rgb,
)
from mojidiff.representation.packed import (  # noqa: E402
    PackedTensorProgram,
    serialize_packed_svg,
    validate_packed_tensor_program,
)

_TINY_PATH = ModelConfig(
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
_FINAL_RUN = "r2s-cur-d02-s120k-seed2-53b3108-46294452-47646604"

STEP_TOLERANCE = 1e-5
"""ONNX Runtime against the exported module in torch (measured 5e-7 on the tiny model;
4.8e-5 on the 9.0M final model, whose logits are larger)."""
MODEL_TOLERANCE = 1e-4
"""ONNX Runtime against `RenderToProgram` itself, which computes the Fourier features
without range reduction (measured 1.2e-6 on the tiny model; 2.4e-4 on the final one)."""

Pieces = tuple[SequenceLayout, list[PackedTensorProgram]]
Tiny = tuple[RenderToProgram, Path]


@pytest.fixture(scope="module")
def pieces() -> Pieces:
    config = load_openmoji_pilot_config(DEFAULT_PILOT_CONFIG)
    codec = _selected_codec(config)
    by_split, _, _ = load_pilot_index(config)
    rows = _select_rows(by_split["primary/validation"], 3, config.seed + 1)
    programs = [_load_program(row, config, codec) for row in rows]
    layout = SequenceLayout(codec=codec, total_segment_slots=config.total_segment_slots)
    return layout, programs


@pytest.fixture(scope="module")
def tiny(pieces: Pieces, tmp_path_factory: pytest.TempPathFactory) -> Tiny:
    layout, _ = pieces
    torch.manual_seed(5)
    model = RenderToProgram(layout, _TINY_PATH).eval()
    directory = tmp_path_factory.mktemp("tiny-onnx")
    export_model(model, directory)
    return model, directory


def _render(program: PackedTensorProgram, layout: SequenceLayout) -> np.ndarray:
    svg = serialize_packed_svg(program, layout.codec, layout.total_segment_slots)
    return render_trusted_rgb(svg, _TINY_PATH.image_size)


def test_step_logits_follow_the_teacher_forced_model(pieces: Pieces, tiny: Tiny) -> None:
    """Chunks of 1 to 32 steps through the cache, from step 0 to the last of 1,376."""

    layout, programs = pieces
    model, directory = tiny
    decoder = OnnxDecoder(directory, layout, threads=2)
    rules = StepRules(layout)
    for program in programs[:2]:
        image = _render(program, layout)
        tokens = flatten_program(program, layout)
        order = path_major_order(tokens, layout)
        with torch.no_grad():
            full = model(torch.from_numpy(image)[None], tokens[None], order[None])[0]
        decoded, steps = tokens.tolist(), order.tolist()
        memory_keys, memory_values = decoder.encode(image)
        past_keys, past_values = decoder.empty_cache(), decoder.empty_cache()
        start, worst, sizes = 0, 0.0, cycle((1, 2, 5, 1, 17, 32, 3))
        while start < layout.length:
            end = min(start + next(sizes), layout.length)
            ids = rules.ids(steps, decoded, start, end)[:, None]
            target = np.array([rules.target(steps[end - 1], decoded)], dtype=np.float32)
            logits, past_keys, past_values = decoder.step(
                ids, target, past_keys, past_values, memory_keys, memory_values
            )
            assert past_keys.shape[3] == past_values.shape[3] == end
            expected = full[steps[end - 1]]
            worst = max(worst, float((torch.from_numpy(logits[0]) - expected).abs().max()))
            start = end
        assert worst < MODEL_TOLERANCE


def test_step_matches_torch_at_milestone_zero_shapes(pieces: Pieces, tiny: Tiny) -> None:
    """Batch 1 to 8, chunks of 1 to 32, caches of 0 to 1,375 steps, random targets."""

    layout, _ = pieces
    model, directory = tiny
    worst = step_parity(model, OnnxDecoder(directory, layout, threads=2))
    assert max(worst["logits"], worst["keys"], worst["values"]) < STEP_TOLERANCE
    assert worst["logits_vs_model_decode"] < MODEL_TOLERANCE


def test_onnx_greedy_equals_greedy_decode(pieces: Pieces, tiny: Tiny) -> None:
    layout, programs = pieces
    model, directory = tiny
    decoder = OnnxDecoder(directory, layout, threads=2)
    for program in programs:
        image = _render(program, layout)
        torch_stats, onnx_stats = DecodeStats(), DecodeStats()
        reference = greedy_decode(model, torch.from_numpy(image)[None], stats=torch_stats)[0]
        tokens = decoder.decode(image, stats=onnx_stats)
        assert torch.equal(tokens, reference)
        assert onnx_stats.model_calls == torch_stats.model_calls > 0
        validate_packed_tensor_program(
            unflatten_program(tokens, program, layout), layout.codec, layout.total_segment_slots
        )


def test_step_rules_match_the_batched_rules(pieces: Pieces) -> None:
    layout, programs = pieces
    rules = StepRules(layout)
    tokens = torch.stack([flatten_program(program, layout) for program in programs])
    role, axis = coordinate_roles(tokens, _static_tables(layout))
    field, path = position_labels(tokens, layout)
    model = RenderToProgram(layout, _TINY_PATH)
    for row in range(tokens.shape[0]):
        decoded = tokens[row].tolist()
        for position in range(layout.length):
            got_role, got_axis = rules.role(position, decoded)
            assert got_role == int(role[row, position])
            if got_role:
                assert got_axis == int(axis[row, position])
            assert rules.labels(position, decoded) == (
                int(field[row, position]),
                int(path[row, position]),
            )
        order = path_major_order(tokens[row], layout)
        ids = rules.ids(order.tolist(), decoded, 0, layout.length)
        inputs, extras = model.step_inputs(tokens[row : row + 1], order[None], 0, layout.length)
        assert np.array_equal(ids[0], inputs[0].numpy())
        assert np.array_equal(ids[1], np.arange(layout.length))
        assert np.array_equal(ids[2], extras["field"][0].numpy())
        assert np.array_equal(ids[3], extras["path"][0].numpy())
        previous = order[:-1]
        assert np.array_equal(ids[5, 1:], role[row, previous].numpy())
        used = ids[5, 1:] > 0
        assert np.array_equal(ids[4, 1:][used], axis[row, previous].numpy()[used])
        targets = [rules.target(position, decoded) for position in order.tolist()]
        assert targets == extras["metric_target"][0].tolist()


def test_graphs_keep_the_browser_contract(pieces: Pieces, tiny: Tiny) -> None:
    """Milestone 0: int64 decoder inputs were silently corrupted on WebGPU; JSEP has no
    And kernel; Range put integer work on the CPU. The manifest must describe the files."""

    layout, _ = pieces
    model, directory = tiny
    manifest = json.loads((directory / MANIFEST_FILE).read_text())
    decoder, encoder = manifest["files"]["decoder_step"], manifest["files"]["encoder"]
    assert [entry["name"] for entry in decoder["inputs"]] == list(DECODER_INPUTS)
    assert [entry["name"] for entry in decoder["outputs"]] == list(DECODER_OUTPUTS)
    assert {entry["dtype"] for entry in decoder["inputs"] + decoder["outputs"]} == {"float32"}
    assert not {"Range", "And", "Einsum"} & set(decoder["ops"])
    assert [(entry["name"], entry["dtype"]) for entry in encoder["inputs"]] == [
        (ENCODER_INPUTS[0], "float32"),
        (ENCODER_INPUTS[1], "int64"),
    ]
    assert [entry["name"] for entry in encoder["outputs"]] == list(ENCODER_OUTPUTS)
    assert encoder["inputs"][0]["shape"] == [1, 32, 32, 3]
    head_dim = _TINY_PATH.d_model // _TINY_PATH.heads
    assert decoder["inputs"][2]["shape"] == [2, "batch", 4, "past", head_dim]
    assert decoder["outputs"][0]["shape"] == ["batch", layout.vocabulary]
    for role, name in (("encoder", ENCODER_FILE), ("decoder_step", DECODER_FILE)):
        data = (directory / name).read_bytes()
        assert manifest["files"][role]["bytes"] == len(data)
        assert manifest["files"][role]["sha256"] == hashlib.sha256(data).hexdigest()
        # No exporter debug metadata: it held stack traces with this checkout's paths.
        graph = onnx.load_from_string(data).graph
        assert not any(node.metadata_props for node in graph.node)
        assert b"stack_trace" not in data
        # Ids reach the embedding Gathers as int32 (index_select exported a cast to int64).
        casts = {
            attribute.i
            for node in graph.node
            if node.op_type == "Cast"
            for attribute in node.attribute
            if attribute.name == "to"
        }
        assert onnx.TensorProto.INT64 not in casts
    constants = manifest["constants"]
    assert (constants["vocabulary"], constants["length"]) == (layout.vocabulary, layout.length)
    assert constants["memory_length"] == model.memory_length == 16
    assert constants["ids_rows"] == ["tokens", "steps", "field", "path", "axis", "role"]
    assert manifest["model_config"] == asdict(_TINY_PATH)


class _Subclass(RenderToProgram):
    """Stands in for a subclass such as `LatentToProgram`, whose encoder differs."""


def test_export_refuses_models_it_would_get_wrong(pieces: Pieces) -> None:
    layout, _ = pieces
    for config in (replace(_TINY_PATH, order="layout"), replace(_TINY_PATH, metric=False)):
        with pytest.raises(ValueError):
            EncoderRows(RenderToProgram(layout, config))
    with pytest.raises(ValueError, match="only RenderToProgram"):
        BrowserDecoderStep(_Subclass(layout, _TINY_PATH))


def test_a_model_with_sources_exports_openmoji_memory(pieces: Pieces, tmp_path: Path) -> None:
    layout, programs = pieces
    torch.manual_seed(2)
    model = RenderToProgram(layout, replace(_TINY_PATH, sources=2)).eval()
    with torch.no_grad():
        model.source_embedding.weight.normal_()
    path = tmp_path / ENCODER_FILE
    export_encoder(model, path)
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    image = _render(programs[0], layout)
    keys, values = session.run(
        None, {"pixels": image[None].astype(np.float32), "rows": np.zeros(3, dtype=np.int64)}
    )
    with torch.no_grad():
        memory = model.encode(torch.from_numpy(image)[None])
    assert keys.shape == (2, 3, 4, 16, 8)
    for layer, (key, value) in enumerate(memory):
        for row in range(3):
            assert np.allclose(keys[layer, row], key[0].numpy(), atol=1e-5)
            assert np.allclose(values[layer, row], value[0].numpy(), atol=1e-5)


def test_cli_exports_a_checkpoint_with_its_provenance(pieces: Pieces, tmp_path: Path) -> None:
    layout, _ = pieces
    torch.manual_seed(4)
    model = RenderToProgram(layout, _TINY_PATH)
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "config": asdict(_TINY_PATH), "step": 3}, checkpoint)
    out = tmp_path / "onnx"
    arguments = ["--run", "tiny-run", "--out", str(out), "--checkpoint", str(checkpoint)]
    assert main([*arguments, "--pilot-config", str(DEFAULT_PILOT_CONFIG)]) == 0
    manifest = json.loads((out / MANIFEST_FILE).read_text())
    assert manifest["run_id"] == "tiny-run"
    assert manifest["checkpoint"]["step"] == 3
    assert manifest["checkpoint"]["sha256"] == hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    assert manifest["pilot_config"] == str(DEFAULT_PILOT_CONFIG)
    assert manifest["parameters"] == sum(p.numel() for p in model.parameters())
    assert manifest["checks"]["step_parity"]["logits"] < STEP_TOLERANCE
    assert "greedy" not in manifest["checks"]
    assert run_pilot_config(_FINAL_RUN) == DEFAULT_PILOT_CONFIG
    assert load_layout(DEFAULT_PILOT_CONFIG) == layout


def test_cli_check_icons_records_greedy_parity_and_fails_on_a_difference(
    pieces: Pieces, tiny: Tiny, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--check-icons` on renders of fixture programs, without the cached corpus: both
    decoders run, the manifest records them, and one differing token makes it exit 1."""

    layout, programs = pieces
    model, _ = tiny
    checkpoint = tmp_path / "best.pt"
    torch.save({"model": model.state_dict(), "config": asdict(_TINY_PATH), "step": 7}, checkpoint)
    images = torch.from_numpy(np.stack([_render(program, layout) for program in programs[:2]]))
    split = SplitData(
        hexcodes=["fixture-0", "fixture-1"],
        tokens=torch.stack([flatten_program(program, layout) for program in programs[:2]]),
        masks=np.zeros((2, 1), dtype=np.uint8),
        images=images,
        targets=images,
        shape=(layout.length, layout.vocabulary),
    )

    def fake_corpus(
        pilot_config: Path, image_size: int, **_: object
    ) -> tuple[dict[str, SplitData], dict[str, SplitData], SequenceLayout, None, str]:
        assert image_size == _TINY_PATH.image_size
        return {"primary/validation": split}, {}, layout, None, "fixture-corpus"

    monkeypatch.setattr(render2svg, "load_corpus", fake_corpus)
    common = ["--run", "tiny-run", "--checkpoint", str(checkpoint)]
    common += ["--pilot-config", str(DEFAULT_PILOT_CONFIG)]
    assert main([*common, "--check-icons", "5", "--out", str(tmp_path / "good")]) == 0
    greedy = json.loads((tmp_path / "good" / MANIFEST_FILE).read_text())["checks"]["greedy"]
    assert (greedy["icons"], greedy["identical"]) == (2, 2)
    assert greedy["dataset_sha256"] == "fixture-corpus"
    assert [row["hexcode"] for row in greedy["rows"]] == split.hexcodes
    assert all(row["model_calls"] == row["torch_model_calls"] > 0 for row in greedy["rows"])

    decode = OnnxDecoder.decode

    def off_by_one_token(
        self: OnnxDecoder, image: np.ndarray, stats: DecodeStats | None = None
    ) -> torch.Tensor:
        tokens = decode(self, image, stats)
        tokens[0] += 1
        return tokens

    monkeypatch.setattr(OnnxDecoder, "decode", off_by_one_token)
    assert main([*common, "--check-icons", "1", "--out", str(tmp_path / "bad")]) == 1
    greedy = json.loads((tmp_path / "bad" / MANIFEST_FILE).read_text())["checks"]["greedy"]
    assert (greedy["icons"], greedy["identical"]) == (1, 0)


def _final_model_ready() -> bool:
    checkpoint = CACHE_ROOT / "runs" / _FINAL_RUN / "best.pt"
    corpus = CACHE_ROOT / f"render2svg-corpus-{_cache_key(DEFAULT_PILOT_CONFIG, 144, True)}.npz"
    return checkpoint.exists() and corpus.exists()


@pytest.mark.skipif(not _final_model_ready(), reason="final checkpoint or its corpus missing")
def test_final_model_export_decodes_validation_icons_exactly(tmp_path: Path) -> None:
    """The 9.0M final model: ONNX Runtime greedy equals torch greedy on 4 validation icons."""

    manifest = export_run(_FINAL_RUN, tmp_path, check_icons=4)
    greedy = manifest["checks"]["greedy"]
    assert greedy["icons"] == greedy["identical"] == 4
    assert all(row["model_calls"] == row["torch_model_calls"] for row in greedy["rows"])
    assert greedy["dataset_sha256"].startswith("47646604")
    parity = manifest["checks"]["step_parity"]
    assert max(parity["logits"], parity["keys"], parity["values"]) < 1e-4
    assert parity["logits_vs_model_decode"] < 1e-3
    assert manifest["parameters"] == 9_025_826
    assert manifest["checkpoint"]["step"] == 64_000
