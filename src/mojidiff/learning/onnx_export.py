"""Export a render-to-SVG model for the browser: encoder.onnx, decoder_step.onnx, manifest.json.

Milestone 0 (findings 2026-10-04, `reports/browser/m0/README.md`) exported several decoder
steps, ran each in Chrome through ONNX Runtime Web, and kept variant B. This module is
variant B and the encoder moved into the package, for any `RenderToProgram` with metric
coordinates and path order (v9 and its descendants):

    python -m mojidiff.learning.onnx_export --run <run_id> [--out DIR] [--check-icons N]

writes `encoder.onnx`, `decoder_step.onnx` and `manifest.json` to DIR (default
`CACHE_ROOT/onnx/<run_id>/`). Run it from the repository root: the pilot configuration's
input paths are relative to it.

Encoder (once per icon): pixels (1, S, S, 3) float32 RGB in [0, 255] on white, NHWC, and
rows (R,) int64, used only for its length -> memory_keys, memory_values
(layers, R, heads, cells, head_dim), the cross-attention keys and values of every decoder
layer, expanded to R decoder rows.

Decoder step (once per free position): L new sequence steps on top of P cached ones.

* ids (6, B, L) float32: per step the token it reads (the previous step's token, 0 at step
  0), its sequence step (position embedding and cache slot), the field type and owning
  path of the position it predicts, and the axis and role of the position its input token
  came from. Every value is a small integer, exact in float32.
* target (B,) float32: the metric target of the last step's position,
  (role - 1) * 2 + axis for an endpoint or control coordinate, otherwise -1.
* past_keys, past_values (layers, B, heads, P, head_dim); memory_keys, memory_values as the
  encoder returns them.
* -> logits (B, vocabulary) of the last step only; present_keys, present_values
  (layers, B, heads, P + L, head_dim), the cache to pass back next time.

The layout keeps every integer computation on the device, because milestone 0 found that
ONNX Runtime Web's WebGPU backend silently corrupted small host-to-device copies inside
a run: with int64 inputs only 75.9% of calls matched torch, with this layout all 11,432.
So all decoder inputs are float32, ids are cast to int32 just before the embedding
lookups, key positions for the causal mask come from a CumSum (no Range), and a product
of floats replaces a boolean And. The head and the metric head are folded into one
constant table W[5, vocabulary, d] and b[5, vocabulary], indexed by target + 1, and the
Fourier features use an exact range reduction (coordinates are quarter units, so
2^k pi v / 72 = 2 pi a / 576 for an integer a, reduced modulo 576 in float32): the
model's own arguments reach about 1,800 rad, where WGSL leaves sin and cos accuracy to
the implementation. Against the training graph (`RenderToProgram.decode`) that moves
logits by at most a few 1e-4 on random inputs (v9 4.1e-4 in milestone 0; the final
model 2.4e-4), and greedy programs stay identical (`--check-icons`).
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn.functional as F
import yaml
from torch import Tensor, nn

from mojidiff.learning.autoregressive import (
    NONE,
    PAD,
    PATH_FIELDS,
    PATH_STRIDE,
    SEGMENT_STRIDE,
    SequenceLayout,
)
from mojidiff.learning.render2svg import (
    _CONTROL_SLOTS,
    CACHE_ROOT,
    FIELD_TYPES,
    REPO_ROOT,
    ROLE_CONTROL,
    ROLE_ENDPOINT,
    ROLE_NONE,
    ModelConfig,
    RenderToProgram,
    _DecoderBlock,
)

if TYPE_CHECKING:
    from mojidiff.learning.onnx_decode import OnnxDecoder

FORMAT = "mojidiff-render2svg-onnx"
FORMAT_VERSION = 1
OPSET = 18
"""As milestone 0: the dynamo exporter at opset 18."""

ENCODER_FILE = "encoder.onnx"
DECODER_FILE = "decoder_step.onnx"
MANIFEST_FILE = "manifest.json"
ID_ROWS = ("tokens", "steps", "field", "path", "axis", "role")
ENCODER_INPUTS = ("pixels", "rows")
ENCODER_OUTPUTS = ("memory_keys", "memory_values")
DECODER_INPUTS = ("ids", "target", "past_keys", "past_values", "memory_keys", "memory_values")
DECODER_OUTPUTS = ("logits", "present_keys", "present_values")
MAX_ROWS = 64
"""Largest decoder batch (and encoder rows) the exported graphs accept."""
DEFAULT_PILOT_CONFIG = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")


# --------------------------------------------------------------------------- modules


def check_exportable(model: RenderToProgram) -> None:
    """Refuse a model the browser graphs would export wrongly without an error."""

    if type(model) is not RenderToProgram:
        raise ValueError(f"only RenderToProgram exports, not {type(model).__name__}")
    config = model.config
    if not config.metric or config.order != "path":
        raise ValueError("the browser export needs metric coordinates and path order")


def folded_head(model: RenderToProgram) -> tuple[Tensor, Tensor]:
    """W (5, V, d), b (5, V): row 0 the plain head, row r + 1 the head plus the metric
    head for target r, so logits = W[target + 1] @ norm(h) + b[target + 1]."""

    with torch.no_grad():
        width = model.config.d_model
        candidates = model.metric_value(cast(Tensor, model._candidate_features))  # (4, V, d)
        query_w, query_b = model.metric_query.weight, model.metric_query.bias
        metric_w = torch.einsum("rvd,de->rve", candidates, query_w) / math.sqrt(width)
        metric_b = torch.einsum("rvd,d->rv", candidates, query_b) / math.sqrt(width)
        head_w, head_b = model.head.weight, model.head.bias
        weight = torch.cat((head_w[None], head_w[None] + metric_w)).contiguous()
        bias = torch.cat((head_b[None], head_b[None] + metric_b)).contiguous()
    return weight.float(), bias.float()


def _two_hot(value: Tensor) -> Tensor:
    """one_hot(value.clamp(0, 1), 2) of a float tensor, by comparisons (no int64)."""

    value = value.clamp(0.0, 1.0)
    return torch.stack(((value == 0.0).to(torch.float32), (value == 1.0).to(torch.float32)), dim=-1)


def _attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None) -> Tensor:
    """Plain matmul-softmax attention (no SDPA), so the exporter emits basic ops."""

    scores = torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(query.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), value)


class EncoderRows(nn.Module):
    """float32 NHWC pixels (1, S, S, 3) in [0, 255] and rows (R,) -> memory K, V
    (layers, R, heads, cells, head_dim). A model with emoji-set sources gets OpenMoji's."""

    def __init__(self, model: RenderToProgram) -> None:
        super().__init__()
        check_exportable(model)
        self.model = model

    def forward(self, pixels: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        model = self.model
        features = model.stem(pixels.permute(0, 3, 1, 2) / 127.5 - 1.0)
        features = features.flatten(2).transpose(1, 2) + model.grid_embedding
        features = features + model.grid_projection(cast(Tensor, model._cell_features))[None]
        for block in model.encoder:
            features = block(features)
        memory = model.encoder_norm(features)
        if model.config.sources:
            memory = memory + model.source_embedding.weight[:1][None]
        keys, values = [], []
        for block in model.decoder:
            key, value = cast(_DecoderBlock, block).cross_attention.keys_values(memory)
            keys.append(key)
            values.append(value)
        stacked_keys, stacked_values = torch.stack(keys), torch.stack(values)
        shape = (
            stacked_keys.shape[0],
            rows.shape[0],
            stacked_keys.shape[2],
            stacked_keys.shape[3],
            stacked_keys.shape[4],
        )
        return stacked_keys.expand(shape).contiguous(), stacked_values.expand(shape).contiguous()


class BrowserDecoderStep(nn.Module):
    """Milestone 0's variant B: the decoder step on float ids with a growing cache.

    Same computation as `RenderToProgram.decode` for the last of L steps (see the module
    docstring for the inputs), up to the exact Fourier range reduction.
    """

    def __init__(self, model: RenderToProgram) -> None:
        super().__init__()
        check_exportable(model)
        self.model = model
        powers = torch.pow(2.0, torch.arange(model.config.fourier)).to(torch.float32)
        self.register_buffer("pow2", powers, persistent=False)
        weight, bias = folded_head(model)
        self.register_buffer("table_w", weight, persistent=False)
        self.register_buffer("table_b", bias, persistent=False)

    def describe(self, value: Tensor, axis: Tensor, role: Tensor) -> Tensor:
        """`RenderToProgram._describe` with exact range reduction before sin and cos."""

        scaled = (value * 4.0)[..., None] * cast(Tensor, self.pow2)
        reduced = scaled - torch.floor(scaled / 576.0 + 0.5) * 576.0
        angle = reduced * (2.0 * math.pi / 576.0)
        return torch.cat(
            (
                (value / 72.0)[..., None],
                torch.sin(angle),
                torch.cos(angle),
                _two_hot(axis),
                _two_hot(role - 1.0),
            ),
            dim=-1,
        )

    def embed(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        model = self.model
        tokens, steps, field, path, axis, role = (row[0] for row in torch.split(ids, 1, dim=0))
        hidden = model.token_embedding(tokens.to(torch.int32))
        hidden = hidden + model.position_embedding(steps.to(torch.int32))
        hidden = hidden + model.field_embedding(field.to(torch.int32))
        hidden = hidden + model.path_embedding(path.to(torch.int32))
        index = (tokens - 1.0).clamp_min(0.0) * 0.25
        value = torch.where(role == float(ROLE_CONTROL), index - 8.0, index)
        described = self.describe(value, axis, role)
        # A product of floats, not a boolean And: WebGPU has no And kernel, so And ran on
        # the CPU with two device copies per call (milestone 0, about 4x slower calls).
        keep = (role > 0.0).to(torch.float32) * (tokens > 0.0).to(torch.float32)
        return hidden + model.coordinate_projection(described * keep[..., None]), steps

    def head(self, normed: Tensor, target: Tensor) -> Tensor:
        index = (target.clamp_min(-1.0) + 1.0).to(torch.int32)
        table_w, table_b = cast(Tensor, self.table_w), cast(Tensor, self.table_b)
        rows, vocabulary, width = table_w.shape
        # F.embedding keeps the int32 index (index_select exported a CPU cast to int64).
        weight = F.embedding(index, table_w.reshape(rows, vocabulary * width))
        bias = F.embedding(index, table_b)
        return (
            torch.matmul(weight.reshape(-1, vocabulary, width), normed[:, :, None])[:, :, 0] + bias
        )

    def forward(
        self,
        ids: Tensor,
        target: Tensor,
        past_keys: Tensor,
        past_values: Tensor,
        memory_keys: Tensor,
        memory_values: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        model = self.model
        hidden, steps = self.embed(ids)
        batch, length = hidden.shape[0], hidden.shape[1]
        new_keys, new_values = [], []
        mask: Tensor | None = None
        for index, block in enumerate(model.decoder):
            block = cast(_DecoderBlock, block)
            normed = block.norm_self(hidden)
            key, value = block.self_attention.keys_values(normed)
            key = torch.cat((past_keys[index], key), dim=2)
            value = torch.cat((past_values[index], value), dim=2)
            new_keys.append(key)
            new_values.append(value)
            if mask is None:
                # Key positions 0..T-1 as (1, 1, 1, T) by a CumSum over ones made from the
                # keys: no Range and no int64. A step attends every slot up to its own.
                ones = key[:1, :1, :, :1] * 0.0 + 1.0
                positions = (torch.cumsum(ones, dim=2) - 1.0).transpose(2, 3)
                mask = positions <= steps[:, None, :, None]  # (B, 1, L, T)
            query = block.self_attention.split(block.self_attention.query(normed))
            attended = _attention(query, key, value, mask)
            hidden = hidden + block.self_attention.project(
                attended.transpose(1, 2).reshape(batch, length, -1)
            )
            crossed = block.cross_attention.split(
                block.cross_attention.query(block.norm_cross(hidden))
            )
            attended = _attention(crossed, memory_keys[index], memory_values[index], None)
            hidden = hidden + block.cross_attention.project(
                attended.transpose(1, 2).reshape(batch, length, -1)
            )
            hidden = hidden + block.feedforward(block.norm_feedforward(hidden))
        normed = model.norm(hidden[:, -1])
        return self.head(normed, target), torch.stack(new_keys), torch.stack(new_values)


def cache_shape(model: RenderToProgram, batch: int, slots: int) -> tuple[int, int, int, int, int]:
    """(layers, batch, heads, slots, head_dim) of a decoder key or value cache."""

    config = model.config
    return (config.decoder_layers, batch, config.heads, slots, config.d_model // config.heads)


def example_step_inputs(
    model: RenderToProgram, batch: int, length: int, past: int, seed: int = 0
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Random decoder-step inputs in the exported layout, every id in its legal range."""

    layout = model.layout
    generator = torch.Generator().manual_seed(seed)

    def ints(high: int, low: int = 0) -> Tensor:
        return torch.randint(low, high, (batch, length), generator=generator)

    steps = (torch.arange(length) + past)[None].expand(batch, -1)
    ids = torch.stack(
        (
            ints(layout.vocabulary),
            steps,
            ints(FIELD_TYPES),
            ints(layout.codec.max_paths + 1),
            ints(2),
            ints(3),
        )
    ).to(torch.float32)
    target = torch.randint(-1, 4, (batch,), generator=generator).to(torch.float32)
    past_keys = torch.randn(cache_shape(model, batch, past), generator=generator)
    past_values = torch.randn(cache_shape(model, batch, past), generator=generator)
    memory = cache_shape(model, batch, model.memory_length)
    memory_keys = torch.randn(memory, generator=generator)
    memory_values = torch.randn(memory, generator=generator)
    return ids, target, past_keys, past_values, memory_keys, memory_values


def torch_step_reference(
    model: RenderToProgram,
    ids: Tensor,
    target: Tensor,
    past_keys: Tensor,
    past_values: Tensor,
    memory_keys: Tensor,
    memory_values: Tensor,
) -> Tensor:
    """Last-step logits (B, V) of `RenderToProgram.decode` itself - the graph training and
    `greedy_decode` run - for decoder-step inputs in the exported layout."""

    from mojidiff.learning.render2svg import coordinate_value

    tokens, steps, field, path, axis, role = (row.to(torch.long) for row in ids)
    offset = int(steps[0, 0])
    expected = torch.arange(offset, offset + steps.shape[1])[None].expand_as(steps)
    if past_keys.shape[3] != offset or not torch.equal(steps, expected):
        raise ValueError("model.decode needs steps past, past + 1, ... in every row")
    keep = ((role > 0) & (tokens > 0))[..., None]
    described = model._describe(coordinate_value(tokens, role), axis, role) * keep
    extras = {
        "field": field,
        "path": path,
        "metric_input": described,
        "metric_target": target.to(torch.long)[:, None].expand_as(tokens),
    }
    layers = range(model.config.decoder_layers)
    cache = [(past_keys[i], past_values[i]) for i in layers] if offset else None
    memory = [(memory_keys[i], memory_values[i]) for i in layers]
    with torch.no_grad():
        logits, _ = model.decode(tokens, memory, cache, offset=offset, extras=extras)
    return logits[:, -1]


PARITY_SHAPES = ((1, 1, 0), (1, 1, 300), (1, 7, 40), (8, 1, 900), (3, 32, 1000), (2, 1, 1375))
"""(batch, length, past) of milestone 0's parity check."""


def step_parity(model: RenderToProgram, decoder: OnnxDecoder) -> dict[str, float]:
    """Largest absolute differences of the ONNX Runtime step on random inputs against the
    same module in torch (logits, keys, values) and against `RenderToProgram.decode`."""

    browser = BrowserDecoderStep(model).eval()
    worst = {"logits": 0.0, "keys": 0.0, "values": 0.0, "logits_vs_model_decode": 0.0}
    for seed, (batch, length, past) in enumerate(PARITY_SHAPES, start=1):
        past = min(past, model.layout.length - length)
        inputs = example_step_inputs(model, batch, length, past, seed=seed)
        with torch.no_grad():
            logits, keys, values = browser(*inputs)
        reference = torch_step_reference(model, *inputs)
        got = decoder.step(*(tensor.numpy() for tensor in inputs))
        names = ("logits", "keys", "values")
        for name, expected, actual in zip(names, (logits, keys, values), got, strict=True):
            worst[name] = max(worst[name], float((torch.from_numpy(actual) - expected).abs().max()))
        difference = float((torch.from_numpy(got[0]) - reference).abs().max())
        worst["logits_vs_model_decode"] = max(worst["logits_vs_model_decode"], difference)
    return worst


# --------------------------------------------------------------------------- export


def _export(
    module: nn.Module,
    inputs: tuple[Tensor, ...],
    path: Path,
    input_names: tuple[str, ...],
    output_names: tuple[str, ...],
    dynamic_shapes: dict[str, Any],
) -> None:
    """Dynamo export at `OPSET` to one self-contained file, without the exporter's
    per-node debug metadata (stack traces with this checkout's absolute paths, about
    250 KB), so the bytes depend on the model and the code, not on where it ran."""

    import onnx

    with torch.no_grad():
        program = torch.onnx.export(
            module,
            inputs,
            dynamo=True,
            verbose=False,
            input_names=list(input_names),
            output_names=list(output_names),
            dynamic_shapes=dynamic_shapes,
            opset_version=OPSET,
        )
    if program is None:
        raise RuntimeError("the dynamo exporter returned no program")
    program.save(str(path), external_data=False)
    graph = onnx.load(str(path))
    for node in graph.graph.node:
        del node.metadata_props[:]
    onnx.save(graph, str(path))


def export_encoder(model: RenderToProgram, path: Path) -> None:
    from torch.export import Dim

    size = model.config.image_size
    generator = torch.Generator().manual_seed(0)
    pixels = torch.randint(0, 256, (1, size, size, 3), generator=generator).to(torch.float32)
    rows = torch.zeros(2, dtype=torch.int64)
    _export(
        EncoderRows(model).eval(),
        (pixels, rows),
        path,
        ENCODER_INPUTS,
        ENCODER_OUTPUTS,
        {"pixels": None, "rows": {0: Dim("rows", min=1, max=MAX_ROWS)}},
    )


def export_decoder_step(model: RenderToProgram, path: Path) -> None:
    from torch.export import Dim

    length = model.layout.length
    batch = Dim("batch", min=1, max=MAX_ROWS)
    steps = Dim("length", min=1, max=length)
    past = Dim("past", min=0, max=length)
    shapes = {
        "ids": {1: batch, 2: steps},
        "target": {0: batch},
        "past_keys": {1: batch, 3: past},
        "past_values": {1: batch, 3: past},
        "memory_keys": {1: batch},
        "memory_values": {1: batch},
    }
    _export(
        BrowserDecoderStep(model).eval(),
        example_step_inputs(model, 2, 3, 5),
        path,
        DECODER_INPUTS,
        DECODER_OUTPUTS,
        shapes,
    )


def graph_signature(path: Path) -> dict[str, Any]:
    """Input and output names, element types and shapes of a saved graph, plus its ops."""

    import onnx
    from onnx import helper

    graph = onnx.load(str(path), load_external_data=False).graph

    def describe(value: Any) -> dict[str, Any]:
        tensor = value.type.tensor_type
        shape: list[int | str] = [
            dim.dim_value if dim.HasField("dim_value") else dim.dim_param
            for dim in tensor.shape.dim
        ]
        dtype = helper.tensor_dtype_to_np_dtype(tensor.elem_type).name
        return {"name": value.name, "dtype": dtype, "shape": shape}

    ops = Counter(node.op_type for node in graph.node)
    return {
        "inputs": [describe(value) for value in graph.input],
        "outputs": [describe(value) for value in graph.output],
        "nodes": len(graph.node),
        "ops": dict(sorted(ops.items())),
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def browser_constants(model: RenderToProgram) -> dict[str, Any]:
    """Everything the JavaScript decoder needs besides the graphs: sequence layout, step
    rules and the grammar's vocabulary sizes (see `autoregressive.legal_mask`)."""

    layout = model.layout
    codec = layout.codec
    config = model.config
    return {
        "vocabulary": layout.vocabulary,
        "length": layout.length,
        "path_positions": layout.path_positions,
        "max_paths": codec.max_paths,
        "max_segments": codec.max_segments,
        "total_segment_slots": layout.total_segment_slots,
        "path_stride": PATH_STRIDE,
        "segment_stride": SEGMENT_STRIDE,
        "path_fields": list(PATH_FIELDS),
        "field_types": FIELD_TYPES,
        "image_size": config.image_size,
        "memory_length": model.memory_length,
        "d_model": config.d_model,
        "decoder_layers": config.decoder_layers,
        "heads": config.heads,
        "head_dim": config.d_model // config.heads,
        "max_rows": MAX_ROWS,
        "ids_rows": list(ID_ROWS),
        "roles": {"none": ROLE_NONE, "endpoint": ROLE_ENDPOINT, "control": ROLE_CONTROL},
        "control_slots": list(_CONTROL_SLOTS),
        "target_rule": "(role - 1) * 2 + axis for an endpoint or control position, else -1",
        "pixels": "NHWC float32 RGB in [0, 255], composited on white",
        "grammar": {
            "pad": PAD,
            "none": NONE,
            "segment_kinds": {"pad": 0, "line": 1, "quad": 2, "cubic": 3, "close": 4},
            "coordinate_bins": codec.coordinate_bins,
            "control_coordinate_bins": codec.effective_control_coordinate_bins,
            "control_coordinate_min": codec.control_coordinate_min,
            "palette": len(codec.palette),
            "opacities": len(codec.opacities),
            "stroke_widths": len(codec.stroke_widths),
            "miter_limits": len(codec.miter_limits),
            "dash_patterns": len(codec.dash_patterns),
            "fill_rules": 2,
            "linecaps": 3,
            "linejoins": 3,
        },
    }


def export_model(
    model: RenderToProgram, out: Path, *, provenance: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Write encoder.onnx, decoder_step.onnx and manifest.json to `out`; the manifest."""

    check_exportable(model)
    model = model.cpu().float().eval()
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, Any] = {}
    for role, name, export in (
        ("encoder", ENCODER_FILE, export_encoder),
        ("decoder_step", DECODER_FILE, export_decoder_step),
    ):
        path = out / name
        started = time.perf_counter()
        export(model, path)
        files[role] = {
            "file": name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "export_seconds": round(time.perf_counter() - started, 2),
            **graph_signature(path),
        }
    manifest: dict[str, Any] = {
        "format": FORMAT,
        "format_version": FORMAT_VERSION,
        **(provenance or {}),
        "model_config": asdict(model.config),
        "export": {
            "variant": "milestone 0 variant B: float32 ids, growing float32 KV cache, "
            "folded head, exact Fourier range reduction",
            "opset": OPSET,
            "exporter": "torch.onnx.export(dynamo=True)",
            "torch": torch.__version__,
            "onnx": importlib.metadata.version("onnx"),
            "onnxscript": importlib.metadata.version("onnxscript"),
        },
        "files": files,
        "constants": browser_constants(model),
    }
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


# --------------------------------------------------------------------------- runs


def load_layout(pilot_config: Path) -> SequenceLayout:
    from mojidiff.learning.openmoji_pilot import _selected_codec, load_openmoji_pilot_config

    pilot = load_openmoji_pilot_config(pilot_config)
    return SequenceLayout(
        codec=_selected_codec(pilot), total_segment_slots=pilot.total_segment_slots
    )


def run_pilot_config(run_id: str) -> Path:
    """The pilot configuration a registered run trained on, from `runs/<run_id>/run.yaml`."""

    record_path = REPO_ROOT / "runs" / run_id / "run.yaml"
    if not record_path.exists():
        raise FileNotFoundError(f"{record_path} not found; pass --pilot-config")
    record = yaml.safe_load(record_path.read_text())
    return Path(str(record["dataset"]["pilot_config"]))


def load_checkpoint_model(
    checkpoint: Path, layout: SequenceLayout
) -> tuple[RenderToProgram, int | None]:
    """A float32 CPU model from a training checkpoint, and the step it was saved at."""

    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = RenderToProgram(layout, ModelConfig(**state["config"]))
    model.load_state_dict(state["model"])
    step = state.get("step")
    return model.eval(), int(step) if step is not None else None


def git_commit() -> str | None:
    from mojidiff.learning.render2svg import _git

    try:
        return _git("rev-parse", "HEAD").strip()
    except Exception:  # noqa: BLE001 - outside a checkout the manifest says so
        return None


def export_run(
    run_id: str,
    out: Path | None = None,
    *,
    checkpoint: Path | None = None,
    pilot_config: Path | None = None,
    check_icons: int = 0,
) -> dict[str, Any]:
    """Export one run's checkpoint; with `check_icons`, also decode that many validation
    icons with ONNX Runtime and torch and record whether the programs agree."""

    checkpoint = checkpoint or CACHE_ROOT / "runs" / run_id / "best.pt"
    out = out or CACHE_ROOT / "onnx" / run_id
    pilot_config = pilot_config or run_pilot_config(run_id)
    layout = load_layout(pilot_config)
    model, step = load_checkpoint_model(checkpoint, layout)
    provenance = {
        "run_id": run_id,
        "checkpoint": {
            "path": str(checkpoint),
            "file": checkpoint.name,
            "sha256": sha256_file(checkpoint),
            "bytes": checkpoint.stat().st_size,
            "step": step,
        },
        "pilot_config": str(pilot_config),
        "parameters": sum(p.numel() for p in model.parameters()),
        "git_commit": git_commit(),
        "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest = export_model(model, out, provenance=provenance)
    from mojidiff.learning.onnx_decode import OnnxDecoder

    decoder = OnnxDecoder(out, layout)
    manifest["checks"] = {"step_parity": step_parity(model, decoder)}
    if check_icons:
        manifest["checks"]["greedy"] = _check_greedy(model, decoder, pilot_config, check_icons)
    (out / MANIFEST_FILE).write_text(json.dumps(manifest, indent=1) + "\n")
    return manifest


def _check_greedy(
    model: RenderToProgram, decoder: OnnxDecoder, pilot_config: Path, icons: int
) -> dict[str, Any]:
    """ONNX Runtime greedy tokens against `render2svg.greedy_decode`, both on the CPU in
    float32, on the first `icons` validation renders of the cached corpus."""

    from mojidiff.learning.render2svg import DecodeStats, greedy_decode, load_corpus

    plain, _, _, _, dataset_hash = load_corpus(pilot_config, model.config.image_size)
    data = plain["primary/validation"]
    count = min(icons, len(data.tokens))
    rows = []
    for index in range(count):
        image = data.images[index]
        torch_stats, onnx_stats = DecodeStats(), DecodeStats()
        started = time.perf_counter()
        reference = greedy_decode(model, image[None], stats=torch_stats)[0]
        torch_seconds = time.perf_counter() - started
        started = time.perf_counter()
        tokens = decoder.decode(image.numpy(), stats=onnx_stats)
        onnx_seconds = time.perf_counter() - started
        rows.append(
            {
                "hexcode": data.hexcodes[index],
                "equal": bool(torch.equal(tokens, reference)),
                "model_calls": onnx_stats.model_calls,
                "torch_model_calls": torch_stats.model_calls,
                "torch_cpu_seconds": round(torch_seconds, 3),
                "onnx_cpu_seconds": round(onnx_seconds, 3),
            }
        )
        print(json.dumps(rows[-1]), flush=True)
    return {
        "split": "primary/validation",
        "dataset_sha256": dataset_hash,
        "icons": count,
        "identical": sum(row["equal"] for row in rows),
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--run", required=True, help="run id; its checkpoint and pilot config")
    parser.add_argument("--out", type=Path, help="output directory (CACHE_ROOT/onnx/<run>)")
    parser.add_argument(
        "--checkpoint", type=Path, help="checkpoint (CACHE_ROOT/runs/<run>/best.pt)"
    )
    parser.add_argument("--pilot-config", type=Path, help="pilot config (from runs/<run>/run.yaml)")
    parser.add_argument(
        "--check-icons", type=int, default=0, help="validation icons to decode with both"
    )
    args = parser.parse_args(argv)
    manifest = export_run(
        args.run,
        args.out,
        checkpoint=args.checkpoint,
        pilot_config=args.pilot_config,
        check_icons=args.check_icons,
    )
    summary = {
        "run_id": manifest["run_id"],
        "files": {role: entry["bytes"] for role, entry in manifest["files"].items()},
        "checks": {
            key: value if key != "greedy" else {k: v for k, v in value.items() if k != "rows"}
            for key, value in manifest["checks"].items()
        },
    }
    print(json.dumps(summary, indent=1))
    greedy = manifest["checks"].get("greedy")
    return 1 if greedy is not None and greedy["identical"] < greedy["icons"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
