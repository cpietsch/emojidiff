"""Milestone 0: export the v9 decoder-step variants and the encoder for the browser replay.

  A   decoder_step_A.onnx   the current export: growing KV cache, plain MatMul/Softmax attention
                            (r2s_export.DecoderStep, use_sdpa=False), six int64 (B, L) inputs + int64
                            target, metric head computed every step (Einsum over 4 x 418 x 256).
  B0  decoder_step_B0.onnx  "browser-safe" growing-cache step, metric head NOT folded:
                            * ids (6, B, L) float32 = tokens, steps, field, path, axis, role and
                              target (B,) float32. All integer work stays on the GPU (Cast float ->
                              int32 for the embedding Gathers; comparisons in float). No int64 tensor
                              and no CPU-placed int op feeds the GPU. ORT Web 1.30 JSEP corrupts small
                              host->device copies inside a run (measured: with A's int64 inputs, or
                              int32 ids split on the CPU, every embedding Gather read the tokens row).
                            * key positions for the causal mask from CumSum over a key slice (no Range).
                            * exact range reduction before sin/cos.
  B   decoder_step_B.onnx   B0 with the head and the metric head folded into one constant table
                            W[5, 418, 256], bias[5, 418]: logits = W[target + 1] @ norm(h) + bias[target + 1]
                            (entry 0 = plain head; entry r + 1 = head + metric_value(candidates_r) @ metric_query
                            / sqrt(d)). Removes the Einsum and the metric_query matmul.
  Bs  decoder_step_Bs.onnx  diagnostic: B with the model's own sin/cos arguments (no range reduction).
  C   decoder_step_C.onnx   B with a static 1,377-slot cache (layers, B, H, 1377, hd); new K/V are written
                            by a one-hot blend (cache * (1 - sum onehot) + onehot^T @ new) instead of
                            ScatterElements, so no int64 indices; constant key positions.
  Cg1, Cg8                  C exported with fixed shapes (batch 1 or 8, L = 1) for ORT Web graph capture,
                            with encoder_g1/encoder_g8.onnx (fixed rows); chunks are fed one step at a time.
  encoder.onnx              float32 NHWC pixels (1, 144, 144, 3) and rows (R,) -> memory K, V (6, R, 8, 324, 32):
                            the encoder runs once at batch 1 and its output is expanded to R decoder rows.

Parity: each variant against torch float32 on CPU on random inputs (A vs r2s_export.DecoderStep;
B0/B/C vs torch FloatStep(fold=False), i.e. the same exact Fourier, so the fold and the static
cache are tested on their own) plus the max difference to the original torch module.

Usage (repo root as cwd): python export_variants.py [enc A B0 B Bs C Cg1 Cg8 parity]
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import cast

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.append(str(HERE.parent / "ortlib"))

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mojidiff.learning.render2svg import ROLE_CONTROL, _DecoderBlock
from r2s_export import STEP_NAMES, DecoderStep, Encoder, attention, example_inputs, load

OUT = HERE / "onnx"
OUT.mkdir(exist_ok=True)


def folded_table(model) -> tuple[Tensor, Tensor]:
    """W (5, V, d), bias (5, V): row 0 the plain head, row r+1 head + metric for target r."""
    with torch.no_grad():
        d = model.config.d_model
        cand = model.metric_value(cast(Tensor, model._candidate_features))  # (4, V, d)
        wq, bq = model.metric_query.weight, model.metric_query.bias  # q = wq n + bq
        # score_r[v] = cand[r, v] . (wq n + bq) / sqrt(d)
        metric_w = torch.einsum("rvd,de->rve", cand, wq) / math.sqrt(d)  # (4, V, d)
        metric_b = torch.einsum("rvd,d->rv", cand, bq) / math.sqrt(d)  # (4, V)
        head_w, head_b = model.head.weight, model.head.bias
        w = torch.cat((head_w[None], head_w[None] + metric_w)).contiguous()
        b = torch.cat((head_b[None], head_b[None] + metric_b)).contiguous()
    return w.float(), b.float()


def two_hot(x: Tensor) -> Tensor:
    """one_hot(x.clamp(0, 1), 2) for a float tensor, by comparisons (no int64)."""
    x = x.clamp(0.0, 1.0)
    return torch.stack(((x == 0.0).to(torch.float32), (x == 1.0).to(torch.float32)), dim=-1)


class FloatStep(nn.Module):
    """Growing-cache decoder step on float ids (variants B0, B). See module docstring."""

    def __init__(self, model, fold: bool, exact: bool = True) -> None:
        super().__init__()
        self.model = model
        self.fold = fold
        self.exact = exact
        k = model.config.fourier
        self.register_buffer("pow2", torch.pow(2.0, torch.arange(k)).to(torch.float32), persistent=False)
        # exactly render2svg.fourier_features' float32 scales (for the diagnostic Bs)
        self.register_buffer("scales", (torch.pow(2.0, torch.arange(k)) * torch.pi / 72.0).to(torch.float32),
                             persistent=False)
        if fold:
            w, b = folded_table(model)
            self.register_buffer("table_w", w)
            self.register_buffer("table_b", b)
        else:
            with torch.no_grad():
                cand = model.metric_value(cast(Tensor, model._candidate_features))
            self.register_buffer("candidates", cand.float())

    def describe(self, value: Tensor, axis: Tensor, role: Tensor) -> Tensor:
        # Exact range reduction: value is a multiple of 1/4, so a = 4 v 2^k is an exact integer
        # and sin(2^k pi v / 72) = sin(2 pi a / 576); reduce a mod 576 exactly in float32.
        # v9 arguments reach ~1800 rad, where WGSL leaves sin/cos accuracy to the implementation.
        if self.exact:
            a = (value * 4.0)[..., None] * self.pow2
            reduced = a - torch.floor(a / 576.0 + 0.5) * 576.0
            angle = reduced * (2.0 * math.pi / 576.0)
        else:  # diagnostic Bs: render2svg.fourier_features as is (float scales buffer)
            angle = value[..., None] * self.scales
        return torch.cat(((value / 72.0)[..., None], torch.sin(angle), torch.cos(angle),
                          two_hot(axis), two_hot(role - 1.0)), dim=-1)

    def embed(self, ids: Tensor) -> tuple[Tensor, Tensor]:
        m = self.model
        tokens, steps, field, path, axis, role = (p[0] for p in torch.split(ids, 1, dim=0))
        i32 = lambda t: t.to(torch.int32)  # noqa: E731
        hidden = m.token_embedding(i32(tokens)) + m.position_embedding(i32(steps))
        hidden = hidden + m.field_embedding(i32(field)) + m.path_embedding(i32(path))
        index = (tokens - 1.0).clamp_min(0.0) * 0.25
        value = torch.where(role == float(ROLE_CONTROL), index - 8.0, index)
        described = self.describe(value, axis, role)
        # product of floats, not a bool And: JSEP has no And kernel, so And ran on the CPU
        # with two GPU->CPU copies per call (measured, ~4x slower calls).
        keep = ((role > 0.0).to(torch.float32) * (tokens > 0.0).to(torch.float32)).unsqueeze(-1)
        return hidden + m.coordinate_projection(described * keep), steps

    def head(self, normed: Tensor, target: Tensor) -> Tensor:
        m = self.model
        if self.fold:
            index = (target.clamp_min(-1.0) + 1.0).to(torch.int32)
            # F.embedding keeps the int32 index (index_select exported a CPU Cast to int64).
            rows, vocabulary, width = self.table_w.shape
            w = F.embedding(index, self.table_w.reshape(rows, vocabulary * width)).reshape(-1, vocabulary, width)
            b = F.embedding(index, self.table_b)  # (B, V)
            return torch.matmul(w, normed[:, :, None])[:, :, 0] + b
        logits = m.head(normed)
        query = m.metric_query(normed)
        scores = torch.einsum("bd,rvd->brv", query, self.candidates) / math.sqrt(query.shape[-1])
        pick = (torch.arange(4, dtype=torch.float32)[None, :] == target[:, None]).to(torch.float32)  # (B, 4)
        return logits + (scores * pick[:, :, None]).sum(1)

    def layers(self, hidden, steps, update, memory_keys, memory_values, positions_of):
        m = self.model
        batch, length = hidden.shape[0], hidden.shape[1]
        new_keys, new_values = [], []
        mask = None
        for index, block in enumerate(m.decoder):
            block = cast(_DecoderBlock, block)
            normed = block.norm_self(hidden)
            key, val = block.self_attention.keys_values(normed)
            key, val = update(index, key, val)
            new_keys.append(key)
            new_values.append(val)
            if mask is None:
                mask = positions_of(key) <= steps[:, None, :, None]  # (B, 1, L, T)
            query = block.self_attention.split(block.self_attention.query(normed))
            att = attention(query, key, val, mask)
            hidden = hidden + block.self_attention.project(att.transpose(1, 2).reshape(batch, length, -1))
            crossed = block.cross_attention.split(block.cross_attention.query(block.norm_cross(hidden)))
            att = attention(crossed, memory_keys[index], memory_values[index], None)
            hidden = hidden + block.cross_attention.project(att.transpose(1, 2).reshape(batch, length, -1))
            hidden = hidden + block.feedforward(block.norm_feedforward(hidden))
        return hidden, torch.stack(new_keys), torch.stack(new_values)

    def forward(self, ids, target, past_keys, past_values, memory_keys, memory_values):
        hidden, steps = self.embed(ids)

        def update(index, key, val):
            return torch.cat((past_keys[index], key), dim=2), torch.cat((past_values[index], val), dim=2)

        def positions_of(key):
            # (1, 1, 1, T) = 0..T-1 by CumSum over ones made from the keys (no Range, no int64)
            ones = key[:1, :1, :, :1] * 0.0 + 1.0  # (1, 1, T, 1)
            return (torch.cumsum(ones, dim=2) - 1.0).transpose(2, 3)

        hidden, keys, values = self.layers(hidden, steps, update, memory_keys, memory_values, positions_of)
        normed = self.model.norm(hidden[:, -1])
        return self.head(normed, target), keys, values


class FloatStatic(FloatStep):
    """Static cache (layers, B, H, 1377, hd); one-hot blend update; slots = steps of row 0."""

    def __init__(self, model) -> None:
        super().__init__(model, fold=True)
        total = model.layout.length + 1
        self.register_buffer("slots", torch.arange(total, dtype=torch.float32), persistent=False)

    def forward(self, ids, target, cache_keys, cache_values, memory_keys, memory_values):
        hidden, steps = self.embed(ids)
        onehot = (steps[0][:, None] == self.slots[None, :]).to(torch.float32)  # (L, T)
        keep = 1.0 - onehot.sum(0)[None, None, :, None]  # (1, 1, T, 1)
        spread = onehot.transpose(0, 1)  # (T, L)

        def update(index, key, val):
            return (cache_keys[index] * keep + torch.matmul(spread, key),
                    cache_values[index] * keep + torch.matmul(spread, val))

        def positions_of(key):
            return self.slots[None, None, None, :]

        hidden, keys, values = self.layers(hidden, steps, update, memory_keys, memory_values, positions_of)
        normed = self.model.norm(hidden[:, -1])
        return self.head(normed, target), keys, values


class EncoderRows(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.encoder = Encoder(model)

    def forward(self, pixels: Tensor, rows: Tensor) -> tuple[Tensor, Tensor]:
        k, v = self.encoder(pixels)  # (6, 1, H, 324, hd)
        shape = (k.shape[0], rows.shape[0], k.shape[2], k.shape[3], k.shape[4])
        return k.expand(shape).contiguous(), v.expand(shape).contiguous()


def export_growing_int64(module: nn.Module, path: Path, model) -> None:
    from torch.export import Dim

    b = Dim("batch", min=1, max=64)
    seq = Dim("length", min=1, max=1376)
    past = Dim("past", min=0, max=1376)
    shapes = {
        "tokens": {0: b, 1: seq}, "steps": {0: b, 1: seq}, "field": {0: b, 1: seq},
        "path": {0: b, 1: seq}, "axis": {0: b, 1: seq}, "role": {0: b, 1: seq}, "target": {0: b},
        "past_keys": {1: b, 3: past}, "past_values": {1: b, 3: past},
        "memory_keys": {1: b}, "memory_values": {1: b},
    }
    args = example_inputs(model, 2, 3, 5)
    with torch.no_grad():
        p = torch.onnx.export(module, args, dynamo=True, input_names=STEP_NAMES,
                              output_names=["logits", "present_keys", "present_values"],
                              dynamic_shapes=shapes, opset_version=18)
    p.save(str(path))


def float_args(args: tuple, static: bool, total: int) -> tuple:
    tokens, steps, field, pth, axis, role, target, pk, pv, mk, mv = args
    ids = torch.stack((tokens, steps, field, pth, axis, role)).to(torch.float32)
    if static:
        batch, past = pk.shape[1], pk.shape[3]
        ck = torch.zeros(6, batch, 8, total, 32)
        cv = torch.zeros(6, batch, 8, total, 32)
        ck[:, :, :, :past] = pk
        cv[:, :, :, :past] = pv
        pk, pv = ck, cv
    return ids, target.to(torch.float32), pk, pv, mk, mv


def export_float(module: nn.Module, path: Path, model, static: bool) -> None:
    from torch.export import Dim

    total = model.layout.length + 1
    b = Dim("batch", min=1, max=64)
    seq = Dim("length", min=1, max=1376)
    args = float_args(example_inputs(model, 2, 3, 5), static, total)
    if static:
        names = ["ids", "target", "cache_keys", "cache_values", "memory_keys", "memory_values"]
        outs = ["logits", "cache_keys_out", "cache_values_out"]
        shapes = {"ids": {1: b, 2: seq}, "target": {0: b}, "cache_keys": {1: b}, "cache_values": {1: b},
                  "memory_keys": {1: b}, "memory_values": {1: b}}
    else:
        past = Dim("past", min=0, max=1376)
        names = ["ids", "target", "past_keys", "past_values", "memory_keys", "memory_values"]
        outs = ["logits", "present_keys", "present_values"]
        shapes = {"ids": {1: b, 2: seq}, "target": {0: b}, "past_keys": {1: b, 3: past},
                  "past_values": {1: b, 3: past}, "memory_keys": {1: b}, "memory_values": {1: b}}
    with torch.no_grad():
        p = torch.onnx.export(module, args, dynamo=True, input_names=names, output_names=outs,
                              dynamic_shapes=shapes, opset_version=18)
    p.save(str(path))


def export_fixed(module: nn.Module, path: Path, model, batch: int) -> None:
    """Static cache, fixed shapes (batch, L = 1): no Shape/Concat nodes, so every node is on the
    GPU and ORT Web can capture the step as a graph (enableGraphCapture)."""
    total = model.layout.length + 1
    args = float_args(example_inputs(model, batch, 1, 5), True, total)
    names = ["ids", "target", "cache_keys", "cache_values", "memory_keys", "memory_values"]
    outs = ["logits", "cache_keys_out", "cache_values_out"]
    with torch.no_grad():
        p = torch.onnx.export(module, args, dynamo=True, input_names=names, output_names=outs, opset_version=18)
    p.save(str(path))


def export_encoder_fixed(module: nn.Module, path: Path, rows: int) -> None:
    pixels = torch.randint(0, 256, (1, 144, 144, 3)).float()
    with torch.no_grad():
        p = torch.onnx.export(module, (pixels, torch.zeros(rows, dtype=torch.int64)), dynamo=True,
                              input_names=["pixels", "rows"], output_names=["memory_keys", "memory_values"],
                              opset_version=18)
    p.save(str(path))


def export_encoder(module: nn.Module, path: Path) -> None:
    from torch.export import Dim

    r = Dim("rows", min=1, max=64)
    pixels = torch.randint(0, 256, (1, 144, 144, 3)).float()
    rows = torch.zeros(2, dtype=torch.int64)
    with torch.no_grad():
        p = torch.onnx.export(module, (pixels, rows), dynamo=True, input_names=["pixels", "rows"],
                              output_names=["memory_keys", "memory_values"],
                              dynamic_shapes={"pixels": None, "rows": {0: r}}, opset_version=18)
    p.save(str(path))


def op_histogram(path: Path) -> dict[str, int]:
    import collections

    import onnx

    return dict(collections.Counter(n.op_type for n in onnx.load(str(path)).graph.node).most_common())


def parity(model) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 8
    ref_a = DecoderStep(model, use_sdpa=False).eval()
    ref_b0 = FloatStep(model, fold=False).eval()
    total = model.layout.length + 1
    res: dict = {}
    g = torch.Generator().manual_seed(1)
    for name in ("A", "B0", "B", "Bs", "C"):
        path = OUT / f"decoder_step_{name}.onnx"
        if not path.exists():
            continue
        sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        worst = {"logits": 0.0, "keys": 0.0, "values": 0.0, "logits_vs_original_torch": 0.0}
        for batch, length, past in ((1, 1, 0), (1, 1, 300), (1, 7, 40), (8, 1, 900), (3, 32, 1000), (2, 1, 1375)):
            args = list(example_inputs(model, batch, length, past))
            args[0] = torch.randint(0, 418, (batch, length), generator=g)
            args[6] = torch.randint(-1, 4, (batch,), generator=g)
            for i in (7, 8, 9, 10):
                args[i] = torch.randn(args[i].shape, generator=g)
            with torch.no_grad():
                orig_logits, orig_k, orig_v = ref_a(*args)
                if name in ("A", "Bs"):
                    ref_logits, ref_k, ref_v = orig_logits, orig_k, orig_v
                else:
                    ref_logits, ref_k, ref_v = ref_b0(*float_args(tuple(args), False, total))
            if name == "A":
                lo, ko, vo = sess.run(None, {n: a.numpy() for n, a in zip(STEP_NAMES, args)})
            else:
                static = name == "C"
                sargs = float_args(tuple(args), static, total)
                keys = ["ids", "target", "cache_keys" if static else "past_keys",
                        "cache_values" if static else "past_values", "memory_keys", "memory_values"]
                lo, ko, vo = sess.run(None, dict(zip(keys, (a.numpy() for a in sargs))))
                ko, vo = ko[:, :, :, : past + length], vo[:, :, :, : past + length]
            worst["logits"] = max(worst["logits"], float(np.abs(lo - ref_logits.numpy()).max()))
            worst["keys"] = max(worst["keys"], float(np.abs(ko - ref_k.numpy()).max()))
            worst["values"] = max(worst["values"], float(np.abs(vo - ref_v.numpy()).max()))
            worst["logits_vs_original_torch"] = max(worst["logits_vs_original_torch"],
                                                    float(np.abs(lo - orig_logits.numpy()).max()))
        res[name] = worst
        print(name, worst, flush=True)
    enc_path = OUT / "encoder.onnx"
    if enc_path.exists():
        sess = ort.InferenceSession(str(enc_path), opts, providers=["CPUExecutionProvider"])
        pixels = torch.randint(0, 256, (1, 144, 144, 3), generator=g).float()
        with torch.no_grad():
            rk, rv = Encoder(model)(pixels)
        k, v = sess.run(None, {"pixels": pixels.numpy(), "rows": np.zeros(8, np.int64)})
        res["encoder"] = {"keys": float(np.abs(k - rk.numpy()).max()), "values": float(np.abs(v - rv.numpy()).max()),
                          "shape": list(k.shape)}
        print("encoder", res["encoder"], flush=True)
    return res


if __name__ == "__main__":
    import json

    which = sys.argv[1:] or ["enc", "A", "B0", "B", "Bs", "C", "parity"]
    model = load()
    info: dict = {}
    for name in which:
        t = time.perf_counter()
        if name == "enc":
            path = OUT / "encoder.onnx"
            export_encoder(EncoderRows(model).eval(), path)
        elif name == "A":
            path = OUT / "decoder_step_A.onnx"
            export_growing_int64(DecoderStep(model, use_sdpa=False).eval(), path, model)
        elif name == "B0":
            path = OUT / "decoder_step_B0.onnx"
            export_float(FloatStep(model, fold=False).eval(), path, model, static=False)
        elif name == "B":
            path = OUT / "decoder_step_B.onnx"
            export_float(FloatStep(model, fold=True).eval(), path, model, static=False)
        elif name == "Bs":
            path = OUT / "decoder_step_Bs.onnx"
            export_float(FloatStep(model, fold=True, exact=False).eval(), path, model, static=False)
        elif name == "C":
            path = OUT / "decoder_step_C.onnx"
            export_float(FloatStatic(model).eval(), path, model, static=True)
        elif name in ("Cg1", "Cg8"):
            batch = int(name[2:])
            path = OUT / f"decoder_step_{name}.onnx"
            export_fixed(FloatStatic(model).eval(), path, model, batch)
            export_encoder_fixed(EncoderRows(model).eval(), OUT / f"encoder_g{batch}.onnx", batch)
        elif name == "parity":
            info["parity"] = parity(model)
            continue
        info[name] = {"file": path.name, "seconds": time.perf_counter() - t, "bytes": path.stat().st_size,
                      "ops": op_histogram(path)}
        print(name, json.dumps(info[name]), flush=True)
    target = HERE / "results" / "export_info.json"
    target.write_text(json.dumps(info if "enc" in which else {**json.loads(target.read_text()), **info}, indent=1)
                      if target.exists() else json.dumps(info, indent=1))
