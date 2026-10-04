"""Milestone 0, step 1-2: record greedy decoder-call traces for the first N held-out
validation icons and check the ONNX variants against them.

Icon order: `data/processed/kitbash/index.json` filtered to split == "primary/validation",
the order `mojidiff.vectorise.server.held_out_icons()` (vectorise demo) and the gallery
(`mojidiff.gallery.server`, which imports `held_out_icons`) offer them. Pixels: the
cached canonical SVG rendered by `render2svg.render_trusted_rgb(svg, 144)`, exactly as
`Vectoriser.icon_png` serves them.

Reference decode: torch float32 on CPU through `r2s_export.DecoderStep` (the module the
ONNX graph A was exported from), with the decode loop of `fast_decode.GraphDecoder` /
`ort_greedy_lib` (grammar on the CPU, forced tokens queued and fed with the next call;
one call per free position, no 32-step chunk cap). The final program is checked against
`render2svg.greedy_decode` (CPU float32, the evaluated decoder) and `GraphDecoder`
(CUDA float32, what the demos serve).

Per decoder call we store: tokens, steps, field, path, axis, role (each of chunk length
L), target, L, the 418-bit legal mask, the torch-chosen token, and the torch margin
(best legal logit minus second best).

Then each ONNX variant (A, B0, B, Bs, C) decodes the same icons free-running on ORT CPU and must
choose identical tokens at every call.

Usage (from the repo root): python trace_record.py [N=32]
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
sys.path.append(str(HERE.parent / "ortlib"))

import numpy as np
import torch

from mojidiff.learning.autoregressive import PATH_STRIDE, SEGMENT_STRIDE, legal_mask
from mojidiff.learning.render2svg import (
    _CONTROL_SLOTS,
    ROLE_CONTROL,
    ROLE_ENDPOINT,
    ROLE_NONE,
    _static_tables,
    path_major_positions,
    render_trusted_rgb,
)
from r2s_export import DecoderStep, Encoder, load

REPO = Path("/home/dev/workspace/mojidiff")
ICONS = REPO / "data/processed/kitbash"
TRACES = HERE / "traces"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

model = load()
layout = model.layout
slot_t, kind_t, axis_t = (t.tolist() for t in _static_tables(layout))
LAYERS, HEADS, HD = model.config.decoder_layers, model.config.heads, model.config.d_model // model.config.heads


def role_of(position: int, decoded: list[int]) -> tuple[int, int]:
    axis = axis_t[position]
    if axis >= 0:
        return ROLE_ENDPOINT, axis
    slot = slot_t[position]
    if slot < 0:
        return ROLE_NONE, 0
    kind = min(max(decoded[kind_t[position]], 0), 4)
    controls = _CONTROL_SLOTS[kind]
    used = controls + 2 if 1 <= kind <= 3 else 0
    if slot >= used:
        return ROLE_NONE, slot % 2
    return (ROLE_CONTROL if slot < controls else ROLE_ENDPOINT), slot % 2


def labels_of(position: int, decoded: list[int]) -> tuple[int, int]:
    if position < layout.path_positions:
        return position % PATH_STRIDE, position // PATH_STRIDE
    offset = position - layout.path_positions
    slot = offset // SEGMENT_STRIDE
    field = PATH_STRIDE + offset % SEGMENT_STRIDE
    end = 0
    for path in range(layout.codec.max_paths):
        end += decoded[path * PATH_STRIDE]
        if slot < end:
            return field, path
    return field, layout.codec.max_paths


def greedy(memory, runner) -> tuple[list[int], list[dict]]:
    """Greedy decode; `runner(rows (6, L) int64, target int, state) -> (logits (V,), state)`."""
    decoded = [0] * layout.length
    tensor = torch.zeros(layout.length, dtype=torch.long)
    walker = path_major_positions(tensor, layout)
    order: list[int] = []
    state = runner.start(memory)
    fed = 0
    calls: list[dict] = []
    for step in range(layout.length):
        position = next(walker)
        order.append(position)
        mask = legal_mask(position, tensor, layout).numpy()
        if mask.sum() == 1:
            token = int(mask.argmax())
            decoded[position] = token
            tensor[position] = token
            continue
        steps = list(range(fed, step + 1))
        rows = np.zeros((6, len(steps)), np.int64)
        for i, current in enumerate(steps):
            previous = order[current - 1] if current > 0 else None
            here = order[current]
            rows[0, i] = decoded[previous] if previous is not None else 0
            rows[1, i] = current
            rows[2, i], rows[3, i] = labels_of(here, decoded)
            if previous is not None:
                role_in, axis_in = role_of(previous, decoded)
                rows[4, i] = axis_in
                rows[5, i] = role_in
        role_out, axis_out = role_of(position, decoded)
        target = (role_out - 1) * 2 + axis_out if role_out != ROLE_NONE else -1
        logits, state = runner(rows, target, state)
        fed = step + 1
        scores = np.where(mask, logits, -np.inf)
        token = int(scores.argmax())
        top2 = np.sort(scores[np.isfinite(scores)])[-2:] if mask.sum() > 1 else np.array([0.0, 0.0])
        calls.append({"rows": rows, "target": target, "mask": mask, "token": token,
                      "margin": float(top2[-1] - top2[0]), "position": position})
        decoded[position] = token
        tensor[position] = token
    return decoded, calls


class TorchRunner:
    def __init__(self) -> None:
        self.step = DecoderStep(model, use_sdpa=False).eval()

    def start(self, memory):
        mk, mv = memory
        return (torch.zeros(LAYERS, 1, HEADS, 0, HD), torch.zeros(LAYERS, 1, HEADS, 0, HD), mk, mv)

    @torch.no_grad()
    def __call__(self, rows, target, state):
        pk, pv, mk, mv = state
        r = torch.from_numpy(rows)[:, None]
        logits, pk, pv = self.step(r[0], r[1], r[2], r[3], r[4], r[5], torch.tensor([target]), pk, pv, mk, mv)
        return logits[0].numpy(), (pk, pv, mk, mv)


class OrtRunner:
    def __init__(self, path: Path, static: bool, packed: bool) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 8
        self.sess = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
        self.static = static
        self.packed = packed

    def start(self, memory):
        mk, mv = (m.numpy() for m in memory)
        if self.static:
            T = layout.length + 1
            return (np.zeros((LAYERS, 1, HEADS, T, HD), np.float32), np.zeros((LAYERS, 1, HEADS, T, HD), np.float32), mk, mv)
        return (np.zeros((LAYERS, 1, HEADS, 0, HD), np.float32), np.zeros((LAYERS, 1, HEADS, 0, HD), np.float32), mk, mv)

    def __call__(self, rows, target, state):
        pk, pv, mk, mv = state
        names = ("tokens", "steps", "field", "path", "axis", "role")
        if self.packed:
            feeds = {"ids": rows[:, None].astype(np.float32), "target": np.array([target], np.float32),
                     "memory_keys": mk, "memory_values": mv}
            if self.static:
                feeds.update(cache_keys=pk, cache_values=pv)
            else:
                feeds.update(past_keys=pk, past_values=pv)
        else:
            feeds = {n: rows[i][None] for i, n in enumerate(names)}
            feeds["target"] = np.array([target], np.int64)
            feeds.update(past_keys=pk, past_values=pv, memory_keys=mk, memory_values=mv)
        logits, pk, pv = self.sess.run(None, feeds)
        return logits[0], (pk, pv, mk, mv)


def icons() -> list[dict]:
    index = json.loads((ICONS / "index.json").read_text())
    return [icon for icon in index if icon["split"] == "primary/validation"][:N]


def main() -> None:
    from mojidiff.learning.render2svg import greedy_decode

    TRACES.mkdir(exist_ok=True)
    chosen = icons()
    images = np.stack([render_trusted_rgb((ICONS / f"{i['hexcode']}.svg").read_bytes(), 144) for i in chosen])
    encoder = Encoder(model).eval()
    torch_runner = TorchRunner()
    manifest: dict = {"source_order": "data/processed/kitbash/index.json, split == primary/validation, first N "
                      "(= mojidiff.vectorise.server.held_out_icons() order, also used by mojidiff.gallery.server)",
                      "checkpoint": "/home/dev/.cache/mojidiff/runs/r2s-full-v9-colour-621bc8e-b26ad95e-47646604/best.pt",
                      "vocabulary": layout.vocabulary, "length": layout.length, "mask_bytes": (layout.vocabulary + 7) // 8,
                      "mask_bitorder": "little", "rows_fields": ["tokens", "steps", "field", "path", "axis", "role"],
                      "calls_fields": ["length", "target", "token", "row_offset"], "icons": []}
    all_calls: list[dict] = []
    row_offset = 0
    reference: dict = {}
    for k, icon in enumerate(chosen):
        image = torch.from_numpy(images[k])
        with torch.no_grad():
            mk, mv = encoder(image[None].float())
        t = time.perf_counter()
        decoded, calls = greedy((mk, mv), torch_runner)
        torch_s = time.perf_counter() - t
        with torch.no_grad():
            ref = greedy_decode(model, image[None])[0].tolist()
        reference[icon["hexcode"]] = {"decoded": decoded, "greedy_decode_equal": ref == decoded}
        lengths = [c["rows"].shape[1] for c in calls]
        manifest["icons"].append({"hexcode": icon["hexcode"], "annotation": icon["annotation"], "calls": len(calls),
                                  "call_offset": len(all_calls), "row_offset": row_offset, "rows": sum(lengths),
                                  "max_chunk": max(lengths), "chunk1_fraction": lengths.count(1) / len(lengths),
                                  "torch_cpu_s": torch_s, "greedy_decode_equal": ref == decoded,
                                  "min_margin": min(c["margin"] for c in calls)})
        for c in calls:
            c["row_offset"] = row_offset
            row_offset += c["rows"].shape[1]
        all_calls.extend(calls)
        print(k, icon["hexcode"], len(calls), "calls", f"{torch_s:.1f}s", "greedy_decode equal:", ref == decoded, flush=True)

    # GraphDecoder (CUDA float32), the server path
    if torch.cuda.is_available():
        from mojidiff.learning.fast_decode import GraphDecoder

        gmodel = load().cuda()
        graph = GraphDecoder(gmodel, dtype=torch.float32)
        for k, icon in enumerate(chosen):
            out = graph.decode(torch.from_numpy(images[k]).cuda())[0].tolist()
            reference[icon["hexcode"]]["graph_decoder_equal"] = out == reference[icon["hexcode"]]["decoded"]
            manifest["icons"][k]["graph_decoder_equal"] = reference[icon["hexcode"]]["graph_decoder_equal"]

    calls_arr = np.array([[c["rows"].shape[1], c["target"], c["token"], c["row_offset"]] for c in all_calls], np.int32)
    rows_arr = np.concatenate([c["rows"] for c in all_calls], axis=1).astype(np.int32)  # (6, R) field-major
    masks_arr = np.stack([np.packbits(c["mask"].astype(np.uint8), bitorder="little") for c in all_calls])
    margins = np.array([c["margin"] for c in all_calls], np.float32)
    calls_arr.tofile(TRACES / "calls.i32")
    rows_arr.tofile(TRACES / "rows.i32")
    masks_arr.tofile(TRACES / "masks.u8")
    margins.tofile(TRACES / "margins.f32")
    images.astype(np.uint8).tofile(TRACES / "images.u8")
    manifest.update(total_calls=len(all_calls), total_rows=int(rows_arr.shape[1]),
                    files={"calls.i32": list(calls_arr.shape), "rows.i32": list(rows_arr.shape),
                           "masks.u8": list(masks_arr.shape), "margins.f32": list(margins.shape),
                           "images.u8": list(images.shape)})
    cs = [i["calls"] for i in manifest["icons"]]
    lens = calls_arr[:, 0]
    manifest["summary"] = {"calls_median": float(np.median(cs)), "calls_p90": float(np.percentile(cs, 90)),
                           "calls_max": int(max(cs)), "calls_min": int(min(cs)),
                           "chunk1_fraction": float((lens == 1).mean()), "max_chunk": int(lens.max()),
                           "calls_over_32": int((lens > 32).sum()),
                           "greedy_decode_equal": sum(i["greedy_decode_equal"] for i in manifest["icons"]),
                           "graph_decoder_equal": sum(bool(i.get("graph_decoder_equal")) for i in manifest["icons"])}
    (TRACES / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest["summary"]), flush=True)

    # ORT CPU free-running greedy per variant: identical tokens at every call?
    parity: dict = {}
    for name, static, packed in (("A", False, False), ("B0", False, True), ("B", False, True), ("Bs", False, True),
                                 ("C", True, True)):
        path = HERE / "onnx" / f"decoder_step_{name}.onnx"
        runner = OrtRunner(path, static, packed)
        same_calls = 0
        same_icons = 0
        total = 0
        first_diff = []
        t = time.perf_counter()
        for k, icon in enumerate(chosen):
            image = torch.from_numpy(images[k])
            with torch.no_grad():
                mk, mv = encoder(image[None].float())
            decoded, calls = greedy((mk, mv), runner)
            ref = manifest["icons"][k]
            ref_calls = all_calls[ref["call_offset"]: ref["call_offset"] + ref["calls"]]
            same = decoded == reference[icon["hexcode"]]["decoded"]
            same_icons += same
            for a, b in zip(calls, ref_calls):
                total += 1
                same_calls += a["token"] == b["token"]
            if not same:
                first_diff.append(icon["hexcode"])
        parity[name] = {"icons_identical": same_icons, "icons": len(chosen), "calls_same_token": same_calls,
                        "calls_compared": total, "differing_icons": first_diff,
                        "ort_cpu_8threads_s": time.perf_counter() - t}
        print(name, parity[name], flush=True)
    (HERE / "results" / "ort_cpu_parity.json").write_text(json.dumps(parity, indent=1))


if __name__ == "__main__":
    main()
