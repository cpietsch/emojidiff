"""Teacher-forced replay of m0/traces through an ONNX variant on ORT CPU (the reference the
browser replay is compared against). Usage: python trace_replay_cpu.py VARIANT ICON NCALLS [ep]
Writes results/ref_logits_<VARIANT>_<ICON>.npy (NCALLS, 418)."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "ortlib"))
import numpy as np, onnxruntime as ort

T = HERE / "traces"
man = json.loads((T / "manifest.json").read_text())
calls = np.fromfile(T / "calls.i32", np.int32).reshape(-1, 4)
rows = np.fromfile(T / "rows.i32", np.int32).reshape(6, -1)
images = np.fromfile(T / "images.u8", np.uint8).reshape(-1, 144, 144, 3)
masks = np.unpackbits(np.fromfile(T / "masks.u8", np.uint8).reshape(len(calls), -1), axis=1, bitorder="little")[:, :418].astype(bool)

def replay(variant, k, n=None):
    o = ort.SessionOptions(); o.intra_op_num_threads = 8
    enc = ort.InferenceSession(str(HERE / "onnx/encoder.onnx"), o, providers=["CPUExecutionProvider"])
    dec = ort.InferenceSession(str(HERE / f"onnx/decoder_step_{variant}.onnx"), o, providers=["CPUExecutionProvider"])
    icon = man["icons"][k]
    mk, mv = enc.run(None, {"pixels": images[k][None].astype(np.float32), "rows": np.zeros(1, np.int64)})
    static = variant == "C"
    S = man["length"] + 1
    pk = np.zeros((6, 1, 8, S if static else 0, 32), np.float32); pv = pk.copy()
    out, agree = [], 0
    names = ["tokens", "steps", "field", "path", "axis", "role"]
    for c in range(icon["call_offset"], icon["call_offset"] + (n or icon["calls"])):
        L, target, token, off = calls[c]
        if variant in ("B0", "B", "Bs", "C"):
            f = {"ids": rows[:, off:off + L][:, None].astype(np.float32), "target": np.array([target], np.float32)}
        else:
            f = {nm: rows[i, off:off + L][None].astype(np.int64) for i, nm in enumerate(names)}
            f["target"] = np.array([target], np.int64)
        if static:
            f.update(cache_keys=pk, cache_values=pv)
        else:
            f.update(past_keys=pk, past_values=pv)
        f.update(memory_keys=mk, memory_values=mv)
        lg, pk, pv = dec.run(None, f)
        out.append(lg[0])
        agree += int(np.where(masks[c], lg[0], -np.inf).argmax() == token)
    return np.stack(out), agree

if __name__ == "__main__":
    v, k, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    lg, agree = replay(v, k, n)
    np.save(HERE / "results" / f"ref_logits_{v}_{k}.npy", lg)
    print(v, k, n, "agree", agree)
