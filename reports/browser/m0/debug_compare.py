"""Compare debug.html dumps with ORT CPU on the same calls; print the first intermediates that diverge."""
import json, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "ortlib"))
import numpy as np, onnxruntime as ort
v, n, tag = sys.argv[1], int(sys.argv[2]), sys.argv[3]
T = HERE / "traces"
calls = np.fromfile(T / "calls.i32", np.int32).reshape(-1, 4)
rows = np.fromfile(T / "rows.i32", np.int32).reshape(6, -1)
images = np.fromfile(T / "images.u8", np.uint8).reshape(-1, 144, 144, 3)
enc = ort.InferenceSession(str(HERE / "onnx/encoder.onnx"), providers=["CPUExecutionProvider"])
dec = ort.InferenceSession(str(HERE / f"onnx/decoder_step_{v}_dbg.onnx"), providers=["CPUExecutionProvider"])
mk, mv = enc.run(None, {"pixels": images[0][None].astype(np.float32), "rows": np.zeros(1, np.int64)})
pk = np.zeros((6, 1, 8, 0, 32), np.float32); pv = pk
names = [o.name for o in dec.get_outputs()]
for c in range(n + 1):
    L, target, token, off = calls[c]
    if v in ("B0", "B", "Bs", "C"):
        f = {"ids": rows[:, off:off + L][:, None].astype(np.float32), "target": np.array([target], np.float32)}
    else:
        f = {nm: rows[i, off:off + L][None].astype(np.int64) for i, nm in enumerate(["tokens", "steps", "field", "path", "axis", "role"])}
        f["target"] = np.array([target], np.int64)
    f.update(past_keys=pk, past_values=pv, memory_keys=mk, memory_values=mv)
    res = dict(zip(names, dec.run(None, f)))
    pk, pv = res["present_keys"], res["present_values"]
    print("call", c, "L", L, "past", pk.shape[3] - L)
d = json.load(open(HERE / f"results/replay_{tag}.json"))["outputs"]
order = (HERE / f"onnx/decoder_step_{v}_dbg.order.txt").read_text().split("\n")
import onnx
g = onnx.load(str(HERE / f"onnx/decoder_step_{v}_dbg.onnx"), load_external_data=False).graph
producer = {o: (nd.op_type, nd.name, list(nd.input)) for nd in g.node for o in nd.output}
shown = 0
for name in ["logits", "present_keys", "present_values"] + order:
    if name not in d: continue
    a = np.array(d[name]["data"], np.float32).reshape(d[name]["dims"]) if d[name]["dims"] else np.array(d[name]["data"], np.float32)
    b = res[name]
    if a.shape != b.shape:
        print("SHAPE", name, a.shape, b.shape); continue
    err = float(np.abs(a - b).max()) if a.size else 0.0
    if name in ("logits", "present_keys", "present_values") or (err > 1e-3 and shown < 12):
        print(f"{err:10.5f} {name} {producer.get(name, ('', '', []))[0]} {a.shape} inputs={producer.get(name, ('', '', []))[2][:3]}")
        if name not in ("logits", "present_keys", "present_values"): shown += 1
