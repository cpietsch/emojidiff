"""Server baseline on the same 32 icons: fast_decode.GraphDecoder, CUDA float32 (what the
vectorise demo and gallery serve), RTX 4080. Times greedy decode (batch 1) and the batch-8
decode_many (greedy + 7 samples at 0.7, seed 0) that `rerank` runs before rendering, plus
full `rerank` (decode + 8 renders + compare). Encoder included in every number (it runs
inside decode). nvidia-smi recorded before and after.

Usage (repo root as cwd): python server_baseline.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import numpy as np
import torch

from mojidiff.learning.fast_decode import GraphDecoder, rerank
from mojidiff.learning.openmoji_pilot import _load_program, load_openmoji_pilot_config, load_pilot_index
from mojidiff.learning.render2svg import DecodeStats
from r2s_export import load

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def smi() -> str:
    return subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                           "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip()


def main() -> None:
    man = json.loads((HERE / "traces/manifest.json").read_text())
    images = np.fromfile(HERE / "traces/images.u8", np.uint8).reshape(-1, 144, 144, 3)
    model = load().cuda()
    pilot = load_openmoji_pilot_config(Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml"))
    by_split, _, _ = load_pilot_index(pilot)
    template = _load_program(by_split["primary/train"][0], pilot, model.layout.codec)
    res: dict = {"smi_before": smi()}
    t = time.perf_counter()
    g1 = GraphDecoder(model, dtype=torch.float32)
    g8 = GraphDecoder(model, dtype=torch.float32, batch=8)
    torch.cuda.synchronize()
    res["graph_capture_s"] = time.perf_counter() - t
    rows = []
    for k, icon in enumerate(man["icons"]):
        image = torch.from_numpy(images[k]).cuda()
        g1.decode(image)  # warm per icon is not needed; graphs are captured, but keep it fair: time the 2nd run
        out: dict = {"hexcode": icon["hexcode"]}
        for name, fn in (("greedy", lambda s: g1.decode(image, stats=s)),
                         ("decode8", lambda s: g8.decode_many(image, temperature=0.7, generator=torch.Generator().manual_seed(0), stats=s)),
                         ("rerank8", lambda s: rerank(g8, image, template, stats=s))):
            stats = DecodeStats()
            torch.cuda.synchronize()
            t = time.perf_counter()
            fn(stats)
            torch.cuda.synchronize()
            out[f"{name}_ms"] = (time.perf_counter() - t) * 1000
            out[f"{name}_calls"] = stats.model_calls
        rows.append(out)
        print(k, json.dumps({a: round(b, 1) if isinstance(b, float) else b for a, b in out.items()}), flush=True)
    res["icons"] = rows
    for name in ("greedy", "decode8", "rerank8"):
        v = [r[f"{name}_ms"] for r in rows]
        c = [r[f"{name}_calls"] for r in rows]
        res[name] = {"ms_median": float(np.median(v)), "ms_p90": float(np.percentile(v, 90)),
                     "calls_median": float(np.median(c)),
                     "ms_per_call_mean": float(sum(v) / sum(c))}
    res["smi_after"] = smi()
    (HERE / "results/server_baseline.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("greedy", "decode8", "rerank8", "smi_before", "smi_after", "graph_capture_s")}, indent=1))


if __name__ == "__main__":
    main()
