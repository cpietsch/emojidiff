"""Collect results/replay_*.json (+ server_baseline.json) into results/summary.json and a
markdown table (results/table.md). Usage: python make_table.py"""
import json
import statistics
from pathlib import Path


def p90(values):
    """Linear-interpolated 90th percentile (numpy's default)."""
    v = sorted(values)
    x = 0.9 * (len(v) - 1)
    lo = int(x)
    return v[lo] + (v[min(lo + 1, len(v) - 1)] - v[lo]) * (x - lo)


HERE = Path(__file__).resolve().parent
R = HERE / "results"
BACKEND = {("wasm", "1"): "WASM 1 thread", ("wasm", "4"): "WASM 4 threads",
           ("webgpu", "jsep"): "WebGPU JSEP", ("webgpu", "native"): "WebGPU native EP"}
ORDER = ["A", "B0", "B", "Bs", "C", "Cg1", "Cg8"]


def backend(d):
    if d["ep"] == "wasm":
        b = f"WASM {d.get('numThreads_effective')} thr"
        if "nocoi" in d["tag"]:
            b += " (no COOP/COEP, asked 4)"
        return b
    if d.get("variant", "").startswith("Cg"):
        return "WebGPU JSEP + graph capture" if d.get("capture") else "WebGPU JSEP fixed-shape"
    return "WebGPU native EP" if "webgpu.min" in d.get("bundle", "") else "WebGPU JSEP"


rows = []
for f in sorted(R.glob("replay_*.json")):
    d = json.loads(f.read_text())
    if "smoke" in f.name or "dbg" in f.name:
        continue
    s = d.get("summary")
    if not s:
        rows.append({"tag": d.get("tag"), "error": (d.get("error") or "")[:200]})
        continue
    smi = d.get("nvidia_smi", {})
    rows.append({
        "tag": d["tag"], "variant": d["variant"], "backend": backend(d), "batch": d["batch"], "icons": s["icons"],
        # per-icon statistics recomputed here (the page's pct() takes the upper middle for even counts)
        "calls": s["calls"], "icon_ms_median": statistics.median(r["icon_ms"] for r in d["icons"]),
        "icon_ms_p90": p90([r["icon_ms"] for r in d["icons"]]),
        "first8_icon_ms_median": statistics.median(r["icon_ms"] for r in d["icons"][:8]),
        "encoder_ms_median": statistics.median(r["encoder_ms"] for r in d["icons"]), "call_ms_median": s["call_ms_median"],
        "call_ms_p90": s["call_ms_p90"], "call_ms_mean": s["call_ms_mean"], "session_ms": s["session_ms"],
        "cold_icon_ms": s["cold_icon_ms"], "cold_encoder_ms": s["cold_encoder_ms"],
        "cold_first_call_ms": s["cold_first_call_ms"], "agreement": s["agreement"],
        "agreement_all_rows": s.get("agreement_all_rows"), "threads": d.get("numThreads_effective"),
        "coi": d.get("crossOriginIsolated"), "adapter": d.get("adapter"),
        "smi_before": smi.get("before", {}).get("gpu"), "smi_after": smi.get("after", {}).get("gpu"),
        "util_max_during": smi.get("during", {}).get("util_max"), "wall_s": d.get("wall_s"),
        "warm_icon0_ms": d["icons"][0]["icon_ms"], "cold_overhead_ms": s["cold_icon_ms"] - d["icons"][0]["icon_ms"],
        "steps": s.get("steps"),
    })
ok = [r for r in rows if "error" not in r]
ok.sort(key=lambda r: (ORDER.index(r["variant"]) if r["variant"] in ORDER else 99, r["batch"], r["backend"]))
server = json.loads((R / "server_baseline.json").read_text()) if (R / "server_baseline.json").exists() else None
(R / "summary.json").write_text(json.dumps({"rows": ok, "errors": [r for r in rows if "error" in r],
                                            "server_baseline": {k: server[k] for k in ("greedy", "decode8", "rerank8", "smi_before", "smi_after")} if server else None}, indent=1))
f = lambda x, n=0: "-" if x is None else f"{x:,.{n}f}"  # noqa: E731
lines = ["| variant | backend | batch | icons | ms/icon median | ms/icon p90 | encoder ms | ms/call median | ms/call p90 | session ms | cold icon 0 ms (warm icon 0; 1st call) | argmax agreement |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in ok:
    lines.append(f"| {r['variant']} | {r['backend']} | {r['batch']} | {r['icons']} | {f(r['icon_ms_median'])} | {f(r['icon_ms_p90'])} | "
                 f"{f(r['encoder_ms_median'], 1)} | {f(r['call_ms_median'], 2)} | {f(r['call_ms_p90'], 2)} | {f(r['session_ms'])} | "
                 f"{f(r['cold_icon_ms'])} ({f(r['warm_icon0_ms'])}; {f(r['cold_first_call_ms'])}) | {r['agreement']:.4f} |")
if server:
    g, d8, r8 = server["greedy"], server["decode8"], server["rerank8"]
    lines.append(f"| server GraphDecoder fp32 | CUDA graphs, Python | 1 | 32 | {f(g['ms_median'])} | {f(g['ms_p90'])} | incl. | {f(g['ms_per_call_mean'], 2)} (mean) | - | - | - | = torch (32/32 icons) |")
    lines.append(f"| server GraphDecoder fp32 | decode_many b8 (real samples, {d8['calls_median']:.0f} calls median) | 8 | 32 | {f(d8['ms_median'])} | {f(d8['ms_p90'])} | incl. | {f(d8['ms_per_call_mean'], 2)} (mean) | - | - | - | - |")
    lines.append(f"| server rerank (best of 8) | + 8 renders + compare | 8 | 32 | {f(r8['ms_median'])} | {f(r8['ms_p90'])} | incl. | - | - | - | - | - |")
(R / "table.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
for r in rows:
    if "error" in r:
        print("ERROR", r)
