# Milestone 0: can a real v9 decoder step run in a browser fast enough for about 440 sequential calls?

2026-10-04, gpubox-4080. RTX 4080 (driver 595.71.05), i7-13700K (24 logical CPUs), headless
Chrome for Testing 153.0.8010.12 (Playwright), ONNX Runtime Web 1.30.0 (same version as
`https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/`, served locally from
`../node_modules`). Checkpoint `r2s-full-v9-colour-621bc8e-b26ad95e-47646604/best.pt`.
Scratch work only: no tracked repo file was edited, and nothing was committed or uploaded.

## Answer

**Yes for correctness. Interactive only by a loose definition (about 1 s per icon), and
the GPU does not help.** The best real browser result is **WASM with 4 threads,
variant B: 881 ms per icon median, p90 1,368 ms** on the first 32 held-out icons (350
calls median), with 100% argmax agreement. The Python server decodes the same icons in
512 ms (p90 711). WebGPU is slower than WASM at 4.7 to 5.6 ms per call because each call
issues about 370 small kernels. Graph capture brings that down to 3.75 ms per call, or
1,472 ms per icon. Scaled to the brief's 437-call median for the full validation set,
expect about 1.2 s per icon on WASM 4 threads and about 2.0 s on WebGPU.

Batch 8 (best of 8) costs almost nothing extra on WebGPU, because the cost is per call
and not per row: 4.9 ms per call at batch 8 against 4.7 ms at batch 1. On WASM it costs
5.4 times more per call (14.4 ms against 2.65 ms). Real best-of-8 sampling needs 1.69
times the greedy calls (server `decode_many`: 591 calls median against 350). Projected
browser best of 8: about 3 s per icon on WebGPU and about 8.5 s on WASM 4 threads,
before the 8 renders. The server does it in 1,153 ms (1,354 ms with rerank).

## What had to change for WebGPU to be correct (the riskiest finding)

The current export (variant **A**, 6 int64 inputs) agreed with torch on only **75.9%** of
calls on WebGPU JSEP, even though ORT CPU, WASM and the WebGPU native EP agreed 100%.
I narrowed it down by exposing every intermediate as an output (`debug_model.py`,
`debug.html`, `debug_compare.py`, `debug_rows.py`):

* On call 1 (step 1, token 5), the position, field and path embedding Gathers returned
  the *token's* row, row 5. The tokens Gather was correct.
* Cause: JSEP places small integer ops on the CPU (the int64 handling, plus the `And`,
  `Range`, `Equal` and `Cast` ops that build the mask and the Fourier input), so it adds
  several small `MemcpyFromHost` copies per run (`results/placement_A_jsep.txt`). In ORT
  Web 1.30 JSEP those small host-to-device copies corrupt one another inside a run.
  `probe/gather2.html` reproduces it in a 10-node graph: int32 ids split on the CPU and
  gathered on the GPU return row `ids[0]` for every table. Packing the ids into one int32
  input still failed in the same way (0.13 to 0.21 agreement; `attic/`).
* Fix, used in B0, B and C: feed `ids (6, B, L)` and `target (B,)` as **float32**. Keep
  every integer step on the GPU (Cast float to int32 for the Gathers, comparisons in
  float, `F.embedding` with int32 for the folded-table lookup, `CumSum` over a key slice
  for the mask positions instead of `Range`), and replace the bool `And` with a float
  product. JSEP now places only two shape `Concat`s on the CPU and inserts no copies.
  Result: **100.000% agreement on all 11,432 calls**, on every backend and batch.
* While debugging I also added exact range reduction before sin/cos. v9's Fourier
  arguments reach about 1,800 rad, and WGSL leaves sin/cos accuracy outside [-pi, pi] to
  the implementation. The diagnostic **Bs** (B with the model's own sin arguments) also
  agrees 100% on this NVIDIA/Vulkan stack, so the reduction is not needed here. It is a
  cheap guard for other GPUs. Its cost: B's logits differ from the original torch module
  by up to 7.8e-4, against 4.9e-5 when compared with the same math in torch. This is
  because torch's own float32 argument rounding at 1,800 rad is about 1e-4 rad.

The native WebGPU EP (`ort.webgpu.min.mjs`, asyncify build) ran A correctly but
slowly: 14 ms per call. The likely cause is its CPU-placed int ops forcing synchronous copies; I did not inspect the native EP's placement. With B
it is correct and close to JSEP at 5.5 ms per call. Neither EP falls back silently. Every
run asserts `adapter.info.vendor == "nvidia"` and `!isFallbackAdapter` before creating
sessions (`results/adapter_probe.json`: the default and high-performance adapters are
NVIDIA Lovelace; only `forceFallbackAdapter` returns SwiftShader).

## Traces (step 1)

`trace_record.py`, run from the repo root.

* **Icon order.** I use `data/processed/kitbash/index.json` filtered to
  `split == "primary/validation"`, taking the first 32. This is the order
  `mojidiff.vectorise.server.held_out_icons()` serves and `mojidiff.gallery.server`
  imports; the vectorise demo filters on `split != "primary/train"`, and validation
  comes before test. Pixels are `render_trusted_rgb(<hex>.svg, 144)`, exactly as
  `Vectoriser.icon_png` produces them. The first icons are 1F0CF, 1F12F, 1F194, 1F199,
  1F1EF and so on.
* **Reference decode.** I ran torch float32 on the CPU through `r2s_export.DecoderStep`,
  with the `GraphDecoder` / `ort_greedy_lib` loop: grammar on the CPU, forced tokens
  queued and fed with the next call. The final programs equal `render2svg.greedy_decode`
  for 32 of 32 icons, and `fast_decode.GraphDecoder` (CUDA float32) for 32 of 32.
* **Stored per call.** Each call records tokens, steps, field, path, axis and role (chunk
  length L), plus target, L, the 418-bit legal mask (packbits, little-endian bit order),
  the torch token and the torch top-2 margin. Files: `traces/calls.i32`, `rows.i32`,
  `masks.u8`, `margins.f32`, `images.u8`, and `manifest.json` with offsets and per-icon
  statistics.
* **Statistics for these 32 icons.** 11,432 calls and 14,535 fed steps. Calls per icon:
  median 350, p90 488, min 87, max 885 (1F0CF). Chunk length is 1 for 91.9% of calls, the
  maximum is 12, and no chunk exceeds 32, so GraphDecoder's call count equals ours. The
  smallest torch margin is 7.2e-4 (2 calls below 1e-3).

## Exports and parity (step 2)

`export_variants.py`: dynamo exporter, opset 18, example dims of 2 or more. Every
variant uses plain MatMul/Softmax attention.

| variant | what | max abs logit diff vs matched torch (random inputs, 6 shapes up to past 1,375 and b 8) | greedy over the 32 traces on ORT CPU |
|---|---|---|---|
| A | current export: growing cache, 6 int64 inputs, Einsum metric head | 5.5e-5 | 11,432 / 11,432 tokens identical |
| B0 | float ids, GPU-only int handling, exact Fourier, unfolded metric head | 4.2e-5 | identical |
| B | B0 + head and metric head folded into one constant table `W[5,418,256]`, `b[5,418]` (`logits = W[target+1] @ norm(h) + b[target+1]`; entry 0 is the plain head) | 4.9e-5 | identical |
| Bs | diagnostic: B with the original sin arguments | 1.0e-4 (vs the original torch) | identical |
| C | B + static 1,377-slot cache, written by a one-hot blend (`cache*(1-Σonehot) + onehotᵀ@new`, no ScatterElements, no int64) | 6.1e-5 | identical |
| Cg1, Cg8 | C exported with fixed shapes (batch 1 or 8, L = 1), so ORT Web graph capture can run it | same graph as C | (replayed in the browser, 100%) |
| encoder | `pixels (1,144,144,3) float32, rows (R,)` gives memory K, V (6,R,8,324,32); runs once per icon at batch 1, expanded to R rows | K 9.5e-6, V 7.6e-6 | - |

`results/ort_cpu_parity.json` holds the free-running greedy runs: each variant decodes
on its own, never teacher-forced. Folding removes the Einsum and the `metric_query`
matmul. On CPU that speeds B up over B0 by 20% (Python ORT, 8 threads: 14.6 s against
18.4 s for the 32 icons). On WASM it is 8% (2.65 against 3.02 ms per call), and on
WebGPU it is about 4%, because WebGPU is dispatch-bound.

## Browser replay (steps 3 and 4)

`replay.html` (pages A, B0, B, Bs, C) and `gc.html` (Cg with graph capture) are driven by
`run_replay.mjs` through `run_matrix.sh`, with one fresh headless browser per config.
Each config creates its sessions, runs icon 0 cold, and then runs every icon warm. Per
icon:

* The encoder runs once.
* Every recorded call runs with the recorded inputs. Logits are read back to the CPU,
  the recorded mask is applied, and the masked argmax is compared with the recorded
  token. The *recorded* token is what the next call feeds, which is teacher forcing.
* On WebGPU, KV stays on the GPU (`preferredOutputLocation: gpu-buffer` for K/V; logits
  go to `cpu`), and replaced KV tensors are disposed.
* Batch 8 repeats the same trace 8 times; agreement is checked on every row.
* `gc.html` binds preallocated GPU buffers and ping-pongs the cache between two captured
  sessions. It feeds a chunk of length L as L single steps and reads back only the last.

`ms/icon` is the wall time of the encoder plus all calls plus the per-call masked
argmax. The JS grammar is excluded because the masks are recorded; the Python
`legal_mask` costs about 12 ms per icon (`../step.json`). Model download is not included
either: the files are fp32, the decoder is 26.6 MB and the encoder 11.6 MB, served from
localhost. `nvidia-smi` was recorded before and after every config and sampled every
250 ms during it (`results/replay_*.json` → `nvidia_smi`). Utilisation was 0 to 6%
before each config, memory stayed at 7,410 MiB held by the operator's two idle servers
(PIDs 5693 and 5699), and nothing else ran.

All 32 icons unless stated. WASM batch 8 used the first 8 icons only, which include the
885-call icon 1F0CF, so its p90 is inflated. Session time is encoder plus decoder
session creation. The cold column shows cold icon 0 time (warm icon 0 time; first call).
All 8-icon rows can be compared through `first8_icon_ms_median` in `results/summary.json`.

| variant | backend | batch | icons | ms/icon median | ms/icon p90 | encoder ms | ms/call median | ms/call p90 | session ms | cold icon 0 ms (warm icon 0; 1st call) | argmax agreement |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A | WASM 1 thread | 1 | 32 | 1,263 | 1,972 | 51.6 | 3.59 | 5.68 | 279 | 6,295 (5,970; 21) | 1.0000 |
| A | WASM 4 threads | 1 | 32 | 1,057 | 1,620 | 16.7 | 3.17 | 4.50 | 280 | 4,772 (4,639; 21) | 1.0000 |
| A | WebGPU JSEP | 1 | 32 | 1,677 | 2,386 | 1.2 | 4.65 | 4.88 | 592 | 4,483 (4,346; 108) | **0.7587** |
| A | WebGPU native EP | 1 | 32 | 4,927 | 6,863 | 1.5 | 14.04 | 15.65 | 609 | 13,650 (13,974; 143) | 1.0000 |
| A | WASM 1 thread | 8 | 8 | 4,172 | 18,184 | 59.5 | 22.15 | 67.60 | 278 | 43,126 (43,187; 34) | 1.0000 |
| A | WASM 4 threads | 8 | 8 | 2,719 | 12,387 | 25.0 | 15.14 | 45.84 | 279 | 29,989 (29,966; 27) | 1.0000 |
| A | WebGPU JSEP | 8 | 32 | 1,698 | 2,356 | 1.1 | 4.67 | 5.03 | 586 | 4,834 (4,464; 130) | **0.7587** |
| A | WebGPU native EP | 8 | 32 | 5,563 | 7,157 | 1.7 | 14.55 | 19.48 | 619 | 13,012 (12,806; 80) | 1.0000 |
| B0 | WASM 1 thread | 1 | 32 | 1,255 | 1,980 | 52.0 | 3.44 | 5.72 | 275 | 6,076 (6,042; 22) | 1.0000 |
| B0 | WASM 4 threads | 1 | 32 | 1,005 | 1,565 | 16.7 | 3.02 | 4.41 | 293 | 4,798 (4,684; 22) | 1.0000 |
| B0 | WebGPU JSEP | 1 | 32 | 1,786 | 2,432 | 1.2 | 4.92 | 5.24 | 556 | 4,737 (4,474; 139) | 1.0000 |
| B0 | WebGPU native EP | 1 | 32 | 1,977 | 2,783 | 2.9 | 5.47 | 6.19 | 584 | 5,139 (5,330; 115) | 1.0000 |
| B0 | WASM 1 thread | 8 | 8 | 4,210 | 18,408 | 64.0 | 22.71 | 66.33 | 270 | 44,132 (43,552; 34) | 1.0000 |
| B0 | WASM 4 threads | 8 | 8 | 2,695 | 12,445 | 25.2 | 14.97 | 47.69 | 311 | 30,094 (30,062; 28) | 1.0000 |
| B0 | WebGPU JSEP | 8 | 32 | 1,795 | 2,502 | 1.2 | 4.98 | 5.43 | 568 | 5,022 (4,775; 148) | 1.0000 |
| B0 | WebGPU native EP | 8 | 32 | 1,963 | 2,824 | 1.7 | 5.50 | 5.81 | 589 | 5,380 (5,174; 96) | 1.0000 |
| **B** | WASM 1 thread | 1 | 32 | 1,132 | 1,832 | 51.4 | 3.15 | 5.34 | 276 | 5,911 (5,686; 17) | 1.0000 |
| **B** | **WASM 4 threads** | 1 | 32 | **881** | **1,368** | 16.6 | 2.65 | 3.95 | 272 | 4,436 (4,255; 17) | 1.0000 |
| B | WASM, no COOP/COEP (asked for 4, got 1) | 1 | 8 | 655 | 2,505 | 56.7 | 3.10 | 8.70 | 326 | 5,749 (5,743; 17) | 1.0000 |
| **B** | WebGPU JSEP | 1 | 32 | 1,731 | 2,406 | 1.3 | 4.71 | 5.08 | 557 | 4,627 (4,291; 132) | 1.0000 |
| B | WebGPU native EP | 1 | 32 | 1,971 | 2,835 | 2.8 | 5.53 | 6.16 | 576 | 5,169 (5,099; 98) | 1.0000 |
| B | WASM 1 thread | 8 | 8 | 4,192 | 18,445 | 65.6 | 22.78 | 65.84 | 277 | 43,653 (43,263; 31) | 1.0000 |
| B | WASM 4 threads | 8 | 8 | 2,604 | 12,196 | 25.5 | 14.35 | 46.31 | 274 | 29,888 (29,527; 23) | 1.0000 |
| **B** | WebGPU JSEP | 8 | 32 | 1,779 | 2,496 | 1.2 | 4.92 | 5.53 | 588 | 5,073 (4,859; 142) | 1.0000 |
| B | WebGPU native EP | 8 | 32 | 1,886 | 2,782 | 1.7 | 5.41 | 5.84 | 599 | 5,488 (5,143; 91) | 1.0000 |
| Bs | WebGPU JSEP | 1 | 32 | 1,729 | 2,503 | 1.2 | 4.84 | 5.07 | 583 | 4,712 (4,488; 128) | 1.0000 |
| Bs | WebGPU native EP | 1 | 32 | 1,945 | 2,758 | 2.8 | 5.48 | 6.20 | 578 | 5,269 (5,319; 90) | 1.0000 |
| C | WASM 1 thread | 1 | 32 | 4,865 | 6,790 | 51.5 | 13.63 | 14.31 | 271 | 12,540 (12,586; 36) | 1.0000 |
| C | WASM 4 threads | 1 | 32 | 3,204 | 4,415 | 15.8 | 9.05 | 9.41 | 286 | 8,353 (8,112; 30) | 1.0000 |
| C | WebGPU JSEP | 1 | 32 | 1,786 | 2,576 | 1.3 | 4.92 | 5.25 | 564 | 4,788 (4,554; 147) | 1.0000 |
| C | WebGPU native EP | 1 | 32 | 2,045 | 2,847 | 1.8 | 5.62 | 5.75 | 574 | 5,335 (5,180; 152) | 1.0000 |
| C | WASM 1 thread | 8 | 8 | 24,221 | 54,129 | 66.1 | 107.12 | 116.51 | 267 | 100,164 (98,237; 216) | 1.0000 |
| C | WASM 4 threads | 8 | 8 | 15,540 | 34,815 | 27.8 | 69.85 | 75.03 | 273 | 63,575 (63,114; 142) | 1.0000 |
| C | WebGPU JSEP | 8 | 32 | 2,015 | 2,797 | 1.1 | 5.44 | 5.76 | 581 | 5,389 (5,180; 245) | 1.0000 |
| C | WebGPU native EP | 8 | 32 | 2,350 | 3,184 | 2.0 | 6.25 | 6.58 | 582 | 6,007 (5,652; 230) | 1.0000 |
| **Cg1** | WebGPU JSEP + graph capture | 1 | 32 | **1,472** | 2,164 | 2.0 | 3.75 | 4.82 | 706 | 4,171 (4,062; 137) | 1.0000 |
| Cg8 | WebGPU JSEP + graph capture | 8 | 32 | 1,828 | 2,791 | 3.5 | 4.37 | 6.29 | 675 | 5,039 (5,106; 151) | 1.0000 |
| server GraphDecoder fp32 (`server_baseline.py`) | CUDA graphs, Python, same 32 icons and images | 1 | 32 | 512 | 711 | included | 1.47 (mean) | - | 0.26 s capture | - | tokens = torch, 32/32 icons |
| server `decode_many` (greedy + 7 samples at 0.7) | same | 8 | 32 | 1,153 | 1,437 | included | 1.97 (mean), 591 calls median | - | - | - | - |
| server `rerank` (best of 8, renders included) | same | 8 | 32 | 1,354 | 1,739 | included | - | - | - | - | - |

First-8-icon medians, for the 8-icon rows: server greedy 330 ms; B on WASM 4 threads at
batch 1, 461 ms; B on WASM 1 thread, 614; B with no COOP/COEP, 655; B on WebGPU JSEP,
1,077; Cg1, 984.

### Why WebGPU is slow here

`replay.html?profile=60` (`results/diagnostics/`) uses JSEP's kernel timestamps on 60
warm calls of icon 0:

* Variant B issues **373 kernels per call** and spends 2.17 ms of GPU kernel time per
  call (median), against a wall time of 5.6 ms with profiling on. MatMul accounts for
  24%, LayerNorm 16% (19 calls at about 17 µs each), Add 15% and Transpose 10%.
* C spends 2.74 ms of GPU kernel time over 396 kernels.

The cost is the number of kernels and the per-dispatch overhead, not arithmetic: about
2 MFLOP per step. Batch 8 therefore costs the same as batch 1, and graph capture, which
removes CPU-side dispatch, saves about 1 ms per call. ORT's BERT fusions
(`attic/fuse_variant.py`) cut only 437 nodes to 407 (Gelu only; pre-norm residuals do not
match SkipLayerNorm), so I dropped them.

### Other observations

* **The static cache (C) loses on WASM** (9 to 14 ms per call): every step blends and
  attends over all 1,377 slots. On WebGPU C is slightly slower than B (4.92 against 4.71
  ms per call); its only use is enabling graph capture (Cg).
* **COOP/COEP.** WASM threads need SharedArrayBuffer, which needs cross-origin isolation.
  GitHub Pages cannot send these headers. On the `NOCOI=1` server, ORT asked for 4
  threads and got `numThreads_effective = 1`, with `crossOriginIsolated = false`. On
  Pages, WASM therefore means the 1-thread row: 1,132 ms per icon median for B. A
  `coi-serviceworker` shim could restore isolation; I did not test it.
* **Cold start is small.** Session creation takes 0.27 s (WASM) to 0.6 to 0.7 s (WebGPU).
  The first call costs about 100 to 250 ms on WebGPU (shader compilation) and about 20 ms
  on WASM. The cold first icon is 100 to 400 ms slower than the same icon warm.
* **Encoder.** It is negligible on WebGPU (1.2 to 3.5 ms) and costs 16 to 17 ms on WASM 4
  threads and 51 to 66 ms on WASM 1 thread.

## Caveats

* The 32 icons are the first 32 held-out validation icons. Their median of 350 calls is
  below the brief's full-set 437. Per-call numbers transfer; per-icon numbers scale with
  calls.
* The replay is teacher-forced with recorded masks, so the JS grammar and SVG
  serialisation costs are excluded. Batch 8 repeats the greedy trace and does not
  reproduce real samples, which diverge and need about 1.69 times the calls.
* WASM batch 8 used only 8 icons. WASM timings share the CPU with the operator's idle
  servers.
* Everything is fp32. The ONNX files total 211 MB in `onnx/`; the deployable pair is the
  B decoder (26.6 MB) and the encoder (11.6 MB). No int8 or fp16 variant was benchmarked
  here.
* B, B0, C and Cg change the arithmetic slightly (exact Fourier). Their logits sit within
  1e-3 of the original torch module and within 6e-5 of the matched torch math, and
  greedy tokens are identical on all 11,432 calls.

## Files

* `trace_record.py`: recorded traces, the torch, greedy_decode and GraphDecoder
  cross-check, and free-running ORT CPU parity for every variant.
* `export_variants.py`: the A, B0, B, Bs, C, Cg1 and Cg8 exports, the encoder export,
  and random-input parity.
* `trace_replay_cpu.py`: teacher-forced ORT CPU replay of one icon (the reference logits
  for the debug pages).
* `replay.html`, `gc.html`, `run_replay.mjs`, `run_matrix.sh`, `server.mjs`: the browser
  harness. The server defaults to port 8941 with COOP/COEP; `NOCOI=1` turns the headers
  off (port 8942 for the Pages-like run). `probe.mjs` is the adapter probe.
* `debug_model.py`, `debug.html`, `run_debug.mjs`, `debug_compare.py`, `debug_rows.py`,
  `probe/`: the bisection of the JSEP failure, with minimal reproductions in
  `probe/make_gather*.py` and `probe/gather*.html`, and node placement in
  `probe/placement.*`.
* `server_baseline.py`: the GraphDecoder baseline, batch 1, batch 8 and rerank.
* `make_table.py`: writes `results/summary.json` and `results/table.md` from the
  `results/replay_*.json` files. Those raw files hold per-icon times, agreement,
  disagreements, the adapter, and nvidia-smi before, during and after.
* `results/export_info.json` (op histograms), `results/export.log` (parity),
  `results/ort_cpu_parity.json`, `results/server_baseline.json`,
  `results/adapter_probe.json`, `results/placement_A_jsep.txt`.
* `attic/`: superseded attempts (int32-packed inputs, BERT fusion), kept as negative
  results.

Reproduce: `cd /home/dev/workspace/mojidiff`, then run `.venv/bin/python
<m0>/export_variants.py`, `<m0>/trace_record.py` and `<m0>/server_baseline.py`. Then in
`<m0>`: `PORT=8941 node server.mjs &`, `NOCOI=1 PORT=8942 node server.mjs &`,
`./run_matrix.sh`, `PAGE=gc.html node run_replay.mjs "Cg1_jsep_capture_b1=b=1"
"Cg8_jsep_capture_b8=b=8"`, and finally `python3 make_table.py`.

## Suggested next step

To make WebGPU worth using, cut the kernel count rather than FLOPs. Options:

* Hand-fused ops: ORT contrib `MultiHeadAttention` with past KV, `SkipLayerNormalization`
  written to match the pre-norm graph, and fused QKV.
* A tiny custom WGSL decoder step: 6 layers in a handful of dispatches.

Either should reach well under 1 ms per call. Short of that, ship WASM with 4 threads
behind a COOP/COEP-capable host or service worker: about 0.9 to 1.2 s per greedy icon.
