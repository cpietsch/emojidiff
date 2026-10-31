# Milestone 1: the final model decodes free-running in the browser, exactly as in Python

2026-10-31, gpubox-4080, branch `browser-m1`. Model: run
`r2s-cur-d02-s120k-seed2-53b3108-46294452-47646604`, `best.pt` at step 64,000 (sha256
`38daeca4…`), 9,025,826 parameters, exported as milestone 0's variant B by
`python -m mojidiff.learning.onnx_export` (fp32; `encoder.onnx` 11,442,722 bytes, sha256
`bde28614…`; `decoder_step.onnx` 26,302,506 bytes, sha256 `969712df…`).

## Answer

**Correct everywhere, about 1.2 s per icon at the median on the best browser backend
(WASM, 4 threads), and the JavaScript grammar costs 0.2% of that.** Nothing is replayed:
the JS grammar (`web/src/grammar.mjs`) picks every legal mask, the loop
(`web/src/decode.mjs`) feeds back the model's own tokens, and the JS codec writes the SVG.

* On 32 validation icons spread over the split, all 1,376 tokens of every icon equal the
  Python ONNX Runtime decoder's (`onnx_decode.OnnxDecoder`). So do the decoder-call counts
  (14,090 calls per config) and the SVG bytes (`packed.serialize_packed_svg` vs
  `web/src/codec.mjs`). This holds on WASM 4 threads, WASM 1 thread and WebGPU in headless
  Chrome, and on onnxruntime-node. In Node the programs also equal torch `greedy_decode`'s,
  read from the committed fixtures (32 of 32).
* WebGPU, free-running through 14,090 calls, is exact too, so variant B's float32-id fix
  holds for the final model.
* Latency is all in the decoder calls: 98.6% of the time on WASM and 99.8% on WebGPU. The
  grammar, ids and masked argmax take 2.7 to 3.8 ms per icon, and writing the SVG takes
  0.08 ms.
* WebGPU is slower than WASM on every icon here, as in milestone 0. It costs a flat 5 ms
  per call because it is dispatch-bound. WASM per-call cost grows with the cache, from
  1.6 ms per call on the lightest icon (113 calls) to 4.2 ms on the heaviest (938 calls),
  so the gap narrows on heavy icons: 3,990 ms on WASM against 4,616 ms on WebGPU.

## Results

32 icons, warm. Icon 0 runs once cold first and is reported separately. Every row has
32 of 32 icons with identical tokens, identical SVG and equal decoder calls. Calls per
icon: median 419, p90 704, total 14,090.

| backend | ms/icon median | ms/icon p90 | mean | max | ms/call (mean) | encoder ms | grammar ms (median) | grammar share | SVG ms | session ms | cold icon 0 ms (warm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **Chrome, WASM 4 threads** | **1,171** | **2,697** | 1,455 | 3,990 | 3.26 | 16.8 | 2.67 | 0.22% | 0.077 | 320 | 2,332 (2,198) |
| same, repeat | 1,179 | 2,625 | 1,434 | 3,933 | 3.21 | 16.2 | 3.00 | 0.23% | 0.075 | 300 | 2,326 (2,202) |
| Chrome, WebGPU (JSEP, `ort.all.min.mjs`) | 2,086 | 3,466 | 2,203 | 4,616 | 4.99 | 1.15 | 3.78 | 0.17% | 0.082 | 616 | 3,218 (2,914) |
| same, repeat | 2,091 | 3,526 | 2,207 | 4,770 | 5.00 | 1.15 | 3.61 | 0.17% | 0.075 | 598 | 3,216 (2,905) |
| Chrome, WASM 1 thread (what GitHub Pages gets without COOP/COEP) | 1,519 | 3,447 | 1,928 | 5,296 | 4.25 | 51.9 | 2.72 | 0.17% | 0.078 | 292 | 3,050 (2,952) |
| Chrome, WASM, no COOP/COEP (`NOCOI=1`: 4 threads asked, 1 effective, not isolated; review check) | 1,506 | 3,439 | 1,913 | 5,179 | 4.22 | 51.6 | 2.75 | 0.16% | 0.100 | 300 | 3,072 (3,070) |
| Node, onnxruntime-node CPU, 4 threads (not a browser) | 746 | 1,681 | 935 | 2,589 | 2.10 | 7.5 | 3.43 | 0.41% | 0.095 | 92 | - |
| Python ORT CPU reference, default threads (indicative, not under the same conditions) | 538 | 999 | - | - | - | - | - | - | - | - | - |

How to read the columns:

* **ms/icon**: from the Uint8 pixels to the SVG string. It includes the float conversion,
  the encoder, every decoder call, the JS grammar, the masked argmax, the codec, and
  releasing the cache. It excludes the model fetch (37.7 MB fp32 from localhost) and
  session creation (`session ms`).
* **grammar share**: summed over icons, the time outside the encoder and decoder calls
  divided by ms/icon. That time covers legal masks, labels, ids, masked argmax, and loop and
  await overhead.
* **ms/call**: summed decoder-call time divided by summed calls.
* Medians and p90 use linear interpolation (numpy's default) over the 32 icons.
* **The p90 understates the whole split.** The 32 icons match the split's median calls
  (419 against 425) but not its tail (p90 704 against 782). Interpolating each config's
  measured ms/icon against calls and evaluating it at the calls of all 339 validation icons
  (from the torch greedy decodes in `web/test/fixtures`) leaves the medians where they are
  (WASM 4 threads 1,173 ms) but raises the p90s to about 3,170 ms (WASM 4 threads), 3,860 ms
  (WebGPU) and 4,130 ms (WASM 1 thread). These are estimates, not measurements.
* The linear fits of ms/icon against calls are `-620 + 4.71 x calls` (WASM 4 threads),
  `25 + 4.95 x calls` (WebGPU) and `-795 + 6.18 x calls` (WASM 1 thread). The negative WASM
  intercepts are another sign that per-call cost grows with the cache.

Comparison with milestone 0: v9 replayed with Python masks on WASM 4 threads took 881 ms
per icon. That was on the first 32 validation icons, which need a median of 350 calls,
and the median per-call time, taken over all calls, was 2.65 ms. This harness records call
time per icon, not per call; the median over icons of each icon's mean per-call time is
2.78 ms. The two statistics are close but not the same, and per-call time excludes the
grammar by construction: the evidence that the free-running grammar adds almost nothing is
its measured share (0.22%), not this comparison. Per-icon times are higher because these
icons are heavier (419 calls against 350) and because heavy icons cost more per call.

## Conditions

* **Machine.** gpubox-4080: RTX 4080 (driver 595.71.05, 16,376 MiB), i7-13700K (24 logical
  CPUs).
* **Software.** Headless Chrome for Testing 153.0.8010.12 through Playwright 1.63.0, with
  milestone 0's WebGPU-on-Vulkan flags (`web/e2e/gpubox-env.sh`); one fresh browser per
  config. ONNX Runtime Web 1.30.0, served from localhost by `web/e2e/server.mjs` with
  COOP/COEP: `crossOriginIsolated` was true and the effective WASM threads were 4 (or 1, as
  requested). Node v22.23.2 with onnxruntime-node 1.30.0. The WebGPU adapter was asserted to
  be NVIDIA Lovelace and not a fallback. The WASM 1 thread row was measured with COOP/COEP
  and 1 thread requested; the `NOCOI=1` row, added by the review (load average 0.9 before,
  GPU idle), serves the page without those headers as GitHub Pages would: ORT asked for 4
  threads, got 1, and timed the same as the 1-thread row.
* **GPU before timing.** nvidia-smi before every config showed 0% utilisation, 30 MiB used
  and no compute processes. The exceptions were 35% before WASM 1 thread (left over from
  the WebGPU config that had just finished) and 5% before the repeated WebGPU run.
* **GPU during WebGPU.** Mean utilisation 35.5%, max 43%, at most 292 MiB.
* **CPU load.** The load average was 1.1 to 3.6 before each config. No other heavy process
  was running; other sessions on the box were idle.
* **Icons.** 32 validation icons spread evenly over the 339 (`web_e2e.spread_indices`:
  indices 0, 11, 22, …, 338). They need a median of 419 calls (p90 704, range 113 to 938),
  against 425 (p90 782) over all 339 icons. The first 16 icons are light, at 260. Inputs are the
  cached corpus renders (`load_corpus`, 144 px); the browser reads the same bytes from
  `images.u8`.
* **Wall time.** Python reference 23 s, Node check 31 s, the three browser configs 3 min
  10 s, and the repeat 2 min.

## Reproduce

From the repository root of a checkout with `data/raw`:

    RUN=r2s-cur-d02-s120k-seed2-53b3108-46294452-47646604
    .venv/bin/python -m mojidiff.learning.onnx_export --run $RUN     # only if /home/dev/.cache/mojidiff/onnx/$RUN is missing
    .venv/bin/python -m mojidiff.learning.web_e2e --run $RUN --icons 32   # Python ORT reference bundle
    cd web && npm ci --ignore-scripts
    M=/home/dev/.cache/mojidiff/onnx/$RUN
    R=/home/dev/.cache/mojidiff/web-reference/$RUN/validation-32
    node e2e/node.mjs --model $M --reference $R --threads 4 --json ../reports/browser/m1/results/node-cpu-4threads.json
    nvidia-smi                       # record the GPU state first
    source e2e/gpubox-env.sh         # gpubox-4080 only: Chrome libraries, browsers, NVIDIA Vulkan ICD
    node e2e/browser.mjs --model $M --reference $R --out ../reports/browser/m1/results --configs wasm4,webgpu,wasm1
    node e2e/browser.mjs --model $M --reference $R --out ../reports/browser/m1/results/repeat --configs wasm4,webgpu
    NOCOI=1 node e2e/browser.mjs --model $M --reference $R --out ../reports/browser/m1/results/nocoi --configs wasm4

Both scripts exit 1 if any icon differs. The tests are `pytest tests/test_web_e2e.py`, which
covers a tiny export and, from the cache, 16 spread icons of the final model in Node.
`MOJIDIFF_BROWSER_TEST=1` after sourcing `gpubox-env.sh` adds the headless-Chrome test.

## Caveats

* **Speed.** This is not fast: about 1.2 s per icon at the median and 2.7 s at p90 over
  these 32 icons (about 3.2 s estimated over all 339), on a desktop CPU (i7-13700K).
  Visitors' machines will be slower. Greedy only; best of 8 was not measured.
* **WASM threads need cross-origin isolation.** GitHub Pages cannot send COOP/COEP, so a
  Pages demo gets the WASM 1 thread row (1,519 ms; 1,506 ms measured with `NOCOI=1`) unless
  a coi-serviceworker shim works. That shim is still untested.
* **Why WASM slows down on long icons.** On WASM, variant B's growing cache crosses the JS
  and WASM heap boundary in both directions on every call, keys and values of up to
  8.5 MB each at the full 1,376 steps, and attention covers the whole cache. That is the likely reason per-call
  cost grows with the cache. WebGPU keeps the cache on the GPU but pays about 5 ms of
  dispatch per call. The calls are where any speed-up must come from: fewer calls, graph
  capture (milestone 0: about 1 ms per call less on WebGPU), smaller weights, or a cache
  that stays inside the WASM heap.
* **Not a registered run.** These are measurements of an existing checkpoint, not a run
  under `runs/`; this task did not touch `runs/` or `state/`.

## Files

* `results/wasm4.json`, `results/webgpu.json`, `results/wasm1.json`, `results/repeat/` and
  `results/nocoi/` (the review's no-COOP/COEP run):
  per config, per icon (index, hexcode, calls, forced steps, ms split, equality flags, first
  differing position), the cold icon, the adapter, threads, the Chrome version, the console,
  and nvidia-smi before, during and after.
* `results/summary.json`: one line per config plus the software versions and Chrome flags.
* `results/node-cpu-4threads.json`: the Node check, per icon, including the torch-fixture
  comparison.
* Code: `src/mojidiff/learning/web_e2e.py` (reference bundle), `web/src/ort-model.mjs` (ONNX
  Runtime step adapter), `web/e2e/` (Node check, page, server, Playwright runner), and
  `tests/test_web_e2e.py`.
