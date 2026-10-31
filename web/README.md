# web/: the render-to-SVG decoder in the browser

ES modules with no dependencies, so the same code runs in a page, a Web Worker and Node:

| file | what | Python reference |
| --- | --- | --- |
| `src/codec.json` | codec and layout constants | `scripts/export_web_fixtures.py codec` |
| `src/codec.mjs` | tokens to SVG text, byte for byte (moved from milestone 0) | `packed.serialize_packed_svg` |
| `src/grammar.mjs` | legal masks, path-major order, per-step labels, variant-B ids | `autoregressive.legal_mask`, `render2svg.path_major_positions`, `fast_decode.GraphDecoder._role/_labels` |
| `src/decode.mjs` | greedy decode loop over an abstract model step | `render2svg.greedy_decode` (batch 1) |
| `src/ort-model.mjs` | the exported model as that step, on ONNX Runtime (Web or Node; the caller passes the module) | `onnx_decode.OnnxDecoder` |

## The step contract

`greedyDecode({grammar, encode, step, image, maxChunk})` calls `encode(image)` once, then
`step(ids, target, past, memory)` once per step whose legal mask leaves a choice. Steps with
exactly one legal token are not predicted: they are queued and fed with the next call, as
`greedy_decode` does.

- `ids`: `Float32Array`, the variant-B `(6, 1, L)` rows `tokens, steps, field, path, axis, role`.
  Float32, not int64: with int64 ids, 24% of calls on WebGPU (JSEP) chose a different token than torch in milestone 0.
- `target`: `Float32Array(1)`, the metric target of the chunk's last step (`(role-1)*2+axis`, or -1).
- `past`: `null` on the first call, then the previous call's `present`, handed over once.
- returns `{logits, present}`, with `logits` the `(1, 418)` logits of the chunk's last step.

`greedyDecode` returns the last `present` and the encoder's `memory` as `past` and `memory`;
`ortModel(...).release(past, memory)` frees them (GPU buffers on WebGPU).

## Tests

    cd web && npm test        # or: node --test test/*.test.mjs   (Node 20 or newer)

`tests/test_web.py::test_node_suite_passes` runs the same suite under `pytest`. It is skipped
when no Node 20 or newer is found (set `MOJIDIFF_NODE` to choose a binary).

The fixtures in `test/fixtures/` come from the Python code (`src/mojidiff/learning/web_reference.py`):

- `programs.json.gz` and `tokens.bin.gz`: every validation program of the corpus (339), the
  final model's greedy decode of each validation icon (339), and 32 samples at temperature 1.
  For each program the fixtures hold its tokens, its batch-1 call count, and 128-bit SHA-256
  digests of (a) its SVG and (b) its decode walk. The walk is a 65-byte record per step:
  position, legal mask, forced, ids, target, token. The Node tests rebuild both digests for
  every program at every position.
- `walks.json.gz`: the full walk of a subset, and each model decode's recorded
  `greedy_decode` call schedule, so a mismatch names the step, position and field.
- Both headers record the splits and the set's counts (`web_reference.fixture_counts`); the
  tests check the set against those counts, not against numbers written into the tests.

Datasets stay out of the repository, so the train and test programs are not committed. The
full set (every corpus split: 3,359 corpus programs plus the same decodes) lives under
`/home/dev/.cache/mojidiff/web-fixtures/<run_id>/full/`, and `MOJIDIFF_WEB_FIXTURES` points
the suite at it (a value that names no directory is an error, not a fallback):

    MOJIDIFF_WEB_FIXTURES=/home/dev/.cache/mojidiff/web-fixtures/<run_id>/full npm test

`tests/test_web.py` runs the suite on both sets; the full-set tests skip when it is missing.

Regenerate both after any change to the grammar, labels, codec or model (the script refuses
to write train or test programs into the repository):

    cd /home/dev/workspace/mojidiff   # reads data/raw and the cached corpus by relative path
    python scripts/export_web_fixtures.py codec
    python scripts/export_web_fixtures.py fixtures --checkpoint /home/dev/.cache/mojidiff/runs/<run_id>/best.pt
    python scripts/export_web_fixtures.py fixtures --checkpoint /home/dev/.cache/mojidiff/runs/<run_id>/best.pt \
        --splits all --out /home/dev/.cache/mojidiff/web-fixtures/<run_id>/full

## End to end with the exported model (`e2e/`)

The development tools pinned in `package.json` (onnxruntime-node, onnxruntime-web 1.30.0,
playwright 1.63.0) are only for these checks; `src/` needs none of them. Install without
install scripts (onnxruntime-node's would download CUDA providers the checks do not use):

    cd web && npm ci --ignore-scripts      # or point MOJIDIFF_NODE_MODULES at a node_modules

Inputs: an export (`python -m mojidiff.learning.onnx_export --run <run_id>`, written to
`/home/dev/.cache/mojidiff/onnx/<run_id>/`) and a reference bundle of validation icons
spread over the split, decoded by the Python ONNX Runtime decoder
(`python -m mojidiff.learning.web_e2e --run <run_id> --icons 32`, written to
`/home/dev/.cache/mojidiff/web-reference/<run_id>/validation-32/`). Both run from the repo root.

- `e2e/node.mjs --model DIR --reference DIR`: decodes every icon with the JS grammar and loop
  on onnxruntime-node and checks all tokens, the call count and the JS codec's SVG bytes
  against the bundle (and against torch `greedy_decode` in `test/fixtures` when the
  checkpoint matches). Exits 1 on any difference.
- `e2e/browser.mjs --model DIR --reference DIR --out DIR --configs wasm4,webgpu`: the same,
  free-running in headless Chrome (`e2e/page.html`, served by `e2e/server.mjs` with
  COOP/COEP), one fresh browser per config, with ms per icon, decoder calls, the grammar's
  share of the time and nvidia-smi before, during and after. On gpubox-4080, `source
  e2e/gpubox-env.sh` first (Chrome's libraries, browsers and the NVIDIA Vulkan ICD).

`tests/test_web_e2e.py` runs `node.mjs` on a tiny random export and, when the final export is
in the cache, on 16 spread validation icons of the final model; both skip without Node 20 or
the packages. The headless-Chrome test runs only with `MOJIDIFF_BROWSER_TEST=1`. Measured
results: `reports/browser/m1/`.
