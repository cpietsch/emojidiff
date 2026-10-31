// Free-running decode of the reference icons in the browser, with ONNX Runtime Web: the JS
// grammar picks every legal mask, the greedy loop feeds back the model's own tokens, and the
// codec writes the SVG. Nothing is replayed from Python; the reference is only compared.
//
// Query: ep=wasm|webgpu  threads=N (WASM threads)  icons=N (limit)  bundle=<onnxruntime-web dist file>
// Served by server.mjs (/web, /ort, /model, /reference), driven by browser.mjs, which reads
// window.__out once document.title is "done".
//
// Per config: sessions are created, icon 0 runs once cold (shader compilation, allocations;
// reported apart), then every icon runs warm. ms_icon is pixels (Uint8Array) in to SVG text
// out, including the encoder, every decoder call, the grammar and the release of the cache;
// model download and session creation are not in it (reported as session_ms).
import { tokensToSvg } from "../src/codec.mjs";
import { greedyDecode } from "../src/decode.mjs";
import { createGrammar } from "../src/grammar.mjs";
import { ortModel, sessionOptions } from "../src/ort-model.mjs";
import { checkBundle, compareIcon } from "./check.mjs";

const q = new URLSearchParams(location.search);
const backend = q.get("ep") || "wasm";
const threads = Number(q.get("threads") || (backend === "wasm" ? 4 : 1));
const limit = Number(q.get("icons") || Infinity);
const bundle = q.get("bundle") || (backend === "webgpu" ? "ort.all.min.mjs" : "ort.wasm.min.mjs");
const now = () => performance.now();
const log = text => { document.getElementById("log").textContent = text; };
const out = {
  backend, threads_requested: threads, bundle, crossOriginIsolated: self.crossOriginIsolated,
  userAgent: navigator.userAgent, hardwareConcurrency: navigator.hardwareConcurrency,
};

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

try {
  if (navigator.gpu) {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    out.adapter = adapter ? {
      vendor: adapter.info.vendor, architecture: adapter.info.architecture,
      description: adapter.info.description, isFallbackAdapter: adapter.info.isFallbackAdapter,
    } : null;
  }
  if (backend === "webgpu" && !(out.adapter && out.adapter.vendor === "nvidia" && !out.adapter.isFallbackAdapter)) {
    throw new Error("WebGPU adapter is not a non-fallback NVIDIA adapter: " + JSON.stringify(out.adapter));
  }
  const [codec, manifest, reference, images] = await Promise.all([
    json("/web/src/codec.json"), json("/model/manifest.json"), json("/reference/reference.json"),
    fetch("/reference/images.u8").then(r => r.arrayBuffer()).then(b => new Uint8Array(b)),
  ]);
  checkBundle(manifest, reference);
  const constants = manifest.constants;
  if (codec.length !== constants.length || codec.vocabulary !== constants.vocabulary) {
    throw new Error("codec.json and the exported model disagree on the sequence layout");
  }
  const grammar = createGrammar(codec);
  const size = constants.image_size, pixels = size * size * 3;
  if (images.length !== reference.icons.length * pixels) throw new Error("images.u8 does not match reference.json");
  out.run_id = reference.run_id;

  const importStart = now();
  const ort = await import(`/ort/${bundle}`);
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = threads;
  out.import_ms = now() - importStart;
  const options = sessionOptions(backend);
  const sessionStart = now();
  const encoder = await ort.InferenceSession.create("/model/encoder.onnx", options.encoder);
  const decoder = await ort.InferenceSession.create("/model/decoder_step.onnx", options.decoder);
  out.session_ms = now() - sessionStart;
  out.numThreads_effective = ort.env.wasm.numThreads;
  out.ort_version = ort.env.versions?.web ?? ort.env.versions?.common ?? null;
  const model = ortModel(ort, { encoder, decoder, constants });

  async function runIcon(k) {
    const ref = reference.icons[k];
    const image = images.subarray(k * pixels, (k + 1) * pixels);
    const t0 = now();
    const r = await greedyDecode({ grammar, encode: model.encode, step: model.step, image });
    const t1 = now();
    const svg = tokensToSvg(r.tokens, codec);
    const t2 = now();
    model.release(r.past, r.memory);
    return {
      index: ref.index, hexcode: ref.hexcode, calls: r.calls, forced: r.forced,
      ms_icon: now() - t0, ms_encode: r.ms.encode, ms_step: r.ms.step, ms_grammar: r.ms.grammar, ms_svg: t2 - t1,
      ...compareIcon(r.tokens, svg, r.calls, ref),
    };
  }

  out.cold = await runIcon(0);
  const count = Math.min(limit, reference.icons.length);
  out.icons = [];
  for (let k = 0; k < count; k++) {
    out.icons.push(await runIcon(k));
    log(`icon ${k + 1}/${count}`);
  }
} catch (error) {
  out.error = String((error && error.stack) || error);
}
window.__out = out;
log(JSON.stringify(out, null, 1).slice(0, 4000));
document.title = "done";
