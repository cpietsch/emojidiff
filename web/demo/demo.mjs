// The demo page: the final model decodes validation renders (or the viewer's own image) in
// this tab, free-running, as web/e2e/page.mjs does for the parity check. Served by
// demo/serve.mjs (/web, /ort, /model, /reference); the reference bundle is
// `python -m mojidiff.learning.web_e2e --icons 339`, whose Python decodes the ticks compare to.
//
// Query: ep=wasm|webgpu (default wasm)  threads=N (WASM threads, default 4; 1 without isolation)
import { tokensToSvg } from "../src/codec.mjs";
import { greedyDecode } from "../src/decode.mjs";
import { createGrammar } from "../src/grammar.mjs";
import { ortModel, sessionOptions } from "../src/ort-model.mjs";
import { checkBundle, compareIcon, quantile } from "../e2e/check.mjs";

const q = new URLSearchParams(location.search);
const backend = q.get("ep") === "webgpu" ? "webgpu" : "wasm";
const threads = Number(q.get("threads") || 4);
const $ = id => document.getElementById(id);
const now = () => performance.now();
const status = html => { $("status").innerHTML = html; };
const buttons = ["next", "eight", "all"];

let model, grammar, codec, reference, images, size, pixels;
let queue = [], done = new Set(), busy = false, stopping = false;
const rows = [];

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

function spreadOrder(n) {
  // Bit-reversal-like spread: halves, then quarters, so any prefix covers the split.
  const order = [], seen = new Set();
  for (let step = n; step >= 1; step = Math.floor(step / 2)) {
    for (let i = 0; i < n; i += step) if (!seen.has(i)) { seen.add(i); order.push(i); }
    if (step === 1) break;
  }
  return order;
}

function shuffled(n) {
  const order = Array.from({ length: n }, (_, i) => i);
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  return order;
}

function drawRender(canvas, rgb) {
  canvas.width = size; canvas.height = size;
  const context = canvas.getContext("2d");
  const data = context.createImageData(size, size);
  for (let p = 0, o = 0; p < rgb.length; p += 3, o += 4) {
    data.data[o] = rgb[p]; data.data[o + 1] = rgb[p + 1]; data.data[o + 2] = rgb[p + 2]; data.data[o + 3] = 255;
  }
  context.putImageData(data, 0, 0);
}

function card(title) {
  const element = document.createElement("article");
  element.className = "card";
  element.innerHTML = `
    <div class="pair">
      <figure><canvas></canvas><figcaption>input render</figcaption></figure>
      <figure><div class="pending">decoding…</div><figcaption>model's SVG</figcaption></figure>
    </div>
    <div class="meta"><span class="name"></span><span class="ms"></span></div>
    <div class="meta"><span class="check"></span><a class="dl" hidden>SVG</a></div>`;
  element.querySelector(".name").textContent = title;
  $("grid").prepend(element);
  return element;
}

function showResult(element, svg, result, title) {
  const blob = new Blob([svg], { type: "image/svg+xml" });
  const url = URL.createObjectURL(blob);
  const img = document.createElement("img");
  img.alt = `SVG written by the model for ${title}`;
  img.src = url;
  element.querySelector(".pending").replaceWith(img);
  element.querySelector(".ms").textContent = `${Math.round(result.ms)} ms · ${result.calls} calls`;
  const link = element.querySelector(".dl");
  link.href = url; link.download = `${title}.svg`; link.hidden = false;
  link.textContent = `SVG (${(svg.length / 1024).toFixed(1)} kB)`;
}

function updateStats() {
  const measured = rows.filter(r => !r.cold);
  const pool = measured.length ? measured : rows;
  $("stats").hidden = rows.length === 0;
  $("s-n").textContent = rows.length;
  if (!pool.length) return;
  const ms = pool.map(r => r.ms), calls = pool.map(r => r.calls);
  $("s-med").textContent = Math.round(quantile(ms, 0.5)).toLocaleString();
  $("s-p90").textContent = pool.length >= 5 ? Math.round(quantile(ms, 0.9)).toLocaleString() : "–";
  $("s-calls").textContent = Math.round(quantile(calls, 0.5));
  const checked = rows.filter(r => r.identical !== null);
  $("s-same").textContent = checked.length ? `${checked.filter(r => r.identical).length} / ${checked.length}` : "–";
}

async function decode(image) {
  const t0 = now();
  const r = await greedyDecode({ grammar, encode: model.encode, step: model.step, image });
  const svg = tokensToSvg(r.tokens, codec);
  model.release(r.past, r.memory);
  return { tokens: r.tokens, calls: r.calls, svg, ms: now() - t0 };
}

async function runIndex(k) {
  const ref = reference.icons[k];
  const image = images.subarray(k * pixels, (k + 1) * pixels);
  const title = ref.hexcode;
  const element = card(title);
  drawRender(element.querySelector("canvas"), image);
  const result = await decode(image);
  const check = compareIcon(result.tokens, result.svg, result.calls, ref);
  showResult(element, result.svg, result, title);
  const mark = element.querySelector(".check");
  mark.className = "check " + (check.identical ? "ok" : "no");
  mark.textContent = check.identical ? "✓ tokens equal Python's" : `✗ differs from Python at position ${check.first_difference}`;
  rows.push({ ms: result.ms, calls: result.calls, identical: check.identical, cold: rows.length === 0 });
  done.add(k);
  updateStats();
}

async function runFile(file) {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = size; canvas.height = size;
  const context = canvas.getContext("2d");
  context.fillStyle = "#fff";
  context.fillRect(0, 0, size, size);
  const scale = Math.min(size / bitmap.width, size / bitmap.height);
  const w = bitmap.width * scale, h = bitmap.height * scale;
  context.drawImage(bitmap, (size - w) / 2, (size - h) / 2, w, h);
  const rgba = context.getImageData(0, 0, size, size).data;
  const image = new Uint8Array(pixels);
  for (let p = 0, o = 0; p < pixels; p += 3, o += 4) {
    image[p] = rgba[o]; image[p + 1] = rgba[o + 1]; image[p + 2] = rgba[o + 2];
  }
  const title = file.name.replace(/\.[^.]+$/, "") || "image";
  const element = card(title);
  drawRender(element.querySelector("canvas"), image);
  const result = await decode(image);
  showResult(element, result.svg, result, title);
  element.querySelector(".check").textContent = "your image (no reference)";
  rows.push({ ms: result.ms, calls: result.calls, identical: null, cold: rows.length === 0 });
  updateStats();
}

function setBusy(value) {
  busy = value;
  for (const id of buttons) $(id).disabled = value || queue.length === 0;
  $("stop").disabled = !value;
  $("file").disabled = value;
}

async function run(count) {
  if (busy) return;
  setBusy(true);
  stopping = false;
  try {
    for (let n = 0; n < count && queue.length && !stopping; n++) {
      const k = queue.shift();
      if (done.has(k)) { n--; continue; }
      await runIndex(k);
    }
  } catch (error) {
    status(`<span class="warn">Decoding failed: ${String(error.message || error)}</span>`);
    console.error(error);
  }
  setBusy(false);
}

function refill() {
  const order = $("order").value === "random" ? shuffled(reference.icons.length) : spreadOrder(reference.icons.length);
  queue = order.filter(k => !done.has(k));
  setBusy(busy);
}

async function main() {
  $("backend").value = backend;
  $("backend").addEventListener("change", () => {
    const next = new URLSearchParams(location.search);
    next.set("ep", $("backend").value);
    location.search = next.toString();
  });
  if (!navigator.gpu) $("backend").querySelector('option[value="webgpu"]').disabled = true;
  if (backend === "webgpu" && !navigator.gpu) throw new Error("WebGPU is not available here (it needs a secure context: https or localhost)");

  const loadStart = now();
  let manifest;
  [codec, manifest, reference] = await Promise.all([
    json("/web/src/codec.json"), json("/model/manifest.json"), json("/reference/reference.json"),
  ]);
  images = new Uint8Array(await (await fetch("/reference/images.u8")).arrayBuffer());
  checkBundle(manifest, reference);
  const constants = manifest.constants;
  if (codec.length !== constants.length || codec.vocabulary !== constants.vocabulary) {
    throw new Error("codec.json and the exported model disagree on the sequence layout");
  }
  grammar = createGrammar(codec);
  size = constants.image_size;
  pixels = size * size * 3;
  if (images.length !== reference.icons.length * pixels) throw new Error("images.u8 does not match reference.json");

  status("Loading ONNX Runtime and the model (38 MB)…");
  const ort = await import(backend === "webgpu" ? "/ort/ort.all.min.mjs" : "/ort/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = "/ort/";
  ort.env.wasm.numThreads = self.crossOriginIsolated ? threads : 1;
  const options = sessionOptions(backend);
  const sessionStart = now();
  const encoder = await ort.InferenceSession.create("/model/encoder.onnx", options.encoder);
  const decoder = await ort.InferenceSession.create("/model/decoder_step.onnx", options.decoder);
  const sessionMs = now() - sessionStart;
  model = ortModel(ort, { encoder, decoder, constants });

  const isolation = self.crossOriginIsolated
    ? `cross-origin isolated, <b>${ort.env.wasm.numThreads}</b> WASM threads`
    : `<span class="warn">not cross-origin isolated (plain http): 1 WASM thread, no WebGPU</span>`;
  status(`Backend <b>${backend.toUpperCase()}</b> · ${isolation} · model <b>${reference.run_id}</b> ·
    loaded in ${Math.round(now() - loadStart).toLocaleString()} ms (sessions ${Math.round(sessionMs)} ms) ·
    ${reference.icons.length} validation icons`);

  $("next").addEventListener("click", () => run(1));
  $("eight").addEventListener("click", () => run(8));
  $("all").addEventListener("click", () => run(Infinity));
  $("stop").addEventListener("click", () => { stopping = true; });
  $("order").addEventListener("change", refill);
  $("file").addEventListener("change", async event => {
    const file = event.target.files[0];
    event.target.value = "";
    if (!file || busy) return;
    setBusy(true);
    try { await runFile(file); } catch (error) { status(`<span class="warn">${String(error.message || error)}</span>`); }
    setBusy(false);
  });
  refill();
  await run(1);
}

main().catch(error => {
  status(`<span class="warn">${String(error.message || error)}</span>`);
  console.error(error);
});
