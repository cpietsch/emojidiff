// The demo page: the final model decodes validation renders (or the viewer's own image) in
// this tab, free-running, as web/e2e/page.mjs does for the parity check. Served by
// demo/serve.mjs (/web, /ort, /model, /reference); the reference bundle is
// `python -m mojidiff.learning.web_e2e --icons 339`, whose Python decodes the ticks compare to.
//
// Query: ep=wasm|webgpu (default wasm)  threads=N (WASM threads, default 4; 1 without isolation)
//
// On WASM each ONNX Runtime call runs synchronously on this thread, so the page yields to the
// browser (a frame, then a task) before every icon, outside the timed window: cards appear one
// by one, Stop is seen between icons, and clicks made during an icon land while the buttons are
// still disabled. The stats tiles count validation icons only; the session's first decode is
// warm-up and stays out of the latency tiles.
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
const showError = text => { $("error").textContent = text; $("error").hidden = !text; };
const buttons = ["next", "eight", "all"];

let model, grammar, codec, reference, images, size, pixels;
let queue = [], busy = false, stopping = false, loaded = false;
const done = new Set(), rows = [];

async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: ${response.status}`);
  return response.json();
}

/** A task boundary that queues behind waiting input and timers. Not scheduler.yield: Chrome
 * runs its continuation ahead of other tasks, so a batch starved clicks and timers anyway. */
function yieldTask() {
  return new Promise(resolve => setTimeout(resolve, 0));
}

/** Let the browser paint what changed (a frame, then a task after it), then go on. */
async function yieldToBrowser() {
  if (!document.hidden) await new Promise(resolve => requestAnimationFrame(() => resolve()));
  await yieldTask();
}

function spreadOrder(n) {
  // Halves, then quarters, and so on, so any prefix covers the split; every index once.
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
  const url = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
  const img = document.createElement("img");
  img.alt = `SVG written by the model for ${title}`;
  img.src = url;
  element.querySelector(".pending").replaceWith(img);
  element.querySelector(".ms").textContent = `${Math.round(result.ms)} ms · ${result.calls} calls`;
  const link = element.querySelector(".dl");
  link.href = url; link.download = `${title}.svg`; link.hidden = false;
  link.textContent = `SVG (${(svg.length / 1024).toFixed(1)} kB)`;
}

function showFailure(element, error) {
  const pending = element.querySelector(".pending");
  if (pending) { pending.textContent = "failed"; pending.classList.add("no"); }
  const mark = element.querySelector(".check");
  mark.className = "check no";
  mark.textContent = String((error && error.message) || error);
}

function updateStats() {
  const validation = rows.filter(r => r.kind === "validation");
  const warm = validation.filter(r => !r.cold);
  $("stats").hidden = validation.length === 0;
  $("s-n").textContent = validation.length;
  const cold = validation.find(r => r.cold);
  $("s-cold").textContent = cold ? `first icon ${Math.round(cold.ms).toLocaleString()} ms with warm-up, not counted` : "";
  $("s-med").textContent = warm.length ? Math.round(quantile(warm.map(r => r.ms), 0.5)).toLocaleString() : "–";
  $("s-p90").textContent = warm.length >= 5 ? Math.round(quantile(warm.map(r => r.ms), 0.9)).toLocaleString() : "–";
  $("s-warm").textContent = warm.length ? `over ${warm.length} icon${warm.length === 1 ? "" : "s"}` : "after warm-up";
  $("s-calls").textContent = validation.length ? Math.round(quantile(validation.map(r => r.calls), 0.5)) : "–";
  $("s-same").textContent = validation.length ? `${validation.filter(r => r.identical).length} / ${validation.length}` : "–";
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
  const element = card(ref.hexcode);
  drawRender(element.querySelector("canvas"), image);
  await yieldToBrowser();
  let result;
  try {
    result = await decode(image);
  } catch (error) {
    showFailure(element, error);
    throw error;
  }
  const check = compareIcon(result.tokens, result.svg, result.calls, ref);
  showResult(element, result.svg, result, ref.hexcode);
  const mark = element.querySelector(".check");
  mark.className = "check " + (check.identical ? "ok" : "no");
  mark.textContent = check.identical ? "✓ tokens equal Python's" : `✗ differs from Python at position ${check.first_difference}`;
  rows.push({ kind: "validation", ms: result.ms, calls: result.calls, identical: check.identical, cold: rows.length === 0 });
  done.add(k);
  updateStats();
}

/** The viewer's file as the encoder's input: composited on white, fitted into size x size. */
async function fileToRender(file) {
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.src = url;
    await img.decode();
    const canvas = document.createElement("canvas");
    canvas.width = size; canvas.height = size;
    const context = canvas.getContext("2d");
    context.fillStyle = "#fff";
    context.fillRect(0, 0, size, size);
    context.imageSmoothingQuality = "high";
    // An SVG with only a viewBox may report no natural size: draw it at the full canvas.
    const width = img.naturalWidth || size, height = img.naturalHeight || size;
    const scale = Math.min(size / width, size / height);
    const w = width * scale, h = height * scale;
    context.drawImage(img, (size - w) / 2, (size - h) / 2, w, h);
    const rgba = context.getImageData(0, 0, size, size).data;
    const image = new Uint8Array(pixels);
    for (let p = 0, o = 0; p < pixels; p += 3, o += 4) {
      image[p] = rgba[o]; image[p + 1] = rgba[o + 1]; image[p + 2] = rgba[o + 2];
    }
    return image;
  } finally {
    URL.revokeObjectURL(url);
  }
}

async function runFile(file) {
  const title = file.name.replace(/\.[^.]+$/, "") || "image";
  const element = card(title);
  try {
    const image = await fileToRender(file);
    drawRender(element.querySelector("canvas"), image);
    await yieldToBrowser();
    const result = await decode(image);
    showResult(element, result.svg, result, title);
    element.querySelector(".check").textContent = "your image (no reference)";
    rows.push({ kind: "upload", ms: result.ms, calls: result.calls, identical: null, cold: rows.length === 0 });
    showError("");
  } catch (error) {
    showFailure(element, error);
    showError(`Could not convert ${file.name}: ${String((error && error.message) || error)}`);
    console.error(error);
  }
}

function setBusy(value) {
  busy = value;
  for (const id of buttons) $(id).disabled = !loaded || value || queue.length === 0;
  $("stop").disabled = !value;
  $("stop").textContent = "Stop";
  $("file").disabled = !loaded || value;
}

async function finish() {
  // Input queued during the last icon is dispatched now, while the buttons are still disabled.
  await yieldTask();
  setBusy(false);
}

async function run(count) {
  if (busy || !loaded) return;
  setBusy(true);
  stopping = false;
  try {
    for (let n = 0; n < count && queue.length && !stopping;) {
      const k = queue.shift();
      if (done.has(k)) continue;
      await runIndex(k);
      n++;
      showError("");
    }
  } catch (error) {
    showError(`Decoding failed: ${String((error && error.message) || error)}`);
    console.error(error);
  }
  await finish();
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
  status("Loading the validation icons (24 MB)…");
  let manifest;
  [codec, manifest, reference] = await Promise.all([
    json("/web/src/codec.json"), json("/model/manifest.json"), json("/reference/reference.json"),
  ]);
  const response = await fetch("/reference/images.u8");
  if (!response.ok) throw new Error(`/reference/images.u8: ${response.status}`);
  images = new Uint8Array(await response.arrayBuffer());
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
  $("stop").addEventListener("click", () => {
    stopping = true;
    $("stop").disabled = true;
    $("stop").textContent = "Stopping after this icon…";
  });
  $("order").addEventListener("change", refill);
  $("file").addEventListener("change", async event => {
    const file = event.target.files[0];
    event.target.value = "";
    if (!file || busy) return;
    setBusy(true);
    await runFile(file);
    await finish();
  });
  loaded = true;
  refill();
  await run(1);
}

main().catch(error => {
  showError(String((error && error.message) || error));
  status(`<span class="warn">The demo could not start.</span>`);
  console.error(error);
});
