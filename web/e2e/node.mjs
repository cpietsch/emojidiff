// End to end in Node: the JavaScript grammar and greedy loop (web/src) drive the exported
// model on onnxruntime-node, and each icon must come out exactly as the Python ONNX Runtime
// decoder wrote it in the reference bundle: all tokens, the decoder-call count, and through
// the JavaScript codec the SVG bytes of packed.serialize_packed_svg.
//
//   node e2e/node.mjs --model <export dir> --reference <bundle dir> [--threads 4] [--icons N] [--json out.json]
//
// <export dir> is written by `python -m mojidiff.learning.onnx_export`, <bundle dir> by
// `python -m mojidiff.learning.web_e2e`. When the fixtures hold torch greedy_decode programs
// of the same checkpoint (web/test/fixtures, or MOJIDIFF_WEB_FIXTURES), the decodes are
// checked against those too (--fixtures off skips that). Prints a JSON summary with per-icon rows; exits 1 if any icon differs. The times are
// onnxruntime-node on the CPU, not a browser number.
import { readFileSync, writeFileSync } from "node:fs";
import os from "node:os";
import path from "node:path";
import { parseArgs } from "node:util";
import { tokensToSvg } from "../src/codec.mjs";
import { greedyDecode } from "../src/decode.mjs";
import { createGrammar } from "../src/grammar.mjs";
import { ortModel, sessionOptions } from "../src/ort-model.mjs";
import { checkBundle, compareIcon, summarize } from "./check.mjs";
import { packageVersion, requirePackage, WEB } from "./modules.mjs";

const { values: args } = parseArgs({
  options: {
    model: { type: "string" }, reference: { type: "string" }, threads: { type: "string", default: "4" },
    icons: { type: "string" }, json: { type: "string" }, fixtures: { type: "string", default: "auto" },
  },
});
if (!args.model || !args.reference) {
  console.error("usage: node e2e/node.mjs --model DIR --reference DIR [--threads N] [--icons N] [--json FILE]");
  process.exit(2);
}

const ort = requirePackage("onnxruntime-node");
const readJson = file => JSON.parse(readFileSync(file, "utf8"));
const manifest = readJson(path.join(args.model, "manifest.json"));
const reference = readJson(path.join(args.reference, "reference.json"));
const images = readFileSync(path.join(args.reference, reference.images.file));
const codec = readJson(path.join(WEB, "src", "codec.json"));
checkBundle(manifest, reference);
const constants = manifest.constants;
if (codec.length !== constants.length || codec.vocabulary !== constants.vocabulary) {
  throw new Error("codec.json and the exported model disagree on the sequence layout");
}
const grammar = createGrammar(codec);
const size = constants.image_size, pixels = size * size * 3;
if (images.length !== reference.icons.length * pixels) throw new Error("images.u8 does not match reference.json");

// torch greedy_decode programs of the same checkpoint, when the committed fixtures have them.
const torchTokens = new Map();
let fixtureTokens = null;
if (args.fixtures !== "off") {
  const { fixtures } = await import("../test/helpers.mjs");
  const { meta, tokens } = fixtures();
  if (meta.checkpoint_sha256 === manifest.checkpoint?.sha256) {
    meta.programs.forEach((p, i) => { if (p.source === "greedy") torchTokens.set(p.hexcode, i); });
    fixtureTokens = tokens;
  }
}

const loadBefore = os.loadavg();
const threads = Number(args.threads);
const options = sessionOptions("cpu", { intraOpNumThreads: threads, interOpNumThreads: 1 });
const started = performance.now();
const encoder = await ort.InferenceSession.create(path.join(args.model, "encoder.onnx"), options.encoder);
const decoder = await ort.InferenceSession.create(path.join(args.model, "decoder_step.onnx"), options.decoder);
const sessionMs = performance.now() - started;
const model = ortModel(ort, { encoder, decoder, constants });

const count = Math.min(Number(args.icons ?? Infinity), reference.icons.length);
const rows = [];
for (let k = 0; k < count; k++) {
  const ref = reference.icons[k];
  const t0 = performance.now();
  const out = await greedyDecode({ grammar, encode: model.encode, step: model.step, image: images.subarray(k * pixels, (k + 1) * pixels) });
  const t1 = performance.now();
  const svg = tokensToSvg(out.tokens, codec);
  const t2 = performance.now();
  model.release(out.past, out.memory);
  const row = {
    index: ref.index, hexcode: ref.hexcode, calls: out.calls, forced: out.forced,
    ms_icon: performance.now() - t0, ms_encode: out.ms.encode, ms_step: out.ms.step, ms_grammar: out.ms.grammar,
    ms_svg: t2 - t1, ...compareIcon(out.tokens, svg, out.calls, ref),
  };
  if (torchTokens.has(ref.hexcode)) {
    const want = fixtureTokens(torchTokens.get(ref.hexcode));
    row.torch_identical = want.every((t, p) => t === out.tokens[p]);
  }
  rows.push(row);
}

const summary = summarize(rows);
summary.torch_checked = rows.filter(r => "torch_identical" in r).length;
summary.torch_identical = rows.filter(r => r.torch_identical).length;
const result = {
  what: "web/src grammar + greedy loop + codec on onnxruntime-node (CPU) against the Python ORT reference",
  run_id: reference.run_id, model: args.model, reference: args.reference,
  conditions: {
    node: process.version, onnxruntime_node: packageVersion("onnxruntime-node"), threads,
    cpu: os.cpus()[0]?.model, logical_cpus: os.cpus().length, loadavg_before: loadBefore, loadavg_after: os.loadavg(),
  },
  session_ms: Math.round(sessionMs), summary, rows,
};
if (args.json) writeFileSync(args.json, JSON.stringify(result, null, 1) + "\n");
process.stdout.write(JSON.stringify(result) + "\n");
const ok = summary.identical_tokens === count && summary.identical_svg === count && summary.equal_calls === count
  && summary.torch_identical === summary.torch_checked;
process.exit(ok ? 0 : 1);
