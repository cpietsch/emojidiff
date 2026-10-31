// Greedy decoding of one render, the JavaScript counterpart of render2svg.greedy_decode at
// batch 1 for a metric, path-order model (fast_decode.GraphDecoder runs the same loop):
//
//   memory = encode(image)                      once per icon
//   for each step in path-major order:
//     mask = legal mask given the decoded prefix
//     one legal token  -> decode it without a model call (queued for the next call)
//     otherwise        -> feed every step since the last call in one step() call and
//                         take the masked argmax of the logits for this step
//
// The model is abstract, so the same loop drives onnxruntime-web, onnxruntime-node or a
// test double:
//
//   encode(image)                     -> memory (opaque; awaited once)
//   step(ids, target, past, memory)   -> {logits, present}   (may return a Promise)
//     ids     Float32Array, the variant-B (6, 1, L) rows [tokens, steps, field, path, axis, role]
//     target  Float32Array(1): metric target of the chunk's last step, -1 for none
//     past    null on the first call, then the previous call's `present`, handed over
//             exactly once; a WebGPU adapter may dispose it once its call has run
//     logits  the (1, vocabulary) logits for the chunk's last step
//
// `maxChunk` splits a long queue into several calls (static-shape graphs feed one step at a
// time); only the last call's logits are used, as GraphDecoder does with its chunk sizes.

import { createGrammar, maskCount, maskedArgmax } from "./grammar.mjs";

const clock = () => (typeof performance !== "undefined" ? performance.now() : Date.now());

/**
 * Decode one render. Returns {tokens (Int32Array, original positions), order (Int32Array,
 * the position decoded at each step), calls, forced (steps decoded without a call), past
 * (the last present) and memory (what encode returned), both for the caller to release,
 * ms: {total, encode, step, grammar}}, where ms.grammar is everything outside encode and
 * step: grammar, ids, masked argmax.
 */
export async function greedyDecode({ grammar, encode, step, image, maxChunk = Infinity, now = clock }) {
  const g = grammar.legalMask ? grammar : createGrammar(grammar);
  if (!(maxChunk === Infinity || (Number.isInteger(maxChunk) && maxChunk >= 1))) {
    throw new Error("maxChunk must be a positive integer or Infinity");
  }
  const L = g.length, V = g.vocabulary;
  const started = now();
  const memory = await encode(image);
  const encoded = now();
  const decoded = new Int32Array(L), order = new Int32Array(L);
  const labels = { field: new Int32Array(L), path: new Int32Array(L), role: new Int32Array(L), axis: new Int32Array(L) };
  const walker = g.pathMajorPositions(decoded);
  const mask = new Uint8Array(V);
  let past = null, fed = 0, calls = 0, forced = 0, stepMs = 0;
  for (let s = 0; s < L; s++) {
    const position = walker.next().value;
    order[s] = position;
    [labels.field[s], labels.path[s]] = g.labelsOf(position, decoded);
    [labels.role[s], labels.axis[s]] = g.roleOf(position, decoded);
    g.legalMask(position, decoded, mask);
    const { count, first } = maskCount(mask);
    if (count === 0) throw new Error(`no legal token at step ${s} (position ${position})`);
    if (count === 1) { decoded[position] = first; forced++; continue; }
    let logits = null;
    for (let start = fed; start <= s; start += maxChunk) {
      const end = Math.min(start + maxChunk, s + 1);
      const ids = g.stepIds(start, end, order, decoded, labels);
      const target = new Float32Array([g.targetOf(labels.role[end - 1], labels.axis[end - 1])]);
      const before = now();
      const out = await step(ids, target, past, memory);
      stepMs += now() - before;
      past = out.present;
      logits = out.logits;
      calls++;
    }
    fed = s + 1;
    if (logits.length < V) throw new Error(`step returned ${logits.length} logits, expected ${V}`);
    const token = maskedArgmax(logits, mask);
    if (token < 0) throw new Error(`no legal logit at step ${s}`);
    decoded[position] = token;
  }
  const total = now() - started;
  return {
    tokens: decoded, order, calls, forced, past, memory,
    ms: { total, encode: encoded - started, step: stepMs, grammar: total - (encoded - started) - stepMs },
  };
}
