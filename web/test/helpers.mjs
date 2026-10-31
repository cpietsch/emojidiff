// Fixture loading and the JavaScript side of web_reference.decode_walk / walk_digest.
import { createHash } from "node:crypto";
import { readFileSync, statSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { gunzipSync } from "node:zlib";
import { createGrammar } from "../src/grammar.mjs";

const here = new URL("./", import.meta.url);
export const codec = JSON.parse(readFileSync(new URL("../src/codec.json", here)));
export const grammar = createGrammar(codec);

/**
 * The fixture directory: MOJIDIFF_WEB_FIXTURES when it is set (the set with every corpus
 * split, under /home/dev/.cache/mojidiff/web-fixtures/<run_id>/full), else the committed
 * validation set in test/fixtures. A set variable that names no directory is an error, so
 * a mistyped path never runs the suite on the committed set instead.
 */
export function fixtureDir() {
  const chosen = process.env.MOJIDIFF_WEB_FIXTURES;
  if (!chosen) return fileURLToPath(new URL("fixtures/", here));
  let directory = false;
  try { directory = statSync(chosen).isDirectory(); } catch { /* reported below */ }
  if (!directory) throw new Error(`MOJIDIFF_WEB_FIXTURES=${chosen} is not a directory`);
  return path.resolve(chosen);
}

const read = name => gunzipSync(readFileSync(path.join(fixtureDir(), name)));
export const loadJsonGz = name => JSON.parse(read(name));

let cache = null;
/**
 * {dir, meta (programs.json.gz), walks (walks.json.gz), tokens(i) -> Int32Array in stored
 * positions}. meta.counts (web_reference.fixture_counts) says what the set holds.
 */
export function fixtures() {
  if (cache) return cache;
  const dir = fixtureDir();
  const meta = loadJsonGz("programs.json.gz");
  const walks = loadJsonGz("walks.json.gz");
  const planes = read("tokens.bin.gz");
  const n = meta.programs.length, L = meta.length;
  if (planes.length !== 2 * n * L) throw new Error(`tokens.bin.gz has ${planes.length} bytes, expected ${2 * n * L}`);
  const tokens = i => {
    const out = new Int32Array(L), lo = i * L, hi = n * L + i * L;
    for (let p = 0; p < L; p++) out[p] = planes[lo + p] | (planes[hi + p] << 8);
    return out;
  };
  cache = { dir, meta, walks, tokens };
  return cache;
}

/** Expand a walks.json.gz mask (a list of [start, stop) ranges) to a Uint8Array. */
export function expandMask(ranges, vocabulary) {
  const mask = new Uint8Array(vocabulary);
  for (const [a, b] of ranges) mask.fill(1, a, b);
  return mask;
}

/**
 * Decode a complete program the way greedy_decode walks it, with the JavaScript grammar:
 * per step the position, the legal mask given the prefix, forced, the variant-B ids rows
 * (6 x L Float32Array via grammar.stepIds) and the target.
 */
export function walk(g, tokens, { keepMasks = false } = {}) {
  const L = g.length, V = g.vocabulary;
  const decoded = new Int32Array(L), order = new Int32Array(L), forced = new Uint8Array(L);
  const target = new Int32Array(L);
  const labels = { field: new Int32Array(L), path: new Int32Array(L), role: new Int32Array(L), axis: new Int32Array(L) };
  const packed = new Uint8Array(L * Math.ceil(V / 8));
  const masks = keepMasks ? [] : null;
  const mask = new Uint8Array(V);
  const walker = g.pathMajorPositions(decoded);
  for (let s = 0; s < L; s++) {
    const position = walker.next().value;
    order[s] = position;
    [labels.field[s], labels.path[s]] = g.labelsOf(position, decoded);
    [labels.role[s], labels.axis[s]] = g.roleOf(position, decoded);
    target[s] = g.targetOf(labels.role[s], labels.axis[s]);
    g.legalMask(position, decoded, mask);
    let count = 0;
    const row = s * Math.ceil(V / 8);
    for (let v = 0; v < V; v++) if (mask[v]) { count++; packed[row + (v >> 3)] |= 1 << (v & 7); }
    forced[s] = count === 1 ? 1 : 0;
    if (masks) masks.push(mask.slice());
    decoded[position] = tokens[position];
  }
  const ids = g.stepIds(0, L, order, decoded, labels);
  return { order, forced, ids, target, packed, masks, tokens: Array.from(order, p => tokens[p]) };
}

/** web_reference.walk_digest: SHA-256 (first 32 hex digits) of a 65-byte record per step. */
export function walkDigest(w, vocabulary) {
  const L = w.order.length, mb = Math.ceil(vocabulary / 8), size = 2 + mb + 1 + 2 + 4 + 1 + 2;
  const bytes = new Uint8Array(L * size), view = new DataView(bytes.buffer);
  for (let s = 0; s < L; s++) {
    let o = s * size;
    view.setUint16(o, w.order[s], true); o += 2;
    bytes.set(w.packed.subarray(s * mb, (s + 1) * mb), o); o += mb;
    bytes[o++] = w.forced[s];
    view.setUint16(o, w.ids[s], true); o += 2; // tokens row
    bytes[o++] = w.ids[2 * L + s]; // field
    bytes[o++] = w.ids[3 * L + s]; // path
    bytes[o++] = w.ids[4 * L + s]; // axis of the input token
    bytes[o++] = w.ids[5 * L + s]; // role of the input token
    view.setInt8(o, w.target[s]); o += 1;
    view.setUint16(o, w.tokens[s], true);
  }
  return createHash("sha256").update(bytes).digest("hex").slice(0, 32);
}

export const sha = text => createHash("sha256").update(text, "utf8").digest("hex").slice(0, 32);
