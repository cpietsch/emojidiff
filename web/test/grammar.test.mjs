// The JavaScript grammar against the Python reference (fixtures from scripts/export_web_fixtures.py).
import assert from "node:assert/strict";
import { test } from "node:test";
import { createGrammar, maskCount, maskedArgmax } from "../src/grammar.mjs";
import { codec, expandMask, fixtures, grammar, walk, walkDigest } from "./helpers.mjs";

test("codec.json and the fixtures describe the same layout", () => {
  const { meta, walks } = fixtures();
  assert.equal(meta.length, codec.length);
  assert.equal(meta.vocabulary, codec.vocabulary);
  assert.equal(walks.length, codec.length);
  assert.equal(grammar.length, 1376);
  assert.equal(grammar.vocabulary, 418);
});

test("the fixtures hold exactly what their metadata counts", t => {
  const { dir, meta, walks } = fixtures();
  const counts = meta.counts;
  t.diagnostic(`fixtures ${dir}: ${counts.programs} programs ${JSON.stringify(counts.sources)}, ` +
    `corpus ${JSON.stringify(counts.corpus_splits)}`);
  assert.deepEqual(walks.counts, counts, "programs.json.gz and walks.json.gz count the same set");
  assert.equal(meta.programs.length, counts.programs);
  const sources = {}, splits = {};
  for (const p of meta.programs) {
    sources[p.source] = (sources[p.source] ?? 0) + 1;
    if (p.source === "corpus") splits[p.split] = (splits[p.split] ?? 0) + 1;
  }
  assert.deepEqual(sources, counts.sources);
  assert.deepEqual(splits, counts.corpus_splits);
  assert.deepEqual(Object.keys(splits).sort(), [...meta.splits].sort(), "corpus programs of the recorded splits only");
  assert.equal(meta.programs.filter(p => p.svg_sha256 !== null).length, counts.svg);
  assert.equal(walks.walks.length, counts.walks);
  assert.equal(walks.schedules.length, counts.schedules);
  for (const source of ["corpus", "greedy", "sample"]) assert.ok(counts.sources[source] > 0, source);
});

test("createGrammar refuses constants it was not written for", () => {
  const changed = structuredClone(codec);
  changed.path_fields = [...codec.path_fields].reverse();
  assert.throws(() => createGrammar(changed), /path_fields/);
  assert.throws(() => createGrammar({ ...codec, segment_stride: 8 }), /strides/);
});

test("an empty prefix: the first path may be absent or take 1..max_segments segments", () => {
  const decoded = new Int32Array(grammar.length);
  const mask = grammar.legalMask(0, decoded);
  assert.deepEqual(maskCount(mask), { count: codec.max_segments + 1, first: 0 });
  // Its layer is any of 1..max_paths; a later path with no active predecessor is PAD only.
  const layer = grammar.legalMask(1, Int32Array.from({ length: grammar.length }, (_, i) => (i === 0 ? 3 : 0)));
  assert.deepEqual(maskCount(layer), { count: codec.max_paths, first: 1 });
  assert.deepEqual(maskCount(grammar.legalMask(16, decoded)), { count: 1, first: 0 });
});

test("maskedArgmax ignores illegal tokens and breaks ties low, like torch.argmax", () => {
  const mask = Uint8Array.from([0, 1, 1, 0, 1]);
  assert.equal(maskedArgmax([9, 1, 3, 9, 3], mask), 2);
  // Unlike torch on an all -inf row (index 0, illegal), a legal token is still returned.
  assert.equal(maskedArgmax([9, -Infinity, -Infinity, 9, -Infinity], mask), 1);
});

function firstDifference(a, b) {
  for (let i = 0; i < Math.max(a.length, b.length); i++) if (a[i] !== b[i]) return i;
  return -1;
}

test("every step of the detailed walks equals Python: order, mask, forced, ids, target", () => {
  const { meta, walks, tokens } = fixtures();
  const L = grammar.length, V = grammar.vocabulary;
  const table = walks.masks.map(ranges => expandMask(ranges, V));
  assert.ok(walks.walks.length >= 3);
  assert.equal(walks.walks.length, meta.counts.walks);
  let steps = 0;
  for (const ref of walks.walks) {
    const program = meta.programs[ref.program];
    const where = `${program.source} ${program.hexcode} (program ${ref.program})`;
    const js = walk(grammar, tokens(ref.program), { keepMasks: true });
    const rows = {
      order: js.order, forced: js.forced, target: js.target,
      token_in: js.ids.subarray(0, L), field: js.ids.subarray(2 * L, 3 * L), path: js.ids.subarray(3 * L, 4 * L),
      axis_in: js.ids.subarray(4 * L, 5 * L), role_in: js.ids.subarray(5 * L, 6 * L),
    };
    for (const [name, got] of Object.entries(rows)) {
      const at = firstDifference(Array.from(got), ref[name]);
      assert.equal(at, -1, `${where}: ${name} differs at step ${at} (position ${ref.order[at]}): ` +
        `js ${got[at]}, python ${ref[name][at]}`);
    }
    assert.deepEqual(Array.from(js.ids.subarray(L, 2 * L)), Array.from({ length: L }, (_, j) => j), `${where}: steps`);
    for (let s = 0; s < L; s++) {
      const expected = table[ref.mask[s]];
      const at = firstDifference(js.masks[s], expected);
      assert.equal(at, -1, `${where}: mask differs at step ${s} (position ${ref.order[s]}), token ${at}: ` +
        `js ${js.masks[s][at]}, python ${expected[at]}`);
    }
    steps += L;
  }
  assert.equal(steps, walks.walks.length * L);
});

test("walk digests of every fixture program equal Python's (all positions, all programs)", t => {
  const { meta, tokens } = fixtures();
  const bad = [];
  let checked = 0;
  const started = performance.now();
  for (let i = 0; i < meta.programs.length; i++) {
    const program = meta.programs[i];
    const js = walk(grammar, tokens(i));
    if (walkDigest(js, grammar.vocabulary) !== program.walk_sha256) bad.push(`${program.source} ${program.hexcode}`);
    const calls = js.forced.reduce((n, f) => n + (f ? 0 : 1), 0);
    if (calls !== program.model_calls) bad.push(`${program.source} ${program.hexcode}: ${calls} calls`);
    checked++;
  }
  const ms = performance.now() - started;
  t.diagnostic(`${checked} programs x ${grammar.length} steps, ${(ms / checked).toFixed(2)} ms per program walk`);
  assert.deepEqual(bad.slice(0, 10), [], `${bad.length} programs differ`);
  assert.equal(checked, meta.counts.programs, "every program the fixtures count");
  const sources = new Set(meta.programs.map(p => p.source));
  assert.ok(sources.has("corpus") && sources.has("greedy") && sources.has("sample"), [...sources].join());
});
