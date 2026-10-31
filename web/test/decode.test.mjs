// The greedy decode loop, driven by a replay step that checks every call against Python.
//
// For each detailed walk the replay step is the model: it asserts that each call's ids and
// target are the Python rows for exactly the steps greedy_decode fed in that call, that the
// KV cache handed in is the one it returned last time, and it answers with logits whose
// largest legal value (by the Python mask) is the program's token, while every illegal
// token scores above every legal one. Decoding therefore
// reproduces the program only if the loop masks exactly as Python does (admitting any
// Python-illegal token changes the result), skips forced steps and chunks its calls as
// greedy_decode does.
import assert from "node:assert/strict";
import { test } from "node:test";
import { greedyDecode } from "../src/decode.mjs";
import { codec, expandMask, fixtures, grammar } from "./helpers.mjs";

const L = grammar.length, V = grammar.vocabulary;

function replay(ref, program, table, maxChunk) {
  const free = ref.forced.flatMap((f, s) => (f ? [] : [s]));
  const memory = { memory: ref.program };
  const calls = [];
  let previous = null, cursor = 0, seed = 1;
  const random = () => ((seed = (seed * 16807) % 2147483647) / 2147483647);
  const step = async (ids, target, past, mem) => {
    assert.equal(mem, memory, "memory is passed through unchanged");
    assert.equal(past, previous, "past is the previous call's present");
    const n = ids.length / 6;
    assert.ok(Number.isInteger(n) && n >= 1 && n <= maxChunk, `chunk of ${n}`);
    const start = cursor, end = start + n, last = end - 1;
    for (let i = 0; i < n; i++) {
      const j = start + i;
      const want = [ref.token_in[j], j, ref.field[j], ref.path[j], ref.axis_in[j], ref.role_in[j]];
      const got = [0, 1, 2, 3, 4, 5].map(r => ids[r * n + i]);
      assert.deepEqual(got, want, `ids of step ${j}`);
    }
    assert.equal(target.length, 1);
    assert.equal(target[0], ref.target[last], `target of step ${last}`);
    cursor = end;
    calls.push([start, end]);
    const mask = table[ref.mask[last]];
    const logits = Float32Array.from({ length: V }, (_, v) => random() + (mask[v] ? 0 : 3));
    logits[program[ref.order[last]]] = 2;
    previous = { present: calls.length };
    return { logits, present: previous };
  };
  return { step, encode: async image => { assert.equal(image, "pixels"); return memory; }, calls, free, memory };
}

async function decodeAll(maxChunk) {
  const { meta, walks, tokens } = fixtures();
  const table = walks.masks.map(ranges => expandMask(ranges, V));
  const schedules = new Map(walks.schedules.map(s => [s.program, s.calls]));
  const results = [];
  for (const ref of walks.walks) {
    const entry = meta.programs[ref.program];
    const where = `${entry.source} ${entry.hexcode}`;
    const program = tokens(ref.program);
    const r = replay(ref, program, table, maxChunk);
    const out = await greedyDecode({ grammar, encode: r.encode, step: r.step, image: "pixels", maxChunk });
    assert.deepEqual(Array.from(out.tokens), Array.from(program), `${where}: decoded program`);
    assert.deepEqual(Array.from(out.order), ref.order, `${where}: order`);
    assert.equal(out.forced, L - r.free.length, `${where}: forced steps`);
    assert.equal(out.memory, r.memory, `${where}: the encoder memory is returned for release`);
    results.push({ ref, entry, out, calls: r.calls, free: r.free, schedule: schedules.get(ref.program) });
  }
  return results;
}

test("replayed decodes reproduce every detailed program with greedy_decode's call schedule", async t => {
  const { walks } = fixtures();
  const results = await decodeAll(Infinity);
  let recorded = 0;
  for (const { entry, out, calls, free, schedule } of results) {
    const where = `${entry.source} ${entry.hexcode}`;
    assert.equal(out.calls, entry.model_calls, `${where}: calls`);
    let start = 0;
    const expected = free.map(s => { const c = [start, s + 1]; start = s + 1; return c; });
    assert.deepEqual(calls, expected, `${where}: one call per free step, forced steps queued`);
    if (schedule) {
      recorded++;
      assert.deepEqual(calls, schedule, `${where}: the calls greedy_decode made`);
      assert.equal(out.calls, entry.recorded_model_calls, `${where}: DecodeStats.model_calls`);
    }
  }
  assert.ok(recorded >= 2, `${recorded} decodes with a recorded schedule`);
  assert.equal(recorded, walks.counts.schedules, "every recorded schedule is replayed");
  const grammarMs = results.map(r => r.out.ms.grammar).sort((a, b) => a - b);
  t.diagnostic(`grammar + loop overhead per icon: median ${grammarMs[grammarMs.length >> 1].toFixed(2)} ms ` +
    `(${results.length} replayed icons, Node ${process.version})`);
});

test("maxChunk splits queued steps into several calls without changing the result", async () => {
  for (const maxChunk of [1, 3]) {
    const results = await decodeAll(maxChunk);
    for (const { entry, out, calls, free } of results) {
      let start = 0, expected = 0;
      for (const s of free) { expected += Math.ceil((s + 1 - start) / maxChunk); start = s + 1; }
      assert.equal(out.calls, expected, `${entry.source} ${entry.hexcode}: calls at maxChunk ${maxChunk}`);
      assert.equal(calls.at(-1)[1], free.at(-1) + 1);
    }
  }
});

test("too few logits or an invalid maxChunk is an error, not a silent decode", async () => {
  const step = async () => ({ logits: new Float32Array(V - 1), present: null });
  await assert.rejects(greedyDecode({ grammar, encode: async () => null, step, image: null }), /logits/);
  // codec.json itself is accepted as the grammar (greedyDecode binds it with createGrammar).
  await assert.rejects(greedyDecode({ grammar: codec, encode: async () => null, step, image: null }), /logits/);
  for (const maxChunk of [0, 2.5, NaN]) {
    await assert.rejects(greedyDecode({ grammar, encode: async () => null, step, image: null, maxChunk }), /maxChunk/);
  }
});
