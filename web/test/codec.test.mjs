// The JavaScript SVG writer against serialize_packed_svg, byte for byte (via SHA-256 of the UTF-8).
import assert from "node:assert/strict";
import { test } from "node:test";
import { num, tokensToSvg } from "../src/codec.mjs";
import { codec, fixtures, sha } from "./helpers.mjs";

test("num() writes floats as Python's repr does", () => {
  assert.equal(num(0), "0");
  assert.equal(num(1), "1.0");
  assert.equal(num(-8), "-8.0");
  assert.equal(num(0.25), "0.25");
  assert.equal(num(2.4999579387999997), "2.4999579387999997");
});

test("every fixture program serialises to Python's SVG, byte for byte", t => {
  const { meta, tokens } = fixtures();
  const bad = [];
  let checked = 0, ms = 0;
  for (let i = 0; i < meta.programs.length; i++) {
    const program = meta.programs[i];
    if (program.svg_sha256 === null) continue;
    const program_tokens = tokens(i);
    const started = performance.now();
    const svg = tokensToSvg(program_tokens, codec);
    ms += performance.now() - started;
    checked++;
    if (sha(svg) !== program.svg_sha256 || Buffer.byteLength(svg) !== program.svg_bytes) {
      bad.push(`${program.source} ${program.hexcode}: ${svg.slice(0, 120)}`);
    }
  }
  t.diagnostic(`${checked} programs, ${(ms / checked).toFixed(4)} ms per SVG`);
  assert.deepEqual(bad.slice(0, 5), [], `${bad.length} of ${checked} SVGs differ`);
  // Every program Python could serialise, as many as the fixtures' own metadata counts.
  assert.ok(checked > 0, "no program checked");
  assert.equal(checked, meta.counts.svg, `${checked} programs checked, the fixtures count ${meta.counts.svg}`);
});
