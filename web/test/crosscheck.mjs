// Command-line side of tests/test_web.py::test_random_legal_programs_match_python.
//
//   node test/crosscheck.mjs programs.json
//
// programs.json is a list of programs (tokens in stored positions). Prints, per program, what
// the JavaScript computes: the walk digest (helpers.walkDigest), the batch-1 call count and
// the SVG text. Python generates programs the committed fixtures never reach (random legal
// tokens at every position) and compares each field with its own reference code.
import { readFileSync } from "node:fs";
import { tokensToSvg } from "../src/codec.mjs";
import { codec, grammar, walk, walkDigest } from "./helpers.mjs";

const programs = JSON.parse(readFileSync(process.argv[2], "utf8"));
const results = programs.map(values => {
  if (values.length !== grammar.length) throw new Error(`program of ${values.length} tokens`);
  const tokens = Int32Array.from(values);
  const w = walk(grammar, tokens);
  return {
    walk_sha256: walkDigest(w, grammar.vocabulary),
    model_calls: w.forced.reduce((n, f) => n + (f ? 0 : 1), 0),
    svg: tokensToSvg(tokens, codec),
  };
});
process.stdout.write(JSON.stringify(results));
