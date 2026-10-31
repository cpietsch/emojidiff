// Shared by the Node end-to-end check (node.mjs) and the browser page (page.mjs): one
// decoded icon against the Python reference bundle (python -m mojidiff.learning.web_e2e),
// and the per-run summary. No dependencies, so it runs in a page as well as in Node.

/** Refuse a reference bundle written from other model files than the export in use. */
export function checkBundle(manifest, reference) {
  for (const role of ["encoder", "decoder_step"]) {
    const want = manifest.files[role].sha256, got = reference.model[`${role}_sha256`];
    if (want !== got) throw new Error(`reference.json was decoded with another ${role} (${got} != ${want})`);
  }
}

/**
 * Compare a decode with its reference icon: every token in original positions, the SVG
 * text the JavaScript codec wrote, and the number of decoder calls.
 */
export function compareIcon(tokens, svg, calls, reference) {
  const want = reference.tokens;
  let first = -1;
  const n = Math.max(tokens.length, want.length);
  for (let p = 0; p < n; p++) if (tokens[p] !== want[p]) { first = p; break; }
  return {
    identical: first < 0,
    first_difference: first,
    svg_identical: svg === reference.svg,
    calls_equal: calls === reference.model_calls,
  };
}

/** Quantile with linear interpolation between order statistics (numpy's default). */
export function quantile(values, q) {
  if (!values.length) return null;
  const s = [...values].sort((a, b) => a - b);
  const at = q * (s.length - 1), low = Math.floor(at), high = Math.ceil(at);
  return s[low] + (s[high] - s[low]) * (at - low);
}

const sum = values => values.reduce((a, b) => a + b, 0);
const round = (x, digits = 2) => (x === null ? null : Math.round(x * 10 ** digits) / 10 ** digits);

/**
 * Summary of per-icon rows {calls, ms_icon, ms_encode, ms_step, ms_grammar, ms_svg,
 * identical, svg_identical, calls_equal}. ms_icon is pixels in to SVG text out; the
 * grammar share is the time outside the encoder and decoder calls (legal masks, labels,
 * ids, masked argmax, loop) over ms_icon, summed over icons.
 */
export function summarize(rows) {
  const pick = key => rows.map(r => r[key]);
  const icon = pick("ms_icon"), calls = pick("calls");
  return {
    icons: rows.length,
    identical_tokens: rows.filter(r => r.identical).length,
    identical_svg: rows.filter(r => r.svg_identical).length,
    equal_calls: rows.filter(r => r.calls_equal).length,
    ms_icon_median: round(quantile(icon, 0.5), 1),
    ms_icon_p90: round(quantile(icon, 0.9), 1),
    ms_icon_mean: round(sum(icon) / rows.length, 1),
    ms_icon_max: round(Math.max(...icon), 1),
    calls_median: quantile(calls, 0.5),
    calls_p90: round(quantile(calls, 0.9), 1),
    calls_total: sum(calls),
    ms_per_call_mean: round(sum(pick("ms_step")) / sum(calls), 3),
    ms_encode_median: round(quantile(pick("ms_encode"), 0.5), 2),
    ms_grammar_median: round(quantile(pick("ms_grammar"), 0.5), 2),
    ms_svg_median: round(quantile(pick("ms_svg"), 0.5), 3),
    grammar_share: round(sum(pick("ms_grammar")) / sum(icon), 4),
    step_share: round(sum(pick("ms_step")) / sum(icon), 4),
  };
}
