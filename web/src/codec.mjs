// Port of autoregressive.unflatten_program + packed.serialize_packed_svg (program.decode_program
// + serialize_float_svg) for programs the grammar admits; moved from milestone 0
// (browser-probe-2026-10-04/codec/codec.mjs). `c` is codec.json (scripts/export_web_fixtures.py
// codec). tokens: Int32Array/Array of length c.length (1376) in ORIGINAL (stored) positions.
// web/test/codec.test.mjs checks it byte for byte against the Python serializer.
const FIELDS = ["path_length","layer","opacity","fill","fill_opacity","stroke","stroke_opacity",
  "stroke_width","linecap","linejoin","miter_limit","dash_pattern","fill_rule"];
const F = Object.fromEntries(FIELDS.map((n, i) => [n, i]));
const PATH_STRIDE = 15, SEGMENT_STRIDE = 7, NONE = 1;
const CAPS = ["butt", "round", "square"], JOINS = ["miter", "round", "bevel"], RULES = ["nonzero", "evenodd"];
const CMD = { 1: "L", 2: "Q", 3: "C", 4: "Z" };
const COUNTS = { 1: [0, 2], 2: [2, 4], 3: [4, 6], 4: [0, 0] }; // [control coords, total coords]

// Python repr(float) for the values this codec can produce (|v| in [1e-4, 1e16)).
export function num(v) {
  if (v === 0) return "0";
  return Number.isInteger(v) ? v.toFixed(1) : String(v);
}

export function tokensToSvg(tokens, c) {
  const endpoint = k => 0 + (72 - 0) * (k - 1) / (c.coordinate_bins - 1);
  const control = k => c.control_coordinate_min +
    (c.control_coordinate_max - c.control_coordinate_min) * (k - 1) / (c.effective_control_coordinate_bins - 1);
  const base0 = c.max_paths * PATH_STRIDE;
  const parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">'];
  let offset = 0, group = null;
  const flush = () => { if (group) parts.push(`<path ${group.attrs(group.d.join(" "))}/>`); };
  for (let p = 0; p < c.max_paths; p++) {
    const b = p * PATH_STRIDE, t = name => tokens[b + F[name]];
    const length = t("path_length");
    if (!length) continue;
    let d = ["M", num(endpoint(tokens[b + 13])), num(endpoint(tokens[b + 14]))];
    for (let s = offset; s < offset + length; s++) {
      const sb = base0 + s * SEGMENT_STRIDE, kind = tokens[sb];
      const [nc, n] = COUNTS[kind];
      d.push(CMD[kind]);
      for (let i = 0; i < n; i++) d.push(num(i < nc ? control(tokens[sb + 1 + i]) : endpoint(tokens[sb + 1 + i])));
    }
    offset += length;
    const layer = t("layer");
    if (group && group.layer === layer) { group.d.push(d.join(" ")); continue; }
    flush();
    const fill = t("fill") === NONE ? null : c.palette[t("fill") - 2];
    const stroke = t("stroke") === NONE ? null : c.palette[t("stroke") - 2];
    const fo = t("fill_opacity") === NONE ? null : c.opacities[t("fill_opacity") - 2];
    const so = t("stroke_opacity") === NONE ? null : c.opacities[t("stroke_opacity") - 2];
    const op = c.opacities[t("opacity") - 2];
    const rule = fill === null ? "nonzero" : RULES[t("fill_rule") - 2];
    const style = { width: c.stroke_widths[t("stroke_width") - 2], cap: CAPS[t("linecap") - 2],
      join: JOINS[t("linejoin") - 2], miter: c.miter_limits[t("miter_limit") - 2],
      dash: t("dash_pattern") === NONE ? null : c.dash_patterns[t("dash_pattern") - 2] };
    group = { layer, d: [d.join(" ")], attrs: dd => {
      const a = [`d="${dd}"`, `fill="${fill ?? "none"}"`];
      if (fo !== null && fo !== 1.0) a.push(`fill-opacity="${num(fo)}"`);
      if (rule !== "nonzero") a.push(`fill-rule="${rule}"`);
      a.push(`stroke="${stroke ?? "none"}"`);
      if (stroke !== null) {
        if (so !== null && so !== 1.0) a.push(`stroke-opacity="${num(so)}"`);
        a.push(`stroke-width="${num(style.width)}"`, `stroke-linecap="${style.cap}"`,
          `stroke-linejoin="${style.join}"`, `stroke-miterlimit="${num(style.miter)}"`);
        if (style.dash) a.push(`stroke-dasharray="${style.dash.map(num).join(" ")}"`);
      }
      if (op !== 1.0) a.push(`opacity="${num(op)}"`);
      return a.join(" ");
    } };
  }
  flush();
  parts.push("</svg>");
  return parts.join("");
}
