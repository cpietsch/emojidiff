// The render-to-SVG decoding grammar, ported from the Python reference:
//   legalMask        <- autoregressive.legal_mask (and its _segment_kind/_coordinate masks)
//   pathMajorPositions <- render2svg.path_major_positions
//   roleOf / labelsOf  <- fast_decode.GraphDecoder._role / _labels
//                         (= render2svg.coordinate_roles / position_labels per position)
//   stepIds / targetOf <- the variant-B decoder step's inputs (reports/browser/m0/export_variants.py):
//                         ids (6, B, L) = tokens, steps, field, path, axis, role; target (B,)
// Every rule reads only tokens decoded earlier in path-major order, so `decoded` holds the
// prefix (later positions still 0), exactly as greedy_decode's tensor does.
// web/test/grammar.test.mjs checks all of it against fixtures written by
// scripts/export_web_fixtures.py from the Python code.

export const PAD = 0, NONE = 1;
export const ROLE_NONE = 0, ROLE_ENDPOINT = 1, ROLE_CONTROL = 2;
export const IDS_ROWS = ["tokens", "steps", "field", "path", "axis", "role"];

const FIELDS = ["path_length", "layer", "opacity", "fill", "fill_opacity", "stroke", "stroke_opacity",
  "stroke_width", "linecap", "linejoin", "miter_limit", "dash_pattern", "fill_rule"];
const F = Object.fromEntries(FIELDS.map((n, i) => [n, i]));
const LINE = 1, QUAD = 2, CUBIC = 3, CLOSE = 4;

function check(condition, what) {
  if (!condition) throw new Error(`codec.json does not match grammar.mjs: ${what}`);
}

/** Bind the grammar to the constants in codec.json (scripts/export_web_fixtures.py codec). */
export function createGrammar(c) {
  check(JSON.stringify(c.path_fields) === JSON.stringify(FIELDS), "path_fields");
  check(c.path_stride === FIELDS.length + 2 && c.segment_stride === 7, "strides");
  check(JSON.stringify(c.segment_types) === JSON.stringify({ pad: 0, line: 1, quad: 2, cubic: 3, close: 4 }),
    "segment_types");
  check(c.tokens.pad === PAD && c.tokens.none === NONE, "pad/none tokens");
  check(c.roles.none === ROLE_NONE && c.roles.endpoint === ROLE_ENDPOINT && c.roles.control === ROLE_CONTROL, "roles");
  check(JSON.stringify(c.ids_rows) === JSON.stringify(IDS_ROWS), "ids_rows");

  const PS = c.path_stride, SS = c.segment_stride;
  const maxPaths = c.max_paths, slots = c.total_segment_slots, length = c.length, V = c.vocabulary;
  const pathPositions = c.path_positions;
  check(pathPositions === maxPaths * PS && length === pathPositions + slots * SS, "length");
  const endpointBins = c.coordinate_bins, controlBins = c.effective_control_coordinate_bins;
  const controlSlots = c.control_slots; // control coordinates per kind token
  const palette = c.palette.length, opacities = c.opacities.length;
  // Legal sizes of the stroke-dependent fields, by field offset.
  const strokeSize = new Map([
    [F.stroke_opacity, opacities], [F.stroke_width, c.stroke_widths.length], [F.linecap, c.linecaps.length],
    [F.linejoin, c.linejoins.length], [F.miter_limit, c.miter_limits.length], [F.dash_pattern, c.dash_patterns.length],
  ]);

  // Static tables (render2svg._static_tables): segment slot, kind position, path-start axis.
  const slot = new Int16Array(length).fill(-1);
  const kindPosition = new Int16Array(length).fill(-1);
  const startAxis = new Int8Array(length).fill(-1);
  for (let s = 0; s < slots; s++) {
    const base = pathPositions + s * SS;
    for (let i = 1; i < SS; i++) { slot[base + i] = i - 1; kindPosition[base + i] = base; }
  }
  for (let p = 0; p < maxPaths; p++) {
    startAxis[p * PS + FIELDS.length] = 0;
    startAxis[p * PS + FIELDS.length + 1] = 1;
  }

  // mask[start:stop] = 1 with Python slice clamping.
  const on = (mask, start, stop) => { if (stop > V) stop = V; for (let v = start; v < stop; v++) mask[v] = 1; };

  function segmentKindMask(mask, position, decoded) {
    const segment = ((position - pathPositions) / SS) | 0;
    let start = 0;
    for (let p = 0; p < maxPaths; p++) {
      const n = decoded[p * PS];
      if (n && start <= segment && segment < start + n) {
        mask[LINE] = mask[QUAD] = mask[CUBIC] = 1;
        if (segment - start === n - 1) mask[CLOSE] = 1; // close may only end a path
        return mask;
      }
      start += n;
    }
    mask[PAD] = 1; // beyond every declared path length
    return mask;
  }

  function coordinateMask(mask, segment, index, decoded) {
    const kind = decoded[pathPositions + segment * SS];
    if (kind !== LINE && kind !== QUAD && kind !== CUBIC) { mask[PAD] = 1; return mask; }
    const controls = controlSlots[kind];
    if (index >= controls + 2) { mask[PAD] = 1; return mask; } // unused slot of a shorter kind
    on(mask, 1, (index < controls ? controlBins : endpointBins) + 1);
    return mask;
  }

  /**
   * Legal tokens at `position` given the decoded prefix: Uint8Array(vocabulary), 1 = legal.
   * Writes into `mask` when given (it is cleared first).
   */
  function legalMask(position, decoded, mask = new Uint8Array(V)) {
    mask.fill(0);
    if (position >= pathPositions) {
      const offset = position - pathPositions, within = offset % SS;
      if (within === 0) return segmentKindMask(mask, position, decoded);
      return coordinateMask(mask, (offset / SS) | 0, within - 1, decoded);
    }
    const path = (position / PS) | 0, within = position % PS, base = path * PS;
    if (within === F.path_length) {
      let used = 0, allActive = true;
      for (let p = 0; p < path; p++) { const n = decoded[p * PS]; used += n; if (!(n > 0)) allActive = false; }
      mask[0] = 1; // a path may always be absent
      if (allActive) on(mask, 1, Math.min(c.max_segments, Math.max(slots - used, 0)) + 1);
      return mask;
    }
    if (!(decoded[base] > 0)) { mask[PAD] = 1; return mask; } // inactive path: all PAD
    const fill = decoded[base + F.fill], stroke = decoded[base + F.stroke];
    // A repeated layer pins every style field to the previous path's value.
    if (path > 0 && within > F.layer && within < FIELDS.length) {
      const here = decoded[base + F.layer], before = decoded[base - PS + F.layer];
      if (here && here === before) { mask[decoded[base - PS + within]] = 1; return mask; }
    }
    if (within === F.layer) {
      let previous = 0;
      for (let p = 0; p < path; p++) previous = Math.max(previous, decoded[p * PS + F.layer]);
      if (previous >= 1) mask[previous] = 1; // reuse the layer
      on(mask, previous + 1, maxPaths + 1);
      return mask;
    }
    if (within === F.opacity) { on(mask, 2, opacities + 2); return mask; }
    if (within === F.fill) { mask[NONE] = 1; on(mask, 2, palette + 2); return mask; }
    if (within === F.stroke) {
      if (fill !== NONE) mask[NONE] = 1; // a fill-painting path may omit its stroke
      on(mask, 2, palette + 2);
      return mask;
    }
    if (within === F.fill_opacity || within === F.fill_rule) {
      if (fill === NONE) mask[NONE] = 1;
      else on(mask, 2, (within === F.fill_opacity ? opacities : c.fill_rules.length) + 2);
      return mask;
    }
    const size = strokeSize.get(within);
    if (size !== undefined) {
      if (stroke === NONE) mask[NONE] = 1;
      else { if (within === F.dash_pattern) mask[NONE] = 1; on(mask, 2, size + 2); }
      return mask;
    }
    on(mask, 1, endpointBins + 1); // the two start coordinates
    return mask;
  }

  /**
   * Original positions in path-major order, read lazily from `decoded`: each path's
   * metadata, then (its length now decoded) its own segment slots; the first inactive path
   * ends the program and every position not yet listed follows in stored order.
   * Fill `decoded` at each yielded position before asking for the next.
   */
  function* pathMajorPositions(decoded) {
    const listed = new Uint8Array(length);
    let offset = 0;
    for (let p = 0; p < maxPaths; p++) {
      const base = p * PS;
      for (let q = base; q < base + PS; q++) { listed[q] = 1; yield q; }
      const n = decoded[base];
      if (n === 0) break;
      const end = Math.min(offset + n, slots);
      for (let s = offset; s < end; s++) {
        const start = pathPositions + s * SS;
        for (let q = start; q < start + SS; q++) { listed[q] = 1; yield q; }
      }
      offset += n;
    }
    for (let q = 0; q < length; q++) if (!listed[q]) yield q;
  }

  /** [role, axis] of an original position, from earlier tokens only. */
  function roleOf(position, decoded) {
    const axis = startAxis[position];
    if (axis >= 0) return [ROLE_ENDPOINT, axis];
    const s = slot[position];
    if (s < 0) return [ROLE_NONE, 0];
    const kind = Math.min(Math.max(decoded[kindPosition[position]], 0), 4);
    const controls = controlSlots[kind];
    const used = kind >= 1 && kind <= 3 ? controls + 2 : 0;
    if (s >= used) return [ROLE_NONE, s % 2];
    return [s < controls ? ROLE_CONTROL : ROLE_ENDPOINT, s % 2];
  }

  /** [field type 0-21, owning path 0-31 or max_paths for padding] of an original position. */
  function labelsOf(position, decoded) {
    if (position < pathPositions) return [position % PS, (position / PS) | 0];
    const offset = position - pathPositions, s = (offset / SS) | 0, field = PS + offset % SS;
    let end = 0;
    for (let p = 0; p < maxPaths; p++) {
      end += decoded[p * PS];
      if (s < end) return [field, p];
    }
    return [field, maxPaths];
  }

  /** The metric target of a step from its own position's role and axis: (role-1)*2+axis, or -1. */
  const targetOf = (role, axis) => (role > 0 ? (role - 1) * 2 + axis : -1);

  /**
   * Variant-B ids for steps start..end-1, as a Float32Array laid out (6, 1, end - start).
   * `order`, `field`, `path`, `role`, `axis` are per step (role/axis of the step's own
   * position); step j reads the token, role and axis of step j - 1.
   */
  function stepIds(start, end, order, decoded, labels) {
    const n = end - start, ids = new Float32Array(6 * n);
    for (let i = 0; i < n; i++) {
      const j = start + i;
      ids[n + i] = j;
      ids[2 * n + i] = labels.field[j];
      ids[3 * n + i] = labels.path[j];
      if (j > 0) {
        ids[i] = decoded[order[j - 1]];
        ids[4 * n + i] = labels.axis[j - 1];
        ids[5 * n + i] = labels.role[j - 1];
      }
    }
    return ids;
  }

  return {
    constants: c, length, vocabulary: V, maxPaths, slots, pathPositions,
    legalMask, pathMajorPositions, roleOf, labelsOf, targetOf, stepIds,
    tables: { slot, kindPosition, startAxis },
  };
}

/** {count, first}: how many tokens a mask admits and the lowest one (-1 if none). */
export function maskCount(mask) {
  let count = 0, first = -1;
  for (let v = 0; v < mask.length; v++) if (mask[v]) { if (first < 0) first = v; count++; }
  return { count, first };
}

/**
 * Index of the largest legal logit; ties go to the lowest index, as torch.argmax does. With
 * finite logits this is greedy_decode's choice; if every legal logit is -Infinity or NaN it
 * returns the lowest legal token (torch would return an illegal index 0 or the NaN).
 */
export function maskedArgmax(logits, mask, offset = 0) {
  let best = -1, value = -Infinity;
  for (let v = 0; v < mask.length; v++) {
    if (!mask[v]) continue;
    const x = logits[offset + v];
    if (x > value || best < 0) { best = v; value = x; }
  }
  return best;
}
