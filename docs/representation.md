# Selected MojiDiff representation

Status: Gate C selection, 2026-08-29. Implementation stress testing remains Gate D.

## Primary representation and fallback

Use the semantic-stroke typed SVG program as the primary representation. It preserves
editable stroke intent and is structurally much smaller than the PicoSVG-outlined
alternative. Normalize with the outlined route only for the 60 reason-coded semantic
failures in OpenMoji 17.0.0: 26 anisotropic stroke transforms, 33 unsupported
presentations, and one same-document resource. The fallback recovers all 60. Record the
route per example; do not mix its hardware or fidelity measurements invisibly with the
semantic route.

## Typed fields

Each ordered contour records its layer, fill/stroke palette categories, exact opacity
categories, fill rule, stroke cap/join, categorical stroke width/miter/dash, start point,
path length, and typed line/quadratic/cubic/close segments. Padding and `NONE` remain
typed. `path_length == 0` is the sole inactive-path signal.

The leading vocabularies are:

- q289 move points and segment endpoints on the quarter-unit [0,72] lattice;
- q417 quadratic/cubic control points on the quarter-unit [-8,96] lattice;
- the 49-width K48+exact-4.1 vocabulary in the style-v2 report;
- all six observed dash patterns exactly;
- miter limits 1.5, 2, 4, 7, and 10, with three near-10 source literals mapped to 10;
- exact opacity values 0.25, 0.4, 0.5, 0.502, 0.6, 0.9969, 0.997, 0.999, and 1.

Keep q145 coordinates and the K32 width vocabulary only as named compact ablations.
Never clamp an in-range control handle to the viewBox; the OOB study demonstrates
catastrophic tail failures. Clamping remains an explicitly labelled safety fallback
only beyond the bounded control vocabulary.

## Capacity layout

Store path metadata in at most 80 ordered path slots and store segment records in one
contiguous, painter-order-preserving packed array with 1,216 total segment slots. Path
lengths define prefix offsets into the segment array. A whole-contour prefix is the only
allowed truncation rule; never emit a partial contour just to fit a budget.

This P80/T1216 envelope covers all 4,006 reviewed primary programs exactly. It allocates
1,296 logical path-plus-segment slots, versus 36,960 for the former dense P96/S384 upper
bound. For training batches, use the exact nested buckets P32/T128, P48/T256, P64/T512,
and P80/T1216. Bucket membership is a batching optimization derived from the known
program length, not a hidden semantic token. Generation without a known length must
remain valid at the outer P80/T1216 bound or use an explicitly modeled length policy.

## Evidence and remaining work

The selection is supported by the representation, typed-codec, aligned-coordinate,
opacity, full-corpus, OOB-control, style-vocabulary, and capacity-layout reports under
`reports/codec/`. Gate D must implement packed tensor conversion and stress it with
round-trip properties, legal/illegal random tensors, malformed inputs, resource caps,
and isolated renderer failures before any serious model training.
