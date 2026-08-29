# Research findings

## 2026-08-23 — Bootstrap boundary audit

**Hypothesis.** The fresh `gtc` environment is a sufficiently isolated control plane
for local reversible work under the YOLO operating contract.

**Observation.** The hostname and canonical workspace match the contract. Docker is
configured through TLS at `tcp://localhost:2376`; known host-socket paths and socket
mounts are absent, and the isolated daemon contains no workloads. The repository has no
commits and consists only of the operator handoff files. No worker is enabled, no
artifact store is configured, and no remote jobs exist.

**Decision.** Continue with local scaffold, probes, tests, and data-curation code. Do
not issue remote commands or start meaningful GPU work. Treat external Tailscale ACLs,
credential scopes, and dedicated-host provenance as unverified rather than inferring
them from local configuration.

## 2026-08-23 — Audit v1 over-rejected internal SVG references

**Hypothesis.** Rejecting every attribute containing `url(...)` would safely identify
resource-bearing SVGs without affecting legitimate OpenMoji assets.

**Observation.** The complete v1 audit retained all 4,495 source rows but classified
two regional flags as parse defects. Inspection showed only same-document references:
Ontario uses `clip-path="url(#ontario-shield)"`, and Castile-La Mancha uses marker
references. Neither accesses an external resource.

**Decision.** Keep audit v1 and its false-positive rows as evidence. Audit v2 permits
only syntactically bounded same-document fragment identifiers and continues to reject
external URLs, embedded data, scripts, event attributes, and resource-bearing tags.

## 2026-08-23 — OpenMoji 17.0.0 reviewed curation

**Hypothesis.** Corpus-tail metrics can identify actual blank or degenerate assets
without treating legitimate minimal symbols or detailed icons as defects.

**Observation.** Audit v2 safely parsed all 4,495 color SVGs. One Castile-La Mancha
regional flag (`1F3F4-E0065-E0073-E0063-E006D-E007F`) consistently fails CairoSVG at
both target sizes because it references an undefined `Dot` marker; it is retained as an
`exclude_defect` row. The empirical bottom/top 0.5% tails nominated 104 assets. Visual
review of every candidate at 72 px and enlarged 18 px found all 104 intentional and
recognizable, including thin symbols, single strokes, and complex family icons.

Exact duplicate accounting found 185 render clusters with 410 members, identical at
both sizes. A deterministic canonical non-flag row is retained per cluster; 218
noncanonical non-flag aliases are preserved in `excluded/exact-duplicates`. The 270
rows whose metadata group is exactly `flags` are preserved in `excluded/flags`.

**Decision.** The reviewed primary manifest contains 4,006 rows: 3,902 ordinary
includes and 104 reason-coded `include_override` rows. No quarantine remains. The
verification artifact confirms raw hashes and immutability, complete row coverage, no
flags in primary, no exact duplicates in primary frequency counts, and no family split
leakage. Representation complexity remains a Gate C measurement rather than a curation
defect rule.

**Reproducibility limitation.** Audit v1 and v2 were run before the repository had its
first local Git commit. Their configs and artifacts are hashed, but the exact pre-run
dirty worktree snapshot was not recorded. This violates the preferred code-identity
practice and is retained as a negative orchestration finding. A local checkpoint is
required before the next material experiment; nothing will be pushed.

## 2026-08-23 — Stratified representation probe v1

**Hypothesis.** Fully outlining OpenMoji will simplify style operations and preserve
renders, but may lengthen programs enough to make a fixed-slot categorical codec less
attractive than preserving semantic strokes.

**Method.** From the reviewed primary manifest, deterministically select eight icons
from each of 11 metadata groups (88 total), combining ink/complexity extremes with
hash-selected examples. Normalize with PicoSVG 0.23.0 and compare source-primitive
semantic proxies against outlined paths. Render both with the pinned Cairo stack at 72
and 18 px. This is a pre-codec proxy, not a final codec comparison.

**Observation.** All 88 assets normalized successfully and all outlined outputs removed
stroke elements. Visual drift was small: median RGBA MAE was 0.000558 at 72 px and
0.001173 at 18 px; median alpha IoU was 0.9990 and 1.0 respectively. Structural cost was
material: outlined segment counts grew 2.241× at the median, 5.287× at p95, and 13.833×
in the worst case. At a 64-segment/path budget, 18 outlined fixtures would truncate
versus 7 semantic proxies. Path-slot pressure was similar (5 versus 4 above 32 paths).
The dotted-line face expanded from 24 to 332 segments; this is precisely the kind of
stroke geometry destroyed by outlining.

**Decision.** Prefer the semantic-stroke representation as the leading candidate, but
do not select it finally yet. Implement a deterministic semantic normalizer/codec next
and measure actual 128/256-bin round trips, transform flattening, unsupported commands,
and truncation. Retain outlined PicoSVG as the required fallback/control and compare it
with the same codec budgets.

## 2026-08-23 — Typed codec probe v1

**Hypothesis.** At matched typed budgets, semantic strokes will retain less structure
than outlined paths, and 256 coordinate values will improve raster fidelity over 128
without exposing a more important codec failure.

**Method.** Normalize the fixed 88-icon fixture into ordered contour slots. Adjacent
contours from one SVG paint operation share a layer ID and serialize back into one
compound path, preserving holes without allowing geometry after `close`. Encode both
source-semantic and PicoSVG-outlined programs at P48/S128 and P64/S384 with 128 or 256
coordinate values. Validate and deterministically re-encode every tensor, serialize
through a fixed SVG allow-list, and render against the original source at 72 and 18 px.

**Observation.** Semantic normalization succeeded for 87/88 icons. `1F250` was
reason-coded `anisotropic_stroke_transform`: its nonuniform transform cannot preserve a
painted stroke with one scalar width. The outlined fallback normalized all 88. At the
coverage P64/S384 budget neither representation truncated. Semantic programs used 53
segments/icon at the median versus 146 outlined, and their median q256 serialization
was 4,936 bytes versus 9,760. At compact P48/S128, semantic dropped no contours and 181
segments across three icons; outlined dropped 13 contours and 509 total segments. The
outlined truncation made the rice ball visibly invalid and removed the UFO dome, while
the semantic versions preserved those structures.

All 700 encoded candidates validated, rendered, and reproduced byte-stable tensor/SVG
hashes; the 28-entry palette had no outliers. At the coverage budget, moving from q128
to q256 lowered median 18 px RGBA MAE from 0.004717 to 0.002971 for semantic programs
and from 0.005490 to 0.003321 for outlined programs. Q256 medians were close between
representations: semantic/outlined RGBA MAE was 0.002922/0.002609 at 72 px and
0.002971/0.003321 at 18 px.

Worst-case inspection exposed a separate quantizer issue. Mapping 256 values uniformly
over the closed 0..72 interval cannot represent ordinary integer coordinates exactly;
for example, the `E2C2` background edge at 4 becomes 3.952941. That tiny movement turns
pixel-aligned edges into antialiased edges and inflates pixelwise MAE despite preserving
recognizability. One semantic and two outlined icons also required explicitly recorded
clamping of a Bézier control point outside the viewBox.

**Decision.** Semantic strokes remain the leading representation and outlined paths
remain the explicit per-icon fallback, but Gate C is not closed. P48/S128 is falsified
as a lossless fixture budget, while P64/S384 is coverage-safe but too generous to adopt
without full-corpus tails. Before interpreting q256 as the coordinate choice, compare
pixel-aligned 145- and 289-value lattices (0.5 and 0.25 unit steps). Preserve this
power-of-two-grid result as a negative finding, then run structural normalization over
the full primary manifest before selecting final slot budgets.

## 2026-08-23 — Pixel-aligned coordinate lattice probe

**Hypothesis.** Much of the q128/q256 error comes from using 127/255 equal intervals
over 0..72, which cannot preserve common integer and half-integer source coordinates.
Lattices with 145 values (0.5-unit steps) and 289 values (0.25-unit steps) should improve
edge fidelity while leaving all structural results unchanged.

**Observation.** The controlled rerun held the 88-icon fixture, normalizers, palette and
style vocabularies, P48/S128 and P64/S384 budgets, renderer, and metrics fixed. All 700
candidates again validated, rendered, and reproduced stable hashes. At P64/S384,
semantic q145 reduced median RGBA MAE from q128's 0.004638/0.004717 to
0.002700/0.002648 at 72/18 px despite a similarly sized vocabulary. Semantic q289
reduced q256's 0.002922/0.002971 to 0.001513/0.001710: improvements of 48.2% and 42.5%.
Outlined q289 improved over q256 by 23.9% at 72 px and 30.2% at 18 px.

The alignment-sensitive `E2C2` case was decisive: semantic 18 px MAE fell from 0.117692
at q256 to 0.001970 at q289, and alpha IoU rose from 0.8 to 1.0. The q289 worst-case
semantic 18 px MAE over the fixture fell to 0.035594 from q256's 0.117692. Visual
inspection confirmed preserved recognition and crisper agreement at ordinary edges.
Aligned coordinates also serialize more compactly because exact quarter-unit values
need shorter decimals: median q289 SVG size was 2,448 semantic and 3,547 outlined bytes,
versus 4,936 and 9,760 for q256.

**Decision.** Adopt the 289-value quarter-unit lattice as the leading codec choice and
retain q145 as the compact coordinate ablation. This result strengthens the semantic
representation choice but does not determine final P/S ceilings: perform a full-primary
structural normalization pass next, retaining outlined fallback counts and every
unsupported semantic reason, before closing Gate C.

## 2026-08-23 — Full-primary structural codec census

**Hypothesis.** Semantic-native programs with a reason-coded outlined fallback will
cover nearly all 4,006 curated primary icons at lower structural cost than outlining
everything, and the corpus tails will identify a compact lossless fixed P/S budget.

**Method.** Hash-check every source in the pinned reviewed-primary manifest, attempt
both semantic and PicoSVG-outlined normalization for every icon, and retain exact
ordered contour/layer/style evidence. Evaluate path-first truncation over the Cartesian
grid P={32,48,64,96,128}, S={64,128,192,256,384,512}. The run made exactly 8,012
attempts and installed four immutable outputs with independently verified hashes and
row counts.

**Observation.** Semantic normalization succeeded for 3,937 icons. Its 69 failures
were 33 unsupported `paint-order` presentations, 26 anisotropically transformed
strokes, nine partial-opacity programs, and one same-document resource. Outlining
recovered 60 of those failures, leaving only the same nine partial-opacity icons
unsupported. The resulting hybrid route covers 3,997/4,006 icons (99.775%). On the
3,937 icons supported by both routes, outlining expands contour count 1.154x and
segment count 2.242x at the median; segment expansion reaches 4.038x at p95 and
13.917x at the maximum.

The structural tails are real rather than percentile-safe: hybrid programs reach 80
contours, 1,211 total segments, and 283 segments in one contour. P64/S128 is therefore
not lossless: it drops 35 contours and 493 segments across eight supported icons,
damaging 30 icon-layer incidences. P96/S384 retains every contour and segment in the
3,997 supported programs, but uses only 0.240% of its 36,864 segment slots. This
falsifies the hypothesis that the census would directly justify a compact dense
lossless rectangle; the tail-safe rectangle is extremely sparse.

Structural success is also not codec losslessness. The semantic route contains 309
literal stroke-width values, six dash patterns, and eight miter-limit values; 344
icons use styles outside the fixture vocabulary. Twenty-two hybrid icons contain 35
coordinate scalars outside 0..72, spanning -6.6875 to 95.0224, so strict q289 clipping
would alter them. All 28 configured palette colors are used and there are no palette
outliers.

**Decision.** Keep semantic-native plus explicit outlined fallback as the leading
route, but do not close Gate C or adopt P96/S384 as the model shape. First add exact
per-path `opacity`, `fill-opacity`, and `stroke-opacity` categories and test the nine
failures. Then measure q289 rendering on the opacity set and selected style/OOB/tail
programs, while evaluating a less wasteful overflow or ragged capacity policy. Preserve
P64/S128 as a falsified lossless budget and P96/S384 as a structural upper bound, not a
final architecture choice.

## 2026-08-29 — Exact opacity recovery probe

**Hypothesis.** A shared categorical vocabulary for per-path element, fill, and stroke
opacity will recover the nine icons rejected by both prior routes without introducing
structural or style approximation. The only remaining loss should be the two previously
observed out-of-bounds moon coordinates.

**Method.** Select exactly the nine `unsupported_both:partial_opacity` rows from the
hash-pinned full-primary census. Normalize both source-semantic and PicoSVG-outlined
forms, then encode each at q289 and P96/S64 with the exact observed opacity, stroke
width, and miter vocabularies. Forbid truncation. Attempt strict encoding first; permit
coordinate clamping only after preserving the strict result. Render every serialized
program against the upstream source at 72 and 18 px. Outputs are create-or-identical
and a second complete invocation verified idempotency.

**Observation.** Both routes normalize all 9/9 icons and all 18 typed programs produce
stable encode/decode/serialize round trips. No contour or segment is dropped, no stroke
width or miter limit is approximated, all colors remain in the 28-entry palette, and
P96/S64 covers the set. Semantic programs remain shorter on this difficult slice:
median contours/segments are 15/64 versus outlined 24/218.

Strict encoding succeeds for 14/18 programs. `1F31A` and `1F31D` each contain one
coordinate scalar outside 0..72 in both representations; the declared projection
clamps exactly one scalar in each affected program. No other safety projection occurs.
Semantic median RGBA MAE is 0.001547 at 72 px and 0.001722 at 18 px, with median 18 px
alpha IoU of 1.0. Outlined medians are 0.002070, 0.002046, and 0.9934 respectively.
Visual inspection of the worst clamped moon (`1F31D`) at 72 and 18 px and the 64-segment
umbrella tail (`26F1`) found no recognizable discrepancy among source, semantic, and
outlined renders.

**Decision.** Accept the three opacity fields and nine-value vocabulary as the leading
codec semantics. They remove the known partial-opacity blocker on its complete pinned
failure set, but do not yet prove a 4,006/4,006 full-primary regression. Keep the two
coordinate clamps explicitly labelled as safety projections; opacity must not be used
to hide the separate OOB-coordinate question. Run an opacity-aware full-primary census
next before closing normalization coverage or choosing the final capacity policy.
