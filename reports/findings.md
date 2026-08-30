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

## 2026-08-29 — Full-primary opacity regression

**Hypothesis.** Adding the exact observed per-path opacity categories will route the
nine former failures through semantic normalization, yielding complete hybrid coverage
without perturbing any non-opacity result or changing the structural tail conclusion.

**Method.** Rerun both normalizers on the unchanged hash-pinned 4,006-icon primary
manifest with schema-v2 explicit opacity vocabulary and the unchanged 30-point P/S
grid. Independently compare all 8,012 attempt rows and all 4,006 hybrid rows with the
v1 census after removing only newly reported opacity fields and the nine formerly
unsupported entries. Verify output hashes, exact row cardinalities, deterministic
ordering, source identity, route coverage, and every capacity loss decomposition.

**Observation.** Semantic normalization now succeeds for 3,946 icons. Its only 60
failures are the unchanged 26 anisotropic stroke transforms, 33 unsupported
presentations, and one same-document resource; outlined fallback recovers all 60.
Hybrid coverage is therefore 4,006/4,006. All 7,994 non-opacity attempt rows and all
3,997 previously supported hybrid rows match v1 exactly after excluding only the new
opacity fields; the 18 old opacity failures are now 18 successes. The nine semantic
opacity icons contain 33 partially opaque layers using only the declared exact
vocabulary.

The full structural result is unchanged except for those nine added programs.
P64/S128 still damages eight icons, dropping 35 contours and 493 segments. P96/S384
is lossless for all 4,006 programs but remains only 0.2403% slot-utilized. The expanded
hybrid set has 24 icons with 37 coordinate scalars outside 0..72, 347 icons outside the
fixture style vocabulary, 311 literal stroke widths, and a maximum of 80 contours,
1,211 total segments, and 283 segments in one path.

**Decision.** Accept opacity-aware semantic normalization plus reason-coded outlined
fallback as complete structural coverage of the reviewed primary dataset. Gate C
remains open: complete normalization is not yet a final categorical codec. First
classify the 37 out-of-bounds scalars by geometric role and measure the q289 safety
projection on all 24 affected icons. Then address exact style categories and the very
sparse lossless capacity rectangle before freezing the representation.

## 2026-08-29 — Out-of-bounds control-handle projection probe

**Hypothesis.** The 37 hybrid coordinates outside 0..72 are off-canvas Bezier control
handles rather than visible endpoints, and clamping them to the viewBox has negligible
render impact relative to the same q289 lattice extended outside the canvas.

**Method.** Select the complete 24-icon OOB set from the hash-pinned opacity census.
Normalize both semantic and outlined forms, classify every excursion by segment type,
coordinate role, axis, side, and distance, then encode at q289 and P48/S64 with exact
fixture styles and no truncation. Compare the actual safe clamped tensor render against
an analysis-only counterfactual using the same quarter-unit lattice without clamping.
The counterfactual is not a model vocabulary. All artifacts are create-or-identical;
a second full invocation reproduced every report hash.

**Observation.** All 37 semantic excursions are cubic control handles: 19 x and 18 y,
22 above and 15 below the viewBox. There are no out-of-range moves or line, quadratic,
or cubic endpoints. The outlined forms contain 27 excursions over 22 icons, also all
cubic controls. Both routes complete all 24 typed round trips with no P/S truncation,
style approximation, palette outlier, or unstable identity.

The negligible-clamp hypothesis is false in the tail. Semantic clamp-only RGBA MAE is
small at the median—0.000114 at 72 px and 0.000277 at 18 px—and seven icons are pixel
identical at 18 px. Five icons exceed 0.001 clamp-only MAE at 18 px, however, and two
exceed 0.01. Clamping `1F4AB` changes the crescent/star silhouette drastically
(18 px MAE 0.03812, alpha IoU 0.8641); clamping the two control handles in
`1F441-FE0F-200D-1F5E8-FE0F` removes most of the eye/speech-bubble interior
(18 px MAE 0.02015). Outlining does not solve the problem: the same two icons remain
the worst outlined clamp cases.

**Decision.** Reject viewBox clamping for Bezier control handles as the primary codec
policy. Keep endpoint coordinates on the 0..72 q289 lattice, but evaluate a distinct
bounded control-coordinate vocabulary before freezing Gate C. The next smallest
candidate is a quarter-unit `[-8, 96]` control lattice (417 values), which covers the
observed -6.6875..95.0224 range while preventing model-generated endpoints from leaving
the canvas. Retain the current clamp as an explicitly labelled fallback projection,
not a lossless normalization rule.

## 2026-08-29 — Role-typed control-coordinate vocabulary

**Hypothesis.** Keeping move/segment endpoints on q289 over 0..72 while assigning only
quadratic/cubic controls to a q417 quarter-unit lattice over [-8,96] will encode the
complete OOB fixture strictly and reproduce the unclamped render evidence without
allowing model-generated endpoints outside the canvas.

**Method.** Add segment-role-aware encoding, decoding, validation, and token bounds to
the typed tensor codec. Hold the 24-icon fixture, both representations, P48/S64, exact
styles, renderer, and q289 endpoint lattice fixed. Disable both truncation and
clamping. Compare every output SVG hash against the corresponding analysis-only
unclamped q289 counterfactual from the parent run. Repeat the full invocation to verify
create-or-identical artifacts.

**Observation.** All 48 semantic/outlined programs are strict-lossless: zero encode
failures, safety projections, dropped contours, dropped segments, style approximations,
palette outliers, or unstable round trips. All 48 SVG hashes exactly match the prior
unclamped quarter-grid counterfactual. Semantic median source MAE is 0.002067 at 72 px
and 0.002752 at 18 px; worst 18 px MAE is 0.005159. The recovered `1F4AB` falls from
the clamped 18 px MAE of 0.03909 to 0.001755, and the eye/speech-bubble case falls from
0.02408 to 0.004100 with alpha IoU restored to 1.0. A second complete run reproduced
all report hashes.

**Decision.** Accept role-typed coordinates as the leading codec: q289 for moves and
segment endpoints over [0,72], q417 for quadratic/cubic controls over [-8,96]. The
control vocabulary adds 128 categories only where segment grammar marks a control
field; legal-token masks must preserve that distinction during corruption and model
prediction. Keep viewBox clamping solely as an explicit fallback for values beyond the
bounded control range. The coordinate blocker is resolved on its complete pinned
failure set; Gate C remains open on style vocabulary and sparse capacity policy.

## 2026-08-29 — Categorical style-vocabulary probe v1

**Hypothesis.** A 48-token stroke-width vocabulary optimized by frequency-weighted
relative L1 error, together with all six observed dash patterns and the five semantic
miter categories, will improve materially on a compact 32-token vocabulary without a
visually meaningful render tail.

**Method.** Derive both width vocabularies deterministically from all 30,222 stroked
contours in the hash-pinned 4,006-row hybrid census. Select a 35-icon render fixture as
the union of each candidate's 12 worst relative and 12 worst absolute width-error
icons, every dashed icon, and every icon containing the three near-10 miter literals.
Hold q289 endpoints, q417 controls over [-8,96], P96/S384, opacity, palette, route, and
renderer fixed. Compare each candidate against an `exact-observed` style control so
the candidate-to-control metric isolates style approximation. Repeat the complete run
and require byte-identical artifacts.

**Observation.** All 105 exact/K32/K48 programs round-trip stably without structural
loss or coordinate projection. K48 reduces the full-corpus worst relative width error
from 22.55% to 9.09%, affected contours from 1,004 to 839, and the fixture's maximum
18 px style-only RGBA MAE from 0.01264 to 0.001698. Its median 18 px style-only MAE is
0.00000908. The all-six dash vocabulary is exact; mapping the eight near-10 miter
contours to 10 produces no leading visual tail.

The negligible-tail hypothesis is nevertheless false at 72 px. `1F4AF` contains five
4.1-width contours, but frequency-weighted K48 maps 4.1 to 4.0. Its style-only RGBA MAE
is 0.01348 at 72 px despite falling to 0.001698 at 18 px. Thus aggregate frequency and
relative-width error alone miss a rare but render-sensitive category. The complete run
reproduced the fixture, metrics, summary, Markdown, and contact-sheet hashes exactly.

**Decision.** Reject unaugmented K48 as the final style vocabulary, while retaining it
as the leading statistical base and K32 as a compact ablation. Test the smallest
render-aware correction next: add exact 4.1 as one explicit sentinel to form K48+1,
using the unchanged fixture and exact-style control. Do not silently tune away this
negative result.

## 2026-08-29 — Render-tail width sentinel

**Hypothesis.** Adding exact width 4.1 as one explicit token to the unchanged K48 base
will remove the only large 72 px style tail without merely moving the error to another
fixture icon.

**Method.** Reuse the hash-pinned 35-icon v1 fixture, exact-style control, hybrid routes,
role-typed coordinates, P96/S384 capacity, palette, opacity, dash, miter, and renderer.
Compare the unmodified 48-token candidate directly with its 49-token union containing
4.1. Repeat the full run and require byte-identical compact and visual artifacts.

**Observation.** All 105 exact/K48/K48+1 programs are stable with no structural loss or
coordinate projection. The sentinel makes `1F4AF` style-exact and reduces the fixture's
maximum style-only RGBA MAE from 0.01348 to 0.002077 at 72 px and from 0.001698 to
0.001510 at 18 px. Median 18 px style-only MAE falls from 0.00000908 to 0.00000303.
No replacement tail appears: the remaining maxima come from different width
approximations, are visually indistinguishable in the inspected contact sheet, and are
more than 6.4x smaller at 72 px than the falsified case. The second invocation
reproduced all report hashes.

**Decision.** Accept K48+1 as the leading stroke-width vocabulary, retain K32 as the
compact ablation, keep all six observed dash patterns exact, and map the three near-10
miter literals to the semantic value 10. This resolves the measured style blocker for
Gate C. The remaining representation question is how to avoid the extremely sparse
P96/S384 dense rectangle while preserving the complete structural tail.

## 2026-08-29 — Packed capacity layout

**Hypothesis.** Packing each icon's segments contiguously while retaining ordered path
lengths will cover the complete corpus at P80/T1216 and reduce the exact worst-case
logical allocation by more than 28x relative to dense P96/S384. Exact nested buckets
should improve typical utilization further without changing program semantics.

**Method.** Recompute loss on every ordered contour-length vector in the hash-pinned
4,006-row hybrid census. Compare dense P64/S128 and P96/S384 with four fixed packed
budgets. Packed truncation retains only a whole-contour prefix so painter order and
contour validity cannot be broken to fit a total-segment budget. Assign every exact
program to the smallest of four explicit packed buckets. Verify row counts, input loss
decomposition, output hashes, and a second create-or-identical invocation.

**Observation.** Corpus maxima are 80 paths, 1,211 total segments, and 283 segments in
one contour. Dense P96/S384 is exact but allocates 36,960 logical path-plus-segment
slots and uses 0.278% on average. Packed P80/T1216 is also exact and allocates 1,296
logical slots, a 28.52x reduction, with 7.92% average utilization. P80/T768 misses only
`E315`, dropping a 12-contour/480-segment suffix, which confirms the single extreme
total-length tail rather than a broad need for dense per-path capacity.

The exact adaptive buckets P32/T128, P48/T256, P64/T512, and P80/T1216 contain
3,359, 553, 88, and 6 icons respectively. They allocate 190.72 logical slots per icon
on average at 53.81% aggregate utilization. All 4,006 assignments are explicit and
hashed. The rerun reproduced every report artifact.

**Decision.** Select semantic-stroke programs with the existing 60-case reason-coded
outlined fallback, role-typed coordinates, K48+1 styles, and packed P80/T1216 capacity.
Use the four nested exact buckets for training efficiency only; bucket identity is not
an unrecorded model semantic. This completes the Gate C representation selection with
explicit tradeoffs. Gate D must implement and fuzz packed conversion, tensor
invariants, safety serialization, and renderer isolation before learning work begins.

## 2026-08-29 — Packed invariant and isolated-render stress

**Hypothesis.** The selected packed layout can reversibly represent diverse legal
programs, reject malformed/corrupted tensors before serialization, and render only the
serializer's bounded XML surface in a resource-limited subprocess, including the full
P80/T1216 outer envelope.

**Method.** Implement reversible dense/packed conversion with path lengths as the sole
segment offsets, validate by expanding through the canonical tensor grammar, and route
serialization through that validator. Add a typed-SVG allowlist followed by a child
renderer with input/output byte ceilings, 512 px size ceiling, CPU/address-space/file/
descriptor limits, a wall timeout, and sanitized exit classification. Generate 200
seeded valid programs; mutate 2,000 packed tensors across ten shape, padding, token,
length, layer, and segment-grammar families; render 12 valid programs at 72/18 px; and
render one synthetic 80-path/1,216-segment outer-bound program.

**Observation.** The first registered invocation failed before artifacts because its
endpoint mutator used token 289 as invalid, although q289 tokens are 1..289. The failed
run is retained. After correcting the mutator to 290 under a new commit/run identity,
all 200 valid programs round-tripped exactly and all 2,000 invalid mutations were
rejected. Rejections span all ten intended violation families. Twenty-four random and
one outer-bound subprocess renders succeeded; the outer program serialized to 15,906
bytes. Malformed XML, an unsafe URL value, and a wrong root/viewBox were rejected with
three explicit classes. The complete corrected run reproduced every artifact hash.

**Decision.** Gate D passes for the selected exposed representation: packed states are
canonical and reversible, invalid states fail closed before XML, serializer output has
a strict allowlist, and rendering is process/resource bounded. Preserve the mutator
failure as a harness-boundary lesson. Begin Gate E with a deterministic CPU tiny-model
proof before requesting billed GPU time; the GPU worker still requires recorded caps
and a verified artifact sink.

## 2026-08-30 — Tiny fixed-topology geometry proof v1

**Hypothesis.** A 241,072-parameter fixed-topology geometry denoiser can overfit one
icon, recover disjoint held-out corruptions for that icon and a four-icon diverse
fixture under predeclared accuracy thresholds, and continue exactly after checkpoint
reload. This is a geometry diagnostic, not a topology or diffusion claim.

**Method.** Pin four semantic-native, in-bounds OpenMoji programs from distinct groups
that fit P16/T128. Corrupt only legal q289 endpoint and q417 control fields at
probability 0.35 while preserving topology and styles. Train eight corruptions of one
icon for 160 full-batch steps and four corruptions each of four icons for 320 steps.
Evaluate on equally sized disjoint corruption draws. Require 99%/95% train/held-out
accuracy for one icon, 98%/85% for four icons, loss ratios at most 0.10/0.20, and exact
160-step checkpoint continuation. Rerun all outputs through create-or-identical guards.

**Observation.** Both cases memorized every training coordinate token and drove the
training loss ratio below 0.0001. Held-out recovery improved far above the untrained
models but missed both predeclared thresholds: one-icon accuracy reached 93.57% rather
than 95%, and diverse-four reached 73.82% rather than 85%. The checkpoint-resumed
diverse run reproduced all 320 loss values and the final model tensors exactly. Visual
inspection agrees with the token metrics: `x_hat_0` recovers coarse palette and
silhouette cues but retains conspicuous misplaced geometry, especially for the apple,
cat, and car at 18 px.

The identical full rerun exposed a separate artifact failure. Legacy `torch.save`
produced different container bytes for the otherwise exact 160-step checkpoint, and
the create-or-identical guard rejected replacement. The first metrics, renders, and
2.9 MB checkpoint are preserved under the failed v1 run identity.

**Decision.** Reject v1 as Gate E exit evidence. It proves training-set learnability
and semantic checkpoint continuation, but not the predeclared held-out recovery or
byte-stable artifact contract. Do not tune the learning thresholds after seeing the
result. Correct only the checkpoint container under a new identity, rerun the unchanged
learning experiment to establish reproducibility, then use the held-out failure to
choose the next smallest scientific ablation.

## 2026-08-30 — Canonical checkpoint correction v2

**Hypothesis.** Replacing only legacy PyTorch checkpoint packaging with a canonical,
pickle-free tensor archive will make the full tiny-learning run byte-reproducible while
leaving every v1 learning result unchanged.

**Method.** Hold the fixture, model, initialization, corruptions, batches, optimizer,
steps, thresholds, renderer, and metrics fixed. Encode model, AdamW, RNG, and step as a
canonically sorted typed JSON tree plus uncompressed `.npy` tensor members in a bounded
ZIP with fixed timestamps and permissions. Reject duplicate, compressed, oversized, or
unexpected members. Run the complete experiment twice through create-or-identical
guards.

**Observation.** Every learning metric, loss-sequence hash, and final model hash exactly
matches v1. The new 3.0 MB checkpoint hash is
`52aee590c65f81f52d49be2626a2398373fe5639c30a3346b1ee30ad5270bcb8`.
Checkpoint continuation is exact, and the second full invocation accepted identical
checkpoint, metrics, render metrics, summary, Markdown, and trajectory bytes. The
artifact correction succeeds without changing the scientific negative result.

**Decision.** Accept the canonical checkpoint format for subsequent local proofs, but
keep Gate E open. V2 is a completed reproducible negative run: fixed-topology geometry
is memorized but does not meet held-out recovery criteria. Before changing capacity,
steps, or corruption, report accuracy separately on actually changed and retained
coordinates; aggregate accuracy may be inflated by coordinates that `x_t` already
reveals unchanged.

## 2026-08-30 — Changed-versus-retained coordinate diagnostic

**Hypothesis.** The diverse-four aggregate held-out score overstates genuine denoising:
accuracy on coordinate tokens actually changed in `x_t` will trail aggregate accuracy
by at least 0.15. No learning input or update is changed from v2.

**Method.** Partition every legal endpoint and control prediction by exact token
comparison between `x_t` and `x_0`. Count replacements that randomly reproduce the clean
token as retained, not changed. Verify that changed and retained counts sum exactly to
the prior aggregate count. Rerun the unchanged 800-step experiment twice through the
canonical artifact guards.

**Observation.** The hypothesis passes narrowly but materially. Diverse-four changed
accuracy is 58.17% (954/1,640), 15.65 percentage points below its 73.82% aggregate.
Retained accuracy is only 82.52% (2,436/2,952), showing that the model also overwrites
many coordinates already correct in `x_t`. The one-icon model generalizes much better:
88.59% changed and 96.11% retained accuracy. All training partitions remain 100%, and
the second full invocation reproduced every compact artifact and checkpoint byte.

**Decision.** The immediate bottleneck is corruption-pattern coverage on the diverse
fixture, not an inability to memorize geometry or resume training. Test deterministic
per-step corruption resampling next while holding batch size, model, corruption
probability, optimizer steps, and held-out draws fixed. This is a smaller and more
diagnostic factor change than increasing model capacity or training duration.

## 2026-08-30 — Per-step corruption coverage treatment

**Hypothesis.** At the same model, batch size, corruption probability, and 320 optimizer
steps, deterministic per-step corruption resampling will raise diverse-four held-out
changed-token accuracy to at least 75% and retained-token accuracy to at least 90%.

**Method.** Keep the one-icon case as the unchanged static control. For diverse-four,
derive each 16-example batch from the global optimizer step, exposing 1,280 deterministic
corruptions per icon over the continuous run without changing compute shape. Use the
same global-step mapping before and after the step-160 resume boundary. Hold held-out
draws and every other learning/render parameter fixed. Repeat the complete run.

**Observation.** The treatment strongly passes. Diverse-four aggregate accuracy rises
from 73.82% to 98.56%, changed-token accuracy from 58.17% to 96.65%, and retained-token
accuracy from 82.52% to 99.63%. Training-probe accuracy is 98.52%. The complete loss
sequence and model are exact across checkpoint continuation, and all artifacts reproduce
byte-for-byte. Visual inspection shows immediately recognizable face, apple, cat, and
car reconstructions at both 72 and 18 px, whereas v3 retained conspicuous geometry
damage. The unchanged one-icon held-out control remains 93.57% against its 95% threshold.

**Decision.** Corruption coverage, not capacity or step count, caused the diverse v3
failure. Accept deterministic per-step resampling for this fixed-topology diagnostic.
Do not close Gate E yet because the generic combined flag retains the one-icon held-out
miss; apply the same factor to one-icon under a final controlled config. This does not
yet answer topology learning or choose the final corruption family.

## 2026-08-30 — All-resampled control and Gate E synthesis

**Hypothesis.** Enabling the already validated per-step resampling treatment for
one-icon will make both v1 predeclared case criteria pass in the same run while
preserving diverse recovery, exact resume, renders, and artifact identity.

**Method.** Change only one-icon `resample_each_step` from false to true. Keep its eight
example batch, 160 steps, seed, held-out draws, and thresholds fixed; keep the entire
diverse v4 branch unchanged. Repeat the full run and inspect the existing paired render
sheet, whose diverse branch should remain byte-identical.

**Observation.** One-icon held-out accuracy rises from 93.57% to 99.26%, with 98.37%
accuracy on changed tokens and 99.72% on retained tokens. Diverse-four remains exactly
98.56% aggregate, 96.65% changed, and 99.63% retained. The render sheet is byte-identical
to v4 and remains recognizable at both sizes. Resume and all artifacts reproduce.

The literal v5 hypothesis is nevertheless false: the fixed one-icon probe scores
97.73%, below the inherited 99% `min_train_accuracy`. Under resampling this probe is an
unseen corruption batch, not a training batch, so it no longer measures memorization.
The threshold result is retained as false rather than renamed after inspection.

**Decision.** Close Gate E from the controlled evidence sequence, not by overriding the
v5 flag. V1 demonstrates exact one-icon memorization at 100%; v5 demonstrates 99.26%
one-icon held-out recovery; v4/v5 demonstrate 98.56% diverse held-out recovery and
recognizable renders; v2-v5 demonstrate exact checkpoint continuation and byte-stable
artifacts. The decisive learning lesson is that fixed corruption examples caused
memorization and poor recovery, while deterministic online coverage solved the tiny
geometry task without more capacity or steps. This remains fixed-topology geometry-only
evidence, not support for topology generation or a final diffusion formulation. Begin
Gate F with matched corruption-family definitions and tiny comparisons.

## 2026-08-30 — Gate F factorized fixed-topology control

**Hypothesis.** Exact role-uniform factorized geometry corruption, with every opened
gate forced to a different legal token, provides a reproducible fixed-topology baseline
for a matched path-correlated comparison.

**Method.** Hold the four-icon fixture, packed codec, 241,072-parameter bidirectional
model, optimizer, seeds, 0.35 marginal corruption probability, online resampling,
320/160-step budgets, checkpoint boundary, held-out draws, and rendering protocol fixed.
Independently gate each legal endpoint/control token and sample from its q289/q417
vocabulary excluding the clean token. Paths, segment types, styles, and padding remain
unchanged. Repeat the complete local CPU run through byte-identity guards.

**Observation.** Both executions are byte-identical, including the canonical diverse
checkpoint and every compact report artifact. Held-out one-icon/diverse-four aggregate
accuracy is 99.33%/98.76%; changed-token accuracy is 98.73%/96.96% and retained-token
accuracy is 99.63%/99.76%. The checkpoint continuation is exact. Paired 72 and 18 px
renders are safe and recognizable for face, apple, cat, and car; the raw factorized
`x_t` states are visibly static-like as expected. The one-icon online training probe is
97.98%, below its inherited 99% threshold because online resampling makes that probe an
unseen batch; this literal threshold miss is retained.

**Decision.** Accept this as the matched factorized control, not as a Gate F choice. Run
the path-correlated geometry treatment next with the same marginal path-field change
rate, data, model, optimizer, steps, and evaluation protocol. The current evidence is
still fixed-topology geometry-only and is not a topology-generation or D3PM result.

## 2026-08-30 — Gate F path-correlated fixed-topology treatment

**Hypothesis.** Sharing one corruption gate across each active path, while preserving
the same 0.35 marginal legal-token corruption rate as the factorized control, learns
faster or recovers changed geometry more effectively.

**Method.** Change only the geometry corruption contract. For each active path, open one
Bernoulli gate and, when open, replace all legal start/control/endpoint tokens uniformly
excluding their clean value. Preserve topology, styles, padding, fixture, model,
optimizer, seeds, online resampling, steps, held-out draws, checkpoint boundary, and
rendering. Repeat the entire local CPU run under canonical artifact guards.

**Observation.** The rerun is byte-identical and checkpoint continuation is exact. The
treatment retains safe recognizable reconstructions, but it does not improve the
factorized control: one-icon held-out/changed accuracy is 95.53%/83.56% versus
99.33%/98.73%; diverse-four is 98.11%/93.86% versus 98.76%/96.96%. Retained-token
accuracy is 100% in both treatment cases, consistent with most paths being entirely
retained, not evidence of stronger restoration.

**Decision.** On this tiny fixed-topology fixture, the path-correlated treatment is a
reproducible negative result: it trails factorized corruption in changed-token recovery
and does not show a trajectory-quality advantage. Do not yet choose a final process or
generalize to topology/style learning. First measure compatible-path availability for
the predeclared whole-path donor ablation; if sparse, create a pinned fixture where that
ablation has real support rather than interpreting mostly retained paths.

## 2026-08-30 — Whole-path donor support audit

**Hypothesis.** The four-icon Gate F fixture contains enough exact segment-signature
matches across different icons to apply external whole-path donor replacement at the
same 35% marginal legal-geometry corruption level as the other arms.

**Method.** Enumerate every active packed path, derive its exact sequence of segment
kinds, and count a path only when an identically typed path occurs in another fixture
icon. Count its start plus legal coordinate fields as eligible. No corruption, model,
or source asset is altered. Repeat the bounded audit and require byte-identical output.

**Observation.** The fixture contains 41 active paths and 1,148 legal geometry fields.
Only 14 paths (34.15%) and 284 fields (24.74%) have an external exact-signature donor.
Thus even a gate probability of 1.0 can change at most 24.74% of fields before accounting
for a donor token that happens to equal the clean token. The repeat is byte-identical.

**Decision.** Falsify feasibility of a matched 35% whole-path ablation on this fixture;
do not lower the control corruption level or interpret a sparsely active treatment as a
comparison. Build a bounded full-primary path-signature census and select a separate
pinned compatible-path fixture before testing whole-path replacement.
