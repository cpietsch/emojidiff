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

## 2026-08-30 — Full-primary path-signature census

**Hypothesis.** The reviewed primary corpus has sufficient exact external donor support
to form a bounded, fair whole-path replacement fixture after the four-icon fixture
failed its support audit.

**Method.** Hash-check and normalize every one of the 4,006 hybrid-selected source
programs using its recorded semantic or outlined route. Enumerate each contour's exact
segment-kind sequence and count a geometry field as donor-compatible only if an identical
sequence occurs in another icon. Repeat the full census through create-or-identical
artifact guards.

**Observation.** The corpus contains 2,532 exact signatures, of which 1,068 occur in at
least two icons. External donors cover 1,488,688 of 1,641,526 legal geometry fields
(90.69%), so a matched whole-path process is feasible at corpus scale. The highest
coverage candidates are not immediately suitable for the tiny comparison: many are
skin-tone or gender family variants, and several exceed the current P16/T128 diagnostic
capacity.

**Decision.** Accept corpus-scale donor feasibility but do not choose an ablation fixture
from raw coverage rankings. Extend the census/selection output with capacity and
family-aware constraints, then pin a diverse compatible fixture before training. This
retains the small-fixture feasibility failure and avoids family leakage.

## 2026-08-30 — Capacity-safe whole-path treatment

**Hypothesis.** Exact compatible external whole-path donor replacement is learnable in
a family-distinct, P16/S32/T128-bounded fixed-topology fixture.

**Method.** The first selected fixture is retained as a harness-boundary failure: its
39-segment path cannot be encoded by the declared S32 diagnostic. The selector was then
restricted by per-path segment capacity and repinned four semantic, family-distinct
icons (`2728`, `1F92F`, `1F953`, `E0C3`), with 72.76–100% per-icon eligible geometry
coverage. Train only the predeclared diverse-four arm with the same model, optimizer,
0.35 path gate, online resampling, 320/160-step checkpoint schedule, and safe render
checks. This fixture is distinct from the earlier factorized/path-correlated fixture.

**Observation.** The corrected local CPU run passes its criteria: held-out aggregate,
changed-token, and retained-token accuracy are all 1.0 (550 changed legal fields over
16 held-out examples); final loss is 0.001014. Checkpoint continuation is exact and the
paired render artifacts pass their safety checks.

**Decision.** Preserve the successful feasibility result and the capacity-selection
failure. Do not compare this 1.0 result numerically with the earlier factorized or
path-correlated arms because their fixtures differ. Run matched controls on this exact
v3 fixture before drawing a corruption-family conclusion.

## 2026-08-31 — Gate F v3 matched path-correlated control

**Hypothesis.** On the capacity-safe v3 fixture, path-correlated geometry corruption
will match or exceed the factorized control's changed-token recovery while retaining
safe fixed-topology renders.

**Method.** Hold the four v3 icons, P16/S32/T128 packed codec, 241,072-parameter
bidirectional model, optimizer, seed 1702, online resampling, 0.35 corruption
probability, 320/160-step schedule, held-out protocol, and renderer fixed. Change only
the geometry corruption family to one gate shared by every legal field in each active
path. Run locally on CPU and repeat through create-or-identical artifact guards.

**Observation.** The two authoritative executions are byte-identical: summary,
metrics, render metrics, and checkpoint SHA-256 values all match. The path-correlated
arm passes its predeclared gate and exact checkpoint continuation, with held-out
aggregate/changed/retained recovery of 92.30%/80.55%/100% over 3,732 changed fields.
The fixture-matched factorized control reaches 96.77%/92.62%/99.06% over 3,345 changed
fields. Thus path-correlated recovery trails factorized by 12.07 percentage points on
the changed-token metric in this single-seed comparison. The whole-path arm reaches
100%, but with only 550 changed fields, so its score demonstrates a feasible, much
lighter corruption task rather than a comparable win.

**Decision.** Retain this negative result: path correlation does not show the expected
advantage on this tiny fixed-topology v3 fixture. Do not select a primary family from
one seed. Commit the compact evidence and add a predeclared additional-seed replication
across all three arms before deciding whether the observed factorized advantage is
robust. This remains a custom iterative denoiser result, not an exact D3PM claim or
evidence for topology generation.

## 2026-08-31 — Gate F independent-seed replication and primary choice

**Hypothesis.** The factorized changed-token advantage over path-correlated corruption
on the capacity-safe v3 fixture persists under the predeclared independent seed, while
whole-path replacement remains a supported but lighter feasibility task.

**Method.** Change only the training/corruption seed from 1702 to 2701 in all three v3
arms. Keep the fixture, packed P16/S32/T128 codec, 241,072-parameter model, optimizer,
online resampling, 0.35 gate probability, 320/160-step schedule, held-out protocol, and
renderer fixed. Repeat all three complete studies through create-or-identical guards.

**Observation.** Every arm passes its predeclared criteria, resumes exactly, renders
safely, and reproduces byte-identically. Factorized held-out aggregate/changed/retained
accuracy is 97.15%/93.47%/99.16%; path-correlated is 92.72%/80.87%/100%. The 12.60-point
factorized changed-token advantage agrees with the first seed's 12.07-point advantage.
Whole-path again reaches 100%, but only 608 held-out fields change versus 3,337 for
factorized and 3,586 for path-correlated.

**Decision.** Close Gate F with factorized role-uniform corruption as the Gate G primary.
It wins changed-token recovery on both predeclared seeds while preserving render safety
and exact recovery. Retain path-correlated and whole-path as named ablations: the former
is a reproducible negative result on this diagnostic, and the latter proves compatible
donor learning but is not a matched-difficulty quality win. This choice applies only to
the current custom iterative-denoiser formulation; it is not an exact D3PM claim and
does not yet establish topology or style generation.

## 2026-09-20 — Gate G dominant-bucket CPU pipeline smoke

**Hypothesis.** The selected factorized formulation can move from a four-icon fixture to
the dominant exact OpenMoji packed bucket without breaking provenance joins, selected
semantic/outlined normalization, structured conditioning, canonical checkpointing, or
locked-path editing semantics.

**Method.** Join the hash-pinned 4,006-row hybrid ledger to the exact adaptive-capacity
assignments, audit split-family isolation, and select the P32/T128 bucket. Deterministically
sample four train and two validation icons, including their selected normalization
routes, then execute one CPU training step with group/subgroup conditioning. Save and
restore the canonical checkpoint and run a path-locked corruption/prediction check. The
complete invocation was repeated through create-or-identical artifact guards.

**Observation.** The join is one-to-one. The bucket contains 3,359 icons: 2,681 train,
339 validation, and 339 test, with no variant family crossing splits. The 577,552-parameter
pipeline completes, its 7,075,309-byte checkpoint restores exactly, the locked path is
unchanged, and every compact artifact repeats byte-identically. First-step train and
validation accuracy are near random, as expected; they are not learning evidence. The
run was registered retroactively after the omission was detected and retains its base
commit plus dirty-source hash.

**Decision.** The local data/model/checkpoint/editing path is ready for a pipeline-specific
GPU smoke, but Gate G is not otherwise passed. This pilot is fixed-topology and
geometry-only. It neither demonstrates unconditional generation nor tests whether
conditioning improves quality. Add a bounded owned-worker pipeline-smoke adapter that
stages only the six selected SVGs plus the palette under verified hashes, then execute
the same config for one step on the RTX 4080 before defining a larger training run.

## 2026-09-20 — Gate G dominant-bucket GPU pipeline smoke on the owned RTX 4080

**Hypothesis.** The exact locally verified Gate G pipeline runs unchanged on the owned
RTX 4080: it loads the real dominant-bucket data through its staged immutable snapshot,
completes one CUDA optimizer step under declared deterministic algorithms, restores a
canonical checkpoint in the persistent artifact volume, and keeps locked paths exact.

**Method.** Commit the code, build a `git archive` snapshot extended with only the six
deterministically selected raw SVGs and the pinned palette, register the run before
execution, stage it under verified archive and extracted-tree hashes, and run the
bounded pipeline smoke in the pinned GPU image with no network, a read-only root
filesystem, dropped capabilities, and only the two Compose-prefixed named volumes
mounted at run-specific subpaths. Repeat the identical invocation to test
create-or-identical behavior.

**Observation.** Three registered attempts were needed, and each failure was preserved
under its own immutable identity rather than retried in place.

1. `openmoji-g1-gpu-cd3250e-0bafd5c-9b9b1699` verified its stage and failed before model
   construction: the staged pilot resolved relative input paths against the image
   working directory, so the pinned palette was missing. No GPU step, no artifacts.
2. `openmoji-g1-gpu-e7dc920-0bafd5c-9b9b1699` loaded the config, data, normalizers,
   model, and optimizer, then was rejected inside the first CUDA step because
   `torch.use_deterministic_algorithms(True)` requires `CUBLAS_WORKSPACE_CONFIG` on
   CUDA >= 10.2. Artifact directory empty. The correction sets that variable explicitly
   in the owned Docker launcher, which makes the declared determinism achievable rather
   than relaxing it.
3. `openmoji-g1-gpu-215bcb8-0bafd5c-9b9b1699` completed the GPU step, wrote a durable
   checkpoint, and passed its round trip and locked-path checks, but failed the adapter
   result contract because the staged wrapper digested `summary.json` from the output
   root instead of the pilot's report root. Its artifacts remain intact on the worker
   and are listed in its `artifacts.json`; nothing was deleted.

`openmoji-g1-gpu-7ba1aa4-0bafd5c-9b9b1699` then completed. `device` is `cuda` with
`deterministic_algorithms` true; train loss is 15.216644 with 0.005068 token accuracy at
step 1; validation loss is 15.344566 with 0.009404 aggregate, 0.008850 changed over 226
changed fields, and 0.009709 retained over 412 retained fields; the model has 577,552
parameters over the 3,359-icon `bucket-p32-t128`; the 7,075,309-byte checkpoint
round-trips and hashes
`4b265e5575e3aa455a0d427e340ec407eaaaf39222709305d060281ae7e453f9`; locked paths are
exact. A second identical invocation returned the same result, and both digests match
attempt 3 exactly, so the result reproduces across separate containers rather than once.
The local CPU pilot checkpoint has the same byte count but a different digest, which is
the expected CPU/GPU floating-point difference and was never part of this contract.

**Decision.** The selected representation, provenance join, conditioning, factorized
corruption, optimizer, canonical checkpoint, and locked-edit path all execute correctly
on the target GPU. First-step accuracies are near random by construction and are not
learning evidence, so Gate G is not passed. The three preserved failures also expose a
real gap: `scripts/remote/` wrapper code had no test coverage, and two of the three
failures were wrapper defects rather than research defects. A CPU regression test now
pins the wrapper to the artifact layout the pilot actually writes. The next step is to
define a bounded dominant-bucket training run with predeclared thresholds, since the
pipeline itself is no longer the open question.

## 2026-09-20 — Gate G dominant-bucket training v1: learnable but overfits 256 icons

**Hypothesis.** Predeclared before launch in
`runs/openmoji-g1-train-ec2436b-f2a06ca5-9b9b1699/run.yaml`. With the selected packed
representation, factorized role-uniform geometry corruption, and group/subgroup
conditioning, a 577,552-parameter denoiser trained on a 256-icon family-disjoint
subsample of the dominant P32/T128 bucket would recover corrupted geometry fields on 128
held-out icons at least 10x its untrained same-input control, preserve at least 0.90 of
already-correct fields, improve monotonically in at least 8 of 10 trace intervals, keep
locked paths exact, and reproduce identically.

**Method.** 600 bounded steps at batch 16, learning rate 0.001, corruption probability
0.35, seed 3101, on the owned RTX 4080 under declared deterministic algorithms. Held-out
evaluation every 60 steps plus an untrained step-0 control on the identical corruption
draw. The tracing is metric-only: `_evaluate` draws from an independent numpy generator
under `no_grad` and the denoiser has no dropout, and a test asserts that a traced and an
untraced run of the same config produce byte-identical checkpoints. A 60-step CPU
preflight preceded the run to size memory and wall time; it is disclosed in the run
record because it informed the thresholds.

**Observation.** The run completed in about 22 seconds and reproduced identically.
Held-out changed-token accuracy rose from 0.003005 untrained to 0.055230, which is
18.38x the control, monotone in 9 of 10 intervals. Held-out retained-token accuracy
reached only 0.325586 against the predeclared 0.90. Held-out loss reached its minimum of
9.7932 at step 240 and then rose steadily to 11.1485, while training token accuracy
climbed to 0.4550 and training loss fell to 2.7221; held-out aggregate accuracy was flat
near 0.231 from step 240. Locked-path exactness and the canonical checkpoint round trip
both held. Totals were 13,978 changed and 26,030 retained held-out fields.

One nuance is recorded rather than smoothed over: held-out changed accuracy continued to
rise across the same interval in which held-out loss worsened, so the monotone criterion
passed while generalization was already degrading. Held-out loss is the more honest
scalar for this comparison, and the monotone criterion should not be reused unqualified.

**Decision.** Predeclared outcome: **falsified**, retained as a negative result rather
than retuned. Two claims survive. The representation is genuinely learnable at 256
diverse icons spanning 10 groups and 75 subgroups, well beyond the 4-icon Gate F
fixtures. And the failure is a data-scale failure, not an optimization or plumbing
failure: the model overfits after roughly 240 steps. The next controlled run should
change only the train-split size, moving to the full 2,681-icon family-disjoint split,
while holding model, seed, corruption, batch, and learning rate fixed, and should select
on held-out loss instead of a fixed step budget. Loading that split costs about three
minutes of CPU at the measured 60 ms per icon, so it remains a bounded run.


## 2026-09-20 — Development moved to gpubox-4080, with a matched environment control

**Hypothesis.** Moving development and execution off the `gtc` control plane onto the
owned RTX 4080 changes the numerical environment from torch 2.8.0a0+5228986c39.nv25.06
with CUDA 12.9, inside a pinned container, to torch 2.14.0a0+4fdf77b940.nv26.08 with
CUDA 13.4, natively. Results obtained before and after the move are therefore not
comparable by assumption, and a matched control is required before the next experiment.

**Method.** Transferred the canonical worktree, full Git history, the 405 MB immutable
raw checkout, reports, and the run registry. Verified the transfer by comparing HEAD,
the tracked tree hash, the 4,495-file count, the read-only mode of the raw checkout, and
an aggregate SHA-256 over every raw file. Built a native virtual environment inheriting
the system NGC torch, installed the project dependencies and the missing Cairo runtime,
and ran the full suite, Ruff, and strict mypy. Then re-executed the **identical** v1
config, `f2a06ca5…`, natively and compared it field by field with the container run.

**Observation.** Every integrity check matched exactly, including the raw aggregate hash
`fe76333c011104a4523635b3946fdc348a3aba8a1f5fb94bf67513938d0a99c4`. The suite passes
108/108 on the box. The native control reproduces the container run closely enough that
no conclusion changes:

| metric | container 2.8 / 12.9 | native 2.14 / 13.4 |
| --- | ---: | ---: |
| final train token accuracy | 0.455046275892 | 0.455046275892 |
| held-out changed accuracy | 0.055229646587 | 0.055229646587 |
| held-out aggregate accuracy | 0.231128774245 | 0.231153769246 |
| held-out retained accuracy | 0.325585862466 | 0.325624279677 |
| held-out loss | 11.148523330688 | 11.148344039917 |

Final train token accuracy and held-out changed-token accuracy are bit-identical.
Exactly one retained token of 26,030 differs. Checkpoint bytes differ, as expected
across framework versions; no cross-environment artifact identity was claimed.

**Decision.** The migration is accepted, and
`openmoji-g1-train-v1-native-c9bf1b9-f2a06ca5-9b9b1699` replaces the container run as
the baseline for the next data-scale experiment. `AGENTS.md` was rewritten for the new
single-machine topology: the strict authorization and immutable-staging rules now apply
only to rented or shared machines, which is what they were designed for, while the run
registry, predeclared criteria, data immutability, and experimental discipline are kept.
The `gtc` copy is retained untouched as a backup.

Two data ideas were measured during this session and deliberately queued rather than
applied, because the corpus already holds ten times the data v1 used. Left-right
mirroring is exactly representable in token space: the quarter-unit lattice is closed
under x to 72-x, endpoint token t maps to 290-t for all 289 bins, and observed control-x
values span only 2.00 to 70.00 across 35,483 samples, so none overflow the -8 to 96
control vocabulary. Separately, content occupies a median 0.753 of the 72-unit box, with
a median bounding box of x [12.00, 60.00] and y [10.62, 61.00] and only 0.5% of icons
using more than 95% of the box, so rescaling would recover roughly 1.33x coordinate
resolution. Coordinate precision is not currently the limiting factor, so neither change
is justified before the full-split run.

## 2026-09-20 — Gate G data-scale v2: the gap closes, the bar does not

**Hypothesis.** The v1 falsification was data-limited, not an optimization or plumbing
failure. Training the identical 577,552-parameter denoiser on the full 2,681-icon
family-disjoint split instead of a 256-icon subsample would lower the held-out loss
minimum, narrow the train/held-out gap, and raise changed-token recovery by at least
1.5x, all read at the checkpoint selected by held-out loss.

**Observation.** Run `openmoji-g1-train-v2-datascale-a50b2e0-c474c94d-9b9b1699` early
stopped at step 1,320 and selected step 840. Against v1 read at its own held-out
optimum, step 240, held-out loss falls 25.1% from 9.7932 to 7.3333, aggregate accuracy
rises 38.0%, retained accuracy 38.6%, and the train-minus-held-out gap narrows 39.7%
from 0.1563 to 0.0942 while training accuracy moves only 4.8%. That is a data-scale
effect, not an optimization one. But changed-token recovery reaches only 0.0573, a
1.307x improvement against a predeclared 1.5x, so the primary recovery criterion is
falsified. The standing 0.90 retained-preservation bar improves from 0.3256 to 0.4128
and remains far out of reach. Structural safety and byte-identical reproducibility both
hold.

The informative part is a disagreement between two scalars. Held-out loss reaches its
minimum at step 840 and never recovers, while held-out accuracy — aggregate, changed,
and retained alike — rises monotonically through the final step 1,320, where
changed-token accuracy is 0.0662, a 1.511x ratio that *would* have met the bar. The
criterion was evaluated exactly as predeclared, at the selected checkpoint; reading it
at the final step after the fact would be picking the stopping rule that passes.

**Decision.** Record v2 as partially falsified and keep it. Data volume is no longer the
binding constraint at this model size: ten times the data bought a large generalization
improvement but sub-proportional changed-field recovery, which points at model capacity,
the corruption schedule, or the single-shot prediction objective instead. Treat the
loss/accuracy disagreement as its own question rather than resolving it by convenience:
after v1 this project declared held-out loss the more honest scalar, and v2 shows that
choice costs 15.6% of the relative changed-token recovery available at the cap. The next
run must declare which scalar governs selection, and why, before it starts.

## 2026-09-20 — The Gate G scalars oversell the picture, and 0.35 may be the wrong regime

**Hypothesis.** Descriptive rather than predictive: at the v2 selected checkpoint's
0.2886 held-out aggregate and 0.0573 changed-token accuracy, what does the predicted
clean state actually look like, and how much does it improve the render over the raw
corrupted state the model was given?

**Observation.** Run `openmoji-g1-train-v2-renders-eff0038-9573bc74-9b9b1699` restored
the hash-verified v2 checkpoint and rendered `x_0`, `x_t`, and `x_hat_0` for twelve
held-out icons under the pilot's own held-out corruption draw. Median RGBA error against
the clean render falls from 0.169546 for `x_t` to 0.142433 for `x_hat_0` at 72 px, a
16.0% reduction, and from 0.182875 to 0.152770 at 18 px. None of the twelve predictions
is a recognizable icon; both `x_t` and `x_hat_0` read as scribble at both sizes. Because
this corruption touches only geometry, fills and topology survive, so several icons keep
a correct palette — `26A0` stays yellow, `1F943` orange, `1F199` green — while the stroke
geometry is destroyed in every case. The rerun reproduced every artifact hash.

**Decision.** Two corrections. First, stop treating Gate G token accuracy as a proxy for
visual recovery: a 16% reduction in render error is not a recovered icon, and no future
Gate G recovery claim should be recorded without a render beside it. Second, and more
consequential, `x_t` at corruption probability 0.35 is already visually destroyed, so
the model is trained and evaluated in a regime where the visual task may be unachievable
at any model size. That probability has been fixed since the Gate F four-icon fixtures
and has never been varied at corpus scale. Promote the corruption schedule to a
first-class candidate factor alongside model capacity, and consider training across a
range of corruption levels rather than at one fixed point — which is also what a
denoiser facing many corruption levels at sampling time would need.

## 2026-09-20 — The checkpoint-selection scalar is a convention, not a finding

**Hypothesis.** v2 left held-out loss and held-out accuracy disagreeing about when to
stop, with its primary criterion passing under one reading and failing under the other.
Because `predict_clean_geometry` decodes by argmax, the artifact depends only on which
token wins and not on the probability mass cross-entropy measures, so the strictly more
accurate step-1,320 checkpoint should render closer to `x_0` than the loss-selected
step-840 one.

**Observation.** The fixed-budget 1,320-step arm reproduces v2's first 1,320 metric rows
exactly and its held-out block matches v2's recorded final step, so both arms are one
trajectory; `x_t` render error is identical across arms, so both saw identical inputs.
The hypothesis is false. On twelve icons the accuracy-selected checkpoint rendered
*worse* by median (0.151687 against 0.142433 at 72 px) despite higher aggregate, changed
and retained accuracy. That run also carried a design defect: comparing each arm's
median independently is the wrong test for paired data, and the per-icon difference was
approximately zero (p = 0.774).

The powered 128-icon rerun settles the magnitude. The mean paired difference is
-0.001638 at 72 px with a 95% interval of -0.004867 to +0.001590, and -0.001392 at 18 px;
the sign test is 65/128 (p = 0.930) and 61/128 (p = 0.659). The scalars are not
separable, but any true difference is now bounded below about 0.0032 RGBA MAE against a
median render error near 0.14 — at most 2.3% of the error already present. The medians
confirmed the defect diagnosis by flipping direction between the two samples: they
favoured the loss arm by 6.5% at twelve icons and the accuracy arm by 4.7% at 128, while
the paired difference stayed near zero in both.

**Decision.** By the decision rule committed before the numbers were read: held-out loss
is the project's checkpoint-selection scalar, chosen by convention as the standard
early-stopping signal and the rule v1 and v2 already used, with the record stating
plainly that it is immaterial for render quality at this stage. Two consequences. v2's
changed-token criterion remains correctly recorded as falsified, and reading it at the
final step would not have been justified by render quality either, because render
quality does not distinguish the checkpoints. And token accuracy is not a stand-in for
render quality even under argmax decoding, which reinforces the render-probe rule: no
Gate G recovery claim without a render beside it. The conclusion is conditional on
argmax decoding; a sampler that draws from the distribution reopens the calibration
question.

## 2026-09-20 — Capacity is not the constraint either; the corruption schedule is what is left

**Hypothesis.** v2 left the 577,552-parameter denoiser with a train/held-out gap of only
0.0942, suggesting it was close to fitting what it could express. At 3.53x the
parameters — 2,040,976, scaling width and depth together with head dimension and the
feedforward ratio held constant — the same denoiser on the same 2,681-icon split should
reach a lower held-out loss, recover at least 1.25x the changed tokens, and render
measurably closer to `x_0`.

**Observation.** Run `openmoji-g1-capacity-v3-3c252d5-ee32665b-9b9b1699` falsified all
three. Held-out loss at the selected checkpoint is 7.3790 against v2's 7.3333 — 0.6%
*worse*, not better. Changed-token recovery is 0.0639, a 1.115x improvement against a
1.25x bar. On the complete 128-icon held-out draw with identical corruption seeds and a
verified identical `x_t` control, the mean paired render difference is +0.001853 with a
95% interval of -0.0026 to +0.0063; the selection-scalar comparison had measured this
interval's half-width at 0.0032 on the same draw, so an effect above roughly 0.0064
would have been detected. Structural safety and byte-identical reproducibility hold.

The shape of the failure is the informative part. The larger model reached essentially
the same held-out loss floor and reached it in 360 steps instead of 840, then
overfitted. More capacity bought faster fitting of the same ceiling, not a lower one.
Both models converge to a floor near 7.35, which looks like a property of the regime
rather than of either model.

**Decision.** Two of three candidate factors are now eliminated on matched, predeclared
comparisons: data volume by v2, model capacity by this run. The remaining candidate is
the corruption schedule, which is where the render probe already pointed: `x_t` at
probability 0.35 is visually destroyed, a third of the geometry is simply gone, and no
model or dataset recovers information that is not there. That probability was chosen for
the Gate F four-icon fixtures and never revisited at corpus scale. The next experiment
varies it, preferably training across a range of levels rather than one fixed point,
which is also what a denoiser facing many levels at sampling time would need.

Worth recording separately: this run's changed-token recovery climbs from 0.0639 at the
loss minimum to 0.0831 by step 840, a 30% relative gain entirely past the point where
held-out loss stopped improving, and 0.0831 would have cleared the criterion. It is
reported and not gated because the selection rule was settled before this run existed.
That is precisely the post-hoc freedom the scalar comparison was run to remove, and it
would have been available here.

## 2026-09-20 — The Gate G denoiser is barely reading its input

**Hypothesis.** Two mechanisms were registered in tension before launch. The *regime*
hypothesis: corruption probability 0.35 destroys information no model can recover, so
one fixed checkpoint evaluated at lower corruption should close a much larger fraction
of the render gap. The *damage* hypothesis: held-out retained-token accuracy is only
0.4128, so the model overwrites about 59% of already-correct fields, and at low
corruption there is more correct material to damage, so recovery could go negative. The
regime hypothesis was the registered prediction.

**Observation.** Run `openmoji-g1-corruption-sweep-71a080a-4levels-9b9b1699` evaluated
v2's loss-selected checkpoint at four corruption levels with identical icons and seeds.
Mean per-icon recovery fraction at 72 px is -4.6517 at p=0.05, -1.2024 at p=0.10,
-0.1315 at p=0.20 and +0.1800 at the trained p=0.35 — strictly increasing in p, the
opposite of the prediction. At p=0.05 the model makes its input 2.8x worse and helps on
zero of thirty-two icons. All three primary criteria are falsified; the damage
hypothesis is confirmed.

The mechanism is visible in one comparison. Across the sweep, mean `x_t` render error
moves by a factor of 3.5, from 0.05098 to 0.17958, while mean `x_hat_0` error moves
about 6%, from 0.13822 to 0.14651. Per icon, predictions at p=0.05 and p=0.35 correlate
at Pearson r = 0.91. The model emits nearly the same reconstruction regardless of how
corrupted its input is. The p=0.05 contact sheet shows it plainly: `x_t` is almost
indistinguishable from `x_0` — legible WC signs, a solid hexagon, an airplane, a
mushroom — and `x_hat_0` is scribble in every row.

`GeometryDenoiser` takes corrupted tokens plus group and subgroup embeddings and nothing
else. There is no timestep or noise-level input and no mask marking which fields were
replaced, and it trained at a single fixed probability, so nothing tells it how much to
trust what it is given.

**Decision.** This reinterprets the whole Gate G sequence, though it invalidates none of
its measurements. The +18% recovery at p=0.35 is a fixed-quality output beating a badly
corrupted baseline, not recovered geometry. Retained-token accuracy of 0.4128 was never
a secondary weakness — it was this finding, visible in token space since v1, and the
standing 0.90 bar has been measuring the real problem all along. The held-out loss floor
near 7.35 that v2 and a 3.53x larger v3 both reach is consistent with both learning the
same prior, which is why capacity did not help. v1 and v2 learned a group-conditioned
prior over icon geometry, not a conditional denoiser.

The next experiment is therefore not a corruption schedule. Give the model a
noise-level or timestep embedding and train across a range of levels, so it can modulate
how much to trust its input; make copying cheap through an edit mask or a residual
against `x_t`, so leaving a correct field alone is the default; and, as a diagnostic
upper bound only, condition on which fields were corrupted, to separate inability to
identify corrupted fields from inability to predict their values.

## 2026-09-20 — The encoder could not tell which coordinate was which

**Hypothesis.** The denoiser's output was near-independent of its input. The
informational explanation was that it trained at one fixed corruption level and was
never told the level, so it could not learn how much to trust `x_t`.

**Observation.** Run `openmoji-g1-noise-conditioned-v4-6b935b2-5a9eaf44-9b9b1699`
sampled the corruption level per example over 0.05 to 0.50 and conditioned the model on
it. All three primary criteria are falsified: recovery at 0.10 is -1.0385 against v2's
-1.2024, the per-icon correlation between predictions at 0.05 and 0.35 is 0.8619 against
0.9122, and recovery at 0.05 is -4.5349 against -4.6517. Every figure moved in the
predicted direction; none moved enough to matter.

Eliminating the informational explanation forced a look at the encoder, and the defect
is there and provable. `GeometryDenoiser` summed six coordinate lookups — all from one
shared embedding table — into a single vector per segment slot, and two start lookups
into the path vector. A sum is commutative, so a segment was an unordered bag of its
values. In float64, a program and its coordinate-swapped variant produce byte-identical
logits: maximum absolute difference exactly 0.0 across 414 swapped segments in six
held-out icons, with identical argmax predictions. The model was never ignoring `x_t`;
it was reading a scrambled copy of it. Evidence is committed as
`scripts/slot_invariance_probe.py` and its output.

Run `openmoji-g1-slot-bound-v5-f502df1-d8af55ea-9b9b1699` binds each value to its slot
by elementwise multiplication with a learned per-slot vector — **768 parameters**, +0.13%
— changing nothing else about v2. It beats the 3.53x capacity run on every measure:
held-out loss 7.1782 against v2's 7.3333 and v3's 7.3790, retained accuracy 0.5408
against 0.4128 and 0.4323, aggregate 0.3793, changed recovery 0.0787. At corruption 0.20
it is now break-even, helping on 20 of 32 icons where v2 helped on 14.

But its prediction correlation at 0.05 against 0.35 is **0.8621** — unchanged from v4's
0.8619 — and mean `x_hat_0` error still moves only 0.124 to 0.143 while the input moves
0.051 to 0.180. Two of its three behavioural criteria are falsified.

**Decision.** Keep slot binding permanently: it is a strict improvement for 768
parameters and it removes a defect that would have confounded every later result. But
record plainly that fixing the encoder raised the ceiling without changing what the model
does — it still emits a prior lightly adjusted by its input.

Five candidate explanations have now been eliminated by predeclared comparison: data
volume, model capacity, the corruption regime, noise-level information, and encoder
slot-blindness. What remains is the prediction objective. Every field is predicted by an
independent single-shot softmax over a 289- or 417-way vocabulary, so there is no cheap
way to express "leave this one alone": copying a field means reconstructing its exact
token from scratch, and at 65% of fields uncorrupted that is most of the task. The next
experiment is an edit-mask or residual formulation in which copying is the default and
the model predicts only what to change.

## 2026-09-20 — Every Gate G model is worse than doing nothing

**Hypothesis.** None. This is a control that should have existed since v1 and did not,
computed from already-committed metrics at no cost.

**Observation.** Under argmax decoding, emitting the input unchanged is a legal policy:
the identity function scores retained-token accuracy 1.0, changed-token accuracy 0.0,
and aggregate accuracy equal to the uncorrupted fraction, 26,030 of 40,008 held-out
fields, or 0.6506 at corruption 0.35. Every trained model in the sequence scores below
it:

| model | aggregate | changed | retained |
| --- | ---: | ---: | ---: |
| v1, 256 icons | 0.2312 | 0.0552 | 0.3256 |
| v2, full split | 0.2886 | 0.0573 | 0.4128 |
| v3, 3.53x capacity | 0.3036 | 0.0639 | 0.4323 |
| v4, noise-conditioned | 0.2980 | 0.0556 | 0.4282 |
| v5, slot-bound | 0.3793 | 0.0787 | 0.5408 |
| **identity, copy `x_t`** | **0.6506** | 0.0000 | 1.0000 |

The best model is 0.2713 below identity. In render terms the same holds: identity has a
recovery fraction of exactly 0 at every corruption level by definition, which beats v2,
v4 and v5 at 0.05, 0.10 and 0.20, and loses only at 0.35.

**Decision.** Record this as a design failure in the experimental sequence, not only a
result about the models. v1 compared held-out recovery against an *untrained* control
and reported 18x above it; that comparison made weak learning look like progress,
because random is the wrong floor. The right floor is the trivial policy, and against it
every model so far is net harmful. The predeclared criteria from v1 through v5 are not
invalidated - they measured what they said - but "learns far above its untrained
control" should never again be reported without the identity baseline beside it.

This also sharpens the remaining candidate rather than changing it. A denoiser that
cannot beat "return the input" is not denoising, and the objective is why: every field
is an independent softmax over a 289- or 417-way vocabulary, so representing identity
requires reconstructing all 26,030 uncorrupted tokens exactly. The next run makes
identity the default by predicting a per-field keep-or-change decision, and its primary
criterion is to beat 0.6506 aggregate - the first Gate G criterion set against the
trivial policy rather than against noise.

## 2026-09-20 — Gate G negative result: the corruption is not identifiable, so it is not invertible

**Hypothesis.** Every trained model scored below the identity policy because the
objective made identity expensive: each field is an independent softmax over a 289- or
417-value vocabulary, so "leave this one alone" costs as much as inventing a value.
Adding a per-field keep-or-change decision, and copying the input where it says keep,
should let the model clear the trivial policy.

**Observation.** Run `openmoji-g1-edit-mask-v6-0228c3b-0f6ce115-9b9b1699` falsified all
four primary criteria and fired the predeclared degenerate-collapse watch. Aggregate
held-out accuracy rose from v5's 0.3793 to 0.6119, still below identity's 0.6506;
retained accuracy rose from 0.5408 to 0.9335, below identity's 1.0; and changed-token
recovery *fell* from 0.0787 to 0.0132. The model predicts keep on 91.4% of held-out
fields. It moved most of the way to the trivial policy and stopped just short, landing
strictly worse than both v5 and identity. Render recovery is near zero at every
corruption level, against v5's -3.1470 at 0.05: the edit mask converted an actively
harmful model into a nearly inert one.

The keep head is a corrupted-field detector, and measuring it directly answers the
question this sequence has been circling. Recall is **0.1058** — one corrupted field in
ten — and precision is 0.4312 against a base rate of 0.3494, a lift of 1.23x. It is
barely better than guessing. That is not an architectural shortcoming. A coordinate
resampled uniformly from a 289-value legal vocabulary lands on a perfectly plausible
coordinate, so telling it apart from a legitimate one requires already knowing what the
icon should look like. Detection is not an easier sub-problem than denoising; it is the
same problem. With 65% of fields uncorrupted and no reliable way to find the rest,
predicting keep is loss-minimising, and the model found it.

**Decision.** Record this as a Gate G-level negative result, as the run's predeclared
falsification meaning required, rather than patching it. Seven candidates have now been
eliminated by predeclared comparison: data volume, model capacity, the corruption level,
noise-level information, encoder slot-blindness, the objective's inability to express
identity, and finally the separability of detection from denoising.

Factorized role-uniform categorical corruption at p=0.35 over a 289/417-value coordinate
vocabulary produces states from which the corruption is not identifiable, and therefore
not invertible, by a model of this class. This bears directly on the project's working
hypothesis, which names *structure-aware* categorical corruption: Gate F selected the
factorized process as primary on four-icon fixtures at 96-99% held-out recovery, and
that selection does not survive contact with 2,681 icons. The Gate F comparison should
be re-read as a memorization comparison rather than a denoising one, and the
path-correlated and whole-path processes it set aside deserve re-examination at corpus
scale with the identity baseline attached.

What this does not touch: Gate C and Gate D stand, the codec is exact and render-safe,
and v5 showed the encoder fix was worth keeping. The negative result is about the
corruption process, not the representation.

## 2026-09-20 — Correction: the corruption IS identifiable; the model failed to learn it

**Hypothesis.** The preceding entry closed Gate G on the reading that factorized
role-uniform corruption is not identifiable, inferred from a trained keep head reaching
0.1058 recall at 0.4312 precision against a 0.3494 base rate. That inference had a
competing explanation it did not rule out: the corruption is identifiable and the model
failed to learn it. The project plan's next action called for settling this without
training, and it should have been settled before the stronger claim was made.

**Observation.** Run `openmoji-g1-detectability-877feff-4processes-9b9b1699` scores
every legal coordinate field by a fixed local-continuity statistic — the mean absolute
distance to the same slot in adjacent segments — with no learned parameters at all. On
48 held-out icons it separates corrupted from retained fields at **AUC 0.7698** under
factorized corruption at p=0.35. At the trained detector's own flag rate of 8.573%, the
statistic reaches **0.7451 precision and 0.1824 recall**, against the trained model's
0.4312 and 0.1058. A zero-parameter heuristic beats a 579,872-parameter model by 73% on
precision at the same operating point.

Across processes: path-correlated 0.9333, factorized at p=0.10 0.8575, factorized at
p=0.35 0.7698, whole-path 0.5318. The whole-path figure is not comparable — donor
compatibility replaced only 32 of 13,128 fields, and a donor path is locally smooth by
construction, so local continuity is the wrong instrument for it.

**Decision.** Withdraw the information-theoretic claim. Factorized corruption at p=0.35
is identifiable; Gate G's failure is one of learning, not of information, and the
previous entry's decisive step was wrong. Every measurement in v1 through v6 stands and
so does every predeclared outcome; what does not stand is the inference from "the model
could not detect it" to "it is not detectable". That inference needed a
model-independent check, and the check was available for the cost of no training.

The Gate G result is therefore narrower and more useful than stated: a 580K-parameter
bidirectional transformer trained this way does not learn a corrupted-field detector
that a trivial local statistic already provides, and it collapses to the trivial policy
instead. That is actionable — a model given the continuity signal, or trained with a
detection objective that the statistic bounds from below, has somewhere to go.

The Gate F implication survives and strengthens. Path-correlated corruption is markedly
more identifiable than factorized, 0.9333 against 0.7698, which is what a structural
argument predicts: a replaced block is inconsistent with its neighbours in a way a
single resampled coordinate is not. Gate F selected factorized on four-icon fixtures
where recovery was memorization, and this is independent, training-free evidence that
the selection should be revisited at corpus scale.

## 2026-09-21 — Two bugs, not a wall: the Gate G plateaus were measurement artifacts

**Hypothesis.** A five-lens diagnostic panel over this codebase, with judges, was run
because the sequence had eliminated five explanations and was about to conclude against
the model class. Two lenses independently identified a defect I had missed, and I
verified it before acting.

**Observation.** Two defects, present through the entire Gate G sequence:

The loss averaged cross-entropy over per-(segment kind, coordinate slot) **groups**, not
fields. QUAD is 0.93% of the corpus, so on the 128-icon held-out draw its four groups
hold one field each and carried 4/13 of the coordinate loss against 40,004 other fields
— a measured 5,287x per-field weight ratio — while the group count flipped between 9 and
13 by batch. And no attention padding mask was passed, so 29.4% of the sequence was
attended as content. Because the evaluation path uses the same loss, held-out loss — the
checkpoint-selection and early-stopping signal — was roughly 31% four individual fields.

Correcting both, with no new parameters and no architectural change, took the trained
corrupted-field detector from **1.365 to 2.806** precision lift and the selected step
from 660 to **6,180 of a 6,300 cap with no early stop at all**. Both lenses' predictions
— 1.70 and 1.75, recorded before the run — were exceeded. The run reproduces identically.

**Decision, part one.** Every plateau in the Gate G sequence was declared by that broken
scalar, on runs stopped at 6–24% of their step cap: v3 at 360, v7 at 660, v2 at 840, v4
at 1,020, v6 at 1,380, v5 at 1,500. The eliminations of **data volume, model capacity,
the corruption regime and noise-level conditioning** are no longer admissible as
measured. Their numbers stand; their reading as plateaus does not, and none should be
cited again without a re-run under sound optimisation. Two of my own conclusions fall
with them: "the encoder does not represent the signal" used a frozen-encoder probe that
shares one weight vector across all six slots, which is mis-specified and a lower bound,
and the architecture lens's ceiling ladder predicted ~1.30 lift at 16 dims per field,
which v8 exceeds at that same width.

**Decision, part two, and it matters more.** A judge found what all five lenses missed.
A detector scoring each field purely by **how rare its own token is for its own role** —
zero parameters, zero context, no neighbours — reaches 0.9887 precision and **2.831 lift**
at the same flag rate, and 0.9962 at a 5% flag rate. v8's 2.806 matches it rather than
beating it. Corruption draws uniformly from the full legal vocabulary, so a large share
of replacements land on tokens real icons essentially never use, and detecting them needs
no geometry whatsoever.

So precision lift under factorized role-uniform corruption is **largely a
marginal-density test**, and v8 has most likely learned the corpus marginal. The
detection metric leaks, which means the corruption process leaks. The next experiment is
to remove the leak: draw replacements from the **corpus marginal for that role** rather
than uniformly, so that a corrupted token is in-distribution by construction and
detecting it actually requires the surrounding geometry. Everything measured on the
uniform process — including v8 — should be re-read against that.

## 2026-09-21 — Detection solved, reconstruction not, and the two cannot yet coexist

**Hypothesis.** With four verified defects fixed — the group-pooled loss, the missing
attention padding mask, the density leak in the corruption process, and coordinates
encoded as unordered categories — the full denoising objective should clear the identity
baseline that every model in this project has failed.

**Observation.** It does not, and the reason is measured precisely.

First the win. Encoding coordinates as magnitudes rather than identities (v10) produced
the project's first trained model to beat a zero-parameter heuristic on a task with no
shortcut in it: detector precision lift **2.781** against a local-continuity statistic's
1.893 and a marginal-density detector's 0.771 — with **54,720 fewer parameters**,
525,152 against 579,872, because the categorical tables are not allocated at all.
Capacity is excluded from both directions.

Then the wall. Turning the value head back on (v11) gives changed-token recovery of
0.0296, **2.24x the best any earlier model managed**, but aggregate accuracy of 0.5839
against identity's 0.6515. Adding that head costs **1.064 of detector lift** — 2.781
down to 1.717 — dropping it back below the free statistic. The two heads share one
encoder, and a 289- or 417-way exact-token objective against a 2-class decision is not a
fair fight; the encoder is shaped by the harder task, which it performs at 0.1087, and
the easier one it had solved is collateral damage. The same pattern appeared between v6
and v7 at 1.234 against 1.365; with a genuinely good detector to lose it is now an order
of magnitude larger.

No decode threshold rescues it. With the value head at 0.1087 on corrupted fields the
break-even detection confidence is p > 1/(1+0.1087) = 0.9019, and sweeping the gate
loses to identity at every flag rate, monotonically: 0.6496 at 1%, 0.6374 at 5%, 0.5839
at the model's own 19.15%. Working back from the 1% point, this degraded joint detector
reaches only about 0.73 precision at its most confident percentile, where v10 reached
0.9691. With v10's detector and v11's value head, flagging the top percentile would have
paid.

**Decision.** The remaining problem is not detection. v10 solved that. It is that
detection and reconstruction cannot currently be learned together, and that
reconstruction itself is weak. Two separable next steps, both measured rather than
guessed:

1. Stop the objectives competing — weight them by field count rather than letting an
   exact-token softmax dominate, or give the heads separate encoders. v10 and v11
   bracket exactly what is at stake on an otherwise identical setup: 2.781 against 1.717.
2. Make reconstruction learnable — the value head predicts an exact bin on a
   quarter-unit metric lattice through a categorical softmax, so being close earns
   nothing. A distance-kernel target spreads mass over nearby bins in proportion to
   distance. The task-formulation lens measured that the model is a calibrated localiser
   being graded pass/fail at plus or minus 0.125 units.

Identity is still not beaten, and both ingredients that would beat it have now been
demonstrated — just never at the same time.

## 2026-09-21 — The first model here to beat doing nothing

**Hypothesis.** v10 beat every free detector at 2.781 lift with detection alone; v11
added the value head and the detector fell to 1.717, below the free continuity statistic,
leaving aggregate accuracy at 0.5839 against identity's 0.6515. The two heads share an
encoder and are unequal at chance — log 2 nats for a 2-class keep decision against log
417 for the value head, a factor of 8.7 — so an unweighted sum lets the value task
dominate. Scaling the value term by log(2)/log(417) = 0.1149, a derived weight rather
than a tuned one, should let both survive.

**Observation.** Both do. The detector recovers to **2.569**, nearly v10's
detection-only 2.781 and well past the free statistic's 1.893, while changed-token
recovery rises to **0.0738** — 2.5x v11's and 5.6x the best any earlier model managed.
At the model's own 0.5 gate aggregate accuracy is 0.6086, so the predeclared
beats-identity criterion is falsified as written.

But the confidence is informative even where the default threshold is not. Sweeping the
gate and reporting on the same icons would be selection on the evaluation set, so the 128
held-out icons were split: the threshold was chosen on 64 and reported on the disjoint 64.
Flagging the most confident 5%, the model reaches **0.6567 against identity's 0.6502** on
icons that played no part in choosing it. That is the first time anything in this project
has beaten the trivial policy of emitting its input unchanged.

**Decision.** Record it as a first, and record its two qualifications in the same breath:
the margin is about 1% relative, and it depends on the calibrated threshold — the model's
own decode still loses by 0.0406. What has changed is not that the model is good but that
it is finally better than nothing, and that the remaining gap is a calibration and
reconstruction problem rather than a representational one.

The arc, five verified defects and a model that shrank from 579,872 parameters to
525,152 along the way: field-pooled loss and an attention padding mask (1.365 -> 2.806 on
the leaky task), marginal-respecting corruption to remove the density shortcut (1.506 on
the honest one), metric coordinate encoding (**2.781**, first trained model to beat a
zero-parameter heuristic), the value head reinstated (1.717, identity still wins), and an
entropy-balanced objective (2.569, identity finally beaten).

What remains is measured. The value head is weak at 0.1233 exact-token accuracy on a
289/417-way vocabulary over a quarter-unit metric lattice, which keeps the break-even
detection confidence high so only the top few percent of flags pay. The next step is the
distance-kernel target the task-formulation lens argued for, so that being close earns
gradient, and calibrating the gate during training rather than after it.

## 2026-09-21 — The identity win, without the asterisk

**Hypothesis.** v12 beat the identity baseline only with a decode threshold fitted on
held-out icons; its own argmax decode lost by 0.0429. That threshold is derivable rather
than arbitrary: keeping a retained field is always right and changing one is right only
if the value head re-predicts the same token, so changing pays above `p = 1/(1+q)` in the
value head's accuracy `q` on corrupted fields.

**Observation.** v13 derived the threshold from train icons and beat identity with
nothing fitted on held-out data — 0.6519 against 0.6515 — but by only 0.0003, and its
threshold criterion was falsified informatively. The value head scores **0.2751** on
icons it trained on against **0.1233** held out, so `q` was inflated and the threshold
came out at 0.784 where the held-out accuracy implies 0.890. The model edited more than
paid.

v14 estimates `q` on 256 icons **withheld from training**: 0.2220, threshold 0.8183,
aggregate **0.6543 against identity's 0.6515**, a margin of **+0.0027**. All four
predeclared criteria pass. Retained-token accuracy reaches **0.9815**, the highest of any
model in this project. The run is handicapped by its own design — withholding those icons
also removes them from training, so it learned from 2,425 rather than 2,681, 9.6% less
than everything it is compared against — so the margin is conservative.

**Decision.** Record the result and its size in the same sentence. Nothing in v14 is
fitted on held-out data: the threshold comes from icons the model never trained on, the
checkpoint is selected on held-out loss as every run here does, and the identity
comparison is a plain read. The project has a denoiser that does better than doing
nothing, with no caveat about how the number was obtained.

It is also +0.0027 on token accuracy, about 0.4% relative. This is not a good model; it
is a model that has stopped being worse than useless. The value head reaches only 0.222
on withheld corrupted fields, which is why the break-even threshold sits at 0.818 and the
model edits just 4% of what it sees.

The next step is unchanged and every term in it is now measured: the value head predicts
an exact bin on a quarter-unit metric lattice through a categorical softmax, so being
close earns nothing. A distance-kernel target should lift `q`, which lowers the
break-even threshold, which lets the model act on more of what it already detects
correctly at 2.569 lift.

A recording note: v13's in-place idempotency rerun failed closed because v14 added two
summary fields afterwards and the artifact writer refuses to replace a differing file.
Re-running v13 to a fresh report root reproduces every other field exactly. The failure
is kept rather than papered over.

## 2026-09-21 — The distance kernel works, and the criteria were measuring the wrong thing

**Hypothesis.** The value head predicts an exact bin on a quarter-unit lattice through a
289/417-way softmax, so a confident one-view-unit miss costs 10.0187 — identical to a
hundred-unit miss. Its accuracy of 0.222 pins the break-even decode threshold at 0.818,
which is why v14 edits only 4% of fields despite a detector at 2.569 lift. Spreading the
target by distance should raise that accuracy and every downstream term with it.

**Observation.** All three primary criteria are falsified, and all three are stated in
exact-token terms: the withheld value accuracy falls to 0.1623, the identity margin to
+0.0014, and the derived threshold rises to 0.8604.

The measures committed to as reported-not-gated say the opposite. Mean absolute error on
corrupted held-out fields falls from **8.2871 to 6.0282 view units**, a 27% reduction,
with the median down 33% from 3.75 to 2.50 and the fraction within one view unit up from
0.2793 to 0.3169. Paired per-icon render recovery, each model decoded at its own derived
threshold, rises from **+0.0447 to +0.1246** at 72 px and +0.0729 to +0.1306 at 18 px,
both intervals excluding zero, helping 10 and 11 of 12 icons. These are the first
positive render recoveries this project has measured under leak-free corruption.

**Decision.** Record the criteria as falsified and the intervention as a success, because
that combination is a verdict on the criteria. Exact-token accuracy is anti-correlated
with what the project is trying to produce — the task-formulation lens said so before any
of this ran, calling the model a calibrated localiser graded pass/fail at ±0.125 units,
and this is the measurement that settles it.

Two consequences. **Demote exact-token accuracy**: mean absolute view-unit error and
paired render recovery become the primary reported metrics, with exact-token accuracy
kept as secondary so nothing already published is withdrawn. And **correct the decode
gate**: the break-even rule `p > 1/(1+q)` is derived in exact-token terms, so a value
head that is closer but not exacter raises its own threshold and edits less of what it
could improve. A gate derived from expected geometric gain is the next run.

A recording note worth keeping. On a first pass I compared the two models by their median
`x_hat_0` error, 0.16074 against 0.16985, and read v15 as worse — the same mistake this
session already caught once. Both models render identical corrupted inputs, so the paired
per-icon difference is the correct statistic, and it says the opposite. On paired data,
compare the pairs.

## 2026-09-21 — Gate on closeness: render recovery doubles again, and the metric question is settled

**Hypothesis.** v15 localised 27% better than v14 yet gated itself more tightly, 0.8604
against 0.8183, because the break-even rule `p > 1/(1+q)` credits only exact tokens.
Choosing the threshold by absolute view-unit error saved on withheld calibration icons
should lower it and let the model act on what it already localises well.

**Observation.** All predeclared criteria pass. The threshold falls to **0.6800**, saving
2.9843 view units per field on calibration, and paired per-icon render recovery rises
from **+0.1246 to +0.2511** at 72 px and +0.1306 to +0.2745 at 18 px, helping 11 of 12
icons with both intervals excluding zero. The trained weights are byte-identical to
v15's, so it is the decode rule alone — 5.6x v14's recovery and 2x v15's.

The two corrections compound because they are the same correction applied twice: train
for closeness, then gate on closeness.

Held-out aggregate token accuracy falls to **0.6354 against identity's 0.6515**, which
the run's record anticipated as expected_tension before it ran.

**Decision.** Between v15 and v16 the metric question is settled rather than merely
argued. **Exact-token accuracy is the wrong primary metric for this project.** v16 is
simultaneously worse than the identity policy on it and the best geometric denoiser the
project has produced; both statements are true and neither is a contradiction. Mean
absolute view-unit error and paired render recovery are the primary metrics from here,
with exact-token accuracy kept as secondary so nothing already published is withdrawn.

Where the sequence lands, from a model that actively destroyed its input to one that
recovers a quarter of the render gap, across seven verified defects and corrections and
a model that shrank from 579,872 parameters to 525,152: identity 0.0000, v14 +0.0447,
v15 +0.1246, v16 **+0.2511**.

And what it does not establish, unchanged: no render in this project is a recognizable
icon. At 35% corruption the input is already scribble and recovering a quarter of that
gap leaves scribble. The direction is measurable and compounding; the generator does not
yet work.

## 2026-09-21 — The repaired measurement restores all four eliminations

**Hypothesis.** Four conclusions in this gate — data volume matters, capacity does not,
the corruption regime is not the constraint, noise-level conditioning changes nothing —
were each declared on a held-out scalar that was ~31% four individual fields, on runs
stopped at 6–24% of their step cap, under a corruption process leaking a density
shortcut, with coordinates as unordered categories, graded on a metric now known to point
the wrong way. All of that is fixed and none had been retested. At least one was expected
to change, capacity most likely.

**Observation.** Three arms off the v16 base, one factor each, judged on paired per-icon
render recovery against the base's +0.2511 and its interval half-width of 0.1103:

| arm | recovery | verdict |
| --- | ---: | --- |
| capacity, 3.69x parameters | +0.2779 (\|diff\| 0.027) | **restores v3** |
| data volume, 256 icons | +0.0766 (0.175 below) | **restores v2** |
| noise conditioning | +0.2096 (\|diff\| 0.042) | **restores v4** |

All three original conclusions survive. The eliminations were right even though they were
measured badly, and the four are admissible again. The data-volume arm is sharpest:
256 icons against 2,425 cuts render recovery 70%, with its interval barely clearing zero.

Capacity is the one I expected to move and did not. v3's finding had been recorded as the
least trustworthy because 768 parameters of slot binding later beat a 3.53x capacity
increase outright; at 3.69x on a sound setup it is still within noise, and the best model
this session built remains the smallest. The two were never in tension — one is a
representational fix, the other raw size.

**Decision.** Restore all four to the record as sound. Two recording notes kept rather
than tidied: the first noise-conditioning arm evaluated at 0.05 rather than 0.35 because
`evaluation_corruption_probability` was unset and defaults to `corruption_probability`,
which the training range had moved — retained under its own identity as failed. And on
first reading I labelled the data-volume arm as overturning v2 by applying a symmetric
rule to a directional criterion; the criterion as predeclared is the one applied.

Separately, the audit script written earlier caught a real error in my own bookkeeping:
the failed arm had been logged under the parent comparison's run_id, marking the whole
three-arm run failed. Corrected append-only, with the superseded row named.

## 2026-09-21 — It produces recognizable icons; the corruption schedule was the ceiling

**Hypothesis.** Every result in this gate has been at corruption probability 0.35, where
the input is already scribble, and every report has carried the line that no render in
this project is a recognizable icon. Was that a statement about the model or about the
regime?

**Observation.** About the regime. v16, rendered at four corruption levels and gated at
the threshold derived for each on the withheld calibration icons:

| p | median `x_t` | median `x_hat_0` | mean recovery | helped |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | 0.03730 | 0.02235 | +0.0308 (median +0.3496) | 9/12 |
| 0.10 | 0.08247 | 0.04206 | **+0.3902** | 11/12 |
| 0.20 | 0.12568 | 0.10014 | +0.2469 | 10/12 |
| 0.35 | 0.16955 | 0.13373 | +0.2511 | 11/12 |

The same sweep over the v2 checkpoint gave −4.6517, −1.2024, −0.1315, +0.1800 and helped
zero of 32 icons at 0.05.

At **0.05** the predictions are recognizable icons and visibly repaired — the hedgehog's
stray diagonal removed, the first-aid kit's slash gone, "WC" cleaned to near-perfect, the
vampire's face restored from under a scribble. At **0.10** the repairs are larger and
still land: `1F3CB` goes from heavy scribble to a clean, recognizable weightlifter, and
`1F199` returns as a clean "UP!" badge. At 0.20 and 0.35 neither input nor output is
recognizable, which is exactly what every earlier report described — correctly, for those
levels.

The `never_damages` criterion is falsified. At 0.05 nine of twelve icons improve, two are
untouched, and one — `26A0` — worsens by 0.05018 absolute; because its `x_t` error was
only 0.01390 the relative statistic turns that into −3.61 and drags the mean to +0.0308
from a median of +0.3496. The absolute difference does not rescue it either, so at twelve
icons one bad case genuinely prevents significance.

**Decision.** Retire the standing line. "No render in this project is a recognizable
icon" was true of probability 0.35 — the only level this gate ever trained or reported at
— and is obsolete as a general claim. The ceiling was set by the corruption schedule, not
by the representation, the model or the objective, each of which was suspected in turn
and none of which was the binding constraint at the end.

Keep the framing narrow. This is a denoiser that removes a handful of stray strokes from
an icon that is mostly intact. It is not a generator, it does not reconstruct destroyed
geometry, and `26A0` shows it can still damage a near-perfect input. But it is the first
result here whose output a person would recognise, and v16 trained only at 0.35, so every
level below is off-distribution — which makes the result stronger rather than weaker.

The next run follows directly: train across a range of corruption levels and report at
each, rather than training and reporting at a single destructive point.

## 2026-09-21 — The training schedule is not the lever either

**Hypothesis.** The sweep showed v16, trained at probability 0.35 alone, producing
recognizable icons at 0.05 and 0.10 — levels it had never seen. A model sampling its
corruption level over 0.05–0.50 and told the level should do better there.

**Observation.** It does not. On 32 icons, both models at four levels with the same
corruption seeds and per-level derived gates, the paired difference (range minus v16) is
−0.00125 at 0.05, −0.00087 at 0.10 and −0.00324 at 0.20, every interval spanning zero —
and **−0.01400 at 0.35 with the interval excluding zero**. Training across levels is
indistinguishable at the low levels it was supposed to help and measurably worse at the
level it was supposed to trade away.

This run's record predicted it before the run: the range-trained model saves fewer view
units per field on its calibration icons at every level, 0.4134 against 0.4636 at 0.05
and 1.9435 against 3.0133 at 0.35, and that was flagged as arguing against the
hypothesis. The calibration estimate is therefore a usable cheap predictor of the render
outcome.

Separately, 32 icons resolves the sample-size limit that falsified `never_damages` last
time. **Both models now improve lightly corrupted inputs significantly** — v16 on 26 of
32 icons at p=0.05, the range model on 29 of 32, both intervals excluding zero. The
sweep's headline is properly supported rather than suggestive.

**Decision.** Every training-side factor this gate identified is now settled: data volume
matters, capacity does not, noise conditioning does not, and the corruption schedule does
not. v16's extrapolation from a single level is as good as training for the regime.

That leaves **Gate I** — the cached autoregressive baseline on the same codec,
PROJECT_PLAN.md section 12 branch 4 — as the only untried route to a generation result
rather than a denoising one. Nothing in the project blocks it, and it has been named as
the remaining branch since the Gate G sequence began going wrong.

## 2026-09-21 — Gate I: the causal model learns the codec

**Hypothesis.** Before anything runs at corpus scale, the causal model should be able to
memorise four icons and reproduce them exactly. Gate E asked this of the denoiser first,
and Gate G then spent months reading plateaus that turned out to be seven defects rather
than limits — every one of which would have shown up here as a model that could not
overfit.

**Observation.** All four predeclared criteria pass. Over the 464 positions per icon
where the grammar leaves more than one legal token, next-token accuracy reaches **1.000**
and the loss falls by **19,902×**, from 4.602 to 0.00023. Greedy decoding through the
KV cache and the dynamic legal-token masks reproduces **4 of 4** programs token for
token, and every sample survives the packed validator and the isolated renderer.
Accuracy hit 1.000 by step 200 of 2,000.

The four icons are drawn from four different subgroups on purpose. The model is
conditioned on group and subgroup only, so two icons sharing a subgroup are literally the
same prompt and no model could reproduce both greedily; a test pins that property,
because if the selector ever broke it the criterion would become unsatisfiable and the
study would read as a model failure rather than a harness one.

**What this is not.** It is a learnability diagnostic and nothing more. Four sequences
into 375,106 parameters is far inside capacity, so this measures whether the objective,
the masks and the cache are wired correctly — not whether the model can generate. Gate G
is the standing reminder of why the distinction is worth stating: a number that looks
like progress is not progress until something honest is standing next to it.

**Decision.** The plumbing is sound, so Gate I proceeds to corpus scale.

## 2026-09-21 — Gate I at corpus scale: the AR baseline barely clears its floor

**Hypothesis.** At a budget matched to Gate G's v16 on every term the plan names —
524,674 parameters against 525,152, the same bucket and hashed corpus, the same
family-disjoint splits, 100,800 icon presentations — an autoregressive model over this
codec should halve the negative log likelihood of a zero-parameter floor.

The floor is not nothing, deliberately. Paired render recovery needs a corrupted input
to recover from and a sampler has none, so identity is meaningless here and quoting it
would be worse than admitting the gap. The floor is a position-marginal policy instead:
it knows the grammar and the empirical token frequency at every position, is
renormalised over exactly the legal set the model is restricted to, and knows nothing
about which icon it is writing.

**Observation. Falsified.** The model reaches **3.6249 nats per free token against the
floor's 3.9290** — a ratio of **0.923** where the predeclared criterion was 0.500. It
gets there at **step 600 of 6,300** and then goes backwards: held-out likelihood rises
monotonically while training loss falls from 3.22 to 1.96, and from **step 1200 the
model is worse than the zero-parameter floor**. Early stopping fired at step 3000.

Everything else it was asked for, it did. All 32 samples validate and render, all 32 are
distinct, so there is no class collapse. Median ink coverage is 0.414 against a corpus
interquartile range of [0.261, 0.387] — outside, but not by much, and that figure was
measured rather than gated.

**The cache buys 16%, not an order of magnitude.** 1.232 s against 1.435 s for a decode
that re-runs the whole prefix at every one of 1,376 positions, with the two paths
agreeing at every position. At 524,674 parameters and d_model 96 the decode is bound by
kernel launches rather than by arithmetic, so re-reading the prefix costs almost nothing.
This is a real measurement on a named GPU and it contradicts the asymptotic argument;
the cache is worth having at scale and is close to free to skip here. Peak VRAM 3.63 GiB,
301 s to train.

**What this is not yet.** It is not a verdict on autoregression over this codec. The
model inherited a defect Gate G already verified on this exact codec — coordinate tokens
encoded as unordered categories on a quarter-unit lattice, where the denoiser's trained
embedding table scored Spearman −0.136 against bin distance. Calling i2 a limit before
testing that would repeat Gate G's mistake with the sign flipped. Arm 3 changes that one
thing and nothing else.

## 2026-09-21 — The coordinate encoding is not the AR model's constraint

**Hypothesis.** Gate G verified a specific defect on this codec: coordinate tokens
encoded as unordered categories on a quarter-unit lattice, where the denoiser's trained
embedding table scored Spearman −0.136 against bin distance. Correcting it took render
recovery from 0.0447 to 0.2511 *while shrinking the model*. The causal model inherited
the defect verbatim, so it should respond the same way.

**Observation. Falsified, and cleanly.** One change from i2, nothing else moved. Best
held-out likelihood **3.6946 nats per free token against i2's 3.6249** — a ratio to the
floor of **0.940 against 0.923**. Not an improvement; a shade worse, while carrying
**1,824 more parameters** than i2, which favoured it. It peaks at step 300 rather than
600 and overfits on the same trajectory.

The features themselves are correct — a test pins that they are monotone in bin distance
at initialisation, being a fixed function of the decoded value rather than something
trained. So the ordering the denoiser needed is present here and the model does not
benefit from it.

That asymmetry is worth stating rather than explaining away. A denoiser is asked to put
back a value *near* the true one, so a representation that knows which values are near
each other is doing most of the work. A generator asked to write 1,376 tokens from a
class label is not failing at precision; the 7.7% it does clear the floor by is not
lost to coordinate arithmetic.

**Method note.** This arm trained for eight minutes and then died at the last step: the
uncached latency diagnostic never passed the segment kinds and raised under metric
coordinates, taking the run with it. Full-sequence and single-token forwards were both
tested and both passed; the growing-prefix shape, used only to measure what the cache is
worth, was not. The failed attempt is preserved under its own run id. Artifacts are now
written before the diagnostics, so a fault in a measurement can no longer destroy the
training it was measuring.

**Decision.** Two arms, the same place. Both peak within two to four epochs and then get
worse while training loss keeps falling, which reads either as too little data or as a
constraint more data will not lift — and those point at entirely different months of
work. Arm 4 measures it with three nested training sets rather than guessing.

## 2026-09-21 — Data scaling: the criterion passed and pointed the wrong way

**Hypothesis.** Two arms had landed in the same place, both peaking within two to four
epochs and then getting worse while training loss kept falling. That signature reads
either as too little data — 2,681 sequences of 1,376 tokens is very little — or as a
constraint more data will not lift, and those point at different months of work. Three
nested training sets, quarter, half and all, everything else fixed.

**Observation.** Both predeclared checks pass. Held-out likelihood improves at every
doubling, **3.6724 → 3.6585 → 3.6480 nats**, and the second doubling still buys **76%**
of what the first did, against a declared floor of 25%. By the criterion as written, the
model is data-limited.

**And the criterion was the wrong instrument.** Quadrupling the data moved the ratio to
the floor from **0.9340 to 0.9285**. At roughly 0.011 nats per doubling, closing the
1.66 nats between here and the standing 0.5 criterion would take on the order of **150
doublings**. There is no corpus of that size and there will not be one.

The criterion asked whether returns had *stopped*. They have not. It never asked whether
they were large enough to matter, and they are not. This is Gate G's failure mode
exactly — a metric that reads well while pointing away from the goal — and it is the
second time in this project that a predeclared criterion has passed without supporting
the decision it existed to inform. A direction without a magnitude does not identify a
lever. The sweep now carries a magnitude check alongside the direction, and arm 5 is the
first study to use it.

Two log-linear points are weak evidence for a 150-doubling extrapolation, and the
extrapolation is not what carries the finding. What carries it is the measured fact that
**4× the data bought 0.0055 of ratio** while the gap to the floor stayed at about 0.28
nats — the same 0.28 that survived changing the coordinate representation.

**Decision.** Data volume is not the lever at any corpus size this project can reach, so
Gate H is not the answer to Gate I's question. Capacity is the one factor left
unmeasured for this model, and it runs next at 9× the parameters — not to succeed, but
to find out whether it behaves like a lever at all.

## 2026-09-21 — Capacity is not a lever either, and is mildly harmful

**Hypothesis.** Three factors had been tested and none moved the result. Capacity was the
last untested one for this model. Gate G found it did not matter for the denoiser, but
that is a different model on a different task and is not evidence here.

This study was also the first to carry a **magnitude** criterion alongside a direction,
which the data-scaling arm taught me to write. The bar was 0.85 — deliberately far below
the 0.5 that i2 and i3 were read against. 0.5 remains the standing bar for "this is a
generator". 0.85 asks a smaller question: at 9× the parameters, does capacity behave like
a lever at all?

**Observation. All three checks fail, and the direction is negative.**

| arm | parameters | held-out nats | ratio to floor | peak at | train |
|---|---|---|---|---|---|
| base | 524,674 | 3.6480 | 0.9285 | step 600 | 306 s |
| mid | 1,614,978 | 3.6565 | 0.9306 | step 300 | 423 s |
| large | 4,817,570 | 3.6899 | 0.9392 | step 300 | 599 s |

Nine times the parameters is **monotonically worse**, and the two larger arms peak at
step 300 — one and a half epochs — rather than 600.

**The standing picture.** The gap to the zero-parameter position-marginal floor is about
0.28 nats, and neither the coordinate representation, nor 4× the data, nor 9× the
capacity moves it. Every arm peaks within two to four epochs and then gets worse.

**What is not yet ruled out.** Every arm failed the same way, and it is a failure with a
standard remedy that none of them had: **no dropout, anywhere, in any arm.** Weight decay
was AdamW's default 0.01 throughout, which is not nothing but is not a response to
overfitting either. Declaring that autoregression fails on this codec without once
applying the standard regulariser for the exact failure mode observed would be a weak
claim, and this project has been strict about not making those. Arm 6 runs it.

## 2026-09-21 — Gate I answered: the samples are scribbles, and every lever is measured

**Regularisation was the last untested remedy and the best one found.** Every arm had
failed the same way — held-out likelihood bottoming out in two to four epochs and then
rising while training loss kept falling — and no arm had carried any dropout at all.

| dropout | held-out nats | ratio to floor | peak at |
|---|---|---|---|
| 0.0 (control) | 3.6480 | 0.9285 | 3.58 epochs |
| 0.1 | 3.5977 | 0.9157 | 3.58 epochs |
| 0.3 | 3.5921 | **0.9143** | 5.37 epochs |

The control reproduces arm 5's base arm at 3.6480 exactly, so this is a controlled
comparison and not three unrelated runs. Dropout is **the largest single effect measured
in this gate** — −0.0142 of ratio, against 4× the data's −0.0055 and 9× the capacity's
+0.0107 the wrong way — and it is still an order of magnitude short of the 0.5 bar. Its
curve has not turned: 0.3 beats 0.1 and peaks later, so the direction is not exhausted.
It is simply far too small to matter.

**Then I looked at the renders, which is the part Gate G taught.**

Eight subgroups picked by hash, three ancestral samples each, beside a real icon from
the same subgroup at the same size on the same ground
(`reports/learning/ar-renders-i2/samples.png`). **The samples are scribbles.** Every one
is a valid program — the legal-token masks guarantee that — drawn in corpus palette
colours, with a median of 8.5 active paths and ink coverage 0.388 against the exemplars'
0.253. Plausible statistics, and not one recognisable shape anywhere on the sheet.

The renders and the numbers agree completely, which is the useful part. A 7.7% gain over
a zero-parameter position-marginal floor is exactly what a model looks like when it has
learned the corpus's colours, its stroke widths and roughly how much ink an icon has,
and nothing about what an icon *is*.

**Gate I's answer.** The typed SVG codec supports generation in the sense that every
sample is a valid, renderable program — that part of the representation works exactly as
designed. It does not support *learning* generation autoregressively at any scale this
project can reach. Five arms measured every factor available and the gap to the floor
moved from 0.28 to 0.26 nats:

- coordinate encoding, the fix that was decisive for the denoiser: **nothing** (slightly worse)
- 4× the data: **−0.0055** of ratio, extrapolating to ~150 doublings to reach the bar
- 9× the capacity: **+0.0107**, the wrong way, monotonically
- dropout 0 → 0.3: **−0.0142**, the best of them, an order of magnitude short

**The cost comparison the plan asked for, delivered.** On gpubox-4080: 301 s to train at
matched budget, 3.63 GiB peak, 1.232 s to decode one 1,376-token icon through the KV
cache against 1.435 s re-reading the whole prefix at every position, the two paths
agreeing at every one of 1,376 positions. **The cache is worth 16%, not an order of
magnitude** — at this size the decode is bound by kernel launches rather than arithmetic.
That is a measurement on a named GPU and it contradicts the asymptotic argument.

**The quality half of that comparison is not well-posed, and saying so is the honest
result.** PROJECT_PLAN.md section 8 asks for a quality comparison against the denoiser.
Paired render recovery needs a corrupted input to recover from and a sampler has none;
held-out likelihood needs an unconditional model and the denoiser is conditioned on a
corrupted program. There is no task both models were trained for, and forcing one — tail
completion, say — would put the denoiser out of distribution and dress the result up as
fair. What would make it well-posed is a denoiser trained on contiguous-region corruption
or a masked, any-order model, and both are new work rather than this gate.

**Decision.** Gate I is closed on the evidence. The remaining branch that is not merely
"more of the same" is the any-order masked model: it would share this codec, these masks
and this measurement harness, and unlike the left-to-right model it could be scored
against the denoiser on one task with one metric. That is the first thing this gate's
result actually recommends.

## 2026-09-21 — Direction: the next result is an editor, and the corpus decides that

**Question.** Gate I closed with every lever measured and the samples scribbles, and the
standing next action was "build the any-order masked model" as a way to score something
against v16 on one metric. Before building it, step back: what has this project actually
established, what is left to win, and what would a result a person would call a success
look like on this corpus with this machine?

**What is established and stays.** The curated corpus, the typed codec with role-typed
coordinates and packed capacity, the safe serializer and isolated renderer, the run
contract, the legal-token grammar over the flattened sequence, and the measurement
discipline Gate G had to learn the hard way — the identity-equivalent baseline, the
zero-parameter floor, the field-pooled loss, the padding mask, the distance kernel, the
render beside every number. None of this is in question and all of it carries forward.

**What the corpus can and cannot support.** The training bucket is 2,681 icons; 1,597 of
its 1,767 variant families are singletons and 53% of it is people-body. Unconditional
generation asks a model to draw a concept it has never seen from a subgroup label,
having seen fewer than three thousand unique programs. No architecture solves that at
this scale, and Gate I's sweeps are best read as measurements of that fact rather than of
autoregression: 4x the data and 9x the capacity were both tested inside the regime where
neither can show, and "not at any size this corpus reaches" is the right reading.
Broader pretraining (Gate H) is the only route to generation, it is an order of magnitude
of data at most, and it has licensing and curation work in front of it. It is not where
the next result is.

**What v16 is, read plainly.** A fixed-topology geometry denoiser that removes some of
the stray strokes from an icon that is mostly intact. The corruption it was built for —
endpoints teleported across the canvas at 35% — is damage no editor produces, and the
regime where it helps, p ≤ 0.10, is a handful of strokes. It cannot add, remove or
restyle a path, and it never predicted topology or style. It beats doing nothing, which
took seven corrections to reach, and it is not a product.

**What is learnable here.** Conditional completion, where the icon itself does most of
the work. The bucket holds 39,535 contours and 228,943 segments; every contour is a
training example for "given the rest of this icon, draw the missing path", and every span
of segments and every style block is another. Context of that strength is what a small
model on a small corpus can use, and it is exactly what an editor needs: lock what you
keep, mask what you want redrawn. PROJECT_PLAN.md section 8 said this before any model
ran — editing may be the strongest product result even if unconditional generation is
mixed — and section 12's third branch is its fixed-topology special case. The evidence
now says it in numbers.

**The instrument.** A bidirectional model over the same flattened sequence, trained to
predict masked tokens from the rest, with mask families that are the editing operations:
a whole path (header and segments, its length kept so the packed layout stays put), a
contiguous span of segments, the style fields of a set of paths, the geometry of a set of
paths, and a uniform random mask so that everything-masked generation is the same model.
Decoding commits tokens in grammatical dependency order — lengths, headers, segment
kinds, coordinates — so `legal_mask` is exact at every commit and a sample is valid by
construction, as in Gate I. It inherits every correction from Gates G and I on day one:
metric coordinate features, the distance-kernel target on coordinates, the loss pooled
over fields, structural padding excluded from attention, dropout from the start, and a
matched-size arm beside a larger regularised one, so the one cell no gate has measured —
capacity together with regularisation — is measured here.

**The measurement.** Whole-path inpainting on held-out icons, scored by paired render
recovery against two zero-parameter policies: the icon with the path dropped, which is
what "do nothing" means for an editor, and a position-marginal sampler decoded through
the same grammar, which is what "knows the corpus statistics and nothing about this
icon" means. Predeclared: beat both with the interval excluding zero, a magnitude bar on
the median recovery, every output valid, and the sheet beside the numbers. This is
well-posed where the v16-versus-AR comparison was not: both baselines see exactly what
the model sees.

**Decision.** Open Gate L — structured editing with a masked any-order model — as the
primary line, and define success for it as an editor that visibly and measurably
completes held-out icons, followed by the Gate K editing viewer over its frozen
artifacts. Close Gate G as answered: generation is not compelling at this scale, blind
denoising helps only for light corruption, and editing is the compelling candidate, now
Gate L's question. Defer Gate H behind Gate L, to be reopened only for generation and
only if Gate L's editor works. Retire further single-factor sweeps on the p = 0.35
denoiser and the pending corruption-process third arm: a masked model makes the
corruption process a mask family, and the question dissolves. Start, as every gate here
has, with an overfit test.

## 2026-09-21 — Gate L's overfit test caught an off-by-one, which is what it is for

**Hypothesis.** Before anything runs at corpus scale, the masked model should memorise
four icons from four distinct subgroups through its mask families and its
grammar-ordered decoder: masked-token accuracy at memorisation, exact-token held-out
likelihood down two orders of magnitude, and a masked whole path reproduced token for
token on every icon.

**Observation. Falsified, by a defect rather than a limit.** Accuracy over the masked
free positions reached only 0.069 - below the position-marginal floor's 0.547 - and the
exact-token likelihood fell 2.7x rather than 100x. Yet the renders said the opposite:
inpainting recovered a median **79%** of the dropped path's render error and helped all
four icons, and the sheet shows the wheelchair user's head, the clock hand and the flag
stripe drawn back almost exactly.

Breaking the accuracy down by position type resolves the contradiction. Every
non-coordinate token is memorised perfectly - lengths, layers, style fields and segment
kinds all at accuracy 1.000 with likelihood near zero - and every one of the 407 masked
coordinates is wrong by **exactly one bin**: median 1, mean 1.0, all 407 within one bin.
Eval-mode and train-mode forwards agree to 1e-5, so the encoder is not it. The soft
coordinate target was centred on token `truth - 1`. The denoiser's distance kernel
subtracts one because its heads index lattice bins; copied into a loss whose logits
index tokens directly, that centred every coordinate target one token low, and the
model learned it faithfully. The unit test that passed compared a near miss against a
far miss, which the shifted kernel also orders correctly.

**Decision.** Fix the centre, and make the test ask the question that catches it: the
loss with logits peaked exactly on the truth must beat logits peaked one bin either
side. The falsified attempt is preserved under its own identity with its artifacts
moved aside, `reason_code: harness_defect`, and the overfit test is re-run on the fixed
revision before the corpus run is registered. Two things worth keeping from the failed
attempt: the structural half of the codec is memorised by step 300, and even a
quarter-unit-shifted completion recovers most of a missing path's render error - which
says the render metric is forgiving of exactly the error a soft target tolerates, and
the exact-token criteria are the ones that catch a shift like this.

## 2026-09-21 — Gate L's plumbing is sound, and one of its three criteria measured the target

**Hypothesis.** With the coordinate target centred on the token, the masked model should
memorise four icons through its mask families and its grammar-ordered decoder.

**Observation.** Masked-token accuracy over the free positions under the fixed masks
reaches **1.000** - from 0.024 untrained, past 0.947 by step 900 - and the structural
half of the codec is memorised by step 300 as before. Greedy grammar-ordered completion
of one masked whole path per icon reproduces **4 of 4** token for token, so the model's
completions render pixel-identical to the clean icons: median RGBA error 0.0 against the
path-dropped icon's 0.0035 and the marginal policy's 0.0106. Every greedy, sampled and
marginal completion validates and renders. On gpubox-4080: 106 s, 0.50 GiB peak.

The loss-reduction check is **falsified as written**: exact-token held-out likelihood
falls from 5.78 to 1.97 nats, 2.9x against a predeclared 100x. That is the target's
entropy, not the model's error. Under `exp(-|b - t| * 0.25 / 1.0)` the truth carries
about 12% of the mass and the rest sits on its neighbours, so a model that has learned
the target exactly still scores about 2 nats at every coordinate, and coordinates are
nine tenths of the masked free positions. The criterion was copied from Gate I's overfit
test, whose model trained on exact targets, and it cannot be met by any model trained
on this objective.

**Decision.** Record the run as falsified rather than re-run it: the accuracy and
exact-reproduction criteria are the ones that test the objective, the masks and the
decoder, and both pass without qualification, while a third attempt to make the
likelihood criterion pass would be tuning the run to the criterion. For later runs a
loss-reduction criterion under a spread target has to be stated against that target's
own floor, or left out; the corpus run's criteria are render recovery against two
policies, a magnitude bar and validity, and do not include it. Gate L proceeds to corpus
scale.

## 2026-09-21 — Gate L at corpus scale: the model beats the corpus and loses to the hole

**Hypothesis.** Matched to v16 and to the causal arm on every term - 526,498 parameters,
the same hashed corpus and family-disjoint splits, 100,800 icon presentations - with
dropout 0.1 from the start, the masked model should draw a removed path back into a
held-out icon better than leaving the hole and better than a position-marginal policy
decoded through the same grammar, with the 95% interval excluding zero on both, a
median recovery of at least 0.30, and every completion valid.

**Observation. Falsified.** It beats the marginal policy cleanly - on **54 of 64** icons,
mean paired difference +0.0112 RGBA MAE, interval [0.0066, 0.0158] - and it is **worse
than leaving the hole on 57 of 64**: mean −0.0040, interval [−0.0069, −0.0012], median
recovery −0.69 against the 0.30 bar. Every completion is valid. Held-out masked
likelihood is 3.869 nats against the floor's 4.111, a ratio of 0.941, flat from step
600 while the training loss is flat from step 300; selection at 3,300, early stop at
5,700. On gpubox-4080: 346 s to train, 102 s to prepare, 1.95 GiB peak.

The sheet says the same thing the numbers do. The fills are in the right colours and
often in roughly the right region, and they are the wrong shape: a stroke where a fill
should be, a blob across a face, a line through the hole rather than the piece that was
there. The marginal policy's fills are worse in the same way.

**Where it learned and where it did not.** A read-only breakdown of the checkpoint by
mask family and token type, model against the floor:

| tokens under a whole-path mask | held-out accuracy | floor | training icons | floor |
| --- | ---: | ---: | ---: | ---: |
| style fields | 0.886 | 0.868 | 0.911 | 0.859 |
| segment kinds | 0.676 | 0.585 | 0.697 | 0.636 |
| start points | 0.043 | 0.027 | 0.126 | 0.058 |
| segment coordinates | 0.023 | 0.010 | 0.073 | 0.054 |

The argmax coordinate sits a mean 54 bins from the truth held out and 44 on training
icons, out of 289. Whole-path inpainting on **training** icons helps **0 of 24**. So this
is not Gate I's failure. The causal model memorised its training set and generalised
nothing; this model does not fit coordinates at corpus scale at all, having memorised
four icons exactly in the overfit test. What it learned is style co-occurrence, which
the floor already knows, and a little about which segment kinds follow which.

**Decision.** Record the falsification and test one factor, chosen by the diagnosis
rather than by the menu. Capacity and the mask mixture are candidates, but the training
curve says the model found the floor's solution by step 300 and stopped, on training
data it sees thirty times over, and that is the signature of information it cannot
reach rather than of parameters it lacks. In the packed layout a segment's owning path
and its index within that path are never given to the encoder: a hole at slot 57 could
belong to any path, and the model has to count `path_length` tokens to find out. The
denoiser had the same defect class as slot blindness, and 768 parameters of binding
beat a 3.5x capacity increase outright. The next arm binds each segment to its path and
its place in it, derived from the visible lengths, and changes nothing else. It runs
through the overfit test first, as every model change here has.

## 2026-09-21 — Path binding passes the plumbing test

**Hypothesis.** With every position carrying its owning path's index and every segment
its index within that path, the model should still memorise four icons exactly through
its masks and its grammar-ordered decoder.

**Observation. Passed** on all three predeclared criteria: masked-token accuracy 1.000,
4 of 4 masked whole paths reproduced token for token with pixel-identical renders,
every completion valid. Memorisation is a touch faster - 0.992 by step 1,800 against
the unbound model's 0.999, both 1.000 by 2,700 - so at four icons binding costs nothing
and buys little, which is what a plumbing test should show. 112 s, 0.50 GiB peak,
542,050 parameters.

**Decision.** The corpus half runs: l2 with binding on and nothing else changed - the
same corpus, splits, budget, masks, 64 held-out icons and removed paths - read on the
same criteria l2 was falsified on.

## 2026-09-21 — Binding changes nothing, and the model is not reading its neighbours at all

**Hypothesis.** l2 sat at the position-marginal floor for coordinates on its own training
icons while memorising four icons exactly, so the suspect was information the packed
layout withholds: which path a segment belongs to and where in that path it sits. The
bound arm gives every position both, derived from the visible lengths, and changes
nothing else.

**Observation. Falsified, and indistinguishable from l2 on every measure.** Held-out
masked likelihood 3.843 against 3.869, accuracy 0.274 against 0.266, 8 held-out icons
helped against 7, worse than the hole on 56 of 64 with the interval [−0.0069, −0.0023],
and icon by icon the two arms differ by 0.0006 RGBA MAE with binding better on 25 of 64.
Every completion valid. Ownership was not the missing information.

**Two read-only probes then found what is.** First, the smallest geometric question a
model can be asked: hide one segment's coordinates with its kind and both neighbours
visible, and compare its predicted endpoint with the truth, in lattice bins of 289:

| policy | held-out, mean / median | training icons, mean / median |
| --- | ---: | ---: |
| position-marginal argmax | 145 / 143 | 145 / 147 |
| **copy the previous endpoint** (zero parameters) | **24.5 / 16.5** | **25.4 / 16.0** |
| l2, unbound | 54.8 / 49.2 | 48.0 / 43.5 |
| l4, bound | 56.4 / 52.2 | 39.4 / 34.0 |

Both models learned coarse location - far better than the marginal - and neither
learned continuity: a hidden segment ends near where the last one did, and a policy
that knows only that beats both by more than two to one, on the icons they trained on.
Second, the probe that explains it: shift the visible previous endpoint by −40 to +40
bins and watch the prediction for the hidden one. It moves by a **median of 0.0** bins
at every shift. The model does not read coordinate context at all. It predicts each
coordinate from its position, the kinds and the styles, and nothing a neighbour says
changes its answer.

**Candidate causes, and what a computation says about the first.** The coordinate
target is `exp(−|b − t| · 0.25 / 1.0)`, one view unit wide, chosen because it was
decisive for the denoiser - whose corrupted input already sat within a few bins of the
truth. Here nothing does, and a target that narrow could leave coarse localisation
without a gradient. But that is not what the loss actually does for a prediction that
is itself spread: a bump eight bins wide scores 4.78 nats sixteen bins off and 7.77
forty bins off under the narrow kernel, a gap of 3.0, and 5.89 against 8.26 under an
equal mixture of one- and eight-unit kernels, a gap of 2.4. The narrow kernel already
distinguishes a coarse hit from a wild miss once the prediction is not a spike. So the
kernel is a candidate, not a finding. The other candidates: the training mixture, in
which seven of every ten masked coordinates have no visible neighbour in their own path
and the signal for continuity is diluted by tasks that are close to generation; the
output head, a linear map from a 96-wide state to 289 unordered bins, which has to
discover a Fourier basis before it can place a bump wherever a neighbour says; and the
input path, which a new test now pins at initialisation.

**Decision.** Discriminate before building. The next run trains the same model on
single-segment masks alone - the continuity task and nothing else - for a fraction of
the corpus budget, and asks one predeclared question: does its predicted endpoint beat
the copy-the-previous-endpoint policy? If it does, the corpus arms failed on their
training distribution and the fix is the mixture. If it does not, the loss and the head
are next, one at a time. The continuity probe is measured by the harness for every run
from here, model against the copy policy, so that every arm is read on the mechanism
and not only on the render.

## 2026-09-21 — Continuity as the only task: still not learned, so the mixture is not it

**Hypothesis.** If the corpus arms failed to learn continuity because seven of ten
masked coordinates in their training mixture had no visible neighbour, a model trained
on single-segment masks and nothing else should beat the copy-the-previous-endpoint
policy on that one task.

**Observation. Falsified.** Over one hidden segment in each of 339 held-out icons, the
model's predicted endpoint is **59.7 bins** from the truth (median 54.5) against **25.6**
(median 17.0) for the copy policy and 140 for the marginal argmax. Held-out masked
likelihood is 4.532 nats against the floor's 4.787, a ratio of 0.947, and the training
loss is flat from step 300 at about 4.5. The mechanism is not merely diluted in the
mixture; it does not appear when the mixture is removed.

**Decision.** The mixture is eliminated as the cause. Two candidates remain and the
output head is the more specific: it is a linear map from a 96-wide state onto 289
unordered bins, so to place a bump at a value a neighbour supplies, attention must learn
to copy that value and the head must independently learn a Fourier ordering over the
bins, and neither is rewarded until the other exists - a trap the position prior sits
comfortably outside of. The metric head makes a coordinate logit the inner product of a
projection of the state with the input side's Fourier features of that bin, plus the
categorical bias, so any state that carries a copied value produces a bump at it from
the first step. It runs through the overfit test and then the same continuity study,
one change from arm 5. If it fails too, the target kernel is next.

## 2026-09-21 — The metric head through the plumbing test: the decoder passes, exact bins lag

**Hypothesis.** The model with the metric output head should still memorise four icons
exactly through its masks and its grammar-ordered decoder.

**Observation. Falsified as written, on one of three criteria.** Greedy completion
reproduces **4 of 4** masked whole paths token for token with pixel-identical renders,
every completion validates, and on the four training icons the continuity probe's error
is **0.0 bins** against 28 for the copy policy. Masked-token accuracy under the fixed
evaluation masks reaches 0.958 against the 0.99 bar, still rising at the last four
evaluations (0.907, 0.919, 0.961, 0.958) where the categorical head reached 1.000 by
step 2,700.

**Reading.** The miss is what the head is, not what it does wrong. Its logits are a
projection of the state against eight Fourier frequencies of each bin's value, and the
finest of those has a period of 4.5 bins, so the bump it places is broad and pinning
an exact bin under a memorisation mask is slower than for a free 289-way head. That
resolution is irrelevant to the question the head was built for - whether the model
can put a bump within tens of bins of a value a neighbour supplies - and the exact-
reproduction criterion, which drives the whole decoder against a known target, passes.

**Decision.** Record the falsification, do not re-run to meet the bar, and proceed to
the continuity half on the strength of the criterion that exercises the pipeline. If
exact-bin precision is ever the question, more frequencies are one number in a config.

## 2026-09-21 — Arm 7 tested nothing: the head was never on at a masked kind

**Observation.** The metric-head continuity study reproduces arm 5's held-out trace to
four decimals at every evaluation - 4.612, 4.596, 4.589, 4.571, 4.552, 4.536, 4.532 -
and its training losses too. The change under test never took effect. The head decides
a coordinate position's role - endpoint or control handle, which fixes the bin-to-value
map - from the kind token in the *input*, and in the span family the kind is hidden
together with its coordinates, as it is in the path family. So at every position the
run trained on, the head found no role and fell back to the categorical logits. Its
1,746 parameters were carried and never used; the continuity probe, which leaves the
kind visible, then drove an untrained projection and scored 64.6 bins.

The overfit half was affected the same way, and passed the decoder criterion because
the decoder commits kinds before coordinates, so at decode time the head was on. That
is also the tell: training and decoding disagreed about when the head applies.

**Decision.** The head takes its roles from the clean kinds under teacher forcing -
exactly as the loss already takes its legal masks from the clean sequence - and from the
committed kinds at decode time, which the tier order guarantees are present. A test now
pins that the head fires at a coordinate whose kind is masked when the kinds are
supplied. Arm 7 is preserved as a harness defect and both halves re-run on the fixed
revision. The lesson is the same one the first overfit test taught: a change can be
carried by a run without being exercised by it, and identical traces are the sign.

## 2026-09-21 — The head is not the trap either, and the causal model says where it is

**Hypothesis.** With coordinate logits projected onto the lattice's Fourier basis, so
that a copied neighbour value becomes a bump without the head first learning an
ordering over 289 bins, the model trained on single-segment masks should beat the
copy-the-previous-endpoint policy.

**Observation. Falsified.** This time the change was exercised - the untrained trace
starts at 6.08 nats against arm 5's 5.20, and the overfit half passed all three of its
criteria at accuracy 0.996 - and the model converges to the same place regardless:
**58.3 bins** against the copy policy's 25.6, held-out likelihood 0.947 of the floor,
training loss flat from step 300.

**What four arms now say together.** Ownership binding, the training mixture and the
output head each changed nothing about whether the model reads its neighbours, and a
40-bin shift of the visible previous endpoint still moves nothing. The model that fails
this is the same model that memorises four icons exactly, so the fault is in what is
learnable at corpus scale, not in capacity. The causal arm of Gate I is the useful
comparison: it drove its training loss to 1.96 and cleared the floor by 8% held out,
and it never had to *find* the previous coordinate - under the causal shift the token
before the one being predicted is the input at the prediction position itself, so
"next is near previous" is a direct map from input to output. Here the previous
endpoint sits seven positions away, in a slot that depends on the previous segment's
kind, behind learned absolute embeddings over packed slots whose contents shift from
icon to icon, and attention has to discover a fetch that is rewarded only once it
exists. The position prior needs none of that and is found by step 300.

**Decision.** Make the masked model's input as local as the causal model's. The codec
chains coordinates - every segment starts where the previous one ended, and a path's
first segment at the header's start point - and that start is already in the sequence,
so it becomes an explicit input feature of each segment block: Fourier features of the
start point when it is visible, a flag when it is not. Nothing about the loss, the head
or the masks changes. If the model then beats the copy policy, the position machinery
was the blocker and the fix is a principled one; if it still cannot, with the value in
its own input, the loss and head are back on the table with the evidence narrowed to
them.

## 2026-09-21 — Start features through the plumbing test: a chained input wants a chained decoder

**Hypothesis.** With each segment block carrying Fourier features of its own start
point, the model should still memorise four icons exactly through its masks and its
grammar-ordered decoder.

**Observation. Falsified as written, on one of three criteria.** Masked-token accuracy
reaches 1.000 under the fixed masks, every completion validates, the continuity probe's
error on the training icons is 0.0 bins, and greedy whole-path completion reproduces
**3 of 4** paths token for token, the fourth rendering at 1.0 recovery all the same.

**Reading.** The decoder, not the training. The coordinate tier commits in confidence
order, in a few parallel passes, and a model whose prediction for a segment depends on
the previous segment's endpoint can be asked to commit that segment while its start is
still a hole; when the hole is later filled the two disagree, and a memorised path comes
back with a token out of place. Single-segment masks never meet this - their start is
always visible - so the continuity half proceeds. For whole-path completion the fix is
the obvious one: commit in chain order, every masked segment whose start is known, each
pass, until none is left. It is built, pinned by a test that no segment is ever
committed while the one before it is a hole, and the overfit test re-runs with it once
the continuity half has the GPU.

## 2026-09-21 — Start features move it: the first curve in Gate L that keeps falling

**Hypothesis.** With each segment block carrying Fourier features of its own start
point, the model trained on single-segment masks should beat the copy-the-previous-
endpoint policy.

**Observation. Falsified, and the first arm to move anything.** Continuity error is
**47.8 bins** (median 38.5) against 58–60 for every arm before it, held-out likelihood
**4.119** nats against 4.532 - 0.861 of the floor where every earlier arm sat at 0.947 -
and the held-out curve is still falling at the last evaluation, 4.393 → 4.291 → 4.228 →
4.183 → 4.158 → 4.141 → 4.119, where every earlier arm was flat from step 300. The copy
policy sits at 25.6. Same corpus, same masks, same budget, same head, same kernel.

**Reading.** The blocker was the fetch. Five arms could not make attention find the
previous endpoint seven slots away behind absolute embeddings over packed slots; put the
value in the segment's own input and the model starts to use it within the same
budget. That it is unconverged at 2,100 steps and still far from a policy that just
copies the value is the next fact, not the last: the value now has to pass from Fourier
features on the input side to a bump on the output side through a categorical head that
has to learn its own ordering - the thing the metric head was built to remove, and could
not show while the value was not there to copy.

**Decision.** Two one-factor follow-ups, chained behind the overfit re-run. The same run
at the full 6,300-step budget, to see where the curve goes; and the same run with the
metric head, at the same 2,100 steps, to see whether copying becomes the linear map it
should be. Whichever wins is the configuration the corpus inpainting run repeats with,
decoded in chain order.

## 2026-09-21 — Correction: the fourth path was one bin off, and that is the target, not the decoder

**Observation.** Decoded in chain order, the start-features overfit test is unchanged:
accuracy 1.000, every completion valid, 3 of 4 masked paths exact, all four renders
pixel-identical. The one differing token is a single coordinate **one bin off** - a
quarter of a view unit - with the truth at probability 0.116 and the decoded neighbour
at 0.100.

**Reading, corrected.** The previous entry read the miss as the decoder committing a
segment before its start; that was wrong, and the chain-order re-run shows it. Under
the one-unit soft target the truth and its adjacent bins are *meant* to be nearly
equiprobable, so exact-bin reproduction of a memorised path is a coin flip at every
coordinate the model has learned the target well at, and l1 and l3 passing 4 of 4 was
the draw going the other way. A quarter-unit miss is invisible in the render, which is
why the four sheets are identical to the clean icons.

**Decision.** Withdraw the decoder-order reading; keep chain-order decoding, which is
the right decoder for an input that hangs each segment from the one before it and was
worth building regardless. Give the overfit criterion a one-bin tolerance on
coordinates - every other token exact - so that it tests the objective, the masks and
the decoder rather than the spread of the target, and record that the exact rate is
still reported beside it.

## 2026-09-21 — With a value to copy, the metric head shows

**Hypothesis.** With each segment's start point in its own input, the metric head should
make copying it a linear map, and the continuity-only model should beat the copy policy.

**Observation. Falsified, and a further move.** Continuity error **41.7 bins** (median
37.0) against arm 10's 47.8 and the 58–60 of every arm before that; held-out likelihood
**4.005** nats, 0.837 of the floor, against 4.119; the curve still falling at the last
evaluation, 4.327 → 4.169 → 4.128 → 4.074 → 4.040 → 4.022 → 4.005. The copy policy sits
at 25.6. Same 2,100 steps as arm 10, so the head is the only difference.

**Reading.** The head that could show nothing while there was nothing to copy shows
now. Two levers, each measured alone against a one-factor baseline, each helping, and
neither run converged at a third of the budget - which is the situation the record
should not be read past. Arm 12, the start-features run at the full budget, is on the
GPU; the combination at the full budget is queued behind it, because it is the
configuration the corpus run would use whichever way arm 12 reads.

## 2026-09-21 — At the full budget, start features nearly reach the copy policy

**Hypothesis.** Arm 10's still-falling curve, given the full 6,300 steps, should reach
the copy-the-previous-endpoint policy.

**Observation. Falsified by 3.5 bins, still falling.** Continuity error **29.1 bins**
(median 22.0) against the copy policy's 25.6 (median 17.0); held-out likelihood
**3.682** nats, 0.769 of the floor, from 4.119 at 2,100 steps; the checkpoint selected
at step 6,300 itself, early stopping never having fired - 3.734, 3.719, 3.683, 3.707,
3.684, 3.682 over the last six evaluations, flattening but not flat.

**Reading.** From 58–60 bins with nothing in the input, to 47.8 at a third of the
budget, to 29.1 at the full budget, by putting one value where the model can see it. The
five arms before arm 10 measured a model that could not begin; this one is a model that
has not finished. The head, which took a third-budget run from 47.8 to 41.7, is running
at the full budget as arm 13, and the corpus inpainting configuration that either
outcome leads to - start features, the head, chain-order decoding - is drafted.

## 2026-09-21 — Gate L's first pass: the model reads its neighbours

**Hypothesis.** Start features and the metric head together, given the full budget,
should beat the copy-the-previous-endpoint policy on the continuity task.

**Observation. Passed.** Over one hidden segment in each of 339 held-out icons the
model's predicted endpoint is **17.9 bins** from the truth (median 10.5) against the
copy policy's 25.6 (median 17.0) and the marginal's 140. Held-out masked likelihood is
**3.309** nats, 0.691 of the floor, and still falling at step 6,300, where the checkpoint
was selected; every completion valid. 419 s to train, 1.9 GiB peak, 531,892 parameters.

**The sequence, in one line each.** Arms 5 and 7b, nothing in the input: 58–60 bins.
Arm 10, the start point in the input, a third of the budget: 47.8. Arm 11, with the
head: 41.7. Arm 12, start features at the full budget: 29.1. Arm 13, both at the full
budget: **17.9**. Each step a single change against a one-factor baseline, on the same
339 icons and the same hidden segments.

**Reading.** This is the first predeclared criterion Gate L has met at corpus scale and
the first time in this project that a model reads its geometric context better than a
zero-parameter policy - the thing v16 never did, since the identity policy it beat was
about leaving tokens alone rather than reading them. The mechanism was never capacity or
data: the same 530k parameters that sat at the floor for five arms learn continuity as
soon as the value they need is in their input and the head can place a bump where it
says. Chain-order decoding hands the model each start before it draws from it, so at
inference the mechanism applies to whole paths as well as single segments. That is what
arm 14 measures: l2's inpainting run, on l2's icons and criteria, with this mechanism.

**What it does not say.** Whole-path completion still asks for the shape of a missing
part, which Gate I says this corpus cannot teach for unseen concepts; continuity gets a
path started and keeps it coherent, and may or may not get it to beat the hole. The
scored task is read next.

## 2026-09-21 — With the mechanism, whole-path completion stops damaging and still loses to the hole

**Hypothesis.** l2's inpainting run, with each segment's start point in its input,
lattice-aware coordinate logits and chain-order decoding, should draw a removed path
back into a held-out icon better than leaving the hole.

**Observation. Falsified, and a different model from l2.** Against the hole the mean
paired difference is **−0.0011** RGBA MAE, interval [−0.0017, −0.0006], 6 of 64 icons
helped, median recovery −0.11 - against l2's −0.0040, 7 helped and −0.69. Against the
marginal policy +0.0141 with the interval excluding zero, on **all 64** icons. Icon by
icon on the same removed paths it beats l2 on **44 of 64**, by 0.0029 on average. Every
completion valid. The sheet shows what the numbers say: the fills are quiet - a small
correct piece here, a short stroke there - rather than the scribbles l2 drew across
faces and flags.

**Two readings, both supported.** The shape of a missing part is what this corpus
cannot teach for an unseen concept - PROJECT_PLAN.md section 15 said so before the
run, and Gate I said it for whole icons - and a model that knows it cannot draw the
part does the next best thing, which is very little. That is not an editor for whole
paths, and it is no longer a model that makes things worse.

And the mechanism is diluted by the mixture it was trained under: continuity error is
**41.8 bins** here against **17.9** for the same architecture trained on single-segment
spans alone (arm 13), and the held-out curve flattened by step 900 where arm 13's fell to
the end. Arm 5 showed the mixture was not the cause when there was no mechanism; now
that there is one, the mixture is what limits it.

**Decision.** Move the scored task to the local edit, as section 15 provided for: a run
of two segments hidden inside a path with a visible segment before and after it, scored
by paired render recovery against a zero-parameter fill that joins the visible ends,
the marginal policy, the same magnitude bar and validity. Read both checkpoints on it
without retraining - arm 14's mixture model and arm 13's span specialist - so that the
choice between a general editor and a specialist is measured on the same 64 icons. The
whole-path result stays reported as the hard case.

## 2026-09-21 — The local edit: the specialist ties the join, the mixture model loses to it

**Hypothesis.** Read on two-segment spans hidden inside held-out paths with visible
neighbours, at least one of the two checkpoints - arm 14's mixture model, arm 13's span
specialist - should complete the span better than a zero-parameter fill that joins the
visible ends, and better than the marginal policy, with the magnitude bar met.

**Observation. Both falsified as written, and they split.** The mixture model is worse
than the join: mean −0.0048 RGBA MAE, interval [−0.0074, −0.0023], 9 of 64 helped,
median recovery −1.35. The specialist - trained on single-segment masks and read on
two-segment spans it never saw - is at **parity** with the join: mean **+0.0011**,
interval [−0.0004, +0.0026] spanning zero, **26 of 64** helped, median recovery −0.035,
and its median error, 0.00112, is below the join's 0.00146. Against the marginal policy
the specialist wins on 60 of 64 with the interval excluding zero; the mixture model on
40. Every completion valid. On the specialist's sheet the model's fills are
indistinguishable from the clean icons in nearly every row, where the join shows its
straight cut.

**Reading.** Two things the numbers settle. The join is a strong identity policy for a
two-segment span - its median error is a tenth of the whole-path hole's - so parity
with it is not nothing, and the magnitude bar of 0.30 of that small error is a hard bar
that the record keeps rather than lowers. And the specialist, whose continuity error is
17.9 bins against the mixture model's 41.8, is the one that carries to the edit: the
mixture dilutes the mechanism, exactly as arm 14's continuity said it would.

**Decision.** The specialist is the line. Its one mismatch with the task is the span
length it trained on; the next arm trains it on spans of one to three segments at the
same budget, with the same mechanism, and reads it on the same 64 icons and
two-segment spans. The whole-path result stays reported as the hard case.

## 2026-09-21 — The specialist beats the join on paired error, and misses the median bar

**Hypothesis.** Trained on spans of one to three segments, the span specialist should
complete two-segment spans of held-out icons better than joining the visible ends, with
the interval excluding zero, the magnitude bar met, and every completion valid.

**Observation. Falsified as predeclared, with both paired tests passed.** Against the
join fill the mean paired difference is **+0.0092** RGBA MAE, interval [+0.0004,
+0.0179] excluding zero - the first time in this gate a model beats the identity policy
on the scored task. Against the marginal policy +0.0171 on 58 of 64. But **26 of 64**
spans helped and the median recovery is −0.03 against the 0.30 bar, so the run is
falsified on the criterion that asked for a typical gain rather than a mean one. Every
completion valid. Continuity 19.9 bins against the copy policy's 25.6; held-out
likelihood 0.782 of its floor and still falling at step 6,300.

**Reading.** The distribution is the finding. On most two-segment spans a straight cut
between the visible ends is already nearly invisible at 72 px, and the model's fill is
equally near-perfect - the median error is 0.0017 against the join's 0.0026, and the
sheet shows the two columns agreeing almost everywhere. On the spans where the cut is
destructive the model wins outright: the "UP!" badge loses its whole fill under the
join and comes back green under the model, and a handful of such cases carry the mean.
An editor that is harmless where a straight cut would do and right where it would not
is the product this gate set out to find; a criterion asking for a 30% typical gain
against a baseline that is already right most of the time was the wrong instrument,
and the record keeps it as falsified rather than moving it.

**A correction the record owes.** The config states this run reads the same 64 icons
and spans as arms 15 and 16. The icons are the same; the spans were drawn with this
run's training seed rather than theirs, and only 2 of 64 coincide. The frozen-
checkpoint mode makes the like-for-like reading a two-minute run, arm 18.

## 2026-09-21 — Span for span: parity with the straight cut, and the task moves to longer spans

**Hypothesis.** Read on exactly the spans arms 15 and 16 used, the specialist trained
on spans of one to three should beat the join fill, and the single-segment specialist.

**Observation. Falsified; a tie on both counts.** Against the join, mean **+0.0007**,
interval [−0.0016, +0.0030] spanning zero, 30 of 64 helped, median recovery **0.0**.
Against the single-segment specialist, span for span: 31 better, 30 worse, 3 equal,
mean −0.0004. Against the marginal policy: 58 of 64 with the interval excluding zero.
Every completion valid.

**Reading.** Arm 17's paired win against the join was a property of its draw, and it
does not replicate on a second one. Across the 128 two-segment spans now measured the
picture is stable and worth stating plainly: the specialist is at parity with a
straight cut at the median, wins outright where the cut is destructive - `1F1FC`
+0.035, `1F62B` +0.021, the "UP!" badge +0.020 - loses badly once (`1F4DC`, −0.052), and
beats the marginal policy every time. A two-segment span is a small edit; at 72 px a
straight line between its ends is usually invisible, and no fill can beat invisible.
That is not the editor's failure; it is the task being too easy for the baseline to
lose.

**Decision.** Move the span to four segments, where the join visibly cuts corners, and
keep everything else: the same 64 icons, the same criteria, the draw pinned by
`inpaint_seed`. Read arm 17's checkpoint there first, as a frozen model on spans it
trained short of, and then a specialist trained on spans of one to four at the same
budget. The two-segment results stay reported as what they are.

## 2026-09-21 — Four-segment spans, the shorter-trained specialist: parity again

**Observation.** Read on four-segment spans of the 56 held-out icons with a path long
enough, arm 17's checkpoint - trained on spans of one to three - ties the join fill:
mean +0.0002, interval [−0.0010, +0.0014], 26 of 56 helped, median recovery −0.01, the
model's median error 0.0046 against the join's 0.0054. It beats the marginal policy on
49 of 56 with the interval excluding zero. The recovery distribution is wide and centred
on zero: the tenth percentile −1.6, the ninetieth +0.48.

**Reading.** At a length where the straight cut visibly loses corners, a model trained
short of that length neither wins nor harms. Arm 20, trained on spans of one to four,
is read on the same spans next. And the one lever every curve in this gate has pointed
at and no arm has pulled is training length: held-out likelihood was still falling at
step 6,300 in every corpus run since the mechanism arrived, and the checkpoint was
selected at the last step each time. A specialist trained three times longer is queued
behind arm 20 on the same spans.

## 2026-09-21 — Trained for the length: parity a third time, with a tighter tail

**Hypothesis.** A specialist trained on spans of one to four, read on four-segment spans,
should beat the join fill where a model trained short of the length only tied it.

**Observation. Falsified; parity.** Mean −0.0006 against the join, interval [−0.0016,
+0.0005] spanning zero, 24 of 56 helped, median recovery −0.01; the marginal policy
beaten on 49 of 56 with the interval excluding zero; span for span against the
shorter-trained model, 30 better and 24 worse, mean difference under 0.001. What did
move is the tail: the tenth percentile of recovery is −0.48 against the shorter model's
−1.58, so the bad cases are less bad, while the centre stays on zero. Held-out
likelihood 0.809 of its floor and still falling at step 6,300; every completion valid.

**Reading.** Three readings at two span lengths with three checkpoints say the same
thing: at this corpus size the specialist's local completion is as good as a straight
cut between the visible ends, and no better, at the median - with the wins and the
losses in the tails roughly balancing. The two levers not yet pulled are the ones every
curve points at: training length, since every corpus run since the mechanism arrived
selected its last step with the curve still falling; and capacity with regularisation,
the one cell no gate in this project has measured. Arm 21 is the first, on the GPU;
arm 22 is the second, queued behind it, at ten times the parameters with the same
dropout. If neither moves the centre, the gate concludes on parity with the mechanism
result standing.

## 2026-09-21 — Three times longer: the mechanism keeps improving, the edit does not move

**Hypothesis.** Every corpus run since the mechanism arrived selected its last step with
the held-out curve still falling; trained three times longer, the specialist should
beat the join fill on four-segment spans.

**Observation. Falsified; parity a fourth time.** Held-out likelihood keeps improving -
**3.593** nats, 0.754 of the floor, selected at step 15,600 of 18,000 - and continuity
reaches **17.6 bins** against the copy policy's 25.6, the best of any run. The edit ties
the join: mean −0.0002, interval [−0.0023, +0.0018] spanning zero, 22 of 56 helped,
median recovery −0.045, and span for span against the 6,300-step model 24 better and
32 worse. The marginal policy beaten on 49 of 56. Every completion valid. 1,199 s to
train.

**Reading.** Training length is not the lever, and the way it is not is the finding.
The two measures the mechanism was built on - masked likelihood and endpoint continuity
- both keep improving with more steps, and the paired render error against a straight
cut does not follow them. So what remains between the model's fill and the join is not
something the model gets closer to with more of what it is learning. Two readings are
consistent with that: on most spans the straight cut is already right to within the
render's resolution, and there is nothing to gain; and on the spans where it is wrong,
what is missing is which of several plausible shapes the hidden run took - which the
visible neighbours do not determine and the corpus cannot teach for an unseen icon. The
recovery distribution says both: centred on zero, with a tenth percentile of −3.3 and a
ninetieth of +0.31.

**Decision.** One lever remains - capacity with regularisation, the cell no gate here
has measured - and it is on the GPU as arm 22. If it does not move the centre either,
Gate L concludes on parity with the mechanism result standing, and the record says
what an editor at this corpus size is: harmless where a straight cut would do, right
where continuity decides, and no better than a guess where shape does.

## 2026-09-21 — Gate L concludes: an editor as good as a straight cut, and a mechanism that is real

**The last lever.** Capacity with regularisation - ten times the parameters, 5,358,516
against 531,892, dropout unchanged - ties the join on four-segment spans like every arm
before it: mean −0.0009, interval [−0.0032, +0.0013], 28 of 56 helped, median recovery
−0.001, span for span against the small model 28 better and 27 worse. The marginal
policy beaten on 49 of 56. 1,236 s to train, 3.9 GiB peak. Every completion valid.

**What twenty-two arms established, in order.** A masked model over this codec cannot
read its neighbours as built: five arms at the floor for geometry on its own training
icons, a 40-bin shift of a visible endpoint moving its prediction by a median of zero.
The cause was the fetch and the head together, not capacity, data, ownership, mixture
or kernel: put each segment's start point in its own input and project coordinate
logits onto the lattice's Fourier basis, and the same 530k parameters learn continuity
- hidden-segment endpoint 17.6 to 19.9 bins from the truth against 25.6 for a policy
that copies the previous endpoint, on 339 held-out icons, in every run that carried
the mechanism. That is the first learned geometry in this project and it is robust.

Carried into editing, the mechanism does what continuity can do and no more. Whole-path
completion: no longer damaging - the gap to the hole a quarter of l2's - and still
short of it, because the shape of a missing part is what the corpus cannot teach for an
unseen icon. Span completion at two and four segments, five readings on pinned spans
with three checkpoints, a budget three times longer and a model ten times larger: parity
with a zero-parameter fill that joins the visible ends, interval spanning zero every
time, wins where the cut is destructive, losses where the hidden run had a shape the
neighbours did not determine, and the marginal policy beaten on 49 of 56 or better every
time. The mechanism's own measures kept improving through all of it - likelihood to
0.754 of the floor, continuity to 17.6 bins - and the render-level edit did not follow.

**Verdict.** PROJECT_PLAN.md section 15 asked for an editor that completes held-out
icons visibly and measurably, or a negative result that names which mask family fails
and why. The answer is both halves at once. What exists is an editor that is harmless
where a straight cut would do, right where continuity decides, and no better than a
guess where shape does - always better than the corpus's statistics, never better than
the simplest geometric policy at the median. The family that fails is any whose answer
is a shape: a whole path, or a span long enough that its interior is not determined by
its ends. Why: 2,681 icons with 1,597 singleton families teach continuity and style,
and do not teach what an unseen icon's parts look like - the same fact Gate I measured
for whole icons, now measured for parts.

**What the gate cost and caught.** Twenty-two registered arms in about five hours of
GPU on gpubox-4080, seven minutes each at 1.9 GiB for the small model. Two harness
defects caught by overfit tests before a corpus run could read them (a soft target
centred a token low; a head that never fired at a masked kind), two criteria found to
measure the target rather than the model (loss reduction under a spread target; exact
bins under a spread target), one evaluation draw that inherited a training seed, and
one decoder-order reading that was wrong and withdrawn. Every one is in the registry
under its own identity.

**Decision.** Gate L is complete. The frozen specialist from arm 21 - the best
continuity, 17.6 bins - is the checkpoint Gate K's editing viewer is built over, showing
what the editor does and does not do beside the join and the hole. Gate H, broader
licensed corpora, is the only route to shape knowledge, for parts as for wholes, and is
the operator's call: it is a licensing and curation project before it is a training one.
No further single-factor arm on this corpus is recommended; the levers are measured.

## 2026-09-22 — Gate M begins: OmniSVG draws an apple zero-shot; the 0.8B cannot route

**OmniSVG 1.1 4B loads and draws.** Its 7.7 GB checkpoint carries an older key layout
under a `transformer.` wrapper; remapped, all 825 tensors load into this transformers
with none missing, at 7.3 GiB on the GPU. Prompted zero-shot with "a red apple with a
green leaf, flat emoji style", both samples are a recognisable flat apple with a leaf
- the first recognisable generated icon in this project's history, from a model that
has never seen OpenMoji. "fish" and "hedgehog" hit a 1,024-token cap I set too low and
came out truncated; the model's own setting is 1,536 and icons run to 2,048. About 15 s
per 1,024 tokens. Fills only, on a 200-unit box, with a non-standard `filling`
attribute that the project's normalizer must strip.

**SemIf on Qwen3.5-0.8B does not route.** Over the 52 owned editing requests the direct
readout scores 0.404: recolour 1.0, ask 0.57, restyle 0.2, generate 0.1, simplify 0.0,
with 23 of 38 actionable requests sent to "ask", and the choice survives reversing the
option order on only 81%. Falsified on all three predeclared criteria. The readout is
real - recolour requests are found every time - but the model is too small to carry
five options; the handoff anticipated escalating to the 4B instruct model, which is
fetched next.

**Qwen3.5 needs its kernels.** With the reference linear-attention path a LoRA step on
the 2B base ran out of memory at 1,024 tokens on 16 GB; the published wheel of
flash-linear-attention ships without its ops, and the source build installs and
imports. Timing follows.

## 2026-09-22 — SemIf on the 4B: never misses an "ask", asks far too often, and follows the last letter

**Observation.** The 4B instruct model through the same one-forward readout scores
0.577 over the 52 owned requests: ask 1.0, recolour 0.7, restyle 0.5, simplify 0.25,
generate 0.2; 22 of the 38 actionable requests are sent to ask; and the choice survives
reversing the option order on only **50%** of decisions. Falsified on accuracy and
order invariance; the clarify-recall criterion passes trivially because everything
tends to ask.

**Reading.** The choice counts by order say what is happening. In the declared order
ask is the last letter and is chosen 36 times; in the reversed order the last letter is
generate, generate rises from 2 to 11 and ask falls to 10. The readout is following the
final position more than the meaning, a known failure of letter readouts that the
reference method's own results discuss under perturbation stability. The cheapest
correction, averaging the two orders' probabilities per option, uses the same two
forwards; it is added to the runner as a secondary readout, reported beside the direct
one and not a criterion, and both models are re-read with it.

**Meanwhile, OmniSVG.** The zero-shot control on 32 held-out annotations is running:
two samples each at 2,048 tokens, decoded, converted into the codec, rendered, scored
against the held-out render and the annotation with a pinned CLIP.

## 2026-09-21 — OmniSVG zero-shot draws emoji, not these emoji; its token language is now exact for OpenMoji

**Observation.** The zero-shot control on 32 held-out annotations, two drawings each at
2,048 tokens: every drawing decodes, 77% end within the budget, 73% enter the codec;
CLIP similarity to the held-out icon's own render is 0.811 on average, to the caption
0.219. The sheet (`reports/learning/omnisvg-m1-zeroshot/samples.png`) is mostly generic
circles and blobs in emoji colours, with a few recognisable subjects: a shopping cart
at 0.944, a tired face at 0.895.

**Reading.** The number needs its floor. Any *other* OpenMoji icon scores 0.777 against
a held-out render, and the top decile of unrelated pairs 0.838
(`clip-chance-levels.json`); the references themselves score 0.268 against their own
captions and 0.206 against unrelated ones. So 0.811 says the drawings look like
OpenMoji-style icons, not that they are these icons, and caption similarity at 0.219
is at the unrelated level. Raw CLIP similarity barely separates icons in this corpus;
the next readings therefore also rank each drawing against every held-out render and
report how often the right icon comes first (chance 1/32). This is the floor the
fine-tune is read against, and the control is re-run under that reading so the
comparison is paired on the same drawings.

**The encoder.** OpenMoji icons now encode into OmniSVG's own token language and decode
back through its released decoder point for point, colours included; a test pins it.
The training repository's YAML lists the command tokens one below where the released
decoder reads them - read at the YAML's values a close decodes as an arc, which is how
the first round trip produced arcs from icons that had none. Every training icon,
outlined, encodes under 2,048 tokens (median 647, maximum 1,935), all 2,681 of them.

**The ceiling.** Only 8 of the 32 held-out icons themselves fit the P32/T128 bucket
after outlining (`oracle-roundtrip.json`), because outlining multiplies segments. So
codec validity is capped at 0.25 for a perfect model in OmniSVG's fills-only language;
it is reported, not judged. The exact icons score 0.994 against their own renders, so
the metric has room above the control's 0.811.

**Decision.** Fine-tune OmniSVG with LoRA in its own tokens on all 2,681 training icons
under the trainer's own prompt, criteria predeclared and registered: a paired gain in
similarity over the control whose bootstrap interval excludes zero, the right icon
retrieved first for at least a quarter of the drawings, 90% of drawings ending, and no
drawing reproducing a training icon. A second control under the trainer's prompt
separates the prompt's share from the training's. All three are queued behind the
text-prior control and the SemIf re-read.
