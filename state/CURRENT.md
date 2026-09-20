# Current research state

Updated: 2026-09-20T17:20:00Z

## Current hypothesis and evidence

Gate B evidence supports retaining visually unusual OpenMoji assets rather than using
metric tails as defects. The reviewed OpenMoji 17.0.0 primary manifest has 4,006 rows:
3,902 ordinary includes and 104 visually reviewed `include_override` rows. It excludes
270 metadata flags, preserves 218 noncanonical exact-render aliases in a reversible
split, and retains one renderer-failing SVG as a defect row. Independent verification
passes raw hash/immutability, row coverage, duplicate, flag-policy, and family-split
invariants.

Gate C probe evidence makes semantic OpenMoji strokes the leading representation
candidate. On an 88-icon stratified fixture, PicoSVG outlining preserved renders closely
but expanded segments 2.241x at the median, 5.287x at p95, and 13.833x in the worst
case. At 64 segments/path, 18 outlined fixtures exceeded budget versus 7 semantic
proxies. A typed contour codec now preserves compound paint operations, semantic
stroke categories, fill rules, dashes, and painter order while enforcing canonical
padding and a safe fixed serializer. Actual typed programs confirm the structural
advantage: median semantic/outlined segment counts are 53/146, and at P48/S128 the
fixture loses 181/509 segments and 0/13 contours. Q256 improves over q128, but its
255-interval grid displaces ordinary pixel-aligned coordinates and inflates antialias
error. This is not yet a final codec selection.

The controlled aligned-grid rerun supports a quarter-unit coordinate lattice. At the
fixture coverage budget, semantic q289 lowers median q256 RGBA MAE by 48.2% at 72 px
and 42.5% at 18 px; outlined improvements are 23.9% and 30.2%. Q145 also beats q128
with a similarly sized vocabulary. Q289 is the leading coordinate vocabulary and q145
is the compact ablation.

The exact opacity extension recovers all nine formerly unsupported icons under both
semantic and outlined normalization and q289/P96/S64 typed round trips. All 18 programs
are stable with no structural loss or style approximation. The observed categorical
values are exactly `0.25, 0.4, 0.5, 0.502, 0.6, 0.9969, 0.997, 0.999, 1.0`; `0.997` is
the PicoSVG-rounded outlined form of source `0.9969`. `1F31A` and `1F31D` each retain
one out-of-bounds scalar in both routes, so 14/18 programs are strict-lossless and four
use one explicitly recorded clamp.

The opacity-aware full-primary regression now proves complete structural coverage:
semantic normalization succeeds for 3,946 icons and the unchanged reason-coded
outlined fallback recovers its 60 failures, for 4,006/4,006 hybrid success. All 7,994
non-opacity attempt rows and 3,997 previously supported hybrid rows match census v1
after removing only newly reported opacity fields; all 18 prior opacity failures are
now successes. P64/S128 remains falsified with 35 dropped contours and 493 dropped
segments across eight icons. P96/S384 is lossless for all 4,006 but only 0.2403%
slot-utilized. Gate C remains open on 37 OOB scalars in 24 icons, 347 nonexact-style
icons with 311 literal stroke widths, and the sparse fixed-capacity tail.

The complete OOB probe confirms that all 37 hybrid excursions are cubic Bezier control
handles; no visible endpoint is outside 0..72. Median clamp-only drift is small, but
the tail falsifies viewBox clamping as a safe primary rule. At 18 px, five semantic
icons exceed 0.001 RGBA MAE and two exceed 0.01. `1F4AB` is visibly broken (MAE
0.03812, alpha IoU 0.8641), while clamping the eye/speech-bubble icon removes most of
its interior (MAE 0.02015). Outlining does not repair those tail cases; this falsified
viewBox clamping and motivated the separate bounded control-handle vocabulary below.

The role-typed follow-up resolves that blocker on the complete pinned set. Q289
endpoints over [0,72] plus q417 quadratic/cubic controls over [-8,96] encode all 48
semantic/outlined programs strictly with zero projection, and every output SVG exactly
matches the prior unclamped quarter-grid counterfactual. The two catastrophic clamp
tails recover. This is the leading coordinate codec; model and corruption masks must
retain the endpoint/control distinction.

The first categorical style study derived exact weighted-relative-L1 K32 and K48 width
vocabularies from all 30,222 stroked contours, then rendered a pinned 35-icon worst-tail
fixture against an exact-style control. K48 strongly improves K32: full-corpus maximum
relative width error falls from 22.55% to 9.09%, and maximum 18 px style-only MAE falls
from 0.01264 to 0.001698. But the hypothesis of a negligible tail is false at 72 px:
K48 maps rare width 4.1 to 4.0, producing 0.01348 style-only MAE on `1F4AF`. K48 is the
statistical base, not yet the final vocabulary; the next falsifiable correction is one
exact 4.1 render-tail sentinel.

The sentinel follow-up supports the smallest correction. On the unchanged 35-icon
fixture, K48+exact-4.1 cuts maximum style-only MAE from 0.01348 to 0.002077 at 72 px and
from 0.001698 to 0.001510 at 18 px, with no new tail and byte-identical rerun artifacts.
The leading style policy is now K48+1 widths, all six observed dash patterns exact, and
five semantic miter values with near-10 literals mapped to 10. K32 remains the compact
ablation. Gate C is open only on the extremely sparse fixed-capacity policy.

The full-corpus capacity study resolves the last Gate C question. Packed P80/T1216 is
exact for all 4,006 programs with 1,296 logical slots versus 36,960 for dense P96/S384,
a 28.52x reduction. Four exact nested buckets place 3,359/553/88/6 icons and achieve
53.81% aggregate utilization. The selected representation is semantic-stroke with the
60-case reason-coded outlined fallback, role-typed coordinates, K48+1 styles, and
packed capacity. Gate C is complete; Gate D packed-codec and renderer stress remains.

Gate D now passes. Reversible packed conversion, canonical expansion validation, a
serializer-only XML allowlist, and a resource-limited subprocess renderer are
implemented. The corrected registered stress run completed 200 stable random packed
round trips, rejected 2,000/2,000 invalid mutations across ten families, classified
three malformed typed-XML cases, and completed 25 isolated renders including an actual
80-path/1,216-segment boundary program. The first mutator-boundary failure is preserved.

Gate E implementation was checkpointed and has now been run. The local CPU harness
uses the selected packed codec, a fixed-topology geometry-only bidirectional transformer,
role-aware q289/q417 corruption, four hash-pinned diverse icons, multiple disjoint
training and held-out corruptions, predeclared accuracy/loss criteria, isolated paired
renders, and exact continuous-versus-checkpoint-resumed comparison. This is explicitly a
diagnostic for geometry learnability, not evidence for topology generation or a final
diffusion process.

The first Gate E run falsified its predeclared held-out recovery thresholds while
confirming memorization and exact semantic resume. Both cases reached 100% training
accuracy, but one-icon held-out accuracy was 93.57% versus 95%, and diverse-four was
73.82% versus 85%. The 160-step diverse checkpoint reproduced the complete continuation
loss sequence and final model tensors exactly. An identical full rerun then failed the
artifact contract because legacy `torch.save` container bytes changed, so v1 is retained
as failed rather than silently accepted or tuned.

Gate F began at commit `333e149`, which defines render-safe fixed-topology contracts for
independent role-uniform geometry corruption, path-correlated geometry blocks, and
compatible whole-path donor replacement; the contracts retain paths, segment types,
styles, and typed padding. The first registered local run is the matched factorized
control. It is a custom iterative-denoiser curriculum, not an exact D3PM claim.

The first matched factorized control, `tiny-geometry-f1-factorized-333e149-e57646b7-32a80ab5`,
completed twice with byte-identical compact artifacts and exact checkpoint continuation.
Held-out one-icon/diverse-four accuracy is 99.33%/98.76%; changed-token recovery is
98.73%/96.96%. Its paired trajectory renders remain valid and recognizable at 72 and
18 px, while raw factorized `x_t` is expectedly static-like. The online training probe
is 97.98% for one icon because it is an unseen resampled batch; retain the literal
threshold miss rather than treating it as an overfit failure.

The matched path-correlated arm, `tiny-geometry-f1-path-correlated-b0983dd-c0acb759-32a80ab5`,
also reproduces exactly and remains render-safe, but trails the factorized control on
changed-token recovery: one-icon 83.56% versus 98.73%, diverse-four 93.86% versus
96.96%. This is a small fixed-topology result, not yet a final corruption choice; it
does falsify an expectation of an obvious path-correlation advantage on this fixture.

The whole-path support audit completes the third planned family check for this fixture:
only 14/41 paths and 284/1,148 legal geometry fields (24.74%) have an exact compatible
external donor. Even a path gate of 1.0 cannot match the 35% field corruption control,
so a whole-path training comparison here would be confounded and is deliberately not run.

The full-primary signature census resolves the corpus-level feasibility question: 1,488,688
of 1,641,526 legal geometry fields (90.69%) have an exact external path-signature donor.
Its candidate rankings needed a family-distinct, P16/T128-bounded selection pass before
a new learning fixture could be pinned. That selection has now completed: the pinned v3
fixture is `2728`, `1F92F`, `1F953`, and `E0C3`, with P16/S32/T128-safe paths and
72.76–100% per-icon external donor coverage. The first selection attempt is retained as
a pre-training capacity failure because it admitted a 39-segment path under S32.

The corrected whole-path arm
`tiny-geometry-f1-whole-path-e13259f-cf59a64e-bd6c4bbc` completed locally. Its
distinct v3 fixture achieves 100% aggregate, changed-token, and retained-token held-out
recovery (550 changed fields) with exact checkpoint continuation and safe paired
renders. This establishes local feasibility only: the earlier factorized and
path-correlated controls use the original fixture, so direct numerical comparison would
be confounded until controls are rerun on v3.

The v3 fixture-matched path-correlated control now completes and reproduces
byte-identically. Its held-out aggregate/changed/retained recovery is
92.30%/80.55%/100% across 3,732 changed fields, versus the matched factorized control's
96.77%/92.62%/99.06% across 3,345. Both meet their predeclared thresholds and have
exact checkpoint continuation. The perfect whole-path result has only 550 changed
fields, so it demonstrates feasibility but cannot establish a corruption-family win.

The predeclared seed-2701 replication closes Gate F. Factorized recovery is
97.15%/93.47%/99.16% aggregate/changed/retained versus path-correlated
92.72%/80.87%/100%; the 12.60-point changed-token advantage agrees with the first
seed's 12.07 points. Whole-path again reaches 100% but changes only 608 held-out fields,
versus 3,337 and 3,586, so it remains a lighter feasibility task. All three replicas
resume exactly and reproduce byte-identically. Factorized role-uniform corruption is
the Gate G primary; path-correlated and whole-path are retained named ablations.

Gate G has begun with a bounded local pipeline smoke over the dominant exact packed
bucket. The hash-pinned hybrid and capacity ledgers join one-to-one; P32/T128 contains
3,359 icons split 2,681/339/339 across train/validation/test with no family leakage.
The new pipeline adds group/subgroup conditioning and exact path locks while retaining
factorized fixed-topology geometry corruption. Its one CPU step, canonical checkpoint
restore, and identical rerun pass. Near-random first-step accuracy is expected and is
not learning evidence. This remains fixed-topology geometry-only, not unconditional
generation.

The bounded Gate G pipeline now also runs on real GPU hardware. On the owned RTX 4080,
the same P32/T128 dominant-bucket config completes one CUDA optimizer step under
declared deterministic algorithms, restores its canonical 7,075,309-byte checkpoint, and
keeps locked paths byte-exact. Train loss is 15.216644 at 0.005068 token accuracy;
validation loss is 15.344566 with 0.009404 aggregate, 0.008850 changed over 226 changed
fields, and 0.009709 retained over 412 retained fields. These first-step accuracies are
near random by construction and are explicitly not learning evidence. Gate G is
therefore unblocked on pipeline mechanics but not passed.


The first predeclared Gate G learning run is falsified, and informatively so. Training
the selected 577,552-parameter denoiser for 600 steps on a 256-icon family-disjoint
subsample lifts held-out changed-token recovery from 0.003005 untrained to 0.055230,
18.38x its own same-input control and monotone in 9 of 10 intervals. But held-out
retained-token accuracy reaches only 0.325586 against the predeclared 0.90, and held-out
loss bottoms out at 9.7932 at step 240 before rising to 11.1485 while training accuracy
climbs to 0.4550. That is overfitting to 256 icons, not an optimization or plumbing
failure. The representation is learnable at real icon diversity; the open question is
now data scale.

A recorded caveat: held-out changed accuracy kept rising while held-out loss worsened,
so the monotone criterion passed during degrading generalization. Held-out loss is the
more honest scalar and the monotone criterion should not be reused unqualified.


Development has moved off the `gtc` control plane onto `gpubox-4080` itself, and the
migration is verified rather than assumed. Re-running the identical v1 config natively
under torch 2.14.0a0+4fdf77b940.nv26.08 and CUDA 13.4, replacing the container
environment of torch 2.8.0a0+5228986c39.nv25.06 and CUDA 12.9, reproduces final train
token accuracy and held-out changed-token accuracy bit-identically. Exactly one retained
token of 26,030 differs, moving aggregate accuracy only in the fifth decimal. Every
conclusion drawn from v1 survives the move, including its falsified
retained-preservation criterion.

The data-scale question v1 left open is now answered, and only partly in the affirmative.
Run `openmoji-g1-train-v2-datascale-a50b2e0-c474c94d-9b9b1699` changes only the train
split, 256 icons to the full 2,681-icon family-disjoint split, and replaces the fixed
step budget with a 6,300-step cap plus held-out-loss early stopping. It stopped at step
1,320 and selected step 840. Against v1 read at its own held-out optimum, step 240,
held-out loss falls 25.1% to 7.3333, aggregate accuracy rises 38.0% to 0.2886, retained
accuracy 38.6% to 0.4128, and the train-minus-held-out gap narrows 39.7% to 0.0942 while
training accuracy moves only 4.8%. The v1 failure was therefore data-limited. But
changed-token recovery reaches only 0.0573, a 1.307x improvement against a predeclared
1.5x, so the primary recovery criterion is falsified and the run is recorded as
partially falsified. Data volume is no longer the binding constraint.

A second, sharper caveat than v1's. Held-out loss bottoms out at step 840 and never
recovers, while held-out accuracy - aggregate, changed and retained alike - rises
monotonically through step 1,320, where changed-token accuracy is 0.0662 and would have
met the 1.5x bar. The criterion was evaluated as predeclared, at the selected
checkpoint. After v1 this project declared held-out loss the more honest scalar; v2
shows that choice costs 15.6% of the relative changed-token recovery available at the
cap. Which scalar governs checkpoint selection is now an open question that the next run
must declare, with a reason, before it starts.

Rendering the v2 checkpoint corrects how those scalars should be read. The read-only
probe `openmoji-g1-train-v2-renders-eff0038-9573bc74-9b9b1699` restored the
hash-verified checkpoint and rendered `x_0`, `x_t` and `x_hat_0` for twelve held-out
icons under the pilot's own held-out corruption draw. Median RGBA error against the
clean render improves only from 0.169546 to 0.142433 at 72 px, a 16.0% reduction, and
none of the twelve predictions is a recognizable icon. Fills and topology are not
corrupted, so several icons keep a correct palette while their stroke geometry is
destroyed. Gate G token accuracy is therefore a poor proxy for visual recovery and no
future recovery claim should be recorded without a render beside it. The probe also
shows that at probability 0.35 the corrupted input is already visually destroyed, so
the corruption schedule - fixed since the Gate F four-icon fixtures and never varied at
corpus scale - is now a first-class candidate factor.

The selection-scalar question v2 opened is now closed, as a convention rather than a
finding. `predict_clean_geometry` decodes by argmax, which suggested the strictly more
accurate checkpoint should render better; that prediction is falsified. Training the
identical v2 configuration at a fixed 1,320-step budget reproduces v2's first 1,320
metric rows exactly and yields its final-step checkpoint, so the loss-selected step 840
and accuracy-selected step 1,320 are two points on one trajectory, and `x_t` render
error is identical across arms. On the complete 128-icon held-out draw the mean paired
difference in `x_hat_0` render error is -0.001638 at 72 px, 95% interval -0.004867 to
+0.001590, with a 65/128 sign test at p = 0.930. The scalars are not separable, but any
true difference is bounded below about 0.0032 RGBA MAE against a median render error
near 0.14. By the decision rule committed before the numbers were read, held-out loss
governs checkpoint selection from now on, chosen as the standard early-stopping signal
rather than on evidence of superiority, and the record states that the choice is
immaterial for render quality at this stage. This is conditional on argmax decoding; a
sampler drawing from the distribution would reopen it.

A methodological correction worth carrying: the twelve-icon predecessor compared each
arm's median independently, which is the wrong test for paired data. Its medians
favoured the loss arm by 6.5%; at 128 icons they favour the accuracy arm by 4.7%, the
opposite direction, while the paired difference stayed near zero in both samples. On
paired data, compare the pairs.



## Last completed action and verification

Implemented optional held-out checkpoint selection and early stopping in the Gate G
pilot at `3b7db57`, gated on a `training.selection_policy` block so every config without
it is untouched: re-running the v1 config reproduces its recorded native summary, trace
and metrics byte-identically, summary SHA-256
`f304a381a6ca57106da5cfe4878b1688e3d6fd558c949243dea01704b7bf6453`. Four focused tests
cover the unchanged fixed-budget shape, argmin selection with a matching written
checkpoint, deterministic patience exhaustion, and rejection of a policy without
periodic evaluation. The full suite passes 112/112; Ruff and strict mypy pass.

Predeclared the data-scale criteria at `a50b2e0` and registered
`openmoji-g1-train-v2-datascale-a50b2e0-c474c94d-9b9b1699` as planned before launch. A
disclosed 60-step sizing preflight on the full split was run first and is reported in
`run.yaml`. The run completed natively on the RTX 4080 in 75.6 s at 290.1 MiB peak CUDA
memory, and a second complete invocation returned an identical JSON result. Locked-path
exactness and canonical checkpoint round-trip both hold at the selected step. Its
7,075,311-byte checkpoint SHA-256 is
`e791493769907ac428db6a081310c0beacc4b88d2072b24fc07df0287e0295b4`.

Added the research weblog: a static site generated from the committed record only -
`state/CURRENT.md`, the new `state/gates.yaml` gate board, `state/runs.jsonl`,
`runs/*/run.yaml` and `result.md`, `reports/findings.md`, and every render under
`reports/`. It invents no status or conclusion and shows failed and superseded runs
exactly like successful ones. It is served on the machine's Tailscale address only.

Pinned OpenMoji 17.0.0 at commit
`f9fc506a3f913be9897ab0181d611d4c910a4104`; hashed and made the 4,495-SVG raw checkout
non-writable; retained a falsified audit v1; completed safe audit v2; generated empirical
distribution reports and seven quarantine contact sheets; visually reviewed all 104
candidates; produced the reviewed v4 manifest; and ran independent curation validation.
Representation, typed-codec, and aligned-lattice evidence is committed locally through
`2099f3d`. The aligned probe completed 700 stable round trips; at q289,
semantic/outlined median 18 px MAE is 0.001710/0.002319, and the alignment-sensitive
`E2C2` semantic case improves from 0.117692 at q256 to 0.001970.

Implemented the immutable full-primary census at `6bb0f10` and bounded Vast stage/smoke
adapters at `c396922`. The census run
`full-primary-structure-v1-6bb0f10-4f1bf442-4e7162ec` completed locally with exactly
8,012 attempts and 4,006 hybrid rows. Actual SHA-256 values match the embedded report
identities; every source has exactly one semantic and one outlined attempt; paired
identity fields and deterministic ordering validate; all three routes contain the
complete 30-point P/S grid; and every loss decomposition balances. Adapter verification
passes the full 49-test suite, Ruff, strict mypy, a local idempotent stage, and a
pre-import tamper regression. GPU smoke has not run because `gtc` has no CUDA and the
Vast worker is not authorized. The completed full-primary evidence and recovery state
are checkpointed locally at `1063fc7`; nothing was pushed.

Implemented exact per-path element, fill, and stroke opacity tokens at `e31273c`, with
compound-layer style equality, canonical PAD/NONE invariants, safe serialization, a
schema-v2 explicit-opacity config path, and an opacity-bearing Vast tiny smoke fixture.
Focused codec tests pass 22/22; the full suite passes 53/53; Ruff and strict mypy pass.
An unregistered read-only nine-icon diagnostic confirmed semantic and outlined
normalization recovery and identified the two one-scalar OOB cases above.

Implemented and checkpointed the reproducible opacity probe at `5164a95`, then ran
`opacity-recovery-v1-5164a95-5f7afab1-49923968` locally. Both routes succeeded for all
nine icons; all 18 tensor/SVG identities are stable; no P/S truncation or style
approximation occurred; and the only four projected programs are the two OOB moons in
both routes. Semantic median RGBA MAE is 0.001547 at 72 px and 0.001722 at 18 px. A
second complete invocation accepted every exact prior artifact, verifying idempotency.
The full suite passes 55/55; Ruff and strict mypy pass. No GPU work was performed.

Ran `full-primary-structure-v2-opacity-a9f1d5c-57a3ebea-4e7162ec` locally. It completed
8,012 normalization attempts and 4,006 hybrid selections with verified hashes and
balanced loss decomposition at all 30 P/S capacities. The regression comparison found
zero unexplained attempt or hybrid mismatches. Its compact evidence and recovery state
are checkpointed locally at `88532a5`. No GPU work was performed.

Implemented and checkpointed the OOB-role/render-impact harness at `bc6459b`, then ran
`oob-control-probe-v1-bc6459b-16c43292-8469ae8f`. All 48 semantic/outlined programs
normalize, encode, decode, serialize, and render stably with no truncation or style
approximation. A second full invocation reproduced every immutable output exactly.
Visual inspection confirms catastrophic tail deformation for the two worst icons.
Focused tests pass 26/26, the full suite passes 57/57, Ruff passes, and strict mypy
passes. The negative result and compact report are checkpointed locally at `45ca009`.
No GPU work was performed.

Implemented and checkpointed role-typed coordinates at `24461b3`, then ran
`control-coordinate-vocabulary-v1-24461b3-4c2f9833-8469ae8f`. All 48 programs are
strict-lossless and stable; all counterfactual SVG hashes match; semantic median MAE is
0.002067/0.002752 at 72/18 px and worst 18 px MAE is 0.005159. A second complete
invocation reproduced every artifact. The full suite passes 59/59; Ruff and strict
mypy pass. The compact result and recovery state are checkpointed locally at `d2b4a1c`.
No GPU work was performed.

Implemented the deterministic corpus/style-tail harness at `7925022`, then ran
`style-vocabulary-v1-7925022-778e6dc5-9b9b1699`. All 105 exact/K32/K48 programs are
stable with no structural loss or coordinate projection. Full-corpus analytics and the
35-icon exact-style-controlled render fixture falsify unaugmented K48 only in the rare
4.1-width 72 px tail described above. A second invocation reproduced all artifact
hashes. The full suite passes 61/61; Ruff and strict mypy pass. No GPU work was
performed.

Implemented pinned-fixture and explicit width-sentinel support at `ecba341`, then ran
`style-vocabulary-v2-render-sentinel-ecba341-858e7513-b13fecb6`. All 105 programs are
stable and the exact 4.1 token removes the falsified 72 px tail without exposing a new
one. A second invocation reproduced all artifact hashes. The full suite remains 61/61;
Ruff and strict mypy pass. No GPU work was performed.

Implemented and checkpointed the capacity audit at `bc52408`, then ran
`capacity-layout-v1-bc52408-6c86cc30-9b9b1699`. All 4,006 contour-length decompositions
balance; packed P80/T1216 and all adaptive assignments are exact; detail and assignment
cardinalities are 24,036 and 4,006. A second invocation reproduced all hashes. The full
suite passes 62/62; Ruff and strict mypy pass. No GPU work was performed.

Implemented reversible packed conversion at `f8b1601`, isolated typed-SVG rendering at
`82c48ec`, and a deterministic Gate D stress harness at `e6c100b`. The unit suite passes
90/90 with 20 seeded random packed round trips and real subprocess rendering. Registered
stress run `packed-render-stress-v1-e6c100b-df13aed7-005dc6b4` failed before artifacts:
the harness used token 289 as an invalid q289 endpoint, but valid coordinate tokens are
1..289. The failure is preserved in the registry; the mutator is corrected to 290 for
a new run identity. No GPU work was performed.

Corrected stress run `packed-render-stress-v1-d5ca696-df13aed7-005dc6b4` passes all
checks above and reproduced every report hash on a second invocation. Packed conversion
is at `f8b1601`, renderer isolation at `82c48ec`, the stress harness at `e6c100b`, and
the boundary correction at `d5ca696`. The full suite passes 90/90; Ruff and strict mypy
pass. Gate D is complete. No GPU work was performed.

Installed pinned CPU-only PyTorch 2.8.0+cpu in the local `.venv`, recorded the explicit
CPU package index and lockfile, and checkpointed the Gate E harness at `49e6e6a`. Its
fixture hash is `32a80ab576a6d5a435e859d38c1ba25e302070e92caae2971b517fe42c4b0d79`
and config hash is `bf71ca62d7f3e3a3a07aa0a6caeb78039785badf2ca63d0884ccf030909769c3`.
All four fixtures load without truncation or projection. The focused learning tests pass
2/2, the full suite passes 92/92, Ruff passes, and strict mypy passes. No registered
learning run or GPU work has occurred yet.

Registered and ran `tiny-geometry-v1-3493eff-bf71ca62-32a80ab5` on local CPU. The first
execution finished within its 800-step and 0.1 GB bounds and produced the negative
learning result above. Its idempotency rerun failed closed on differing checkpoint
container bytes. Compact metrics, renders, summary, run record, and the original 2.9 MB
checkpoint are preserved; no remote action or external spend occurred.

Replaced only the v1 checkpoint container under commit `0b8535b`: v2 stores a
canonically ordered JSON tree and bounded pickle-free `.npy` tensors in a ZIP with fixed
metadata, then restores model, optimizer, RNG, and step. A focused round trip proves
independent saves and load-resave are byte-identical. The full suite passes 93/93;
Ruff and strict mypy pass. The learning fixture, architecture, seeds, corruption,
steps, and predeclared scientific thresholds are unchanged in the v2 config, whose hash
is `75d558c97c106c127f026a229613afcde35da13cf7668c2c950c4b1c5d955ebb`.

Registered and completed `tiny-geometry-v2-47811d0-75d558c9-32a80ab5`. Both complete
invocations reproduced every learning metric and final model hash from v1. The canonical
checkpoint, summary, metrics, render metrics, Markdown, and trajectory are byte-identical
across reruns. V2 therefore resolves artifact idempotency but intentionally preserves
the missed held-out thresholds; Gate E remains open.

Implemented the metric-only v3 diagnostic at `a10ef83`. It leaves learning unchanged
and partitions legal-coordinate accuracy by whether each `x_t` token actually differs
from `x_0`. The partition sums back to the existing total in tests. Its config hash is
`ff42f4dd0b8915f00d8deea020b46e6891a588751e651d50c1e75bced3135a30`.
The full suite remains 93/93; Ruff and strict mypy pass.

Registered and completed `tiny-geometry-v3-93ed354-ff42f4dd-32a80ab5` twice with exact
artifact identity. Diverse-four changed-token accuracy is 58.17%, 15.65 points below
aggregate, while retained-token accuracy is 82.52%. One-icon changed/retained accuracy
is 88.59%/96.11%. The diagnostic hypothesis passes and localizes the next question to
corruption-pattern coverage rather than basic memorization or checkpointing.

Implemented the controlled v4 corruption-coverage treatment at `f0c265c`. One-icon
training remains the fixed v3 control. Diverse-four now derives a fresh deterministic
16-example corruption batch from the global optimizer step, so continuous and resumed
training see the same sequence without a hidden data cursor. Model, probability, batch
size, 320-step budget, held-out draws, and resume boundary are unchanged. Config hash:
`7ccbc64e51a0f009178c06b262fd6bc10bab3deb2c013fcdc825e29618e465c8`.
The full suite passes 93/93; Ruff and strict mypy pass.

Registered and completed `tiny-geometry-v4-ede1009-7ccbc64e-32a80ab5` twice with exact
artifact identity. Diverse-four aggregate/changed/retained held-out accuracy improves to
98.56%/96.65%/99.63% without increasing the model, batch, probability, or step budget.
Checkpoint continuation is exact, and visual inspection finds all four reconstructions
recognizable at 72 and 18 px. The treatment hypothesis passes. The unchanged one-icon
control remains 93.57% held-out, leaving the generic combined flag false.

Added the final all-resampled config at `f33acb7`. It changes only the one-icon
`resample_each_step` flag from false to true; diverse-four and every other setting are
identical to v4. Config hash:
`a34456a8e50877002d339aec9167e77501fb5edab831568626bb995554b3657e`.

Registered and completed `tiny-geometry-v5-44ce3de-a34456a8-32a80ab5` twice with exact
artifact identity. One-icon held-out accuracy rises to 99.26% and diverse remains
98.56%. The literal v5 combined flag stays false because its now-unseen fixed probe is
97.73% versus the inherited 99% memorization threshold. Gate E nevertheless closes from
the controlled sequence: v1 supplies 100% actual one-icon overfit, v5 supplies one-icon
held-out recovery, v4/v5 supply diverse recovery and recognizable renders, and v2-v5
supply exact resume and byte-stable artifacts. The representation is learnable for
fixed-topology geometry; topology and corruption-family questions remain open.

Adapted the owned-worker contract to its sibling-container topology. The adapter now
requires and preflights the exact Compose-prefixed workspace and artifact volumes,
mounts only run-specific subpaths, bypasses the NGC entrypoint for clean machine-readable
output, and has dry-run/fake-Docker coverage. The complete local suite passes 101/101;
Ruff and strict mypy pass. The pinned image is
`mojidiff/owned-gpu-smoke:2c3b248` with image ID
`sha256:fd065ec98130193b324cf9462fa97e025041bd6df72313249edd902608a2ac00`.

The owned RTX 4080 adapter smoke `owned-gpu-smoke-5da02d3-7c03a644` completed twice.
It verified immutable staging, typed codec and Cairo rendering, one CUDA forward/backward
step, checkpoint write/read, transfer into the persistent artifact volume, and exact
create-or-identical rerun behavior. The 16,179-byte checkpoint SHA-256 is
`77e93414e04707efb0469718426bb2364d5a7b22e575e519124d5c628442348f`;
the result JSON SHA-256 is
`837850bd34e3b82c978989cc229a52db45599c8659e1c0a5b137bc25c1809837`.
Three preceding adapter failures are preserved with reason codes; one additional parent
run completed its GPU/artifact work but failed only because the NGC banner violated the
JSON stdout contract.

Ran the owned-worker Gate G pipeline smoke through four registered identities and
preserved all three failures. `openmoji-g1-gpu-cd3250e-0bafd5c-9b9b1699` failed before
model construction on relative input paths resolved against the image working directory.
`openmoji-g1-gpu-e7dc920-0bafd5c-9b9b1699` reached the first CUDA step and was rejected
because `torch.use_deterministic_algorithms(True)` needs `CUBLAS_WORKSPACE_CONFIG` on
CUDA >= 10.2; its artifact directory stayed empty.
`openmoji-g1-gpu-215bcb8-0bafd5c-9b9b1699` completed the GPU step and wrote durable
artifacts but failed the adapter result contract by digesting `summary.json` from the
output root instead of the pilot report root; its artifacts are intact on the worker and
listed in its `artifacts.json`.

`openmoji-g1-gpu-7ba1aa4-0bafd5c-9b9b1699` completed. `device` is `cuda`, the staged
archive and extracted-tree hashes verified, the checkpoint round-trips, locked paths are
exact, and a second identical invocation returned the same JSON result. Its checkpoint
digest `4b265e5575e3aa455a0d427e340ec407eaaaf39222709305d060281ae7e453f9` and summary
digest `15ede088423678eb308548481e1c348850c04bf588101d4f74fecaaa224d07b7` also match the
preserved third attempt exactly, so the GPU result reproduces across separate containers.
The full suite passes 105/105 including a new CPU regression test that pins the staged
wrapper to the artifact layout the pilot actually writes; Ruff and strict mypy pass.
Two of the three failures were `scripts/remote/` wrapper defects, which had no test
coverage before this session.


Added metric-only held-out tracing to the pilot: a new `eval_every` config field, an
untrained step-0 control on the identical corruption draw, and a `validation.jsonl`
trace. The change is provably inert. Configs without `eval_every` reproduce the recorded
local CPU checkpoint `d11efa00...` exactly, and a test asserts that a traced and an
untraced run of the same config produce byte-identical checkpoints. Also generalized the
staged wrapper so each smoke id pins exactly one committed config and an unknown id
fails closed. The full suite passes 108/108; Ruff and strict mypy pass.

Predeclared and completed `openmoji-g1-train-ec2436b-f2a06ca5-9b9b1699` on the owned
RTX 4080 in about 22 seconds, with criteria committed to Git before launch at `63074d9`.
Four of five predeclared criteria passed; retained preservation failed at 0.325586
against 0.90, so the overall outcome is recorded as falsified and retained rather than
retuned. Two complete invocations returned identical checkpoint
`ac696b81b3c9cd6f01f4cfba0adc5301b3db59a189f7cd007dc41778ad45a49d` and summary
`b892acbf28078405a5954b7fa4e0db3704dd0675bebddd2ba53522add77fd14d`.


Moved the canonical worktree, full Git history, the 405 MB immutable raw checkout,
reports, and the run registry to `/home/dev/workspace/mojidiff` on `gpubox-4080`. The
move was verified rather than trusted: identical HEAD and tracked tree hash, identical
4,495-file count, preserved read-only mode on the raw checkout, and an identical
aggregate SHA-256 over every raw file (`fe76333c011104a4523635b3946fdc348a3aba8a1f5fb94bf67513938d0a99c4`).
Built a native `.venv` that inherits the system NGC torch through
`--system-site-packages`, installed the project dependencies and the missing Cairo
runtime, and confirmed 108/108 tests, Ruff, and strict mypy on the box.

Rewrote `AGENTS.md` for the new single-machine topology and repointed the inventory
orchestrator to `gpubox-4080`. The stricter authorization rules now apply only to rented
or shared machines, which is what they were written for. The run registry, predeclared
criteria, data immutability, and experimental discipline are kept. One test that asserted
`hostname == "gtc"` now asserts sanitization instead of a specific machine.

Registered and completed `openmoji-g1-train-v1-native-c9bf1b9-f2a06ca5-9b9b1699` in
about 18 seconds as the matched environment control described above.

The `gtc` copy is left intact and untouched as a backup. Nothing was deleted.


## Active jobs

One, and it is not compute. The research weblog is served by
`scripts/serve_weblog.py` in tmux session `mojidiff-weblog`, bound to
`100.69.189.78:8787` on the Tailscale interface only, not to `0.0.0.0`. It is a
read-only static file server over `site/` and holds no GPU or lock; stop it with
`tmux kill-session -t mojidiff-weblog`.

No training job is active. The v2 data-scale run and its reproducibility rerun both
finished in tmux sessions that have exited. The owned worker is enabled for bounded work
under its recorded 20,000-step and 50 GB cap. Vast and A100 workers remain disabled.

## Artifact durability

Raw source (405 MB), audit renders/tables, and derived fixture SVGs remain on the
persistent workspace. Curation, representation, typed-codec, and aligned-grid compact
evidence is versioned locally through `2099f3d`; the 26 MB full-primary report is
versioned at `1063fc7`, the opacity codec at `e31273c`, and the 64 KB opacity evidence
at `86f7d03`; the 32 MB full-primary opacity report is versioned at `88532a5`, and the
216 KB OOB compact report at `45ca009`. Nothing was pushed. The OOB probe's 812 KB
derived SVGs and opacity probe's 260 KB derived SVGs are reproducible but local-only.
The 152 KB role-typed coordinate report is versioned at `d2b4a1c`; its 500 KB derived
SVGs are reproducible and local-only. The owned-worker persistent artifact sink is now
configured and verified; earlier control-plane-only artifacts have not been copied to it.

The compact style-vocabulary, capacity, and packed/render stress evidence is versioned
locally through `791af1f`; their reproducible bulk SVG/raster derivatives remain ignored.
The Gate E implementation, fixture, config, and CPU dependency lock are versioned at
`49e6e6a`, and the v1 negative compact report is versioned at `471899a`. Its 2.9 MB
resume checkpoint is reproducible in model/optimizer content but not byte-stable in the
legacy PyTorch container and remains local-only. Nothing was pushed.

The v2 compact report is versioned locally at `2d75672`. Its canonical 3.0 MB checkpoint
is verified byte-stable but has not been copied to the newly verified worker sink.

The v3 compact diagnostic is versioned locally at `99106ed`; its canonical 3.0 MB
checkpoint is byte-stable and remains local-only. The v4 compact report is versioned
locally at `fecc4c0`; its 3.0 MB canonical checkpoint is also byte-stable and local-only.

The v5 compact report is pending a local Git checkpoint. Its canonical 3.0 MB checkpoint
is byte-identical to v4 because the diverse branch is unchanged; it remains local-only.

The owned worker's `/home/dev/workspace` and `/home/dev/.cache` live on persistent
Compose volumes. The artifact sink write/read/hash preflight succeeded, and the complete
smoke checkpoint and result are retained under
`/home/dev/.cache/owned-gpu-smoke-5da02d3-7c03a644/`. Identical workspace copies are
retained under `/home/dev/workspace/owned-gpu-smoke-5da02d3-7c03a644/`. No artifact was
deleted or published.

The local Gate G pilot's 7,075,309-byte canonical checkpoint is reproducible and
byte-stable at SHA-256
`d11efa006657dafc2edf10bc5fe5725cad09e6beb40167032606f6c9f7c8407a`.
It remains on the control-plane volume pending the pipeline-specific owned-worker smoke.

The Gate G GPU smoke artifacts are durable in the owned worker's persistent artifact
volume under `/home/dev/.cache/openmoji-g1-gpu-7ba1aa4-0bafd5c-9b9b1699/`, totalling
7,077,100 bytes, well inside the 50 GB cap. The compact `summary.json` and
`metrics.jsonl` were copied to the control plane under
`runs/openmoji-g1-gpu-7ba1aa4-0bafd5c-9b9b1699/` and verified by hash against the worker
originals. The 7,075,309-byte GPU checkpoint deliberately stays on the worker sink. The
third attempt's identical artifacts are also retained on the worker; nothing was deleted.


The training run's artifacts are durable on the owned worker under
`/home/dev/.cache/openmoji-g1-train-ec2436b-f2a06ca5-9b9b1699/`, totalling 7,140,279
bytes. Its compact `summary.json`, `validation.jsonl`, and `metrics.jsonl` were copied
to the control plane under that run's directory and verified by hash against the worker
originals; the 7,075,310-byte checkpoint deliberately stays on the worker sink.


The v2 data-scale run's artifacts are durable under
`/home/dev/.cache/openmoji-g1-train-v2-datascale-a50b2e0-c474c94d-9b9b1699/`: a
7,075,311-byte canonical checkpoint at SHA-256
`e791493769907ac428db6a081310c0beacc4b88d2072b24fc07df0287e0295b4`, plus the report
files. Its `summary.json`, `metrics.jsonl` and `validation.jsonl`, about 200 KB in all,
are copied into the run directory in the repository; the checkpoint deliberately stays
on the durable sink. Nothing was deleted, and nothing was pushed.

The v2 render probe's 336 KB contact sheet, 16 KB render metrics, and summary are
versioned in the repository under `reports/learning/openmoji-g1-train-v2-renders/`, with
the metrics and summary also copied into its run directory. It is reproducible from the
durable v2 checkpoint in about a minute on CPU.

The selection-scalar comparison's durable checkpoint is
`/home/dev/.cache/openmoji-g1-selection-scalar-c4e6a8c-3a34e64b-9b9b1699/checkpoint/`
at SHA-256 `359b11a1020ea07b43bca82be254cac791793dc9166f7828c9038d5ca5b830af`. Its four
render reports cover the 12-icon and 128-icon probes for both arms and are versioned
under `reports/learning/`: about 8.3 MB, dominated by the two 128-icon contact sheets at
3.7 MB each. That is large for a render but well inside this repository's precedent for
versioned evidence, and the sheets are the visual record the conclusion rests on, so
they are kept rather than reduced to their metrics. They are duplicated on the durable
sink under
`/home/dev/.cache/openmoji-g1-selection-scalar-full-c4e6a8c-powered-9b9b1699/` and are
reproducible from the two durable checkpoints on CPU in about ten minutes.

The weblog's generated `site/` directory is derived output and is Git-ignored: it is
rebuilt from the committed record in about a second by `python -m mojidiff.weblog.build`.
The generator, its stylesheet and script, and `state/gates.yaml` are versioned.

## Current blockers and missing authorization

- YOLO checks passed for hostname, persistent repository location, TLS-only isolated
  DinD, absent host Docker sockets, absent worker/cloud/kube/GitHub credentials, and an
  empty running-container inventory. Dedicated Tailscale tag/ACL restrictions and
  external credential scopes cannot be verified from inside this container.
- The owned worker is authorized and its persistent paths are verified. Its SSH endpoint
  is a devbox controlling sibling Docker containers: use only the recorded
  `code-server-gpu_gpubox-workspace` and `code-server-gpu_gpubox-home` volumes. An
  unprefixed volume name silently creates the wrong volume, and devbox-local bind paths
  do not resolve in sibling containers.
- `vast_5090` is disabled; `resource_cap.max_steps`, `max_spend_usd`, and
  `max_storage_gb` are null.
- The prior Vast endpoint may be stale after shutdown. A future Vast launch still needs
  its exact current SSH endpoint and complete billed resource cap before use.
- `a100_cluster` is disabled; `workspace_root`, `namespace`, `service_account`, `pvc`,
  `resource_cap.max_jobs`, `max_steps_per_job`, and `max_storage_gb` are null.
- The owned-GPU endpoint was explicitly replaced by the operator. The hardened
  `owned-gpu` alias now resolves to `dev@100.69.189.78:22`; first-use pinning recorded
  ED25519 fingerprint `SHA256:kBhpBUsFqhneFvZUdwEnqt7URk/7+MaiJ7FxUBA09vA`.
  A probe identifies host `gpubox-4080`, Linux 6.17.0-41, NVIDIA GeForce RTX 4080
  (16,376 MiB), driver 595.71.05, and 714 GiB free on `/`. This remains an RTX 4080
  result, not a 4090 result. The sibling Docker daemon is reachable and its NVIDIA
  runtime successfully executed the complete project smoke using the pinned image.

## Next smallest evidence-producing action

The selection scalar is settled: held-out loss, by convention, with the effect on render
quality bounded below 0.0032 RGBA MAE. Runs no longer need to argue the point; they
record the rule and move on.

The next controlled experiment is model capacity, not more data. v2 showed that
10.5x the data narrows the generalization gap 39.7% but lifts changed-token recovery
only 1.307x, and its train/held-out gap at the selected step is down to 0.0942, so the
577,552-parameter denoiser is close to fitting what it can express. Change only
`d_model`, `layers`, and `feedforward`, hold the full 2,681-icon split, seed 3101,
corruption probability 0.35, batch size 16, and learning rate 0.001 fixed, and predeclare
a changed-token recovery threshold against v2's 0.0573 at its selected step. Keep the
0.90 retained-preservation bar standing and expect it to remain falsified.

The corruption schedule is now the co-equal candidate, not a secondary one, and it
should be a separate one-factor run rather than folded into the capacity test. The
render probe showed that at probability 0.35 the corrupted input is already visually
destroyed, so a third of the geometry is simply gone and the visual task may be
unachievable at any model size. That probability has been fixed since the Gate F
four-icon fixtures and has never been varied at corpus scale. Vary it, or train across
a range of levels, which is also what a denoiser facing many corruption levels at
sampling time would need.

Whichever factor goes first, render the result. The scalars oversold the v2 output by a
wide margin, and `python -m mojidiff.learning.pilot_renders` now makes the check cheap.

Two augmentation ideas remain measured and queued but deliberately unapplied, since data
volume is no longer the binding constraint. Left-right mirroring is exactly
representable: the quarter-unit lattice is closed under x to 72-x, endpoint token t maps
to 290-t for all 289 bins, and observed control-x spans only 2.00 to 70.00, so nothing
overflows the -8 to 96 control vocabulary. Content also occupies a median 0.753 of the
72-unit box, so rescaling would recover about 1.33x coordinate resolution, though
coordinate precision is not currently the limiting factor.

Regenerate the weblog with `python -m mojidiff.weblog.build` after any material result,
so the served view does not drift from the committed record.
