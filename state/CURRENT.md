# Current research state

Updated: 2026-09-24T21:31:44Z

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

Model capacity is now eliminated as well. Run
`openmoji-g1-capacity-v3-3c252d5-ee32665b-9b9b1699` scales the denoiser 3.53x to
2,040,976 parameters, changing nothing else, and falsifies all three of its primary
criteria. Held-out loss at the selected checkpoint is 7.3790 against v2's 7.3333, 0.6%
worse rather than better; changed-token recovery is 0.0639, a 1.115x improvement against
a 1.25x bar; and on the full 128-icon draw the mean paired render difference is +0.001853
with a 95% interval of -0.0026 to +0.0063, against a detection threshold of roughly
0.0064 established on that same draw. The larger model reached the same held-out loss
floor in 360 steps instead of 840 and then overfitted, which is the signature of a
task-limited rather than capacity-limited problem; both models converge near 7.35.

Two of the three candidate factors are therefore eliminated on matched, predeclared
comparisons - data volume by v2, capacity by v3 - and the remaining candidate is the
corruption schedule, exactly where the render probe pointed.

Testing that last candidate produced the most important result of the Gate G sequence,
and it is not the one expected. Run
`openmoji-g1-corruption-sweep-71a080a-4levels-9b9b1699` evaluated v2's loss-selected
checkpoint at corruption probabilities 0.05, 0.10, 0.20 and 0.35 with identical icons
and seeds, and no training. Mean per-icon recovery fraction at 72 px is -4.6517,
-1.2024, -0.1315 and +0.1800 - strictly increasing in the corruption level, the opposite
of the prediction. At 0.05 the model makes its input 2.8x worse and helps on zero of
thirty-two icons. All three primary criteria are falsified and the competing damage
hypothesis, registered in the same record before launch, is confirmed.

The mechanism: mean `x_t` render error varies by a factor of 3.5 across the sweep while
mean `x_hat_0` error moves about 6%, and per-icon predictions at 0.05 and 0.35 correlate
at Pearson r = 0.91. The model emits nearly the same reconstruction whatever it is
given. The p=0.05 contact sheet shows `x_t` almost indistinguishable from `x_0` and
`x_hat_0` as scribble in every row. `GeometryDenoiser` has no timestep or noise-level
input and no corrupted-field mask, and it trained at one fixed probability, so nothing
tells it how much to trust its input.

This reinterprets the sequence without invalidating any measurement. The +18% recovery
at 0.35 is a fixed-quality output beating a badly corrupted baseline, not recovered
geometry. Retained-token accuracy of 0.4128 was this same finding in token space since
v1, and the standing 0.90 bar has been measuring the real problem all along. The shared
loss floor near 7.35 across a 3.53x capacity change is consistent with both models
learning the same prior. What v1 and v2 learned is a group-conditioned prior over icon
geometry, not a conditional denoiser.

The cause is now identified and partly fixed. Noise-level conditioning was the obvious
informational explanation and run
`openmoji-g1-noise-conditioned-v4-6b935b2-5a9eaf44-9b9b1699` falsified it: sampling the
corruption level per example over 0.05 to 0.50 and telling the model the level moved
every figure in the predicted direction and none enough to matter, with the prediction
correlation only 0.9122 to 0.8619. Eliminating that forced a look at the encoder, where
the defect is provable. `GeometryDenoiser` summed six coordinate lookups from one shared
embedding table into a single vector per segment, so a segment was an unordered bag of
its values; in float64 a program and its coordinate-swapped variant produce
byte-identical logits, maximum absolute difference exactly 0.0 across 414 swapped
segments. The model was never ignoring `x_t` - it was reading a scrambled copy.

Run `openmoji-g1-slot-bound-v5-f502df1-d8af55ea-9b9b1699` binds each value to its slot
for 768 parameters, +0.13%, changing nothing else about v2. It beats the 3.53x capacity
run on every measure: held-out loss 7.1782 against 7.3333 and 7.3790, retained accuracy
0.5408 against 0.4128 and 0.4323, aggregate 0.3793, changed recovery 0.0787, and the
latest held-out optimum of any run at step 1,500. At corruption 0.20 it is break-even,
helping on 20 of 32 icons where v2 helped on 14. But its prediction correlation is
0.8621, unchanged from v4, and two of its three behavioural criteria are falsified.
Fixing the encoder raised the ceiling without changing what the model does.

Slot binding is kept permanently: a strict improvement for 768 parameters that removes a
defect which would confound every later result. Five candidates are now eliminated by
predeclared comparison - data volume, capacity, the corruption regime, noise-level
information, and encoder slot-blindness - leaving the prediction objective.

The prediction objective was the last candidate, and testing it produced a control that
should have existed since v1. Under argmax decoding, emitting `x_t` unchanged is a legal
policy: the identity function scores 1.0 retained, 0.0 changed and 0.6506 aggregate at
corruption 0.35. Every trained model scores below it, the best by 0.2713. v1 reported
recovery as 18x its *untrained* control, which made weak learning look like progress
because random is the wrong floor.

Run `openmoji-g1-edit-mask-v6-0228c3b-0f6ce115-9b9b1699` added per-field keep-or-change
heads so identity is predict-keep-everywhere, 1,552 parameters on v5. All four primary
criteria are falsified and the predeclared degenerate-collapse watch fired. Aggregate
rises to 0.6119, still below identity; retained to 0.9335, below identity's 1.0; and
changed recovery *falls* from 0.0787 to 0.0132. The model predicts keep on 91.4% of
fields, landing strictly worse than both v5 and identity. Render recovery is near zero
at every level: the edit mask converted an actively harmful model into a nearly inert
one.

Measuring the keep head as a corrupted-field detector gives the answer. Recall is 0.1058
and precision 0.4312 against a 0.3494 base rate, a lift of 1.23x - barely better than
guessing. A coordinate resampled uniformly from a 289-value legal vocabulary is a
plausible coordinate, so identifying it requires already knowing the icon. Detection is
not an easier sub-problem than denoising; it is the same problem, and with 65% of fields
uncorrupted predicting keep is loss-minimising.

A five-lens diagnostic panel over this codebase then overturned most of the sequence, and
four verified defects were found and fixed. None was a tuning knob.

The loss averaged cross-entropy over per-(segment kind, coordinate slot) GROUPS rather
than fields. QUAD is 0.93% of the corpus, so its four groups hold one field each and
carried 4/13 of the coordinate loss against 40,004 others - a measured 5,287x per-field
weight ratio - while the group count flipped between 9 and 13 by batch. No attention
padding mask was passed, so 29.4% of the sequence was attended as content. Because the
evaluation path uses the same loss, held-out loss - the selection and early-stopping
signal - was ~31% four individual fields, which is why every run stopped at 6-24% of its
cap. Fixing both took the detector from 1.365 to 2.806 and the selected step from 660 to
6,180 of 6,300 with no early stop.

The corruption process leaked. Replacements were drawn uniformly from the full legal
vocabulary, so many landed on tokens real icons never use, and a zero-parameter detector
knowing only the corpus marginal reached 2.84 lift with no context at all - above v8's
2.806. Drawing replacements from the corpus marginal for the field's role instead drops
that detector to 0.771 while the local-continuity signal survives at 1.893. On that
honest task the categorical model reaches only 1.506.

Coordinates entered as unordered categories. The trained input table carried no order -
Spearman -0.136 between embedding distance and bin distance, adjacent bins at 0.989 of
the all-pairs mean - while the output head, which gets a direct gradient from the corpus
marginal, reached +0.143 and 0.882. Decoding each token to its view-unit value with the
role-correct affine map and projecting Fourier features of it takes the detector from
1.506 to 2.781, past the free statistic's 1.893, with 54,720 FEWER parameters. That is
the project's first trained model to beat a zero-parameter heuristic on a leak-free task.

Turning the value head back on then loses it again. v11 reaches changed-token recovery
of 0.0296, 2.24x the best any earlier model managed, but aggregate 0.5839 against
identity's 0.6515, because the value objective costs 1.064 of detector lift - 2.781 down
to 1.717. The two heads share one encoder and a 289/417-way exact-token objective against
a 2-class decision is not a fair fight. No decode threshold rescues it: break-even
detection confidence is 1/(1+0.1087) = 0.9019 and every flag rate loses to identity
monotonically.

Balancing the two objectives lets them coexist, and the identity baseline finally falls.
The keep head decides 2 classes and the value head 289 or 417, so at chance they are log
2 and log 417 nats - a factor of 8.7 - and an unweighted sum lets the value task dominate
a shared encoder. Scaling it by log(2)/log(417) = 0.1149, a derived weight rather than a
tuned one, recovers the detector to 2.569 while changed-token recovery rises to 0.0738,
2.5x v11's and 5.6x the best any earlier model managed.

At the model's own 0.5 gate aggregate is 0.6086, so v12's predeclared beats-identity
criterion is falsified as written. But with the decode threshold chosen on 64 held-out
icons and reported on the disjoint other 64 - so the figure is not selection on its own
evaluation set, and the primary/test split stays untouched - the model reaches 0.6567
against identity's 0.6502 by flagging its most confident 5%. That is the first time
anything in this project has beaten the trivial policy of emitting its input unchanged.

The calibration asterisk is then removed. The threshold is derivable: changing a field
pays only above p = 1/(1+q) in the value head's accuracy q on corrupted fields. v13
derived it from train icons and beat identity with nothing fitted on held-out data,
0.6519 against 0.6515, but by only 0.0003 - the value head scores 0.2751 on what it
trained on against 0.1233 held out, so q was inflated and the threshold came out at 0.784
where 0.890 was implied. v14 estimates q on 256 icons withheld from training: q 0.2220,
threshold 0.8183, aggregate 0.6543 against identity's 0.6515, margin +0.0027, all four
predeclared criteria passed. Retained-token accuracy is 0.9815, the highest here, and the
run is handicapped by its own design - it trains on 2,425 icons rather than 2,681 - so
the margin is conservative.

Nothing in v14 is fitted on held-out data. The project has a denoiser that does better
than doing nothing, with no caveat about how the number was obtained. It is also +0.0027
on token accuracy, about 0.4% relative: not a good model, but one that has stopped being
worse than useless.

The distance-kernel target then falsified its own criteria and succeeded anyway, which is
a verdict on the criteria. v15 spreads the value target as exp(-|b-t|*0.25/tau) with tau
one view unit. All three primary criteria - stated in exact-token terms - fail: withheld
value accuracy 0.1623 against 0.2220, identity margin +0.0014 against +0.0027, derived
threshold 0.8604 against 0.8183. But mean absolute error on corrupted held-out fields
falls from 8.2871 to 6.0282 view units, 27%, the median 33% from 3.75 to 2.50, and the
fraction within one view unit rises from 0.2793 to 0.3169. Paired per-icon render
recovery nearly triples, +0.0447 to +0.1246 at 72 px and +0.0729 to +0.1306 at 18 px,
both intervals excluding zero, helping 10 and 11 of 12 icons. Those are the first
positive render recoveries this project has measured under leak-free corruption.

Exact-token accuracy is therefore anti-correlated with what the project is trying to
produce, as the task-formulation lens argued before any of it ran. From here mean
absolute view-unit error and paired render recovery are the primary reported metrics and
exact-token accuracy is secondary, so nothing already published is withdrawn.

Correcting the decode gate the same way doubles it again. v15 localised better yet gated
itself MORE tightly, 0.8604 against v14's 0.8183, because the break-even rule credits
only exact tokens. v16 picks the threshold by absolute view-unit error saved on the
withheld calibration icons: it falls to 0.6800, saving 2.9843 view units per field, and
paired render recovery rises from +0.1246 to +0.2511 at 72 px and +0.1306 to +0.2745 at
18 px, helping 11 of 12 icons with both intervals excluding zero. The trained weights are
byte-identical to v15's, so that is the decode rule alone - 5.6x v14 and 2x v15. The two
corrections compound because they are the same correction applied twice: train for
closeness, then gate on closeness.

v16's held-out aggregate token accuracy is 0.6354 against identity's 0.6515, which its
record anticipated before the run. Between v15 and v16 the metric question is settled:
v16 is simultaneously worse than the identity policy on exact tokens and the best
geometric denoiser this project has produced. Both are true, and that is a statement
about the metric.

The sequence, from a model that actively destroyed its input to one that recovers a
quarter of the render gap: identity 0.0000, v14 +0.0447, v15 +0.1246, v16 +0.2511. Seven
verified defects and corrections, and the model shrank from 579,872 parameters to
525,152. No render is a recognizable icon; at 35% corruption the input is already
scribble and recovering a quarter of that gap leaves scribble.

The four eliminations are then retested off that base and all survive. Judged on paired
render recovery against v16's +0.2511 and its interval half-width of 0.1103: capacity at
3.69x parameters gives +0.2779, restoring v3; 256 training icons give +0.0766, 0.175
below, restoring v2; noise conditioning gives +0.2096, restoring v4. All three reproduce
identically. The eliminations were right even though they were measured badly, and the
four are admissible again.

Capacity is the one that was expected to move and did not. It had been recorded as the
least trustworthy of the four, because 768 parameters of slot binding later beat a 3.53x
capacity increase outright. At 3.69x on a sound setup it is still within noise, and the
best model this session built remains the smallest one. The two results were never in
tension: one is a representational fix, the other raw size.

Sweeping v16 across corruption levels then retires the standing line about renders. At
probability 0.05 and 0.10 the predictions are RECOGNIZABLE ICONS and visibly repaired:
the hedgehog's stray diagonal removed, the first-aid kit's slash gone, "WC" cleaned to
near-perfect, the vampire's face restored, and at 0.10 the weightlifter 1F3CB going from
heavy scribble to a clean figure. Recovery by level, each gated at the threshold derived
for it on withheld icons, is +0.0308, +0.3902, +0.2469 and +0.2511, against the v2
checkpoint's -4.6517, -1.2024, -0.1315 and +0.1800 on the same sweep - where v2 helped
zero of 32 icons at 0.05.

"No render in this project is a recognizable icon" was true of probability 0.35, the only
level this gate ever trained or reported at, and is obsolete as a general claim. The
ceiling was set by the corruption schedule, not by the representation, the model or the
objective - each suspected in turn and none the binding constraint at the end.

The never_damages criterion is falsified and the diagnosis is kept: at 0.05 nine of
twelve icons improve, two are untouched and one, 26A0, worsens by 0.05018 absolute;
because its x_t error was only 0.01390 the relative statistic turns that into -3.61 and
drags the mean to +0.0308 from a median of +0.3496. The absolute difference does not
rescue it either, so twelve icons is genuinely too few for one bad case.

The framing stays narrow. This removes a handful of stray strokes from an icon that is
mostly intact. It is not a generator, it does not reconstruct destroyed geometry, and
26A0 shows it can still damage a near-perfect input. v16 trained only at 0.35, so every
level below is off-distribution, which makes the result stronger rather than weaker.

Training across corruption levels was then tested directly and is not the lever. On 32
icons, both models at four levels with the same seeds and per-level derived gates, the
paired difference between a range-trained model and v16 is -0.00125 at 0.05, -0.00087 at
0.10 and -0.00324 at 0.20, every interval spanning zero, and -0.01400 at 0.35 with the
interval excluding zero. It buys nothing at the low levels it was meant to help and costs
performance at the level it was meant to trade away. That run's record predicted it
beforehand from the calibration figures, which makes the calibration estimate a usable
cheap predictor of the render outcome.

Those 32 icons also resolve the sample-size limit that falsified `never_damages`. Both
models now improve lightly corrupted inputs significantly - v16 on 26 of 32 icons at
p=0.05, the range-trained model on 29 of 32, both intervals excluding zero - so the
sweep's headline is properly supported rather than suggestive.

Every training-side factor this gate identified is therefore settled: data volume
matters, capacity does not, noise conditioning does not, and the corruption schedule does
not.

Every plateau the sequence had declared - data volume, capacity, the corruption regime,
noise conditioning - was measured under the broken loss and is no longer admissible as
read. The numbers stand; the interpretation does not.

That inference was then checked without training, and it was wrong on its decisive
point. Run `openmoji-g1-detectability-877feff-4processes-9b9b1699` scores every legal
field by a parameter-free local-continuity statistic - mean absolute distance to the
same slot in adjacent segments - and separates corrupted from retained fields at AUC
0.7698 under factorized corruption at p=0.35. At the trained detector's own 8.573% flag
rate it reaches 0.7451 precision and 0.1824 recall, against the model's 0.4312 and
0.1058: a zero-parameter heuristic beats a 579,872-parameter model by 73% on precision.

The information-theoretic claim is therefore withdrawn. The corruption is identifiable;
Gate G's failure is one of learning, not of information. Every measurement in v1 through
v6 stands and so does every predeclared outcome - what does not stand is the inference
from "the model could not detect it" to "it is not detectable", which needed a
model-independent check that cost no training.

The Gate G result is narrower and more useful than first stated: a 580K-parameter
bidirectional transformer trained this way does not learn a corrupted-field detector
that a trivial local statistic already provides, and collapses to the trivial policy
instead. The Gate F implication strengthens rather than weakens: path-correlated
corruption is markedly more identifiable than factorized, 0.9333 against 0.7698, exactly
as a structural argument predicts, and Gate F chose factorized on four-icon fixtures
where recovery was memorization. Gate C and Gate D are untouched: the codec is exact and
render-safe, and none of this is about the representation.


Gate I is complete and its answer is negative in a specific, useful way. A causal model
over the typed SVG codec, matched to Gate G's v16 on parameters, corpus, splits and
training budget, clears a zero-parameter position-marginal floor by 7.7% in nats against
a predeclared 50%, and the renders show what that gain is made of: valid programs in
corpus palette colours with plausible ink coverage and no recognisable shape anywhere.
Five arms measured every factor available - coordinate representation, data volume,
capacity, regularisation - and the gap to the floor moved from 0.28 to 0.26 nats. The
codec supports generation, in that every sample is a valid renderable program and the
legal-token masks make an invalid one unreachable. It does not support learning
generation left to right at any scale this project can reach. The branch this recommends
is an any-order masked model, which is also the only one that could be scored against the
denoiser on one task with one metric - the comparison PROJECT_PLAN.md section 8 asks for
and that a left-to-right sampler cannot enter.

The direction is now set, and the corpus set it. The training bucket is 2,681 unique
programs, 1,597 of its 1,767 variant families are singletons, and 53% of it is
people-body; unconditional generation of unseen concepts from that is not achievable by
any method, and Gate I's sweeps are measurements of that fact. Conditional completion
is achievable: the bucket holds 39,535 contours and every one is a training example for
drawing a missing path from the rest of its icon. The project's result is an editor.
Gate L opens with a masked any-order model over the same flattened codec sequence - mask
families that are the editing operations, decoding in grammatical dependency order so
every output is valid, every Gate G and Gate I correction built in from the first run -
scored by paired render recovery of whole-path inpainting on held-out icons against the
path-dropped icon and a position-marginal sampler. Gate G is closed with the assessment
it asked for, Gate H is deferred behind Gate L, and the single-factor denoiser sweeps and
the corruption-process third arm are retired. PROJECT_PLAN.md section 15 and the
findings entry of the same date carry the argument.

Gate L is complete. Arm 22 - ten times the parameters with the same dropout - ties the
join on four-segment spans like every arm before it (-0.0009, interval spanning zero,
28 of 56 helped, median recovery -0.001), beats the marginal on 49 of 56, and is
indistinguishable span for span from the small model. Every lever available has been
pulled once on the same icons and spans: ownership, mixture, head, start features,
budget, span length, training length, capacity with regularisation. The mechanism is
real and robust - continuity 17.6 to 19.9 bins against 25.6 in every run that carried
it, the first learned geometry in this project - and the editor it makes is harmless
where a straight cut would do, right where continuity decides, and no better than a
guess where shape does; always better than the corpus's statistics, never better than
the simplest geometric policy at the median. What the corpus cannot teach is shape, for
parts as Gate I found for wholes. The findings entry of this date carries the verdict,
the gate board its evidence, PROJECT_PLAN.md section 15 its conclusion.

The operator chose the pretrained route. PROJECT_PLAN.md section 16 opens Gate M: a
permissively licensed pretrained code model fine-tuned with LoRA to write the project's
canonical SVG text from a caption, every output parsed by the typed codec and rendered
by the safe renderer, read against its own zero-shot output and against memorisation on
the family-disjoint splits. Model chosen on license files at pinned revisions:
Qwen3-4B-Base (Apache 2.0) primary, Qwen2.5-Coder-1.5B (Apache 2.0) comparison,
Qwen2.5-Coder-3B excluded under the Qwen Research License. The stack is installed into
`.venv` without displacing the NGC torch - an install that pulled a PyPI torch in was
undone and the suite re-verified on the original. Weights are downloading to the
persistent Hugging Face cache under `/home/dev/.cache`.

Gate M is closed on its answer. Three experiments on the operator's decision, licenses
set aside for research use, every run registered with numbers, sheets and outcomes,
every failure preserved. **OmniSVG 1.1 4B** fine-tuned with LoRA in its own token
language (exact tested encoder) learns OpenMoji's palette and stock parts, not icons:
falsified under three decoders. **Qwen3.5-2B-Base** fine-tuned to write the codec's
canonical SVG from a caption is the best drawing model the project has had - 44 of 64
held-out drawings enter the codec against 3 for its control, a paired CLIP gain of
+0.040 with an interval excluding zero - and is falsified on codec validity (0.69
against 0.8) and retrieval (0.094 against 0.25); the full-corpus arm at 4,096 tokens
writes long icons that do not finish and is worse. **SemIf** is falsified on both
models; averaged over option orders the 4B reaches 0.75, a secondary signal. Retrieval
of the right icon among the 32 held-out renders never rose above 0.094 in any arm
(chance 0.031): neither prior names an icon from a caption at this corpus size. The
direction that remains open is conditioning on more than a caption - a partial icon, a
family sibling, or a reference render - with the text prior as the vehicle, because it
already writes the codec's form. The synthesis is in `reports/findings.md`.

Gate N is closed on its answer, and it is the project's first satisfying result. Given
a held-out icon's own 448 px render, OmniSVG's released image branch returns the icon:
first among the 32 held-out renders for 39 of 64 drawings zero-shot, **49 of 64 with
best-of-six decoding** and 53 of 64 with best-of-twelve - six candidates, the two closest to the input render in pixels
kept - at 0.944 similarity, passing predeclared bars with no training; the sheet is
recognisable almost everywhere (the hedgehog, the anatomical heart and the vulcan
salute still fail). Edits made in pixels on exactly edited programs - recolour, erase,
move - re-vectorised the same way identify the edited icon 71% of the time with half
the pixel error of plain decoding; the recolour is reflected 87% of the time, the
8-unit move is at the limit of what CLIP sees. Fine-tuning the image branch on
OpenMoji's own render-to-program pairs does not help: three arms in two dialects, with
the merger trainable and selection on free-running drawings, fitted the corpus
(likelihood 2.6 to 1.0 nats) and drew worse from their first 50 steps; the failure is
located at the first point and the close, which teacher forcing cannot reach; the
pipeline test on the model's own drawings proves the code sound. The product loop this
gives today: edit in pixels, re-vectorise with best-of-K against the edited render, take
the program. Next, in order of cost: twelve candidates and a pixel-error stop for the
loop; then a sequence-level objective as a new gate if compact programs are needed.

After Gate N the operator asked for something vector-native and proposed merging
icons. Two things came of it. The kitbash tool (`scripts/serve_kitbash.py`, tmux
`mojidiff-kitbash`, port 8789, CPU only, no model) composes parts of two icons as
vectors on the 72-box and exports through the codec at browser speed. Gate O asked
whether the Qwen3.5-2B text prior can learn OpenMoji's own merges from its 1,015 ZWJ
sequences and is closed on no: the fine-tune beats stacking the parts and ties the
first-component baseline - it copies the base figure - and the path provenance of all
merged icons says why: only 17% of a merged icon's paths are exact copies of its
components' paths, 40% have no structural match, and the median icon has 0.895 of its
paths unexplained by its parts. OpenMoji redraws a merge rather than composing it, so
the part of merging that is composition is exactly what the kitbash tool does, and the
rest is the drawing problem Gates M and N measured. The raster demo of Gate N is
stopped; its code stays.

## Last completed action and verification

Closed Gate L on the evidence of twenty-two registered arms, all on gpubox-4080, all in
`state/runs.jsonl` with predeclared criteria and outcomes, preserved failures under
their own identities, and a sheet beside every number under `reports/learning/masked-*`.
The instrument is `src/mojidiff/learning/masked.py` (model, mask families, soft
coordinate target, grammar-ordered and chain-order decoding, start features, metric
head, path binding, span join baseline) and `masked_inpaint.py` (training, selection
against the position-marginal floor under fixed masks, whole-path and span inpainting
against identity and marginal policies, the continuity probe, frozen-checkpoint
evaluation, an explicit evaluation seed). Every option is off by default so every
earlier arm reproduces. Ruff and strict mypy pass; 183 tests pass; the run-record audit
reports the registry consistent; the weblog is rebuilt and serves the conclusion on
its front page.

## Active jobs

The research weblog is served by `scripts/serve_weblog.py` in tmux session
`mojidiff-weblog`, bound to `100.69.189.78:8787` on the Tailscale interface only. It is
a read-only static file server over `site/` and holds no GPU or lock; stop it with
`tmux kill-session -t mojidiff-weblog`. Rebuild its content with
`python -m mojidiff.weblog.build` after any material result. It went dark once, on
2026-09-21, while alive and listening: the server was single-threaded and one
connection that opened and sent nothing blocked every other request. It is threaded
now with a ten-second per-connection timeout, verified to serve while a connection is
stalled; if it ever stops answering again, `tmux kill-session -t mojidiff-weblog` and
relaunch with `--rebuild`. Do not `pkill -f serve_weblog` from a script whose own
command line contains that text.

The kitbash tool is served by `scripts/serve_kitbash.py` in tmux session
`mojidiff-kitbash`, bound to `100.69.189.78:8789`, CPU only, no model; it reads the
cached canonical SVGs under `data/processed/kitbash/`. The raster demo
(`scripts/serve_demo.py`, port 8788) is stopped because it holds the card at 7.9 GiB;
start it only when no registered run needs the GPU, and stop it with
`tmux kill-session -t mojidiff-demo` followed by `pgrep -af serve_demo`.

Gate L runs execute detached in tmux sessions named `masked-<arm>` with stdout under
the run's `data/processed/<study>/stdout.log`; a run that has exited leaves `EXIT=<code>`
as the log's last line. Check `tmux ls` before assuming the GPU is free. The owned
worker is enabled for bounded work under its recorded 20,000-step and 50 GB cap. Vast
and A100 workers remain disabled.

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

The capacity run's artifacts are durable under
`/home/dev/.cache/openmoji-g1-capacity-v3-3c252d5-ee32665b-9b9b1699/` with checkpoint
SHA-256 `93344f19969eb06eb2d99a16967201bd4bb1e7e330cb690551118d81aca17e5c`. Its summary,
metrics, trace, and 128-icon render metrics are copied into the run directory, and its
128-icon contact sheet is versioned under `reports/learning/`.

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

Read the OmniSVG zero-shot control when it finishes; then run the Qwen3.5-2B-Base
control (`configs/learning/prior-m2-control.yaml`) on the same 32 icons, and the
permutation-averaged SemIf re-read of the 4B. With both controls in hand, register the
two fine-tunes with criteria stated against their own controls: codec validity above a
bar, CLIP-to-reference above the control with the interval excluding zero, exact
memorisation bounded, and the sheet. OmniSVG fine-tunes in its native vocabulary on
outlined OpenMoji with LoRA (its training repository assumes full fine-tuning and an
older transformers, so the loop is ours); the text prior fine-tunes on the compact SVG
text with the chunked loss.

Environment note: `transformers`, `peft`, `accelerate`, `flash-linear-attention` (from
source), `qwen-vl-utils`, `shapely`, `networkx`, `einops` live in `.venv`; the NGC torch
stays the one in use. OmniSVG's inference code is checked out at
`/home/dev/workspace/external/OmniSVG` (moviepy is stubbed at import), its training
repository beside it, and SemIf's reference beside those.
