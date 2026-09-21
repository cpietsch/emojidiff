# Current research state

Updated: 2026-09-21T18:50:00Z

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

## Last completed action and verification

Implemented optional held-out checkpoint selection and early stopping in the Gate G
pilot at `3b7db57`, gated on a `training.selection_policy` block so every config without
it is untouched: re-running the v1 config reproduces its recorded native summary, trace
and metrics byte-identically, summary SHA-256
`f304a381a6ca57106da5cfe4878b1688e3d6fd558c949243dea01704b7bf6453`. Four focused tests
cover the unchanged fixed-budget shape, argmin selection with a matching written
checkpoint, deterministic patience exhaustion, and rejection of a policy without
periodic evaluation. The full suite passes 112/112; Ruff and strict mypy pass.

Registered and completed the edit-mask run v6 at commit `0228c3b`. All four primary
criteria are falsified, the degenerate-collapse watch fired, and the keep-head detector
diagnostics were measured directly. Its artifacts reproduce identically. The identity
baseline was computed from already-committed metrics at no cost and is recorded in
`reports/learning/identity-baseline.json`.

Registered and completed the noise-conditioned run v4 and the slot-bound run v5, at
commits `6b935b2` and `f502df1`. v4 is falsified; v5 is partially falsified and its
768-parameter change is retained. Both reproduce identical artifacts, and the v1 config
still reproduces its recorded native summary byte-identically after every change.
`GeometryDenoiser` gained optional noise-level conditioning and optional slot binding,
both off by default so every earlier checkpoint still loads and every earlier probe
still reproduces. The slot-invariance evidence is committed as
`scripts/slot_invariance_probe.py` with its output, and a test pins both halves: the
unbound encoder is permutation-invariant by construction, and binding removes it.

Registered and completed the corruption-level sweep
`openmoji-g1-corruption-sweep-71a080a-4levels-9b9b1699` at commit `71a080a`: four
evaluation levels over one fixed checkpoint, CPU only, no training, with every level
reproducing identical artifacts. Its three primary criteria are falsified and the
registered competing hypothesis is confirmed.

Registered and completed the capacity run
`openmoji-g1-capacity-v3-3c252d5-ee32665b-9b9b1699` at commit `3c252d5`, 72.1 s and
342.5 MiB peak CUDA memory, with an identical rerun and a powered 128-icon render check.
Its predeclared criteria are falsified and the negative result is retained.

Registered and completed the selection-scalar comparison
`openmoji-g1-selection-scalar-c4e6a8c-3a34e64b-9b9b1699` and its powered follow-up
`openmoji-g1-selection-scalar-full-c4e6a8c-powered-9b9b1699`. The twelve-icon run
falsified its hypothesis and carried a recorded design defect; the 128-icon run met its
bound criterion and settled the project's selection scalar by a rule declared before the
numbers were read. Both probes reproduce identical artifacts.

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


Gate L's instrument is built, tested and proven on its overfit test. `masked.py` holds
the model, the five mask families, the soft coordinate loss and the grammar-ordered
decoder; `masked_inpaint.py` the harness that trains, selects on held-out masked
likelihood under fixed masks against the position-marginal floor, writes the trace and
checkpoint before any diagnostic, and scores whole-path inpainting by paired render
recovery against the path-dropped icon and the marginal policy. Eleven tests pin it. The
first overfit attempt was falsified by an off-by-one the test exists to catch - every
coordinate exactly one bin off because the soft target was centred a token low - and is
preserved under `masked-overfit-l1-01917d5-4icons-9b9b1699` with `reason_code:
harness_defect`. The rerun on the fixed revision, `masked-overfit-l1-f51c119-4icons-
9b9b1699`, reaches masked-token accuracy 1.000, reproduces 4 of 4 masked whole paths
token for token with pixel-identical renders, and validates every completion; its
loss-reduction criterion is falsified as written because a spread target bounds the
exact-token likelihood at its own entropy, and that is recorded rather than re-run.
Ruff and strict mypy pass; 176 tests pass. The weblog is rebuilt and shows both runs,
the attempt against its own artifacts.

The corpus run `masked-inpaint-l2-54061b2-2681icons-9b9b1699` is falsified as
predeclared: matched to v16 and the causal arm with dropout 0.1, it beats the
position-marginal policy on 54 of 64 held-out icons with the interval excluding zero and
is worse than leaving the hole on 57 of 64, mean paired difference -0.0040 RGBA MAE,
interval [-0.0069, -0.0012], median recovery -0.69 against the 0.30 bar; every
completion valid; 346 s to train, 1.95 GiB peak. A read-only breakdown of the
checkpoint says this is not Gate I's failure: segment-coordinate accuracy is at the
floor on the TRAINING icons too - 0.073 against 0.054, whole-path inpainting helping 0
of 24 - so the model never fit geometry at corpus scale, while it memorises four icons
exactly. It learned style co-occurrence, which the floor already knows, and a little
about segment kinds. The single-factor response is path binding: every position carries
its owning path's index and every segment its index within that path, derived from the
visible lengths, because the packed layout otherwise makes the encoder count
path_length tokens to know which hole is which path's. Implemented behind
`model.path_binding`, off by default, with a test that the ownership follows the visible
lengths and never guesses past a masked one; 177 tests pass. Its overfit half is
registered as `masked-overfit-l3-binding-438a98f-4icons-9b9b1699` and its corpus half is
drafted as `configs/learning/masked-inpaint-l4-binding.yaml`, unregistered until the
overfit passes.

The bound corpus arm `masked-inpaint-l4-binding-438a98f-2681icons-9b9b1699` is falsified
and indistinguishable from l2 on every measure - held-out likelihood 3.843 against 3.869,
8 held-out icons helped against 7, icon by icon a 0.0006 RGBA MAE difference with binding
better on 25 of 64 - so ownership was not the missing information. Two read-only probes
then found the failure. With one segment's coordinates hidden and both neighbours
visible, the model's predicted endpoint sits 55 bins from the truth held out and 39-48
on its own training icons, against 24.5 (median 16.5) for a zero-parameter policy that
copies the previous endpoint and 145 for the marginal argmax; and shifting that visible
previous endpoint by anything from -40 to +40 bins moves the prediction by a median of
0.0. The model never reads coordinate context; it predicts each coordinate from its
position, kinds and styles. Three causes are live - the training mixture, in which
seven of ten masked coordinates have no visible in-path neighbour; the one-unit target
kernel, though a computation on the record shows it already orders coarse misses for a
spread prediction; and the categorical output head - and the harness now measures the
continuity probe against the copy policy on every run. Arm 5,
`masked-continuity-l5-78d4545-2681icons-9b9b1699`, is running: the same model trained on
single-segment masks alone at a third of the budget, predeclared to beat the copy
policy. It separates the mixture from the rest.

Arm 5 answers the mixture question: `masked-continuity-l5-78d4545-2681icons-9b9b1699`,
the same model trained on single-segment masks and nothing else, puts its endpoint 59.7
bins from the truth against 25.6 for the copy-the-previous-endpoint policy, at 0.947 of
the marginal floor with the training loss flat from step 300. Continuity is not learned
even as the only task, so the training distribution is eliminated. The output head is
the next single factor: a linear map onto 289 unordered bins cannot place a bump at a
copied value until it has learned an ordering over them, and nothing rewards the copy
until it can. The metric head - coordinate logits as the inner product of a projection
of the state with the input side's Fourier features of each bin, plus the categorical
bias - is implemented behind `model.metric_head`, off by default, pinned by a test that
it touches coordinate logits only and still completes valid programs. Arms 6 and 7 are
chained in one tmux session: the overfit half `masked-overfit-l6-metric-head-5475283-
4icons-9b9b1699` and, only if it passes, `masked-continuity-l7-metric-head-5475283-
2681icons-9b9b1699`, arm 5 with the head and nothing else changed. If the head is not
it either, the target kernel mixture is next, then the harder question of whether a
289-way categorical over an absolute lattice is the right output at all.

Arm 6, the metric head's overfit half, passed exact reproduction (4 of 4 masked paths,
pixel-identical renders, continuity error 0.0 bins on the training icons) and missed the
0.99 accuracy bar at 0.958, still rising - a property of a head whose finest Fourier
period is 4.5 bins, recorded as falsified as written and not re-run. Arm 7 then tested
nothing: its held-out trace reproduces arm 5's to four decimals, because the head read
a coordinate's role from the kind token in the input and every editing family hides the
kind with its coordinates, so the head never fired at a trained position. Preserved as
`masked-continuity-l7-metric-head-5475283-2681icons-9b9b1699` with `reason_code:
harness_defect`. The head now takes teacher-forced kinds during training - as the loss
takes its legal masks from the clean sequence - and the committed kinds at decode time,
pinned by a test that it fires at a hidden coordinate when the kinds are supplied. Both
halves are re-running chained on revision `62db169` as `masked-overfit-l6b-metric-head-
62db169-4icons-9b9b1699` and `masked-continuity-l7b-metric-head-62db169-2681icons-
9b9b1699`; the continuity half launches only if the overfit half passes exact
reproduction and validity. The lesson, the same one the first overfit test taught: a
change can be carried by a run without being exercised by it, and an identical trace is
the sign.

Arms 6b and 7b close the head question. With the head on at every trained position the
overfit half passes all three criteria (accuracy 0.996, 4 of 4 exact, all valid), and
the continuity half - whose untrained trace starts at 6.08 against arm 5's 5.20, so the
change was exercised - converges to the same place: 58.3 bins against the copy policy's
25.6, held-out likelihood 0.947 of the floor. Four single-factor arms have now left the
model unable to read its neighbours. The causal model of Gate I is the useful contrast:
it could fit its coordinates because under the causal shift the previous coordinate is
the input at the prediction position itself; here it sits seven slots away, in a
position that depends on the previous kind, behind learned absolute embeddings over
packed slots, and attention has to discover a fetch that is rewarded only once it
exists. Arm 9/10 makes the masked model's input as local as the causal model's: every
segment block carries Fourier features of its own start point - the previous segment's
endpoint, or the header's start for a path's first segment - when visible, nothing when
hidden, behind `model.start_features` with a test. Both halves are chained on revision
`4c9d1e3` as `masked-overfit-l9-start-4c9d1e3-4icons-9b9b1699` and
`masked-continuity-l10-start-4c9d1e3-2681icons-9b9b1699`.

Arm 9, the start-features overfit half, reached accuracy 1.000 and validity and
reproduced 3 of 4 masked paths exactly - the fourth committed out of chain order, a
segment decoded while the segment it hangs from was still a hole, which is what a
confidence-ordered coordinate tier does to a model that reads its start. Chain-order
decoding is built behind `decoding.chain_order` - each pass commits every masked segment
whose start is known - pinned by a test that no segment is committed before the one it
hangs from, and the overfit half re-runs with it as `masked-overfit-l9b-start-chain-
e52b43e-4icons-9b9b1699`, queued behind arm 10. Arm 10, the continuity half, was
launched on accuracy and validity because single-segment masks always have a visible
start and the decoder order does not enter.

Arm 10 is the first arm in Gate L that moves anything. With each segment's start point
in its own input, continuity error falls from 58-60 bins to 47.8 (median 38.5), held-out
likelihood from 4.532 to 4.119 - 0.861 of the floor where every earlier arm sat at
0.947 - and the held-out curve is still falling at the last evaluation where every
earlier arm was flat from step 300. Still short of the copy policy's 25.6, and
unconverged at a third of the budget. The blocker was the fetch: five arms could not
make attention find the previous endpoint seven slots away, and putting the value in
the segment's own input starts the mechanism within the same budget. Two one-factor
follow-ups are queued behind the chain-order overfit re-run: arm 11, the same run with
the metric head at the same 2,100 steps, which should make copying the start a linear
map now that the value is there to copy; and arm 12, the same run at the full 6,300
steps, to see where the curve goes.

## Active jobs

The research weblog is served by `scripts/serve_weblog.py` in tmux session
`mojidiff-weblog`, bound to `100.69.189.78:8787` on the Tailscale interface only. It is
a read-only static file server over `site/` and holds no GPU or lock; stop it with
`tmux kill-session -t mojidiff-weblog`. Rebuild its content with
`python -m mojidiff.weblog.build` after any material result.

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

Read arms 9b, 11 and 12. Whichever of 11 and 12 comes closest to - or beats - the copy
policy is the configuration the corpus inpainting run repeats with, decoded in chain
order, on l2's criteria with the continuity probe beside the render. If neither beats
the copy policy, the two combine (start features, metric head, full budget) as one
further arm before the output representation itself is questioned - predicting an
offset from the start point rather than an absolute bin is a codec-level change and
would be recorded as such before any run.

Once the mechanism exists, the scored task follows it: the editing result this gate can
reach is the local one - spans, refinements, restyles - with whole-path completion kept
as the reported hard case rather than the gate. Whole-path completion of an arbitrary
path is close to generation of a part, and Gate I says parts of unseen concepts are as
unseen as wholes.

What is retired, so it is not picked up again by habit: further single-factor sweeps on
the p = 0.35 denoiser, further left-to-right arms, and the corpus-scale corruption-process
third arm (`openmoji-g1-corruption-process-corpus-76f41a3-2arms-9b9b1699` stays as
recorded). The KV-cache measurement stands: 16% at this scale, re-measure rather than
argue.
