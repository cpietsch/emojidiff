# The selection scalar does not matter, and that is the answer

The twelve-icon comparison falsified the only directional prediction there were grounds
for — that the strictly more accurate checkpoint renders better — but left the two
candidate scalars statistically indistinguishable. This run re-measures the same paired
quantity on the complete 128-icon held-out draw, with the decision rule declared before
the numbers were read.

## Result

Control: `x_t` render error is identical across the two arms for all 128 icons at both
sizes, so the two checkpoints saw exactly the same inputs.

| | 72 px | 18 px |
| --- | ---: | ---: |
| median RGBA MAE, loss-selected step 840 | 0.141215 | 0.144372 |
| median RGBA MAE, accuracy-selected step 1,320 | 0.134577 | 0.138735 |
| mean paired difference (positive favours loss) | -0.001638 | -0.001392 |
| 95% CI on that mean | -0.004867 .. +0.001590 | -0.004982 .. +0.002198 |
| CI half-width | 0.003228 | 0.003590 |
| icons where loss-selected is better | 65 of 128 | 61 of 128 |
| two-sided sign test | p = 0.930 | p = 0.659 |

## Predeclared criteria

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| control_inputs_identical | identical for all 128 | identical | pass |
| separation_or_bound | CI excludes zero, **or** half-width < 0.01 | half-width 0.003228 | pass, by the bound |
| reproducibility | identical artifacts on rerun | identical | pass |

The confidence interval does not exclude zero and the sign test is a coin flip at both
sizes, so the two scalars are not separable. But the interval is now tight: any true
difference is below about 0.0032 RGBA MAE at 72 px, against a median render error of
about 0.14. The choice of selection scalar moves render quality by at most 2.3% of the
error that is already there.

## The declared decision rule, applied

The rule committed before the numbers were read says that if the effect is bounded below
0.01, held-out loss governs selection by default — on the separate ground that it is the
standard early-stopping signal and the rule v1 and v2 already used — and that the record
must state explicitly that the choice is immaterial for render quality at this stage.

**Held-out loss is the project's checkpoint-selection scalar.** It is chosen by
convention, not by evidence of superiority, and the record says so. v2's changed-token
criterion, evaluated at the loss-selected checkpoint, remains correctly recorded as
falsified: reading it at the final step would not have been justified by render quality,
because render quality does not distinguish the two checkpoints at all.

## The twelve-icon medians were an artifact, confirmed

At twelve icons the medians favoured the loss arm by 6.5% (0.142433 against 0.151687).
At 128 icons they favour the *accuracy* arm by 4.7% (0.141215 against 0.134577) — the
opposite direction — while the paired difference stays near zero in both samples. That
is exactly the instability the twelve-icon run's design defect predicted, and it is a
useful reminder: on paired data, compare the pairs.

## Scope

One training trajectory, one seed, one corruption probability of 0.35, one corruption
draw per icon. The conclusion is conditional on argmax decoding, which discards the
distribution beyond the winning token; a sampler that draws from the distribution would
reopen the calibration question and this result would not transfer. Both checkpoints
render as scribble, so this bounds the effect of a selection rule, not output quality.
