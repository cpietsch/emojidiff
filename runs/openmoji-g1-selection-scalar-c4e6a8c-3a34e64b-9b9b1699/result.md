# Which scalar should select the checkpoint?

v2 left a question its own artifacts could not answer: held-out loss and held-out
accuracy disagreed about when to stop, and v2's primary criterion passed under one
reading and failed under the other.

The question looked decidable by measurement rather than argument. `predict_clean_geometry`
decodes by argmax over the legal vocabulary, so the rendered artifact depends only on
which token wins, not on the probability mass cross-entropy measures. The two candidate
rules select two concrete checkpoints from one training trajectory — step 840 by
held-out loss, step 1,320 by held-out accuracy — and whichever renders closer to `x_0`
should be the better rule for what this project actually delivers.

The predicted answer was accuracy, because the step-1,320 checkpoint is strictly better
on all three accuracy measures.

## The two arms are the same trajectory

The fixed-budget 1,320-step run reproduces v2's first 1,320 metric rows **exactly**, and
its held-out block matches v2's recorded `validation_final_step` to the last digit. The
`x_t` render errors are identical across both probes, confirming the two checkpoints saw
identical inputs. The comparison is valid.

| | loss-selected, step 840 | accuracy-selected, step 1,320 |
| --- | ---: | ---: |
| held-out loss | **7.3333** | 7.5275 |
| held-out aggregate accuracy | 0.2886 | **0.2999** |
| held-out changed accuracy | 0.0573 | **0.0662** |
| held-out retained accuracy | 0.4128 | **0.4254** |
| median RGBA MAE, 72 px | **0.142433** | 0.151687 |
| median RGBA MAE, 18 px | **0.152770** | 0.165363 |

## Predeclared criteria

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| trajectory_reproduces | exact match on 1,320 rows | exact | pass |
| accuracy_selected_renders_better_72 | < 0.142433 | 0.151687 | **falsified** |
| accuracy_selected_renders_better_18 | < 0.152770 | 0.165363 | **falsified** |
| direction_is_consistent | >= 7 of 12 icons | 5 of 12 | **falsified** |
| reproducibility | identical artifacts | identical | pass |

The hypothesis is falsified. Higher token accuracy did **not** produce better renders.

## But the predeclared statistic was the wrong one

Comparing each arm's median is the wrong test for paired data, and it is worth being
explicit that this was a mistake in the run's design rather than a property of the
result. The two probes render the *same* twelve icons, so the informative quantity is
the per-icon difference. It is approximately zero:

| | 72 px | 18 px |
| --- | ---: | ---: |
| mean paired difference (positive favours loss) | -0.00030 | +0.00083 |
| 95% CI on that mean | -0.0194 .. +0.0188 | -0.0196 .. +0.0213 |
| icons where loss-selected is better | 7 of 12 | 7 of 12 |
| two-sided sign test | p = 0.774 | p = 0.774 |

The 6.5% gap between the medians is an artifact of an asymmetric per-icon error
distribution, not an effect. Per-icon outcomes are close to a coin flip and the
individual swings are large in both directions: `1F199` improves from 0.204 to 0.133
under accuracy selection, while `1F6BE` degrades from 0.190 to 0.262.

## What is settled and what is not

Settled: the argmax argument that motivated this run is wrong, or at least incomplete.
Selecting the strictly more accurate checkpoint does not yield a better-rendering one,
so token accuracy is not a reliable stand-in for render quality even under argmax
decoding. v2's changed-token criterion therefore remains correctly recorded as
falsified, and reading it at the final step would not have been justified by render
quality either.

Not settled: which scalar is better. At twelve icons the two checkpoints are
statistically indistinguishable, and this run does not license a preference in either
direction. A follow-up renders the complete 128-icon held-out draw for both arms, which
shrinks the confidence interval by about 3.3x and can either separate them or bound the
effect small enough that the choice can be made on other grounds.
