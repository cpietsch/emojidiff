# Gate G data-scale run v2

The v1 falsification left one question: was the failure a matter of data scale? v1
trained the selected 577,552-parameter geometry denoiser on a 256-icon family-disjoint
subsample and overfitted it - held-out loss bottomed out at step 240 and then rose to
11.1485 while training accuracy climbed to 0.4550.

This run changes the train split to the complete 2,681-icon family-disjoint split and
nothing else about the learning problem: model, seed 3101, corruption probability 0.35,
batch size 16, learning rate 0.001, the 128-icon validation draw, the held-out
corruption seeds, and the evaluation cadence are all unchanged. The fixed step budget
becomes a 6,300-step cap plus held-out early stopping, because a fixed budget is
precisely what made v1 report an overfitted model. That stopping rule is a second
change, so v1 is read at *its own* held-out loss minimum, step 240, rather than at its
reported step 600. Both arms are therefore compared under the same selection rule.

The predeclared criteria were committed at `a50b2e0` before launch.

## Result

Completed natively on the owned RTX 4080 in 75.6 seconds, peak 290.1 MiB CUDA memory.
Early stopping ended the run at step 1,320 after eight evaluations without improvement;
the selected checkpoint is step 840. A second complete invocation returned an identical
JSON result.

### Held-out trace

| step | held-out aggregate | changed | retained | held-out loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 (untrained) | 0.0037 | 0.0030 | 0.0041 | 15.3908 |
| 60 | 0.0700 | 0.0228 | 0.0954 | 10.5741 |
| 120 | 0.1475 | 0.0328 | 0.2091 | 9.3253 |
| 180 | 0.2002 | 0.0383 | 0.2871 | 8.6219 |
| 240 | 0.2253 | 0.0408 | 0.3243 | 8.2993 |
| 300 | 0.2466 | 0.0420 | 0.3564 | 8.0193 |
| 360 | 0.2566 | 0.0445 | 0.3706 | 7.6764 |
| 420 | 0.2627 | 0.0469 | 0.3786 | 7.6276 |
| 480 | 0.2692 | 0.0484 | 0.3878 | 7.5239 |
| 540 | 0.2735 | 0.0493 | 0.3940 | 7.5360 |
| 600 | 0.2785 | 0.0519 | 0.4002 | 7.4659 |
| 660 | 0.2823 | 0.0532 | 0.4054 | 7.4575 |
| 720 | 0.2824 | 0.0532 | 0.4055 | 7.4163 |
| 780 | 0.2844 | 0.0542 | 0.4080 | 7.3897 |
| **840** | 0.2886 | 0.0573 | 0.4128 | **7.3333** |
| 900 | 0.2877 | 0.0559 | 0.4122 | 7.3702 |
| 960 | 0.2902 | 0.0584 | 0.4146 | 7.5195 |
| 1020 | 0.2920 | 0.0600 | 0.4167 | 7.4133 |
| 1080 | 0.2912 | 0.0615 | 0.4146 | 7.5393 |
| 1140 | 0.2950 | 0.0617 | 0.4203 | 7.5728 |
| 1200 | 0.2958 | 0.0637 | 0.4204 | 7.6322 |
| 1260 | 0.2961 | 0.0646 | 0.4205 | 7.6349 |
| 1320 | 0.2999 | 0.0662 | 0.4254 | 7.5275 |

Held-out totals are 13,978 changed and 26,030 retained fields, identical to v1 because
the validation draw and corruption seeds are unchanged. The bold row is the selected
checkpoint.

### Against the v1 baseline, both read at their own held-out optimum

| measure | v1 @ step 240 | v2 @ step 840 | change |
| --- | ---: | ---: | ---: |
| held-out loss | 9.7932 | 7.3333 | -25.1% |
| held-out aggregate accuracy | 0.2091 | 0.2886 | +38.0% |
| held-out changed-token accuracy | 0.0439 | 0.0573 | +30.7% |
| held-out retained-token accuracy | 0.2978 | 0.4128 | +38.6% |
| train accuracy, mean of 10 steps to the selected step | 0.3654 | 0.3828 | +4.8% |
| train minus held-out aggregate gap | 0.1563 | 0.0942 | -39.7% |

Training accuracy is nearly unchanged while every held-out measure improves, so the
generalization gap narrows by 39.7%. That is the signature of a data-scale effect
rather than an optimization one.

## Predeclared criteria

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| data_scale_lowers_held_out_loss | < 9.7932 | 7.3333 | pass |
| generalization_gap_narrows | < 0.1831 | 0.0942 | pass |
| held_out_changed_recovery_improves | >= 0.065782 (1.5x) | 0.0573 (1.307x) | **falsified** |
| later_held_out_optimum | > step 240 | step 840 | pass |
| structural_safety | locked-path exact, checkpoint round-trips | both true | pass |
| reproducibility | identical rerun JSON | identical | pass |

Standing Gate G bar, declared as a bar rather than as a prediction of this run:

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| retained_preservation | >= 0.90 | 0.4128 | falsified, as expected |

Overall: **partially falsified**. Ten times the data buys a large, consistent
generalization improvement, but not the predicted 1.5x in changed-token recovery.

## The finding this run actually produced

The predeclared 1.5x bar is missed at the selected checkpoint and met at the last
trained step. Changed-token accuracy at step 1,320 is 0.0662, which is 1.511x the
baseline. The two scalars disagree about when to stop:

- held-out **loss** reaches its minimum at step 840 and never recovers;
- held-out **accuracy** - aggregate, changed, and retained alike - keeps rising
  monotonically through step 1,320, well past that minimum.

The criterion was evaluated exactly as predeclared, at the selected checkpoint, so it is
recorded as falsified. Reading it at the final step instead would be choosing the
stopping rule after seeing which one passed.

The disagreement itself is the result worth carrying forward. v1 saw a weaker version of
it and concluded that held-out loss was the more honest scalar. v2 shows that acting on
that conclusion has a measurable cost: loss-based selection gives up 15.6% of the
relative changed-token recovery available at the cap. Cross-entropy punishes confident
errors while accuracy counts only the argmax, so a model can keep getting more answers
right while becoming worse calibrated. Which scalar should govern selection is now an
open question in its own right, and it should be settled deliberately - on a criterion
declared before the next run - rather than by whichever reading is convenient.

## What this does and does not establish

It establishes that at this model size the 256-icon result was data-limited, and that
the full split substantially closes the generalization gap. It does not establish that
data scale alone reaches useful recovery: at 0.0573 changed-token accuracy the model
recovers about one corrupted geometry field in seventeen. Retained-token preservation
improves from 0.2978 to 0.4128 but remains far from the 0.90 bar.

The bottleneck has moved. Ten times the data no longer produces a proportional gain in
changed-field recovery, which points at model capacity, the corruption schedule, or the
single-shot prediction objective rather than at data volume. This remains
fixed-topology, geometry-only, single-seed work; it is not evidence about unconditional
generation, topology, or style.
