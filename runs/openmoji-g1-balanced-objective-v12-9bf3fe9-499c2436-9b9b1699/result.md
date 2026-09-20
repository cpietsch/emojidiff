# The first model here to beat doing nothing

Every trained model in this project has scored below the identity policy — emit the
corrupted input unchanged, which is legal under argmax decoding. The best managed 0.6119
against identity's 0.6515. This one clears it.

The change is one number. The keep head decides 2 classes and the value head decides
289 or 417, so at chance they are log 2 and log 417 nats — a factor of **8.7**. An
unweighted sum of the two lets the value task, which the model performs at 0.11,
dominate a shared encoder, and v11 measured the cost: the detector fell from 2.781 to
1.717, below the free continuity statistic. Scaling the value term by log(2)/log(417) =
**0.1149** rebalances them. The weight is derived, not tuned.

## Result

| | v11 unweighted | **v12 balanced** | identity |
| --- | ---: | ---: | ---: |
| detector lift | 1.717 | **2.569** | — |
| changed-token accuracy | 0.0296 | **0.0738** | 0.0000 |
| retained-token accuracy | 0.8803 | 0.8947 | 1.0000 |
| aggregate, model's own gate | 0.5839 | 0.6086 | 0.6515 |

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| restores_the_detector | > 1.893 | 2.569 | pass |
| keeps_a_usable_value_head | >= 0.0296 | 0.0738 | pass |
| beats_identity (own gate) | > 0.6515 | 0.6086 | **falsified** |
| structural_safety | locked-path exact, round-trips | both true | pass |
| reproducibility | identical rerun | identical | pass |

Both ingredients now coexist, which is what this run was for: the detector is nearly
back to v10's detection-only 2.781, and changed-token recovery is 2.5x v11's and 5.6x
the best any earlier model managed. But at the model's own 0.5 threshold it still loses
to identity, so the predeclared criterion is falsified as written.

## Beating identity, honestly

The model's confidence is informative even though its default threshold is not. Sweeping
the gate and reporting on the same icons would be selection on the evaluation set, so the
128 held-out icons were split in half: the threshold was chosen on 64 and reported on the
disjoint 64, which were never used to choose anything.

Chosen on the calibration half: flag the most confident **5%**, p >= 0.9237.

| evaluation half, 64 icons, 19,856 fields | aggregate |
| --- | ---: |
| identity | 0.6502 |
| **model, calibrated gate** | **0.6567** |
| model, own 0.5 gate | 0.6096 |

**+0.0065 over identity**, on icons that played no part in choosing the threshold. It is
the first time anything in this project has beaten the trivial policy.

The margin is small — about 1% relative — and it depends on the calibrated threshold:
the model's own decode still loses by 0.0406. Both facts belong in the claim. What has
changed is not that the model is good, but that it is finally better than nothing, and
that the remaining gap is now a calibration problem rather than a representation one.

## The arc

| run | change | detector | aggregate vs identity |
| --- | --- | ---: | --- |
| v7 | starting point | 1.365 | — |
| v8 | field-pooled loss, padding mask | 2.806 | — (leaky task) |
| v9 | marginal-respecting corruption | 1.506 | — (honest task) |
| v10 | metric coordinate encoding | **2.781** | — (detection only) |
| v11 | value head on | 1.717 | 0.5839 < 0.6515 |
| **v12** | **entropy-balanced objective** | **2.569** | **0.6567 > 0.6502** |

Five verified defects, none of them tuning knobs, and the model shrank along the way:
579,872 parameters at v7, 525,152 at v12.

## What remains

The value head is still weak at 0.1233 exact-token accuracy on corrupted fields, which
is why the break-even detection confidence is high and only the top few percent of flags
pay. The measured next step is the distance-kernel target: the value head predicts an
exact bin on a quarter-unit metric lattice through a categorical softmax, so being close
earns nothing. And the model's own gate needs calibrating during training rather than
after it.

## Scope

Fixed-topology and geometry-only, single seed, marginal-respecting corruption at 0.35,
token accuracy rather than renders. The project's primary/test split remains untouched.
Beating identity by 0.0065 on token accuracy is not a recognizable icon, and no render
in this project has produced one yet.
