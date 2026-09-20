# Detection-only: the shared encoder was part of it, not most of it

The registered corpus-scale process comparison assumes a trained model can learn
corrupted-field detection on some corruption process. That assumption had never been
tested, and the evidence against it was stark: a zero-parameter local-continuity
statistic reaches 2.13x precision lift on factorized corruption, and the v6 keep head
reached 1.23x.

The suspected reason was objective balance. The edit-mask loss is a 2-class keep
decision plus a 289- or 417-class value prediction, sharing one encoder, so the encoder
is shaped almost entirely by a value task the model demonstrably cannot do. This run
drops the value term entirely. Nothing else changes.

## Result

Precision lift at a matched 8.573% flag rate, over a 0.3494 base rate:

| detector | parameters | precision | lift | recall |
| --- | ---: | ---: | ---: | ---: |
| local-continuity statistic | **0** | 0.7451 | **2.13** | 0.1824 |
| v6 keep head, joint objective | 579,872 | 0.4312 | 1.234 | 0.1058 |
| **v7 keep head, detection only** | 579,872 | 0.4770 | **1.365** | 0.1170 |

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| beats_the_previous_trained_detector | > 1.234 | 1.365 | pass |
| matches_the_free_detector | >= 2.13 | 1.365 | **falsified** |
| reproducibility | identical rerun | identical | pass |

The shared-encoder explanation is real but small. Removing a 289-class objective that
was consuming the entire representation bought 0.13 of lift, closing 15% of the gap to
a heuristic with no parameters at all. The model still cannot see what the statistic
sees.

Held-out loss selected step 660 of a 6,300 cap and early stopping ended the run at
1,140, so it is not budget-limited: the model converged to this.

## What it settles

The predeclared falsification meaning applies. The failure is not about objective
balance and it is not about any particular corruption process, so comparing corruption
processes is premature and the registered two-arm comparison is deferred rather than
run. A comparison between processes is only meaningful once a trained model can learn
the signal on at least one of them.

The conclusion is about this model class and training setup. That is a narrower claim
than the one briefly recorded and withdrawn earlier — that the corruption is not
identifiable — and it is better supported: the signal is demonstrably there, extractable
by a statistic with no parameters, and this network trained this way does not find it.

## Scope

One corruption process at one probability, one seed, one architecture. A detection-only
model reconstructs nothing, so it is a diagnostic rather than a candidate design;
held-out accuracy is meaningless here because the value head is untrained.
