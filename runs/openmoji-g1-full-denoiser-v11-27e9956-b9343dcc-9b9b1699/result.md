# The full objective wrecks the detector it needs

v10 produced the project's first trained model to beat a zero-parameter heuristic on a
leak-free task: detector lift 2.781 against a continuity statistic's 1.893. This run
turns the value head back on, so the model reconstructs as well as detects, and tests
the bar every model here has failed — the identity policy of emitting the input
unchanged.

## Result

| | v11 | identity | best prior (v6) |
| --- | ---: | ---: | ---: |
| aggregate token accuracy | 0.5839 | **0.6515** | 0.6119 |
| changed-token accuracy | **0.0296** | 0.0000 | 0.0132 |
| retained-token accuracy | 0.8803 | 1.0000 | 0.9335 |

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| beats_identity | > 0.6515 | 0.5839 | **falsified** |
| recovers_more_than_identity | > 0.0132 | 0.0296 | pass |
| preserves_what_identity_preserves | >= 0.90 | 0.8803 | **falsified** |
| structural_safety | locked-path exact, round-trips | both true | pass |
| reproducibility | identical rerun | identical | pass |

Changed-token recovery is **2.24x the best any earlier model managed**, which is real.
Everything else falls short, and the reason is exact.

## The value objective destroys the detector

| model | objective | detector lift | detector precision |
| --- | --- | ---: | ---: |
| v10 | detection only | **2.781** | 0.9691 |
| **v11** | **joint** | **1.717** | 0.5983 |
| — | free continuity statistic | 1.893 | — |

Adding the value head costs **1.064 of detector lift** and drops it back below the free
statistic. The two heads share one encoder, and a 289- or 417-way exact-token objective
against a 2-class decision is not a fair fight: the encoder is shaped by the harder task,
which it performs at 0.1087, and the easier one it had solved is collateral damage. The
same pattern appeared at small scale between v6 and v7 (1.234 against 1.365); with a
genuinely good detector to lose, it is now an order of magnitude larger.

## Why no threshold rescues it

Flagging a field is only worth it if the expected gain beats keeping it. Keeping a
retained field is always right; flagging one is right only if the value head happens to
re-predict the same token. With the value head at **0.1087** on genuinely corrupted
fields, the break-even detection confidence is

    p > 1 / (1 + 0.1087) = 0.9019

The model flags 19.15% of fields, far past where it is that confident. But sweeping the
threshold does not save it either — every flag rate loses to identity, monotonically:

| flag rate | aggregate | vs identity |
| ---: | ---: | ---: |
| 0.00 (identity) | 0.6515 | — |
| 0.01 | 0.6496 | −0.0020 |
| 0.05 | 0.6374 | −0.0141 |
| 0.1915 (the model's own gate) | 0.5839 | −0.0668 |
| 0.35 | 0.5137 | −0.1378 |

Working backwards from the 1% point, this detector's precision at its most confident 1%
is only about 0.73 — not the 0.9691 v10 achieved, because this is the degraded joint
detector. With a detector at v10's precision and a value head at 0.1087, flagging the
top percentile would have paid.

So identity is not beaten, and the two ingredients that would beat it have both been
demonstrated — just never at the same time.

## What this localises

The remaining problem is **not** detection: v10 solved that. It is that detection and
reconstruction cannot currently be learned together, and that reconstruction itself is
weak at 0.1087 exact-token accuracy on a 289- or 417-way vocabulary.

Both have concrete, measured next steps, and they are separable:

1. **Stop the two objectives competing.** Weight them by field count rather than letting
   an exact-token softmax dominate, or give the heads separate encoders. v10 and v11
   bracket exactly what is at stake: 2.781 against 1.717 on the identical setup.
2. **Make reconstruction learnable.** The value head predicts an exact bin on a
   quarter-unit metric lattice through a categorical softmax. The task-formulation lens
   argued for a distance-kernel target — mass spread over nearby bins in proportion to
   their distance — so that being close earns gradient. That lens also measured that the
   model is a calibrated localiser being graded pass/fail at ±0.125 units.

## Scope

Fixed-topology and geometry-only, single seed, marginal-respecting corruption at 0.35.
Selected step 840 of 1,320 before early stopping, against v10's 5,340 — the joint
objective also collapses training far sooner. No render was produced for this run; the
token numbers do not warrant one, and that is itself consistent with the standing rule
that no recovery claim is made without one.
