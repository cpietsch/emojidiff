# Slot binding: a proven defect, a real gain, and still not enough

The encoder summed six coordinate lookups — all from one shared embedding table — into a
single vector per segment slot. A sum is commutative, so a segment was represented by an
unordered bag of its coordinate values. In float64, a program and its coordinate-swapped
variant produce byte-identical logits: maximum absolute difference exactly 0.0 across 414
swapped segments in six held-out icons, with identical argmax predictions. The model
provably could not tell which coordinate held which value, while still having to predict
all six separately.

This run binds each value to its slot by elementwise multiplication with a learned
per-slot vector. Everything else is the v2 configuration exactly: same split, seed,
fixed corruption 0.35, batch size, learning rate, step cap and selection policy. Noise
conditioning is off, since v4 falsified it. The change costs **768 parameters**, +0.13%.

## Result

Early stopping ended the run at step 1,980; the selected checkpoint is step 1,500 — the
latest optimum of any run so far. A second complete invocation returned an identical
JSON result.

| run | parameters | held-out loss | aggregate | changed | retained | selected step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v2 baseline | 577,552 | 7.3333 | 0.2886 | 0.0573 | 0.4128 | 840 |
| v3 capacity 3.53x | 2,040,976 | 7.3790 | 0.3036 | 0.0639 | 0.4323 | 360 |
| v4 noise-conditioned | 580,720 | 7.4285 | 0.2980 | 0.0556 | 0.4282 | 1,020 |
| **v5 slot-bound** | **578,320** | **7.1782** | **0.3793** | **0.0787** | **0.5408** | **1,500** |

768 parameters beat a 3.53x capacity increase on every measure, which is the clearest
confirmation that the defect was real and structural rather than a matter of resources.
Retained-token accuracy rises 31%, aggregate accuracy 31%, changed-token recovery 37%,
and held-out loss improves for the first time since v2.

## Predeclared criteria

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| retained_preservation_moves | >= 0.70 | 0.5408 | **falsified** |
| loss_improves | < 7.3333 | 7.1782 | pass |
| stops_destroying_light_corruption | recovery at 0.10 above 0 | -0.7607 | **falsified** |
| output_depends_on_input | Pearson r below 0.70 | 0.8621 | **falsified** |
| structural_safety | locked-path exact, round-trips | both true | pass |
| reproducibility | identical rerun | identical | pass |

Standing Gate G bar, unmet since v1: retained-token accuracy >= 0.90, observed 0.5408.

Overall: **partially falsified**.

## The gain is real; the mechanism is not fixed

| corruption p | v2 | v4 | v5 |
| ---: | ---: | ---: | ---: |
| 0.05 | -4.6517 | -4.5349 | -3.1470 |
| 0.10 | -1.2024 | -1.0385 | -0.7607 |
| 0.20 | -0.1315 | -0.0910 | -0.0253 |
| 0.35 | +0.1800 | +0.2083 | +0.2098 |
| Pearson r, 0.05 vs 0.35 | 0.9122 | 0.8619 | **0.8621** |

Every recovery figure improves, and at corruption 0.20 the model is now essentially
break-even where v2 lost ground, helping on 20 of 32 icons rather than 14. But the
correlation between its predictions at 0.05 and at 0.35 is **unchanged** — 0.8621 against
v4's 0.8619 — and mean `x_hat_0` error still moves only from 0.124 to 0.143 across a
sweep where the input moves from 0.051 to 0.180.

So the model reads its input better than it did, and uses it barely more. Fixing the
encoder raised the ceiling of what it produces without changing what it fundamentally
does: emit a prior lightly adjusted by its input.

## What remains

The predeclared falsification meaning holds, and it is now the only candidate left
standing after data volume, capacity, the corruption regime, noise-level information and
encoder slot-blindness have each been eliminated by a predeclared comparison:

The prediction objective itself. The model predicts every field through an independent
single-shot softmax over a 289- or 417-way vocabulary. There is no cheap way for it to
express "leave this one alone" — copying a field requires reconstructing its exact token
from scratch through the transformer, and at 65% of fields uncorrupted that is most of
the work it is being asked to do. An edit-mask or residual formulation, where copying is
the default and the model predicts only what to change, is the change that addresses
this directly.

Slot binding should be kept regardless: it is a strict improvement for 768 parameters
and it removes a defect that would have confounded every later result.

## Scope

Fixed-topology and geometry-only, single seed, one corruption draw per icon at each
sweep level. No run in this sequence has yet produced a recognizable icon, and this one
does not either. It is a better-conditioned model, not a working one.
