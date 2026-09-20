# Gate on closeness, and the render recovery doubles again

v15 localised 27% better than v14 yet **gated itself more tightly** — 0.8604 against
0.8183 — and edited fewer fields. The break-even rule `p > 1/(1+q)` counts a field as
recovered only when the predicted token is exact, so a model that is closer but not
exacter is punished for it. This run picks the threshold by absolute view-unit error
saved on the withheld calibration icons instead.

The trained weights are **identical to v15's** — the checkpoint hash matches — so
everything below is attributable to the gate alone.

## Result

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| lowers_the_threshold | < 0.8604 | **0.6800** | pass |
| improves_paired_render_recovery | > +0.1246, CI excludes zero | **+0.2511** | pass |
| saves_error_on_calibration | > 0 | 2.9843 view units/field | pass |
| structural_safety | locked-path exact, round-trips | both true | pass |
| reproducibility | identical rerun | identical | pass |

**Paired per-icon render recovery**, 12 held-out icons, each model at its own threshold:

| | 72 px | 95% CI | helped | 18 px | helped |
| --- | ---: | :---: | ---: | ---: | ---: |
| identity | 0.0000 | — | — | 0.0000 | — |
| v14, exact-token target, break-even gate | +0.0447 | [+0.008, +0.081] | 8/12 | +0.0729 | 10/12 |
| v15, distance kernel, break-even gate | +0.1246 | [+0.061, +0.188] | 10/12 | +0.1306 | 11/12 |
| **v16, distance kernel, geometric gate** | **+0.2511** | [+0.141, +0.361] | **11/12** | **+0.2745** | **11/12** |

**5.6x v14 and 2x v15**, from changing nothing but the decode rule.

The two corrections compound, and they are the same correction applied twice: train for
closeness, then gate on closeness. Either alone gets part way; together they recover a
quarter of the render gap between the corrupted input and the clean program.

## The tension, which was predeclared

Held-out aggregate token accuracy falls to **0.6354 against identity's 0.6515**. The
model now recovers 25% of the render gap while scoring *worse than doing nothing* on
exact tokens.

That was written into this run's record before it ran, as expected_tension: editing more
fields lowers exact-token accuracy while improving geometry, because most edits land near
rather than on the true bin. It is the v15 finding repeating one level down, and between
the two runs it is now conclusive rather than suggestive:

**Exact-token accuracy is the wrong metric for this project.** A model can be
simultaneously worse than the identity policy on it and the best geometric denoiser the
project has produced. Both statements are true of v16 and neither is a contradiction —
they are a statement about the metric.

## Where the whole sequence lands

| run | change | render recovery, 72 px |
| --- | --- | ---: |
| identity | do nothing | 0.0000 |
| v2 | the model before any of this | strongly negative, destroyed its input |
| v14 | five defects fixed, exact-token target, break-even gate | +0.0447 |
| v15 | distance-kernel target | +0.1246 |
| **v16** | **geometric gate** | **+0.2511** |

Seven verified defects and corrections, none of them tuning knobs, and the model shrank
from 579,872 parameters to 525,152 along the way.

## Scope, unchanged and still binding

Fixed-topology and geometry-only, single seed, marginal-respecting corruption at 0.35,
12 icons for renders. `x_hat_0` is visibly tidier than `x_t` on most rows — the warning
triangle, the tumbler, the badge — but **no render in this project is a recognizable
icon**, and this one is not either. At 35% corruption the input is already scribble, and
recovering a quarter of that gap leaves scribble. What has been established is a
measurable, compounding direction, not a working generator.
