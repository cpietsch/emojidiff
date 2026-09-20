# The distance kernel works; my criteria measured the wrong thing

The value head predicts an exact bin on a quarter-unit lattice through a 289/417-way
softmax, so a confident one-view-unit miss costs 10.0187 — identical to a miss of a
hundred units. This run spreads the target mass by distance, `exp(-|b−t|·0.25/τ)` with
τ = 1 view unit, so proximity earns gradient. Nothing else changes from v14.

## Predeclared criteria: falsified

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| lifts_the_value_head | q > 0.2220 | 0.1623 | **falsified** |
| widens_the_margin_over_identity | > +0.0027 | +0.0014 | **falsified** |
| lowers_the_derived_threshold | < 0.8183 | 0.8604 | **falsified** |
| structural_safety | locked-path exact, round-trips | both true | pass |
| reproducibility | identical rerun | identical | pass |

All three primary criteria fail. Every one of them is stated in **exact-token** terms.

## Reported not gated: the intervention worked

The measures I committed to reporting tell the opposite story, and they are the ones the
task-formulation lens argued for before this run existed.

**Geometric accuracy of the value head**, on genuinely corrupted held-out fields:

| | v14 exact-token target | **v15 distance kernel** |
| --- | ---: | ---: |
| exact-token accuracy | 0.1210 | 0.1000 |
| mean absolute error, view units | 8.2871 | **6.0282** |
| median absolute error | 3.7500 | **2.5000** |
| within 1 view unit | 0.2793 | **0.3169** |
| within 2 view units | 0.3887 | **0.4590** |

Mean error falls **27%** and median **33%**. The model localises substantially better and
hits the exact bin slightly less often, which is exactly what a distance-aware target is
for.

**Paired per-icon render recovery**, 12 held-out icons, each model decoded at its own
derived threshold:

| | 72 px | 95% CI | 18 px | 95% CI |
| --- | ---: | :---: | ---: | :---: |
| v14 | +0.0447 | [+0.008, +0.081] | +0.0729 | [+0.031, +0.115] |
| **v15** | **+0.1246** | [+0.061, +0.188] | **+0.1306** | [+0.070, +0.191] |
| identity | 0.0000 | — | 0.0000 | — |

v15 recovers nearly **three times** as much of the render gap as v14, helping 10 of 12
icons at 72 px and 11 of 12 at 18 px, with both intervals excluding zero. These are the
first positive render recoveries this project has measured under leak-free corruption.

## The lesson, which is about the criteria

I nearly recorded this as a failure twice over. The predeclared criteria were all in
exact-token terms, and on a first pass I also compared the two models by their **median**
`x_hat_0` error — 0.16074 against 0.16985 — and read v15 as worse. That is the same
mistake this session already caught once: on paired data, compare the pairs. The paired
per-icon recovery says v15 is far better, and it is the correct statistic because both
models render the identical corrupted inputs.

So the honest record is: the criteria are falsified as written, and the intervention is a
clear success on both of the measures I committed to reporting. That combination is a
verdict on the criteria. Exact-token accuracy is anti-correlated with what this project
is trying to produce, which is what the task-formulation lens said before any of this
ran — a calibrated localiser graded pass/fail at ±0.125 units.

## What it changes

**Demote exact-token accuracy.** Mean absolute view-unit error and paired render
recovery should be the primary reported metrics from here, with exact-token accuracy kept
as a secondary so nothing already published is withdrawn.

**The decode gate needs the same correction.** The break-even rule `p > 1/(1+q)` is
derived in exact-token terms, so a value head that is closer but not exacter *raises* its
own threshold — v15's went to 0.8604 — and edits less of what it could improve. A gate
derived from expected geometric gain rather than exact-token gain would let v15 act on
more of what it localises well, and that is the next run.

## Scope

Fixed-topology and geometry-only, single seed, marginal-respecting corruption at 0.35,
12 icons for renders. `x_hat_0` now tracks `x_t` with small corrections, where the same
probe on v2 showed predictions that destroyed their input outright. No render in this
project has yet produced a recognizable icon, and this one does not either — at 35%
corruption `x_t` is already scribble, and a 12% recovery of that gap is still scribble.
