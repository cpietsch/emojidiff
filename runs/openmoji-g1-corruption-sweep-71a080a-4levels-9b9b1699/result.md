# The denoiser is barely reading its input

v2 eliminated data volume as the Gate G constraint and v3 eliminated model capacity,
both on matched predeclared comparisons. The corruption schedule was the only candidate
left, and the render probe had suggested why: at probability 0.35 the input `x_t` is
already visually destroyed, so perhaps no model can recover information that is gone.

This run tests that before spending any training on it. One fixed checkpoint — v2's
loss-selected step 840 — evaluated and rendered at four corruption levels, with 0.35 as
the control because that is what it trained at. Identical icons, sizes and corruption
seeds at every level; only the Bernoulli rate differs. No training.

The statistic is the per-icon **recovery fraction**, `(MAE(x_t) - MAE(x_hat_0)) / MAE(x_t)`
— the share of the render gap the model closes. Absolute error is not comparable across
levels, because `x_t` itself moves closer to `x_0` as corruption falls.

## Result

| corruption p | median `x_t` error | median `x_hat_0` error | mean recovery fraction | 95% CI | icons helped |
| ---: | ---: | ---: | ---: | :---: | ---: |
| 0.05 | 0.04973 | 0.13853 | **-4.6517** | -7.672 .. -1.632 | 0 of 32 |
| 0.10 | 0.08116 | 0.13914 | **-1.2024** | -2.118 .. -0.287 | 4 of 32 |
| 0.20 | 0.13864 | 0.13717 | -0.1315 | -0.285 .. +0.022 | 14 of 32 |
| 0.35 (control) | 0.17606 | 0.14045 | +0.1800 | +0.103 .. +0.257 | 26 of 32 |

At 18 px the picture is the same: -4.07, -1.12, -0.12, +0.19.

## Predeclared criteria

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| recovery_improves_at_lower_corruption | recovery at 0.10 >= 2x recovery at 0.35 | -1.2024 against +0.3600 | **falsified** |
| monotone_in_corruption | non-increasing across 0.05, 0.10, 0.20, 0.35 | -4.65, -1.20, -0.13, +0.18 — strictly *increasing* | **falsified** |
| a_useful_regime_exists | some level exceeds 0.50 recovery | best is +0.18 | **falsified** |
| reproducibility | identical artifacts per level | identical | pass |

The regime hypothesis is falsified and the competing damage hypothesis — registered in
the same record before launch — is confirmed. At probability 0.05 the model makes its
input **2.8x worse**, and it helps on zero of thirty-two icons.

## Why: the output barely depends on the input

The decisive number is not in the table above. It is that `x_hat_0` error hardly moves
at all while `x_t` error moves by a factor of 3.5:

| corruption p | mean `x_t` error | mean `x_hat_0` error |
| ---: | ---: | ---: |
| 0.05 | 0.05098 | 0.13822 |
| 0.10 | 0.08739 | 0.13820 |
| 0.20 | 0.13592 | 0.14728 |
| 0.35 | 0.17958 | 0.14651 |

Per icon, the prediction at p=0.05 correlates with the prediction at p=0.35 at
**Pearson r = 0.91**, with a mean absolute difference of 0.019 against a mean error of
0.14. The model emits nearly the same reconstruction for a given icon whether 5% or 35%
of its geometry has been replaced.

It is not denoising. It has learned a group- and subgroup-conditioned prior over icon
geometry and emits approximately that, largely ignoring `x_t`.

The architecture makes this easy to believe. `GeometryDenoiser` takes the corrupted
tokens plus group and subgroup embeddings and nothing else: there is no noise-level or
timestep input and no mask marking which fields were replaced. Trained at a single fixed
probability of 0.35, it never needed to behave differently at any other level, and
nothing in its input tells it how much to trust what it is given.

## This corrects how the earlier Gate G numbers should be read

The measurements stand; their interpretation does not.

- The +18% recovery at p=0.35 in the first render probe is not the model recovering
  geometry. It is a fixed-quality output happening to beat a badly corrupted baseline.
  The same output scored against a lightly corrupted baseline loses badly.
- Held-out retained-token accuracy of 0.4128 was never a secondary weakness. It was this
  finding, visible in token space the whole time: the model overwrites correct fields
  because it is not reading them. The standing 0.90 bar has been measuring the real
  problem since v1.
- The held-out loss floor near 7.35 that v2 and a 3.53x larger v3 both converge to is
  consistent with both models learning the same prior. Capacity would not help, and
  measurably did not.
- v1's 18x-above-untrained result and v2's 25% loss reduction remain real learning. What
  was learned is a prior over icons, not a conditional denoiser.

## What this redirects the project toward

Not a corruption schedule. The falsification meaning was registered before launch and it
holds: the binding problem is that the model cannot leave correct fields alone, so the
next experiment is an objective or architecture that lets it.

The concrete, cheap candidates, in the order their cost suggests:

1. **Tell the model the noise level.** A timestep or corruption-level embedding, and
   training across a range of levels rather than one fixed point. Without it the model
   cannot modulate how much to trust its input, which is exactly the failure here.
2. **Make copying easy.** Predict an edit mask, or a residual against `x_t`, so that
   "leave this field alone" is the default rather than something the model must
   reconstruct token by token.
3. **Condition on which fields were corrupted**, as a diagnostic upper bound. That is
   not a realistic sampling-time signal, but it cleanly separates "cannot identify the
   corrupted fields" from "cannot predict their values", and that is worth knowing
   before designing around either.

The first is the smallest change that addresses the mechanism observed here, and it is
the one this project should run next.

## Scope

One checkpoint, one seed, 32 held-out icons per level, one corruption draw per icon. The
model trained only at 0.35, so evaluating it at 0.05 is off-distribution by construction
— that is the point of the sweep, but it means a poor result at low corruption does not
by itself condemn a model that was trained across levels. Establishing that is exactly
what candidate 1 above would do.
