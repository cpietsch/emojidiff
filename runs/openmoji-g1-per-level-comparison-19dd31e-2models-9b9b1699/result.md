# Training across corruption levels is not the lever

The sweep showed v16 — trained at probability 0.35 alone — producing recognizable icons
at 0.05 and 0.10, levels it had never seen. The obvious inference was that a model which
samples its corruption level over 0.05–0.50 and is told the level would do better there.
It does not.

Both checkpoints already existed, so this is an evaluation study: 32 icons rather than
the previous 12, both models at four levels on the same icons with the same corruption
seeds, each gated at the threshold derived for that level on its own withheld icons, and
reported per level. The statistic is the paired per-icon **absolute** improvement, because
at low corruption the relative form divides by a denominator going to zero — which is
what prevented significance last time.

## Result

| p | model | mean absolute improvement | 95% CI | helped |
| ---: | --- | ---: | :---: | ---: |
| 0.05 | v16 | +0.01797 | [+0.0098, +0.0261] | 26/32 |
| 0.05 | range | +0.01671 | [+0.0114, +0.0220] | 29/32 |
| 0.10 | v16 | +0.02987 | [+0.0226, +0.0372] | 30/32 |
| 0.10 | range | +0.02899 | [+0.0221, +0.0359] | 31/32 |
| 0.20 | v16 | +0.03684 | [+0.0244, +0.0493] | 28/32 |
| 0.20 | range | +0.03360 | [+0.0251, +0.0421] | 30/32 |
| 0.35 | v16 | **+0.03846** | [+0.0245, +0.0524] | 27/32 |
| 0.35 | range | +0.02446 | [+0.0161, +0.0328] | 28/32 |

Paired difference, range minus v16:

| p | difference | 95% CI | range better on |
| ---: | ---: | :---: | ---: |
| 0.05 | −0.00125 | [−0.0092, +0.0067] | 17/32 |
| 0.10 | −0.00087 | [−0.0068, +0.0050] | 14/32 |
| 0.20 | −0.00324 | [−0.0138, +0.0073] | 17/32 |
| 0.35 | **−0.01400** | **[−0.0264, −0.0016]** | 13/32 |

| criterion | observed | outcome |
| --- | --- | --- |
| range_model_wins_at_low_corruption | −0.00125, CI spans zero | **falsified** |
| range_model_does_not_lose_at_the_trained_level | −0.01400, CI excludes zero | **falsified** |
| both_models_help_at_every_level | every CI excludes zero | pass |
| reproducibility | identical artifacts, all eight runs | pass |

Training across levels is **indistinguishable** from single-level training at 0.05, 0.10
and 0.20, and **measurably worse** at 0.35, the level it was supposed to trade away. It
buys nothing and costs something.

## The calibration figures said so first

This run's record, written before it ran, noted that the range-trained model saves fewer
view units per field on its calibration icons at every level — 0.4134 against 0.4636 at
0.05, and 1.9435 against 3.0133 at 0.35 — and flagged that as arguing against the
hypothesis. It was right, and it cost nothing to check: the calibration estimate is a
usable predictor of the render outcome, which is worth knowing for future runs.

## What 32 icons settles

The previous sweep could not clear zero at p=0.05: nine of twelve icons improved but one
bad case dragged the interval across zero. At 32 icons **both models improve lightly
corrupted inputs significantly** — v16 on 26 of 32, the range model on 29 of 32, both
intervals excluding zero. The earlier failure was sample size, and the corrected rule
about using absolute differences at low corruption did its job.

So the sweep's headline result is now properly supported rather than suggestive: this
model repairs lightly corrupted icons, significantly, at every level tested.

## What is left

The predeclared falsification meaning applies. v16's extrapolation from a single level is
as good as training across levels, so the training schedule is not the remaining lever.
Together with the earlier retests — data volume matters, capacity does not, noise
conditioning does not — every training-side factor this gate has identified is now
settled.

That leaves **Gate I**, the cached autoregressive baseline on the same codec, which is
PROJECT_PLAN.md section 12 branch 4. It is the only untried route to a generation result
rather than a denoising one, and nothing in the project blocks it.

## Scope

No training; both checkpoints predate this run. 32 icons, one corruption draw per icon
per level, single seed, fixed topology and geometry only. This compares two denoisers;
neither is a generator.
