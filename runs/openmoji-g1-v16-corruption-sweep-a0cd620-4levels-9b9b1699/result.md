# It produces recognizable icons — the regime was the problem, not the model

Every result in this gate has been at corruption probability 0.35, where the input is
already scribble, and every report has carried the same line: no render in this project
is a recognizable icon. This run asks whether that was a statement about the model or
about the regime.

It was about the regime.

## Result

v16 rendered at four corruption levels, each gated at the threshold derived for that
level on the withheld calibration icons:

| p | threshold | median `x_t` | median `x_hat_0` | mean recovery | 95% CI | helped |
| ---: | ---: | ---: | ---: | ---: | :---: | ---: |
| 0.05 | 0.870 | 0.03730 | **0.02235** | +0.0308 | [−0.629, +0.690] | 9/12 |
| 0.10 | 0.815 | 0.08247 | **0.04206** | **+0.3902** | [+0.227, +0.554] | 11/12 |
| 0.20 | 0.760 | 0.12568 | 0.10014 | +0.2469 | [+0.089, +0.405] | 10/12 |
| 0.35 | 0.680 | 0.16955 | 0.13373 | +0.2511 | [+0.141, +0.361] | 11/12 |

The same sweep over the v2 checkpoint gave −4.6517, −1.2024, −0.1315, +0.1800 and helped
**zero** of 32 icons at p=0.05.

| criterion | observed | outcome |
| --- | --- | --- |
| never_damages | CI excludes zero at 0.10, 0.20, 0.35 but not 0.05 | **falsified** |
| improves_a_lightly_corrupted_input | median 0.02235 against 0.03730, a 40% improvement | pass |
| reproducibility | identical artifacts at every level | pass |

## The p=0.05 failure is one icon and a fragile statistic

Nine of twelve icons improve, two are untouched, and exactly one worsens: `26A0`, the
warning triangle, by 0.05018 absolute. Because its `x_t` error was only 0.01390, the
relative statistic turns that into −3.61 and drags a mean of twelve to +0.0308 from a
median of **+0.3496**.

Switching to the absolute difference does not rescue it either — mean +0.01070 with a
95% interval of [−0.00204, +0.02344]. So this is not only a bad choice of statistic; at
twelve icons one bad case is genuinely enough to prevent significance. The criterion is
falsified as written and the honest summary is that the model improves nine of twelve
lightly corrupted icons by a median 35% and damages one.

## What the renders show

At **p=0.05** the predictions are recognizable icons, and visibly repaired: the
hedgehog's stray diagonal slash removed, the first-aid kit's diagonal gone, "WC" cleaned
to near-perfect, the vampire's face restored from under a scribble. `26A0` is the
visible failure — a near-perfect input triangle comes out broken.

At **p=0.10** the repairs are larger and still land. `1F3CB`, the weightlifter, goes from
heavy scribble to a clean, recognizable figure; `1F199` comes back as a clean "UP!"
badge; the vampire and the WC sign are both recognizable again. This is the level with
the best recovery of the four, +0.3902.

At **0.20 and 0.35** neither input nor output is recognizable, which is what every
earlier report described — correctly, for those levels.

## What this changes

The standing line that "no render in this project is a recognizable icon" was true and
is now obsolete. It described corruption probability 0.35, which has been the only level
this gate ever trained or reported at, and at which a third of the geometry is simply
gone. At 0.05 and 0.10 the same checkpoint produces recognizable output and performs
real repairs.

The honest framing is narrow and should stay narrow. This is a denoiser that removes a
handful of stray strokes from an icon that is mostly intact — not a generator, and not
a model that reconstructs destroyed geometry. But it is the first result in this project
where the output is something a person would recognise, and it means the ceiling was set
by the corruption schedule rather than by the representation, the model or the objective.

## Scope

v16 trained only at 0.35, so 0.05 through 0.20 are off-distribution by construction —
which makes the result stronger, not weaker, since the model was never shown these
levels. Twelve icons, one corruption draw each, single seed, no training. Recognisability
is a judgement and is reported as one.

The obvious next run follows: train across a range of corruption levels and report at
each, rather than training and reporting at a single destructive point.
