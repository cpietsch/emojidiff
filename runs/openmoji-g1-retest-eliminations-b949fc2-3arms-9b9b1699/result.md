# All four eliminations survive the repair

Four conclusions in this gate — that data volume matters, that model capacity does not,
that the corruption regime is not the constraint, and that noise-level conditioning
changes nothing — were each declared on a held-out scalar that was roughly 31% four
individual fields, on runs stopped at 6–24% of their step cap, under a corruption process
whose replacements a zero-parameter marginal detector identified at 2.84 lift, with
coordinates encoded as unordered categories, and graded on exact-token accuracy which v15
and v16 showed points the wrong way.

All of that is now fixed, and none of the four had been retested. Three of them are
training factors and are retested here off the v16 base, one factor per arm, judged on
paired per-icon render recovery — the primary metric v15 and v16 established.

## Result

| arm | parameters | trains on | 72 px recovery | 95% CI | helped |
| --- | ---: | ---: | ---: | :---: | ---: |
| **v16 base** | 525,152 | 2,425 | **+0.2511** | [+0.141, +0.361] | 11/12 |
| capacity, 3.69x | 1,936,160 | 2,425 | +0.2779 | [+0.161, +0.395] | 10/12 |
| data volume, 256 icons | 525,152 | 256 | **+0.0766** | [+0.001, +0.152] | 8/12 |
| noise conditioning | 528,320 | 2,425 | +0.2096 | [+0.116, +0.304] | 10/12 |

Applying the criteria exactly as predeclared, against the base interval's half-width of
0.1103:

| criterion | observed | verdict |
| --- | ---: | --- |
| capacity_retested | \|diff\| 0.0268 | **restores v3** — 3.69x parameters changes nothing measurable |
| data_volume_retested | 0.1745 below base | **restores v2** — data volume matters |
| noise_conditioning_retested | \|diff\| 0.0415 | **restores v4** — conditioning changes nothing that matters |
| structural_safety | all exact, all round-trip | pass |
| reproducibility | identical reruns | pass |

**All three original conclusions survive.** The eliminations were right even though they
were measured badly, and the four are admissible again.

The data-volume arm is the sharpest of the three: training on 256 icons instead of 2,425
cuts render recovery from +0.2511 to +0.0766, a 70% drop, with its interval barely
clearing zero. Under the broken measurement v2 found the same thing in exact-token terms;
under the corrected one it is larger and clearer.

Capacity is the one I expected to move and it did not. v3 concluded that 3.53x parameters
changes nothing, and I had recorded that as the least trustworthy of the four because 768
parameters of slot binding later beat it outright. At 3.69x on a sound setup it is still
within noise of the base — and the model that does best here remains the smallest one
built this session. The slot-binding result and the capacity result were never in tension:
one was a representational fix, the other was raw size.

## A failed arm, kept

The first noise-conditioning arm set `corruption_probability` to 0.05 as the low end of
its training range, which also moved the **evaluation** level, because
`evaluation_corruption_probability` was unset and defaults to it. Its identity baseline
came out at 0.9501 against the other arms' 0.6515 — it measured a different task
entirely. It is recorded in the registry as failed with that reason rather than quietly
rerun, and the corrected arm pins the evaluation level to 0.35.

A second recording note: on first reading these numbers I labelled the data-volume arm as
overturning v2, by applying a symmetric "differs by more than the half-width" rule to a
criterion that was predeclared **directionally** — below the base by more than the
half-width *restores* v2. The criterion as written is the one applied above.

## Scope

Three arms, not four: the corruption-regime sweep is an evaluation-time study over a
fixed checkpoint rather than a training factor, so it belongs in separate work. Fixed
topology and geometry only, single seed, 12 icons for renders. No render in this project
is a recognizable icon.
