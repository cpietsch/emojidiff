# With the leak closed, the model learns geometry — and still loses to nine parameters

v8 reached 2.806 precision lift, which looked like a breakthrough until a judge pointed
out that a detector knowing only the corpus token marginal reaches **2.831** on the same
data with zero parameters and zero context. Factorized corruption draws replacements
uniformly from the full legal vocabulary, so a large share land on tokens real icons
essentially never use, and spotting them needs no geometry at all.

This run removes that shortcut. Replacements are drawn from the corpus distribution for
the field's own role — keyed by segment kind as well as slot, since slot 0 is an endpoint
for a LINE and a control handle for a CUBIC — so every corrupted token is
in-distribution by construction. The marginal is estimated from train programs only.
Nothing else changes from v8.

Measured on 128 held-out icons before the run, at the matched 8.573% flag rate:

| free detector | under factorized | under marginal |
| --- | ---: | ---: |
| marginal density (0 params, no context) | 2.840 | **0.771** |
| local continuity (0 params, neighbours) | 2.126 | **1.893** |

The density shortcut is gone — below chance, because corrupted tokens are now typical —
while the genuine geometric signal survives.

## Result

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| beats_the_free_geometric_detector | > 1.893 | 1.506 | **falsified** |
| not_explained_by_density | > 1.20 | 1.506 | pass |
| structural_safety | locked-path exact, round-trips | both true | pass |
| reproducibility | identical rerun | identical | pass |

Selected step 2,340 of 2,820 run, early stopped — against v8's full 6,300 with no stop.
The harder task converges sooner.

## What 1.506 means

It is the first honest detection number this project has produced, and it says two
things at once.

**The model does learn geometry.** At 1.506 it is far above the 0.771 a density detector
manages here, and above the 1.20 bar set to rule that out. Whatever it is using, it is
not the marginal, because the marginal no longer helps.

**It still loses to a zero-parameter statistic.** The local-continuity heuristic reaches
1.893 on the identical corrupted inputs. A 579,872-parameter transformer, trained under
sound optimisation for 2,340 steps, extracts less of the geometric signal than the mean
absolute distance to the same slot in adjacent segments.

And v8's 2.806 is now properly read: mostly the leak. The honest ranking under leak-free
corruption is continuity 1.893, **v9 1.506**, density 0.771.

## What this settles and what it opens

Settled: the corruption process was leaking, v8's apparent success was largely that
leak, and the fix is real and cheap. Every detector number measured under factorized
corruption — v6's 1.234, v7's 1.365, v8's 2.806, and the 2.13 and 2.83 free references —
describes a task that is partly a density test. This run's 1.506 replaces them as the
reference point.

Opened, and exactly as this run's predeclared falsification meaning anticipated: the
model cannot beat a trivial geometric statistic even with the shortcut removed and the
optimisation fixed, so the remaining candidates are representational. The diagnostic
panel measured two, independently:

* **coordinate values enter as identities, not magnitudes.** The trained input embedding
  is statistically indistinguishable from random initialisation — I confirmed Spearman
  −0.136 between embedding distance and bin distance, adjacent-bin distance 0.989 of the
  all-pairs mean — while the *output* head, where the corpus marginal supplies a direct
  gradient, does show order at +0.143 and 0.882. The continuity statistic works by
  subtracting decoded view-unit values, which is precisely the operation a table of 418
  unordered categories cannot support.
* **six fields are summed into one 96-dim slot.** About 16 dimensions per field, before
  attention runs.

The representation lens measured the first in a controlled simulation: with the current
encoding a probe on the neighbour-difference target scores AUC 0.5024, chance; replacing
the categorical lookup with a numeric code of the same value gives 0.9981. That is the
next run, and it is a single factor.

## Scope

Detection-only, single seed, probability 0.35. Not comparable with v8 in absolute terms:
the task is strictly harder because the easy fraction of corruptions was removed, so
1.506 against 2.806 is not a regression.
