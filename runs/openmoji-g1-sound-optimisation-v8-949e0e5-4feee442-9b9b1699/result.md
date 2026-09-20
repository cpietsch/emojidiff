# Two bug fixes move the detector from below-trivial to state of the art, and reveal that the metric leaks

A five-lens diagnostic panel over this codebase found two defects that had been present
through the entire Gate G sequence. Both were verified independently before being acted
on. Neither is a tuning knob.

**The loss averaged over groups, not fields.** `_legal_fields` emits one group per
(segment kind, coordinate slot), and both loss functions averaged across groups. QUAD
segments are 0.93% of the corpus, so on the 128-icon held-out draw the four QUAD groups
hold exactly **one field each** and carried 4/13 — 30.8% — of the coordinate loss against
40,004 other fields. Measured per-field weight ratio between smallest and largest group:
**5,287x**. Groups with no fields are skipped, so the denominator also flipped between 9
and 13 depending on whether a batch happened to contain a QUAD.

**No attention padding mask.** `self.encoder(hidden)` was called bare, so typed padding
slots received attention mass. Measured padding share of the sequence: **29.4%**.

Because `_evaluate` calls the same loss, held-out loss — the checkpoint-selection and
early-stopping signal — was roughly 31% four individual fields.

## Result

| | v7 (defects present) | **v8 (both fixed)** |
| --- | ---: | ---: |
| detector precision at the matched flag rate | 0.4770 | **0.9805** |
| precision lift over the 0.3494 base rate | 1.365 | **2.806** |
| recall at that flag rate | 0.1170 | 0.2406 |
| selected step, of a 6,300 cap | 660 | **6,180** |
| ended by | early stopping at 1,140 | **the cap; still improving** |

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| beats_the_broken_baseline | > 1.365 | 2.806 | pass |
| reaches_the_panel_lower_bound | >= 1.55 | 2.806 | pass |
| uses_more_of_its_budget | step > 660, ran >= 1,140 | 6,180 of 6,300 | pass |
| reproducibility | identical rerun | identical | pass |

Every criterion passes, and by a margin far beyond either lens's prediction — the
optimization lens predicted 1.70 (interval 1.55–1.95), the objective lens 1.75
(1.6–1.9). Both were recorded before the run. The run did not early-stop at all: it
consumed its full budget and was still improving, against v7's stop at 1,140.

**No parameters were added and nothing architectural changed.** 579,872 parameters in
both arms.

## The caveat that matters more than the result

One of the judges found something all five lenses missed, and it reframes the metric.
A detector that scores each field purely by **how rare its own token is for its own
role** — zero parameters, zero context, no neighbours, marginal estimated on clean train
icons — achieves:

| flag rate | precision | lift |
| ---: | ---: | ---: |
| 5.00% | 0.9962 | 2.853 |
| 8.573% (matched) | 0.9887 | **2.831** |

So v8's 2.806 does not beat that; it **matches** it. The mechanism is plain once seen:
corruption draws uniformly from the *full legal* vocabulary, so a large share of
replacements land on tokens real icons essentially never use. Detecting them needs no
geometry at all.

The honest reading of this run is therefore two-sided. The fixes are real, large and
necessary — the model went from losing to a zero-parameter heuristic to matching the
best one available. But precision lift at a fixed flag rate under factorized
role-uniform corruption is **mostly a marginal-density test**, and v8 has most likely
learned the corpus marginal rather than the geometry. The earlier references now sort
differently: continuity 2.13, local features 2.69, marginal density 2.831, v8 2.806.

## What this invalidates

Every plateau in the Gate G sequence was declared by a held-out scalar that was ~31%
four fields, on runs stopped at 6–24% of their step cap: v3 at 360, v7 at 660, v2 at 840,
v4 at 1,020, v6 at 1,380, v5 at 1,500 of 6,300. v8 shows what changes when that signal
is sound — the same model, same data, same hyperparameters runs 6,180 steps and doubles
its detector.

So the eliminations of **data volume (v2), model capacity (v3), the corruption regime
(sweep), and noise-level conditioning (v4)** are no longer admissible as measured. Their
numbers stand; their interpretation as plateaus does not. They should be re-run under
sound optimisation before any of them is cited again.

Two conclusions I drew earlier also fall. That "the encoder does not represent the
signal" rested on a frozen-encoder probe that shares one weight vector across all six
slots — a mis-specified, lower-bound probe — and on v7's broken training; the same probe
on v8 reaches AUC 0.6604 against v7's 0.5797. And the architecture lens's ceiling ladder
predicted ~1.30 lift at 16 dims per field; v8 reaches 2.806 with that same 16 dims, so
the ladder does not bound the trained model.

## Scope

Detection-only, so this model reconstructs nothing and its held-out accuracy is
meaningless. Field-pooled held-out loss is a different scalar from v7's and the two are
not comparable in value. Single seed, factorized p=0.35. The two fixes were applied
together and that is disclosed: separating them would require a baseline that is sound
in neither arm.
