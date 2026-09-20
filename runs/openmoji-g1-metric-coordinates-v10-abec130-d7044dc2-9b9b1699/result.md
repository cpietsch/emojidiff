# Encoding coordinates as magnitudes beats every free detector, with fewer parameters

For the entire Gate G sequence no trained model had beaten a zero-parameter heuristic on
a task without a shortcut in it. This one does.

The diagnostic panel measured why the earlier ones could not. The continuity statistic
that kept winning works by **subtracting decoded view-unit values**, and the encoder had
no such operand: coordinates entered as an `nn.Embedding` over 418 unordered categories,
and the trained table carried no order whatsoever — Spearman **−0.136** between embedding
distance and bin distance, adjacent bins at **0.989** of the all-pairs mean. The *output*
head, which gets a direct gradient from the corpus marginal, did learn order: **+0.143**
and **0.882**. The input side never had a reason to.

This run decodes each token to its view-unit value using the role-correct affine map —
the segment kind selects it, since slot 0 is an endpoint for a LINE and a control handle
for a CUBIC — and projects Fourier features of it, **replacing** the categorical tables
rather than augmenting them.

## Result

| detector | parameters | precision | lift | recall |
| --- | ---: | ---: | ---: | ---: |
| marginal density (0 params, no context) | 0 | — | 0.771 | — |
| v9, categorical input | 579,872 | 0.5248 | 1.506 | 0.1291 |
| local-continuity statistic | 0 | — | 1.893 | — |
| **v10, metric input** | **525,152** | **0.9691** | **2.781** | 0.2384 |

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| beats_the_free_geometric_detector | > 1.893 | 2.781 | **pass** |
| improves_on_categorical_input | > 1.506 | 2.781 | **pass** |
| structural_safety | locked-path exact, round-trips | both true | pass |
| reproducibility | identical rerun | identical | pass |

Every criterion passes. At the base flag rate it reaches 0.7182 precision and recall for
a 2.061 lift, against v9's 1.232.

**The model got smaller.** 579,872 → 525,152 parameters, because the categorical tables
are not allocated at all under metric encoding. A 54,720-parameter *reduction* nearly
doubles the detector. That rules out capacity as the explanation, which the earlier
capacity arm had already eliminated from the other direction.

The run used 5,340 of 6,300 steps before early stopping, against v9's 2,340 — with a
usable input representation there is more to learn and it keeps learning longer.

## The arc this completes

Four defects, each verified before it was acted on, none of them a tuning knob:

| # | defect | fix | detector lift |
| ---: | --- | --- | ---: |
| — | starting point (v7) | — | 1.365 |
| 1 | loss averaged over groups, not fields — four fields carried 31% of it | pool over fields | |
| 2 | no attention padding mask — 29.4% of the sequence attended as content | mask padding | 2.806 (v8) |
| 3 | corruption drew uniformly, so a token-marginal detector scored 2.84 with no context | draw from the corpus marginal | 1.506 (v9, honest task) |
| 4 | coordinates encoded as unordered categories, so no difference was computable | decode to view units | **2.781 (v10)** |

Defect 3 is the one that matters for reading the others: it revealed that v8's 2.806 was
mostly the leak, and it is why v9's 1.506 — not v8's number — is the baseline this run
improves on. Without closing the leak first, v10's 2.781 would have been unreadable too.

## Scope, honestly

This is detection, not reconstruction. The model identifies which fields were corrupted;
it does not yet put back what was there. Held-out accuracy is meaningless for a
detection-only model and the identity baseline is untouched by this run. No render in
this project has yet produced a recognizable icon.

What it establishes is that the representation and the training setup can support
learning a real geometric relation — which nothing before it had shown. The natural next
run applies all four fixes to the full denoising objective and re-tests the identity
baseline, which is the bar every earlier model failed.

Single seed, probability 0.35, marginal-respecting corruption. Slot binding was switched
off in the same change, disclosed in the run record: it existed to break the permutation
symmetry of six summed embeddings, and concatenated per-slot features already distinguish
them, so leaving it on would have carried a mechanism that no longer does anything.
