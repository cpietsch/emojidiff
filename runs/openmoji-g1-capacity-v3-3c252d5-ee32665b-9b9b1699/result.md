# Gate G capacity run v3: capacity is not the constraint either

v2 established that data volume is no longer the binding constraint — 10.5x the data
narrowed the train/held-out gap 39.7% but lifted changed-token recovery only 1.307x,
leaving that gap at 0.0942 and suggesting the 577,552-parameter denoiser was close to
fitting what it could express. This run tests that suggestion directly.

Capacity is the single factor. Width and depth scale together, which is how a
transformer's capacity is conventionally expressed, with head dimension held at 24 and
the feedforward ratio at 2x d_model so no other shape parameter moves: 577,552 ->
2,040,976 parameters, 3.53x. Split, seed, corruption probability, batch size, learning
rate, step cap, evaluation cadence, and the held-out-loss selection policy are all
unchanged.

The selection scalar was declared before launch, which matters here: as in v2, held-out
accuracy keeps climbing well past the loss minimum, and at the final step changed-token
recovery reaches 0.0831 — 1.45x the baseline, which would have passed the criterion.
The settled rule says read at the loss-selected checkpoint, and it was settled before
this run existed.

## Result

Completed natively on the RTX 4080 in 72.1 seconds at 342.5 MiB peak CUDA memory. Early
stopping ended it at step 840; the selected checkpoint is step 360. A second complete
invocation returned an identical JSON result.

### Held-out trace

| step | aggregate | changed | retained | held-out loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 (untrained) | 0.0035 | 0.0034 | 0.0036 | 15.7289 |
| 60 | 0.1608 | 0.0341 | 0.2288 | 9.4942 |
| 120 | 0.2485 | 0.0432 | 0.3587 | 8.1581 |
| 180 | 0.2752 | 0.0474 | 0.3975 | 7.6501 |
| 240 | 0.2872 | 0.0544 | 0.4123 | 7.5938 |
| 300 | 0.2982 | 0.0607 | 0.4257 | 7.5106 |
| **360** | 0.3036 | 0.0639 | 0.4323 | **7.3790** |
| 420 | 0.3055 | 0.0680 | 0.4330 | 7.6706 |
| 480 | 0.3054 | 0.0680 | 0.4330 | 7.5067 |
| 540 | 0.3135 | 0.0715 | 0.4434 | 7.5109 |
| 600 | 0.3114 | 0.0755 | 0.4381 | 7.6677 |
| 660 | 0.3145 | 0.0771 | 0.4420 | 7.5621 |
| 720 | 0.3164 | 0.0772 | 0.4449 | 7.6622 |
| 780 | 0.3139 | 0.0823 | 0.4382 | 7.8262 |
| 840 | 0.3171 | 0.0831 | 0.4427 | 7.6919 |

### Against v2, both read at their own held-out loss minimum

| | v2, 577,552 params | v3, 2,040,976 params | |
| --- | ---: | ---: | ---: |
| selected step | 840 | 360 | 2.3x sooner |
| held-out loss | **7.3333** | 7.3790 | +0.6% worse |
| aggregate accuracy | 0.2886 | 0.3036 | 1.052x |
| changed-token accuracy | 0.0573 | 0.0639 | 1.115x |
| retained-token accuracy | 0.4128 | 0.4323 | 1.047x |
| median render RGBA MAE, 72 px, 128 icons | 0.141215 | 0.136983 | |
| mean paired render difference, 72 px | — | +0.001853 | 95% CI -0.0026 .. +0.0063 |

## Predeclared criteria

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| capacity_lowers_held_out_loss | < 7.3333 | 7.3790 | **falsified** |
| changed_recovery_improves | >= 0.071630 (1.25x) | 0.063886 (1.115x) | **falsified** |
| renders_measurably_closer | CI on paired difference excludes zero | -0.0026 .. +0.0063 | **falsified** |
| structural_safety | locked-path exact, checkpoint round-trips | both true | pass |
| reproducibility | identical rerun JSON | identical | pass |

Standing Gate G bar, not a prediction of this run:

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| retained_preservation | >= 0.90 | 0.4323 | falsified, as expected |

Overall: **falsified**. Capacity is not the binding constraint.

## What the shape of the failure says

3.53x the parameters did not reach a better held-out loss. It reached essentially the
same floor — 7.3790 against 7.3333, within 0.6% — and reached it in 360 steps instead of
840, then overfitted from there. That is the signature of a problem that is limited by
the task rather than by the model: more capacity buys faster fitting of the same
ceiling, not a lower one.

The render check agrees and is the more important of the two, because it is powered. The
selection-scalar comparison measured the paired-difference interval's half-width at
0.0032 on this exact 128-icon draw, so an effect above roughly 0.0064 would have been
detected. The observed effect is +0.001853 with an interval spanning zero. A 3.53x model
does not render measurably closer to `x_0`.

The accuracy-versus-loss divergence is also sharper here than in v2: changed-token
recovery climbs from 0.0639 at the loss minimum to 0.0831 by step 840, a 30% relative
gain entirely on the far side of the point where held-out loss stopped improving.
Because the selection rule was settled beforehand, that number is reported and not
gated — which is exactly the post-hoc freedom the scalar comparison was run to remove.

## Where this leaves Gate G

Two of the three candidate factors are now eliminated on matched, predeclared
comparisons. Data volume is not the constraint: v2. Model capacity is not the
constraint: this run. The remaining candidate is the one the render probe pointed at —
the corruption schedule.

That probe showed `x_t` at corruption probability 0.35 is already visually destroyed: a
third of the geometry is simply gone, and no amount of model or data recovers
information that is not there. The probability has been fixed at 0.35 since the Gate F
four-icon fixtures, chosen for a four-icon experiment and never revisited at corpus
scale. The held-out loss floor near 7.35 that both v2 and v3 converge to looks like a
property of that regime rather than of either model.

The next experiment should vary it — ideally training across a range of corruption
levels rather than a single fixed point, which is also what a denoiser facing many
levels at sampling time would need.
