# Gate G dominant-bucket training run v1

Status: staged; training run not yet started.

First Gate G run whose outcome is a learning claim rather than pipeline liveness. It
trains the selected 577,552-parameter geometry denoiser for 600 bounded steps on a
256-icon family-disjoint subsample of the exact P32/T128 bucket and evaluates on 128
held-out icons from the disjoint validation split, tracing held-out recovery every 60
steps against an untrained step-0 control on the identical corruption draw.

The predeclared pass/fail criteria are recorded in `run.yaml` before launch. They are
deliberately falsifiable: the retained-token bar in particular is a real bet, because a
60-step CPU preflight reached only 0.0934 retained accuracy.

## Result

Completed on the owned RTX 4080 in about 22 seconds including stage verification. Two
complete invocations produced identical checkpoint and summary digests.

| step | held-out aggregate | changed | retained | held-out loss |
| ---: | ---: | ---: | ---: | ---: |
| 0 (untrained) | 0.0037 | 0.0030 | 0.0041 | 15.3908 |
| 60 | 0.0684 | 0.0219 | 0.0934 | 10.7155 |
| 120 | 0.1453 | 0.0346 | 0.2048 | 9.9711 |
| 180 | 0.1893 | 0.0420 | 0.2685 | 9.7997 |
| 240 | 0.2090 | 0.0439 | 0.2977 | **9.7932** |
| 300 | 0.2186 | 0.0458 | 0.3114 | 10.1371 |
| 360 | 0.2216 | 0.0465 | 0.3157 | 10.2798 |
| 420 | 0.2269 | 0.0499 | 0.3219 | 10.5504 |
| 480 | 0.2311 | 0.0530 | 0.3267 | 10.7512 |
| 540 | 0.2308 | 0.0559 | 0.3247 | 11.1266 |
| 600 | 0.2311 | 0.0552 | 0.3256 | 11.1485 |

Held-out totals are 13,978 changed and 26,030 retained fields.

## Predeclared criteria

| criterion | threshold | observed | outcome |
| --- | --- | --- | --- |
| recovery above control | >= 10x | 18.38x | pass |
| retained preservation | >= 0.90 | 0.3256 | **fail** |
| monotone learning | >= 8/10 | 9/10 | pass |
| structural safety | locked + round trip | both true | pass |
| reproducibility | identical rerun | identical | pass |

**Overall: falsified.** The run is retained as a negative result rather than retuned.

## Interpretation

The representation is clearly learnable: held-out changed-token recovery reaches 18.4x
its own untrained same-input control, and the improvement is monotone. That is a real
result at 256 diverse icons across 10 groups and 75 subgroups, far beyond the 4-icon
Gate F fixtures.

But the run also falsifies the retained-preservation hypothesis at this data scale, and
the trace shows why. Held-out loss reaches its minimum of 9.7932 at step 240 and then
rises steadily to 11.1485 while training token accuracy climbs to 0.4550 and training
loss falls to 2.7221. Held-out aggregate accuracy is flat at about 0.231 from step 240
onward. This is overfitting to 256 icons, not an optimization failure.

One nuance worth recording rather than smoothing over: held-out changed accuracy keeps
creeping up (0.0439 to 0.0552) across the same interval in which held-out loss worsens.
The monotone criterion therefore passed while the model was already generalizing worse
overall, so that criterion is weaker evidence than it looks in isolation. Aggregate
held-out loss is the more honest scalar here.

The 0.90 retained bar was a deliberate bet and it lost by a wide margin. Nothing about
this run suggests the bar itself was wrong for a usable denoiser; it suggests 256 icons
is too little data for this model to reach it.

## Next experiment this justifies

More data, not more steps. The dominant bucket has a 2,681-icon family-disjoint train
split, ten times what this run used, and loading it costs about three minutes of CPU at
the measured 60 ms per icon. The controlled next run should change only the train-split
size, hold the model, seed, corruption, batch, and learning rate fixed, and use an
early-stopping or best-checkpoint policy keyed on held-out loss rather than a fixed
600-step budget.
