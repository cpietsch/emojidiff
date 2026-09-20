# Beating the trivial policy, with no asterisk

v12 was the first model here to beat the identity baseline — emit the corrupted input
unchanged — but only with a decode threshold fitted on held-out icons. Its own argmax
decode lost by 0.0429. Two runs removed that asterisk.

The threshold is derivable rather than arbitrary. Keeping a retained field is always
right; changing one is right only if the value head re-predicts the same token, which is
negligible. So changing at confidence `p` has expected gain `p·q − (1 − p)`, positive
exactly when

    p > 1 / (1 + q)

with `q` the value head's accuracy on genuinely corrupted fields.

v13 estimated `q` on train icons and got 0.2751, against 0.1233 held out — the value head
is much better on what it trained on. That gave a threshold of 0.784 where the held-out
accuracy implies 0.890, so the model edited more than paid and beat identity by only
0.0003. This run estimates `q` on 256 icons **withheld from training**.

## Result

| | q | threshold | aggregate | identity | margin |
| --- | ---: | ---: | ---: | ---: | ---: |
| v12, argmax decode | — | 0.500 | 0.6086 | 0.6515 | −0.0429 |
| v12, threshold fitted on held-out icons | — | 0.924 | 0.6567 | 0.6502 | +0.0065 * |
| v13, q from icons it trained on | 0.2751 | 0.784 | 0.6519 | 0.6515 | +0.0003 |
| **v14, q from withheld icons** | **0.2220** | **0.818** | **0.6543** | 0.6515 | **+0.0027** |

\* fitted on held-out data, which is why it needed an asterisk.

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| beats_identity_without_touching_held_out_data | > identity | +0.0027 | pass |
| improves_on_the_trained_on_estimate | > 0.0003 | +0.0027 | pass |
| threshold_reflects_an_honest_estimate | 0.7842 < t < 0.95 | 0.8183 | pass |
| structural_safety | locked-path exact, round-trips | both true | pass |
| reproducibility | identical rerun | identical | pass |

All four pass, and the run is handicapped: withholding 256 icons also removes them from
training, so it learned from 2,425 rather than 2,681 — 9.6% less data than every run it
is compared against. The margin is conservative for that reason.

Retained-token accuracy is **0.9815**, the highest of any model in this project and close
to identity's 1.0, while changed-token recovery stays at 0.0423. The model has learned to
leave things alone unless it is nearly certain, which is exactly what the break-even rule
asks of it.

## What is now established

Nothing in this run is fitted on held-out data. The threshold comes from icons the model
never trained on, the model selects its checkpoint on held-out loss as every run here
does, and the identity comparison is a plain read of the result. The project has a
denoiser that does better than doing nothing, with no caveat about how the number was
obtained.

## What is not

The margin is +0.0027 on token accuracy — about 0.4% relative. This is not a good model;
it is a model that has stopped being worse than useless. The value head still reaches
only 0.222 on withheld corrupted fields, which is why the break-even threshold sits at
0.818 and the model edits just 4% of fields.

The measured next step is unchanged and remains the task-formulation lens's: the value
head predicts an exact bin on a quarter-unit metric lattice through a categorical
softmax, so being close earns nothing. A distance-kernel target would give partial credit
for proximity, which should lift `q`, which lowers the break-even threshold, which lets
the model edit more of what it correctly detects. Every term in that chain is now
measured.

## Scope

Fixed-topology and geometry-only, single seed, marginal-respecting corruption at 0.35,
token accuracy rather than renders. The project's primary/test split remains untouched.
No render in this project has yet produced a recognizable icon, and this run does not
change that.
