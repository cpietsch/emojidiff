# Noise-level conditioning is not the missing piece

The corruption sweep found the Gate G denoiser's output near-independent of its input.
The obvious informational explanation was that it trained at one fixed corruption level
and was never told the level, so it had no reason to learn how much to trust `x_t`. This
run supplies both halves of that fix: a corruption level sampled per example over 0.05
to 0.50, and 16 sinusoidal conditioning features telling the model the level.

Model size is v2's plus 3,168 parameters of noise projection. Selection and the headline
held-out numbers stay at corruption 0.35, exactly the task v2 and v3 were measured on.
This is a deliberate two-factor change — sampling without telling the model would leave
it unable to use the level, telling it without varying it would leave it a constant —
and is recorded as such rather than presented as one factor.

## Result

Early stopping ended the run at step 1,500; the selected checkpoint is step 1,020. A
second complete invocation returned an identical JSON result.

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| stops_destroying_light_corruption | recovery at 0.10 above 0 | -1.0385 | **falsified** |
| output_depends_on_input | Pearson r below 0.70 | 0.8619 | **falsified** |
| light_corruption_damage_collapses | recovery at 0.05 above -0.5 | -4.5349 | **falsified** |
| does_not_wreck_the_trained_task | held-out loss below 8.0 | 7.4285 | pass |
| structural_safety | locked-path exact, round-trips | both true | pass |
| reproducibility | identical rerun | identical | pass |

Against v2 on the identical 32-icon sweep draw:

| corruption p | v2 recovery | v4 recovery |
| ---: | ---: | ---: |
| 0.05 | -4.6517 | -4.5349 |
| 0.10 | -1.2024 | -1.0385 |
| 0.20 | -0.1315 | -0.0910 |
| 0.35 | +0.1800 | +0.2083 |
| Pearson r, 0.05 vs 0.35 | 0.9122 | 0.8619 |

Every number moves in the predicted direction and none moves enough to matter. The model
still destroys lightly corrupted inputs and still emits nearly the same picture whatever
it is given.

Overall: **falsified**. Noise-level information was not the missing piece.

## What it ruled out, and what that forced

This eliminated the informational explanation, which is what made it worth running: the
model was not failing for want of knowing how corrupted its input was. That pushed the
search to the encoder, where the actual defect turned out to be — see
`openmoji-g1-slot-bound-v5-f502df1-d8af55ea-9b9b1699`. The encoder summed six coordinate
lookups from one shared embedding table into a single vector per segment, so a segment
was an unordered bag of its values and the model provably could not tell which
coordinate held which. It was never ignoring `x_t`; it was reading a scrambled copy.

This run's negative result is what made that search necessary, and its 0.4282 retained
accuracy — barely above v2's 0.4128 despite the extra conditioning — is consistent with
a model whose input was scrambled before the conditioning could help.
