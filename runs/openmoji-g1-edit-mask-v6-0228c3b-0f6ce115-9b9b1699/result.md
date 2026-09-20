# The edit mask works, the model collapses onto it, and Gate G has its answer

Every trained Gate G model scored below the identity policy — emit `x_t` unchanged —
which is legal under argmax decoding and scores 0.6506 aggregate held-out accuracy. The
objective was the suspected reason: each field is an independent softmax over a 289- or
417-way vocabulary, so "leave this one alone" costs as much as inventing a value, and
26,030 of 40,008 held-out fields are uncorrupted.

This run adds a per-field keep-or-change head and copies the input wherever the model
says keep, making identity reachable as predict-keep-everywhere. 1,552 parameters on top
of v5, +0.27%. Everything else is v5 exactly.

## Result

| model | aggregate | changed | retained |
| --- | ---: | ---: | ---: |
| v5 slot-bound | 0.3793 | **0.0787** | 0.5408 |
| **v6 edit-mask** | 0.6119 | 0.0132 | 0.9335 |
| identity | **0.6506** | 0.0000 | **1.0000** |

| criterion | threshold | observed | outcome |
| --- | --- | ---: | --- |
| beats_identity | > 0.6506 | 0.6119 | **falsified** |
| retains_what_identity_retains | >= 0.95 | 0.9335 | **falsified** |
| recovers_more_than_identity | > 0.0787 | 0.0132 | **falsified** |
| stops_damaging_light_corruption | recovery at 0.10 >= 0 | -0.1087 | **falsified** |
| structural_safety | locked-path exact, round-trips | both true | pass |
| reproducibility | identical rerun | identical | pass |

Overall: **falsified**, and the predeclared degenerate-collapse watch has fired. The
model predicts keep on 91.4% of held-out fields. It moved most of the way to the trivial
policy and stopped just short of it, arriving somewhere strictly worse than both v5 and
identity: it recovers six times fewer corrupted fields than v5 while still not keeping
as reliably as doing nothing.

The renders say the same thing. Mean `x_hat_0` error now tracks the input almost exactly
— 0.0580, 0.0895, 0.1362, 0.1767 against inputs of 0.0510, 0.0874, 0.1359, 0.1796 — and
recovery is near zero everywhere:

| corruption p | v2 | v5 | **v6** | identity |
| ---: | ---: | ---: | ---: | ---: |
| 0.05 | -4.6517 | -3.1470 | **-0.3473** | 0.0000 |
| 0.10 | -1.2024 | -0.7607 | **-0.1087** | 0.0000 |
| 0.20 | -0.1315 | -0.0253 | **-0.0101** | 0.0000 |
| 0.35 | +0.1800 | +0.2098 | **+0.0147** | 0.0000 |

The edit mask did what it was built to do: it stopped the model destroying its input.
It converted an actively harmful model into a nearly inert one.

## Why it collapsed: corrupted fields are not detectable

The keep head is a corrupted-field detector, and measuring it directly answers the
question this project has been circling since the corruption sweep.

| quantity | value |
| --- | ---: |
| fields predicted keep | 0.9143 |
| corrupted-field recall | **0.1058** (1,479 of 13,978) |
| corrupted-field precision | 0.4312 (1,479 of 3,430 flagged) |
| base rate of corrupted fields | 0.3494 |
| false-alarm rate on uncorrupted fields | 0.0750 |

The detector finds one corrupted field in ten, and when it does flag one it is right
43% of the time against a 34.9% base rate — a lift of 1.23x. It is barely better than
guessing.

That is not an architectural shortcoming, and it is why collapsing to keep is the
rational thing for this model to do. A coordinate resampled uniformly from a 289-value
legal vocabulary lands on a perfectly plausible coordinate. Telling it apart from a
legitimate one requires already knowing what the icon should look like — which is the
denoising problem itself. **Detection is not an easier sub-problem than denoising; it is
the same problem.** With 65% of fields uncorrupted and no reliable way to find the other
35%, predicting keep is the loss-minimising policy, and the model found it.

## What this closes

Seven candidate explanations have now been eliminated by predeclared comparison:

| candidate | eliminated by |
| --- | --- |
| too little data | v2: 10.5x data, changed recovery 1.307x against a 1.5x bar |
| too little capacity | v3: 3.53x parameters, same loss floor reached sooner |
| wrong corruption level | sweep: output near-independent of input, r = 0.91 |
| no noise-level information | v4: every figure moved, none enough to matter |
| encoder slot-blindness | v5: real defect, 768 parameters, still r = 0.8621 |
| objective cannot express identity | v6: it can now, and the model collapses onto it |
| — | detection is not separable from denoising |

The predeclared falsification meaning for this run said that if the model could not clear
the trivial policy with identity one prediction away, the formulation itself is in
question and it should be written up as a Gate G-level negative result rather than
patched. That is the situation, and this is that write-up.

**Factorized role-uniform categorical corruption at p=0.35 over a 289/417-value
coordinate vocabulary produces states from which the corruption is not identifiable, and
therefore not invertible, by a model of this class.** The project's working hypothesis
names *structure-aware* categorical corruption; Gate F selected the factorized process as
primary on four-icon fixtures where held-out recovery ran at 96–99% because the model had
memorized the fixtures. That selection does not survive contact with 2,681 icons, and the
Gate F comparison should be re-read as a memorization comparison rather than a denoising
one.

## Scope

Fixed-topology and geometry-only, single seed, one corruption draw per icon per level.
This is a negative result about one corruption process at one probability with one model
class. It does not show that the typed-SVG representation is unlearnable — Gate C and
Gate D stand, the codec is exact, and v5 showed the encoder fix was worth 768 parameters.
It shows that this corruption process does not define an invertible problem.
