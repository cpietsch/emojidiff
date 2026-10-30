# vt-v3-d02-s3-d05a169-0f827702-47646604

**Hypothesis.** A variational transcriber - the new final transcriber whose 18 x 18 stem grid passes through a per-cell Gaussian latent (c = 16 channels, chosen by the pre-flight) held at a KL budget of C = 2,000 nats per icon - reconstructs held-out icons from its posterior mean close to its parent while its latent is smooth at the posterior scale. Pass, on the first 339 validation icons, IEEE float32 (TF32 off) greedy decoding, 72 px pixel error, bootstrap intervals, all comparisons paired, requires A1 and A3; A2 is a collapse check. (A1) reconstruction from mu is non-inferior to the parent re-decoded in float32 on the same icons - the upper 95% bound of (VT - parent) is at most +0.015 - and beats the nearest training render or its mirror, the 95% interval on the reduction excluding zero. (A2) held-out KL within [0.8C, 1.2C] with beta above its floor. (A3) decoding one posterior sample instead of mu costs at most 0.010 pixel error, paired mean. Note: the parent was chosen as the best of four seeds on these validation icons, so its score is optimistic and non-inferiority is the harder half. Decision for the line (three seeds): at least two runs pass A1 and A3 - fit the flow prior; otherwise stop.

## Criteria

| criterion | measure | pass |
| --- | --- | --- |
| A1 | VT - v9 0.0117 [0.0081, 0.0154]; reduction vs nearest 0.0089 [0.0047, 0.0130] | False |
| A2 | held-out KL 1815.4443 [1768.0638, 1863.8206] nats; beta at floor False | True |
| A3 | sample - mu 0.0005 [-0.0025, 0.0034] | True |

## Result (339 validation icons, IEEE float32 greedy decoding (TF32 off: matmul and cuDNN); pixel metrics at 72 px)

| measure | value |
| --- | --- |
| VT from mu, pixel error | 0.0760 [0.0706, 0.0815] |
| VT from one posterior sample | 0.0765 [0.0711, 0.0819] |
| v9 (parent), same icons, float32 | 0.0642 [0.0595, 0.0692] |
| nearest training render or mirror | 0.0849 [0.0795, 0.0902] |
| nearest training render, no mirrors | 0.0897 [0.0848, 0.0947] |
| z = 0 decode (control) | 0.1720 [0.1650, 0.1794] |
| blank canvas | 0.1720 |
| icons VT beats nearest / beats v9 | 242 / 117 |
| exact reconstructions | 0.035 |
| held-out KL per dimension | 0.350 |
| N(0, I) samples distinct / rendered | 64 / 64 of 64 |
| N(0, I) samples with ink < 0.10 | 0.156 (validation icons 0.021) |
| selected step | 12000 (fallback: False) |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,038,402 |
| train seconds | 1544 |
| peak VRAM GiB | 4.3444905281066895 |
| graph decode ms per icon, batch 1, IEEE float32 | 1241.3 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Sheets: `reconstructions.png` (reference, VT from mu, VT from a posterior sample,
v9, nearest training render or mirror), `interpolations.png` (lerp of posterior
means, a canvas dissolve), `prior-samples.png` (N(0, I), the prior-hole control).
