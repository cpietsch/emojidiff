# vt-v1-74dfbd2-46380052-47646604

**Hypothesis.** A variational transcriber - v9 whose 18 x 18 stem grid passes through a per-cell Gaussian latent (c = 8 channels) held at a KL budget of C = 2,000 nats per icon - keeps v9's persistent spatial record of what is still undrawn and so reconstructs held-out icons from its posterior mean close to v9, while its latent is smooth at the posterior scale. Pass, on the first 339 validation icons, IEEE float32 (TF32 off) greedy decoding, 72 px pixel error, bootstrap intervals, all comparisons paired, requires A1 and A3; A2 is a collapse check. (A1) reconstruction from mu is non-inferior to v9 re-decoded in float32 on the same icons - the upper 95% bound of (VT - v9) is at most +0.015 - and beats the nearest training render or its mirror (every model trains with mirrors), the 95% interval on the reduction excluding zero. (A2) held-out KL within [0.8C, 1.2C] = [1,600, 2,400] nats per icon with beta above its floor of 1e-4; the controller enforces this band, so it detects collapse and is not evidence of anything else. (A3) decoding one posterior sample instead of mu costs at most 0.010 pixel error, paired mean. Reported without a criterion: the z = 0 decode, N(0, I) samples as the prior-hole control (ink coverage, distinct, rendered), 8 lerp strips of posterior means (a canvas dissolve), graph-decoder latency. Decision: A1 and A3 pass - fit the flow prior (arm 3, lfp-v1). A1 fails - run once more at c = 16 under a new run identity; a second failure stops the canvas line.

## Criteria

| criterion | measure | pass |
| --- | --- | --- |
| A1 | VT - v9 0.0093 [0.0052, 0.0133]; reduction vs nearest 0.0019 [-0.0024, 0.0061] | False |
| A2 | held-out KL 1943.6445 [1901.2430, 1987.5275] nats; beta at floor False | True |
| A3 | sample - mu 0.0003 [-0.0034, 0.0041] | True |

## Result (339 validation icons, IEEE float32 greedy decoding (TF32 off: matmul and cuDNN); pixel metrics at 72 px)

| measure | value |
| --- | --- |
| VT from mu, pixel error | 0.0829 [0.0780, 0.0880] |
| VT from one posterior sample | 0.0832 [0.0778, 0.0889] |
| v9 (parent), same icons, float32 | 0.0737 [0.0689, 0.0785] |
| nearest training render or mirror | 0.0849 [0.0795, 0.0902] |
| nearest training render, no mirrors | 0.0897 [0.0848, 0.0947] |
| z = 0 decode (control) | 0.1720 [0.1650, 0.1794] |
| blank canvas | 0.1720 |
| icons VT beats nearest / beats v9 | 181 / 126 |
| exact reconstructions | 0.009 |
| held-out KL per dimension | 0.750 |
| N(0, I) samples distinct / rendered | 64 / 64 of 64 |
| N(0, I) samples with ink < 0.10 | 0.328 (validation icons 0.021) |
| selected step | 12000 (fallback: False) |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA GeForce RTX 4080 |
| parameters | 9,032,242 |
| train seconds | 3418 |
| peak VRAM GiB | 4.298437595367432 |
| graph decode ms per icon, batch 1, IEEE float32 | 1206.6 |
| torch / CUDA | 2.14.0a0+4fdf77b940.nv26.08 / 13.4 |

Sheets: `reconstructions.png` (reference, VT from mu, VT from a posterior sample,
v9, nearest training render or mirror), `interpolations.png` (lerp of posterior
means, a canvas dissolve), `prior-samples.png` (N(0, I), the prior-hole control).
