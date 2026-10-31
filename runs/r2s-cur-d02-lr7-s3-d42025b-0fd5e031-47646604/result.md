# r2s-cur-d02-lr7-s3-d42025b-0fd5e031-47646604

**Hypothesis.** With a learning rate of 7e-4 instead of 5e-4 and everything else as dropout 0.2 at 60k steps, the model transcribes the 339 validation renders better than the dropout-0.2 run of the same seed. Judged over seeds 7001-7008: the mean seed-matched difference with a bootstrap interval over seeds excluding zero.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0673 [0.0622, 0.0728] |
| model pixel error, median | 0.0589 |
| exact program rate | 0.041 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0224 [0.0180, 0.0265] |
| icons where model beats nearest icon | 265 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.625 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 2598 |
| peak VRAM GiB | 4.388660430908203 |
| latency ms/icon, batch 1, median | 3250.1 |
| latency ms/icon, batch 1, p95 | 3251.5 |
| throughput ms/icon, batch 64 | 118.4 |
| model calls per icon (decoding) | 657.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
