# r2s-cur-d02-lr3-s1-8e367a7-d88d8f3a-47646604

**Hypothesis.** With a learning rate of 3e-4 instead of 5e-4 and everything else as dropout 0.2 at 60k steps, the model transcribes the 339 validation renders better than the dropout-0.2 run of the same seed. Judged over seeds 7001-7008: the mean seed-matched difference with a bootstrap interval over seeds excluding zero.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0749 [0.0697, 0.0803] |
| model pixel error, median | 0.0694 |
| exact program rate | 0.012 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0148 [0.0106, 0.0190] |
| icons where model beats nearest icon | 245 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.531 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 2630 |
| peak VRAM GiB | 4.388660430908203 |
| latency ms/icon, batch 1, median | 2122.2 |
| latency ms/icon, batch 1, p95 | 2141.6 |
| throughput ms/icon, batch 64 | 73.8 |
| model calls per icon (decoding) | 743.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
