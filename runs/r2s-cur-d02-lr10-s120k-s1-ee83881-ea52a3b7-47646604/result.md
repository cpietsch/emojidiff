# r2s-cur-d02-lr10-s120k-s1-ee83881-ea52a3b7-47646604

**Hypothesis.** Dropout 0.2 for 120k steps with learning rate 0.001 instead of 5e-4 (seed 7001) transcribes the 339 validation renders better than r2s-cur-d02-s120k of the same seed. Judged over seeds 7001-7008: the mean seed-matched difference with a bootstrap interval over seeds excluding zero. Final-model rule (state/CURRENT.md, declared 2026-10-31 11:50): if it passes, the run with the lowest float32 greedy validation error is the candidate; one paired test look against the current final model decides, its interval reported either way.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0694 [0.0647, 0.0743] |
| model pixel error, median | 0.0664 |
| exact program rate | 0.029 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0203 [0.0162, 0.0244] |
| icons where model beats nearest icon | 245 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.594 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 5259 |
| peak VRAM GiB | 4.388660430908203 |
| latency ms/icon, batch 1, median | 3954.4 |
| latency ms/icon, batch 1, p95 | 3963.9 |
| throughput ms/icon, batch 64 | 101.4 |
| model calls per icon (decoding) | 806.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
