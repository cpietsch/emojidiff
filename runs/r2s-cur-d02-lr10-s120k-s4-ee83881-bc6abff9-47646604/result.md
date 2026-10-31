# r2s-cur-d02-lr10-s120k-s4-ee83881-bc6abff9-47646604

**Hypothesis.** Dropout 0.2 for 120k steps with learning rate 0.001 instead of 5e-4 (seed 7004) transcribes the 339 validation renders better than r2s-cur-d02-s120k of the same seed. Judged over seeds 7001-7008: the mean seed-matched difference with a bootstrap interval over seeds excluding zero. Final-model rule (state/CURRENT.md, declared 2026-10-31 11:50): if it passes, the run with the lowest float32 greedy validation error is the candidate; one paired test look against the current final model decides, its interval reported either way.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0742 [0.0692, 0.0795] |
| model pixel error, median | 0.0728 |
| exact program rate | 0.047 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0155 [0.0111, 0.0199] |
| icons where model beats nearest icon | 232 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.656 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 5602 |
| peak VRAM GiB | 4.388660430908203 |
| latency ms/icon, batch 1, median | 2277.1 |
| latency ms/icon, batch 1, p95 | 2279.0 |
| throughput ms/icon, batch 64 | 115.0 |
| model calls per icon (decoding) | 456.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
