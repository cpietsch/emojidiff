# r2s-cur-d025-s120k-seed4-8775698-9fdb5d80-47646604

**Hypothesis.** Dropout 0.25 trained for 120k steps (seed 4) transcribes validation renders better than v9. Pass requires the paired 95% bootstrap interval on the per-icon reduction in mean 72 px pixel error on the 339 validation renders to exclude zero against each normal v9 run (RTX 4080 original, H100 rerun, B300 seeds 2 and 3). Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0740 [0.0688, 0.0795] |
| model pixel error, median | 0.0705 |
| exact program rate | 0.024 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0158 [0.0117, 0.0197] |
| icons where model beats nearest icon | 233 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.531 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 5218 |
| peak VRAM GiB | 4.388660430908203 |
| latency ms/icon, batch 1, median | 1107.7 |
| latency ms/icon, batch 1, p95 | 1108.8 |
| throughput ms/icon, batch 64 | 64.5 |
| model calls per icon (decoding) | 375.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
