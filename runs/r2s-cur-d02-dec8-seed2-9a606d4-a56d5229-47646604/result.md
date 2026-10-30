# r2s-cur-d02-dec8-seed2-9a606d4-a56d5229-47646604

**Hypothesis.** Dropout 0.2 with 8 decoder layers (seed 2) transcribes validation renders better than v9. Pass requires the paired 95% bootstrap interval on the per-icon reduction in mean 72 px pixel error on the 339 validation renders to exclude zero against each normal v9 run (RTX 4080 original, H100 rerun, B300 seeds 2 and 3) and, for depth arms, a direction consistent across seeds against dropout 0.2 at 6 layers. Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0720 [0.0670, 0.0772] |
| model pixel error, median | 0.0659 |
| exact program rate | 0.018 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0177 [0.0135, 0.0218] |
| icons where model beats nearest icon | 224 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.750 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 11,132,706 |
| train seconds | 3170 |
| peak VRAM GiB | 5.548205375671387 |
| latency ms/icon, batch 1, median | 4305.1 |
| latency ms/icon, batch 1, p95 | 4355.3 |
| throughput ms/icon, batch 64 | 122.4 |
| model calls per icon (decoding) | 752.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
