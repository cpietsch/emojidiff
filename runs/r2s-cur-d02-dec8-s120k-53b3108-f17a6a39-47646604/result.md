# r2s-cur-d02-dec8-s120k-53b3108-f17a6a39-47646604

**Hypothesis.** With dropout 0.2 and 120k steps, 8 decoder layers transcribe validation renders better than v9. Pass requires the paired 95% bootstrap interval on the per-icon reduction in mean 72 px pixel error on the 339 validation renders to exclude zero against each normal v9 run (RTX 4080 original, H100 rerun, B300 seeds 2 and 3). Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0706 [0.0657, 0.0756] |
| model pixel error, median | 0.0647 |
| exact program rate | 0.006 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0191 [0.0152, 0.0230] |
| icons where model beats nearest icon | 260 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.812 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 11,132,706 |
| train seconds | 5830 |
| peak VRAM GiB | 5.548205375671387 |
| latency ms/icon, batch 1, median | 3957.2 |
| latency ms/icon, batch 1, p95 | 3964.6 |
| throughput ms/icon, batch 64 | 128.2 |
| model calls per icon (decoding) | 686.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
