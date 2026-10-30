# r2s-cur-d02-w384-508979a-1a8a1eac-47646604

**Hypothesis.** With dropout 0.2 holding back overfitting, width 384 transcribes validation renders better than v9. Pass requires the paired 95% bootstrap interval on the per-icon reduction in mean 72 px pixel error on the 339 validation renders to exclude zero against each v9 run of the same protocol (seeds 7001-7003 on the B300, the H100 rerun, the RTX 4080 original). Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0704 [0.0654, 0.0755] |
| model pixel error, median | 0.0678 |
| exact program rate | 0.050 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0194 [0.0153, 0.0233] |
| icons where model beats nearest icon | 245 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.562 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 19,762,402 |
| train seconds | 4823 |
| peak VRAM GiB | 6.41351842880249 |
| latency ms/icon, batch 1, median | 2595.1 |
| latency ms/icon, batch 1, p95 | 4752.6 |
| throughput ms/icon, batch 64 | 82.6 |
| model calls per icon (decoding) | 898.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
