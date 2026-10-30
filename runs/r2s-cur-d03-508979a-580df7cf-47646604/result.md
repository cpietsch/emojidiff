# r2s-cur-d03-508979a-580df7cf-47646604

**Hypothesis.** With dropout 0.3, more regularisation than 0.2, the model transcribes validation renders better than v9. Pass requires the paired 95% bootstrap interval on the per-icon reduction in mean 72 px pixel error on the 339 validation renders to exclude zero against each v9 run of the same protocol (seeds 7001-7003 on the B300, the H100 rerun, the RTX 4080 original). Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0790 [0.0738, 0.0845] |
| model pixel error, median | 0.0759 |
| exact program rate | 0.012 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0107 [0.0058, 0.0155] |
| icons where model beats nearest icon | 222 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.531 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 4794 |
| peak VRAM GiB | 4.388385772705078 |
| latency ms/icon, batch 1, median | 2380.2 |
| latency ms/icon, batch 1, p95 | 2386.3 |
| throughput ms/icon, batch 64 | 100.7 |
| model calls per icon (decoding) | 447.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
