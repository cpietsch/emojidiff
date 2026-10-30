# r2s-cur-d02-seed2-508979a-8a9c9811-47646604

**Hypothesis.** With dropout 0.2 (a second seed), the model transcribes validation renders better than v9. Pass requires the paired 95% bootstrap interval on the per-icon reduction in mean 72 px pixel error on the 339 validation renders to exclude zero against each v9 run of the same protocol (seeds 7001-7003 on the B300, the H100 rerun, the RTX 4080 original). Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0690 [0.0643, 0.0739] |
| model pixel error, median | 0.0655 |
| exact program rate | 0.006 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0207 [0.0169, 0.0244] |
| icons where model beats nearest icon | 258 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.656 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 4801 |
| peak VRAM GiB | 4.388385772705078 |
| latency ms/icon, batch 1, median | 3158.8 |
| latency ms/icon, batch 1, p95 | 3215.5 |
| throughput ms/icon, batch 64 | 123.4 |
| model calls per icon (decoding) | 626.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
