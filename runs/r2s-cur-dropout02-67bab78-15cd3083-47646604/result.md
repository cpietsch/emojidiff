# r2s-cur-dropout02-67bab78-15cd3083-47646604

**Hypothesis.** With dropout 0.2 instead of 0.1, and everything else as in v9, the render-to-program model transcribes the 339 validation renders with lower mean 72 px pixel error than v9 trained on the same machine and seed. Pass requires the paired 95% bootstrap interval on the per-icon reduction against that v9 rerun to exclude zero. Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0696 [0.0649, 0.0746] |
| model pixel error, median | 0.0658 |
| exact program rate | 0.029 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0201 [0.0160, 0.0241] |
| icons where model beats nearest icon | 246 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.625 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 5855 |
| peak VRAM GiB | 4.388385772705078 |
| latency ms/icon, batch 1, median | 2523.4 |
| latency ms/icon, batch 1, p95 | 2582.0 |
| throughput ms/icon, batch 64 | 109.7 |
| model calls per icon (decoding) | 493.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
