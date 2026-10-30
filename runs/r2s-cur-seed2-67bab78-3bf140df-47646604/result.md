# r2s-cur-seed2-67bab78-3bf140df-47646604

**Hypothesis.** With a second training and augmentation seed, and everything else as in v9, the render-to-program model transcribes the 339 validation renders with lower mean 72 px pixel error than v9 trained on the same machine and seed. Pass requires the paired 95% bootstrap interval on the per-icon reduction against that v9 rerun to exclude zero. Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0728 [0.0676, 0.0780] |
| model pixel error, median | 0.0709 |
| exact program rate | 0.041 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0169 [0.0126, 0.0212] |
| icons where model beats nearest icon | 241 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.750 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 5895 |
| peak VRAM GiB | 4.388385772705078 |
| latency ms/icon, batch 1, median | 2551.6 |
| latency ms/icon, batch 1, p95 | 2596.2 |
| throughput ms/icon, batch 64 | 82.5 |
| model calls per icon (decoding) | 831.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
