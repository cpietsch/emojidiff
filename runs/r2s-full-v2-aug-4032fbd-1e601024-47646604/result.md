# r2s-full-v2-aug-4032fbd-1e601024-47646604

**Hypothesis.** Trained on the 2,681 family-disjoint training icons, the 8.9M-parameter render-to-program model transcribes held-out validation renders better than retrieving the closest training icon. Pass requires all three: (1) mean 72 px pixel error below the nearest-training-icon baseline with the paired 95% interval on the reduction excluding zero; (2) CLIP top-1 on Gate N's 32 validation icons at or above OmniSVG 4B zero-shot, 0.609; (3) median single-icon latency on the RTX 4080 under 500 ms.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.1301 [0.1248, 0.1360] |
| model pixel error, median | 0.1273 |
| exact program rate | 0.000 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | -0.0404 [-0.0448, -0.0361] |
| icons where model beats nearest icon | 50 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.156 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA GeForce RTX 4080 |
| parameters | 8,921,634 |
| train seconds | 1654 |
| peak VRAM GiB | 4.089363098144531 |
| latency ms/icon, batch 1, median | 828.0 |
| latency ms/icon, batch 1, p95 | 829.6 |
| throughput ms/icon, batch 64 | 62.3 |
| model calls per icon (decoding) | 641.0 |
| torch / CUDA | 2.14.0a0+4fdf77b940.nv26.08 / 13.4 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
