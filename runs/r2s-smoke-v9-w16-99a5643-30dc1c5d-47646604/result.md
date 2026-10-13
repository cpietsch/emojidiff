# r2s-smoke-v9-w16-99a5643-30dc1c5d-47646604

**Hypothesis.** Trained on the 2,681 family-disjoint training icons with metric coordinates, path-major order, online exact augmentation and compositions, the 9.0M-parameter render-to-program model transcribes held-out validation renders better than retrieving the closest training icon. Pass requires all three: (1) mean 72 px pixel error below the nearest-training-icon baseline with the paired 95% interval on the reduction excluding zero; (2) CLIP top-1 on Gate N's 32 validation icons at or above OmniSVG 4B zero-shot, 0.609; (3) median single-icon latency on the RTX 4080 under 500 ms.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 32 (primary/validation) |
| model pixel error, mean [95% CI] | 0.1632 [0.1382, 0.1909] |
| model pixel error, median | 0.1210 |
| exact program rate | 0.000 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1627 |
| nearest training icon pixel error | 0.0943 [0.0816, 0.1074] |
| error reduction vs nearest icon | -0.0689 [-0.0977, -0.0425] |
| icons where model beats nearest icon | 5 |
| CLIP top-1, model greedy (8 Gate N icons) | 0.000 |
| CLIP top-1, nearest training icon | 0.375 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA GeForce RTX 4080 |
| parameters | 9,025,826 |
| train seconds | 115 |
| peak VRAM GiB | 4.341529369354248 |
| latency ms/icon, batch 1, median | 1767.7 |
| latency ms/icon, batch 1, p95 | 1772.8 |
| throughput ms/icon, batch 32 | 109.7 |
| model calls per icon (decoding) | 918.0 |
| torch / CUDA | 2.14.0a0+4fdf77b940.nv26.08 / 13.4 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
