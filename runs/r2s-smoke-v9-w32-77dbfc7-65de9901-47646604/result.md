# r2s-smoke-v9-w32-77dbfc7-65de9901-47646604

**Hypothesis.** Trained on the 2,681 family-disjoint training icons with metric coordinates, path-major order, online exact augmentation and compositions, the 9.0M-parameter render-to-program model transcribes held-out validation renders better than retrieving the closest training icon. Pass requires all three: (1) mean 72 px pixel error below the nearest-training-icon baseline with the paired 95% interval on the reduction excluding zero; (2) CLIP top-1 on Gate N's 32 validation icons at or above OmniSVG 4B zero-shot, 0.609; (3) median single-icon latency on the RTX 4080 under 500 ms.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 32 (primary/validation) |
| model pixel error, mean [95% CI] | 0.1632 [0.1392, 0.1896] |
| model pixel error, median | 0.1241 |
| exact program rate | 0.000 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1627 |
| nearest training icon pixel error | 0.0943 [0.0816, 0.1074] |
| error reduction vs nearest icon | -0.0688 [-0.0959, -0.0440] |
| icons where model beats nearest icon | 0 |
| CLIP top-1, model greedy (8 Gate N icons) | 0.375 |
| CLIP top-1, nearest training icon | 0.375 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA H100 |
| parameters | 9,025,826 |
| train seconds | 127 |
| peak VRAM GiB | 4.38838529586792 |
| latency ms/icon, batch 1, median | 2019.1 |
| latency ms/icon, batch 1, p95 | 2024.0 |
| throughput ms/icon, batch 32 | 156.5 |
| model calls per icon (decoding) | 499.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
