# r2s-smoke-v9-w16-a4d3d37-30dc1c5d-47646604

**Hypothesis.** Trained on the 2,681 family-disjoint training icons with metric coordinates, path-major order, online exact augmentation and compositions, the 9.0M-parameter render-to-program model transcribes held-out validation renders better than retrieving the closest training icon. Pass requires all three: (1) mean 72 px pixel error below the nearest-training-icon baseline with the paired 95% interval on the reduction excluding zero; (2) CLIP top-1 on Gate N's 32 validation icons at or above OmniSVG 4B zero-shot, 0.609; (3) median single-icon latency on the RTX 4080 under 500 ms.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 32 (primary/validation) |
| model pixel error, mean [95% CI] | 1.0000 [1.0000, 1.0000] |
| model pixel error, median | 1.0000 |
| exact program rate | 0.000 |
| rendered rate | 0.000 |
| blank canvas pixel error | 0.1627 |
| nearest training icon pixel error | 0.0943 [0.0816, 0.1074] |
| error reduction vs nearest icon | -0.9057 [-0.9184, -0.8926] |
| icons where model beats nearest icon | 0 |
| CLIP top-1, model greedy (8 Gate N icons) | 0.000 |
| CLIP top-1, nearest training icon | 0.375 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 138 |
| peak VRAM GiB | 4.38838529586792 |
| latency ms/icon, batch 1, median | 3092.3 |
| latency ms/icon, batch 1, p95 | 3099.5 |
| throughput ms/icon, batch 32 | 209.2 |
| model calls per icon (decoding) | 627.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.

## Recorded as failed (2026-10-30)

Curiosity cluster (B300) environment smoke, Slurm job 13937. Training matched the RTX 4080
reference (`r2s-smoke-v9-w16-77dbfc7-30dc1c5d-47646604`): loss 2.6940 against 2.6947 and
free-token accuracy 0.3507 against 0.3504 at step 1,200. Every evaluation render failed
(pixel error 1.0, rendered rate 0): the isolated renderer starts `python -I`, which ignores
`PYTHONPATH`, and this checkout was not installed in its environment. The scores above are
therefore not results. Fixed in 67bab78 (one environment per checkout, project installed).
