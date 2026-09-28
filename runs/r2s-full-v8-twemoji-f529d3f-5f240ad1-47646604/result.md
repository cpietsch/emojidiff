# r2s-full-v8-twemoji-f529d3f-5f240ad1-47646604

**Hypothesis.** Adding the Twemoji icons that fit the codec to v7's training data lowers held-out OpenMoji pixel error below v7's at the same budget, greedy decoding, with the paired 95% interval on the per-icon difference excluding zero. It also transcribes held-out validation renders better than retrieving the closest training icon. Pass requires all three: (1) mean 72 px pixel error below the nearest-training-icon baseline with the paired 95% interval on the reduction excluding zero; (2) CLIP top-1 on Gate N's 32 validation icons at or above OmniSVG 4B zero-shot, 0.609; (3) median single-icon latency on the RTX 4080 under 500 ms.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0911 [0.0851, 0.0972] |
| model pixel error, median | 0.0890 |
| exact program rate | 0.015 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | -0.0013 [-0.0062, 0.0035] |
| icons where model beats nearest icon | 190 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.531 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA GeForce RTX 4080 |
| parameters | 9,025,826 |
| train seconds | 5455 |
| peak VRAM GiB | 4.341529369354248 |
| latency ms/icon, batch 1, median | 1078.4 |
| latency ms/icon, batch 1, p95 | 1082.5 |
| throughput ms/icon, batch 64 | 79.6 |
| model calls per icon (decoding) | 540.0 |
| torch / CUDA | 2.14.0a0+4fdf77b940.nv26.08 / 13.4 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
