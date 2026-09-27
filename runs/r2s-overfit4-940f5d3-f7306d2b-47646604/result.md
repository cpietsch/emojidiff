# r2s-overfit4-940f5d3-f7306d2b-47646604

**Hypothesis.** An 8.9M-parameter render-to-program encoder-decoder, trained on four OpenMoji icons, decodes those four icons greedily from their 144 px renders to programs whose 72 px renders are within 0.01 mean RGB error of the originals, in under 1,000 steps.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 4 (primary/train) |
| model pixel error, mean [95% CI] | 0.0000 [0.0000, 0.0000] |
| model pixel error, median | 0.0000 |
| exact program rate | 1.000 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1657 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA GeForce RTX 4080 |
| parameters | 8,921,634 |
| train seconds | 14 |
| peak VRAM GiB | 0.6286492347717285 |
| latency ms/icon, batch 1, median | 147.9 |
| latency ms/icon, batch 1, p95 | 149.0 |
| throughput ms/icon, batch 4 | 56.0 |
| model calls per icon (decoding) | 97.0 |
| torch / CUDA | 2.14.0a0+4fdf77b940.nv26.08 / 13.4 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
