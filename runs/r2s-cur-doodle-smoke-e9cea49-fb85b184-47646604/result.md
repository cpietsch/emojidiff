# r2s-cur-doodle-smoke-e9cea49-fb85b184-47646604

**Hypothesis.** Fine-tuned for 30k steps on renders degraded to look hand-drawn (probability 0.9), the final model turns degraded validation renders into programs closer to the clean icon than the parent model does, and closer than the degraded input itself (what a faithful tracer returns), while its error on clean validation renders rises by at most 0.010. Scored by doodle_eval on the sets declared in state/CURRENT.md (2026-10-31 16:40): clean, the training kinds and the held-out kinds pencil, hatched, affine, thumb.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 16 (primary/validation) |
| model pixel error, mean [95% CI] | 0.1633 [0.1138, 0.2157] |
| model pixel error, median | 0.1273 |
| exact program rate | 0.000 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1829 |
| nearest training icon pixel error | 0.0775 [0.0569, 0.1018] |
| error reduction vs nearest icon | -0.0859 [-0.1470, -0.0305] |
| icons where model beats nearest icon | 5 |
| CLIP top-1, model greedy (4 Gate N icons) | 1.000 |
| CLIP top-1, nearest training icon | 0.250 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA GeForce RTX 4080 |
| parameters | 9,025,826 |
| train seconds | 12 |
| peak VRAM GiB | 4.341804504394531 |
| latency ms/icon, batch 1, median | 2457.6 |
| latency ms/icon, batch 1, p95 | 2674.7 |
| throughput ms/icon, batch 16 | 177.5 |
| model calls per icon (decoding) | 932.0 |
| torch / CUDA | 2.14.0a0+4fdf77b940.nv26.08 / 13.4 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
