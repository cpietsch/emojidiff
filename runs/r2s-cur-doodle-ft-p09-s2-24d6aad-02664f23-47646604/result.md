# r2s-cur-doodle-ft-p09-s2-24d6aad-02664f23-47646604

**Hypothesis.** Fine-tuned for 30k steps on renders degraded to look hand-drawn (probability 0.9), the final model turns degraded validation renders into programs closer to the clean icon than the parent model does, and closer than the degraded input itself (what a faithful tracer returns), while its error on clean validation renders rises by at most 0.010. Scored by doodle_eval on the sets declared in state/CURRENT.md (2026-10-31 16:00): clean, the training kinds and the held-out kinds pencil, hatched, affine, thumb.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0916 [0.0862, 0.0973] |
| model pixel error, median | 0.0833 |
| exact program rate | 0.006 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | -0.0019 [-0.0063, 0.0022] |
| icons where model beats nearest icon | 174 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.656 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 1681 |
| peak VRAM GiB | 4.388660430908203 |
| latency ms/icon, batch 1, median | 3637.8 |
| latency ms/icon, batch 1, p95 | 3669.9 |
| throughput ms/icon, batch 64 | 126.4 |
| model calls per icon (decoding) | 745.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
