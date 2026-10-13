# r2s-v10-b128-e526a4e-f66c07e7-47646604

**Hypothesis.** With batch 32 -> 128 with learning rate 5e-4 -> 1e-3 (square-root rule), 60k steps, so four times the samples, and everything else as in v9, the render-to-program model transcribes the 339 validation renders with lower mean 72 px pixel error than v9 trained on the same machine and seed. Pass requires the paired 95% bootstrap interval on the per-icon reduction against that v9 rerun to exclude zero. Its greedy latency on the RTX 4080 is recorded; a pass that is too slow for the browser demo is a teacher for distillation, not the demo model.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0751 [0.0695, 0.0806] |
| model pixel error, median | 0.0707 |
| exact program rate | 0.047 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0147 [0.0103, 0.0189] |
| icons where model beats nearest icon | 226 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.625 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA H100 |
| parameters | 9,025,826 |
| train seconds | 11444 |
| peak VRAM GiB | 16.676997661590576 |
| latency ms/icon, batch 1, median | 3887.4 |
| latency ms/icon, batch 1, p95 | 3927.9 |
| throughput ms/icon, batch 64 | 86.3 |
| model calls per icon (decoding) | 1005.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
