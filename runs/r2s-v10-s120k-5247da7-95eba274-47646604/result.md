# r2s-v10-s120k-5247da7-95eba274-47646604

**Hypothesis.** With training steps 60k -> 120k, and everything else as in v9, the render-to-program model transcribes the 339 validation renders with lower mean 72 px pixel error than v9 trained on the same machine and seed. Pass requires the paired 95% bootstrap interval on the per-icon reduction against that v9 rerun to exclude zero. Its greedy latency on the RTX 4080 is recorded; a pass that is too slow for the browser demo is a teacher for distillation, not the demo model.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0727 [0.0677, 0.0780] |
| model pixel error, median | 0.0678 |
| exact program rate | 0.032 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0170 [0.0131, 0.0209] |
| icons where model beats nearest icon | 237 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.750 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA H100 |
| parameters | 9,025,826 |
| train seconds | 6360 |
| peak VRAM GiB | 4.38838529586792 |
| latency ms/icon, batch 1, median | 2993.2 |
| latency ms/icon, batch 1, p95 | 8018.8 |
| throughput ms/icon, batch 64 | 79.3 |
| model calls per icon (decoding) | 763.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
