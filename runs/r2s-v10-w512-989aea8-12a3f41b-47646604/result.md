# r2s-v10-w512-989aea8-12a3f41b-47646604

**Hypothesis.** With width 256 -> 512 (feedforward 1024 -> 2048) and everything else as in v9, the render-to-program model transcribes the 339 validation renders with lower mean 72 px pixel error than v9 trained on the same machine and seed. Pass requires the paired 95% bootstrap interval on the per-icon reduction against that v9 rerun to exclude zero. Its greedy latency on the RTX 4080 is recorded; a pass that is too slow for the browser demo is a teacher for distillation, not the demo model.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0789 [0.0735, 0.0842] |
| model pixel error, median | 0.0753 |
| exact program rate | 0.050 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0109 [0.0066, 0.0150] |
| icons where model beats nearest icon | 216 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.531 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA H100 |
| parameters | 34,648,226 |
| train seconds | 4169 |
| peak VRAM GiB | 8.525425434112549 |
| latency ms/icon, batch 1, median | 1905.8 |
| latency ms/icon, batch 1, p95 | 1913.6 |
| throughput ms/icon, batch 64 | 89.9 |
| model calls per icon (decoding) | 489.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
