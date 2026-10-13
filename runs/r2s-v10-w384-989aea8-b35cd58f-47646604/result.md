# r2s-v10-w384-989aea8-b35cd58f-47646604

**Hypothesis.** With width 256 -> 384 (feedforward 1024 -> 1536) and everything else as in v9, the render-to-program model transcribes the 339 validation renders with lower mean 72 px pixel error than v9 trained on the same machine and seed. Pass requires the paired 95% bootstrap interval on the per-icon reduction against that v9 rerun to exclude zero. Its greedy latency on the RTX 4080 is recorded; a pass that is too slow for the browser demo is a teacher for distillation, not the demo model.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0736 [0.0683, 0.0791] |
| model pixel error, median | 0.0720 |
| exact program rate | 0.041 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0161 [0.0119, 0.0202] |
| icons where model beats nearest icon | 229 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.625 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA H100 |
| parameters | 19,762,402 |
| train seconds | 3655 |
| peak VRAM GiB | 6.413517951965332 |
| latency ms/icon, batch 1, median | 3167.7 |
| latency ms/icon, batch 1, p95 | 7408.5 |
| throughput ms/icon, batch 64 | 89.1 |
| model calls per icon (decoding) | 637.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
