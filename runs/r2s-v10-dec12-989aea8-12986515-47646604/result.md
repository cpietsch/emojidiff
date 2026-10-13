# r2s-v10-dec12-989aea8-12986515-47646604

**Hypothesis.** With decoder layers 6 -> 12 and everything else as in v9, the render-to-program model transcribes the 339 validation renders with lower mean 72 px pixel error than v9 trained on the same machine and seed. Pass requires the paired 95% bootstrap interval on the per-icon reduction against that v9 rerun to exclude zero. Its greedy latency on the RTX 4080 is recorded; a pass that is too slow for the browser demo is a teacher for distillation, not the demo model.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0755 [0.0703, 0.0807] |
| model pixel error, median | 0.0742 |
| exact program rate | 0.047 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0143 [0.0101, 0.0182] |
| icons where model beats nearest icon | 219 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.594 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA H100 |
| parameters | 15,346,466 |
| train seconds | 4851 |
| peak VRAM GiB | 7.8711838722229 |
| latency ms/icon, batch 1, median | 5868.3 |
| latency ms/icon, batch 1, p95 | 11061.2 |
| throughput ms/icon, batch 64 | 136.4 |
| model calls per icon (decoding) | 862.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
