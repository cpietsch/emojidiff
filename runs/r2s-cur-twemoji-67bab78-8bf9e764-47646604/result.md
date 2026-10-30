# r2s-cur-twemoji-67bab78-8bf9e764-47646604

**Hypothesis.** With the Twemoji icons that fit the codec added to the training split (held-out concepts excluded), and everything else as in v9, the render-to-program model transcribes the 339 validation renders with lower mean 72 px pixel error than v9 trained on the same machine and seed. Pass requires the paired 95% bootstrap interval on the per-icon reduction against that v9 rerun to exclude zero. Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0844 [0.0791, 0.0898] |
| model pixel error, median | 0.0801 |
| exact program rate | 0.012 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0054 [0.0009, 0.0098] |
| icons where model beats nearest icon | 195 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.469 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 5753 |
| peak VRAM GiB | 4.388385772705078 |
| latency ms/icon, batch 1, median | 4905.1 |
| latency ms/icon, batch 1, p95 | 5123.5 |
| throughput ms/icon, batch 64 | 124.1 |
| model calls per icon (decoding) | 921.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
