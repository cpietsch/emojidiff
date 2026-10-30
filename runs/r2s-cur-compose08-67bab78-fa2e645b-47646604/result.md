# r2s-cur-compose08-67bab78-fa2e645b-47646604

**Hypothesis.** With compositions presented with probability 0.8 instead of 0.5, and everything else as in v9, the render-to-program model transcribes the 339 validation renders with lower mean 72 px pixel error than v9 trained on the same machine and seed. Pass requires the paired 95% bootstrap interval on the per-icon reduction against that v9 rerun to exclude zero. Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0831 [0.0779, 0.0884] |
| model pixel error, median | 0.0791 |
| exact program rate | 0.009 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0067 [0.0022, 0.0109] |
| icons where model beats nearest icon | 223 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.531 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 6466 |
| peak VRAM GiB | 4.388385772705078 |
| latency ms/icon, batch 1, median | 4678.3 |
| latency ms/icon, batch 1, p95 | 4695.9 |
| throughput ms/icon, batch 64 | 129.8 |
| model calls per icon (decoding) | 936.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
