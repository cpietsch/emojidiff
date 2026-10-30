# r2s-cur-diag-init7004-3d24194-505ccea1-47646604

**Hypothesis.** Diagnostic, not a candidate: if seed 4's initialisation is what makes its runs bad, this v9 run scores near seed 4's 0.0861 rather than v9's 0.072; it is reported, not passed.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0816 [0.0765, 0.0870] |
| model pixel error, median | 0.0776 |
| exact program rate | 0.006 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0081 [0.0035, 0.0127] |
| icons where model beats nearest icon | 204 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.531 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 2890 |
| peak VRAM GiB | 4.388660430908203 |
| latency ms/icon, batch 1, median | 4678.0 |
| latency ms/icon, batch 1, p95 | 4709.3 |
| throughput ms/icon, batch 64 | 118.9 |
| model calls per icon (decoding) | 962.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
