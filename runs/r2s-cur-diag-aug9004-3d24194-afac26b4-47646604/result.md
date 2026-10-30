# r2s-cur-diag-aug9004-3d24194-afac26b4-47646604

**Hypothesis.** Diagnostic, not a candidate: if seed 4's augmentation stream is what makes its runs bad, this v9 run scores near seed 4's 0.0861 rather than v9's 0.072; it is reported, not passed.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0727 [0.0679, 0.0776] |
| model pixel error, median | 0.0681 |
| exact program rate | 0.024 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0170 [0.0130, 0.0209] |
| icons where model beats nearest icon | 240 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.500 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 2878 |
| peak VRAM GiB | 4.388660430908203 |
| latency ms/icon, batch 1, median | 3799.0 |
| latency ms/icon, batch 1, p95 | 3800.5 |
| throughput ms/icon, batch 64 | 112.3 |
| model calls per icon (decoding) | 775.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
