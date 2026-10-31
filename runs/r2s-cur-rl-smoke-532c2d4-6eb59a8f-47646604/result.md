# r2s-cur-rl-smoke-532c2d4-6eb59a8f-47646604

**Hypothesis.** Smoke test: render-reward fine-tuning from the final model runs end to end (6 steps, 4 icons x 4 samples on the 16 GiB card).

## Result

| measure | value |
| --- | --- |
| icons evaluated | 16 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0642 [0.0456, 0.0830] |
| model pixel error, median | 0.0606 |
| exact program rate | 0.000 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1829 |
| nearest training icon pixel error | 0.0775 [0.0569, 0.1018] |
| error reduction vs nearest icon | 0.0133 [-0.0093, 0.0349] |
| icons where model beats nearest icon | 11 |
| CLIP top-1, model greedy (4 Gate N icons) | 1.000 |
| CLIP top-1, nearest training icon | 0.250 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA GeForce RTX 4080 |
| parameters | 9,025,826 |
| train seconds | 21 |
| peak VRAM GiB | 2.6904897689819336 |
| latency ms/icon, batch 1, median | 1266.5 |
| latency ms/icon, batch 1, p95 | 1268.6 |
| throughput ms/icon, batch 16 | 122.4 |
| model calls per icon (decoding) | 621.0 |
| torch / CUDA | 2.14.0a0+4fdf77b940.nv26.08 / 13.4 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
