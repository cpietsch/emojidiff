# r2s-cur-rl-smoke-online-532c2d4-5b2bd54c-47646604

**Hypothesis.** Render-reward fine-tuning of the final model for 800 steps at lr 2e-5 on the d02 recipe's online stream. Declared in state/CURRENT.md (2026-10-31 19:20): the RL arm's greedy float32 pixel error on the 339 validation renders beats the control's, seed-matched over seeds 7001-7004 with a bootstrap interval excluding zero, and the best RL run is more often recognisable than the final model on the same blind visual review.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 16 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0653 [0.0480, 0.0821] |
| model pixel error, median | 0.0686 |
| exact program rate | 0.000 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1829 |
| nearest training icon pixel error | 0.0775 [0.0569, 0.1018] |
| error reduction vs nearest icon | 0.0121 [-0.0120, 0.0358] |
| icons where model beats nearest icon | 11 |
| CLIP top-1, model greedy (4 Gate N icons) | 1.000 |
| CLIP top-1, nearest training icon | 0.250 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA GeForce RTX 4080 |
| parameters | 9,025,826 |
| train seconds | 20 |
| peak VRAM GiB | 2.6880550384521484 |
| latency ms/icon, batch 1, median | 1519.2 |
| latency ms/icon, batch 1, p95 | 1524.4 |
| throughput ms/icon, batch 16 | 122.5 |
| model calls per icon (decoding) | 747.0 |
| torch / CUDA | 2.14.0a0+4fdf77b940.nv26.08 / 13.4 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
