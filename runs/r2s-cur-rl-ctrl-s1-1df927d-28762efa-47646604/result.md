# r2s-cur-rl-ctrl-s1-1df927d-28762efa-47646604

**Hypothesis.** Supervised (control) fine-tuning of the final model for 800 steps at lr 2e-5 on the d02 recipe's online stream. Declared in state/CURRENT.md (2026-10-31 18:45): the RL arm's greedy float32 pixel error on the 339 validation renders beats the control's, seed-matched over seeds 7001-7004 with a bootstrap interval excluding zero, and the best RL run is more often recognisable than the final model on the same blind visual review.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0635 [0.0590, 0.0683] |
| model pixel error, median | 0.0587 |
| exact program rate | 0.032 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0262 [0.0222, 0.0302] |
| icons where model beats nearest icon | 268 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.750 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 173 |
| peak VRAM GiB | 2.303131580352783 |
| latency ms/icon, batch 1, median | 4327.6 |
| latency ms/icon, batch 1, p95 | 4339.8 |
| throughput ms/icon, batch 64 | 110.2 |
| model calls per icon (decoding) | 888.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
