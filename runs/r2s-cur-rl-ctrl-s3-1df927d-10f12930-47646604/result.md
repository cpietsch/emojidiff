# r2s-cur-rl-ctrl-s3-1df927d-10f12930-47646604

**Hypothesis.** Supervised (control) fine-tuning of the final model for 800 steps at lr 2e-5 on the d02 recipe's online stream. Declared in state/CURRENT.md (2026-10-31 18:45): the RL arm's greedy float32 pixel error on the 339 validation renders beats the control's, seed-matched over seeds 7001-7004 with a bootstrap interval excluding zero, and the best RL run is more often recognisable than the final model on the same blind visual review.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0664 [0.0616, 0.0713] |
| model pixel error, median | 0.0606 |
| exact program rate | 0.029 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0234 [0.0194, 0.0273] |
| icons where model beats nearest icon | 249 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.688 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 187 |
| peak VRAM GiB | 2.303131580352783 |
| latency ms/icon, batch 1, median | 2139.8 |
| latency ms/icon, batch 1, p95 | 2150.6 |
| throughput ms/icon, batch 64 | 67.7 |
| model calls per icon (decoding) | 763.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
