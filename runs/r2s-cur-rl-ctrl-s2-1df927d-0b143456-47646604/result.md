# r2s-cur-rl-ctrl-s2-1df927d-0b143456-47646604

**Hypothesis.** Supervised (control) fine-tuning of the final model for 800 steps at lr 2e-5 on the d02 recipe's online stream. Declared in state/CURRENT.md (2026-10-31 18:45): the RL arm's greedy float32 pixel error on the 339 validation renders beats the control's, seed-matched over seeds 7001-7004 with a bootstrap interval excluding zero, and the best RL run is more often recognisable than the final model on the same blind visual review.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0636 [0.0588, 0.0686] |
| model pixel error, median | 0.0567 |
| exact program rate | 0.032 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0261 [0.0220, 0.0302] |
| icons where model beats nearest icon | 269 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.688 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,025,826 |
| train seconds | 186 |
| peak VRAM GiB | 2.303131580352783 |
| latency ms/icon, batch 1, median | 3436.5 |
| latency ms/icon, batch 1, p95 | 3457.9 |
| throughput ms/icon, batch 64 | 99.0 |
| model calls per icon (decoding) | 697.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
