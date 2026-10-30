# r2s-cur-twemoji-src-67bab78-4d130310-47646604

**Hypothesis.** Told which emoji set each training icon comes from (OpenMoji strokes or Twemoji fills), and decoding as OpenMoji, the model trained on OpenMoji plus the Twemoji icons that fit the codec transcribes OpenMoji validation renders better than v9, instead of hedging between two conventions as v8 did. Pass requires the paired 95% bootstrap interval on the per-icon reduction in mean 72 px pixel error on the 339 validation renders, against the v9 rerun, to exclude zero. Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0835 [0.0782, 0.0889] |
| model pixel error, median | 0.0768 |
| exact program rate | 0.009 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0062 [0.0020, 0.0104] |
| icons where model beats nearest icon | 194 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.531 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 9,026,338 |
| train seconds | 5771 |
| peak VRAM GiB | 4.386440277099609 |
| latency ms/icon, batch 1, median | 3592.8 |
| latency ms/icon, batch 1, p95 | 3836.4 |
| throughput ms/icon, batch 64 | 75.6 |
| model calls per icon (decoding) | 677.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
