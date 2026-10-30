# r2s-cur-twemoji-src-dec8-67bab78-5117c6cd-47646604

**Hypothesis.** With the emoji-set input, the Twemoji data and 8 decoder layers, the model transcribes OpenMoji validation renders better than v9; with nearly twice the training icons, the depth that only leaned better on 2,681 icons now has data to use. Pass requires the paired 95% bootstrap interval on the per-icon reduction in mean 72 px pixel error on the 339 validation renders, against the v9 rerun, to exclude zero. Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0813 [0.0760, 0.0869] |
| model pixel error, median | 0.0767 |
| exact program rate | 0.012 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0084 [0.0040, 0.0126] |
| icons where model beats nearest icon | 212 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.562 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 11,133,218 |
| train seconds | 6040 |
| peak VRAM GiB | 5.5490922927856445 |
| latency ms/icon, batch 1, median | 3796.0 |
| latency ms/icon, batch 1, p95 | 3815.4 |
| throughput ms/icon, batch 64 | 121.0 |
| model calls per icon (decoding) | 648.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
