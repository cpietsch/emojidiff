# r2s-cur-d02-dec8-508979a-0eedf911-47646604

**Hypothesis.** With dropout 0.2 holding back overfitting, 8 decoder layers (which only leaned better on MUSICA) transcribe validation renders better than v9. Pass requires the paired 95% bootstrap interval on the per-icon reduction in mean 72 px pixel error on the 339 validation renders to exclude zero against each v9 run of the same protocol (seeds 7001-7003 on the B300, the H100 rerun, the RTX 4080 original). Its greedy latency on the RTX 4080 is recorded.

## Result

| measure | value |
| --- | --- |
| icons evaluated | 339 (primary/validation) |
| model pixel error, mean [95% CI] | 0.0684 [0.0635, 0.0733] |
| model pixel error, median | 0.0639 |
| exact program rate | 0.044 |
| rendered rate | 1.000 |
| blank canvas pixel error | 0.1720 |
| nearest training icon pixel error | 0.0897 [0.0848, 0.0947] |
| error reduction vs nearest icon | 0.0213 [0.0172, 0.0254] |
| icons where model beats nearest icon | 247 |
| CLIP top-1, model greedy (32 Gate N icons) | 0.594 |
| CLIP top-1, nearest training icon | 0.344 |
| CLIP top-1, OmniSVG 4B zero-shot (Gate N) | 0.609 |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA B300 SXM6 AC |
| parameters | 11,132,706 |
| train seconds | 4818 |
| peak VRAM GiB | 5.548108100891113 |
| latency ms/icon, batch 1, median | 2053.4 |
| latency ms/icon, batch 1, p95 | 2066.6 |
| throughput ms/icon, batch 64 | 84.7 |
| model calls per icon (decoding) | 553.0 |
| torch / CUDA | 2.14.1+cu130 / 13.0 |

Latency covers encoding and decoding to a validated token program; it excludes
rasterising the SVG. Samples: `samples.png`, rows are reference, model, nearest
training icon.
