# Typed semantic-stroke versus outlined codec probe

Fixture: 24 icons. PicoSVG 0.23.0; CairoSVG 2.9.0.

| representation / budget / bins | ok | MAE 72 median | MAE 18 median | alpha IoU 18 median | truncated icons (P/S) | projected icons |
|---|---:|---:|---:|---:|---:|---:|
| outlined / coverage-p48-s64 / q289 / control-q417@-8..96 | 24/24 | 0.00270875 | 0.00333606 | 0.994328 | 0/0 | 0 |
| semantic / coverage-p48-s64 / q289 / control-q417@-8..96 | 24/24 | 0.00206745 | 0.00275206 | 1 | 0/0 | 0 |

Normalization failures and exact worst-case lists are retained in `summary.json`;
per-icon evidence is in `normalization.jsonl` and `metrics.jsonl`.
