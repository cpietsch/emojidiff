# Typed semantic-stroke versus outlined codec probe

Fixture: 9 icons. PicoSVG 0.23.0; CairoSVG 2.9.0.

| representation / budget / bins | ok | MAE 72 median | MAE 18 median | alpha IoU 18 median | truncated icons (P/S) | projected icons |
|---|---:|---:|---:|---:|---:|---:|
| outlined / coverage-p96-s64 / q289 | 9/9 | 0.0020701 | 0.00204551 | 0.993421 | 0/0 | 2 |
| semantic / coverage-p96-s64 / q289 | 9/9 | 0.00154699 | 0.00172174 | 1 | 0/0 | 2 |

Normalization failures and exact worst-case lists are retained in `summary.json`;
per-icon evidence is in `normalization.jsonl` and `metrics.jsonl`.
