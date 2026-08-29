# Typed semantic-stroke versus outlined codec probe

Fixture: 24 icons. PicoSVG 0.23.0; CairoSVG 2.9.0.

| representation / budget / bins | ok | MAE 72 median | MAE 18 median | alpha IoU 18 median | truncated icons (P/S) | projected icons |
|---|---:|---:|---:|---:|---:|---:|
| outlined / coverage-p48-s64 / q289 | 24/24 | 0.0027983 | 0.00373245 | 0.993391 | 0/0 | 22 |
| semantic / coverage-p48-s64 / q289 | 24/24 | 0.00243396 | 0.00292 | 0.996124 | 0/0 | 24 |

Normalization failures and exact worst-case lists are retained in `summary.json`;
per-icon evidence is in `normalization.jsonl` and `metrics.jsonl`.

## Out-of-bounds coordinate projection

The unclamped comparison extends the same coordinate lattice outside 0..72 for rendering only. It is not a proposed model vocabulary.

| representation / budget / bins | clamp MAE 72 median/max | clamp MAE 18 median/max |
|---|---:|---:|
| outlined / coverage-p48-s64 / q289 | 6.0518e-06/0.0282022 | 0.000217865/0.0314391 |
| semantic / coverage-p48-s64 / q289 | 0.000114228/0.0313243 | 0.00027687/0.0381173 |
