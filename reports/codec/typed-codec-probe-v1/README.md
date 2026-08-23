# Typed semantic-stroke versus outlined codec probe

Fixture: 88 icons. PicoSVG 0.23.0; CairoSVG 2.9.0.

| representation / budget / bins | ok | MAE 72 median | MAE 18 median | alpha IoU 18 median | truncated icons (P/S) |
|---|---:|---:|---:|---:|---:|
| outlined / compact-p48-s128 / q128 | 88/88 | 0.00507538 | 0.00554648 | 0.988768 | 2/3 |
| outlined / compact-p48-s128 / q256 | 88/88 | 0.00266866 | 0.0033527 | 0.993443 | 2/3 |
| outlined / coverage-p64-s384 / q128 | 88/88 | 0.00492948 | 0.0054905 | 0.989386 | 0/0 |
| outlined / coverage-p64-s384 / q256 | 88/88 | 0.00260937 | 0.00332093 | 0.996732 | 0/0 |
| semantic / compact-p48-s128 / q128 | 87/87 | 0.00463776 | 0.00471738 | 0.992806 | 0/3 |
| semantic / compact-p48-s128 / q256 | 87/87 | 0.00292226 | 0.00313181 | 0.991379 | 0/3 |
| semantic / coverage-p64-s384 / q128 | 87/87 | 0.00463776 | 0.00471738 | 0.992806 | 0/0 |
| semantic / coverage-p64-s384 / q256 | 87/87 | 0.00292226 | 0.00297144 | 0.991379 | 0/0 |

Normalization failures and exact worst-case lists are retained in `summary.json`;
per-icon evidence is in `normalization.jsonl` and `metrics.jsonl`.

Semantic normalization rejected `1F250` because a nonuniform transform on its stroke
cannot be represented by one scalar width. The outlined codec is its documented
fallback. At P48/S128, semantic programs dropped 181 segments and no contours; outlined
programs dropped 509 segments and 13 contours. P64/S384 covered the fixture without
truncation.

Worst-case inspection at [72 px](worst-coverage-q256.png) and
[18 px](worst-18-q256.png) showed that the untruncated q256 programs remain recognizable,
but ordinary pixel-aligned coordinates are displaced by the 255-interval grid. That
alignment artifact motivates the next 0.5/0.25-unit lattice probe.
