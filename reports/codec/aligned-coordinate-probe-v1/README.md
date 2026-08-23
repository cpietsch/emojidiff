# Typed semantic-stroke versus outlined codec probe

Fixture: 88 icons. PicoSVG 0.23.0; CairoSVG 2.9.0.

| representation / budget / bins | ok | MAE 72 median | MAE 18 median | alpha IoU 18 median | truncated icons (P/S) |
|---|---:|---:|---:|---:|---:|
| outlined / compact-p48-s128 / q145 | 88/88 | 0.00327913 | 0.00337842 | 0.993484 | 2/3 |
| outlined / compact-p48-s128 / q289 | 88/88 | 0.00204267 | 0.00231935 | 1 | 2/3 |
| outlined / coverage-p64-s384 / q145 | 88/88 | 0.00321918 | 0.00337842 | 0.994666 | 0/0 |
| outlined / coverage-p64-s384 / q289 | 88/88 | 0.00198565 | 0.00231935 | 1 | 0/0 |
| semantic / compact-p48-s128 / q145 | 87/87 | 0.00269986 | 0.00264766 | 0.994872 | 0/3 |
| semantic / compact-p48-s128 / q289 | 87/87 | 0.00153867 | 0.00172174 | 1 | 0/3 |
| semantic / coverage-p64-s384 / q145 | 87/87 | 0.00269986 | 0.00264766 | 0.994872 | 0/0 |
| semantic / coverage-p64-s384 / q289 | 87/87 | 0.00151295 | 0.00170963 | 1 | 0/0 |

Normalization failures and exact worst-case lists are retained in `summary.json`;
per-icon evidence is in `normalization.jsonl` and `metrics.jsonl`.

The aligned lattices support the hypothesis. At the coverage budget, semantic q145
beats q128 despite using a comparably small vocabulary, and q289 lowers median MAE
relative to q256 by 48.2% at 72 px and 42.5% at 18 px. Outlined q289 improves by 23.9%
and 30.2%. The `E2C2` semantic 18 px MAE falls from 0.117692 at q256 to 0.001970 at
q289, while alpha IoU rises from 0.8 to 1.0.

The [18 px grid comparison](grid-comparison-semantic-18.png) confirms that this is an
edge-alignment effect rather than a change in recognizability. Q289 is the leading
fidelity vocabulary; q145 remains a compact ablation. Slot/segment truncation is
unchanged because only the coordinate vocabulary changed.
