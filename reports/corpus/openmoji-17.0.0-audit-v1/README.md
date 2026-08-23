# OpenMoji corpus audit distributions

Audited rows: 4495

Parse failures: 2
72 px render failures: 2
18 px render failures: 2

Numeric quarantine thresholds are intentionally unset in the initial pass.
The committed `distributions.json` contains observed quantiles used to select them.

| metric | min | q0.1% | q0.5% | median | q99.5% | max |
|---|---:|---:|---:|---:|---:|---:|
| alpha_fraction_72 | 0.0162037 | 0.0455123 | 0.0878588 | 0.322724 | 0.601852 | 0.806906 |
| ink_fraction_72 | 0.0160221 | 0.0318523 | 0.0640519 | 0.294765 | 0.577963 | 0.792486 |
| bbox_width_72 | 2 | 15 | 20 | 50 | 68 | 72 |
| bbox_height_72 | 10 | 13.984 | 25 | 51 | 67 | 70 |
| bbox_area_72 | 84 | 557.712 | 945.46 | 2520 | 4391.64 | 4761 |
| alpha_fraction_18 | 0.0740741 | 0.0956296 | 0.149568 | 0.401235 | 0.694444 | 0.858025 |
| segment_count | 0 | 0 | 0 | 59 | 328 | 1167 |
| path_count | 0 | 0 | 0 | 8 | 33 | 78 |
| component_count_72 | 1 | 1 | 1 | 1 | 5 | 21 |
| total_path_length_estimate | 0 | 0 | 0 | 506.643 | 1447.18 | 8139.3 |
