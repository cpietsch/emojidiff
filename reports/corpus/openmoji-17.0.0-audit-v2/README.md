# OpenMoji corpus audit distributions

Audited rows: 4495

Parse failures: 0
72 px render failures: 1
18 px render failures: 1

Numeric quarantine thresholds are intentionally unset in the initial pass.
The committed `distributions.json` contains observed quantiles used to select them.

| metric | min | q0.1% | q0.5% | median | q99.5% | max |
|---|---:|---:|---:|---:|---:|---:|
| alpha_fraction_72 | 0.0162037 | 0.0455139 | 0.0878598 | 0.322917 | 0.601852 | 0.806906 |
| ink_fraction_72 | 0.0160221 | 0.0318549 | 0.0640548 | 0.294765 | 0.577946 | 0.792486 |
| bbox_width_72 | 2 | 15 | 20 | 50 | 68 | 72 |
| bbox_height_72 | 10 | 13.986 | 25 | 51 | 67 | 70 |
| bbox_area_72 | 84 | 557.748 | 945.465 | 2520 | 4391.31 | 4761 |
| alpha_fraction_18 | 0.0740741 | 0.0956358 | 0.149583 | 0.401235 | 0.694444 | 0.858025 |
| segment_count | 0 | 0 | 0 | 58 | 328 | 1167 |
| path_count | 0 | 0 | 0 | 8 | 33 | 78 |
| component_count_72 | 1 | 1 | 1 | 1 | 5 | 21 |
| total_path_length_estimate | 0 | 0 | 0 | 506.643 | 1447.18 | 8139.3 |
