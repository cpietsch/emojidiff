# Dense versus packed capacity study

Complete hybrid corpus: 4006 icons.

| candidate | layout | logical slots | exact icons | dropped contours | dropped segments | mean utilization |
|---|---|---:|---:|---:|---:|---:|
| dense-p64-s128 | dense | 8256 | 3998 | 35 | 493 | 1.2414% |
| dense-p96-s384 | dense | 36960 | 4006 | 0 | 0 | 0.2777% |
| packed-p48-t256 | packed | 304 | 3912 | 716 | 7467 | 33.0865% |
| packed-p64-t512 | packed | 576 | 4000 | 63 | 1005 | 17.7707% |
| packed-p80-t768 | packed | 848 | 4005 | 12 | 480 | 12.0876% |
| packed-p80-t1216 | packed | 1296 | 4006 | 0 | 0 | 7.9186% |

Adaptive packed buckets are exact for every icon, allocate 190.72 logical slots per icon on average, and use 53.81% of them.

Packed truncation retains a whole-contour prefix; it never creates a partial contour merely to fit the total-segment budget.
