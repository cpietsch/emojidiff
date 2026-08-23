# Full primary structural codec census

Pinned manifest: 4006 icons; exactly 8012 representation attempts.

| representation | successful | failed | median contours | median segments |
|---|---:|---:|---:|---:|
| semantic | 3937 | 69 | 12.0 | 77.0 |
| outlined | 3997 | 9 | 15.0 | 184.0 |

`attempts.jsonl` preserves ordered contour lengths/layers and exact styles; `hybrid.jsonl` records semantic-native versus outlined-fallback routing. Candidate capacity loss is exact in `summary.json` (CLOSE consumes a slot). Percentiles use higher order statistics and are not integer safety bounds.
