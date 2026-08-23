# Semantic-stroke versus outlined representation probe

Fixture: 88 primary-manifest icons; PicoSVG successes: 88.

This is a pre-codec structural proxy, not a representation selection. Source SVG
primitives approximate semantic paths; PicoSVG materializes shapes, transforms,
clip paths, and strokes into filled paths.

| metric | median | p95 | max |
|---|---:|---:|---:|
| semantic_proxy_path_count | 10 | 30.95 | 48 |
| outline_path_count | 10 | 32.65 | 48 |
| path_expansion_ratio | 1 | 1.3408 | 1.66667 |
| semantic_proxy_segment_count | 51.5 | 323.4 | 1167 |
| outline_segment_count | 146 | 514.4 | 1167 |
| segment_expansion_ratio | 2.241 | 5.28711 | 13.8333 |
| rgba_mae_72 | 0.000558184 | 0.00182756 | 0.00248956 |
| rgba_mae_18 | 0.00117254 | 0.00349189 | 0.00603667 |
| alpha_iou_72 | 0.999005 | 1 | 1 |
| alpha_iou_18 | 1 | 1 | 1 |

## Truncation counts

```json
{
  "outline": {
    "path_slots": {
      "16": 19,
      "32": 5,
      "64": 0
    },
    "segments_per_path": {
      "128": 8,
      "32": 43,
      "64": 18
    }
  },
  "semantic_proxy": {
    "path_slots": {
      "16": 19,
      "32": 4,
      "64": 0
    },
    "segments_per_path": {
      "128": 5,
      "32": 16,
      "64": 7
    }
  }
}
```
