# Visual review: the final model's greedy SVGs on all 339 validation icons

2026-10-31. Model `r2s-cur-d02-s120k-seed2-53b3108-46294452-47646604`, greedy decoding: the SVGs
the browser demo shows (Python ONNX Runtime, token-identical to the browser), from the bundle
`web_e2e --icons 339`. The operator asked for a visual judgement because pixel error does not say
whether an icon is usable.

## Method

- `make_sheets.py` renders 15 contact sheets: input render on the left, the model's SVG (cairosvg,
  144 px, on white) on the right, under neutral ids in a shuffled order (seed 0). No pixel error and no
  hexcode on the sheets, so the grading is blind to the metric and to the emoji's name.
- One test per icon: would a viewer who sees only the model's image name the same emoji?
  **3 faithful** (yes, minor flaws only; it could stand in for the icon), **2 recognisable** (yes, with
  obvious flaws), **1 partial** (no, but clear resemblance: layout, colours, a fragment), **0 garbage**
  (no meaningful resemblance). A lost defining attribute (vampire's cape, fairy's wings, a sign's letter)
  makes it 1 unless the object stays clearly that emoji.
- Graded by eight parallel model graders (two sheets each, the rubric with eleven calibration
  examples), plus a ninth that re-graded one grader's two sheets blind. I graded sheets 0 and 7 myself
  before seeing theirs (`my-grades-sheets-00-07.json`) and checked 12 random icons per grade
  (`by-grade-*.png`). Per-icon grades, subjects, descriptions and defect tags: `grades.json`.

## Result

| grade | icons | share | pixel error, median (IQR) |
| --- | --- | --- | --- |
| 3 faithful | 53 | 16% | 0.003 (0.000-0.008) |
| 2 recognisable | 124 | 37% | 0.051 (0.037-0.071) |
| 1 partial | 156 | 46% | 0.076 (0.056-0.099) |
| 0 garbage | 6 | 2% | 0.112 (0.095-0.141) |

**About half the icons are recognisable (53%), one in six could stand in for the original (16%), and
almost none are nonsense (2%).** The other 46% keep the layout and colours or a fragment but lose the
subject. Weighting each of the 177 emoji families once gives the same picture (53% and 15%).

Agreement: a grader and the blind re-grade agree exactly on 46 of 48 icons (96%, never more than one
grade apart); my grades and the graders' agree on 45 of 48 (94%, never more than one apart). Pixel
error tracks the grade (Spearman -0.55): faithful icons sit below 0.01, recognisable ones near 0.05,
partial ones near 0.08. The model's mean of 0.064 is on the boundary between recognisable and partial,
so the 0.003-0.005 gains of the recent recipe changes are small in visual terms.

### What works and what fails

- **Works:** single simple shapes and plain figures: raised and folded hands, hearts, faces, busts,
  walking people (all 12 walking-person variants are faithful), hexagons, the NO sign.
- **Complexity:** icons that need fewer than 300 decoder calls are 68% recognisable and 38% faithful;
  above 450 calls almost none are faithful (1% and 0%).
- **Defects** (tags over all 339): missing parts 153, distorted shape 96, fragmented 85, wrong colour 82,
  missing fill 68, extra strokes 26, broken outline 23, lost text or symbol 18.
- **It draws familiar templates and drops what it has not seen.** Vampires, fairies and older people
  (36 icons; the family-disjoint split keeps these families out of training) become plain people
  in 35 of 36. Multi-part action
  scenes (water polo 18 variants, weightlifting 18) break into fragments: the water stays, the figure goes.
- **A systematic colour error:** in couple and kiss icons the red heart comes out brown in 18 of 26.
- **Not just recall.** 72% of the faithful icons have no near-twin in training (nearest training render
  above 0.02). On the 237 icons with nothing close in training, 45% are recognisable and 11% faithful.

## Reading

The model is a usable vectoriser for simple, bold icons and a poor one for detailed scenes; it
transcribes shapes it knows and replaces unfamiliar parts with the nearest familiar template. At 2,681
training icons, recipe tuning moves pixel error by a few percent and does not change this picture.
