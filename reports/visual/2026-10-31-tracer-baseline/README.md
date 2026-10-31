# Reference point: a classical tracer (vtracer) on the same 339 validation icons

2026-10-31. The 144 px input renders the model reads (the `web_e2e --icons 339` bundle) traced by
[vtracer](https://github.com/visioncortex/vtracer) 0.6.15 in colour mode (`trace.py`: its defaults, and a
"compact" preset with coarser colours and a larger speckle filter), installed in a separate tool venv under
`/home/dev/.cache/mojidiff/tools/`. Every SVG is scored like the model's: cairo at 72 px, composited on
white, `render2svg.pixel_error` against the corpus target (the same cairo path as `render_trusted_rgb`,
minus the codec check, verified identical on the model's SVGs). `metrics.json` holds per-icon values.

| method | pixel error (mean, 95% CI) | median | paths, median (p90) | bytes, median | path commands, median | time per icon |
| --- | --- | --- | --- | --- | --- | --- |
| final model, greedy | 0.0642 [0.0595, 0.0692] | 0.0597 | 10 (18) | 2,590 | 68 | 388 ms (RTX 4080), ~1.2 s (browser WASM) |
| vtracer, default | 0.0219 [0.0211, 0.0227] | 0.0209 | 9 (16) | 13,051 | 169 | 1.8 ms (CPU) |
| vtracer, compact | 0.0246 [0.0236, 0.0256] | 0.0238 | 7 (11) | 4,326 | 130 | 1.6 ms (CPU) |

Paired: vtracer default beats the model by 0.0424 [0.0381, 0.0469] and on 87% of icons; compact by 0.0397
[0.0353, 0.0442], 84%. The model's best of 8 (validation 0.0450) is still twice the tracer's error.

`compare-input-model-vtracer.png` (12 icons of contact sheet 0): the compact trace is recognisable on all
twelve, including the ones the model breaks (water polo, picnic basket, fairy, bus stop, key, "OFF"); its flaws
are slightly darker skin tones, soft edges and a full-canvas background rectangle.

What the tracer does not give: OpenMoji's structure. Its black outlines are filled ring shapes, not strokes
(no editable line width), colours are quantised layers stacked from the background up, and the file is
1.7-5x larger. The model writes strokes as strokes in OpenMoji's 72-unit frame through the typed codec.

## Visual grades: greedy, best of 8 and the tracer on the same blind sheets

The best-of-8 SVGs (`rerank_svgs.py`: `rerank_decode`, temperature 0.7, seed = icon index; mean pixel error
0.0450, the committed `eval-validation-rerank8` value exactly; a sample beat greedy on 81% of icons) and the
vtracer compact SVGs were put on contact sheets with the same neutral ids, order and layout as the greedy review
(`../2026-10-31-final-greedy/`) and graded by the same workflow: the same rubric and test (would a viewer name the
same emoji from the right image alone?), eight graders and a blind re-grade of two sheets.

| method | faithful | recognisable | partial | garbage | recognisable or better | mean grade |
| --- | --- | --- | --- | --- | --- | --- |
| model, greedy | 16% | 37% | 46% | 2% | 52% | 1.66 |
| model, best of 8 | 22% | 43% | 34% | 1% | 65% | 1.87 |
| vtracer, compact | 52% | 47% | 1% | 0% | **99%** | 2.52 |

- Best of 8 against greedy: 22% of icons up a grade, 4% down; mean grade +0.21 [+0.15, +0.27].
- vtracer against best of 8: the tracer is better on 60% of icons, level on 32%, worse on 9%.
- Agreement: blind re-grades 98% (best of 8) and 96% (vtracer) exact; my own grades of best-of-8 sheet 0 agree
  71% exactly with the graders, never more than one grade apart (`my-grades-rerank8-sheet-00.json`).
- The tracer's defects come from the compact preset: small facial features vanish (32 icons with blank faces) and
  small outlined heads or balls fill dark (81). The 9% where the model is better are exactly those: walking people,
  busts, couples, whose faces the model draws as clean templates. The default preset keeps more detail (pixel
  error 0.0219 against 0.0246) at three times the file size.
- Per-icon grades for all three: `grades-greedy-best8-vtracer.json`; sheet 0 of each: `sheet-00-*.png`.
