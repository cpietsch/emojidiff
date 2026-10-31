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
