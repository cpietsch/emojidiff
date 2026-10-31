# Current research state

Updated: 2026-10-31 21:00 — wrapped up at 17:30; reopened 18:30-19:55 for render-reward RL (stopped by the operator); OmniLottie zero-shot probe 20:30-21:00

## Where the project stands

The first phase (2026-08-23 to 2026-09-24, Gates A to O) built a render-safe typed SVG codec that
round-trips all 4,006 curated OpenMoji icons, fixed hashed family-disjoint splits (2,681 train / 339
val / 339 test), a paired render evaluation with bootstrap intervals and a run registry. Since
2026-09-27 the model is a small render-to-SVG transcriber (`mojidiff.learning.render2svg`). Full
narrative: `reports/findings.md`; the wrap-up entry of 2026-10-31 summarises the end state.

**Final model:** `r2s-cur-d02-s120k-seed2-53b3108-46294452-47646604` (9.0M parameters, dropout 0.2,
120k steps; best.pt at step 64k), chosen by a declared rule. Untouched test split, float32: greedy
0.0617 [0.0567, 0.0667] against v7 0.0710; best of 8 0.0397. RTX 4080: greedy 388 ms, best of 8
1,022 ms (16 light icons). In Chrome it decodes token for token as Python, WASM 4 threads 1,171 ms per
icon median (`web/`, `reports/browser/m1/`). Checkpoint in `/home/dev/.cache/mojidiff/runs/` and on
the public HF repo `chrispie/mojidiff-checkpoints`.

**What the evidence says** (findings 2026-10-30/31):
- Recipe tuning is closed: dropout 0.2 is the one lever; learning rate, length, depth, width,
  compositions, translation range and Twemoji are level at 2,681 training icons.
- Visually (blind grades, 339 validation icons): greedy 52% recognisable (16% faithful), best of 8 65%
  (22%); vtracer, a classical tracer, 99% (52%) at 1.6 ms on a CPU, and 3x lower pixel error
  (`reports/visual/`). As an image-to-SVG converter the model is not competitive with a tracer; it
  draws familiar templates and drops unfamiliar parts.
- Doodle to emoji (a job a tracer cannot do) failed its declared criterion and fragments real doodles
  (`reports/doodle/2026-10-31-wave-d1/`). The canvas latent and flow prior stopped by their own rules;
  their variations work only for busts, faces, couples and stick figures.

## Last completed action and verification

2026-10-31 21:00: OmniLottie zero-shot probe (Curiosity job 14197; findings 2026-10-31,
`reports/omnilottie/2026-10-31-zero-shot/`): image plus prompt 15/30 recognisable, none faithful, almost no motion,
64.5 s per icon on a B300. No fine-tuning arm.

2026-10-31 17:30: wrap-up. The five browser-milestone agent worktrees were removed with the operator's
approval (their data symlinks unlinked first; main's 67,977 data files unchanged; branches kept); their
Node packages now live in `web/node_modules` (ignored by git; `e2e/modules.mjs` resolves them there).
The recurring check-in loop was cancelled.

## Active jobs

None. Render-reward RL (wave R1/R1b, reopened 18:30) was stopped by the operator at 19:55 before any criterion
could be scored: the supervised controls completed (mean 0.0649 against the parent's 0.0641), the 8 RL runs reached
steps 250-300 with 64-icon evaluations inside the controls' range and are recorded as cancelled (findings
2026-10-31). The Curiosity queue is empty; nothing runs locally; the demo server is off.

## Artifact durability

- Checkpoints: the final model, vt-v3-d02-s5 and lfp-v2x on `chrispie/mojidiff-checkpoints` (index
  `reports/checkpoints-hf.json`); every Curiosity run in the private dataset `chrispie/mojidiff-transfer`
  (`results/<job>/`), the doodle fine-tunes' best.pt also under `/home/dev/.cache/mojidiff/runs/`.
  Cluster storage is purged after the hackathon; nothing there is needed.
- Visual reviews, tracer and doodle evaluations: `reports/visual/`, `reports/doodle/`; raw sheets,
  degraded sets and traces under `/home/dev/.cache/mojidiff/{visual-review-2026-10-31,doodle-eval,quickdraw}/`;
  vtracer in its own venv at `/home/dev/.cache/mojidiff/tools/vtracer-venv`.
- Speed-up and ensemble work: `/home/dev/.cache/mojidiff/{speedup,ensemble}-2026-10-30/`.
- Code on GitHub `cpietsch/emojidiff`; local main is ahead of origin (not pushed).

## Open items for the operator

- Rotate the Hugging Face token (pasted into the session; stored mode 600 at
  `~/.cache/huggingface/token` and `~/.config/mojidiff/hf_token` on Curiosity).
- Push local main to GitHub if wanted (not done).
- Unanswered demo questions: go/no-go latency thresholds, best of 8 on WebGPU only, ONNX hosting,
  test-icon labels.

## Next smallest evidence-producing action, if the project resumes

The model only beats a tracer if it can draw shapes it has not memorised. Options with evidence behind
them: best of 8 in the browser (the one visible model-side gain, 52% to 65% recognisable); a variations
page for the dense families; or a different training signal (synthetic random-program pretraining was
proposed, never run). Recipe arms on the transcriber are not worth more compute.
