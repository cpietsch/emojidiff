# Current research state

Updated: 2026-10-31 18:45 — wrapped up at 17:30; reopened at 18:30 for one bet on the remaining cluster hours

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

2026-10-31 17:30: wrap-up. The five browser-milestone agent worktrees were removed with the operator's
approval (their data symlinks unlinked first; main's 67,977 data files unchanged; branches kept); their
Node packages now live in `web/node_modules` (ignored by git; `e2e/modules.mjs` resolves them there).
The recurring check-in loop was cancelled.

## Active jobs

Wave R1, render-reward RL (operator's choice, 18:30): 8 one-GPU Curiosity jobs from the final model, 800 steps
at lr 2e-5 on the d02 recipe's online stream, seeds 7001-7004 in two arms that differ only in rl_samples:
RL (16 icons x 8 samples per step, reward minus pixel error against the clean render, group-normalised
advantages, plus 0.5 x the supervised loss; `render2svg.render_reward_loss`) and a supervised-only control.
Declared before results (18:45): (1) RL's greedy float32 pixel error on the 339 validation renders beats the
control's, seed-matched with a bootstrap interval excluding zero; (2) the best RL run (lowest validation
error) is recognisable more often than the final model's 52% on the same blind visual review. Both pass:
the RL model becomes a candidate for the demo. (1) fails: RL adds nothing over continued training; stop.
Added 19:00 (operator approved) on the GPUs the controls free: wave R1b, the RL arm at lr 5e-5, same seeds; judged by the same criteria against the same control. Smoke runs on the RTX 4080: r2s-cur-rl-smoke-*. The allocation ends about 22:45.

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
