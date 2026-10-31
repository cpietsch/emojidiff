# Current research state

Updated: 2026-10-31 15:00

## Current hypothesis and evidence

The first phase (2026-08-23 to 2026-09-24, Gates A to O) built a render-safe typed SVG codec
that round-trips all 4,006 curated OpenMoji icons, fixed hashed family-disjoint splits
(P32/T128 bucket: 2,681 train / 339 val / 339 test), a paired render evaluation with
bootstrap intervals, and a run registry. Full narrative: `reports/findings.md`; last long
handoff: `reports/state-history-2026-09-24.md`.

The deliverable is in `AGENTS.md`. Since 2026-09-27 the direction is a small render-to-SVG
transcriber over the codec (`mojidiff.learning.render2svg`).

**Final model since 2026-10-30:** `r2s-cur-d02-s120k-seed2-53b3108-46294452-47646604` (9.0M
parameters; v9's recipe with dropout 0.2 and 120k steps; best.pt at step 64k), chosen by a rule
declared before its results. Untouched test split, float32: greedy 0.0617 [0.0567, 0.0667]
against v7 0.0710 (paired +0.0093 [+0.0056, +0.0131]); best of 8 0.0397 (v7 0.0506). RTX 4080,
16-icon median: greedy 388 ms, best of 8 1,022 ms. Checkpoint in
`/home/dev/.cache/mojidiff/runs/<run_id>/` and on the HF checkpoints repo.

What the Curiosity night established (findings 2026-10-30, three entries): at 2,681 training
icons dropout (0.15-0.3) is the one lever that holds, about 5%; length, depth, width, Twemoji
(with or without an emoji-set input) and more compositions do not help. Seeds scatter by
0.005-0.010 per configuration, and the 64-icon checkpoint selection adds noise without gain,
so configurations are compared by seed means and last checkpoints are reported too. Choosing
per icon among several seeds' greedy decodes beats sampling one model (validation: best of 4
models 0.048 against best of 4 samples 0.051; 8 models 0.043). Training runs at about 21
steps/s per run, twice wave 1, with byte-identical batches (`reports/curiosity/speedup.md`).

**Direction for the next 24 hours (operator granted them at ~22:45):** seeds 6-9 of the final
configuration as a model pool; a proper best-of-models decoder with one declared test look;
the browser demo on the new model; the canvas-latent line (vt, flow prior) rebuilt on the
dropout recipe, the operator's preferred generative direction. A read-only investigation of
the three code paths (browser, latent, decoding) is running before the plan is fixed.

## Last completed action and verification

2026-10-30 22:50: final model scored once on the test split (float32 greedy and best of 8) and
for latency on the RTX 4080; paired test comparison against v7 by re-decoding both
(`/home/dev/.cache/mojidiff/ensemble-2026-10-30/paired_test_*`). Findings entry written. The
first test attempt never ran (zsh passed `$args` as one argument); no number was seen before
the recorded run.

## Active jobs

- Curiosity (`ssh curiosity`, account eai-hackathon, QoS eai-team11, user eai-jdkxz). One-GPU
  jobs; job scripts ~/eai-team11/mojidiff/jobs/arms-<commit>.slurm (MODULE/EXTRA). Submit and wait
  only through scratchpad `submit_wave.sh` / `wait_jobs.sh` (bash; zsh loops broke waiters twice).
  Nothing running (14:25). Wave 12 done: dropout 0.2 + lr 1e-3 + 120k is level with d02-s120k
  (-0.0009 [-0.0028, +0.0011], 4/8); the final-model rule's stage 1 failed, so the final model stays
  and recipe tuning is closed (findings 2026-10-31). Waves 9-11 are complete (findings 2026-10-31): translation range 64 +0.0000 [-0.0020, +0.0023].
  Learning rate at 60k, seed-matched against 5e-4 (8 seeds unless noted): 3e-4 -0.0042, 7e-4
  +0.0028 [+0.0011, +0.0046], 1e-3 +0.0035 [+0.0002, +0.0066], 1.5e-3 +0.0001 (7 seeds; -0.0043
  [-0.0068, -0.0015] against 1e-3). Earlier: dropout 0.2 vs v9 +0.0039, compose 0.3 -0.0040,
  compose 0.7 -0.0013 (n.s.), 120k vs 60k +0.0024.
- Browser demo (web/demo, 92089f7): the operator closed the tunnel and the demo server at 16:05; restart with
  `node web/demo/serve.mjs` (see its header) when needed. Its review fixes are in (95c3348).
- Visual review (findings 2026-10-31, reports/visual/2026-10-31-final-greedy/): of the greedy SVGs 16%
  faithful, 37% recognisable, 46% partial, 2% garbage; offered the operator a best-of-8 visual check.
- Browser milestone 1 merged (2b2ab8f, squash of browser-m1; findings 2026-10-31): the final model
  decodes in Chrome token for token as Python, WASM 4 threads 1,171 ms per icon median. Agent
  worktrees (mojidiff-browser-m1*, mojidiff-m1js-review, mojidiff-review-m1py) are still on disk.
- d02, vt-v3-d02-s5 and lfp-v2x are on the public HF checkpoints repo (operator approved 2026-10-31;
  sha256 verified, index reports/checkpoints-hf.json) and back in configs/gallery/models.yaml.
- Hugging Face token: `~/.cache/huggingface/token` here, `~/.config/mojidiff/hf_token` there
  (mode 600). It was pasted into the session; rotate it after the hackathon.
- Local servers stay stopped; the RTX 4080 runs short evaluations only.

## Artifact durability

- Checkpoints: pre-cluster ones on https://huggingface.co/chrispie/mojidiff-checkpoints (index
  `reports/checkpoints-hf.json`); Curiosity ones arrive through the private dataset
  `chrispie/mojidiff-transfer` (`results/<job>/checkpoints`) and are copied to
  `/home/dev/.cache/mojidiff/runs/` when needed. Cluster storage is purged after the hackathon.
- Speed-up research scripts: `/home/dev/.cache/mojidiff/speedup-2026-10-30/`; ensemble and paired
  test: `/home/dev/.cache/mojidiff/ensemble-2026-10-30/`.
- MUSICA runs stay in `$DATA/predictable/cache` (no backup) until copied here.
- Browser probe: `/home/dev/.cache/mojidiff/browser-probe-2026-10-04/`; results `reports/browser/m0/`.
- Code on GitHub `cpietsch/emojidiff`; local main is ahead of origin (not pushed).

## Current blockers

None for training. Browser demo, open with the operator: the go/no-go thresholds, best of 8
on WebGPU only, ONNX files under `onnx/` in the checkpoints repo, test-split icons labelled.

## Next smallest evidence-producing action

Decision for the operator (asked 15:00): vtracer, a classical tracer, beats the transcriber 3x on pixel error
(0.022 vs 0.064; findings 2026-10-31, reports/visual/2026-10-31-tracer-baseline/) at 1.8 ms on a CPU. Either
measure the demo openly against vtracer, or move the model's role to what a tracer cannot do (OpenMoji-
structured output: strokes, the 72-unit frame; or generation). No new cluster training until decided; the
cluster is idle with ~7.5 h left. Best-of-8 SVGs for the 339 icons are being generated on the 4080.
