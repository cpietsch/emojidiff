# Current research state

Updated: 2026-10-31 17:10

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

What the cluster established (findings 2026-10-30/31): recipe tuning is closed (dropout 0.2 the one
lever; lr, length, depth, width, compositions, translation, Twemoji level). Visually the final model's greedy
SVGs are 52% recognisable, best of 8 65%; vtracer, a classical tracer, 99% at 1.6 ms on a CPU
(reports/visual/). So the model's job changes to what a tracer cannot do (below).

## Last completed action and verification

2026-10-31 15:58: doodle path (mojidiff.learning.doodle, render2svg degrade hooks, fine-tuning via init_run)
smoke-tested on the RTX 4080 (r2s-cur-doodle-smoke-*: parent loaded with its sha256, degraded selection set,
completed); ruff, strict mypy, 25 targeted tests.

## Active jobs

- Curiosity (`ssh curiosity`, account eai-hackathon, QoS eai-team11, user eai-jdkxz). One-GPU
  jobs; job scripts ~/eai-team11/mojidiff/jobs/arms-<commit>.slurm (MODULE/EXTRA). Submit and wait
  only through scratchpad `submit_wave.sh` / `wait_jobs.sh` (bash; zsh loops broke waiters twice).
  Nothing running. Waves 1-12 and doodle wave D1 are complete and recorded.
- Browser demo (web/demo, 92089f7): the operator closed the tunnel and the demo server at 16:05; restart with
  `node web/demo/serve.mjs` (see its header) when needed. Its review fixes are in (95c3348).
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

Doodle to emoji (wave D1, 8 fine-tunes) failed its declared criterion (2): on lineart and pencil the model's
output is further from the clean icon than the degraded input itself (lineart 0.105 vs 0.097), level with
vtracer on lineart; real QuickDraw doodles fragment (findings 2026-10-31, reports/doodle/2026-10-31-wave-d1/).
By the declared rule the line stops. Asked the operator (17:10): build the fallback variations page from the
existing flow prior (no training; works for busts, faces, couples, stick figures, fragments elsewhere) or wrap
up. The cluster is idle (allocation ends about 22:45); the demo server is off.
