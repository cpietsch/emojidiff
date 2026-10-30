# Current research state

Updated: 2026-10-30

## Current hypothesis and evidence

The first phase (2026-08-23 to 2026-09-24, Gates A to O) built a render-safe typed SVG codec
that round-trips all 4,006 curated OpenMoji icons, fixed hashed family-disjoint splits
(P32/T128 bucket: 2,681 train / 339 val / 339 test), a paired render evaluation with
bootstrap intervals, and a run registry. Full narrative: `reports/findings.md`; last long
handoff: `reports/state-history-2026-09-24.md`.

The deliverable is in `AGENTS.md`. Since 2026-09-27 the direction is a small render-to-SVG
transcriber over the codec (`mojidiff.learning.render2svg`). v9
(`r2s-full-v9-colour-621bc8e-b26ad95e-47646604`, 9.0M parameters) is the model: 0.072 mean
72 px pixel error on validation against 0.090 for the nearest training icon; v7, its parent,
beats retrieval on the untouched test split (0.071 against 0.087). RTX 4080, float32: greedy
381 ms per icon median, best of 8 1,154 ms. Browser milestone 0: WASM 881 ms per icon median
on 32 replayed traces (findings, 2026-10-04).

Every from-scratch result reduces to one limit: 2,681 training icons. The MUSICA sweep
(findings, 2026-10-14) made it direct: width, depth, resolution, steps and batch all fail
against a v9 rerun; best checkpoints come early, so extra capacity fits the training set.
The only levers that ever moved the transcriber were data-shaped (exact augmentation,
path order, compositions). Twemoji (v8) cost 0.015 because it writes outlines as fills
where OpenMoji uses strokes, so one render had two right programs.

**Direction since 2026-10-30 (operator, ~20 hours on the Curiosity cluster, 8 B300 GPUs):**
attack the data limit, one factor per arm, paired against a v9 rerun on the same machine.
Wave 1 (eight arms, one job): v9 rerun; second seed (noise floor); Twemoji without and with
an emoji-set input (`model.sources`, commit 67bab78: a learned vector per set on the encoder
memory, OpenMoji at evaluation; single-set compositions); Twemoji + set input + 8 decoder
layers; compositions 0.8; dropout 0.2; weight decay 0.1. Wave 2: a second seed of the best
arm, then that arm with dec8, w384, 120k steps. Wave 3: third seed, test split for the final
winner only. 4080 latency of the winner measured locally afterwards.

## Last completed action and verification

2026-10-30: cluster bring-up. Transfer goes through the private Hugging Face dataset
`chrispie/mojidiff-transfer` (code bundles, raw OpenMoji and Twemoji, corpus cache
5590a1948f5f = dataset 47646604, Twemoji cache; `SHA256SUMS` checked on arrival); jobs
return records, logs and checkpoints under `results/<job id>/` and a `DONE` file the
workstation polls, so waiting needs no ssh. Scripts: `scripts/curiosity/setup.sh`
(per-checkout environment, torch 2.14.1+cu130) and `scripts/curiosity/arms.slurm` (one
config per GPU). The first smoke (job 13937) trained like the 4080 reference (loss 2.6940
against 2.6947 at step 1,200) but every evaluation render failed: the isolated renderer's
`python -I` ignores PYTHONPATH. Recorded as failed; fixed in 67bab78. The emoji-set input:
ruff, strict mypy, three new tests, full pytest 327 passed; its training-loop path is first
run by the token smoke on the cluster.

## Active jobs

- Curiosity (`ssh curiosity`, account eai-hackathon, QoS eai-team11, user eai-jdkxz): wave 1,
  job 13939 on dgx04 (8 GPUs, 67bab78, submitted 2026-10-30 ~14:00 CET, limit 6 h): v9 rerun,
  cur-seed2, cur-twemoji, cur-twemoji-src, cur-twemoji-src-dec8, cur-compose08,
  cur-dropout02, cur-wd01. Smokes 13938 passed (findings to follow with wave 1). Project folder
  `~/eai-team11/mojidiff` (checkouts `repo-<commit>`, caches, `results/<job>`). No backup
  there; storage is purged after the hackathon, so results come home after every job.
- Token for Hugging Face: `~/.cache/huggingface/token` here, `~/.config/mojidiff/hf_token`
  there (mode 600). It was pasted into the session; rotate it after the hackathon.
- Local servers stay stopped; this GPU serves other work (short smoke tests allowed).

## Artifact durability

- Checkpoints: every pre-cluster checkpoint on https://huggingface.co/chrispie/mojidiff-checkpoints
  (index `reports/checkpoints-hf.json`); code on GitHub `cpietsch/emojidiff`.
- Curiosity checkpoints arrive through `chrispie/mojidiff-transfer` (`results/<job>/checkpoints`)
  and are copied to `/home/dev/.cache/mojidiff/runs/`.
- MUSICA runs stay in `$DATA/predictable/cache` (no backup) until copied here.
- Browser probe: `/home/dev/.cache/mojidiff/browser-probe-2026-10-04/`; results `reports/browser/m0/`.

## Current blockers

None for training. Browser demo, open with the operator: the go/no-go thresholds, best of 8
on WebGPU only, ONNX files under `onnx/` in the checkpoints repo, test-split icons labelled.

## Next smallest evidence-producing action

When wave 1 (job 13939) returns, score each arm paired against the v9 rerun with
`python -m mojidiff.learning.paired runs/<arm> runs/<v9 rerun>`, then launch wave 2.
