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
Wave 1 (findings 2026-10-30): Twemoji hurts with or without an emoji-set input
(`model.sources`, 67bab78), compositions 0.8 hurt, weight decay is neutral, dropout 0.2 leans
better (0.0696; +0.0024-0.0032 against three v9 runs, intervals include zero); a same-seed v9
rerun on the B300 went bad late (0.0790). Wave 2 replicates dropout with seeds and tests it
with dose, weight decay, depth, width and length. Wave 3: confirm the winner with seeds, test
split for the final winner only; 4080 latency measured locally afterwards.

**Final-model rule (declared 2026-10-30 20:30 CET, before any 120k replication, 180k, 240k or
dose-at-120k result):** the Curiosity final model is the configuration with the lowest mean
339-icon validation pixel error over its seeds (the runs' own bf16 scores), among dropout 0.2 at
120k (seeds 1-4), 180k, 240k, dropout 0.25 and 0.3 at 120k, and 8 decoder layers at 120k. A
one-seed configuration qualifies only if it beats the 120k seed mean by more than the 120k seeds'
range; a tie goes to fewer steps. From that configuration, the seed with the lowest validation
error is re-scored in float32 (`python -m mojidiff.learning.render2svg_eval`) on validation and
once on the 339 test icons, greedy and best of 8, with GraphDecoder latency on the RTX 4080.

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

- Curiosity (`ssh curiosity`, account eai-hackathon, QoS eai-team11, user eai-jdkxz; the Axis user
  id changed at 17:45, `~/.ssh/config` updated). One-GPU jobs on dgx01/dgx03, job script
  ~/eai-team11/mojidiff/jobs/arms-8099e35.slurm. Wave 3b (53b3108, started 19:20): 13981
  cur-d02-s120k-seed2, 13982 cur-d02-s120k-seed3, 13983 cur-d02-dec8-s120k, 13984 cur-d02-s240k.
  Wave 4a (3681e08, started 20:17): 13986 cur-d02-s120k-seed4, 13987 cur-d02-s180k, 13988
  cur-d025-s120k, 13989 cur-d03-s120k. Wave 3 60k arms recorded (13974/13975/13979/13980
  recorded as cancelled). Jobs sharing a checkout upload each other's partial run dirs: take
  each run's record from the job that ran it (the copy with stdout.log). Project folder
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

Wave 2 recorded (findings 2026-10-30): dropout 0.2 at 120k steps reached 0.0654, ahead of every v9
run. Score wave 3 the same way; the final winner alone goes to the test split and to 4080 latency.
