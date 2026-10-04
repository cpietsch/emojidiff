# Current research state

Updated: 2026-10-03

## Current hypothesis and evidence

The first phase (2026-08-23 to 2026-09-24, Gates A to O) established a render-safe
typed SVG codec that round-trips all 4,006 curated OpenMoji icons, fixed hashed
family-disjoint splits (P32/T128 bucket: 2,681 train / 339 val / 339 test), a paired
render evaluation with bootstrap intervals, and a run registry. Its full narrative is
`reports/findings.md`; the last long-form handoff is `reports/state-history-2026-09-24.md`.

That phase's models (label-conditioned generators, a denoiser, 2B-4B fine-tunes) never
beat their no-model baselines and never measured latency; see findings through Gate O.

The deliverable is stated in `AGENTS.md`: a small fast model with a demo and a measured
per-icon latency. The operator chose the direction on 2026-09-27: a small from-scratch
render-to-SVG transcriber over the existing codec (`mojidiff.learning.render2svg`),
with wider vector data, distillation, and a continuous latent model queued behind it.

Render-to-SVG evidence (339 validation icons; nearest training icon scores pixel error
0.090; full tables in `reports/findings.md`):

| run | change | pixel error | beats nearest | CLIP top-1 |
| --- | --- | --- | --- | --- |
| v1 | none | 0.149 | 16 | 0.22 |
| v2 / v4 | cached / online exact variants | 0.130 / 0.121 | 50 / 62 | 0.16 / 0.25 |
| v3 | metric coordinates | 0.106 | 138 | 0.22 |
| v5 | + path-major order | 0.098 | 153 | 0.44 |
| v6 | compositions (vs v4) | 0.111 | 96 | 0.31 |
| v7 | all of the above, 60k steps | 0.077 | 225 | 0.47 bf16 / 0.63 fp32 |
| v7 best of 8 | render-and-compare decoding | 0.057 | 298 | 0.66 |
| v9 | palette permutation 0.1 (vs v7) | 0.072 | 227 | 0.53 bf16 |

v7 (`r2s-full-v7-systems-42762c2-e9842024-47646604`, 9.0M parameters) is the final model
by the rule declared before v8's result. On the untouched test split it beats retrieval
greedy (pixel error 0.071 against 0.087, 244 of 339 icons) and best of 8 (0.051, 286 of
339). Idle RTX 4080, float32, rendering included for best-of: greedy 381 ms per icon
median, best of 4 753 ms, best of 8 1,154 ms. It keeps pixel edits as often as OmniSVG
4B on Gate N's edit test (best of 8: 0.74 reflected against 0.70). CLIP top-1 on Gate
N's 32 icons clears OmniSVG zero-shot (0.609) in float32 (greedy 0.63, best of 8 0.66)
but not in the run's own bfloat16 evaluation (0.47), so v7 is recorded as falsified on
that criterion.

Other directions (findings, 2026-09-28):

- 2, wider data: adapter v2 fits Twemoji 67%, Noto 27%, Blobmoji 33% as the codec
  stands. v8 (v7 + 2,211 Twemoji icons) is worse by 0.015 paired: Twemoji writes
  outlines as filled shapes, OpenMoji as strokes; mixing conventions blind hurts.
- 3, distillation: OmniSVG draws more faithfully, but 62-86% of its output misses the
  codec, at 40-250x the student's time. Set aside for render-to-SVG.
- 4, latent model: latent-v2 (beta 0.1) passes its criteria - own latent beats the
  prior-mean decode by 0.013, 32 of 32 samples distinct - but reconstructs near blank
  (0.164). Interpolations decode valid programs that jump between emoji modes. Next test
  for this direction: per-path latents (DeepSVG-style), not a bigger single latent.

## Last completed action and verification

2026-10-03, recovery after a power failure. The container came back from its base image:
volumes intact, apt packages lost. `libcairo2` (renderer; see `docs/data-curation.md`)
and `iproute2` reinstalled; `cairosvg` imports. Both jobs of 2026-09-29 had finished
before the outage (vt-v2-c16 `EXIT=0`, harness `HARNESS-CANVAS-DONE`, every step
`EXIT=0`); checkpoints present; `scripts/audit_run_records.py` consistent. Their results
are recorded in findings (2026-10-03): vt-v2-c16 fails A1 again, so the canvas line stops
as declared; lfp-v1x-exploratory fails B1, B2, B4, B5 (B3 passes), though it is the best
latent sampler so far (fragments 0.18, CLIP recall 0.42).

## Active jobs

No training or scoring jobs. Services in tmux on gpubox-4080, restarted 2026-10-03:
`weblog` :8787, `vectorise` :8790 (v9), `gallery` :8791 (15 models; logs under
`/home/dev/.cache/mojidiff/`).

## Artifact durability

- Checkpoints: `data/processed/openmoji-g1-geometric-gate-v16/checkpoint.zip` (v16
  repairer, 6.2 MB) and `data/processed/masked-span-l21-specialist-4-long/checkpoint.zip`
  (Gate L arm 21), both git-ignored, on the persistent volume.
- HF weights for OmniSVG 1.1 and Qwen3.5 under `/home/dev/.cache/huggingface` (~34 GB).
- Run records under `runs/`, registry in `state/runs.jsonl`, all committed.
- Off-machine copy (2026-10-04): every runs/*.pt and data/processed/*/checkpoint.zip
  mirrored to https://huggingface.co/chrispie/mojidiff-checkpoints (public, CC BY-SA 4.0);
  sha256 index `reports/checkpoints-hf.json`, verified against the remote. Code: GitHub
  `cpietsch/emojidiff` (private).

## Current blockers

Direction after the canvas line: the operator's call (see next action).

## Next smallest evidence-producing action

Operator focus since 2026-09-29: latent models, judged by prior samples, novelty and
interpolation in the gallery. The canvas line (vt-v1, vt-v2-c16) stopped by its declared
rule; the program-latent line (latent-v1/v2/v3-blind) ended at "full-size but blobby".
Candidate next arms, one to be chosen:

1. A larger flow prior on vt-v2-c16, explicitly exploratory (the visually best sampler
   so far; its gate is gone, so outcomes are gallery evidence, not a pass).
2. Per-path latents (DeepSVG-style), the program line's queued next test.
3. Return to the render-to-SVG deliverable (v7/v9) and the demo.
