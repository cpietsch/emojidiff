# Current research state

Updated: 2026-09-29

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

2026-09-28: v7 scored once on the untouched test split (predeclared rule); v8 Twemoji,
latent-v1/v2, external probe v1/v2, distillation and edit probes recorded in findings.
Verified by the tests in `tests/test_render2svg.py`, `test_fast_decode.py`,
`test_latent.py`, `test_vectorise.py`, ruff, strict mypy, and the run-record audit.

## Active jobs

In tmux on gpubox-4080; logs under `/home/dev/.cache/mojidiff/`, each ending in `EXIT=`:

1. `lfp`: `configs/latent/lfp-v1x-exploratory.yaml`, the flow prior over vt-v1 (declared
   exploratory: vt-v1 failed A1), log `lfp-v1x.log`.
2. `vt-v2`: waits for 1, then `configs/latent/vt-v2-c16.yaml` (the declared c = 16
   rerun), log `vt-v2-c16.log`.
3. `gallery`: http://100.69.189.78:8791/ - 13 models; blind Compare ratings append to
   `reports/gallery/ratings.jsonl`. 4. `vectorise`: :8790 (v9). 5. `weblog`: :8787.

After 1: add the gallery entry (backend canvas-flow), restart the gallery, score it with
`latent_metrics` (samples, and `--interpolation backend` for slerp).

## Artifact durability

- Checkpoints: `data/processed/openmoji-g1-geometric-gate-v16/checkpoint.zip` (v16
  repairer, 6.2 MB) and `data/processed/masked-span-l21-specialist-4-long/checkpoint.zip`
  (Gate L arm 21), both git-ignored, on the persistent volume.
- HF weights for OmniSVG 1.1 and Qwen3.5 under `/home/dev/.cache/huggingface` (~34 GB).
- Run records under `runs/`, registry in `state/runs.jsonl`, all committed.

## Current blockers

None.

## Next smallest evidence-producing action

Direction set by the operator on 2026-09-29, after browsing every model in the gallery:
focus on the latent-model approach. They like the interpolations and prior samples most
("you can see that the model learned how to draw even if it is very bad"). Latent work
is judged by prior-sample quality, novelty and interpolation smoothness, shown in the
gallery's latent tab, not by reconstruction error alone. A design panel is choosing the
first arms and a metrics harness; latent-v1 and latent-v2 are the baselines.
