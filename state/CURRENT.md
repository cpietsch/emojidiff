# Current research state

Updated: 2026-09-27 (evening)

## Current hypothesis and evidence

The first phase (2026-08-23 to 2026-09-24, Gates A to O) established a render-safe
typed SVG codec that round-trips all 4,006 curated OpenMoji icons, fixed hashed
family-disjoint splits (P32/T128 bucket: 2,681 train / 339 val / 339 test), a paired
render evaluation with bootstrap intervals, and a run registry. Its full narrative is
`reports/findings.md`; the last long-form handoff is `reports/state-history-2026-09-24.md`.

What that phase showed about models, in one place:

- From scratch on 2,681 icons, neither a categorical denoiser nor a KV-cached
  autoregressive model learns shape. The AR model reaches 0.92 of a zero-parameter
  position-marginal floor (criterion was 0.50); 4x data and 9x capacity do not help.
- The v16 denoiser (525k params) is a fixed-topology geometry repairer: it improves 26
  of 32 lightly corrupted icons but never predicts topology or style. Its inference
  latency was never measured.
- Conditioning on a render works where a label does not: OmniSVG 4B reconstructs a
  held-out icon from its 72 px render top-1 in 39/64 zero-shot, at 6.5 to 106 s per
  icon. Pixel edits survive re-vectorisation 71% of the time.
- Fine-tuning 2B to 4B pretrained priors on OpenMoji (Gates M, N, O) never beat the
  no-model baselines and is 40 to 100 s per icon, the opposite of the deliverable.
- No multi-step sampler from noise was ever run at corpus scale. No latency or VRAM
  benchmark was ever recorded; `reports/benchmarks/` is empty.

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

v7 (`r2s-full-v7-systems-42762c2-e9842024-47646604`) is the current model: greedy it
beats retrieval (+0.013 [+0.009, +0.017]) at 381 ms per icon median on an idle RTX 4080
(9.0M parameters, CUDA-graph decoder, float32). Its CLIP criterion passes only in the
float32 re-score, so the run is recorded as falsified on criterion 2. Best-of-8 clears
every quality bar including OmniSVG zero-shot (0.61); its graph-based latency is queued.

Other directions (findings, 2026-09-28):

- 2, wider data: Twemoji fits the codec at 66%; 2,211 icons kept after excluding
  held-out concepts. v8 (v7 + Twemoji) is training.
- 3, distillation: OmniSVG draws more faithfully, but 62-86% of its output misses the
  codec, at 40-250x the student's time. Set aside for render-to-SVG.
- 4, latent model: latent-v1 samples 32 distinct emoji-like programs but cannot
  reconstruct (KL about 16 nats per icon). latent-v2 with less KL pressure is queued.

## Last completed action and verification

2026-09-28: v5 recorded; idle-GPU latency measured for v3 and v5; compositional training
icons, whole-program masks, reranked decoding and a uniform float32 evaluator added.
Verified by `tests/test_render2svg.py`, `tests/test_fast_decode.py` (graph decoder equals
the reference in float32 for all three model variants), ruff, strict mypy.

## Active jobs

In tmux on gpubox-4080; logs under `/home/dev/.cache/mojidiff/`, each ending in `EXIT=`:

1. `r2s-chain4`: `full-v8-twemoji.yaml` (v7 plus 2,211 Twemoji icons, 60,000 steps).
2. `r2s-chain5`: after 1, idle best-of-8 and best-of-4 latency for v7, then v8's
   float32 and best-of-8 evaluations (`latency.log`, `eval-v8-*.log`).
3. `r2s-chain6`: after 2, `configs/latent/latent-v2-kl.yaml`.
4. `vectorise`: the demo on http://100.69.189.78:8790/ serving v7 (greedy or best of 8).
5. `weblog`: http://100.69.189.78:8787/.

## Artifact durability

- Checkpoints: `data/processed/openmoji-g1-geometric-gate-v16/checkpoint.zip` (v16
  repairer, 6.2 MB) and `data/processed/masked-span-l21-specialist-4-long/checkpoint.zip`
  (Gate L arm 21), both git-ignored, on the persistent volume.
- HF weights for OmniSVG 1.1 and Qwen3.5 under `/home/dev/.cache/huggingface` (~34 GB).
- Run records under `runs/`, registry in `state/runs.jsonl`, all committed.

## Current blockers

None.

## Next smallest evidence-producing action

Compare v8 with v7 per icon (paired). Choose the final configuration, then score it once
on the untouched test split, greedy and best of 8. Then: lower palette permutation (v7's
remaining errors are mostly colour), and read latent-v2.
