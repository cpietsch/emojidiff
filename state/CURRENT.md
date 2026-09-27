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

Render-to-SVG evidence so far:

- Four-icon overfit (`r2s-overfit4-940f5d3-f7306d2b-47646604`): all four programs
  reproduced exactly from their renders by step 100. 8.9M parameters, 148 ms per icon
  at batch 1 on the RTX 4080, 97 decoder calls per icon, 0.63 GiB peak VRAM.
- Full corpus without augmentation (`r2s-full-v1-940f5d3-1974cf82-47646604`): falsified.
  Memorises; pixel error 0.149 against 0.090 for the nearest training icon; CLIP top-1
  0.22 against 0.61 for OmniSVG zero-shot; 629 ms per icon.
- 16 cached exact variants per icon (`r2s-full-v2-aug-4032fbd-1e601024-47646604`):
  falsified, but better. Pixel error 0.130; beats the nearest icon on 50 of 339;
  draws the first, largest shape right and loses later paths; 828 ms per icon.
- Metric coordinates (`r2s-full-v3-metric`, running): ahead of v2 at every step so far
  (step 6,000: held-out accuracy 0.48 against 0.35, pixel error 0.113 on 64 icons).

## Last completed action and verification

2026-09-27: render-to-SVG model, exact augmentation (mirror, translation, palette
permutation; 42,896 variants cached), metric-coordinate option, and the re-vectorise
demo (`scripts/serve_vectorise.py`, port 8790) written and committed. Verified by
`tests/test_render2svg.py` (fast decoding equals a full-prefix decode; augmentation is
pixel-exact and preserves the masks the loss reads; cached metric decoding equals a
full forward), `tests/test_vectorise.py`, ruff, strict mypy.

## Active jobs

Chained in tmux on gpubox-4080, each writing `EXIT=` to its log under
`/home/dev/.cache/mojidiff/`:

1. `r2s-v3`: `configs/render2svg/full-v3-metric.yaml`, log `r2s-full-v3-metric.log`.
2. `r2s-v4`: waits for 1, then `full-v4-online.yaml` (online augmentation, one change
   from v2), log `r2s-full-v4-online.log`.
3. `r2s-v5`: waits for 2, then `full-v5-path.yaml` (path-major order, one change from
   v3), log `r2s-full-v5-path.log`.

Each run registers itself in `state/runs.jsonl` and writes `runs/<run_id>/`.

## Artifact durability

- Checkpoints: `data/processed/openmoji-g1-geometric-gate-v16/checkpoint.zip` (v16
  repairer, 6.2 MB) and `data/processed/masked-span-l21-specialist-4-long/checkpoint.zip`
  (Gate L arm 21), both git-ignored, on the persistent volume.
- HF weights for OmniSVG 1.1 and Qwen3.5 under `/home/dev/.cache/huggingface` (~34 GB).
- Run records under `runs/`, registry in `state/runs.jsonl`, all committed.

## Current blockers

None.

## Next smallest evidence-producing action

Read v2 and v3 against the nearest-training-icon baseline and Gate N's CLIP top-1.
If either passes, serve its best checkpoint in the re-vectorise demo and measure it on
the untouched test split. If neither lifts held-out accuracy, the next factor is the
encoder: a stride-4 grid or a pretrained image backbone.
