# Current research state

Updated: 2026-09-27

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

The deliverable is now stated in `AGENTS.md`: a small fast model with a demo and a
measured per-icon latency. The direction to reach it is not yet chosen. The candidates
on the table, from the 2026-09-27 review, are: (1) a small from-scratch render-to-SVG
model over the existing codec with affine and colour augmentation; (2) widening the
corpus with Noto, Twemoji, Blobmoji and CC0 icon sets; (3) distilling the big prior into
a small student; (4) a continuous latent (DeepSVG-style) model; (5) shipping the v16
repairer inside the kitbash tool. The recommendation is (1), later adding (2).

## Last completed action and verification

2026-09-27: process reset. `AGENTS.md` rewritten around the deliverable (one page);
this note cut to under 100 lines; gates and predeclared criteria dropped from the
contract; `mojidiff.learning.telemetry` added for per-icon latency and peak VRAM;
`mojidiff.orchestration.contract` and `scripts/audit_run_records.py` now require a
`resource` block and a baseline on every completed run registered from 2026-09-27.
Verified: `ruff`, strict `mypy`, `tests/test_telemetry.py`, `tests/test_contract.py`,
full `pytest`, and the audit script against the 59 existing records (consistent).

## Active jobs

None. No tmux sessions or containers are running project work.

## Artifact durability

- Checkpoints: `data/processed/openmoji-g1-geometric-gate-v16/checkpoint.zip` (v16
  repairer, 6.2 MB) and `data/processed/masked-span-l21-specialist-4-long/checkpoint.zip`
  (Gate L arm 21), both git-ignored, on the persistent volume.
- HF weights for OmniSVG 1.1 and Qwen3.5 under `/home/dev/.cache/huggingface` (~34 GB).
- Run records under `runs/`, registry in `state/runs.jsonl`, all committed.

## Current blockers

Direction not chosen by the operator. Nothing else blocks work on `gpubox-4080`.

## Next smallest evidence-producing action

Once a direction is chosen: a four-icon fixture overfit of the chosen model that writes a
`resource` block (latency per icon, peak VRAM) and an identity baseline through the new
contract, so the first run of the new phase already answers "how fast".
