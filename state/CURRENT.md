# Current research state

Updated: 2026-10-04

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

Other directions (findings, 2026-09-28 to 2026-10-03): wider data (v8 + Twemoji, worse
by 0.015: fills against strokes); distillation from OmniSVG (62-86% misses the codec);
program latents (latent-v1/v2/v3: distinct but blobby); canvas latent vt-v1/vt-v2-c16
(stopped after two A1 failures); flow prior lfp-v1x (best sampler so far, exploratory).

## Last completed action and verification

2026-10-04: public release and browser milestone 0 (findings, 2026-10-04). The repo
`cpietsch/emojidiff` is public (CC BY-SA 4.0); the weblog deploys to
https://cpietsch.github.io/emojidiff/ from GitHub Actions (first deploy checked: pages
return 200). Hugging Face Docker Spaces need a paid PRO account, so none was created;
the operator chose in-browser inference. Milestone 0, replaying 32 recorded v9 greedy
traces in headless Chrome on the 4080: WASM 4 threads 881 ms per icon median (about
1.2 s projected over the validation set), WebGPU 1,731 ms, server 512 ms; argmax 100%
for the float32-id export (the int64 export is wrong on WebGPU JSEP: 75.9%). An
independent re-run matched within 2%. Full suite 321 passed, ruff and strict mypy clean
at the release commit.

## Active jobs

None. 2026-10-10: the operator paused MojiDiff and reassigned gpubox-4080's compute to
other work; the gallery (:8791), vectorise (:8790) and weblog (:8787) servers were
stopped. Do not restart them or launch GPU jobs until the operator resumes the project.
The public weblog on Pages is unaffected.

## Artifact durability

- Checkpoints: `data/processed/openmoji-g1-geometric-gate-v16/checkpoint.zip` (v16
  repairer, 6.2 MB) and `data/processed/masked-span-l21-specialist-4-long/checkpoint.zip`
  (Gate L arm 21), both git-ignored, on the persistent volume.
- HF weights for OmniSVG 1.1 and Qwen3.5 under `/home/dev/.cache/huggingface` (~34 GB).
- Run records under `runs/`, registry in `state/runs.jsonl`, all committed.
- Off-machine copy (2026-10-04): every runs/*.pt and data/processed/*/checkpoint.zip
  mirrored to https://huggingface.co/chrispie/mojidiff-checkpoints (public, CC BY-SA 4.0);
  sha256 index `reports/checkpoints-hf.json`, verified against the remote. Code: GitHub
  `cpietsch/emojidiff` (public).
- Browser probe (exports, Chrome harness, traces): `/home/dev/.cache/mojidiff/
  browser-probe-2026-10-04/`; compact results in `reports/browser/m0/`.

## Current blockers

Operator to confirm: the go/no-go thresholds (greedy median 3 s go, 3-6 s
click-to-vectorise, above 6 s stop), best of 8 on WebGPU only, ONNX files uploaded to
the public checkpoints repo under `onnx/`, and test-split icons labelled on the page.

## Next smallest evidence-producing action

Operator direction 2026-10-04: public demos run in the visitor's browser (ONNX Runtime
Web), hosted on Pages; latent models stay the research focus (2026-09-29). Plan
(milestones 1-8, in findings 2026-10-04 and `reports/browser/m0/README.md`): next is
milestone 1, `scripts/export_onnx.py` with wrappers (plain attention, float32 ids,
folded head table, float Fourier scales, LayerNorm bias patch for the flow DiT) and
pytest parity (v9 greedy 10 of 10 identical, vt 4 of 4, flow 50 steps within 1e-4).
Then the JS core with Node tests, v9 greedy end to end, best of 8, the vectorise page,
the gallery (flow prior samples and interpolation), Pages integration.
