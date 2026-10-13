# Current research state

Updated: 2026-10-13

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

2026-10-13: the operator resumed MojiDiff for training on the MUSICA cluster (ASC; H100 nodes,
about 80 credits approved), where it runs under the codename "predictable": checkout
`$DATA/predictable/repo`, checkpoints in `$DATA/predictable/cache/runs`, job scripts and the
record fetch helper in the operator's private cluster repo (`jobs/predictable_*.slurm`,
`scripts/predictable_fetch.sh`). Records come back here and are committed from this machine.
Smoke runs passed on the RTX 4080 and on one H100 node (about 14.5 steps/s per H100 against
10.7 here; not loader-bound; same corpus file, dataset hash 47646604; Cairo 1.18.2 there).

- Slurm 1822825 (queued): v9 rerun (`full-v9-colour.yaml`, the sweep's paired parent),
  `v10-w384`, `v10-w512`, `v10-dec12`, one per H100 on one node, 4 h limit.

The local servers stay stopped; this machine's GPU serves other work, short smoke tests allowed.

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

Operator plan of 2026-10-13 (training on MUSICA): (1) the scaling sweep, one change from v9 at a
time (width, depth, steps 120k/240k, batch 128, image size 192), each scored paired against the
v9 rerun on the same machine; (2) seeds for the top two and one test-split score for the final
choice, with RTX 4080 and browser latency, distilling into a small student if the winner is too
slow; (3) the canvas latent and a larger flow prior reopened on the winning backbone (the
operator lifted the stop rule for that). The browser milestones (findings 2026-10-04) wait.
