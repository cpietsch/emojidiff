# MojiDiff

MojiDiff trains small models from scratch on one RTX 4080 to write OpenMoji-style
emoji as editable SVG programs. The programs use a typed, render-safe codec. Coding
agents run the project under a written operating contract (`AGENTS.md`), and every
result is published in a dated record, including the ones that failed.

The GitHub repository is named `emojidiff`; the Python package is `mojidiff`.

- Weblog (findings, runs, gallery snapshots): https://cpietsch.github.io/emojidiff/
- Findings, the single dated narrative: [`reports/findings.md`](reports/findings.md)
- Checkpoints, all of them: https://huggingface.co/chrispie/mojidiff-checkpoints
  (sha256 index in [`reports/checkpoints-hf.json`](reports/checkpoints-hf.json))
- Demos: in-browser versions of the model gallery and of edit pixels, re-vectorise are
  being built (the models run in the visitor's browser, served from Pages). Until then
  the demos run on the project's own machine only. `space/` and
  `scripts/build_space.py` also package them as Hugging Face Docker Spaces, which need
  a PRO account on the free CPU tier; in a container limited to 2 CPUs, v9 greedy took
  a median of about 730 ms per icon on the first 40 held-out icons the page lists,
  against 381 ms for v7 on the RTX 4080 (below).

## What it does

The goal is a model that is small and fast, that draws recognisable emoji, and that
writes SVG a person can edit. Every icon is a typed program: paths, styles, and metric
coordinates on a fixed grid. The codec round-trips all 4,006 curated OpenMoji 17.0.0
icons. The splits are fixed, hashed, and family-disjoint; the bucket the current models
train on (at most 32 paths and 128 segments per icon, 3,359 of the 4,006 icons) has
2,681 training, 339 validation, and 339 test icons.

Two lines of work produced models that are worth keeping:

- **Render to SVG.** A 9.0M-parameter transcriber takes a rendered emoji and writes
  its SVG program. This is the main deliverable: it vectorises an image you have
  edited.
- **Latent models.** Variational models over programs or over the transcriber's
  image latent, plus a rectified-flow prior. They are judged on prior samples,
  novelty, and interpolation.

## Results so far

Pixel error is the mean absolute error between renders at 72 px, so lower is better.
The baseline is retrieval: copy the nearest training icon. Full tables, intervals, and
run ids are in `reports/findings.md`.

| model | what it is | result |
| --- | --- | --- |
| v7 (`r2s-full-v7-systems-*`) | render-to-SVG transcriber, 9.0M params | Test split: 0.071 greedy, 0.051 best of 8. Retrieval scores 0.087. Greedy beats retrieval on 244 of 339 icons. |
| v7 latency | idle RTX 4080, CUDA-graph decoder, float32, batch 1, 16 validation icons | Median 381 ms per icon greedy, 753 ms best of 4, 1,154 ms best of 8 (rendering every candidate included) |
| v9 (`r2s-full-v9-colour-*`) | v7 with less palette permutation | Validation 0.072 vs v7's 0.076 (paired, interval clear of zero); the colour errors it targeted are gone from the sample sheet, though some glyphs still fail. Serves the vectorise demo. |
| v7 edit test (Gate N) | edit pixels, re-vectorise | Edits reflected: 0.74 best of 8, vs 0.70 for OmniSVG 4B (93 edits on 32 validation icons) |
| vt-v1, vt-v2-c16 | variational canvas latent on v9 | Reconstruct at 0.083 and 0.081. Retrieval with mirrors scores 0.085, and the paired interval of that gap includes zero, so both fail their declared criterion A1. The canvas line stopped by its declared rule. |
| lfp-v1x (exploratory) | flow prior over vt-v1's latent | Best latent sampler so far (fragments 0.18, CLIP recall 0.42, no copies), but it misses its declared thresholds. About 1 s per sample. |

What did not work, briefly:

- The first phase (2026-08-23 to 2026-09-24) never beat its no-model baselines. That
  phase tried label-conditioned generators, a categorical denoiser, and 2B to 4B
  fine-tunes.
- Adding 2,211 Twemoji icons (v8) made OpenMoji transcription worse by 0.015. Twemoji
  draws outlines as filled shapes, and OpenMoji draws them as strokes.
- Distillation from OmniSVG was set aside after a probe: 62 to 86% of its drawings do
  not fit the codec.
- Single-vector program latents (latent-v1/v2/v3) sample distinct programs, but their
  reconstructions are blobby or near blank.
- v7 misses CLIP top-1 against OmniSVG zero-shot on Gate N's 32 icons in its own
  bfloat16 evaluation (0.47 against 0.609). It clears that bar only in a float32
  re-score (0.63).

For the current direction and the next step, see [`state/CURRENT.md`](state/CURRENT.md).

## Repository map

| path | contents |
| --- | --- |
| `src/mojidiff/representation` | the typed SVG codec, normaliser, safe renderer |
| `src/mojidiff/curation` | OpenMoji acquisition, audit, curation manifests, external-set probes |
| `src/mojidiff/learning` | models and studies: `render2svg`, `fast_decode` (CUDA-graph decoder), `latent`, `pixel_latent`, `latent_flow`, `latent_metrics`, `telemetry`, first-phase models |
| `src/mojidiff/orchestration` | run registry, run contract, local worker |
| `src/mojidiff/gallery`, `vectorise`, `demo`, `kitbash` | small web servers for the browser demos |
| `src/mojidiff/weblog` | static site generator for the weblog |
| `configs/` | versioned configs: `data`, `codec`, `learning`, `render2svg`, `latent`, `gallery` |
| `data/manifests/` | versioned curation manifests and fixtures (raw data and processed outputs are git-ignored) |
| `runs/` | one directory per run: `run.yaml`, `metrics.jsonl`, `stdout.log`, `artifacts.json`, `result.md` |
| `state/` | `CURRENT.md` (current direction), `runs.jsonl` (append-only run registry), `gates.yaml` (first-phase gates) |
| `reports/` | `findings.md`, per-study reports, latent harness reports, checkpoint index |
| `scripts/` | servers (`serve_*.py`), `audit_run_records.py`, probes (`scripts/remote/` is first-phase worker tooling, kept for history) |
| `infra/` | the pinned NVIDIA PyTorch container image used for the GPU smoke test |
| `.github/workflows/` | the GitHub Pages build of the weblog |
| `docs/` | codec representation, corruption contracts, data-curation reproduction |
| `tests/` | pytest suite |
| `PROJECT_PLAN.md`, `DATA_CURATION.md` | reference from the first phase: codec, corpus, and what failed |

## Quickstart

You need Python 3.12, [`uv`](https://docs.astral.sh/uv/), git, and Cairo for
rendering. Most tests read the OpenMoji checkout, so acquire it before running them
(see [Data and checkpoints](#data-and-checkpoints)):

```bash
git clone https://github.com/cpietsch/emojidiff.git mojidiff && cd mojidiff
sudo apt-get install --no-install-recommends libcairo2
uv sync --extra dev --extra model-cpu   # model-cpu adds CPU PyTorch 2.8.0
uv run mojidiff-curate acquire --config configs/data/openmoji-17.0.0.yaml
uv run pytest                           # tests that need CUDA are skipped on CPU
uv run ruff check src tests
uv run mypy                             # strict, as configured in pyproject.toml
```

Cloning into a directory named `mojidiff` matters: one orchestration test expects the
repository path to end in `/mojidiff`. On a fresh clone (CPU, 2026-10-04) the suite
still has known environment failures outside that: the gallery and vectorise tests need
the icon cache in `data/processed/kitbash` (see Local demos), and one OmniSVG
tokenizer test needs an OmniSVG checkout at `/home/dev/workspace/external/OmniSVG`.

On the RTX 4080 machine, the environment runs inside an NVIDIA PyTorch container. The
`.venv` there is created with system site packages, so it uses the container's CUDA
PyTorch rather than the `model-cpu` extra.

**Cache path.** Checkpoints and derived caches live under `CACHE_ROOT`, which is
hard-coded to `/home/dev/.cache/mojidiff` in `src/mojidiff/learning/render2svg.py`.
On any other machine, either create that directory or edit the constant before
training, evaluating, or serving models.

**Weblog.** Build the static site into `site/` (git-ignored):

```bash
uv run python -m mojidiff.weblog.build --root . --out site
```

The public copy on GitHub Pages is built by `.github/workflows/pages.yml` on every
push to `main`, with `--publication pages`.

`scripts/serve_weblog.py --rebuild --host 127.0.0.1` rebuilds the site and serves it
on port 8787.

**Local demos.** Like the weblog server, these servers bind one explicit address.
Their default is a machine-specific private address, so pass `--host` yourself:

```bash
uv run python scripts/serve_gallery.py --host 127.0.0.1           # port 8791
uv run python scripts/serve_vectorise.py --host 127.0.0.1 \
    --checkpoint /home/dev/.cache/mojidiff/runs/<run_id>/best.pt  # port 8790
```

The gallery loads every model listed in `configs/gallery/models.yaml` from
`CACHE_ROOT/runs/<run_id>/best.pt`. If any checkpoint is missing, it stops
before it binds. Both servers also need the OpenMoji checkout (see below) and read
their icons from a cache in `data/processed/kitbash` (git-ignored), which
`python -m mojidiff.vectorise.icon_cache` builds from that checkout.

## Data and checkpoints

**OpenMoji.** The corpus is OpenMoji 17.0.0 at commit
`f9fc506a3f913be9897ab0181d611d4c910a4104`. The checkout is not part of this
repository. It is cloned into `data/raw/openmoji/17.0.0` (pinned in
`configs/data/openmoji-17.0.0.yaml`), hash-verified, and never modified:

```bash
uv run mojidiff-curate acquire --config configs/data/openmoji-17.0.0.yaml
```

To reproduce the audit and curation layers, see
[`docs/data-curation.md`](docs/data-curation.md). The reviewed manifests are versioned
in `data/manifests/`.

**Twemoji.** Only the v8 arm uses Twemoji. It reads a checkout of
[jdecked/twemoji](https://github.com/jdecked/twemoji) under `data/raw/external/twemoji`
(tag v17.0.3 on the training machine).

**Checkpoints.** All run checkpoints are on Hugging Face, at the same `runs/<run_id>/`
layout the code expects under `CACHE_ROOT`. For example, for v9:

```bash
uvx --from huggingface_hub hf download chrispie/mojidiff-checkpoints \
    --include 'runs/r2s-full-v9-colour-621bc8e-b26ad95e-47646604/*' \
    --local-dir /home/dev/.cache/mojidiff
```

`best.pt` is the checkpoint chosen by the run's declared rule, and `latest.pt` is the
resumable end state. First-phase models are under `legacy/`.

## How the project is run

[`AGENTS.md`](AGENTS.md) is the operating contract that the agents follow. The rules
that matter most:

- Every training or sampling run has a stable `run_id`: a slug, plus short hashes of
  the code, config, and data. Each run has a directory under `runs/`.
- Each `run.yaml` records the commit, config hash, dataset manifest hash, and seed.
  From 2026-09-27 it also records a `resource` block (device, peak VRAM, training
  seconds, end-to-end milliseconds per icon) and a baseline measured on the same split.
- State changes are appended to `state/runs.jsonl` and never edited. Failed runs are
  kept under their own id.
- Criteria are declared before a run, and the outcome is recorded against them, even
  when the run fails.
- `scripts/audit_run_records.py` checks that the registry and the run records agree.
  Run it before committing run records.
- `state/CURRENT.md` holds the current direction in at most 100 lines.
  `reports/findings.md` holds one entry per material result: hypothesis, observation,
  numbers, decision.

## Origins

The project started from a handoff package for `gtc`, a Codex control-plane machine
that was meant to dispatch work to rented GPU workers. That plan was only a starting
point and an inspiration; the project has since evolved into something else. It now
runs directly on one owned RTX 4080, and its goal has narrowed to a small, fast model
with a measured latency. The handoff files are kept for history and are not current
instructions: `DEVBOX_BOOTSTRAP.md`, `START_PROMPT.md`, `hosts.example.yaml`,
`ssh_config.example`, `scripts/remote/`, and the compute sections of
`PROJECT_PLAN.md`.

## Licence and attribution

Everything in this repository is licensed under
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/): code, documentation,
reports, and the published checkpoints. The full text is in [`LICENSE`](LICENSE). This
matches the licence of the OpenMoji data the models are trained on.

- **OpenMoji**: All emojis designed by [OpenMoji](https://openmoji.org/) – the
  open-source emoji and icon project. License:
  [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). This project
  normalises the OpenMoji SVGs into its typed codec and renders them; the curated
  manifests, renders, and every model trained on them are adaptations of OpenMoji.
- **Twemoji**: the v8 checkpoint (`r2s-full-v8-twemoji-*`) was also trained on 2,211
  [Twemoji](https://github.com/jdecked/twemoji) graphics, licensed under
  [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). They were adapted:
  palette-snapped to OpenMoji's colours and converted into the codec.
- Third-party models: OmniSVG 1.1 (4B) and Qwen3.5 models (0.8B to 4B) were compared
  against and, in the first phase, fine-tuned with LoRA; CLIP ViT-B/32 was used to
  score results. None of their weights, and none of the fine-tuned adapters, are
  redistributed here or in the checkpoint repository. Their outputs appear in
  `reports/` only as evaluation records and sample sheets.
