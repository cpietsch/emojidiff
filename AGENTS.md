# MojiDiff Operating Contract

MojiDiff is an engineering project with one deliverable: **a small model, running fast
on the operator's RTX 4080, that produces recognisable OpenMoji-style emoji as editable
SVG programs, shown in a browser demo with a measured per-icon latency.** Every session
moves that deliverable forward or explains, in `reports/findings.md`, why it cannot.

Research rigour serves the deliverable, not the other way round. Negative results are
kept, but the question is always "does this get us a working, fast model?", not "is
this hypothesis true?".

If this file changes, restart the agent session before relying on it.

## Read order

1. `AGENTS.md`
2. `state/CURRENT.md` (at most 100 lines: direction, last result, active jobs, next step)
3. The last two or three entries of `reports/findings.md`

`PROJECT_PLAN.md` and `DATA_CURATION.md` are reference material from the first phase
(2026-08-23 to 2026-09-24, Gates A to O). Consult them for the codec, corpus, and the
record of what has already failed; do not treat their gates or hypotheses as the plan.

## Machine

`gpubox-4080` is the operator's own, persistent, trusted machine: RTX 4080 (16 GiB),
repository at `/home/dev/workspace/mojidiff`, large artifacts under `/home/dev/.cache`,
native CUDA PyTorch in `.venv`, `uv`, `tmux`, Docker with the NVIDIA runtime. Edit, run,
train, and iterate directly; no snapshotting or permission ceremony for bounded work.

If you launch a sibling container, mount the Compose-prefixed volumes
`code-server-gpu_gpubox-workspace` and `code-server-gpu_gpubox-home`. Any other
volume name silently creates an empty one.

Rented or shared machines are out of scope. Never rent, stop, resize, or bill anything.

## Data

- The upstream OpenMoji checkout is immutable. Exclusions live in the versioned
  manifest with reason codes; the `flags` group stays a reversible named split.
- Splits are fixed, hashed, and family-disjoint. Do not construct new ones casually.
- Any augmentation must be exactly invertible in token space or must record the
  transform it applied. Verify it with a test.

## Run contract

Every training or sampling run has a stable `run_id` (slug plus short code, config, and
data hashes) and a directory under `runs/` with `run.yaml`, `metrics.jsonl`,
`stdout.log`, `artifacts.json`, and `result.md`. Append state changes to
`state/runs.jsonl`; never edit history. Preserve failed runs under their own id.

`run.yaml` must record the commit, config and its hash, dataset manifest hash, seed,
and, for every completed run registered from 2026-09-27:

- a `resource` block written through `mojidiff.learning.telemetry`: device, peak VRAM,
  training seconds, and end-to-end inference milliseconds per icon (all steps, all
  positions, any projection). A null value needs a `*_null_reason`.
- a `baseline` or `baselines` block: the identity policy, a zero-parameter policy, or
  the parent run, measured on the same split with the same metric.

`scripts/audit_run_records.py` enforces both. Run it before committing run records.

Long runs go in `tmux` with a recorded job identity. Register before launch, record on
completion. Overnight chained runs are welcome as long as the morning finds recorded
outcomes, not open questions.

## Experimental discipline

- Cheap before expensive: deterministic fixture overfit, then a small corpus run, then
  the full corpus. Do not scale a model that has not beaten its baseline.
- Change one primary factor at a time. Report uncertainty across seeds for small gaps.
- Speed is a result. Never say "fast" without milliseconds per icon on the named GPU
  under the conditions the demo would use.
- Compare against the nearest honest alternative at matched size and data.
- Label raw corrupted state `x_t` and predicted clean state `x_hat_0`; label any
  projection or constrained decoding.

## Checks

Before committing a material code change: `ruff`, strict `mypy`, the narrow test, then
`pytest`. Say exactly what was tested. Code that runs only on a path with no test does
not count as done.

## Artifacts

Metadata, compact reports, representative renders, and demo assets stay in the
repository. Checkpoints and datasets stay under `/home/dev/.cache`. Keep best, latest
recoverable, and instructive failure checkpoints. Never delete artifacts or run records
without asking.

## Record

- `state/CURRENT.md`: at most 100 lines. Current direction and the evidence for it,
  last completed action and how it was verified, active jobs, blockers, next smallest
  step. Rewrite it; do not append to it.
- `reports/findings.md`: the one dated narrative. Each material result gets one entry
  with hypothesis, observation, numbers, and decision. Nothing is duplicated into
  `state/CURRENT.md` or `PROJECT_PLAN.md`.
- The weblog (`scripts/serve_weblog.py`, port 8787) is rebuilt after every material
  result. The operator reads it, not the terminal.

## Stop and ask

Stop for anything that costs money, deletes or overwrites data, needs credentials that
are missing, or changes the deliverable itself. Ordinary edits, tests, and bounded runs
on `gpubox-4080` proceed without asking.
