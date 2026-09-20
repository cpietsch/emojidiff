# MojiDiff Operating Contract

This repository is an exploratory research project for a small categorical diffusion
model over editable SVG programs.

Development and execution both happen on `gpubox-4080`, the operator's own machine. The
canonical Git worktree, run registry, reports, data, and GPU all live there. There is no
control-plane/worker split for ordinary work.

If this file changes, restart the agent session before relying on the new instructions.

## Read order

At the start of every session, read these files completely in this order:

1. `AGENTS.md`
2. `PROJECT_PLAN.md`
3. `DATA_CURATION.md`
4. `state/CURRENT.md`, if present
5. `state/hosts.local.yaml`, if present
6. The most recent relevant report under `reports/`

## Operating objective

Run this as an evidence-driven experiment, not a deadline-driven build. Advance from
cheap tests to expensive tests only when the preceding evidence justifies it. Keep
negative results. Do not force the research to support a predetermined claim.

The working hypothesis is that a typed, render-safe SVG program plus structure-aware
categorical corruption can support useful generation, editing, and visually meaningful
denoising trajectories. The comparison against a cached autoregressive model must be
honest and hardware-specific.

## Machine

`gpubox-4080` is an owned, persistent, trusted machine:

- NVIDIA GeForce RTX 4080, 16,376 MiB, driver 595.71.05;
- repository at `/home/dev/workspace/mojidiff` on a persistent volume;
- durable artifacts under `/home/dev/.cache`, also persistent;
- native CUDA PyTorch in `.venv`, which inherits the system NGC torch through
  `--system-site-packages`;
- passwordless `sudo`, Docker with the NVIDIA runtime, `uv`, `tmux`.

Ordinary development, testing, training, evaluation, and artifact writing on this
machine need no ceremony. Edit, run, and iterate directly. Do not stage an immutable
snapshot or ask permission to use the GPU for a bounded experiment.

It is a devbox that drives **sibling** Docker containers, not children. If you launch a
container, mount the Compose-prefixed volumes `code-server-gpu_gpubox-workspace` and
`code-server-gpu_gpubox-home`; an unprefixed name silently creates a new empty volume,
and devbox-local bind paths do not resolve inside siblings.

## Other machines

Rented or shared machines are a different trust class and keep the stricter rules.
`state/hosts.local.yaml` remains the machine-readable authorization record for them.

- Never rent, stop, destroy, resize, or change the billing cap of a billed instance.
- A billed worker needs `enabled: true` plus a complete `resource_cap` before any
  compute. For Vast that means `max_steps`, `max_spend_usd`, and a storage bound; for
  the cluster, `max_jobs` and `max_steps_per_job`.
- Stop before any action that would create or raise external cost, and ask for a revised
  cap rather than inferring one.
- Use named aliases from `~/.ssh/config` with `BatchMode yes` and `ForwardAgent no`.
  Never use `StrictHostKeyChecking no`; treat a changed host key as a blocker.
- Keep credentials out of the repository, logs, and command output. Never print a
  complete environment, secret-bearing config, private key, or token.
- Stage an immutable snapshot identified by Git commit plus config hash, and verify
  hashes on arrival. Never assume remote source or data is current.
- Treat a remote checkout as disposable and pull results back before releasing a
  machine.

## Data

- Keep the upstream OpenMoji checkout immutable and non-writable. Never "clean" the data
  by deleting or rewriting source files.
- Exclusions belong in a versioned manifest with reason codes.
- Exclude the OpenMoji `flags` group from the primary dataset, retaining it as a
  reversible named split for a later ablation.
- Quarantine visually blank or degenerate candidates for measured or manual review as
  defined in `DATA_CURATION.md`.
- Any augmentation or renormalization must be exactly invertible in token space, or must
  record the transform it applied. Verify that claim rather than asserting it.

## Run contract

The run registry is what makes this research reproducible. Keep it even though the
machine is trusted.

Every run receives a stable `run_id`: a readable experiment slug plus short code, config,
and data hashes. Random seeds alone are not run identities.

Before launch, record the hypothesis, the parent or baseline run, the Git commit and any
dirty-patch hash, the exact config and its hash, the dataset manifest and split hashes,
the GPU and driver, the framework versions, the seed set and determinism settings, and
the output locations. For any run whose outcome is a learning claim, predeclare
falsifiable pass/fail criteria and commit them before the run starts.

A run directory contains at least `run.yaml`, `metrics.jsonl`, `stdout.log`,
`artifacts.json`, and `result.md`. Append state transitions to `state/runs.jsonl`; do
not silently edit history. Use explicit states such as `planned`, `running`,
`completed`, `failed`, and `cancelled`, and record the reason for failure.

Preserve failed runs under their own identity rather than retrying in place. A falsified
predeclared criterion is a result, not a mistake to tune away.

Long runs must survive a dropped connection: use `tmux`, a detached process, or a
detached container with a recorded stable job identity. Do not leave an active job known
only in terminal scrollback.

## Experimental discipline

- Change one primary factor at a time unless the run is explicitly a systems test.
- Start with deterministic tiny fixtures and overfit tests.
- Keep train/validation/test split construction fixed, hashed, and family-disjoint.
- Report uncertainty across seeds when interpreting small differences.
- Compare quality at matched parameter count, data, codec, and engineering effort. The
  AR baseline must use KV caching.
- Never claim diffusion is faster merely because it uses fewer nominal decoding steps.
  Measure end-to-end latency, throughput, peak VRAM, and quality on the same named GPU.
- Label the raw corrupted state `x_t` and the model's predicted clean state `x_hat_0`.
  If safety projection or constrained decoding is applied, label it.
- Do not call a correlated denoiser an exact D3PM unless its transition and posterior are
  mathematically defined and implemented.
- Record the framework and CUDA versions with every result. A change of environment
  breaks numeric comparability with earlier runs; establish a matched control rather
  than comparing across environments silently.
- Preserve failed hypotheses and surprising samples in `reports/findings.md`.

## Artifact policy

- Keep metadata, compact reports, representative renders, and selected final artifacts
  in the repository.
- Keep full checkpoints and large datasets under `/home/dev/.cache`, not in Git.
- Retention is explicit: keep best, latest recoverable, and scientifically relevant
  failure checkpoints. Propose deletions separately; never delete artifacts or run
  records without asking.

## Local development checks

Keep the cheapest relevant checks first and runnable without a GPU:

- `ruff` and strict `mypy`;
- parser/serializer round-trip tests;
- property/fuzz tests for render safety and tensor invariants;
- deterministic data-manifest and split tests;
- CPU or tiny-GPU model smoke test;
- checkpoint save/resume test.

Code that only ever executes elsewhere is code that is never tested. Anything under
`scripts/remote/` needs a local test that exercises its real contract.

For every material code change, run the narrow relevant test first, then the broader
suite. State exactly what was tested and what was not.

## Session continuity

Maintain `state/CURRENT.md` as the concise handoff between sessions. It must say the
current hypothesis and evidence, the last completed action and its verification, active
jobs, artifact durability, current blockers, and the next smallest evidence-producing
action. Update it after every material result, failure, and before ending a session.

## Stop and ask

Stop and ask the operator when an action would create or raise external cost, when a
destructive or data-losing action is involved, when credentials or access are
insufficient, or when two plausible choices would materially change the scientific
conclusion.

Ordinary code edits, local tests, and bounded runs on `gpubox-4080` proceed without
asking.
