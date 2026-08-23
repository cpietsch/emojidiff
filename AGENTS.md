# MojiDiff Codex Operating Contract

This repository is an exploratory research project for a small categorical diffusion
model over editable SVG programs. `gtc` is the persistent control plane. GPU
machines are execution workers, never the source of truth.

## Read order

At the start of every session, read these files completely in this order:

1. `AGENTS.md`
2. `PROJECT_PLAN.md`
3. `DATA_CURATION.md`
4. `state/CURRENT.md`, if present
5. `state/hosts.local.yaml`, if present; otherwise `hosts.example.yaml`
6. The most recent relevant report under `reports/`

If this file changes, tell the operator to restart Codex before relying on the new
instructions.

## Operating objective

Run this as an evidence-driven experiment, not a deadline-driven build. Advance from
cheap tests to expensive tests only when the preceding evidence justifies it. Keep
negative results. Do not force the research to support a predetermined claim.

The working hypothesis is that a typed, render-safe SVG program plus structure-aware
categorical corruption can support useful generation, editing, and visually meaningful
denoising trajectories. The comparison against a cached autoregressive model must be
honest and hardware-specific.

## System topology

- `gtc`: a fresh control-plane container built from
  `https://github.com/cpietsch/code-server-cpu`. It hosts Codex CLI, the canonical Git
  worktree, configuration, run registry, reports, lightweight evaluation artifacts,
  and orchestration scripts. Its persistent workspace is `/home/dev/workspace`.
- Vast.ai RTX 5090: ephemeral worker reached through its public SSH endpoint. The Vast
  instance is already a container; do not add Docker-in-Docker.
- Owned GPU machine: persistent worker reached over Tailscale SSH. Docker with NVIDIA
  Container Toolkit may be used there.
- A100 cluster: worker reached through its SSH gateway. Prefer a Kubernetes Job and
  durable PVC or configured artifact store over a long process in a login shell.
- Codex and OpenAI credentials remain on `gtc`. Never copy them to a worker.

Do not attempt cross-host distributed training. Treat each GPU as an independent
worker for replication, ablations, evaluation, or a separate run. A run's latency and
throughput results belong to the exact hardware and software environment that produced
them.

## Authorization boundary

`state/hosts.local.yaml` is the machine-readable authorization record.

- A worker with `enabled: true` and a complete `resource_cap` is authorized for normal,
  non-destructive probes, staging, training, evaluation, checkpointing, and artifact
  transfer within that cap. A map whose values are all null is not a cap.
- For a billed Vast worker, both `max_steps` and `max_spend_usd` must be non-null. For an
  owned worker, `max_steps` must be non-null. For the cluster, `max_jobs` and
  `max_steps_per_job` must be non-null. Storage bounds are also required before a run
  can produce substantial checkpoints or datasets.
- `enabled: false`, a missing host, or an incomplete cap is not authorized for compute
  jobs. Read-only connectivity probes are allowed only when the operator has already
  supplied that endpoint.
- Never rent, stop, destroy, resize, or change the billing cap of a Vast instance
  without explicit operator approval.
- Never delete a remote workspace, checkpoint, PVC, object-store prefix, or local run
  record without explicit operator approval.
- Do not submit a Kubernetes job until its namespace, image, storage, and resource cap
  are recorded.
- Stop before any action that would exceed the configured step, sample, storage, or
  spend cap. Ask for a revised cap; do not infer one.

## YOLO execution profile

Codex is intentionally started with `codex --yolo`. This bypasses both the Codex
sandbox and interactive approvals. It changes command execution, not the authorization
boundary above. Treat every prohibition in this file as binding even though the CLI
will not enforce it.

Because no approval barrier exists:

- `gtc` must be a fresh dedicated VM/container environment, not a shell on a shared
  production host;
- never mount the Coolify/host Docker socket into the dev container;
- the isolated DinD daemon may contain only disposable project workloads;
- do not place a Vast API key or cloud billing credential on `gtc`; the operator
  provisions, resizes, and terminates billed machines outside Codex;
- give `gtc` a dedicated tagged Tailscale identity whose ACL permits only the intended
  GPU workers and explicitly required services, not unrelated tailnet machines or admin
  interfaces;
- use a separate least-privilege SSH key for each worker class; never use a personal
  all-host key or SSH agent forwarding;
- use a repository-scoped GitHub credential and an artifact credential scoped to the
  MojiDiff prefix, without delete permission where the provider supports it;
- use a namespace-scoped Kubernetes identity with ResourceQuota and no cluster-admin
  permission;
- record a clean Git checkpoint before every remote launch;
- never interpret `--yolo` as permission to push, publish, delete, spend, or widen
  credentials.

If these external controls are missing, continue with local read-only or reversible
work and report the missing control. Do not compensate by trusting a prompt alone.

## Security and SSH rules

- Use named aliases from `~/.ssh/config`; do not scatter IP addresses, ports, or user
  names through scripts.
- Keep private keys and tokens out of the repository, logs, shell history, configs, and
  command output. Use the SSH agent or root/user-readable secret files outside the repo.
- Set `BatchMode yes`, `ForwardAgent no`, and finite connection keepalives/timeouts.
- When the operator supplies the exact SSH user, host, and port for a newly provisioned
  ephemeral worker, first-use pinning is sufficient: use per-alias
  `StrictHostKeyChecking accept-new`, then record the observed fingerprint for recovery
  and audit. Out-of-band fingerprint confirmation is not required unless the operator
  explicitly requests high-assurance verification. Never use
  `StrictHostKeyChecking no`, and treat any later key change as a blocker.
- Remove or replace stale host-key entries narrowly, never the whole known-hosts file.
- Prefer pulling results from workers to `gtc` or a configured artifact sink. Do
  not open inbound services merely for convenience.
- Never print a complete environment, secret-bearing config, SSH private key, or token.

## Canonical repository and staging

The canonical source is the Git worktree on `gtc`. A worker checkout is disposable.

- Preserve unrelated or pre-existing operator changes.
- Never use destructive reset, force-push, broad recursive deletion, or `rsync --delete`.
- Never push to a remote repository unless explicitly asked.
- Before a nontrivial run, make the code identity reproducible. Prefer a local Git
  commit. If a commit would mix unrelated user work, record the base commit, a patch,
  the patch hash, and the list of untracked experiment files instead.
- Stage into a run-specific directory, not a mutable shared directory. Use a snapshot
  identified by Git commit plus patch/config hash.
- Never assume worker-local source or data is current. Verify hashes.

Recommended repository layout:

```text
configs/             versioned experiment configurations
data/manifests/      provenance, licenses, splits, and content hashes
docs/                representation and methodology notes
infra/               container and Kubernetes definitions
reports/             generated audits, evaluations, plots, and findings
runs/                lightweight immutable run metadata; no giant checkpoints
scripts/remote/      worker probe, stage, launch, monitor, sync, and evacuate tools
src/                 library and command-line code
state/               local orchestration state; secrets and host details are ignored
tests/               unit, property, round-trip, and smoke tests
```

## Worker adapters

Expose the same conceptual operations for every worker:

1. `probe`: read-only GPU, driver, storage, image, Python, framework, and connectivity
   inspection.
2. `stage`: transfer or check out an immutable code/config snapshot and verify hashes.
3. `smoke`: run tokenizer, renderer, one forward/backward pass, checkpoint write/read,
   and artifact transfer on a tiny fixture.
4. `launch`: start a bounded detached job and return a stable job identity.
5. `status`: report scheduler/process state and recent sanitized logs.
6. `sync`: copy selected checkpoints, metrics, samples, and metadata to durable storage.
7. `cancel`: graceful checkpoint-and-stop first; destructive termination requires the
   operator's approval when data could be lost.

Implementation by worker type:

- Vast container: use its shell directly. Launch with remote `tmux`, `systemd-run`
  where available, or a carefully recorded detached process. Never rely on the local
  SSH connection remaining open.
- Owned GPU host: execute through the versioned project container using Docker and the
  NVIDIA runtime. Persist data and artifacts in explicit mounted paths.
- A100 cluster: render and submit a versioned Kubernetes Job YAML. Record namespace,
  Job name/UID, image digest, PVC, resource requests/limits, and log commands. A
  notebook may be used for inspection, not as the sole home of a long training run.

Remote commands must be non-interactive, quote-safe, idempotent where practical, and
log their exit status. Do not compose remote shell commands from unvalidated strings.

## Run contract

Every run receives a stable `run_id`; use a readable experiment slug plus short code,
config, and data hashes. Random seeds alone are not run identities.

Before launch, record:

- hypothesis and expected information gain;
- parent/baseline run, if any;
- Git commit and dirty-patch hash;
- exact config and config hash;
- dataset manifest and split hashes;
- worker alias, GPU model, driver, container tag and digest;
- PyTorch, CUDA, cuDNN, TensorRT, and relevant package versions;
- seed set and deterministic/nondeterministic settings;
- resource cap and checkpoint/evacuation policy;
- full sanitized command and output locations.

After launch, immediately record the PID, tmux session, Kubernetes Job UID, or other
stable identity. Append state transitions to `state/runs.jsonl`; do not silently edit
history. A run directory should contain at least:

```text
run.yaml
metrics.jsonl
stdout.log or durable log URI
artifacts.json
result.md
```

Use explicit states such as `planned`, `staged`, `running`, `checkpointed`, `completed`,
`failed`, `cancelled`, and `evacuated`. Record the reason for failure or cancellation.

An expensive run may start only after its exact pipeline has passed the smoke adapter
on the target worker. Do not launch many speculative runs merely because compute is
available. Prefer the experiment with the highest expected information gain.

## Artifact policy

`gtc` is a small control machine. It is not implicitly the bulk checkpoint store.

- Before meaningful training, probe available disk and configure a durable artifact
  sink in `state/hosts.local.yaml`.
- Keep metadata, compact reports, representative renders, and selected final artifacts
  on `gtc`. Put full checkpoints and large datasets in the configured sink.
- On an ephemeral Vast worker, checkpoint and evacuate selected artifacts by training
  step. Verify destination size and hash before considering them safe.
- Never stop an ephemeral worker until the chosen artifacts and run metadata have been
  verified off-worker.
- Retention must be explicit: keep best, latest recoverable, and scientifically relevant
  failure checkpoints; propose deletions separately.

## Experimental discipline

- Change one primary factor at a time unless the run is explicitly a systems test.
- Start with deterministic tiny fixtures and overfit tests.
- Keep train/validation/test split construction fixed and hashed for comparisons.
- Report uncertainty across seeds when interpreting small differences.
- Compare quality at matched parameter count, data, codec, and reasonable engineering
  effort. The AR baseline must use KV caching.
- Never claim diffusion is faster merely because it uses fewer nominal decoding steps.
  Measure end-to-end latency, throughput, peak VRAM, and quality on the same named GPU.
- Label the raw corrupted state `x_t` and the model's predicted clean state `x_hat_0`.
  If safety projection or constrained decoding is applied, label it.
- Preserve failed hypotheses and surprising samples in `reports/findings.md`.
- Do not call a correlated denoiser an exact D3PM unless its transition and posterior
  are mathematically defined and implemented.
- Keep the upstream OpenMoji checkout immutable. Exclusions belong in a versioned
  manifest with reason codes; never "clean" the data by deleting source files.
- Exclude the OpenMoji `flags` group from the primary dataset, while retaining it as a
  reversible named split for a later ablation. Quarantine visually blank or degenerate
  candidates for measured/manual review as defined in `DATA_CURATION.md`.

## Local development checks

Add the cheapest relevant checks first and keep them runnable without a GPU:

- formatter, linter, and type checks;
- parser/serializer round-trip tests;
- property/fuzz tests for render safety and tensor invariants;
- deterministic data-manifest and split tests;
- CPU or tiny-GPU model smoke test;
- checkpoint save/resume test;
- remote adapter dry-run tests.

For every material code change, run the narrow relevant test first, then the broader
suite. State exactly what was tested and what was not.

## Session continuity

Maintain `state/CURRENT.md` as the concise handoff between Codex sessions. It must say:

- current hypothesis and evidence;
- last completed action and verification;
- active jobs with worker and stable job IDs;
- artifact durability status;
- current blockers or decisions needed;
- next smallest evidence-producing action.

Update it after every launch, material result, failure, and before ending a session.
Do not leave an active job known only in terminal scrollback.

## Stop conditions

Stop and ask the operator when:

- a worker, artifact sink, namespace, or resource cap is missing;
- an action would create or raise external cost;
- credentials or access permissions are insufficient;
- host identity is ambiguous or changed unexpectedly;
- a destructive action or data-loss risk is involved;
- two plausible choices would materially change the scientific conclusion;
- a run would exceed the recorded authorization boundary.

Ordinary code edits, local tests, read-only probes, and bounded runs on explicitly
enabled workers should proceed without unnecessary questions.

## Initial bootstrap

When first dropped into the repository:

1. Confirm that the host is `gtc`, the workspace is persistent, the host Docker socket
   is absent, and Docker points only to the isolated DinD daemon.
2. Inspect the repository, Git status, disk, installed CLI tools, and existing state.
3. Create missing directories and a secret-safe `.gitignore`.
4. Copy `hosts.example.yaml` to `state/hosts.local.yaml` if absent, but leave workers
   disabled until the operator fills the real aliases and caps.
5. Create `state/CURRENT.md` and an append-only run registry.
6. Implement a local environment probe and the remote adapter interfaces before model
   code.
7. Build the OpenMoji audit and reversible curation manifest before preprocessing or
   model code.
8. Report only the missing decisions that block the next evidence-producing action.
9. Once a worker is enabled, probe it, smoke-test artifact recovery, then begin the
   evidence gates in `PROJECT_PLAN.md`.
