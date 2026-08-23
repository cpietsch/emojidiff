# `gtc` Codex handoff for MojiDiff

This package turns a fresh `cpietsch/code-server-cpu` deployment named `gtc` into the
persistent orchestration and research-control machine while GPU systems act as
disposable or scheduler-managed workers. Codex runs in YOLO mode inside this isolated
control plane.

## Files

- `AGENTS.md` — binding Codex operating, security, SSH, authorization, and recovery
  rules.
- `PROJECT_PLAN.md` — evidence-gated SVG diffusion research plan without a calendar.
- `DATA_CURATION.md` — reversible OpenMoji cleaning and flags policy.
- `DEVBOX_BOOTSTRAP.md` — fresh `gtc` deployment, Codex install, hardening, and startup.
- `hosts.example.yaml` — non-secret inventory and authorization template.
- `ssh_config.example` — hardened SSH alias examples for Vast, Tailscale, and the A100
  gateway.
- `START_PROMPT.md` — first prompt to give Codex after placing the files in the repo.

## Install into the canonical repository

Deploy `https://github.com/cpietsch/code-server-cpu` with `TS_HOSTNAME=gtc`, then follow
`DEVBOX_BOOTSTRAP.md`. Copy these files into the root of the MojiDiff repository on
`gtc`. Review any
existing `AGENTS.md` before merging; do not overwrite repository-specific instructions
blindly.

Then start or attach to the persistent shell:

```bash
tmux new -As mojidiff-codex
cd /home/dev/workspace/mojidiff
codex --yolo
```

Paste the contents of `START_PROMPT.md` as the first instruction. Codex will create a
local `state/hosts.local.yaml` from the example and identify the few fields needed before
it can use a remote GPU.

To return later, attach the tmux session and resume Codex from the same repository:

```bash
tmux attach -t mojidiff-codex
cd /home/dev/workspace/mojidiff
codex --yolo resume --last
```

Do not place SSH private keys, cloud credentials, tokens, or passwords in the
repository. Fill only aliases, non-secret paths, storage URIs, and explicit resource
caps in `state/hosts.local.yaml`.

Before enabling a worker, merge the relevant `ssh_config.example` block into the real
`~/.ssh/config`, verify the host key/fingerprint through a trusted source, and confirm
that `ssh <alias> true` succeeds non-interactively. A newly rented Vast instance needs a
new exact host-key entry if its endpoint or key changes.

`--yolo` is the documented alias for bypassing both approvals and sandboxing. It should
only be used in an externally hardened environment. Here, that means a fresh dedicated
host, no Coolify/host Docker socket, no Vast billing API key, and least-privilege
credentials. Restrict the `gtc` Tailscale identity with ACLs so it can reach only the
intended GPU workers. `AGENTS.md` encodes these requirements, but technical credential
and infrastructure boundaries are still necessary because YOLO mode does not enforce
them.
