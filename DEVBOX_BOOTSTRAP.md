# Fresh `gtc` control-plane bootstrap

This procedure assumes a fresh dedicated VM using the deployment from
`https://github.com/cpietsch/code-server-cpu`. The dev container is the Codex control
plane; GPU training runs elsewhere over SSH.

## Deploy

In Coolify, deploy the repository's Compose definition and set at least:

```text
TS_HOSTNAME=gtc
TS_AUTHKEY=<set only in Coolify's secret environment>
SSH_AUTHORIZED_KEYS=<public keys only>
```

Do not publish SSH on the public host. Reach `gtc` through Tailscale. Use fresh named
volumes for the fresh control plane, and do not mount `/var/run/docker.sock` or another
host socket into the dev container.

Use a dedicated Tailscale tag/auth key for `gtc`. Its tailnet ACL should allow only the
specific owned GPU and A100 gateway targets needed by the project. It should not reach
unrelated servers, NAS devices, routers, hypervisors, or administration interfaces.

The repository currently supplies:

- Ubuntu 24.04;
- Node 22 and Python 3.12;
- Git, Git LFS, GitHub CLI, `uv`, `ripgrep`, `tmux`, and SSH;
- a Tailscale sidecar;
- an isolated Docker-in-Docker daemon reached over TLS.

It does not currently install Codex. Container names may still contain `devbox`; the
tailnet hostname is controlled by `TS_HOSTNAME` and should be `gtc`.

## Verify the external boundary

From a separate tailnet machine, first verify that the new node is visible and that SSH
works only through the intended Tailscale path:

```bash
tailscale status
ssh dev@gtc true
```

After connecting as `dev`, verify and record:

```bash
hostname
docker context show
docker info
test ! -S /var/run/docker.sock
git --version
gh --version
node --version
python --version
uv --version
tmux -V
df -h /home/dev/workspace
```

The expected Docker daemon is the isolated DinD endpoint from `DOCKER_HOST`, not the
Coolify host. If `/var/run/docker.sock` is mounted, stop and correct the deployment
before starting Codex YOLO mode.

Also keep these credentials absent from `gtc`:

- Vast.ai API/billing token;
- Coolify administration token;
- cloud account owner credentials;
- cluster-admin kubeconfig;
- broad personal SSH key or forwarded SSH agent.

Use project-scoped credentials only. The operator creates and terminates Vast instances
outside Codex, then supplies a verified SSH alias and explicit cap.

## Install and authenticate Codex

Install the current Codex CLI as user `dev` using OpenAI's Linux installer:

```bash
curl -fsSL https://chatgpt.com/codex/install.sh | sh
codex --version
```

Run `codex` once and complete the ChatGPT sign-in flow. Do not copy the resulting Codex
credentials to GPU workers.

## Create the canonical project

Use the persistent workspace volume:

```bash
cd /home/dev/workspace
git clone <MOJIDIFF_REPOSITORY_URL> mojidiff
cd mojidiff
```

Place this handoff's files in the repository root before starting the durable Codex
session. Create dedicated SSH keys for worker classes rather than reusing a broad
personal key. Keep private keys outside the repository and disable agent forwarding.

## Start the durable YOLO session

```bash
tmux new -As mojidiff-codex
cd /home/dev/workspace/mojidiff
codex --yolo
```

Paste `START_PROMPT.md`. YOLO bypasses Codex approvals and sandboxing; it does not
override the repository's authorization policy. `state/hosts.local.yaml` remains the
execution boundary for remote compute.

To return:

```bash
ssh dev@gtc
tmux attach -t mojidiff-codex
```

If the Codex process ended, start it from the same repository with:

```bash
cd /home/dev/workspace/mojidiff
codex --yolo resume --last
```
