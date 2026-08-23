# `gtc` YOLO boundary verification

Verified locally on 2026-08-23:

- hostname is `gtc`;
- canonical repository is `/home/dev/workspace/mojidiff` on the persistent
  `/home/dev/workspace` volume;
- `DOCKER_HOST=tcp://localhost:2376`, TLS verification is enabled, and the selected
  Docker context resolves to the same endpoint;
- `/var/run/docker.sock`, `/run/docker.sock`, and
  `/run/host-services/docker.sock` are absent and no Docker socket appears in the
  container mount table;
- the DinD daemon identifies itself as `gtc`, uses `/var/lib/docker`, and has no active
  containers; its cached `python:3.12-slim` image is a disposable local test artifact;
- `~/.ssh` contains no private worker key or worker config, no SSH agent is forwarded,
  and no kubeconfig, GitHub credential file, or Vast/cloud/billing-token environment
  variable name was detected;
- `state/hosts.local.yaml` is byte-identical to `hosts.example.yaml`, remains ignored,
  and authorizes no worker.

Not verifiable from inside this container:

- whether the external Tailscale identity has a dedicated project tag and restrictive
  ACL;
- whether this deployment is fresh/dedicated beyond the observed container and volume
  topology;
- scopes of credentials that have not yet been installed, including future SSH,
  artifact-store, GitHub, and Kubernetes identities.

Therefore the local execution boundary passes, but external hardening is only partially
verified. Local reversible research may continue. Remote or cost-bearing work remains
blocked until the external controls and machine-readable authorization fields are
configured.
