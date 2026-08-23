# Current research state

Updated: 2026-08-23T14:54:00Z

## Current hypothesis and evidence

Gate B evidence supports retaining visually unusual OpenMoji assets rather than using
metric tails as defects. The reviewed OpenMoji 17.0.0 primary manifest has 4,006 rows:
3,902 ordinary includes and 104 visually reviewed `include_override` rows. It excludes
270 metadata flags, preserves 218 noncanonical exact-render aliases in a reversible
split, and retains one renderer-failing SVG as a defect row. Independent verification
passes raw hash/immutability, row coverage, duplicate, flag-policy, and family-split
invariants.

The next research hypothesis is unresolved: semantic OpenMoji strokes may preserve
editability with shorter programs, while outlining may simplify style fields at the
cost of path/segment expansion and fidelity drift.

## Last completed action and verification

Pinned OpenMoji 17.0.0 at commit
`f9fc506a3f913be9897ab0181d611d4c910a4104`; hashed and made the 4,495-SVG raw checkout
non-writable; retained a falsified audit v1; completed safe audit v2; generated empirical
distribution reports and seven quarantine contact sheets; visually reviewed all 104
candidates; produced the reviewed v4 manifest; and ran independent curation validation.
Formatter, linter, strict type checking, and 15 tests pass.

## Active jobs

None. Every worker is disabled, the isolated Docker daemon has no running container,
and no remote command has been issued.

## Artifact durability

Raw source (405 MB), audit v1/v2 renders and tables (about 102 MB), and compact reports
are on the persistent `gtc` workspace. Compact manifests and reports are ready for a
local Git checkpoint. No external artifact sink is configured or verified, so these
bulk artifacts are not durable against loss of the control-plane volume and meaningful
GPU work remains blocked.

## Current blockers and missing authorization

- YOLO checks passed for hostname, persistent repository location, TLS-only isolated
  DinD, absent host Docker sockets, absent worker/cloud/kube/GitHub credentials, and an
  empty running-container inventory. Dedicated Tailscale tag/ACL restrictions and
  external credential scopes cannot be verified from inside this container.
- `artifact_store.type`, `artifact_store.uri`, and
  `artifact_store.credentials_source` are unset.
- `vast_5090` is disabled; `resource_cap.max_steps`, `max_spend_usd`, and
  `max_storage_gb` are null.
- `owned_gpu` is disabled; `resource_cap.max_steps` and `max_storage_gb` are null.
- `a100_cluster` is disabled; `workspace_root`, `namespace`, `service_account`, `pvc`,
  `resource_cap.max_jobs`, `max_steps_per_job`, and `max_storage_gb` are null.
- The template SSH alias names exist in local authorization state, but no real aliases,
  endpoints, per-worker keys, or verified host fingerprints exist in `~/.ssh/config`.

## Next smallest evidence-producing action

Create a local Git checkpoint (no push), then begin Gate C with a CPU-only measured
semantic-versus-outlined representation probe: path/segment distributions, transform
coverage, candidate slot truncation, and 72/18 px round-trip fidelity on a deterministic
stratified fixture before implementing the full codecs.
