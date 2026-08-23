# Current research state

Updated: 2026-08-23T15:03:00Z

## Current hypothesis and evidence

Gate B evidence supports retaining visually unusual OpenMoji assets rather than using
metric tails as defects. The reviewed OpenMoji 17.0.0 primary manifest has 4,006 rows:
3,902 ordinary includes and 104 visually reviewed `include_override` rows. It excludes
270 metadata flags, preserves 218 noncanonical exact-render aliases in a reversible
split, and retains one renderer-failing SVG as a defect row. Independent verification
passes raw hash/immutability, row coverage, duplicate, flag-policy, and family-split
invariants.

Gate C probe evidence makes semantic OpenMoji strokes the leading representation
candidate. On an 88-icon stratified fixture, PicoSVG outlining preserved renders closely
but expanded segments 2.241x at the median, 5.287x at p95, and 13.833x in the worst
case. At 64 segments/path, 18 outlined fixtures exceeded budget versus 7 semantic
proxies. This is not yet a final codec selection.

## Last completed action and verification

Pinned OpenMoji 17.0.0 at commit
`f9fc506a3f913be9897ab0181d611d4c910a4104`; hashed and made the 4,495-SVG raw checkout
non-writable; retained a falsified audit v1; completed safe audit v2; generated empirical
distribution reports and seven quarantine contact sheets; visually reviewed all 104
candidates; produced the reviewed v4 manifest; and ran independent curation validation.
After local checkpoint `a324c57`, ran representation probe v1 with an exact config and
primary-manifest hash. All 88/88 PicoSVG 0.23.0 normalizations succeeded. Median RGBA
MAE was 0.000558 at 72 px and 0.001173 at 18 px; all outlined outputs removed strokes.
Formatter, linter, strict type checking, and 19 tests pass.

## Active jobs

None. Every worker is disabled, the isolated Docker daemon has no running container,
and no remote command has been issued.

## Artifact durability

Raw source (405 MB), audit v1/v2 renders and tables (about 102 MB), and derived outlined
fixture SVGs (660 KB) are on the persistent workspace. Compact reports
are on the persistent `gtc` workspace. Compact manifests and reports are versioned in
local Git checkpoint `0b96473`; nothing was pushed. No external artifact sink is configured or verified, so these
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

Implement a deterministic semantic-stroke normalizer/codec on the fixed fixture and
measure transform flattening, unsupported path commands, 128-versus-256 coordinate-bin
round-trip fidelity, and actual fixed-slot truncation. Apply the identical typed codec
budgets to outlined paths as the fallback/control before making the Gate C selection.
