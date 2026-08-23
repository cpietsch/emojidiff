# Current research state

Updated: 2026-08-23T16:09:40Z

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
proxies. A typed contour codec now preserves compound paint operations, semantic
stroke categories, fill rules, dashes, and painter order while enforcing canonical
padding and a safe fixed serializer. Actual typed programs confirm the structural
advantage: median semantic/outlined segment counts are 53/146, and at P48/S128 the
fixture loses 181/509 segments and 0/13 contours. Q256 improves over q128, but its
255-interval grid displaces ordinary pixel-aligned coordinates and inflates antialias
error. This is not yet a final codec selection.

## Last completed action and verification

Pinned OpenMoji 17.0.0 at commit
`f9fc506a3f913be9897ab0181d611d4c910a4104`; hashed and made the 4,495-SVG raw checkout
non-writable; retained a falsified audit v1; completed safe audit v2; generated empirical
distribution reports and seven quarantine contact sheets; visually reviewed all 104
candidates; produced the reviewed v4 manifest; and ran independent curation validation.
After local checkpoint `a324c57`, ran representation probe v1 with an exact config and
primary-manifest hash. All 88/88 PicoSVG 0.23.0 normalizations succeeded. Median RGBA
MAE was 0.000558 at 72 px and 0.001173 at 18 px; all outlined outputs removed strokes.
Compact evidence and the completed run record are committed locally at `6478654`; no
push occurred. Implemented and checkpointed the deterministic semantic normalizer,
typed codec, actual truncation reporting, outlined control harness, and standalone safe
serializer at `60e3dd7`. The full local suite passes (36 tests), as do Ruff and strict
mypy. Typed codec probe v1 completed 700 stable renderable round trips. Semantic
normalization succeeded for 87/88 icons; the reason-coded nonuniform-stroke failure has
the outlined representation as fallback. P64/S384 covered both fixture representations
without truncation. Worst-case 72/18 px contact sheets were inspected; compact outlined
truncation visibly corrupted the rice ball and UFO, while untruncated q256 candidates
remained recognizable despite alignment-sensitive pixel metrics.

## Active jobs

None. Every worker is disabled, the isolated Docker daemon has no running container,
and no authenticated remote command has been issued. The operator reports a running
Vast RTX 5090 instance with a 100 GB attached volume, but it remains disabled in the
authorization record and has no active MojiDiff job.

## Artifact durability

Raw source (405 MB), audit v1/v2 renders and tables (about 102 MB), and derived outlined
fixture SVGs (660 KB) are on the persistent workspace. Compact reports
are on the persistent `gtc` workspace. Curation artifacts are versioned at `0b96473`
and representation evidence at `6478654`; typed-codec derived SVGs add 11 MB and the
compact typed-codec report adds 1.2 MB pending its local evidence checkpoint. Nothing
was pushed. No external artifact sink is configured or verified, so these
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
- The operator supplied two distinct Vast port references. A non-authenticating host-key
  scan found a different SSH host on each, so the exact SSH command, username/port, and
  expected host fingerprint remain ambiguous. No host key was trusted and no login was
  attempted.
- The reported 100 GB attached volume has no recorded mount path or persistence
  guarantee and therefore is not yet treated as the configured durable artifact sink.
- `owned_gpu` is disabled; `resource_cap.max_steps` and `max_storage_gb` are null.
- `a100_cluster` is disabled; `workspace_root`, `namespace`, `service_account`, `pvc`,
  `resource_cap.max_jobs`, `max_steps_per_job`, and `max_storage_gb` are null.
- The template SSH alias names exist in local authorization state, but no real aliases,
  endpoints, per-worker keys, or verified host fingerprints exist in `~/.ssh/config`.

## Next smallest evidence-producing action

Checkpoint typed-codec evidence, then compare pixel-aligned 145- and 289-value
coordinate lattices (0.5 and 0.25 unit steps) on the identical fixture and budgets. If
that resolves the observed edge-alignment error, run structural normalization over the
full 4,006-row primary manifest before selecting final path/segment budgets.
