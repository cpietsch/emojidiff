# Current research state

Updated: 2026-08-23T17:21:47Z

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

The controlled aligned-grid rerun supports a quarter-unit coordinate lattice. At the
fixture coverage budget, semantic q289 lowers median q256 RGBA MAE by 48.2% at 72 px
and 42.5% at 18 px; outlined improvements are 23.9% and 30.2%. Q145 also beats q128
with a similarly sized vocabulary. Q289 is the leading coordinate vocabulary and q145
is the compact ablation.

The exact full-primary census attempted semantic and outlined normalization for every
one of 4,006 icons. Semantic succeeds for 3,937; outlined fallback recovers 60 of its
69 failures; nine partial-opacity icons remain unsupported, for hybrid coverage of
3,997/4,006 (99.775%). P64/S128 is falsified as lossless: it drops 35 contours and 493
segments across eight supported icons. P96/S384 is structurally lossless for all 3,997
supported programs but only 0.240% slot-utilized. Style vocabulary (344 nonexact icons),
35 out-of-bounds coordinate scalars in 22 icons, opacity, and sparse-tail handling keep
Gate C open.

## Last completed action and verification

Pinned OpenMoji 17.0.0 at commit
`f9fc506a3f913be9897ab0181d611d4c910a4104`; hashed and made the 4,495-SVG raw checkout
non-writable; retained a falsified audit v1; completed safe audit v2; generated empirical
distribution reports and seven quarantine contact sheets; visually reviewed all 104
candidates; produced the reviewed v4 manifest; and ran independent curation validation.
Representation, typed-codec, and aligned-lattice evidence is committed locally through
`2099f3d`. The aligned probe completed 700 stable round trips; at q289,
semantic/outlined median 18 px MAE is 0.001710/0.002319, and the alignment-sensitive
`E2C2` semantic case improves from 0.117692 at q256 to 0.001970.

Implemented the immutable full-primary census at `6bb0f10` and bounded Vast stage/smoke
adapters at `c396922`. The census run
`full-primary-structure-v1-6bb0f10-4f1bf442-4e7162ec` completed locally with exactly
8,012 attempts and 4,006 hybrid rows. Actual SHA-256 values match the embedded report
identities; every source has exactly one semantic and one outlined attempt; paired
identity fields and deterministic ordering validate; all three routes contain the
complete 30-point P/S grid; and every loss decomposition balances. Adapter verification
passes the full 49-test suite, Ruff, strict mypy, a local idempotent stage, and a
pre-import tamper regression. GPU smoke has not run because `gtc` has no CUDA and the
Vast worker is not authorized.

## Active jobs

None. Every worker is disabled, the isolated Docker daemon has no running container,
and no authenticated remote command has been issued. The operator reports a running
Vast RTX 5090 instance with a 100 GB attached volume, but it remains disabled in the
authorization record and has no active MojiDiff job.

## Artifact durability

Raw source (405 MB), audit renders/tables, and derived fixture SVGs remain on the
persistent workspace. Curation, representation, typed-codec, and aligned-grid compact
evidence is versioned locally through `2099f3d`. The new full-primary report is 26 MB
and is pending its local evidence checkpoint. Nothing was pushed. No external artifact
sink is configured or verified, so bulk artifacts are not durable against loss of the
control-plane volume and meaningful GPU work remains blocked.

## Current blockers and missing authorization

- YOLO checks passed for hostname, persistent repository location, TLS-only isolated
  DinD, absent host Docker sockets, absent worker/cloud/kube/GitHub credentials, and an
  empty running-container inventory. Dedicated Tailscale tag/ACL restrictions and
  external credential scopes cannot be verified from inside this container.
- `artifact_store.type`, `artifact_store.uri`, and
  `artifact_store.credentials_source` are unset.
- `vast_5090` is disabled; `resource_cap.max_steps`, `max_spend_usd`, and
  `max_storage_gb` are null.
- The operator supplied a public endpoint but two distinct Vast port references and no
  SSH username. The exact username and one exact SSH port are still required. Under the
  current contract, per-alias `StrictHostKeyChecking accept-new` is sufficient for first
  use; no out-of-band fingerprint is required. No host key has been trusted and no login
  has been attempted.
- The reported 100 GB attached volume has no recorded mount path or persistence
  guarantee and therefore is not yet treated as the configured durable artifact sink.
- `owned_gpu` is disabled; `resource_cap.max_steps` and `max_storage_gb` are null.
- `a100_cluster` is disabled; `workspace_root`, `namespace`, `service_account`, `pvc`,
  `resource_cap.max_jobs`, `max_steps_per_job`, and `max_storage_gb` are null.
- The template SSH alias names exist in local authorization state, but no real named
  Vast alias exists in `~/.ssh/config`. A local SSH key is present, but its username and
  endpoint have not been recorded or used.

## Next smallest evidence-producing action

Checkpoint the completed full-primary evidence, then add exact per-path `opacity`,
`fill-opacity`, and `stroke-opacity` categorical fields and run the nine-icon recovery
test. Follow with q289 render/round-trip evidence on the opacity set and selected
style/OOB/structural tails before choosing a fixed, ragged, or overflow capacity policy.
