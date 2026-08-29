# Current research state

Updated: 2026-08-29T17:34:49Z

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

The exact opacity extension recovers all nine formerly unsupported icons under both
semantic and outlined normalization and q289/P96/S64 typed round trips. All 18 programs
are stable with no structural loss or style approximation. The observed categorical
values are exactly `0.25, 0.4, 0.5, 0.502, 0.6, 0.9969, 0.997, 0.999, 1.0`; `0.997` is
the PicoSVG-rounded outlined form of source `0.9969`. `1F31A` and `1F31D` each retain
one out-of-bounds scalar in both routes, so 14/18 programs are strict-lossless and four
use one explicitly recorded clamp.

The opacity-aware full-primary regression now proves complete structural coverage:
semantic normalization succeeds for 3,946 icons and the unchanged reason-coded
outlined fallback recovers its 60 failures, for 4,006/4,006 hybrid success. All 7,994
non-opacity attempt rows and 3,997 previously supported hybrid rows match census v1
after removing only newly reported opacity fields; all 18 prior opacity failures are
now successes. P64/S128 remains falsified with 35 dropped contours and 493 dropped
segments across eight icons. P96/S384 is lossless for all 4,006 but only 0.2403%
slot-utilized. Gate C remains open on 37 OOB scalars in 24 icons, 347 nonexact-style
icons with 311 literal stroke widths, and the sparse fixed-capacity tail.

A complete read-only classification of the 37 OOB scalars found that every one is a
cubic Bezier control handle; no move, line, quadratic, or cubic endpoint is outside
0..72. This narrows the question from invalid visible geometry to whether clamping
legitimate off-canvas curve handles measurably changes the clipped render. The pinned
24-icon q289 counterfactual harness is implemented but has not yet been registered or
run, so this classification remains diagnostic rather than completed run evidence.

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
Vast worker is not authorized. The completed full-primary evidence and recovery state
are checkpointed locally at `1063fc7`; nothing was pushed.

Implemented exact per-path element, fill, and stroke opacity tokens at `e31273c`, with
compound-layer style equality, canonical PAD/NONE invariants, safe serialization, a
schema-v2 explicit-opacity config path, and an opacity-bearing Vast tiny smoke fixture.
Focused codec tests pass 22/22; the full suite passes 53/53; Ruff and strict mypy pass.
An unregistered read-only nine-icon diagnostic confirmed semantic and outlined
normalization recovery and identified the two one-scalar OOB cases above.

Implemented and checkpointed the reproducible opacity probe at `5164a95`, then ran
`opacity-recovery-v1-5164a95-5f7afab1-49923968` locally. Both routes succeeded for all
nine icons; all 18 tensor/SVG identities are stable; no P/S truncation or style
approximation occurred; and the only four projected programs are the two OOB moons in
both routes. Semantic median RGBA MAE is 0.001547 at 72 px and 0.001722 at 18 px. A
second complete invocation accepted every exact prior artifact, verifying idempotency.
The full suite passes 55/55; Ruff and strict mypy pass. No GPU work was performed.

Ran `full-primary-structure-v2-opacity-a9f1d5c-57a3ebea-4e7162ec` locally. It completed
8,012 normalization attempts and 4,006 hybrid selections with verified hashes and
balanced loss decomposition at all 30 P/S capacities. The regression comparison found
zero unexplained attempt or hybrid mismatches. Its compact evidence and recovery state
are checkpointed locally at `88532a5`. No GPU work was performed.

Implemented an OOB-role and render-impact extension for the codec study, plus a pinned
24-icon fixture selected exactly from the opacity census. It compares the safe q289
clamp with the same quarter-unit lattice extended outside 0..72 for rendering only;
P48/S64 covers both representations and exact style vocabularies remove truncation and
style approximation as confounds. Focused tests pass 26/26, the full suite passes
57/57, Ruff passes, and strict mypy passes. The implementation is not yet checkpointed.

## Active jobs

None. Every worker is disabled, the isolated Docker daemon has no running container,
and no authenticated remote command has been issued. The operator is keeping the Vast
GPU server off until Codex explicitly requests it; no active MojiDiff job exists.

## Artifact durability

Raw source (405 MB), audit renders/tables, and derived fixture SVGs remain on the
persistent workspace. Curation, representation, typed-codec, and aligned-grid compact
evidence is versioned locally through `2099f3d`; the 26 MB full-primary report is
versioned at `1063fc7`, the opacity codec at `e31273c`, and the 64 KB opacity evidence
at `86f7d03`; the 32 MB full-primary opacity report is versioned at `88532a5`. Nothing
was pushed. The opacity probe's 260 KB derived SVGs are reproducible but local-only. No
external artifact sink is
configured or verified, so bulk artifacts are not durable against loss of the
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
- The prior Vast endpoint may be stale after shutdown. Once the next server exists, the
  exact current SSH username, public host, and one exact SSH port are required. Under
  the current contract, per-alias `StrictHostKeyChecking accept-new` is sufficient for
  first use; no out-of-band fingerprint is required. No host key has been trusted and
  no login has been attempted.
- The reported 100 GB attached volume has no recorded mount path or persistence
  guarantee and therefore is not yet treated as the configured durable artifact sink.
- `owned_gpu` is disabled; `resource_cap.max_steps` and `max_storage_gb` are null.
- `a100_cluster` is disabled; `workspace_root`, `namespace`, `service_account`, `pvc`,
  `resource_cap.max_jobs`, `max_steps_per_job`, and `max_storage_gb` are null.
- The template SSH alias names exist in local authorization state, but no real named
  Vast alias exists in `~/.ssh/config`. A local SSH key is present, but its username and
  endpoint have not been recorded or used.

## Next smallest evidence-producing action

Checkpoint the OOB probe implementation, register its exact config/fixture identities,
then measure the q289 safety projection over all 24 affected icons. Keep the GPU server
off during this CPU-only Gate C probe.
