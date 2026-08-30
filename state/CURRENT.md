# Current research state

Updated: 2026-08-30T06:01:12Z

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

The complete OOB probe confirms that all 37 hybrid excursions are cubic Bezier control
handles; no visible endpoint is outside 0..72. Median clamp-only drift is small, but
the tail falsifies viewBox clamping as a safe primary rule. At 18 px, five semantic
icons exceed 0.001 RGBA MAE and two exceed 0.01. `1F4AB` is visibly broken (MAE
0.03812, alpha IoU 0.8641), while clamping the eye/speech-bubble icon removes most of
its interior (MAE 0.02015). Outlining does not repair those tail cases; this falsified
viewBox clamping and motivated the separate bounded control-handle vocabulary below.

The role-typed follow-up resolves that blocker on the complete pinned set. Q289
endpoints over [0,72] plus q417 quadratic/cubic controls over [-8,96] encode all 48
semantic/outlined programs strictly with zero projection, and every output SVG exactly
matches the prior unclamped quarter-grid counterfactual. The two catastrophic clamp
tails recover. This is the leading coordinate codec; model and corruption masks must
retain the endpoint/control distinction.

The first categorical style study derived exact weighted-relative-L1 K32 and K48 width
vocabularies from all 30,222 stroked contours, then rendered a pinned 35-icon worst-tail
fixture against an exact-style control. K48 strongly improves K32: full-corpus maximum
relative width error falls from 22.55% to 9.09%, and maximum 18 px style-only MAE falls
from 0.01264 to 0.001698. But the hypothesis of a negligible tail is false at 72 px:
K48 maps rare width 4.1 to 4.0, producing 0.01348 style-only MAE on `1F4AF`. K48 is the
statistical base, not yet the final vocabulary; the next falsifiable correction is one
exact 4.1 render-tail sentinel.

The sentinel follow-up supports the smallest correction. On the unchanged 35-icon
fixture, K48+exact-4.1 cuts maximum style-only MAE from 0.01348 to 0.002077 at 72 px and
from 0.001698 to 0.001510 at 18 px, with no new tail and byte-identical rerun artifacts.
The leading style policy is now K48+1 widths, all six observed dash patterns exact, and
five semantic miter values with near-10 literals mapped to 10. K32 remains the compact
ablation. Gate C is open only on the extremely sparse fixed-capacity policy.

The full-corpus capacity study resolves the last Gate C question. Packed P80/T1216 is
exact for all 4,006 programs with 1,296 logical slots versus 36,960 for dense P96/S384,
a 28.52x reduction. Four exact nested buckets place 3,359/553/88/6 icons and achieve
53.81% aggregate utilization. The selected representation is semantic-stroke with the
60-case reason-coded outlined fallback, role-typed coordinates, K48+1 styles, and
packed capacity. Gate C is complete; Gate D packed-codec and renderer stress remains.

Gate D now passes. Reversible packed conversion, canonical expansion validation, a
serializer-only XML allowlist, and a resource-limited subprocess renderer are
implemented. The corrected registered stress run completed 200 stable random packed
round trips, rejected 2,000/2,000 invalid mutations across ten families, classified
three malformed typed-XML cases, and completed 25 isolated renders including an actual
80-path/1,216-segment boundary program. The first mutator-boundary failure is preserved.

Gate E implementation was checkpointed and has now been run. The local CPU harness
uses the selected packed codec, a fixed-topology geometry-only bidirectional transformer,
role-aware q289/q417 corruption, four hash-pinned diverse icons, multiple disjoint
training and held-out corruptions, predeclared accuracy/loss criteria, isolated paired
renders, and exact continuous-versus-checkpoint-resumed comparison. This is explicitly a
diagnostic for geometry learnability, not evidence for topology generation or a final
diffusion process.

The first Gate E run falsified its predeclared held-out recovery thresholds while
confirming memorization and exact semantic resume. Both cases reached 100% training
accuracy, but one-icon held-out accuracy was 93.57% versus 95%, and diverse-four was
73.82% versus 85%. The 160-step diverse checkpoint reproduced the complete continuation
loss sequence and final model tensors exactly. An identical full rerun then failed the
artifact contract because legacy `torch.save` container bytes changed, so v1 is retained
as failed rather than silently accepted or tuned.

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

Implemented and checkpointed the OOB-role/render-impact harness at `bc6459b`, then ran
`oob-control-probe-v1-bc6459b-16c43292-8469ae8f`. All 48 semantic/outlined programs
normalize, encode, decode, serialize, and render stably with no truncation or style
approximation. A second full invocation reproduced every immutable output exactly.
Visual inspection confirms catastrophic tail deformation for the two worst icons.
Focused tests pass 26/26, the full suite passes 57/57, Ruff passes, and strict mypy
passes. The negative result and compact report are checkpointed locally at `45ca009`.
No GPU work was performed.

Implemented and checkpointed role-typed coordinates at `24461b3`, then ran
`control-coordinate-vocabulary-v1-24461b3-4c2f9833-8469ae8f`. All 48 programs are
strict-lossless and stable; all counterfactual SVG hashes match; semantic median MAE is
0.002067/0.002752 at 72/18 px and worst 18 px MAE is 0.005159. A second complete
invocation reproduced every artifact. The full suite passes 59/59; Ruff and strict
mypy pass. The compact result and recovery state are checkpointed locally at `d2b4a1c`.
No GPU work was performed.

Implemented the deterministic corpus/style-tail harness at `7925022`, then ran
`style-vocabulary-v1-7925022-778e6dc5-9b9b1699`. All 105 exact/K32/K48 programs are
stable with no structural loss or coordinate projection. Full-corpus analytics and the
35-icon exact-style-controlled render fixture falsify unaugmented K48 only in the rare
4.1-width 72 px tail described above. A second invocation reproduced all artifact
hashes. The full suite passes 61/61; Ruff and strict mypy pass. No GPU work was
performed.

Implemented pinned-fixture and explicit width-sentinel support at `ecba341`, then ran
`style-vocabulary-v2-render-sentinel-ecba341-858e7513-b13fecb6`. All 105 programs are
stable and the exact 4.1 token removes the falsified 72 px tail without exposing a new
one. A second invocation reproduced all artifact hashes. The full suite remains 61/61;
Ruff and strict mypy pass. No GPU work was performed.

Implemented and checkpointed the capacity audit at `bc52408`, then ran
`capacity-layout-v1-bc52408-6c86cc30-9b9b1699`. All 4,006 contour-length decompositions
balance; packed P80/T1216 and all adaptive assignments are exact; detail and assignment
cardinalities are 24,036 and 4,006. A second invocation reproduced all hashes. The full
suite passes 62/62; Ruff and strict mypy pass. No GPU work was performed.

Implemented reversible packed conversion at `f8b1601`, isolated typed-SVG rendering at
`82c48ec`, and a deterministic Gate D stress harness at `e6c100b`. The unit suite passes
90/90 with 20 seeded random packed round trips and real subprocess rendering. Registered
stress run `packed-render-stress-v1-e6c100b-df13aed7-005dc6b4` failed before artifacts:
the harness used token 289 as an invalid q289 endpoint, but valid coordinate tokens are
1..289. The failure is preserved in the registry; the mutator is corrected to 290 for
a new run identity. No GPU work was performed.

Corrected stress run `packed-render-stress-v1-d5ca696-df13aed7-005dc6b4` passes all
checks above and reproduced every report hash on a second invocation. Packed conversion
is at `f8b1601`, renderer isolation at `82c48ec`, the stress harness at `e6c100b`, and
the boundary correction at `d5ca696`. The full suite passes 90/90; Ruff and strict mypy
pass. Gate D is complete. No GPU work was performed.

Installed pinned CPU-only PyTorch 2.8.0+cpu in the local `.venv`, recorded the explicit
CPU package index and lockfile, and checkpointed the Gate E harness at `49e6e6a`. Its
fixture hash is `32a80ab576a6d5a435e859d38c1ba25e302070e92caae2971b517fe42c4b0d79`
and config hash is `bf71ca62d7f3e3a3a07aa0a6caeb78039785badf2ca63d0884ccf030909769c3`.
All four fixtures load without truncation or projection. The focused learning tests pass
2/2, the full suite passes 92/92, Ruff passes, and strict mypy passes. No registered
learning run or GPU work has occurred yet.

Registered and ran `tiny-geometry-v1-3493eff-bf71ca62-32a80ab5` on local CPU. The first
execution finished within its 800-step and 0.1 GB bounds and produced the negative
learning result above. Its idempotency rerun failed closed on differing checkpoint
container bytes. Compact metrics, renders, summary, run record, and the original 2.9 MB
checkpoint are preserved; no remote action or external spend occurred.

Replaced only the v1 checkpoint container under commit `0b8535b`: v2 stores a
canonically ordered JSON tree and bounded pickle-free `.npy` tensors in a ZIP with fixed
metadata, then restores model, optimizer, RNG, and step. A focused round trip proves
independent saves and load-resave are byte-identical. The full suite passes 93/93;
Ruff and strict mypy pass. The learning fixture, architecture, seeds, corruption,
steps, and predeclared scientific thresholds are unchanged in the v2 config, whose hash
is `75d558c97c106c127f026a229613afcde35da13cf7668c2c950c4b1c5d955ebb`.

Registered and completed `tiny-geometry-v2-47811d0-75d558c9-32a80ab5`. Both complete
invocations reproduced every learning metric and final model hash from v1. The canonical
checkpoint, summary, metrics, render metrics, Markdown, and trajectory are byte-identical
across reruns. V2 therefore resolves artifact idempotency but intentionally preserves
the missed held-out thresholds; Gate E remains open.

Implemented the metric-only v3 diagnostic at `a10ef83`. It leaves learning unchanged
and partitions legal-coordinate accuracy by whether each `x_t` token actually differs
from `x_0`. The partition sums back to the existing total in tests. Its config hash is
`ff42f4dd0b8915f00d8deea020b46e6891a588751e651d50c1e75bced3135a30`.
The full suite remains 93/93; Ruff and strict mypy pass.

Registered and completed `tiny-geometry-v3-93ed354-ff42f4dd-32a80ab5` twice with exact
artifact identity. Diverse-four changed-token accuracy is 58.17%, 15.65 points below
aggregate, while retained-token accuracy is 82.52%. One-icon changed/retained accuracy
is 88.59%/96.11%. The diagnostic hypothesis passes and localizes the next question to
corruption-pattern coverage rather than basic memorization or checkpointing.

## Active jobs

None. Every worker is disabled, the isolated Docker daemon has no running container,
and no authenticated remote command has been issued. The operator is keeping the Vast
GPU server off until Codex explicitly requests it; no active MojiDiff job exists.

## Artifact durability

Raw source (405 MB), audit renders/tables, and derived fixture SVGs remain on the
persistent workspace. Curation, representation, typed-codec, and aligned-grid compact
evidence is versioned locally through `2099f3d`; the 26 MB full-primary report is
versioned at `1063fc7`, the opacity codec at `e31273c`, and the 64 KB opacity evidence
at `86f7d03`; the 32 MB full-primary opacity report is versioned at `88532a5`, and the
216 KB OOB compact report at `45ca009`. Nothing was pushed. The OOB probe's 812 KB
derived SVGs and opacity probe's 260 KB derived SVGs are reproducible but local-only.
The 152 KB role-typed coordinate report is versioned at `d2b4a1c`; its 500 KB derived
SVGs are reproducible and local-only. No external artifact sink is
configured or verified, so bulk artifacts are not durable against loss of the
control-plane volume and meaningful GPU work remains blocked.

The compact style-vocabulary, capacity, and packed/render stress evidence is versioned
locally through `791af1f`; their reproducible bulk SVG/raster derivatives remain ignored.
The Gate E implementation, fixture, config, and CPU dependency lock are versioned at
`49e6e6a`, and the v1 negative compact report is versioned at `471899a`. Its 2.9 MB
resume checkpoint is reproducible in model/optimizer content but not byte-stable in the
legacy PyTorch container and remains local-only. Nothing was pushed.

The v2 compact report is versioned locally at `2d75672`. Its canonical 3.0 MB checkpoint
is verified byte-stable but remains local-only because no external artifact sink exists.

The v3 compact diagnostic is pending a local Git checkpoint; its canonical 3.0 MB
checkpoint is byte-stable and remains local-only.

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

Implement deterministic per-step corruption resampling with the same batch size,
probability, model, held-out draws, optimizer steps, and exact resume boundary. Compare
against v3 changed/retained accuracy under a new run identity. Keep the GPU server off.
