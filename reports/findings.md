# Research findings

## 2026-08-23 — Bootstrap boundary audit

**Hypothesis.** The fresh `gtc` environment is a sufficiently isolated control plane
for local reversible work under the YOLO operating contract.

**Observation.** The hostname and canonical workspace match the contract. Docker is
configured through TLS at `tcp://localhost:2376`; known host-socket paths and socket
mounts are absent, and the isolated daemon contains no workloads. The repository has no
commits and consists only of the operator handoff files. No worker is enabled, no
artifact store is configured, and no remote jobs exist.

**Decision.** Continue with local scaffold, probes, tests, and data-curation code. Do
not issue remote commands or start meaningful GPU work. Treat external Tailscale ACLs,
credential scopes, and dedicated-host provenance as unverified rather than inferring
them from local configuration.

## 2026-08-23 — Audit v1 over-rejected internal SVG references

**Hypothesis.** Rejecting every attribute containing `url(...)` would safely identify
resource-bearing SVGs without affecting legitimate OpenMoji assets.

**Observation.** The complete v1 audit retained all 4,495 source rows but classified
two regional flags as parse defects. Inspection showed only same-document references:
Ontario uses `clip-path="url(#ontario-shield)"`, and Castile-La Mancha uses marker
references. Neither accesses an external resource.

**Decision.** Keep audit v1 and its false-positive rows as evidence. Audit v2 permits
only syntactically bounded same-document fragment identifiers and continues to reject
external URLs, embedded data, scripts, event attributes, and resource-bearing tags.

## 2026-08-23 — OpenMoji 17.0.0 reviewed curation

**Hypothesis.** Corpus-tail metrics can identify actual blank or degenerate assets
without treating legitimate minimal symbols or detailed icons as defects.

**Observation.** Audit v2 safely parsed all 4,495 color SVGs. One Castile-La Mancha
regional flag (`1F3F4-E0065-E0073-E0063-E006D-E007F`) consistently fails CairoSVG at
both target sizes because it references an undefined `Dot` marker; it is retained as an
`exclude_defect` row. The empirical bottom/top 0.5% tails nominated 104 assets. Visual
review of every candidate at 72 px and enlarged 18 px found all 104 intentional and
recognizable, including thin symbols, single strokes, and complex family icons.

Exact duplicate accounting found 185 render clusters with 410 members, identical at
both sizes. A deterministic canonical non-flag row is retained per cluster; 218
noncanonical non-flag aliases are preserved in `excluded/exact-duplicates`. The 270
rows whose metadata group is exactly `flags` are preserved in `excluded/flags`.

**Decision.** The reviewed primary manifest contains 4,006 rows: 3,902 ordinary
includes and 104 reason-coded `include_override` rows. No quarantine remains. The
verification artifact confirms raw hashes and immutability, complete row coverage, no
flags in primary, no exact duplicates in primary frequency counts, and no family split
leakage. Representation complexity remains a Gate C measurement rather than a curation
defect rule.

**Reproducibility limitation.** Audit v1 and v2 were run before the repository had its
first local Git commit. Their configs and artifacts are hashed, but the exact pre-run
dirty worktree snapshot was not recorded. This violates the preferred code-identity
practice and is retained as a negative orchestration finding. A local checkpoint is
required before the next material experiment; nothing will be pushed.
