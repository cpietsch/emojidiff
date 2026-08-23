# OpenMoji curation policy

## Decision

Use a curated, reproducible subset for the primary experiment:

- exclude the OpenMoji metadata group `flags` from primary training and evaluation;
- preserve flags as the reversible named split `excluded/flags`;
- detect blank, invisible, nearly blank, out-of-bounds, corrupt, and degenerate-looking
  assets automatically;
- review all nominated assets in labelled contact sheets;
- never modify or delete the immutable upstream checkout.

Flags are not defective data. They are a scope exclusion: a large repetitive domain of
rectangular layouts, stripes, and emblems that can distort frequency, duplication, and
memorization measurements without strongly testing the intended object/character SVG
generation problem. Their separate split supports a later include/downsample ablation.

## Dataset layers

Maintain four distinct layers:

```text
data/raw/openmoji/<pinned-revision>/       immutable upstream files
data/audit/<audit-version>/                metrics, renders, and contact sheets
data/manifests/<curation-version>.parquet  one decision row per asset
data/processed/<codec-version>/            derived tensors from included assets only
```

The processed layer must be reproducible from raw data, the audit implementation, the
curation manifest, and the codec configuration.

## Audit record

Record at least these fields per icon:

- source revision, path, hexcode, Unicode sequence, annotation, group, and subgroup;
- source SVG hash and deterministic raster hashes at 72 px and 18 px;
- XML/SVG parse status and renderer status;
- visible alpha-pixel count and fraction;
- non-background/ink mass and fraction;
- visible bounding box width, height, area, and center;
- geometry inside versus outside the 72×72 viewBox;
- element, path, subpath, and segment counts;
- open versus closed path counts;
- total stroked path length and estimated filled area;
- fill/stroke/opacity usage and unique visible colors;
- connected-component count and tiny-component statistics;
- 18 px survival/legibility indicators;
- exact raster duplicate cluster and perceptual near-neighbor cluster;
- curation status, reason codes, rule version, reviewer, and note.

Store raw metrics before applying thresholds. Generate distribution plots so thresholds
are selected from corpus evidence and committed in configuration.

## Hard defect rules

The following can be marked `exclude_defect` after deterministic verification:

- SVG cannot be parsed safely;
- renderer fails under the resource limits;
- no visible painted pixel at both audit sizes;
- all painted geometry lies outside the viewBox;
- every visible shape has zero opacity, no paint, or zero geometric extent;
- source and normalized render are empty because of a confirmed asset defect.

Keep the source row and failure artifact in the manifest/report.

## Quarantine rules

Nominate, but do not automatically delete, assets with:

- extremely low visible coverage relative to the measured corpus distribution;
- a bounding box with an extremely small width, height, or area;
- only one short or open stroke and no meaningful filled region;
- visible content at 72 px that disappears or becomes ambiguous at 18 px;
- almost all geometry clipped at the viewBox boundary;
- an extreme path/segment count;
- exact or near duplication not explained by a variant family;
- normalization that materially changes the render.

These candidates receive `quarantine_auto` and one or more reason codes. Examples:

```text
PARSE_ERROR
RENDER_ERROR
NO_VISIBLE_PAINT
OUTSIDE_VIEWBOX
ZERO_EXTENT
LOW_INK
THIN_BBOX
SINGLE_OPEN_STROKE
FAILS_AT_18PX
EXCESSIVE_COMPLEXITY
EXACT_DUPLICATE
NEAR_DUPLICATE
NORMALIZATION_DRIFT
```

## Manual review

Create deterministic contact sheets grouped by reason code. Each cell shows:

- 72 px render and enlarged 18 px render;
- hexcode, annotation, group/subgroup, and filename;
- the metrics that triggered quarantine;
- raw and normalized render side by side where relevant.

Review decisions are:

- `include_override`: unusual but intentional and useful;
- `exclude_defect`: confirmed broken or visually empty;
- `exclude_policy`: valid but outside the experiment's documented scope;
- `quarantine_manual`: unresolved and excluded from training until decided.

A thin line alone is never sufficient for exclusion. Minimal Unicode symbols may be
legitimate; the contact sheet prevents the audit from confusing simplicity with a bad
asset.

## Flags policy

Apply the explicit metadata rule:

```text
group == "flags" -> exclude_policy, split = "excluded/flags",
reason = "SCOPE_FLAGS"
```

Do not use filename patterns or country-code heuristics when metadata is available.
Report the number and fraction excluded after pinning the actual OpenMoji revision.

Possible later comparisons:

- primary curated data without flags;
- flags included at natural frequency;
- flags downsampled;
- flags used only for codec/structure pretraining, not style specialization.

Do not run these comparisons until the main codec and learning pipeline are stable.

## Duplicate and family policy

Deduplicate exact SVG/raster duplicates for frequency accounting, but preserve all
metadata aliases in a mapping table. Assign near-duplicates and related Unicode variants
as a family before train/validation/test splitting. Skin-tone, gender, ZWJ, country-flag,
and alias relationships must not leak across splits.

## Required curation outputs

- immutable raw-source manifest with hashes and license metadata;
- audit table and versioned threshold configuration;
- corpus distribution report;
- all quarantine contact sheets;
- reviewed decision manifest with reason codes;
- primary curated manifest;
- separate flags manifest;
- duplicate/family mapping;
- summary counts before and after each rule;
- tests proving deterministic decisions and no missing source rows.

Model training must consume a named manifest version, never a directory glob.
