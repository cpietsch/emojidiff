# Reproducing the OpenMoji curation artifacts

The primary source is OpenMoji 17.0.0 at commit
`f9fc506a3f913be9897ab0181d611d4c910a4104`. The raw checkout is never normalized,
edited, or deleted. Acquisition hashes every color SVG and removes write bits after
verification.

The control plane needs Python 3.12, the locked `uv` environment, and Cairo 1.18. On
Ubuntu 24.04 the native render dependency used for the recorded audit is:

```bash
sudo apt-get install --no-install-recommends libcairo2
uv sync --all-extras
```

Reproduce each layer with:

```bash
uv run mojidiff-curate acquire --config configs/data/openmoji-17.0.0.yaml
uv run mojidiff-curate audit --config configs/data/openmoji-audit-v2.yaml
uv run mojidiff-curate curate --config configs/data/openmoji-curation-v3.yaml
uv run mojidiff-curate curate --config configs/data/openmoji-curation-v4-reviewed.yaml
uv run mojidiff-curate verify \
  --source-config configs/data/openmoji-17.0.0.yaml \
  --decisions data/manifests/openmoji-17.0.0-curation-v4-reviewed-decisions.parquet \
  --output reports/corpus/openmoji-17.0.0-curation-v4-reviewed/verification.json
```

Audit v1 is deliberately retained. It over-rejected two safe same-document SVG
references and is an artifact of a falsified safety rule. Audit v2 permits only bounded
fragment references and rejects scripts, event attributes, external URLs, embedded
data, resource-bearing elements, excessive input bytes, and excessive element counts.

Numeric thresholds nominate only the observed 0.5% corpus tails. The reviewed manifest
is anchored to the exact candidate set and contact-sheet hashes. Exact 72 px duplicate
clusters were independently confirmed identical at 18 px; one canonical row is kept
for frequency accounting and every alias remains in a reversible named manifest.
