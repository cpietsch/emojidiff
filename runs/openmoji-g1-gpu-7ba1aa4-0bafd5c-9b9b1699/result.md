# Owned RTX 4080 Gate G pipeline smoke (corrected result contract)

Status: planned; no remote stage or GPU step yet.

This run is bounded to the exact one-step local pilot and stages only its six selected
raw SVGs plus the pinned palette with the committed source snapshot.

It retries `openmoji-g1-gpu-215bcb8-0bafd5c-9b9b1699`, whose GPU step, checkpoint
round trip, and locked-path check all succeeded and whose artifacts remain durable on
the worker. That run failed only because the staged wrapper digested `summary.json`
from the output root rather than the pilot's report root. The only changed factor is
that path, now derived from the resolved pilot configuration, plus a new CPU regression
test that pins the wrapper to the layout the pilot actually writes.

Because nothing that affects computation changed, this run is also a same-worker
reproduction check against the retry parent's recorded checkpoint and summary digests.
