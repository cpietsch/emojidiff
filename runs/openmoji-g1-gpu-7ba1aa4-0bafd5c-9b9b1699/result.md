# Owned RTX 4080 Gate G pipeline smoke (corrected result contract)

Status: completed.

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

## Result

The complete Gate G pipeline smoke passed on the owned RTX 4080. `device` is `cuda`,
`deterministic_algorithms` is true under `CUBLAS_WORKSPACE_CONFIG=:4096:8`, the staged
identity verified, one bounded optimizer step ran over the real dominant-bucket data,
the canonical checkpoint round-tripped, and locked paths stayed byte-exact.

Recorded values:

- train loss 15.216644, train token accuracy 0.005068 at step 1;
- validation loss 15.344566; aggregate 0.009404, changed 0.008850 over 226 changed
  fields, retained 0.009709 over 412 retained fields;
- 577,552 model parameters; `bucket-p32-t128` with 3,359 icons;
- checkpoint 7,075,309 bytes, sha256
  `4b265e5575e3aa455a0d427e340ec407eaaaf39222709305d060281ae7e453f9`;
- summary sha256 `15ede088423678eb308548481e1c348850c04bf588101d4f74fecaaa224d07b7`;
- torch 2.8.0a0+5228986c39.nv25.06, CUDA 12.9, driver 595.71.05.

These first-step accuracies are near-random by construction. They are a pipeline
liveness check, not learning evidence.

## Reproduction

A second identical invocation returned the same JSON result, so the adapter's
create-or-identical behavior holds. The checkpoint and summary digests also match the
retry parent `openmoji-g1-gpu-215bcb8-0bafd5c-9b9b1699` exactly, which makes this an
independent same-worker reproduction across separate containers rather than a single
observation.

The local CPU pilot checkpoint has the same 7,075,309 bytes but a different digest
(`d11efa00...`). That is the expected CPU/GPU floating-point difference; cross-device
artifact identity was never part of this run's contract.

## Scope

Fixed-topology, geometry-only, one optimizer step. This does not establish learning,
topology generation, or unconditional generation. It establishes that the selected
representation, data join, conditioning, corruption, optimizer, checkpoint, and
locked-edit path all execute correctly in the target GPU environment.
