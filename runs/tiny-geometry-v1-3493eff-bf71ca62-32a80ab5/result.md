# Tiny geometry v1 result

Failed the predeclared learning criterion and the artifact-idempotency rerun.

Both cases reached 100% training-token accuracy. One-icon held-out accuracy was
93.57% against a 95% threshold; diverse-four held-out accuracy was 73.82% against an
85% threshold. The diverse checkpoint continuation reproduced every loss and final
model tensor exactly.

The identical full rerun then failed closed because legacy `torch.save` checkpoint
container bytes differed. The original checkpoint and all first-execution evidence are
preserved. This does not invalidate the learning metrics, but it prevents v1 from
passing the project's reproducible-artifact contract.
