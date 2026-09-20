# Gate G dominant-bucket CPU pipeline smoke

Status: completed twice with exact artifact identity.

The complete 4,006-row hybrid and capacity ledgers joined one-to-one. The exact
P32/T128 bucket contains 3,359 icons split into 2,681 train, 339 validation, and 339
test rows with no variant-family crossing those splits. The bounded smoke selected four
train and two validation rows, performed one conditioned factorized-geometry step, and
restored its canonical 7,075,309-byte checkpoint exactly.

The first-step metrics are intentionally near random and are not a learning result.
The evidence is pipeline execution, deterministic recovery, and exact locked-path
preservation. This remains fixed-topology geometry-only work, not unconditional SVG
generation.
