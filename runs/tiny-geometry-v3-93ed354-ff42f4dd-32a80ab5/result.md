# Tiny geometry v3 result

Completed reproducibly; the predeclared diagnostic passed.

For diverse-four, actually-changed-token accuracy is 58.17%, 15.65 points below the
73.82% aggregate. Retained-token accuracy is also limited at 82.52%, so the model often
overwrites coordinates already correct in `x_t`. The one-icon case is much stronger:
88.59% changed and 96.11% retained accuracy. A complete rerun reproduced all artifacts.

This supports testing broader deterministic corruption coverage at the same batch size,
model, probability, and optimizer-step count before increasing capacity or steps.
