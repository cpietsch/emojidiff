# Tiny fixed-topology geometry learning proof

PyTorch 2.8.0+cpu on deterministic CPU; 241072 parameters.

| case | icons | steps | train accuracy | held-out accuracy | loss ratio | resume exact | passes |
|---|---:|---:|---:|---:|---:|---:|---:|
| one-icon | 1 | 160 | 1.0000 | 0.9357 | 0.000095 | None | False |
| diverse-four | 4 | 320 | 1.0000 | 0.7382 | 0.000053 | True | False |

This is a fixed-topology, geometry-only diagnostic. It is evidence that the packed pipeline can learn and resume; it is not evidence for the final corruption process or unconditional generation.
