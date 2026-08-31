# Tiny fixed-topology geometry learning proof

PyTorch 2.8.0+cpu on deterministic CPU; 241072 parameters.

| case | train accuracy | held-out accuracy | changed accuracy | retained accuracy | resume exact | passes |
|---|---:|---:|---:|---:|---:|---:|
| diverse-four | 0.9279 | 0.9230 | 0.8055 | 1.0000 | True | True |

This is a fixed-topology, geometry-only diagnostic. It is evidence that the packed pipeline can learn and resume; it is not evidence for the final corruption process or unconditional generation.
