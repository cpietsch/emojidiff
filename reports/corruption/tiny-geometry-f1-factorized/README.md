# Tiny fixed-topology geometry learning proof

PyTorch 2.8.0+cpu on deterministic CPU; 241072 parameters.

| case | train accuracy | held-out accuracy | changed accuracy | retained accuracy | resume exact | passes |
|---|---:|---:|---:|---:|---:|---:|
| one-icon | 0.9798 | 0.9933 | 0.9873 | 0.9963 | None | False |
| diverse-four | 0.9878 | 0.9876 | 0.9696 | 0.9976 | True | True |

This is a fixed-topology, geometry-only diagnostic. It is evidence that the packed pipeline can learn and resume; it is not evidence for the final corruption process or unconditional generation.
