# Held-out render probe

Checkpoint step 4860, 12 held-out icons, corruption probability 0.05 (trained at 0.35).

`x_0` is the clean program, `x_t` the raw corrupted state the model is given, and `x_hat_0` the model's predicted clean state. No safety projection or constrained decoding is applied.

| state and size | median RGBA MAE against x_0 | renders |
| --- | ---: | ---: |
| x_hat_0-18 | 0.022561 | 12 |
| x_hat_0-72 | 0.022350 | 12 |
| x_t-18 | 0.037480 | 12 |
| x_t-72 | 0.037302 | 12 |

This probe trains nothing and changes no learning artifact.
