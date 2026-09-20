# Held-out render probe

Checkpoint step 1020, 32 held-out icons, corruption probability 0.35 (trained at 0.05).

`x_0` is the clean program, `x_t` the raw corrupted state the model is given, and `x_hat_0` the model's predicted clean state. No safety projection or constrained decoding is applied.

| state and size | median RGBA MAE against x_0 | renders |
| --- | ---: | ---: |
| x_hat_0-18 | 0.146992 | 32 |
| x_hat_0-72 | 0.147318 | 32 |
| x_t-18 | 0.180683 | 32 |
| x_t-72 | 0.176058 | 32 |

This probe trains nothing and changes no learning artifact.
