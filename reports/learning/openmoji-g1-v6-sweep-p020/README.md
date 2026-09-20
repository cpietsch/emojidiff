# Held-out render probe

Checkpoint step 1380, 32 held-out icons, corruption probability 0.2 (trained at 0.35).

`x_0` is the clean program, `x_t` the raw corrupted state the model is given, and `x_hat_0` the model's predicted clean state. No safety projection or constrained decoding is applied.

| state and size | median RGBA MAE against x_0 | renders |
| --- | ---: | ---: |
| x_hat_0-18 | 0.143659 | 32 |
| x_hat_0-72 | 0.134922 | 32 |
| x_t-18 | 0.147594 | 32 |
| x_t-72 | 0.138642 | 32 |

This probe trains nothing and changes no learning artifact.
