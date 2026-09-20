# Held-out render probe

Checkpoint step 840, 32 held-out icons, corruption probability 0.1 (trained at 0.35).

`x_0` is the clean program, `x_t` the raw corrupted state the model is given, and `x_hat_0` the model's predicted clean state. No safety projection or constrained decoding is applied.

| state and size | median RGBA MAE against x_0 | renders |
| --- | ---: | ---: |
| x_hat_0-18 | 0.140910 | 32 |
| x_hat_0-72 | 0.139142 | 32 |
| x_t-18 | 0.094103 | 32 |
| x_t-72 | 0.081163 | 32 |

This probe trains nothing and changes no learning artifact.
