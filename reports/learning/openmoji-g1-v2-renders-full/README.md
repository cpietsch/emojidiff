# Held-out render probe

Checkpoint step 840, 128 held-out icons, corruption probability 0.35.

`x_0` is the clean program, `x_t` the raw corrupted state the model is given, and `x_hat_0` the model's predicted clean state. No safety projection or constrained decoding is applied.

| state and size | median RGBA MAE against x_0 | renders |
| --- | ---: | ---: |
| x_hat_0-18 | 0.144372 | 128 |
| x_hat_0-72 | 0.141215 | 128 |
| x_t-18 | 0.177560 | 128 |
| x_t-72 | 0.168048 | 128 |

This probe trains nothing and changes no learning artifact.
