# latent-v2-kl-efbd2e2-451268e6-47646604

**Hypothesis.** A variational autoencoder over codec programs, trained on the 2,681 training icons and their compositions, keeps information in its 64-dimensional latent. Pass requires (1) reconstruction pixel error on the 339 validation icons below the error of the single prior-mean decode, paired, with the 95% interval on the reduction excluding zero; (2) at least 29 of 32 prior samples distinct. Reported without a criterion: reconstruction against the nearest training icon, interpolation and prior-sample sheets, and decode latency.

| measure | value |
| --- | --- |
| validation icons | 339 |
| reconstruction pixel error | 0.1638 [0.1571, 0.1705] |
| nearest training icon pixel error | 0.0897 |
| prior-mean decode pixel error (control) | 0.1768 |
| error reduction, own latent vs prior mean | 0.0130 [0.0091, 0.0168] |
| exact reconstructions | 0.000 |
| prior samples distinct / rendered | 32 / 32 of 32 |
| prior samples, median pixel distance to nearest training icon | 0.0369 |
| parameters | 10,114,082 |
| train seconds | 3100 |
| decode ms per icon (graph, batch 1) | 454.0 |

Sheets: `reconstructions.png`, `interpolations.png`, `prior-samples.png`.
