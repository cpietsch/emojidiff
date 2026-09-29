# lfp-v1x-exploratory-81eb2e2-5ae4d777-47646604

**Hypothesis.** A rectified-flow prior fitted to the aggregate posterior of the frozen canvas latent (vt-v1, c = 8) replaces its N(0, I) prior, whose draws decode to fragments, and so generates whole, novel OpenMoji-like drawings and interpolates through plausible drawings. The prior is a 9.7M-parameter DiT over standardised (8, 18, 18) grids (x_t = (1 - t) x_0 + t eps, velocity target, logit-normal t, EMA 0.999), trained on posterior draws of 32 exact variants per training icon (no compositions) and selected by EMA flow loss on the validation icons' latents. Samples: 50 Euler steps from noise scale 1 to z_hat_0, greedy IEEE float32 decoding by the VT. Scored afterwards by latent_metrics through the gallery backend canvas-flow on its fixed sets (339 samples, seed 23; 32 validation pairs x 9 frames, rng 17; CLIP ViT-B/32 at 72 px against the 339 validation renders; bootstrap intervals, paired where paired; copy rule without-twins): B1-B3 on the report with default settings (its samples block); B4 and B5 on the report run with --interpolation backend, whose interpolation block is canvas-flow's own path, the slerp below (the report records interpolation.path slerp-through-prior-noise). The default report's interpolation block is the frozen VT's lerp of posterior means and scores neither B4 nor B5. Pass requires B1-B3; B4 and B5 are the interpolation claims. (B1) fragment rate at most 10%. (B2) CLIP k-NN precision at least 2x the best latent-v2 prior row (N(0, I) or refit), and recall above it, both with intervals excluding zero. (B3) copy rate at most 10% and all 339 samples distinct. (B4) slerp through the prior's noise space (reverse-ODE inversion of both posterior means, slerp, forward integration): interior fragment rate at most 15%, jump share below latent-v2's (paired over pairs, interval excluding zero), detour rate at most 20%. (B5) slerp interior precision above v9 on pixel crossfades of the same pairs, paired, interval excluding zero; a failure is recorded as "the latent adds nothing over crossfade plus transcriber". Decision: B1-B3 pass - the gallery's main latent model. B3 fails - add compositions to the prior's data, then an earlier-stopped prior. B1 or B2 fails while B3 passes - a larger prior. Only B5 fails - keep the model; its interpolation is a dissolve. Reported without a criterion: validation flow loss against the Gaussian velocity, vt-v1's N(0, I) samples on the same seeds, lerp strips, watch-it-draw strips, ms per sample.

## Criteria

B1-B5 are scored after this run by `latent_metrics` through the gallery backend
`canvas-flow` (339 samples, 32 pairs, CLIP, copy rule without-twins): B1-B3 on its
default report, B4 and B5 on its `--interpolation backend` report, whose
interpolation block is the slerp through the prior's noise (the default report's is
the frozen VT's lerp). The numbers below are this run's own checks, not the criteria.

| criterion | claim |
| --- | --- |
| B1 | fragment rate at most 10% |
| B2 | CLIP precision at least 2x the best latent-v2 prior row, and recall above it, both with intervals excluding zero |
| B3 | copy rate at most 10% under copy rule without-twins, and all samples distinct |
| B4 | slerp interior fragments at most 15%, jump share below latent-v2's (paired), detours at most 20% |
| B5 | slerp interior CLIP precision above v9 on crossfades, paired |

## Result (IEEE float32 greedy decoding (TF32 off: matmul and cuDNN); pixel metrics at 72 px)

| measure | value |
| --- | --- |
| validation flow loss, EMA (served) | 1.2587 |
| validation flow loss, Gaussian velocity (zero parameters) | 1.7629 |
| selected step | 38000 |
| flow samples distinct / rendered | 64 / 64 of 64 |
| flow samples with ink < 0.10 | 0.141 |
| parent VT, same N(0, I) seeds, ink < 0.10 | 0.328 |
| validation icons with ink < 0.10 | 0.021 |
| slerp interior frames with ink < 0.10 | 0.089 |
| lerp interior frames with ink < 0.10 | 0.071 |
| slerp endpoint pixel error (inversion round trip) | 0.0923 [0.0678, 0.1188] |
| lerp endpoint pixel error (VT from mu) | 0.0875 [0.0627, 0.1159] |

## Resources

| measure | value |
| --- | --- |
| device | NVIDIA GeForce RTX 4080 |
| flow parameters (trained) | 9,747,744 |
| served parameters (flow + frozen VT) | 18,779,986 |
| train seconds | 1179 |
| peak VRAM GiB | 3.4414305686950684 |
| flow + graph decode, ms per sample, batch 1 | 1048.4 |
| flow alone, ms per sample, batch 1 | 60.3 |
| batched, ms per sample | 143.0 |
| torch / CUDA | 2.14.0a0+4fdf77b940.nv26.08 / 13.4 |

Sheets: `prior-samples.png`, `vt-normal-samples.png` (the parent's own prior on the
same seeds), `slerp-interpolations.png` and `lerp-interpolations.png` (A, 9 frames,
B), `watch-it-draw.png` (z_hat_0 at 8 flow times, then the sample).
