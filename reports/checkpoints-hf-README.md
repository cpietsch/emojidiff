---
license: cc-by-sa-4.0
tags:
- svg
- emoji
- openmoji
- vector-graphics
---

# MojiDiff checkpoints

Research checkpoints from MojiDiff: small models, trained from scratch on an RTX 4080,
that write OpenMoji-style emoji as SVG programs in a typed codec. These are research
artifacts with honest, mostly negative results, not a polished release. Loading them
needs the MojiDiff code (`mojidiff.learning.*`) from
[github.com/cpietsch/emojidiff](https://github.com/cpietsch/emojidiff), which also holds
every run record and the research log
([weblog](https://cpietsch.github.io/emojidiff/)). Live demos:
[gallery](https://huggingface.co/spaces/chrispie/mojidiff-gallery) and
[edit pixels, re-vectorise](https://huggingface.co/spaces/chrispie/mojidiff-vectorise).

## Layout

- `runs/<run_id>/best.pt`: the checkpoint selected by the run's declared rule;
  `latest.pt` is the resumable end state (with optimizer); `best-any.pt` is the best
  without the selection constraint. `run.yaml` and `result.md` record the config,
  hashes, criteria and results of each run.
- `legacy/<name>/checkpoint.zip`: first-phase models (manifest plus numpy tensors).
- `checkpoints-hf.json`: sha256 and size of every file.

## Main models

| run | what | validation pixel error (72 px) |
| --- | --- | --- |
| `r2s-full-v7-systems-*` | render-to-SVG transcriber, 9.0M parameters | 0.077 greedy, 0.057 best of 8 |
| `r2s-full-v9-colour-*` | v7 with palette permutation cut from 0.5 to 0.1 | 0.072 |
| `vt-v1-*`, `vt-v2-c16-*` | variational canvas latent on v9 (c = 8, 16) | 0.083, 0.081 from the posterior mean |
| `lfp-v1x-exploratory-*` | rectified-flow prior over vt-v1's latent | samples: 18% fragments, CLIP recall 0.42 |
| `latent-v1/v2/v3-*` | single-vector program latents | baselines |

Retrieval of the nearest training icon scores 0.090 on the same split, and 0.085 when
mirrored training renders are allowed (the baseline for the `vt` rows).

## Data and licence

Trained on [OpenMoji](https://openmoji.org/) 17.0.0. All emojis designed by OpenMoji –
the open-source emoji and icon project. License:
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
`r2s-full-v8-twemoji-*` was also trained on adapted
[Twemoji](https://github.com/jdecked/twemoji) graphics
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)). The checkpoints, like the
MojiDiff code and documentation, are shared under CC BY-SA 4.0 with attribution to both
projects.
