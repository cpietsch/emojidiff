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
needs the MojiDiff code (`mojidiff.learning.*`), which is not public yet.

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
| `r2s-full-v9-colour-*` | v7 + palette permutation | 0.072 |
| `vt-v1-*`, `vt-v2-c16-*` | variational canvas latent on v9 (c = 8, 16) | 0.083, 0.081 from the posterior mean |
| `lfp-v1x-exploratory-*` | rectified-flow prior over vt-v1's latent | samples: 18% fragments, CLIP recall 0.42 |
| `latent-v1/v2/v3-*` | single-vector program latents | baselines |

Nearest-training-icon retrieval scores 0.085 on the same split.

## Data and licence

Trained on [OpenMoji](https://openmoji.org/) (CC BY-SA 4.0); `r2s-full-v8-twemoji-*`
also on [Twemoji](https://github.com/jdecked/twemoji) graphics (CC BY 4.0). The
checkpoints are shared under CC BY-SA 4.0 with attribution to both projects.
