---
title: MojiDiff gallery
emoji: 🧩
colorFrom: yellow
colorTo: green
sdk: docker
app_port: 7860
license: cc-by-sa-4.0
short_description: Every MojiDiff emoji-to-SVG model side by side, on CPU
models:
  - chrispie/mojidiff-checkpoints
pinned: false
---

# MojiDiff gallery

Every model of MojiDiff's render-to-SVG and latent phase behind one page, so you can
compare them by eye on the same input. MojiDiff trains small models (9 to 10 million
parameters each; the flow prior adds 9.7 million to its frozen canvas latent) that turn OpenMoji-style emoji into editable SVG programs in a typed
codec. Every output on this page is decoded under that codec's grammar; nothing is
retrieved, and only held-out icons (validation and test, which no model trained on)
are offered.

- **Transcribers** (v1 to v9, the overfit fixture): pick a held-out icon, paint on it or
  draw from scratch, and each model transcribes the canvas into an SVG program, greedily
  or best of 8 by render-and-compare.
- **Latent models** (three VAEs, the canvas latent and its flow prior): reconstruct a
  held-out icon, interpolate between two, or decode samples from the prior.

The numbers next to each model come from its run record. Pixel error is measured on the
339 validation icons; lower is better, and the nearest training icon scores 0.090.

## This Space

- Runs on the free CPU tier (2 vCPU, 16 GB RAM), float32, one decode at a time across
  models. The page shows each request's model time. Measured in this image on CPU, the
  container limited to 2 CPUs (`docker run --cpus=2`, no GPU), on the first 40 held-out
  icons the page lists: v9 greedy transcription takes a median 729 ms per icon (10th to
  90th percentile 371 to 991 ms). Best of 8 is several times slower (12 s on one dense
  icon whose greedy decode took 2.1 s); a latent reconstruction takes 1 to 2 s, and 7
  interpolation frames or 8 prior samples 3 to 12 s. Hugging Face's own vCPUs may be
  slower. For comparison, the training machine's gallery decodes v7 (the same
  architecture) greedily in a median 381 ms per icon on an RTX 4080 with CUDA graphs.
- All 15 models in `configs/gallery/models.yaml` load at startup, in about 4 s, using
  about 0.9 GB of RAM (1.2 GB after every model has run).
- Ratings are switched off: the operator's ratings are a research record kept on the
  training machine, so this copy accepts none and hides the rating controls.
- The image fetches the OpenMoji 17.0.0 sources at their pinned tag (each file is checked
  against the corpus ledger's sha256) and the checkpoints from
  [chrispie/mojidiff-checkpoints](https://huggingface.co/chrispie/mojidiff-checkpoints),
  then builds the icon and corpus caches, so startup only loads.
- `space-source.json` names the MojiDiff commit this Space was assembled from
  (`scripts/build_space.py gallery <dir>`).

## Links

- Code and research log: [github.com/cpietsch/emojidiff](https://github.com/cpietsch/emojidiff)
- Weblog: [cpietsch.github.io/emojidiff](https://cpietsch.github.io/emojidiff/)
- Checkpoints: [chrispie/mojidiff-checkpoints](https://huggingface.co/chrispie/mojidiff-checkpoints)
- The edit-pixels, re-vectorise demo:
  [chrispie/mojidiff-vectorise](https://huggingface.co/spaces/chrispie/mojidiff-vectorise)

MojiDiff started from a handoff package for `gtc`, a Codex control-plane machine meant
to dispatch work to rented GPUs. That was only a starting point and an inspiration; the
project has since evolved into something else.

## Licence

CC BY-SA 4.0, like OpenMoji: code, models and this Space. The full licence text is in
`LICENSE`. The models are trained on [OpenMoji](https://openmoji.org/) 17.0.0, and
every icon shown here is derived from OpenMoji. All emojis designed by
[OpenMoji](https://openmoji.org/) – the open-source emoji and icon project. License:
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/). The v8 model was also
trained on adapted [Twemoji](https://github.com/jdecked/twemoji) graphics
([CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)).
