---
title: MojiDiff vectorise
emoji: ✏️
colorFrom: yellow
colorTo: blue
sdk: docker
app_port: 7860
license: cc-by-sa-4.0
short_description: Paint on an emoji; a 9M model redraws it as an SVG program
models:
  - chrispie/mojidiff-checkpoints
pinned: false
---

# MojiDiff: edit pixels, re-vectorise

The page shows a held-out OpenMoji icon as pixels. Paint on it, erase, or draw from
scratch; every change is sent to MojiDiff's render-to-SVG model (v9 colour, about 9
million parameters), which transcribes the canvas back into an editable SVG program in
MojiDiff's typed codec. You get the SVG, the model's time, and how many decoder calls it
took. Nothing is looked up: held-out icons are ones the model never trained on.

## This Space

- Runs on the free CPU tier (2 vCPU, 16 GB RAM), float32, one request at a time. Measured
  in this image on CPU, the container limited to 2 CPUs (`docker run --cpus=2`, no GPU),
  on the first 40 held-out icons the page lists: greedy decoding takes a median 731 ms per
  icon (10th to 90th percentile 371 to 983 ms). Best of 8, which samples seven more
  candidates and keeps the one that redraws the canvas best, takes a median 3.3 s on the
  first 10 (up to 11.7 s on a dense icon). Hugging Face's own vCPUs may be slower. For
  comparison, the training machine's demo decodes v7 (the same architecture) greedily in
  a median 381 ms per icon on an RTX 4080 with CUDA graphs.
- Starts in about 2 s and uses 0.25 to 0.5 GB of RAM.
- The image fetches the OpenMoji 17.0.0 sources at their pinned tag (each file is checked
  against the corpus ledger's sha256) and v9's checkpoint from
  [chrispie/mojidiff-checkpoints](https://huggingface.co/chrispie/mojidiff-checkpoints),
  and builds the icon cache, so startup only loads.
- `space-source.json` names the MojiDiff commit this Space was assembled from
  (`scripts/build_space.py vectorise <dir>`).

## Links

- Code and research log: [github.com/cpietsch/emojidiff](https://github.com/cpietsch/emojidiff)
- Weblog: [cpietsch.github.io/emojidiff](https://cpietsch.github.io/emojidiff/)
- Checkpoints: [chrispie/mojidiff-checkpoints](https://huggingface.co/chrispie/mojidiff-checkpoints)
- Every model side by side:
  [chrispie/mojidiff-gallery](https://huggingface.co/spaces/chrispie/mojidiff-gallery)

MojiDiff started from a handoff package for `gtc`, a Codex control-plane machine meant
to dispatch work to rented GPUs. That was only a starting point and an inspiration; the
project has since evolved into something else.

## Licence

CC BY-SA 4.0, like OpenMoji: code, model and this Space. The full licence text is in
`LICENSE`. The model is trained on [OpenMoji](https://openmoji.org/) 17.0.0, and every
icon shown here is derived from OpenMoji. All emojis designed by
[OpenMoji](https://openmoji.org/) – the open-source emoji and icon project. License:
[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).
