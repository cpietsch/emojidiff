# OmniLottie zero-shot on OpenMoji (2026-10-31)

Can a pretrained image/text-to-Lottie model (OmniLottie, 4B, Qwen2.5-VL-3B based; github OpenVGLab/OmniLottie at
26131278, weights OmniLottie/OmniLottie at 10b0d6bb) animate OpenMoji icons in their style, with no training?

**Setup.** 30 validation icons (618 px renders) with a short prompt each: "A flat OpenMoji-style emoji icon with
thick black outlines and flat colors: <what>. <motion>." (`items.jsonl`). Two modes per icon: image plus prompt
(the repository's Text-Image-to-Lottie) and prompt only. Repository default sampling (temperature 0.9, top-p 0.25,
top-k 5, repetition penalty 1.01, at most 4,096 new tokens), bf16, SDPA attention, batch 1. Curiosity job 14197, one
shard per B300. Rendered with lottie-web 5.12.2 (SVG renderer) in headless Chromium, 8 frames per animation
(`scripts/`). Lottie JSON, all frames and GIFs: `/home/dev/.cache/mojidiff/omnilottie/probe-14197/`.

| | image + prompt | prompt only |
|---|---|---|
| valid Lottie | 30 / 30 | 30 / 30 |
| hit the 4,096-token limit (no end token) | 14 | 10 |
| empty render | 4 | 1 |
| recognisable or better (grade >= 2, `grades.json`) | 15 / 30 | 1 / 30 |
| faithful (grade 3) | 0 | 0 |
| any visible motion (mean frame change > 1 grey level) | 6 | 10 |
| seconds per icon, B300, median (p90) | 64.5 (128) | 57.5 (123) |
| peak VRAM | 7.9 GB | 7.9 GB |

Every animation is 16 frames at 8 fps (2 s). Decoding runs at about 34 tokens/s and a median output is about 2,050
tokens.

**What the sheets show** (`sheet-image-*.png`, `sheet-text-*.png`: input, then 8 frames).
- With the image, simple flat objects come back as recognisable outlines: heart, mushroom, bomb, leaf, seedling,
  battery, bubbles, cloud, wheel, airplane, umbrella, warning sign, cross mark. Nine of these sit on a solid
  green or navy square that is not in the input, and several lose their fills (battery, warning sign). Not one
  keeps the OpenMoji look exactly.
- Faces, animals and dense icons fail: whale, spouting whale, bird, dog and mosquito are fragments or empty, and
  most faces keep a few features only. These are the icons that ran into the token limit.
- Motion is mostly absent. The recognisable outputs are static (the red heart's transforms are all constant).
  The one clear motion that matches its prompt is the spinning wheel. The rest is pop-in or jitter.
- The prompt alone yields a different object almost every time: one recognisable umbrella out of 30.

**Grading caveat.** Graded by one grader (Claude) by eye, not blind: the grader saw each input next to the output.
