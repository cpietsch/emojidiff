# MojiDiff: learnings

2026-10-31, at the wrap-up. The narrative and every number are in `reports/findings.md`; this page
draws the lessons from them. Evidence for each point is named in brackets.

## What the project delivered

A 9.0M-parameter transcriber (`r2s-cur-d02-s120k-seed2-*`) that turns a 144 px OpenMoji-style render
into an editable SVG program in a typed codec. Test pixel error 0.062 greedy, 0.040 best of 8. 388 ms per
icon on the RTX 4080, about 1.2 s in Chrome (WASM, 4 threads), token-identical to Python. A browser demo
(`web/demo`) shows validation icons, uploads and a drawing pad with per-icon latency.

## Learnings

**1. Measure the simplest non-learned alternative first.** A ten-minute run of vtracer, a classical
tracer, outdid five weeks of transcriber work: 99% of icons recognisable against 52%, a third of the
pixel error, 1.6 ms on a CPU. It should have been the first measurement when the direction was chosen on
2026-09-27, not the last. [findings 2026-10-31, "A classical tracer beats the transcriber";
`reports/visual/2026-10-31-tracer-baseline/`]

**2. Look at the outputs; the metric hid how bad they were.** Pixel error ranked models correctly
(Spearman -0.55 with the visual grade), but the recipe gains of the cluster days (0.003-0.005) were
invisible, and the operator saw "mostly garbage" while the numbers improved. A blind visual grading of
all 339 validation icons took minutes with parallel graders and agreed with itself 94-98% exactly. It
belongs next to every number shown to the operator. [`reports/visual/2026-10-31-final-greedy/`]

**3. The limit was the data, not the settings.** With 2,681 training icons the model learns a library
of shapes to copy, not a skill of drawing what it sees. Every direction hit the same wall: caption-driven
generation (Gate M, September), the canvas latent and flow prior, best of N, doodles. Vampires, fairies
and older people became plain people in 35 of 36 icons, and real doodles became fragments, for the same
reason. No hyperparameter gets past it. [visual review; findings on Gate M, vt-v3 and lfp-v2x, doodle D1]

**4. At this scale the training seed matters more than most settings.** The spread between seeds (sd
about 0.005) was as large as the effects being tested. Seed-matched comparisons over eight seeds turned
contradictory unpaired readings into clean answers (dropout 0.2: +0.005, 7 of 7 pairs); before that,
earlier claims about training length and dropout had to be corrected three times. [findings 2026-10-31,
"The training seed carries the variance"]

**5. Stopping rules written before the results keep the work honest and fast.** The combined recipe
(wave 12) and doodle to emoji each stopped within minutes of their results, by rules fixed in
`state/CURRENT.md` beforehand, instead of being argued into partial successes.

**6. Synthetic augmentation only teaches invariances inside its own distribution.** Degraded OpenMoji
renders kept OpenMoji's shapes, proportions and framing, so the doodle fine-tune learned to repair shapes
it already knew; on synthetic doodles its output was still further from the clean icon than the doodle
itself, and on 130 real QuickDraw doodles it fragmented. [`reports/doodle/2026-10-31-wave-d1/`]

**7. Compute was never the bottleneck; orchestration was.** Training got 2.3x faster with byte-identical
batches and eight B300 GPUs were available, yet the idle hours came from orchestration mistakes: a wave
submitted without a waiter, a zsh loop that silently made a waiter for a bogus job id, a `pkill -f` that
killed its own shell, shared-memory crashes when other jobs started on the node. Fixes that stuck: start
the waiter in the same step as every submission, bash scripts with one job id per argument, a loader
timeout, re-collation on the shared-memory error. None of the speed mattered, because the problem was the
approach. [`reports/curiosity/speedup.md`; findings 2026-10-30/31]

**8. The goal contained a tension.** "Small, trained from scratch, OpenMoji-style SVG" is easy to satisfy
for conversion, where a tracer already does it. A learned model is only justified by something a tracer
cannot do, such as generation or editing, and that needs far more data or a pretrained prior, which
conflicts with "small and from scratch".

## What worked

- The typed codec round-trips all 4,006 curated OpenMoji icons exactly.
- In-browser decoding matches Python token for token, on WASM and WebGPU.
- For a small autoregressive model, WASM beat WebGPU: each decoder step is too small to pay WebGPU's
  per-call overhead. The JavaScript grammar costs 0.2% of decode time.
- Best of 8 (render-and-compare reranking) is the one model-side gain that shows by eye: 52% to 65%
  recognisable.
- Run records stayed rigorous throughout: run ids with hashes, a registry, audited resource and baseline
  blocks, failed runs kept.

## What I would do differently

Spend the first day on a classical tracer and a blind visual check, decide the direction from those two
numbers, and only then tune. If the project resumes, the evidence points to best of 8 in the browser, a
variations page for the families where the corpus is dense (busts, faces, couples, stick figures), or a
training signal that teaches drawing unfamiliar shapes; not more recipe arms.
