# MojiDiff Exploratory Research Plan

## 1. Research objective

Build and study a compact categorical denoising model that generates and edits
OpenMoji-style vector graphics as typed SVG programs. Every exposed state should be
serializable to a safe, renderable SVG, while the model remains small enough for
single-GPU experimentation.

This is an experiment, not a commitment to prove that diffusion beats autoregression.
The useful outcome may be a successful diffusion model, a narrower editing/inpainting
system, a better SVG representation, or a well-supported negative result.

The central visual is a paired trajectory:

- raw current state, labelled `x_t`;
- predicted clean reconstruction, labelled `x_hat_0`.

The pair matters. At high uniform noise, `x_t` may correctly look like random static;
`x_hat_0` can reveal the model's evolving global hypothesis. Never present one as the
other, and label any grammar or safety projection.

## 2. Questions and falsifiable hypotheses

Treat each item as a question to answer with an artifact, metric, or controlled
comparison.

### Representation

1. Does a typed SVG program make arbitrary corrupted states safe and renderable without
   forcing the model to learn XML syntax?
2. Which source representation is more learnable and faithful: OpenMoji's semantic
   strokes or fully outlined paths produced by normalization?
3. What coordinate resolution preserves recognizable 18 px and 72 px renders without
   needlessly lengthening the program?
4. Is a fixed-slot representation sufficient, or does truncation erase important icon
   structure?

### Corruption and learning

5. Does path-correlated, invariant-preserving corruption learn faster or produce better
   samples than independent per-field uniform corruption?
6. Which corruption choices create informative renderable trajectories rather than
   visually meaningless static?
7. Does learning path structure and coordinates jointly help, or does a fixed-topology
   diagnostic reveal that geometry alone is the real problem?
8. Is a compact model sufficient for the data, or is performance representation/data
   limited rather than capacity limited?

### Data and conditioning

9. Does broad permissively licensed vector pretraining improve SVG competence before
   OpenMoji specialization, or dilute the target style?
10. Do structured OpenMoji labels and tags provide controllability beyond memorization?
11. Can family-aware splits expose leakage among Unicode, gender, skin-tone, and ZWJ
   variants?

### Utility and systems

12. At matched codec, data, model scale, and quality, how does iterative denoising
   compare with a properly KV-cached autoregressive baseline?
13. Is constrained inpainting or path locking a stronger use case than unconditional
   generation?
14. Which optimizations actually improve end-to-end latency or memory on a named GPU?

## 3. Scope

### Core scope

- deterministic OpenMoji ingestion and licensing manifest;
- two representation candidates and a measured selection;
- typed codec, serializer, renderer, and linter;
- factorized uniform corruption control;
- path-correlated corruption treatment;
- compact bidirectional denoising transformer;
- cached causal autoregressive baseline;
- conditional generation from structured tags;
- inpainting and locked-path editing;
- paired trajectory renderer;
- validity, fidelity, novelty, quality, latency, throughput, and VRAM evaluation;
- reproducible worker images and remote-run adapters.

### Deferred until evidence supports them

- free-text encoder;
- WebGPU/browser deployment;
- multi-node or cross-host distributed training;
- a large hyperparameter sweep;
- TensorRT or quantization before an eager PyTorch baseline is correct;
- broad pretraining before the OpenMoji-only pipeline is validated;
- a polished public interface before model and evaluation semantics stabilize.

### Explicit non-goals

- modeling arbitrary raw SVG/XML;
- claiming every internal random tensor is aesthetically meaningful;
- claiming novelty or speed superiority without a literature check and measurements;
- using raster fallback results as if they were vector-native generation;
- hiding invalid samples, unfavorable seeds, or failed hypotheses.

## 4. Compute architecture

### Control plane: `gtc`

`gtc` hosts the long-lived Codex session in `tmux`, the canonical Git repository,
run/config metadata, orchestration code, compact reports, and the research log. It must
be able to recover the complete experiment state after an SSH disconnect or worker
loss, apart from artifacts explicitly recorded as non-durable.

The host is a fresh deployment built from `cpietsch/code-server-cpu`. That environment
provides Ubuntu 24.04, Node, Python, Git/GitHub CLI, `uv`, `ripgrep`, `tmux`, SSH, a
Tailscale sidecar, and an isolated Docker-in-Docker daemon. It does not currently bake
Codex into the image, so install Codex after deployment as described in
`DEVBOX_BOOTSTRAP.md`. Set `TS_HOSTNAME=gtc` in the deployment environment.

This is a suitable YOLO control plane only while it remains dedicated and externally
hardened: no host Docker socket, no production credentials, no cloud billing API key,
and least-privilege worker/storage identities. The DinD sidecar is a disposable build
environment, not the canonical repository or artifact store.

Use:

```bash
tmux new -As mojidiff-codex
cd /home/dev/workspace/mojidiff
codex --yolo
```

Resume from the same repository directory with:

```bash
codex --yolo resume --last
```

### Worker: Vast.ai RTX 5090

Use for most single-GPU development and training when an instance has been explicitly
provisioned and enabled. Connect over the Vast public SSH mapping. The selected Vast
image is already the runtime container; do not nest Docker.

Preferred image candidate:

```text
nvcr.io/nvidia/pytorch:25.06-py3
```

Validate the actual host driver before use. This image expects a sufficiently recent
driver and includes a Blackwell-capable PyTorch/CUDA stack plus TensorRT-related NVIDIA
components. If the host driver cannot support it, select a compatible pinned image and
record the reason; never silently use `latest`.

The 5090's benchmark results are specific to that worker offer, power limit, driver,
image, and load. Record all of them. Because the worker is ephemeral and billed, its
checkpoints must be copied to durable storage and verified before shutdown.

### Worker: owned GPU machine over Tailscale

Use as a persistent development, replication, rendering, or overflow worker. Connect
through a stable SSH alias backed by Tailscale/MagicDNS. Run the versioned project
container with NVIDIA Container Toolkit and explicit data/output mounts. Keep its
results separate from the Vast 5090 benchmark unless environments are demonstrably
matched.

### Worker: A100 cluster

Use when the experiment benefits from A100 memory, cluster durability, independent
replication, or a controlled comparison. SSH is the control path; Kubernetes is the
execution path. Codex should render a Job manifest, validate it, submit it to the
configured namespace, and record the Job UID and PVC/artifact destination. Jupyter is
appropriate for interactive diagnosis but not the only process holding a training run.

Do not combine these machines into one distributed job. Scheduling independent runs is
simpler, more reproducible, and scientifically cleaner for this project.

## 5. Data, rights, and provenance

### Primary data

OpenMoji provides SVGs, raster previews, and semantic annotations. Pin an exact source
revision and build a manifest containing:

- source URL and revision;
- upstream file path and stable icon identifier;
- upstream license and attribution text;
- file content hash;
- name, group, subgroup, and tags;
- Unicode sequence and variant-family identifier;
- preprocessing/codec version;
- curation status and reason codes;
- train, validation, test, quarantine, or named-exclusion assignment.

OpenMoji is CC BY-SA 4.0. Treat attribution and downstream licensing as a first-class
research artifact. Document the chosen position on data, generated outputs, code, and
weights rather than implying legal certainty about unsettled model-weight questions.
Do not publish weights until the operator has reviewed that position.

### Curation before preprocessing

The upstream repository is immutable raw data. Build a measured audit and a reversible
curation manifest before normalization or tokenization. OpenMoji contains candidates
that render blank, nearly blank, outside the viewBox, or as degenerate single strokes.
Do not silently let them become easy low-loss training examples.

For each icon, record parse/render success, visible alpha/ink coverage, visible bounding
box, path and segment counts, fill/stroke usage, connected components, palette usage,
18 px legibility, duplicate/near-duplicate cluster, and SVG/render hashes. Use the
observed distributions to nominate outliers; do not invent one universal pixel cutoff
before seeing the corpus.

Classify each asset as:

- `include`: accepted for the primary model dataset;
- `exclude_policy`: intentionally outside scope, with a policy reason;
- `quarantine_auto`: metric/rule candidate awaiting review;
- `exclude_defect`: confirmed blank, invisible, corrupt, or unusably degenerate;
- `include_override`: unusual but legitimate icon restored by review;
- `quarantine_manual`: unresolved and excluded from training until decided.

Generate labelled contact sheets for every quarantined candidate. Thin or single-line
geometry is a review signal, not an automatic defect: some legitimate symbols are
supposed to be minimal.

Exclude `group == flags` from the primary experimental dataset. Flags form a large,
highly repetitive special domain, encourage memorization of near-rectangular templates,
and do little to test the desired object/character drawing behavior. Preserve all flags
as a named reversible split, `excluded/flags`, so a later inclusion or downsampling
ablation can measure the decision. See `DATA_CURATION.md` for the exact policy.

### Optional broad corpus

Candidate sources include Twemoji, Noto Emoji, Material Design Icons, Bootstrap Icons,
and Lucide. No source enters training based on reputation alone. For each source, verify
the exact asset license at the pinned revision, attribution obligations, redistribution
terms, provenance, and compatibility with the intended release.

Start broad-data experiments only after the OpenMoji-only pipeline is correct. This
makes pretraining a controlled question rather than an invisible confound.

### Leakage-resistant splits

Random file splits are inadequate. Group visually and semantically linked variants
before assigning a split, including:

- base emoji and skin-tone variants;
- gender variants;
- family and other ZWJ sequences;
- stylistic or Unicode aliases;
- duplicated assets and near-identical normalized geometry.

Hash the grouping logic and final manifest. Add nearest-neighbor inspection across
splits before interpreting generalization.

## 6. Representation and codec

### Design principle

Model a bounded typed drawing program, not source SVG text. A deterministic serializer
owns XML, namespaces, `viewBox`, path syntax, separators, defaults, clamping, escaping,
and safety. The model never emits arbitrary tags or attributes.

Use a fixed 72×72 coordinate system and ordered path slots. Each path slot contains:

- `path_length` in `0..S`; zero means inactive;
- layer/order value;
- fill palette entry or none;
- stroke palette entry or none;
- stroke width, cap, and join categories;
- quantized start point;
- up to `S` typed segments;
- segment type such as line, quadratic, cubic, or close;
- a fixed set of coordinate fields, with unused fields represented by typed padding.

Do not add a redundant presence bit or independent active mask. `path_length == 0` is
the sole source of path inactivity. Derived masks come from it.

### Invariants

The codec and all corruption operators must enforce:

- inactive paths have zero active segments and typed padding elsewhere;
- active segment count agrees with `path_length`;
- segment types only activate their legal coordinate fields;
- a close segment cannot be followed by active geometry in the same path;
- all coordinates and style values are in vocabulary range;
- palette references resolve;
- serialization cannot introduce scripts, external resources, event handlers, or
  unbounded complexity;
- render resource limits are enforced independently of syntax validity.

### Representation candidates

Evaluate both before committing:

1. **Semantic-stroke representation**: preserve OpenMoji fill/stroke semantics,
   normalized transforms, ordering, and style categories.
2. **Outlined representation**: use a pinned normalizer such as `picosvg` to flatten
   transforms and convert shapes/strokes into filled paths.

The first better preserves editable intent and OpenMoji conventions. The second reduces
style operations but may add many paths/segments and destroy stroke semantics. Compare
them using measured program lengths, round-trip fidelity, complexity, and downstream
tiny-model learning rather than choosing by intuition.

### Codec study

For each candidate:

- plot paths/icon and segments/path distributions, including tails;
- quantify truncation at candidate slot and segment budgets;
- compare coordinate quantization, initially 128 and 256 bins;
- render original and round-tripped images at 72 px and 18 px;
- compute raster similarity and inspect worst cases;
- record palette coverage and outliers;
- fuzz valid and invalid tensor programs;
- verify deterministic encode/decode and stable hashes.

Do not begin serious model training until a codec report establishes acceptable
round-trip fidelity, bounded lengths, and robust render safety.

## 7. Corruption processes

### Control: factorized field-uniform corruption

Implement a simple type-aware control in which each field is independently retained or
replaced by a value from its own legal vocabulary. Never draw a coordinate token for a
segment type or a color for a length. Apply deterministic safety projection before
serialization.

This control is useful precisely because it exposes the coupling problem: independently
noised length, segment type, and coordinates can yield incoherent programs even when
they remain serializable.

### Primary treatment: path-correlated corruption

Use a shared corruption gate or time for each path block, then corrupt fields subject
to structural invariants:

1. sample or corrupt `path_length` from an empirical or explicitly defined prior;
2. derive active segment masks from that length;
3. retain typed padding for inactive paths and fields;
4. corrupt only legal fields for each active segment type;
5. corrupt structural fields more conservatively than local geometry where useful;
6. combine local coordinate perturbations with occasional global replacement so the
   task spans refinement and synthesis;
7. define exactly whether style, topology, and geometry share or have separate gates.

Also test whole-path replacement from the empirical path distribution as a strong
block-level ablation.

If the correlated transition has a tractable forward distribution and exact posterior,
document the mathematics and call it a structured D3PM. If training instead predicts a
clean program under a custom noising curriculum without that posterior, call it a
structured iterative denoiser. Precision in naming is part of the result.

### Diagnostic: fixed topology

Hold path lengths and segment types fixed while noising coordinates and styles. If this
works while topology generation fails, the experiment has isolated the bottleneck and
supports a semi-autoregressive or path-template formulation.

### Trajectory evaluation

For fixed seeds, save:

- raw `x_t` render;
- projected `x_t`, if projection changes it;
- predicted `x_hat_0` render;
- final sampled state;
- field-level change counts by step;
- parser/renderer validity and safety flags.

Compare factorized and path-correlated trajectories side by side. Do not hide the
expected static-like appearance of a high-noise raw uniform state.

## 8. Models

### Denoising model

Start with a compact bidirectional transformer in roughly the 20M-parameter region, but
treat parameter count as a measured variable rather than a promise. Use:

- BF16 where supported;
- embeddings and output heads that respect field type;
- timestep/noise-level conditioning;
- structured tag conditioning;
- masked cross-entropy for legal active fields;
- loss balancing so abundant coordinate fields do not swamp topology/style;
- exponential moving average only if an ablation shows value;
- checkpointed optimizer, scheduler, scaler, RNG, data cursor, and config state.

First overfit one icon, then a tiny diverse fixture. A model that cannot do this does
not justify a larger run.

### Autoregressive baseline

Train a causal transformer on the identical information content, data splits,
conditioning, and comparable parameter/training budget. Give it a production-reasonable
implementation:

- legal-token masks;
- KV caching;
- batched decoding where applicable;
- the same serializer, safety checks, and evaluation suite.

Diffusion uses full-sequence model calls for multiple denoising iterations; AR performs
incremental cached decoding. Nominal step counts are not a speed result.

### Conditional editing

Support conditioning from OpenMoji's structured metadata before adding a text encoder.
The editing interface should permit:

- locked path blocks;
- locked style fields with geometry resampled;
- masked region/path inpainting;
- controlled tags;
- seed and denoising-strength control.

Editing may be the strongest product result even if unconditional generation is mixed.

## 9. Linter and safe rendering

Build the linter as a standalone CLI and reusable library. Separate hard grammar/safety
validity from soft OpenMoji convention scores.

Hard checks include:

- tensor/program invariants;
- safe SVG element and attribute allow-list;
- fixed viewBox and coordinate bounds;
- no external references, scripts, events, filters, or resource bombs;
- path/segment and serialized-byte complexity ceilings;
- successful sandboxed render at target sizes.

Soft checks include:

- palette membership and color contrast;
- OpenMoji-like stroke width, caps, joins, and layering;
- geometry within visual bounds;
- legibility at 18 px;
- disconnected specks, near-zero segments, and excessive self-overlap;
- style consistency across paths.

The decoder may reuse hard constraints. Soft convention checks should normally score or
warn rather than silently rewrite a sample.

## 10. Evaluation

### Codec and validity

- encode/decode exactness where applicable;
- original versus round-trip raster similarity at 72 px and 18 px;
- truncation rate and complexity distribution;
- program invariant pass rate;
- SVG parse, safety, and render success rate;
- failure taxonomy, including hidden projection rate.

### Visual quality and style

Use more than one metric:

- a perceptual embedding or image similarity metric suited to emoji renders;
- palette/stroke/layer convention scores;
- human inspection on a fixed, predeclared sample grid;
- prompt/tag adherence;
- small-icon legibility;
- uncertainty across seeds.

Avoid treating raster FID alone as decisive on a small vector corpus.

### Diversity and memorization

- nearest training and validation neighbors in raster and program space;
- exact and near-duplicate rates;
- pairwise diversity at fixed condition;
- train/validation gap by Unicode family;
- interpolation or editing consistency, labelled as such rather than presented as
  independent generation.

### Diffusion versus AR

Compare at matched codec, data, model scale, conditioning, and reasonable tuning. Sweep
denoising iterations and AR sampling settings to build a quality/latency frontier.

Report for every point:

- named GPU and worker alias;
- driver, image digest, framework versions, precision, and batch size;
- end-to-end latency distribution, including p50 and p95;
- throughput;
- peak VRAM;
- model and artifact size;
- validity and projection rates;
- quality, diversity, and adherence metrics.

Warm-up and measurement protocol must be versioned. Never merge A100 and RTX 5090
latency into a single curve.

### Optimization sequence

Only optimize a correct frozen model/configuration:

1. eager PyTorch correctness baseline;
2. BF16 and inference-mode cleanup;
3. `torch.compile` where stable;
4. TensorRT/Torch-TensorRT when operators and dynamic shapes permit;
5. Model Optimizer or lower precision as a separate quality/size experiment.

Record compile/build time separately from steady-state inference and keep failed engine
conversions as findings.

## 11. Evidence gates

These gates are ordered by dependency, not calendar. Each gate ends in a committed
artifact and a short decision note. If evidence fails, narrow or revise the next gate.

### Gate A — orchestration and recovery

Produce:

- local and remote environment probes;
- enabled worker inventory;
- durable artifact sink configuration;
- detached launch/status/sync adapter;
- successful tiny artifact round trip with verified hash;
- recovery drill from a disconnected SSH session.

Exit evidence: a worker loss would not erase the experiment ledger, and an active job
can be found without terminal history.

### Gate B — corpus and rights audit

Produce a versioned OpenMoji manifest, license/attribution report, family-aware splits,
duplicate analysis, corpus statistics, automated outlier report, labelled quarantine
contact sheets, reviewed inclusion/exclusion decisions, and a separate flags split.

Exit evidence: every sample and split is traceable and hashed; every exclusion has a
reason; the curated SVGs render successfully at 72 px and remain legible or intentionally
minimal at 18 px; no source file was deleted or modified.

### Gate C — representation study

Implement semantic-stroke and outlined codecs and create the codec report described
above.

Exit evidence: one representation is selected with explicit tradeoffs and a documented
fallback.

### Gate D — invariant and renderer stress test

Run property tests and fuzzing over random and corrupted programs. Test parser failure,
resource ceilings, malformed inputs, and renderer isolation.

Exit evidence: exposed states serialize safely, render reliably, and failures are
classified rather than swallowed.

### Gate E — tiny learning proof

Overfit one icon and a tiny diverse fixture with the denoiser. Verify checkpoint resume
reproduces the expected continuation and the sampler can recover held-out corruptions.

Exit evidence: loss, reconstruction, and renders demonstrate an end-to-end learnable
pipeline.

### Gate F — corruption comparison

Compare factorized uniform, path-correlated, fixed-topology, and whole-path replacement
under matched data and model settings.

Exit evidence: choose a primary corruption based on learning curves, validity,
trajectory behavior, and sample quality—not just aesthetics.

### Gate G — OpenMoji generation and editing

Train the selected formulation on OpenMoji, evaluate structured conditioning, and test
locked-path/style inpainting.

Exit evidence: a fixed sample set, failures, nearest neighbors, metrics, and an honest
assessment of whether generation, editing, or neither is compelling.

### Gate H — broad pretraining question

Only if the OpenMoji pipeline is stable, add individually licensed corpora and compare
OpenMoji-only against pretrain-then-specialize with matched downstream training.

Exit evidence: quantified benefit or style dilution, plus complete provenance.

### Gate I — cached AR comparison

Train and tune the matched causal baseline. Produce quality/latency frontiers on the
same worker and test set.

Exit evidence: a defensible conclusion, including cases where AR wins.

### Gate J — deployment optimization

Apply the optimization sequence to frozen checkpoints. Package reproducible inference
and benchmark environments.

Exit evidence: verified numerical/visual parity and measured gains or a documented
reason the conversion is not worthwhile.

### Gate K — research interface and narrative

Build the paired trajectory viewer, inpainting controls, linter output, nearest-neighbor
panel, and benchmark explorer from frozen run artifacts.

Exit evidence: every displayed claim links to a reproducible run and environment.

## 12. Fallback interpretations

Fallbacks are scientific branches, not emergency deadline substitutions:

1. full structured categorical diffusion over path topology, style, and geometry;
2. semi-autoregressive path construction with within-path denoising;
3. fixed topology with geometry/style denoising for variation and editing;
4. cached autoregressive vector model using the same codec;
5. raster generative model plus clearly labelled vectorization, only as a separate
   comparison rather than a vector-native claim.

All branches retain the corpus manifest, codec study, renderer, linter, evaluation,
remote infrastructure, and provenance work.

## 13. Research record and outputs

Maintain:

- `reports/findings.md`: append hypotheses, observations, contradictions, and decisions;
- `reports/codec/`: distributions, round trips, worst cases, and representation choice;
- `reports/corruption/`: trajectory grids and controlled comparisons;
- `reports/evaluation/`: fixed sample sets, metrics, nearest neighbors, and uncertainty;
- `reports/benchmarks/`: hardware-specific quality/latency frontiers;
- `data/manifests/`: provenance, licenses, splits, and hashes;
- `runs/`: immutable lightweight run records;
- model cards and dataset/process cards for artifacts that may be shared.

A mature repository should expose reproducible commands for:

- data audit and preprocessing;
- codec report generation;
- linter and safe rendering;
- tiny smoke training;
- denoiser training and sampling;
- cached AR training and sampling;
- evaluation and nearest-neighbor analysis;
- paired trajectory rendering;
- remote probe/launch/status/sync;
- container and Kubernetes execution.

The project succeeds when it produces clear evidence about the representation,
corruption process, and vector-generation use case—even if the original diffusion
hypothesis is rejected.

## 14. Research weblog

The operator needs a readable running account of the research that is separate from the
machine-readable run registry: what was decided, why, what each experiment predicted,
what it actually showed, and the visual output that makes a claim checkable by eye.

Requirements:

- A static site generated from the committed evidence already in the repository —
  `state/CURRENT.md`, `state/runs.jsonl`, `runs/*/run.yaml`, `reports/**/summary.json`,
  and the derived renders. It is a view over the research record, never a second,
  hand-maintained source of truth that can silently disagree with it.
- It shows the decision trail (gates, hypotheses, predeclared criteria, and whether each
  was met or falsified), the experiment timeline, and the visual artifacts: rendered
  icons, paired clean/corrupted/predicted trajectories, and contact sheets.
- Falsified and superseded runs stay visible. A weblog that only shows successes would
  misrepresent the research.
- Served over the machine's Tailscale address, `100.69.189.78`, not published to any
  external host and not a Claude artifact.
- Regenerated and extended during training downtime, so it costs no GPU time and never
  delays an experiment.

## 15. Direction after Gate I

Written 2026-09-21, after Gates A–F closed, Gate G produced a denoiser that beats doing
nothing only at light corruption, and Gate I showed a matched autoregressive model over
the same codec generates scribbles at every setting this project can reach. The
evidence for each decision here is in `reports/findings.md` under the same date.

### The result to aim for

An **editor**, not a generator. The training bucket is 2,681 unique programs and most
of its variant families are singletons; unconditional generation of unseen concepts is
not achievable at that scale by any method, and Gate I measured it. Conditional
completion is: the corpus holds 39,535 contours and every one is a training example for
"given the rest of this icon, draw the missing path". Section 8 anticipated this — editing
may be the strongest product result even if unconditional generation is mixed — and
the fixed-topology fallback in section 12 is its special case.

### Gate L — structured editing with a masked any-order model

One bidirectional model over the flattened typed sequence, trained to predict masked
tokens from the rest. The mask families are the editing operations, and they replace
the corruption process rather than compete with it:

- a whole path, with its length kept so the packed layout does not move;
- a contiguous span of segments inside a path;
- the style fields of a set of paths;
- the geometry of a set of paths, styles and topology kept;
- a uniform random mask, so everything-masked generation is the same model.

Decoding commits tokens in grammatical dependency order — path lengths, path headers,
segment kinds, coordinates — with `legal_mask` recomputed at every commit, so a sample
is valid by construction. The model carries every verified correction from Gates G and
I from its first run: metric coordinate features, the distance-kernel target on
coordinates, the loss pooled over fields, structural padding excluded from attention,
dropout from the start, and a matched-size arm beside a larger regularised one.

Primary metric: paired per-icon render recovery of whole-path inpainting on held-out
icons against two zero-parameter policies, the icon with the path dropped and a
position-marginal sampler decoded through the same grammar. Predeclare: beats both with
the 95% interval excluding zero, a magnitude bar on the median recovery, every output
valid, and a render sheet beside the numbers.

Exit evidence: an editor that completes held-out icons visibly and measurably, or a
negative result that names which mask family fails and why.

### What follows and what is retired

- **Gate K** builds the editing viewer over Gate L's frozen artifacts: mask, fill, lock.
- **Gate H** is deferred behind Gate L, and is reopened only for generation, only if the
  editor works, and only with the licensing audit section 5 requires.
- Retired: further single-factor sweeps on the p = 0.35 denoiser, the corruption-process
  third arm, and further left-to-right arms. Their results stand as recorded.
- Gate G is complete: its assessment is that generation is not compelling at this
  scale, blind denoising helps only for light corruption, and editing is the candidate.

### Gate L's conclusion, 2026-09-21

Twenty-two arms. The mechanism a masked editor over this codec was missing is found
and measured: with each segment's start point in its own input and coordinate logits
projected onto the lattice's Fourier basis, the same 530k parameters learn continuity
- 17.6 bins against a zero-parameter copy policy's 25.6 - where five arms without it
sat at the marginal floor on their own training icons. Carried into editing, the
mechanism does what continuity can do and no more: whole-path completion stops damaging
and still loses to the hole; span completion ties a zero-parameter fill that joins the
visible ends across five readings, longer training and ten times the capacity, while
beating the corpus's statistics every time. What the corpus cannot teach is shape, for
parts as Gate I found for wholes. The gate is complete; the frozen specialist of arm 21
is the checkpoint Gate K's viewer is built over; Gate H is the only route to shape and
is the operator's call.

## 16. Direction after Gate L: a pretrained prior

Written 2026-09-21 with the operator's decision. Every model in this project so far was
trained from nothing on 2,681 icons, and twenty-two arms in Gate L plus Gate I before it
established what that teaches - colour, style, local continuity - and what it cannot:
what an unseen icon's parts look like. No further from-scratch arm on this corpus is
planned. The premise changes: start from a model that already knows what shapes look
like, and fine-tune it on OpenMoji under the same codec, splits, renderer, baselines and
run contract.

### Gate M - a pretrained SVG prior, fine-tuned on OpenMoji

The model is a permissively licensed pretrained code language model, pinned by hub
revision with its license recorded at that revision, fine-tuned with LoRA on the
training split. It reads the icon's caption - OpenMoji's annotation and tags, which
every icon carries - and writes the icon as SVG text in the project's canonical
serialization: fixed viewBox, only `<path d fill stroke ...>` elements on the
quarter-unit lattice. The text is what the pretrained prior knows; the typed codec
parses every output, so validity, safety and rendering are measured exactly as before,
and an output the codec rejects is a counted failure rather than a hidden one.

This is a change to two rules in section 3: the model is no longer compact, and it
does model an SVG text form - the project's own safe subset, not arbitrary XML. Both
are recorded here as deliberate.

Steps, each a registered run with predeclared criteria:

1. **Zero-shot control.** The pinned model, unfine-tuned, prompted with held-out
   captions. Validity through the codec, render success, and a sheet. This is the
   floor every fine-tune is read against, and it is not nothing: a code model may
   already draw.
2. **Overfit test.** Four icons memorised through the same prompt format and decoder,
   as every model change here has been.
3. **Fine-tune on the training split**, LoRA, evaluated on the family-disjoint
   validation split: validity and render rate, a perceptual similarity between the
   rendered output and the held-out icon's render and between the output and its
   caption, nearest-neighbour memorisation against the training set, and the sheet.
   Predeclared: validity above a stated bar, similarity to the reference above the
   zero-shot control with the interval excluding zero, and renders a person would
   recognise, judged on a fixed grid.
4. **Editing** through fill-in-the-middle prompts over the same canonical text, scored
   as Gate L scored it, only once generation has a result.

Exit evidence: recognisable held-out icons from captions, measured against the
zero-shot control and against memorisation, or a negative result that says what the
prior did not carry. Compute is the owned RTX 4080: a 3B model fits in bf16 with LoRA
and gradient checkpointing; 7B needs 4-bit weights and is the capacity step, not the
start.

### What carries over

The corpus manifest and family-disjoint splits, the typed codec and its serializer,
the safe renderer, the run registry, the weblog, and the discipline of a zero-parameter
or zero-training baseline beside every number. Gate L's frozen specialist and its
harness remain the record of what a from-scratch model does here.

### The model, chosen 2026-09-21

Checked at pinned hub revisions, license file read rather than the card's field:

- `Qwen/Qwen3-4B-Base` at `906bfd4b` - Apache 2.0; 4B parameters, 36 layers, hidden
  2560, grouped-query attention, 151,936-token vocabulary. **Primary.** Fits the RTX
  4080 in bf16 with LoRA and gradient checkpointing at batch 1.
- `Qwen/Qwen2.5-Coder-1.5B` at `df3ce67c` - Apache 2.0; the cheap comparison arm, a
  code-specialised prior at a third of the size.
- `Qwen/Qwen2.5-Coder-3B` at `09d9bc5d` - the card says "other"; the license file is the
  Qwen Research License, non-commercial. **Excluded.**
- `Qwen/Qwen2.5-Coder-7B` - Apache 2.0; needs 4-bit weights on 16 GB and is the
  capacity step once the 4B has a result, not the start.

The fine-tuning stack (`transformers`, `peft`, `accelerate`) is installed into `.venv`
without its own torch: the NGC torch the whole record was produced on stays the one in
use, and an install that drags a PyPI torch in must be undone before anything runs.
