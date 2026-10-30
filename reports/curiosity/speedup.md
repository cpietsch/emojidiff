# Research note: training throughput on the Curiosity cluster

Started 2026-10-30. Living note; the dated narrative stays in `reports/findings.md`.

## Question

The Curiosity allocation gives the team one node: 8 NVIDIA B300 (275 GB each), 128 cores /
256 threads (Xeon 6776P), 1.8 TB RAM, for about 20 hours. The render-to-SVG transcriber is a
9.0M-parameter model trained at batch 32. How do we turn that node into the most useful
training per hour: faster single runs, more runs per node, and fewer wasted run-hours per
decision?

## Bottom line

The node is limited by **CPU work per training sample**, not by the GPUs and not by
kernel launches. Each sample costs a loader worker about 15 ms of CPU, and almost none of
that is drawing: the codec validates every program three times and indexes tokens one
element at a time in Python. With 8 runs on 256 threads the loaders, the 8 main processes and
the evaluation decodes all compete for cores.

In order of payoff per hour of work:

1. **Cap the main processes' CPU threads.** Done in wave 2: about 25% more steps per hour,
   measured.
2. **A faster, byte-identical loader.** A prototype reproduces the real stream exactly on
   12,000 items and runs 2.5 times faster. Expected: runs reach the single-run step rate of
   about 18 steps/s, up from 12-13.
3. **Render once, train many.** Arms that differ only in model or optimizer settings consume
   the identical data stream. Wave 2's 8 runs need only 3 streams.
4. **Only after that, the GPU side:** `torch.compile`, CUDA graphs, several runs per GPU. None
   of it pays while the CPU is the limit.

GPU rasterisation, larger batches, MIG and more loader workers do not help (see below).

## Measured throughput

Steps per second include the in-loop evaluation every 2,000 steps (the full protocol),
unless marked otherwise.

| setting | steps/s per run | source |
| --- | --- | --- |
| RTX 4080, one run, full protocol | 11.56 | `r2s-full-v9-colour-621bc8e` |
| H100, full protocol, 4 runs on the node | 17.96 | `r2s-full-v9-colour-989aea8` |
| H100, full protocol, s120k / s240k | 18.87 / 19.02 | `r2s-v10-s120k`, `r2s-v10-s240k` |
| B300, 2 smokes at once on 64 CPUs (eval every 400 steps) | 11.26 / 11.47 | `r2s-smoke-*-67bab78` |
| B300 node, 8 runs, no thread cap (wave 1) | about 10.2 | job 13939 |
| B300 node, 8 runs, `OMP_NUM_THREADS=8` (wave 2, first 12 min) | 11.8-13.4 | job 13958 |

The 14.5 steps/s quoted earlier for one H100 run was the mean of three concurrent 1,200-step
smokes, whose evaluation every 400 steps takes 11-28% of their time. The right single-run
reference is about 18-19 steps/s. **No single B300 run with ample CPU has been measured yet**;
that is the first missing number (see the end).

Wave 1 itself is a natural experiment: per-run rates rose as arms finished, from about 10.2
with 8 runs training to 11.7-11.9 with 6, 14.25 with 2 (dec8) and 12.2-12.3 for compose08
alone. On MUSICA, 4 runs per H100 node cost about 10% per run against one run alone.

## Where the time goes

### The loader (per sample, CPU)

Measured locally (i7-13700K P-core at 5.3 GHz, one thread, as in a DataLoader worker), on the
exact v9 stream, and confirmed by an independent re-run:

| item | ms per item | items/s per thread |
| --- | --- | --- |
| original icon (pre-rendered) | 0.14 | 7,195 |
| exact variant | 13.7 | 73.5 |
| composition | 20.1 | 50.5 |
| v9 mix (compose 0.5, original 0.1) | 15.6 | 64.3 |
| compose 0.8 mix | 18.5 | 54.1 |
| labelled Twemoji mix | 15.4 | 64.9 |

Inside one exact variant (13.7 ms): `serialize_packed_svg` 5.6 ms (mostly two more passes of
`validate_tensor_program`), `validate_packed_tensor_program` 2.6, `augment_program_tokens` 1.6,
`unflatten_program` 1.3, and **all of Cairo 2.4 ms** (parse 0.13, draw 1.12, PNG encode 0.66, PIL
decode 0.21, compositing 0.23). Each program is validated three times (four for
compositions), about 6,000 tiny `np.any` calls per item. A composition adds unflattening of
2-4 source programs, its own validation and `program_legal_masks` (1.1 ms).

On the cluster's CPUs a worker makes about 30 items/s when uncontended and about 20 when the
node is packed (inferred from the wave-1 records), against 60-63 locally. At 20 items/s, one
run at 18 steps/s needs 29 workers, so eight runs would need 232 threads for workers alone.

### The main process (per step, CPU)

About 3.4 ms of CPU per batch: the queue receive, `np.unpackbits` of the masks and `.bool()`.
`.bool()` allocates a fresh 18.4 MB tensor every step. `torch.from_numpy(bits).view(torch.bool)`
gives the identical tensor in 1.3 ms instead of 4.0 at one thread, and removes the only
OpenMP parallel region in the step. Without a cap, each of the 8 main processes sizes its
thread pool to the machine, and libgomp threads spin: 16 threads burn 5.9-13 ms of CPU per
step for 1.4-1.9 ms of useful work. That, multiplied by 8, is the load wave 1 showed.

Every step copies 21.1 MB to the GPU, 18.4 MB of it the boolean legal-token mask, through
synchronous pageable copies. Sending the packed bits and unpacking on the GPU
(shift-and-`&1`) is bit-exact and cuts the copy to 5.0 MB.

### The training step (GPU)

On the RTX 4080 the step is GPU-bound: 83.6 ms per step, 79 ms of it kernel time (about 95%
busy), 1,210 kernels, 9 stream synchronisations, about 2.8-2.9 TFLOP of executed work (34
TFLOPS against 103 measured peak bf16 matmul). There, only `torch.compile` helps (61.2 ms with
model plus loss compiled, 1.37x); fused AdamW, pinned memory, GPU unpacking and a whole-step
CUDA graph all land at 81-84 ms. Two or three processes on the 4080 give 4% less in total.

On the H100 and B300 the step is not GPU-bound, but the evidence points to the per-sample CPU
bound above rather than launch overhead. The width and depth slowdowns on MUSICA fit
"max(loader, compute)" as well as a fixed launch cost, and batch 128 gave only about 10% more
samples per second, which a fixed per-step overhead would not allow. Host enqueue is about
13-14 ms per step on a fast desktop core. So CUDA graphs and compilation become levers only
once the loader supplies well over about 1,300 samples/s per run, 2-3 times what it does now.

Two code facts matter for that later stage:

- The loss indexes logits with a boolean mask (`[free]`), which forces device-to-host syncs
  every step. A whole-step CUDA graph needs a masked mean instead (`reduction='none'`, times
  the mask, over its sum). That is mathematically equal but not bit-identical (summation
  order).
- A whole-step graph keeps the computation: the step-1 loss was bit-identical and 10-step
  losses stayed within the 4e-5 run-to-run spread.

### Evaluation and fixed costs

- Greedy decoding is host-bound Python (a `legal_mask` per position): 2.0 ms per model call on
  the 4080, 3.8-4.3 on an H100, 4.9-5.1 on a B300.
- Every evaluation render starts a `python -I` subprocess (the isolated renderer): 118-149 ms
  of CPU per icon against about 5 ms in process.
- Time outside training steps: 4.7-5.0% of a 60k run on the 4080, 10.6-16% on an H100. The
  final phase took 212-305 s on an H100 (74 s on the 4080), of which the latency measurement
  is 48-111 s. On a busy cluster node that latency is not the deliverable's number, which is
  measured on the RTX 4080.
- `teacher_forced_loss` runs with dropout active and draws from the CUDA RNG. Changing how
  often or where evaluation runs therefore changes the training trajectory, which is not just
  a measurement change.
- `fast_decode.GraphDecoder` (CUDA-graph decoding, 381 ms per icon on an idle 4080 against
  about 1.5 s for the reference decoder) exists and is used by the demo, but not by training.

## Levers

| # | lever | gain | effort | changes results? | status |
| --- | --- | --- | --- | --- | --- |
| 1 | `OMP_NUM_THREADS=8` per run | +25% steps/s (measured) | minutes | no | in wave 2 |
| 2 | `.view(torch.bool)` instead of `.bool()` | about 2.7 ms less main CPU per step | minutes | no (identical tensor) | done, d54e21d |
| 3 | packed masks unpacked on the GPU | 16 MB less copied per step | hours | no (bit-exact) | done, d54e21d |
| 4 | fast loader: one validation, numpy index tables, no PNG round trip | 2.3x per worker (measured); runs up to about 18 steps/s | about a day with tests | no (byte-identical, below) | done, d54e21d (token augmentation still in torch) |
| 5 | one loader feeding several models with the same stream | CPU cut by streams/runs (wave 2: 3/8) | about a day | no per model | proposed |
| 6 | evaluation in a separate process; GraphDecoder; one persistent render worker instead of a subprocess per icon | 10-16% of wall on H100-class hosts | one day | yes: new baseline (RNG, selection) | proposed |
| 7 | skip the latency measurement on the cluster (null with reason; measure on the 4080) | 1-2% of wall | minutes | no | proposed |
| 8 | `torch.compile` of model and loss | 1.37x on a GPU-bound 4080; on B300 only after 4-5 | hours | numerics within run-to-run spread | after 4-5 |
| 9 | whole-step CUDA graph (needs a masked-mean loss) | up to 1.3x per run once not CPU-bound | one day | not bit-identical (sum order) | after 4-5 |
| 10 | several runs per GPU (time slicing or user-started CUDA MPS) | only with spare CPU; literature: MPS up to 1.8x over naive sharing on A100 | hours | no | after 4-5 |

### Implemented: levers 2-4 (commit d54e21d)

- `serialize_packed_svg` validates once: `unpack_tensor_program` already validates the dense
  program it serializes, so `decode_program`'s second pass is skipped, and the stream no
  longer validates separately before serializing.
- `flatten_program` and `unflatten_program` use index arrays instead of element loops.
- `render_trusted_rgb` reads cairo's ARGB32 surface and un-premultiplies it exactly as cairo's
  PNG writer does, instead of encoding and decoding a PNG.
- Training batches carry packed masks, unpacked on the GPU; host expansion uses
  `.view(torch.bool)`.

Verification. The streams of d54e21d and of 508979a (the code waves 1 and 2 ran), side by
side on the real corpus, agree on every item: render, tokens, packed masks, path order and
set label.

| stream | items | identical | old items/s | new items/s | speed-up |
| --- | --- | --- | --- | --- | --- |
| v9 mix, worker 0 | 3,000 | 3,000 | 62.7 | 142.1 | 2.27x |
| v9 mix, worker 7 | 1,500 | 1,500 | 61.7 | 141.0 | 2.29x |
| compositions only | 1,500 | 1,500 | 50.5 | 122.2 | 2.42x |
| labelled Twemoji (shared hyperthread) | 1,500 | 1,500 | 45.6 | 79.7 | 1.75x |

The corpus cache rebuilt with the new code (its key hashes `autoregressive.py`, so it moved to
35a6ce034e7e) has the same sha256 as before (dataset 47646604): every icon of every split,
plain and mirrored, renders, flattens and masks to the same bytes. Unit tests pin
flatten/unflatten against the element loops, the surface render against the PNG path,
`unpack_masks` against `np.unpackbits`, and one validation per serialization.

How 4 stays exact: the stream depends only on the augmentation seed, the worker count (one
generator per worker, batches dealt round-robin), batch size, compose settings and training
data. A faster producer that keeps the same 16 logical streams and the same random draws gives
the same batches. Changing `loader_workers`, by contrast, changes the data: the w16/w32/w40
smokes differ at step 400 (loss 3.127, 3.116, 3.044).

Further down the same road: a counter-based generator seeded by (augment_seed, item index)
would make the stream independent of the worker count, so any number of producers gives the
same batches. That is a one-time change of data that needs a new v9 baseline. Precomputed
streams also fit in memory: a 60k-step stream is about 19 GB compressed (0.2 ms decode per
item) and would fit in a B300's free HBM or the node's 1.8 TB of RAM.

## More decisions per GPU-hour

From the 12 comparable full runs and their 64-icon evaluations:

- The 64-icon evaluation at 10k steps does not predict the final 339-icon ranking (Spearman
  about 0.08). At 20k it reaches 0.56 (p = 0.059); the best-so-far value at 30k reaches 0.67
  (p = 0.017).
- Successive halving that keeps the top third at 20k would have kept the eventual best arm
  65-96% of the time (33-38% by chance), with mean regret at most 0.0006. But it is biased
  against slow starters: batch 128 ranked first at 6k, 10k and 30k and finished 7th; v6's
  compositions trailed v4 early and ended 0.010 better. Waves 1-2 are mostly regularisers,
  which start slow. Use a score relative to a control run, not to the rung leader.
- Halving only saves GPU-hours with a launcher that starts a new arm when one is stopped.
  `arms.slurm` runs a fixed list and waits for all of it.
- Best checkpoints fall at 0.77-0.97 of the 60k schedule in 9 of 10 runs; 120k and 240k gave no
  gain. Evaluating every 10k steps and every 2k in the last third uses 14 of 30 evaluations and
  costs at most 0.0013 in the selected 64-icon error. Patience-based early stopping loses
  quality (patience 3: +0.011 median).
- Noise sets the design: one 64-icon point jitters with SD about 0.003; identical configs
  differ at matched steps with SD about 0.005; the paired standard error between two arms is
  0.0042 on 64 icons and 0.0018 on 339. The B300 v9 rerun (0.0790 against 0.0724 for the same
  seed on an H100) shows single runs mislead at the 0.003 effects this project is chasing.
  Seeds are worth more than extra arms.
- Teacher-forced NLL as an early signal is unresolved: one analysis found Spearman 0.73-0.78
  against the final error from 16k to 40k (9 MUSICA arms), another 0.36 at 20-30k (12 arms).

## What does not help

- **GPU rasterisation.** Cairo's own drawing is 0.19-1.1 ms of a 13.7 ms item.
- **Larger batches.** About 10% more samples per second on the recorded hardware, and batch
  128 was worse on MUSICA (0.0751 against 0.0724).
- **More loader workers.** This changes the data stream, and at 8 runs the node has no spare
  cores anyway.
- **MIG.** It needs root and splits a GPU into fixed slices.
- **Packing more runs per GPU now.** That adds loaders to a CPU that is already the limit.

## Next measurements

1. **One B300 run with ample CPU** (1 GPU, 32 or more CPUs, full protocol, with a per-step
   timer for the wait in `next(batches)`): the true single-run ceiling on this node, and
   whether a run is loader-bound or main-process-bound. `py-spy dump` through
   `srun --jobid=<id> --overlap` answers the same question for a running job.
2. **Wave 2's full-run rates** against wave 1's 10.2 steps/s: the thread cap over a whole run,
   not only its first 12 minutes.
3. **The fast loader on a cluster CPU:** per-worker items/s on the Xeon, which is what sets
   workers per run.

## Sources and artifacts

- Measurement scripts, the fast-loader prototype (`loader/fast_stream.py`), raw results and GPU
  traces: `/home/dev/.cache/mojidiff/speedup-2026-10-30/`. Four investigations (loader, GPU
  step, run records, outside sources), each checked by an independent verifier who re-ran or
  re-read the main numbers; corrected values are used here.
- Outside sources: NVIDIA MPS documentation (single-user MPS without root); arXiv 2209.06018
  (MPS against MIG and naive sharing, A100); HFTA (horizontally fused training arrays, MLSys
  2021: 2.4-11.5x for fused same-architecture jobs on V100/A100); PyTorch CUDA-graphs blog (1.12-1.7x
  on MLPerf models); Triton sm_103 issue #8473 and fix #8299. On Curiosity, `torch.compile`
  already ran on these B300s with torch 2.14.1+cu130 (`/home/dev/curiosity-cluster.md`).
