# Research note: training throughput on the Curiosity cluster

Started 2026-10-30. Living note; the dated narrative stays in `reports/findings.md`.

## Question

The Curiosity allocation gives the team one node: 8 NVIDIA B300 (275 GB each), 128 cores /
256 threads, 1.8 TB RAM, for about 20 hours. The render-to-SVG transcriber is a 9.0M-parameter
model trained at batch 32. How do we turn that node into the most useful training per hour:
faster single runs, more runs per node, and fewer wasted run-hours per decision?

## What is measured so far

| setting | steps/s per run | GPU utilization | source |
| --- | --- | --- | --- |
| RTX 4080, one run | 10.7 | | MUSICA findings, 2026-10-14 |
| H100, one run (16, 32 or 40 loader workers) | ~14.5 | | `r2s-smoke-v9-w16/w32/w40-77dbfc7` |
| B300, one run, 1,200-step smoke | | | `r2s-smoke-v9-w16-67bab78` |
| B300 node, 8 runs at once (wave 1, job 13939) | ~10.1 | 27-51% (samples 0-50%) | `nvidia-smi` at 48 min |

Node at 48 minutes into wave 1: load average 397 on 256 hardware threads; 5.7 GB of 275 GB
used per GPU; 250-310 W per GPU. Eight runs of 16 loader workers each are 128 rendering
processes, plus each main process's PyTorch CPU thread pool, which defaults to every core.

First reading: the node is CPU-bound, not GPU-bound. A B300 is far larger than this model
needs, so the GPUs wait for rendered batches, and the main processes compete with the
loader workers for cores.

## Under investigation

1. Loader: items per second per worker for originals, exact variants and compositions;
   where the time goes (Cairo, masks, Python); cores needed per run.
2. Training step: GPU kernel time against wall time (launch-bound?), and the effect of
   fused AdamW, `torch.compile`, CUDA graphs, packed masks unpacked on the GPU, pinned
   memory, larger batches, and several processes per GPU.
3. Run records: evaluation overhead per run, and whether early evaluations predict the final
   ranking well enough for successive halving.
4. Outside practice: CUDA MPS and MIG on B300 under Slurm, batched small-model training,
   GPU rasterisation of SVG paths.

Results, levers and a recommended setup for wave 2 follow below once measured.
