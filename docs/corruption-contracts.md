# Corruption contracts

Gate F separates corruption correlation from grammar validity. All contracts operate on
the typed packed program and validate against the same codec. None is described as an
exact D3PM: these are custom structured iterative-denoiser curricula with no claimed
closed-form posterior.

## Factorized uniform control

With fixed topology, each legal geometry token independently receives a Bernoulli gate
with probability `p`. Open gates replace the token uniformly from its role-specific
vocabulary excluding the original token. Endpoints use q289; quadratic and cubic control
coordinates use q417. Path lengths, segment kinds, styles, and padding remain unchanged.

## Path-correlated treatment

Each active path receives one Bernoulli gate with probability `p`. An open gate replaces
every legal geometry token in that path, again uniformly excluding the original value.
The marginal expected fraction of changed legal geometry fields is `p`, matching the
factorized control, while the within-path correlation differs. Topology, styles, and
padding remain locked.

## Fixed-topology diagnostic

Both initial Gate F arms are fixed-topology geometry-only experiments. They use the Gate
E model and the same data, seeds, model, optimizer, per-step resampling, held-out draws,
checkpoint boundary, and render protocol. This is deliberately narrower than a joint
topology/style model and must be reported that way.

## Whole-path replacement ablation

The safe fixed-topology implementation accepts a donor packed program and copies a whole
path's start and legal coordinate fields only when length and segment-kind sequence are
identical. Destination styles, topology, and padding are retained. Incompatible paths
are unchanged rather than projected. A later Gate F arm must pin an empirical donor
selection rule and report the compatible-path rate before this ablation is interpreted.
