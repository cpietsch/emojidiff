# Owned RTX 4080 Gate G pipeline smoke

Status: failed before model construction or a GPU step.

This run is bounded to the exact one-step local pilot and stages only its six selected
raw SVGs plus the pinned palette with the committed source snapshot.

The immutable stage verified, but the pilot resolved its relative input paths against
the image working directory rather than the staged source root. It failed on the pinned
palette path without writing a checkpoint or result. This run remains preserved; the
wrapper working-directory fix requires a new code and run identity.
