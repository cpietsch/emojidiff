# Owned RTX 4080 Gate G pipeline smoke (corrected working-directory contract)

Status: failed during the first CUDA step; no checkpoint written.

This run is bounded to the exact one-step local pilot and stages only its six selected
raw SVGs plus the pinned palette with the committed source snapshot.

It retries `openmoji-g1-gpu-cd3250e-0bafd5c-9b9b1699`, which verified its stage but
failed before model construction because the staged pilot resolved relative input paths
against the image working directory. The only changed factor is the wrapper's working
directory; the config, fixture, seed, step budget, image, and cap are unchanged.

The working-directory correction worked: the staged config, pinned palette, six raw
SVGs, selected normalizers, conditioned model, and optimizer all loaded from the
immutable source root. The run then failed inside the first CUDA step because the
pipeline declares `torch.use_deterministic_algorithms(True)` and CUDA >= 10.2 cuBLAS
needs `CUBLAS_WORKSPACE_CONFIG` to honor that declaration. The container left an empty
artifact directory, no checkpoint, and no running container.

This is a launcher environment gap, not a weakening of the determinism contract. The
correction sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` explicitly in the owned Docker smoke
launcher, which makes the declared determinism achievable rather than relaxing it.
