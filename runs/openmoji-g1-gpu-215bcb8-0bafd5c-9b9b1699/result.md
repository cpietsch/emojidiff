# Owned RTX 4080 Gate G pipeline smoke (deterministic cuBLAS workspace)

Status: planned; no remote stage or GPU step yet.

This run is bounded to the exact one-step local pilot and stages only its six selected
raw SVGs plus the pinned palette with the committed source snapshot.

It retries `openmoji-g1-gpu-e7dc920-0bafd5c-9b9b1699`, which loaded the staged config,
data, normalizers, model, and optimizer correctly but was rejected inside the first CUDA
step because `torch.use_deterministic_algorithms(True)` requires `CUBLAS_WORKSPACE_CONFIG`
on CUDA >= 10.2. The only changed factor is that launcher environment variable. The
config, fixture, seed, step budget, image, and cap are unchanged.
