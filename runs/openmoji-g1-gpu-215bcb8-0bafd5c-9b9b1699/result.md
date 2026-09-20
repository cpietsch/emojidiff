# Owned RTX 4080 Gate G pipeline smoke (deterministic cuBLAS workspace)

Status: failed on the adapter result contract after the GPU step completed.

This run is bounded to the exact one-step local pilot and stages only its six selected
raw SVGs plus the pinned palette with the committed source snapshot.

It retries `openmoji-g1-gpu-e7dc920-0bafd5c-9b9b1699`, which loaded the staged config,
data, normalizers, model, and optimizer correctly but was rejected inside the first CUDA
step because `torch.use_deterministic_algorithms(True)` requires `CUBLAS_WORKSPACE_CONFIG`
on CUDA >= 10.2. The only changed factor is that launcher environment variable. The
config, fixture, seed, step budget, image, and cap are unchanged.

The deterministic-cuBLAS correction worked. The pipeline ran to completion on the RTX
4080: `device` is `cuda`, `deterministic_algorithms` is true, one optimizer step
produced train loss 15.216644 and token accuracy 0.005068, validation loss is 15.344566
with 0.009404 aggregate / 0.008850 changed / 0.009709 retained accuracy over 226 changed
and 412 retained fields, `checkpoint_round_trip` is true, and `locked_path_exact` is
true. Near-random first-step accuracy is expected and is not learning evidence.

The run is still recorded as failed because the adapter's result contract failed: the
staged wrapper computed its summary digest from the output root rather than the pilot's
report root, so it raised before printing the JSON result. The GPU work and its
artifacts are intact and durable on the worker and are listed in `artifacts.json`;
nothing was deleted. The correction is confined to the wrapper's summary path and
requires a new code and run identity.

Recovered evidence on the worker:

- checkpoint.zip, 7,075,309 bytes, sha256
  `4b265e5575e3aa455a0d427e340ec407eaaaf39222709305d060281ae7e453f9`;
- summary.json sha256
  `15ede088423678eb308548481e1c348850c04bf588101d4f74fecaaa224d07b7`;
- 577,552 model parameters over the 3,359-icon `bucket-p32-t128`;
- torch 2.8.0a0+5228986c39.nv25.06, CUDA 12.9.

The local CPU pilot checkpoint is the same 7,075,309 bytes but hashes differently
(`d11efa00...`), which is the expected CPU/GPU floating-point difference. This run was
never intended to establish cross-device artifact identity.
