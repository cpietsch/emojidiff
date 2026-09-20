# Owned RTX 4080 Gate G pipeline smoke (corrected working-directory contract)

Status: planned; no remote stage or GPU step yet.

This run is bounded to the exact one-step local pilot and stages only its six selected
raw SVGs plus the pinned palette with the committed source snapshot.

It retries `openmoji-g1-gpu-cd3250e-0bafd5c-9b9b1699`, which verified its stage but
failed before model construction because the staged pilot resolved relative input paths
against the image working directory. The only changed factor is the wrapper's working
directory; the config, fixture, seed, step budget, image, and cap are unchanged.
