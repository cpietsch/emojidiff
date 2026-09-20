# Owned GPU tiny smoke with derived runtime

Status: failed before the CUDA step.

The local authorization record accidentally assigned the derived image to the disabled
Vast worker while `owned_gpu` still selected the upstream image. The adapter correctly
obeyed that record, so CairoSVG was again unavailable. The durable run directory is
empty. This is preserved as a configuration-targeting failure.
