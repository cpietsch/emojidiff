# Owned GPU tiny smoke retry

Status: failed before the CUDA step.

The sibling container and both scoped named-volume mounts started successfully,
resolving the parent failure. The pinned upstream NGC image lacks CairoSVG 2.9.0, so
the tokenizer/renderer import failed before forward/backward execution. The durable
run directory is empty. Preserve this as evidence that the upstream image alone is not
the complete project runtime.
