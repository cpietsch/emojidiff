# Owned GPU tiny smoke

Status: failed before container creation.

Docker 29.7.2 rejected the generated volume mount because bare `rw` is not a
key/value `--mount` field. The durable run directory exists and is empty; no
container or GPU step was created. Preserve this as an adapter compatibility failure.
