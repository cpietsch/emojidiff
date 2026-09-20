# Owned GPU adapter-contract smoke

Status: completed.

This retry changed only entrypoint handling after the parent run completed GPU and
artifact work but failed clean-JSON result parsing. The adapter staged the immutable
Git snapshot, ran the typed-codec/render and one-step GPU forward/backward smoke, wrote
and reloaded the checkpoint, and copied selected artifacts into the persistent artifact
volume.

The RTX 4080 step used PyTorch `2.8.0a0+5228986c39.nv25.06` with CUDA 12.9. Loss was
`0.20076081156730652` and gradient norm was `0.3963698446750641`. Tensor fields were
stable through the codec round trip. The 16,179-byte checkpoint has SHA-256
`77e93414e04707efb0469718426bb2364d5a7b22e575e519124d5c628442348f`.

A second complete invocation used create-or-identical semantics and reproduced both
the workspace and durable artifacts exactly. No detached job remains active.
