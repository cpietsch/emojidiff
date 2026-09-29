"""IEEE float32 on demand.

This environment routes float32 matrix multiplies and cuDNN convolutions through TF32
by default (the NGC image sets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1`, and torch reports
`allow_tf32` True for both). TF32 keeps a 10-bit mantissa, so a float32 greedy decode
can depend on the batch it sits in near an argmax tie: on two vt-v1 smoke checkpoints,
1 and 3 of 4 N(0, I) latents decoded to different programs at batch 1 and batch 4.
With the in-process flags switched off - which wins over the override variable in this
build (a 512 x 1024 x 512 GPU matmul: max error 0.046 against float64 with TF32, 6e-5
without) - batch 1, batch 4 and the float32 CUDA-graph decoder agreed exactly there.
IEEE float32 is not guaranteed batch-invariant either, only far less tie-prone.

`strict_float32` switches TF32 off for a block and restores the previous settings after
it; `float32_flags` records what is in force. The flags are process-global: callers that
decode from several threads must serialise (the gallery's GPU lock does). A CUDA graph
bakes in the kernels it was captured with, so capture it inside the block too.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch

OVERRIDE_VARIABLE = "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"


def float32_flags() -> dict[str, Any]:
    """The float32 math settings in force, for a run or report record."""

    return {
        "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "tf32_cublas_override_variable": os.environ.get(OVERRIDE_VARIABLE),
    }


def tf32_enabled() -> bool:
    """Whether any float32 matmul or convolution may run in TF32 right now."""

    return bool(
        torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
        or torch.get_float32_matmul_precision() != "highest"
    )


def precision_tag() -> str:
    """A short tag for cache keys: "tf32" or "ieee"."""

    return "tf32" if tf32_enabled() else "ieee"


@contextmanager
def strict_float32() -> Iterator[dict[str, Any]]:
    """TF32 off for matmuls and cuDNN inside the block; the flags in force, yielded."""

    saved = (
        bool(torch.backends.cuda.matmul.allow_tf32),
        bool(torch.backends.cudnn.allow_tf32),
        torch.get_float32_matmul_precision(),
    )
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        yield float32_flags()
    finally:
        # The legacy matmul flag also sets the precision, so the precision goes last.
        torch.backends.cuda.matmul.allow_tf32 = saved[0]
        torch.backends.cudnn.allow_tf32 = saved[1]
        torch.set_float32_matmul_precision(saved[2])
