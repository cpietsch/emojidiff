"""`strict_float32`: TF32 off inside the block, the previous settings restored after it,
even when the block raises; and what `float32_flags` records."""

from __future__ import annotations

import pytest
import torch

from mojidiff.learning.precision import (
    float32_flags,
    precision_tag,
    strict_float32,
    tf32_enabled,
)


@pytest.fixture
def tf32_on() -> object:
    saved = float32_flags()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")
    yield
    torch.backends.cuda.matmul.allow_tf32 = saved["matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = saved["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(saved["float32_matmul_precision"])


def test_strict_float32_switches_tf32_off_and_restores_it(tf32_on: object) -> None:
    before = float32_flags()
    assert tf32_enabled() and precision_tag() == "tf32"
    with strict_float32() as inside:
        assert inside["matmul_allow_tf32"] is False and inside["cudnn_allow_tf32"] is False
        assert inside["float32_matmul_precision"] == "highest"
        assert not tf32_enabled() and precision_tag() == "ieee"
        with strict_float32():  # nesting is harmless
            pass
        assert not tf32_enabled()
    assert float32_flags() == before
    with pytest.raises(RuntimeError, match="inside"), strict_float32():
        raise RuntimeError("inside")
    assert float32_flags() == before


def test_strict_float32_leaves_ieee_settings_as_they_were() -> None:
    saved = float32_flags()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        before = float32_flags()
        with strict_float32():
            pass
        assert float32_flags() == before and not tf32_enabled()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = saved["matmul_allow_tf32"]
        torch.backends.cudnn.allow_tf32 = saved["cudnn_allow_tf32"]
        torch.set_float32_matmul_precision(saved["float32_matmul_precision"])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="TF32 needs a GPU")
def test_strict_float32_gives_ieee_matmuls_on_the_gpu(tf32_on: object) -> None:
    generator = torch.Generator().manual_seed(0)
    a = torch.randn(64, 1024, generator=generator)
    b = torch.randn(1024, 64, generator=generator)
    exact = a.double() @ b.double()
    with strict_float32():
        strict = (a.cuda() @ b.cuda()).cpu().double()
    assert float((strict - exact).abs().max()) < 1e-3
