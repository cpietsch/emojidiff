"""Latency and memory telemetry that every training or sampling run must report.

The project's goal is a small, fast model, so speed is a first-class result, not a
follow-up. This module gives every study one way to measure it, so the numbers in
`run.yaml` are comparable across runs on the same named GPU.

`measure_latency` times a callable that processes one icon (or one batch of icons)
end to end, after warmup, with CUDA synchronised, and reports per-icon milliseconds.
`resource_summary` collects the device name, peak VRAM, and framework versions.
`ResourceReport.as_record` is the block a run writes under `resource` in `run.yaml`;
`scripts/audit_run_records.py` requires it for every completed run registered after
`RESOURCE_REQUIRED_FROM`.
"""

from __future__ import annotations

import platform
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import torch

from mojidiff.orchestration.contract import RESOURCE_REQUIRED_FROM, RESOURCE_REQUIRED_KEYS

__all__ = [
    "RESOURCE_REQUIRED_FROM",
    "RESOURCE_REQUIRED_KEYS",
    "LatencyReport",
    "ResourceReport",
    "measure_latency",
    "resource_summary",
    "reset_peak_memory",
    "peak_vram_gib",
    "device_name",
]


@dataclass(frozen=True)
class LatencyReport:
    """Per-icon wall-clock latency of one callable, measured after warmup."""

    repeats: int
    icons_per_call: int
    median_ms_per_icon: float
    p95_ms_per_icon: float
    min_ms_per_icon: float
    icons_per_second: float


@dataclass(frozen=True)
class ResourceReport:
    """Everything a run must record about the hardware it used."""

    device: str
    torch_version: str
    cuda_version: str | None
    python_version: str
    peak_vram_gib: float | None
    train_seconds: float | None
    inference_ms_per_icon: float | None
    inference_p95_ms_per_icon: float | None
    icons_per_second: float | None

    def as_record(self) -> dict[str, Any]:
        return asdict(self)


def _synchronise(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure_latency(
    step: Callable[[], Any],
    *,
    device: torch.device,
    icons_per_call: int = 1,
    warmup: int = 3,
    repeats: int = 20,
) -> LatencyReport:
    """Time `step` end to end and report milliseconds per icon.

    `step` runs one complete inference for `icons_per_call` icons: all denoising
    steps, all decoding positions, any projection or constrained decoding. Only the
    whole thing is a latency; a single forward pass is not.
    """

    if icons_per_call < 1:
        raise ValueError("icons_per_call must be at least 1")
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    for _ in range(warmup):
        step()
    _synchronise(device)
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        step()
        _synchronise(device)
        samples.append((time.perf_counter() - started) * 1000.0 / icons_per_call)
    samples.sort()
    median = statistics.median(samples)
    p95 = samples[min(len(samples) - 1, int(round(0.95 * (len(samples) - 1))))]
    return LatencyReport(
        repeats=repeats,
        icons_per_call=icons_per_call,
        median_ms_per_icon=median,
        p95_ms_per_icon=p95,
        min_ms_per_icon=samples[0],
        icons_per_second=1000.0 / median if median > 0 else float("inf"),
    )


def reset_peak_memory(device: torch.device) -> None:
    """Call before training so `peak_vram_gib` measures this run alone."""

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_vram_gib(device: torch.device) -> float | None:
    if device.type != "cuda":
        return None
    return float(torch.cuda.max_memory_allocated(device)) / 2**30


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"{device.type}:{platform.processor() or platform.machine()}"


def resource_summary(
    device: torch.device,
    *,
    train_seconds: float | None,
    latency: LatencyReport | None,
) -> ResourceReport:
    """The `resource` block for `run.yaml`; pass `latency=None` only for non-model runs."""

    return ResourceReport(
        device=device_name(device),
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda) if device.type == "cuda" else None,
        python_version=platform.python_version(),
        peak_vram_gib=peak_vram_gib(device),
        train_seconds=train_seconds,
        inference_ms_per_icon=latency.median_ms_per_icon if latency else None,
        inference_p95_ms_per_icon=latency.p95_ms_per_icon if latency else None,
        icons_per_second=latency.icons_per_second if latency else None,
    )
