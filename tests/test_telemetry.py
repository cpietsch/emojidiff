from __future__ import annotations

import time

import torch

from mojidiff.learning.telemetry import (
    RESOURCE_REQUIRED_KEYS,
    measure_latency,
    resource_summary,
)


def test_measure_latency_reports_per_icon_milliseconds() -> None:
    device = torch.device("cpu")
    calls = 0

    def step() -> None:
        nonlocal calls
        calls += 1
        time.sleep(0.002)

    report = measure_latency(step, device=device, icons_per_call=4, warmup=2, repeats=5)
    assert calls == 7
    assert report.repeats == 5 and report.icons_per_call == 4
    # 2 ms for four icons is at least 0.5 ms per icon; sleep never undershoots.
    assert report.min_ms_per_icon >= 0.5
    assert report.median_ms_per_icon >= report.min_ms_per_icon
    assert report.p95_ms_per_icon >= report.median_ms_per_icon
    assert report.icons_per_second > 0


def test_resource_summary_has_every_required_key() -> None:
    device = torch.device("cpu")
    report = measure_latency(lambda: None, device=device, warmup=0, repeats=1)
    record = resource_summary(device, train_seconds=1.5, latency=report).as_record()
    for key in RESOURCE_REQUIRED_KEYS:
        assert key in record
    assert record["peak_vram_gib"] is None
    assert record["cuda_version"] is None
    assert record["train_seconds"] == 1.5
    assert record["inference_ms_per_icon"] == report.median_ms_per_icon


def test_resource_summary_without_latency_leaves_inference_fields_null() -> None:
    record = resource_summary(torch.device("cpu"), train_seconds=None, latency=None).as_record()
    assert record["inference_ms_per_icon"] is None
    assert record["icons_per_second"] is None
