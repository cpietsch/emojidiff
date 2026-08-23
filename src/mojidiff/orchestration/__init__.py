"""Authorization-aware worker orchestration."""

from mojidiff.orchestration.adapters import (
    SmokeRequest,
    SnapshotIdentity,
    StageRequest,
    WorkerAdapter,
    adapter_for,
    snapshot_identity,
)
from mojidiff.orchestration.config import HostInventory, WorkerSpec, load_inventory

__all__ = [
    "HostInventory",
    "SmokeRequest",
    "SnapshotIdentity",
    "StageRequest",
    "WorkerAdapter",
    "WorkerSpec",
    "adapter_for",
    "load_inventory",
    "snapshot_identity",
]
