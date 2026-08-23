"""Authorization-aware worker orchestration."""

from mojidiff.orchestration.adapters import WorkerAdapter, adapter_for
from mojidiff.orchestration.config import HostInventory, WorkerSpec, load_inventory

__all__ = ["HostInventory", "WorkerAdapter", "WorkerSpec", "adapter_for", "load_inventory"]
