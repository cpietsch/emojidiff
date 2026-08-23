"""Load the local machine-readable authorization record without exposing secrets."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class ConfigurationError(ValueError):
    """Raised when the authorization record is malformed."""


CAP_FIELDS: dict[str, tuple[str, ...]] = {
    "vast-ephemeral": ("max_steps", "max_spend_usd", "max_storage_gb"),
    "owned-persistent": ("max_steps", "max_storage_gb"),
    "cluster": ("max_jobs", "max_steps_per_job", "max_storage_gb"),
}


@dataclass(frozen=True)
class ArtifactStore:
    type: str
    uri: str | None
    credentials_source: str | None

    def missing_fields(self) -> tuple[str, ...]:
        missing: list[str] = []
        if not self.type or self.type == "unset":
            missing.append("artifact_store.type")
        if not self.uri:
            missing.append("artifact_store.uri")
        if not self.credentials_source:
            missing.append("artifact_store.credentials_source")
        return tuple(missing)


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    enabled: bool
    ssh_alias: str | None
    kind: str
    execution: str
    workspace_root: str | None
    image: str | None
    resource_cap: Mapping[str, Any]
    raw: Mapping[str, Any]

    def missing_cap_fields(self) -> tuple[str, ...]:
        required = CAP_FIELDS.get(self.kind)
        if required is None:
            return (f"workers.{self.name}.kind(unsupported:{self.kind})",)
        missing: list[str] = []
        for field in required:
            value = self.resource_cap.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                missing.append(f"workers.{self.name}.resource_cap.{field}")
        return tuple(missing)

    def missing_launch_fields(self) -> tuple[str, ...]:
        missing = list(self.missing_cap_fields())
        if not self.ssh_alias:
            missing.append(f"workers.{self.name}.ssh_alias")
        if not self.workspace_root:
            missing.append(f"workers.{self.name}.workspace_root")
        if not self.image:
            missing.append(f"workers.{self.name}.image")
        if self.kind == "cluster":
            for field in ("namespace", "service_account", "pvc"):
                if not self.raw.get(field):
                    missing.append(f"workers.{self.name}.{field}")
        return tuple(missing)

    @property
    def compute_authorized(self) -> bool:
        return self.enabled and not self.missing_launch_fields()


@dataclass(frozen=True)
class HostInventory:
    path: Path
    orchestrator_name: str
    artifact_store: ArtifactStore
    workers: Mapping[str, WorkerSpec]

    def enabled_workers(self) -> tuple[WorkerSpec, ...]:
        return tuple(worker for worker in self.workers.values() if worker.enabled)


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigurationError(f"{field} must be a mapping")
    return value


def load_inventory(path: Path) -> HostInventory:
    """Parse host authorization configuration and retain no implicit defaults."""

    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigurationError(f"cannot read {path}: {exc}") from exc
    root = _mapping(loaded, "root")
    if root.get("version") != 1:
        raise ConfigurationError("version must be 1")

    orchestrator = _mapping(root.get("orchestrator"), "orchestrator")
    artifact = _mapping(root.get("artifact_store"), "artifact_store")
    workers_raw = _mapping(root.get("workers"), "workers")

    workers: dict[str, WorkerSpec] = {}
    for name, item in workers_raw.items():
        if not isinstance(name, str):
            raise ConfigurationError("worker names must be strings")
        worker = _mapping(item, f"workers.{name}")
        cap = _mapping(worker.get("resource_cap"), f"workers.{name}.resource_cap")
        enabled = worker.get("enabled")
        if not isinstance(enabled, bool):
            raise ConfigurationError(f"workers.{name}.enabled must be boolean")
        workers[name] = WorkerSpec(
            name=name,
            enabled=enabled,
            ssh_alias=_optional_str(worker.get("ssh_alias"), f"workers.{name}.ssh_alias"),
            kind=_required_str(worker.get("kind"), f"workers.{name}.kind"),
            execution=_required_str(worker.get("execution"), f"workers.{name}.execution"),
            workspace_root=_optional_str(
                worker.get("workspace_root"), f"workers.{name}.workspace_root"
            ),
            image=_optional_str(worker.get("image"), f"workers.{name}.image"),
            resource_cap=cap,
            raw=worker,
        )

    return HostInventory(
        path=path,
        orchestrator_name=_required_str(orchestrator.get("name"), "orchestrator.name"),
        artifact_store=ArtifactStore(
            type=_required_str(artifact.get("type"), "artifact_store.type"),
            uri=_optional_str(artifact.get("uri"), "artifact_store.uri"),
            credentials_source=_optional_str(
                artifact.get("credentials_source"), "artifact_store.credentials_source"
            ),
        ),
        workers=workers,
    )


def _required_str(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigurationError(f"{field} must be a non-empty string")
    return value


def _optional_str(value: object, field: str) -> str | None:
    if value is None:
        return None
    return _required_str(value, field)
