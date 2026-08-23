"""Pinned, immutable raw-source acquisition and provenance manifest."""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


class SourceError(RuntimeError):
    """Pinned source configuration or acquisition failed safely."""


_REVISION = re.compile(r"[0-9a-f]{40}\Z")
_TAG = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")


@dataclass(frozen=True)
class OpenMojiSource:
    name: str
    url: str
    tag: str
    revision: str
    license: str
    license_path: str
    metadata_path: str
    palette_path: str
    svg_glob: str
    raw_root: Path
    source_manifest: Path


def load_source_config(path: Path) -> OpenMojiSource:
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SourceError(f"cannot load source config {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise SourceError("source config schema_version must be 1")
    source = document.get("source")
    if not isinstance(source, dict):
        raise SourceError("source must be a mapping")

    def required(mapping: dict[str, Any], key: str) -> str:
        value = mapping.get(key)
        if not isinstance(value, str) or not value:
            raise SourceError(f"{key} must be a non-empty string")
        return value

    tag = required(source, "tag")
    revision = required(source, "revision")
    if not _TAG.fullmatch(tag):
        raise SourceError("source tag contains unsafe characters")
    if not _REVISION.fullmatch(revision):
        raise SourceError("source revision must be a full lowercase Git commit")
    url = required(source, "url")
    if not url.startswith("https://github.com/") or not url.endswith(".git"):
        raise SourceError("source URL must be an HTTPS GitHub repository")

    return OpenMojiSource(
        name=required(source, "name"),
        url=url,
        tag=tag,
        revision=revision,
        license=required(source, "license"),
        license_path=required(source, "license_path"),
        metadata_path=required(source, "metadata_path"),
        palette_path=required(source, "palette_path"),
        svg_glob=required(source, "svg_glob"),
        raw_root=Path(required(document, "raw_root")),
        source_manifest=Path(required(document, "source_manifest")),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_head(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
        timeout=30,
        shell=False,
    )
    if completed.returncode != 0:
        raise SourceError(f"cannot verify raw checkout: {completed.stderr.strip()[-500:]}")
    return completed.stdout.strip()


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), reverse=True):
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode & ~0o222)
    mode = stat.S_IMODE(root.stat().st_mode)
    root.chmod(mode & ~0o222)


def acquire(source: OpenMojiSource) -> dict[str, Any]:
    """Clone once, verify the exact commit, manifest every SVG, and remove write bits."""

    if source.raw_root.exists():
        actual = _git_head(source.raw_root)
        if actual != source.revision:
            raise SourceError(
                f"existing raw checkout revision {actual} does not match {source.revision}"
            )
    else:
        source.raw_root.parent.mkdir(parents=True, exist_ok=True)
        staging_parent = Path("data/downloads")
        staging_parent.mkdir(parents=True, exist_ok=True)
        staging = staging_parent / f"openmoji-{source.tag}.partial-{os.getpid()}"
        if staging.exists():
            raise SourceError(f"refusing to reuse acquisition staging path: {staging}")
        completed = subprocess.run(
            [
                "git",
                "clone",
                "--depth",
                "1",
                "--branch",
                source.tag,
                "--single-branch",
                "--",
                source.url,
                str(staging),
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=900,
            shell=False,
        )
        if completed.returncode != 0:
            raise SourceError(
                f"clone failed; staging preserved at {staging}: {completed.stderr.strip()[-1000:]}"
            )
        actual = _git_head(staging)
        if actual != source.revision:
            raise SourceError(
                f"tag resolved to unexpected revision {actual}; staging preserved at {staging}"
            )
        staging.replace(source.raw_root)

    required_paths = (
        source.raw_root / source.license_path,
        source.raw_root / source.metadata_path,
        source.raw_root / source.palette_path,
    )
    missing = [str(path) for path in required_paths if not path.is_file()]
    if missing:
        raise SourceError(f"pinned checkout lacks required files: {', '.join(missing)}")
    svg_paths = sorted(source.raw_root.glob(source.svg_glob))
    if not svg_paths:
        raise SourceError(f"no SVGs matched {source.svg_glob}")

    rows = [
        {
            "path": path.relative_to(source.raw_root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in svg_paths
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "source": source.name,
        "url": source.url,
        "tag": source.tag,
        "revision": source.revision,
        "license": source.license,
        "license_path": source.license_path,
        "license_sha256": sha256_file(required_paths[0]),
        "metadata_path": source.metadata_path,
        "metadata_sha256": sha256_file(required_paths[1]),
        "palette_path": source.palette_path,
        "palette_sha256": sha256_file(required_paths[2]),
        "svg_count": len(rows),
        "svgs": rows,
    }
    source.source_manifest.parent.mkdir(parents=True, exist_ok=True)
    source.source_manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _make_read_only(source.raw_root)
    return manifest
