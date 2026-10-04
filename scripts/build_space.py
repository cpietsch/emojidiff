#!/usr/bin/env python3
"""Assemble a Hugging Face Docker Space for one of the live demos.

    python scripts/build_space.py gallery|vectorise <outdir> [--allow-dirty]

The Space repository is built from this repository's files, never by cloning it: the
package (`src/mojidiff`, without the tools a Space does not serve), the demo's serve
script, the pilot config and the three ledgers it pins, `pyproject.toml` and `uv.lock`,
and for the gallery its registry and each model's `runs/<run_id>/run.yaml` (the stats
the page shows). From `space/` it takes the demo's Dockerfile and Space README and the
build-time helpers, and it writes `space/checkpoints.json`: the model repo files the
image downloads, each with the size and sha256 that `reports/checkpoints-hf.json`
records. `space-source.json` names the commit the files came from.

Everything else - the OpenMoji 17.0.0 sources, the checkpoints, the icon cache, the
gallery's corpus - is fetched or built inside `docker build`; see the Dockerfiles.

`<outdir>` must be empty or absent. Files with uncommitted changes refuse the build
unless `--allow-dirty`, which records them in `space-source.json`; upload a Space only
from a clean tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import cast

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEMOS = ("gallery", "vectorise")
PACKAGES = ("curation", "gallery", "learning", "orchestration", "representation", "vectorise")
"""Subpackages of `mojidiff` the demos import (the weblog, kitbash and demo tools stay)."""
COMMON = (
    "pyproject.toml",
    "uv.lock",
    "src/mojidiff/__init__.py",
    "configs/learning/openmoji-g1-geometric-gate-v16.yaml",
    "reports/codec/full-primary-structure-v2-opacity/hybrid.jsonl",
    "reports/codec/capacity-layout-v1/bucket-assignments.jsonl",
    "reports/codec/style-vocabulary-v2-render-sentinel/summary.json",
    "space/fetch_checkpoints.py",
    "space/warm_cache.py",
)
OPTIONAL = ("LICENSE",)
"""Copied when present: the repository's licence, CC BY-SA 4.0 like OpenMoji."""
REGISTRY = "configs/gallery/models.yaml"
CHECKPOINT_INDEX = "reports/checkpoints-hf.json"
CACHE_ROOT = "/home/dev/.cache/mojidiff"
"""Where the image keeps checkpoints: `render2svg.CACHE_ROOT`, under uid 1000's home."""
VECTORISE_MODEL = "v9"
"""The registry id of the model the vectorise Space serves."""
LARGE = 10 * 2**20
"""Files above this go through LFS/Xet when the Space is pushed with git."""


def _run_ids(registry: Path) -> dict[str, str]:
    """Registry id to run id, in display order."""

    root = yaml.safe_load(registry.read_text())
    return {str(entry["id"]): str(entry["run_id"]) for entry in root["models"]}


def files_for(demo: str, root: Path = REPO_ROOT) -> list[str]:
    """Repository-relative paths the Space copies, sorted."""

    if demo not in DEMOS:
        raise ValueError(f"demo must be one of {', '.join(DEMOS)}, got {demo!r}")
    paths = set(COMMON)
    for package in PACKAGES:
        for path in (root / "src" / "mojidiff" / package).rglob("*"):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                paths.add(path.relative_to(root).as_posix())
    paths.add(f"scripts/serve_{demo}.py")
    if demo == "gallery":
        paths.add(REGISTRY)
        paths.update(f"runs/{run_id}/run.yaml" for run_id in _run_ids(root / REGISTRY).values())
    paths.update(name for name in OPTIONAL if (root / name).is_file())
    return sorted(paths)


def checkpoints_for(demo: str, root: Path = REPO_ROOT) -> dict[str, object]:
    """`space/checkpoints.json`: the model repo files the image downloads, with the size
    and sha256 the mirror's index records for each."""

    index = json.loads((root / CHECKPOINT_INDEX).read_text())
    run_ids = _run_ids(root / REGISTRY)
    wanted = list(run_ids.values()) if demo == "gallery" else [run_ids[VECTORISE_MODEL]]
    files: dict[str, dict[str, object]] = {}
    for run_id in wanted:
        name = f"runs/{run_id}/best.pt"
        entry = index["files"].get(name)
        if entry is None:
            raise ValueError(f"{name} is not in {CHECKPOINT_INDEX}")
        files[name] = {"bytes": int(entry["bytes"]), "sha256": str(entry["sha256"])}
    repo = str(index["repo"]).removeprefix("https://huggingface.co/").strip("/")
    return {"repo": repo, "revision": "main", "target": CACHE_ROOT, "files": files}


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=root, capture_output=True, text=True, check=True
    ).stdout


def dirty_files(paths: list[str], root: Path = REPO_ROOT) -> list[str]:
    """Paths among `paths` that are untracked or differ from HEAD."""

    status = _git(root, "status", "--porcelain", "--untracked-files=all", "--", *paths)
    return sorted(line[3:].strip() for line in status.splitlines() if line.strip())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assemble(
    demo: str, out: Path, *, root: Path = REPO_ROOT, allow_dirty: bool = False
) -> dict[str, object]:
    """Write the Space for `demo` into `out` (empty or absent); its source record."""

    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"{out} is not empty")
    paths = files_for(demo, root)
    missing = [path for path in paths if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError("missing: " + ", ".join(missing))
    # The checkpoint index and the registry decide what the image downloads, so they count
    # even where the Space does not copy them.
    inputs = [f"space/{demo}/Dockerfile", f"space/{demo}/README.md", CHECKPOINT_INDEX, REGISTRY]
    dirty = dirty_files(sorted({*paths, *inputs}), root)
    if dirty and not allow_dirty:
        raise RuntimeError(
            f"{len(dirty)} file(s) differ from HEAD (commit them, or --allow-dirty): "
            + ", ".join(dirty[:10])
        )
    space = root / "space" / demo
    dockerfile = (space / "Dockerfile").read_text()
    if demo == "vectorise":
        run_id = _run_ids(root / REGISTRY)[VECTORISE_MODEL]
        if f"/runs/{run_id}/best.pt" not in dockerfile:
            raise ValueError(f"space/vectorise/Dockerfile does not serve {run_id}")
    out.mkdir(parents=True, exist_ok=True)
    for path in paths:
        target = out / path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / path, target)
    (out / "Dockerfile").write_text(dockerfile)
    shutil.copyfile(space / "README.md", out / "README.md")
    checkpoints = checkpoints_for(demo, root)
    checkpoint_files = cast(dict[str, object], checkpoints["files"])
    (out / "space" / "checkpoints.json").write_text(json.dumps(checkpoints, indent=1) + "\n")
    large = sorted(path for path in paths if (root / path).stat().st_size > LARGE)
    (out / ".gitattributes").write_text(
        "".join(f"{path} filter=lfs diff=lfs merge=lfs -text\n" for path in large)
    )
    record: dict[str, object] = {
        "demo": demo,
        "repository": "https://github.com/cpietsch/emojidiff",
        "commit": _git(root, "rev-parse", "HEAD").strip(),
        "dirty": dirty,
        "files": {path: _sha256(root / path) for path in paths},
        "checkpoints": sorted(checkpoint_files),
    }
    (out / "space-source.json").write_text(json.dumps(record, indent=1) + "\n")
    return record


def _front_matter(readme: str) -> dict[str, object]:
    match = re.match(r"---\n(.*?)\n---\n", readme, re.DOTALL)
    if match is None:
        raise ValueError("README.md has no front matter")
    loaded = yaml.safe_load(match.group(1))
    if not isinstance(loaded, dict):
        raise ValueError("README.md front matter is not a mapping")
    return loaded


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("demo", choices=DEMOS)
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--allow-dirty", action="store_true", help="copy uncommitted changes, recorded"
    )
    args = parser.parse_args()
    try:
        record = assemble(args.demo, args.out, allow_dirty=args.allow_dirty)
    except (FileExistsError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"build_space: {error}", file=sys.stderr)
        return 1
    meta = _front_matter((args.out / "README.md").read_text())
    files = cast(dict[str, str], record["files"])
    dirty = cast(list[str], record["dirty"])
    checkpoints = cast(list[str], record["checkpoints"])
    print(
        f"{args.demo} Space in {args.out}: {len(files)} files from {str(record['commit'])[:10]}"
        f"{f' ({len(dirty)} dirty)' if dirty else ''}, {len(checkpoints)} checkpoint(s), "
        f"sdk {meta.get('sdk')}, port {meta.get('app_port')}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
