#!/usr/bin/env python3
"""Download a Space's checkpoints from the public Hugging Face model repo, at build time.

    python space/fetch_checkpoints.py space/checkpoints.json

`checkpoints.json` is written by `scripts/build_space.py` from the repository's
`reports/checkpoints-hf.json`: the model repo, a revision, the target directory, and per
file its byte count and sha256. Each file lands at `<target>/<path in the repo>`, so
`runs/<run_id>/best.pt` lands where the demos look for it
(`/home/dev/.cache/mojidiff/runs/<run_id>/best.pt`, `render2svg.CACHE_ROOT`). A file
whose size or sha256 differs from the index stops the build.

Runs in a throwaway environment with only `huggingface_hub`; it imports nothing of
MojiDiff.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    spec = json.loads(Path(argv[1]).read_text())
    target = Path(spec["target"])
    target.mkdir(parents=True, exist_ok=True)
    # The hub's own cache would keep a second copy of every file in the image.
    scratch = Path(tempfile.mkdtemp(prefix="hf-"))
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ["HF_HUB_CACHE"] = str(scratch / "hub")
    os.environ["HF_XET_CACHE"] = str(scratch / "xet")

    from huggingface_hub import hf_hub_download

    started = time.perf_counter()
    total = 0
    try:
        for name, expected in sorted(spec["files"].items()):
            path = Path(
                hf_hub_download(
                    repo_id=spec["repo"],
                    filename=name,
                    revision=spec.get("revision", "main"),
                    local_dir=target,
                )
            )
            size, digest = path.stat().st_size, _sha256(path)
            if size != expected["bytes"] or digest != expected["sha256"]:
                print(
                    f"{name}: {size} bytes sha256 {digest}, expected {expected['bytes']} bytes "
                    f"sha256 {expected['sha256']}",
                    file=sys.stderr,
                )
                return 1
            total += size
            print(f"{name}: {size:,} bytes, sha256 ok", file=sys.stderr, flush=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
        shutil.rmtree(target / ".cache", ignore_errors=True)  # hub download metadata
    elapsed = time.perf_counter() - started
    print(
        f"{len(spec['files'])} checkpoints, {total / 2**20:.0f} MiB, in {elapsed:.0f} s",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
