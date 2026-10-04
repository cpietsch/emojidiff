"""`scripts/build_space.py`: the Hugging Face Spaces are assembled from repository files."""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


def _module() -> ModuleType:
    script = ROOT / "scripts" / "build_space.py"
    spec = importlib.util.spec_from_file_location("build_space", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build_space = _module()


def _front_matter(readme: str) -> dict[str, object]:
    match = re.match(r"---\n(.*?)\n---\n", readme, re.DOTALL)
    assert match is not None
    loaded = yaml.safe_load(match.group(1))
    assert isinstance(loaded, dict)
    return loaded


@pytest.mark.parametrize("demo", ["gallery", "vectorise"])
def test_a_space_holds_what_its_demo_reads_and_a_docker_space_header(
    demo: str, tmp_path: Path
) -> None:
    out = tmp_path / demo
    record = build_space.assemble(demo, out, allow_dirty=True)
    files = set(record["files"])
    for path in (
        "pyproject.toml",
        "uv.lock",
        f"scripts/serve_{demo}.py",
        "src/mojidiff/representation/render_worker.py",
        "src/mojidiff/vectorise/icon_cache.py",
        "configs/learning/openmoji-g1-geometric-gate-v16.yaml",
        "reports/codec/full-primary-structure-v2-opacity/hybrid.jsonl",
        "space/fetch_checkpoints.py",
        "space/warm_cache.py",
    ):
        assert path in files and (out / path).read_bytes() == (ROOT / path).read_bytes(), path
    assert not any("__pycache__" in path or "/weblog/" in path for path in files)
    # Byte-identical copies: the corpus cache key hashes these two files.
    for name in ("autoregressive.py", "openmoji_pilot.py"):
        path = f"src/mojidiff/learning/{name}"
        assert (out / path).read_bytes() == (ROOT / path).read_bytes()

    meta = _front_matter((out / "README.md").read_text())
    assert meta["sdk"] == "docker" and meta["app_port"] == 7860
    assert meta["license"] == "cc-by-sa-4.0"
    assert isinstance(meta["short_description"], str) and len(meta["short_description"]) <= 60
    assert {"title", "emoji", "colorFrom", "colorTo"} <= set(meta)
    readme = (out / "README.md").read_text()
    assert "__" not in readme.replace("__init__", ""), "a measurement placeholder is unfilled"

    dockerfile = (out / "Dockerfile").read_text()
    assert dockerfile == (ROOT / "space" / demo / "Dockerfile").read_text()
    assert '"--host", "0.0.0.0", "--port", "7860"' in dockerfile
    assert "useradd --create-home --uid 1000 --home-dir /home/dev" in dockerfile
    assert "--extra model-cpu" in dockerfile

    index = json.loads((ROOT / "reports" / "checkpoints-hf.json").read_text())
    checkpoints = json.loads((out / "space" / "checkpoints.json").read_text())
    assert checkpoints["repo"] == "chrispie/mojidiff-checkpoints"
    assert checkpoints["target"] == "/home/dev/.cache/mojidiff"
    for name, entry in checkpoints["files"].items():
        assert entry == {key: index["files"][name][key] for key in ("bytes", "sha256")}
    registry = yaml.safe_load((ROOT / "configs" / "gallery" / "models.yaml").read_text())
    run_ids = {entry["id"]: entry["run_id"] for entry in registry["models"]}
    if demo == "gallery":
        assert set(checkpoints["files"]) == {f"runs/{r}/best.pt" for r in run_ids.values()}
        assert {f"runs/{r}/run.yaml" for r in run_ids.values()} <= files
        assert '"--no-ratings"' in dockerfile
    else:
        assert list(checkpoints["files"]) == [f"runs/{run_ids['v9']}/best.pt"]
        assert f"/runs/{run_ids['v9']}/best.pt" in dockerfile
    source = json.loads((out / "space-source.json").read_text())
    assert source["demo"] == demo and re.fullmatch(r"[0-9a-f]{40}", source["commit"])


def test_assembly_refuses_a_non_empty_outdir_and_uncommitted_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "busy").mkdir()
    (tmp_path / "busy" / "file").write_text("x")
    with pytest.raises(FileExistsError):
        build_space.assemble("vectorise", tmp_path / "busy", allow_dirty=True)
    asked: list[list[str]] = []

    def dirty(paths: list[str], root: Path) -> list[str]:
        asked.append(paths)
        return ["src/x.py"]

    monkeypatch.setattr(build_space, "dirty_files", dirty)
    with pytest.raises(RuntimeError, match="differ from HEAD"):
        build_space.assemble("vectorise", tmp_path / "clean")
    assert not (tmp_path / "clean").exists()
    # What decides the downloaded checkpoints is checked too, though the Space omits it.
    assert {"reports/checkpoints-hf.json", "configs/gallery/models.yaml"} <= set(asked[0])
    with pytest.raises(ValueError):
        build_space.files_for("kitbash")
