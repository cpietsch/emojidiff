"""The staged remote wrapper must agree with the pilot's real artifact layout."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from mojidiff.learning.openmoji_pilot import SUMMARY_FILENAME, load_openmoji_pilot_config

_REPO_ROOT = Path(__file__).resolve().parents[1]
_REMOTE_ROOT = _REPO_ROOT / "scripts" / "remote"
_CONFIG = _REPO_ROOT / "configs" / "learning" / "openmoji-g1-dominant-bucket-smoke.yaml"


def _load_wrapper() -> ModuleType:
    """Import the staged wrapper the way the worker does, without installing it."""

    if str(_REMOTE_ROOT) not in sys.path:
        sys.path.insert(0, str(_REMOTE_ROOT))
    spec = importlib.util.spec_from_file_location(
        "openmoji_pilot_smoke", _REMOTE_ROOT / "openmoji_pilot_smoke.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_wrapper_reads_the_summary_the_pilot_actually_writes(tmp_path: Path) -> None:
    wrapper = _load_wrapper()
    pilot = load_openmoji_pilot_config(_CONFIG)
    output = tmp_path / "smoke"

    summary, configured = wrapper.run_openmoji_pilot_with_roots(pilot, _CONFIG, output)

    assert configured.report_root == output / "report"
    assert configured.checkpoint_root == output / "checkpoint"
    assert (configured.report_root / SUMMARY_FILENAME).is_file()
    assert (configured.checkpoint_root / "checkpoint.zip").is_file()
    assert summary["checkpoint_round_trip"] is True
    assert summary["locked_path_exact"] is True
    # The wrapper previously digested `output / summary.json`, which never exists. A
    # complete GPU run then failed only on its result contract.
    assert not (output / SUMMARY_FILENAME).exists()
