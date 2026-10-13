"""Paired comparison of two render-to-SVG runs on the icons both scored.

Each run writes `per_icon.jsonl` (hexcode and 72 px pixel error per validation icon).
A candidate is compared with its parent icon by icon: the reduction is parent error
minus candidate error, so positive means the candidate is better, with the same
bootstrap interval as the runs\x27 own summaries.

    python -m mojidiff.learning.paired runs/<candidate> runs/<parent>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from mojidiff.learning.render2svg import bootstrap_mean_interval


def load_errors(run_dir: Path) -> dict[str, float]:
    errors: dict[str, float] = {}
    for line in (run_dir / "per_icon.jsonl").read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            errors[row["hexcode"]] = float(row["model_pixel_error"])
    return errors


def paired_reduction(candidate: dict[str, float], parent: dict[str, float]) -> dict[str, Any]:
    if set(candidate) != set(parent):
        raise ValueError(f"runs scored different icons: {len(set(candidate) ^ set(parent))} differ")
    icons = sorted(candidate)
    reductions = [parent[icon] - candidate[icon] for icon in icons]
    mean, low, high = bootstrap_mean_interval(reductions)
    return {
        "icons": len(icons),
        "candidate_pixel_error": bootstrap_mean_interval([candidate[i] for i in icons]),
        "parent_pixel_error": bootstrap_mean_interval([parent[i] for i in icons]),
        "parent_minus_candidate": [mean, low, high],
        "candidate_better_icons": sum(r > 0 for r in reductions),
        "parent_better_icons": sum(r < 0 for r in reductions),
        "interval_excludes_zero_in_favour": low > 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("candidate", type=Path)
    parser.add_argument("parent", type=Path)
    args = parser.parse_args()
    result = paired_reduction(load_errors(args.candidate), load_errors(args.parent))
    print(json.dumps({"candidate": args.candidate.name, "parent": args.parent.name, **result}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
