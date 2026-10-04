"""Build the icon cache the demos read: `data/processed/kitbash`.

One file per icon of the pilot bucket (`<hexcode>.svg`), each the canonical program the
typed codec makes of the OpenMoji source - the packed serializer's own output - and an
`index.json` listing every icon in split order (train, validation, test) with its
annotation, split, group and subgroup. The vectorise demo, the gallery and the kitbash
tool all read it; only held-out icons are ever offered for transcription.

Sources are read from the immutable OpenMoji checkout through `_load_program`, which
refuses any file whose sha256 differs from the tracked hybrid ledger, so a cache built
here is the cache the models were evaluated against or no cache at all.

    python -m mojidiff.vectorise.icon_cache [--out data/processed/kitbash] [--check]

`--check` builds into memory and compares with what is on disk instead of writing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator
from pathlib import Path

from mojidiff.vectorise.server import ICON_CACHE, PILOT_CONFIG

SPLITS = ("primary/train", "primary/validation", "primary/test")
INDEX = "index.json"


def icon_cache_files(
    pilot_config: Path = PILOT_CONFIG, *, limit: int | None = None
) -> Iterator[tuple[str, bytes]]:
    """(file name, bytes) of every cache file: the icons' SVGs, then `index.json`.

    `limit` keeps the first icons of each split only (for tests); the index then lists
    exactly the icons written.
    """

    from mojidiff.learning.openmoji_pilot import (
        _load_program,
        _selected_codec,
        load_openmoji_pilot_config,
        load_pilot_index,
    )
    from mojidiff.representation.packed import serialize_packed_svg

    pilot = load_openmoji_pilot_config(pilot_config)
    codec = _selected_codec(pilot)
    by_split, _, _ = load_pilot_index(pilot)
    annotations = {
        str(entry["hexcode"]): str(entry["annotation"])
        for entry in json.loads((pilot.raw_root / "data" / "openmoji.json").read_text())
    }
    index: list[dict[str, str]] = []
    for split in SPLITS:
        rows = by_split[split] if limit is None else by_split[split][:limit]
        for row in rows:
            program = _load_program(row, pilot, codec)
            svg = serialize_packed_svg(program, codec, pilot.total_segment_slots)
            yield f"{row.hexcode}.svg", svg
            index.append(
                {
                    "hexcode": row.hexcode,
                    "annotation": annotations[row.hexcode],
                    "split": row.split,
                    "group": row.group,
                    "subgroup": row.subgroup,
                }
            )
    yield INDEX, json.dumps(index).encode()


def build_icon_cache(
    out: Path = ICON_CACHE, pilot_config: Path = PILOT_CONFIG, *, limit: int | None = None
) -> int:
    """Write the cache into `out`; the number of icons written. `index.json` is written
    last, so a cache cut short has no index and is never served."""

    out.mkdir(parents=True, exist_ok=True)
    icons = 0
    for name, data in icon_cache_files(pilot_config, limit=limit):
        target = out / name
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(target)
        icons += name != INDEX
    return icons


def check_icon_cache(out: Path = ICON_CACHE, pilot_config: Path = PILOT_CONFIG) -> list[str]:
    """Names of cache files that are missing from `out` or differ from a fresh build."""

    return [
        name
        for name, data in icon_cache_files(pilot_config)
        if not (out / name).is_file() or (out / name).read_bytes() != data
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=ICON_CACHE, help="cache directory")
    parser.add_argument(
        "--pilot-config", type=Path, default=PILOT_CONFIG, help="pilot config naming the bucket"
    )
    parser.add_argument("--check", action="store_true", help="compare with disk, write nothing")
    args = parser.parse_args(argv)
    if args.check:
        differing = check_icon_cache(args.out, args.pilot_config)
        print(f"{len(differing)} file(s) missing or different in {args.out}", file=sys.stderr)
        for name in differing[:20]:
            print(f"  {name}", file=sys.stderr)
        return 1 if differing else 0
    icons = build_icon_cache(args.out, args.pilot_config)
    print(f"{icons} icons and {INDEX} written to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
