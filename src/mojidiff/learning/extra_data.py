"""Extra training icons from other emoji sets, through the OpenMoji codec.

Only icons that fit the codec's budget as they are - no projection, no clamping - are
kept, and only for training. Held-out concepts stay held out: a source icon is dropped
when its base emoji, the codepoints left after removing variation selectors, skin
tones, gender signs and joiners, matches the base of any OpenMoji validation or test
icon. The adapter is `curation.external_probe.adapt` (PicoSVG, gradients to mean
colour, palette snapping), so colours are OpenMoji's palette, not the source's.
"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mojidiff.learning.autoregressive import SequenceLayout, flatten_program
from mojidiff.learning.render2svg import (
    CACHE_ROOT,
    EVAL_SIZE,
    SplitData,
    program_legal_masks,
    render_trusted_rgb,
)

EXTRA_FORMAT = 1
_MODIFIERS = {"FE0F", "FE0E", "200D", "2640", "2642", "1F3FB", "1F3FC", "1F3FD", "1F3FE", "1F3FF"}


def base_key(hexcode: str) -> tuple[str, ...]:
    """An emoji's base: its codepoints without selectors, skin tones, genders, joiners."""

    parts = [part.upper() for part in hexcode.replace("_", "-").split("-") if part]
    return tuple(part for part in parts if part not in _MODIFIERS)


def held_out_keys(pilot: Any) -> set[tuple[str, ...]]:
    """Bases of every OpenMoji icon outside the training split, and of their families."""

    from mojidiff.learning.openmoji_pilot import load_pilot_index

    by_split, _, _ = load_pilot_index(pilot)
    rows = [row for rows in by_split.values() for row in rows]
    held_families = {row.variant_family_id for row in rows if row.split != "primary/train"}
    return {
        base_key(row.hexcode)
        for row in rows
        if row.split != "primary/train" or row.variant_family_id in held_families
    }


def _twemoji_hexcode(path: Path) -> str:
    return path.stem


def _prepare(args: tuple[str, bytes, SequenceLayout, Any, int]) -> dict[str, Any] | None:
    from mojidiff.curation.external_probe import adapt
    from mojidiff.learning.omnisvg import parse_into_codec
    from mojidiff.representation.packed import serialize_packed_svg

    key, source, layout, template, size = args
    try:
        adapted, _ = adapt(source, layout.codec.palette)
    except Exception:  # noqa: BLE001 - an icon the adapter refuses is simply not used
        return None
    program, _ = parse_into_codec(adapted, layout.codec, layout.total_segment_slots)
    if program is None:
        return None
    svg = serialize_packed_svg(program, layout.codec, layout.total_segment_slots)
    tokens = flatten_program(program, layout)
    return {
        "key": key,
        "tokens": tokens.numpy().astype(np.int16),
        "mask": np.packbits(program_legal_masks(tokens, layout).reshape(-1)),
        "image": render_trusted_rgb(svg, size),
        "target": render_trusted_rgb(svg, EVAL_SIZE),
    }


def load_twemoji(
    layout: SequenceLayout, pilot: Any, template: Any, image_size: int, *, workers: int = 12
) -> tuple[SplitData, dict[str, Any]]:
    """Twemoji icons that fit the codec, minus held-out concepts, as a training split."""

    from mojidiff.curation.external_probe import EXTERNAL, _commit

    commit = _commit("twemoji")
    paths = sorted((EXTERNAL / "twemoji/assets/svg").glob("*.svg"))
    excluded = held_out_keys(pilot)
    kept_paths = [p for p in paths if base_key(_twemoji_hexcode(p)) not in excluded]
    settings = {
        "format": EXTRA_FORMAT,
        "commit": commit,
        "size": image_size,
        "palette": list(layout.codec.palette),
        "slots": layout.total_segment_slots,
        "excluded_bases": len(excluded),
        "candidates": len(kept_paths),
    }
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:12]
    path = CACHE_ROOT / f"render2svg-extra-twemoji-{digest}.npz"
    if not path.exists():
        jobs = [(p.stem, p.read_bytes(), layout, template, image_size) for p in kept_paths]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            records = [r for r in pool.map(_prepare, jobs, chunksize=8) if r is not None]
        temporary = path.with_suffix(".tmp.npz")
        np.savez(
            temporary,
            hexcodes=np.array([r["key"] for r in records]),
            tokens=np.stack([r["tokens"] for r in records]),
            masks=np.stack([r["mask"] for r in records]),
            images=np.stack([r["image"] for r in records]),
            targets=np.stack([r["target"] for r in records]),
        )
        temporary.rename(path)
    data = np.load(path)
    split = SplitData(
        hexcodes=[f"twemoji:{h}" for h in data["hexcodes"]],
        tokens=torch.from_numpy(data["tokens"].astype(np.int64)),
        masks=data["masks"],
        images=torch.from_numpy(data["images"]),
        targets=torch.from_numpy(data["targets"]),
        shape=(layout.length, layout.vocabulary),
    )
    report = {
        **settings,
        "source_files": len(paths),
        "excluded_as_held_out": len(paths) - len(kept_paths),
        "fit_and_kept": len(split.tokens),
        "cache": str(path),
        "cache_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    return split, report


def concatenate(first: SplitData, second: SplitData) -> SplitData:
    return SplitData(
        hexcodes=first.hexcodes + second.hexcodes,
        tokens=torch.cat((first.tokens, second.tokens)),
        masks=np.concatenate((first.masks, second.masks)),
        images=torch.cat((first.images, second.images)),
        targets=torch.cat((first.targets, second.targets)),
        shape=first.shape,
    )
