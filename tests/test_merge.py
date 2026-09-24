"""The merge gate's data: ZWJ targets with their components, the prompt, the overlay baseline."""

from __future__ import annotations

from pathlib import Path

from mojidiff.learning.merge_study import merge_pairs, merge_prompt, overlay
from mojidiff.learning.omnisvg import parse_into_codec
from mojidiff.learning.openmoji_pilot import (
    _load_program,
    _selected_codec,
    load_openmoji_pilot_config,
    load_pilot_index,
)
from mojidiff.learning.prior import SVG_CLOSE, SVG_OPEN, compact_svg

_PILOT = Path("configs/learning/openmoji-g1-geometric-gate-v16.yaml")


def test_zwj_targets_pair_with_their_components_by_split() -> None:
    pilot = load_openmoji_pilot_config(_PILOT)
    by_split, _, _ = load_pilot_index(pilot)
    pairs = merge_pairs(by_split)
    assert set(pairs) <= set(by_split)
    total = sum(len(items) for items in pairs.values())
    assert total >= 1000
    for split, items in pairs.items():
        for target, components in items:
            assert target.split == split and "200D" in target.hexcode
            assert len(components) >= 2
            assert all("200D" not in c.hexcode for c in components)


def test_prompt_and_overlay_are_well_formed_and_the_overlay_parses() -> None:
    pilot = load_openmoji_pilot_config(_PILOT)
    by_split, _, _ = load_pilot_index(pilot)
    codec = _selected_codec(pilot)
    target, components = merge_pairs(by_split)["primary/validation"][0]
    svgs = [compact_svg(_load_program(c, pilot, codec), codec, pilot.total_segment_slots) for c in components]
    prompt = merge_prompt("woman surfing", [("person surfing", svgs[0]), ("female sign", svgs[1])])
    assert prompt.startswith("<!-- merge: woman surfing -->\n<!-- part: person surfing -->\n<svg")
    assert prompt.endswith("<!-- merged -->\n")
    stacked = overlay(svgs)
    assert stacked.startswith(SVG_OPEN) and stacked.endswith(SVG_CLOSE)
    program, info = parse_into_codec(stacked.encode(), codec, pilot.total_segment_slots)
    # Two icons' paths on one box either fit the bucket or fail for capacity, never for grammar.
    assert program is not None or info["failure"].startswith(("pack", "encode", "projection_required"))
