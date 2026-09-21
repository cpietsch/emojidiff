"""OmniSVG's output must land in the project's codec, or say why it cannot."""

from __future__ import annotations

from pathlib import Path

from mojidiff.learning.omnisvg import _remap_key, parse_into_codec, snap_to_palette, to_project_svg
from mojidiff.learning.openmoji_pilot import _selected_codec, load_openmoji_pilot_config

_PILOT = Path("configs/learning/openmoji-g1-dominant-bucket-smoke.yaml")

_OMNISVG_STYLE = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0.0 0.0 200.0 200.0" height="200px" '
    'width="200px"><path fill="#aadd66" fill-opacity="1.0"  filling="0" '
    'd="M40.0 40.0 L160.0 40.0 L160.0 160.0 L40.0 160.0 Z"/>'
    '<path fill="#ff0000" fill-opacity="1.0"  filling="0" '
    'd="M60.0 60.0 C60.0 40.0 140.0 40.0 140.0 60.0 L100.0 140.0 Z"/></svg>'
)


def test_fills_snap_to_the_palette_and_report_the_move() -> None:
    palette = ("#000000", "#92d3f5", "#ffffff", "#e67a94")
    snapped, info = snap_to_palette(_OMNISVG_STYLE, palette)
    assert 'fill="#aadd66"' not in snapped and 'fill="#ff0000"' not in snapped
    assert info["fills"] == 2 and info["fills_moved"] == 2 and info["mean_snap_distance_rgb"] > 0
    already, info = snap_to_palette('<path fill="#92d3f5"/>', palette)
    assert 'fill="#92d3f5"' in already and info["fills_moved"] == 0


def test_an_omnisvg_drawing_enters_the_codec_scaled_and_in_palette() -> None:
    pilot = load_openmoji_pilot_config(_PILOT)
    codec = _selected_codec(pilot)
    projected, snap = to_project_svg(_OMNISVG_STYLE, codec.palette)
    assert b"filling=" not in projected and b'viewBox="0 0 72 72"' in projected
    program, info = parse_into_codec(projected, codec, pilot.total_segment_slots)
    assert program is not None, info
    assert info["active_paths"] == 2 and info["segments"] >= 6
    assert "failure" not in info
    # A drawing with more paths than the packed bucket allows is a counted failure.
    many = _OMNISVG_STYLE.replace(
        "</svg>",
        "".join(
            f'<path fill="#000000" d="M{i} {i} L{i + 5} {i} L{i + 5} {i + 5} Z"/>'
            for i in range(40)
        )
        + "</svg>",
    )
    projected, _ = to_project_svg(many, codec.palette)
    program, info = parse_into_codec(projected, codec, pilot.total_segment_slots)
    assert program is None and info["failure"].startswith(("projection_required", "pack", "encode"))


def test_checkpoint_keys_remap_onto_this_layout() -> None:
    expected = {
        "model.visual.blocks.0.norm1.weight",
        "model.language_model.layers.0.mlp.up_proj.weight",
        "lm_head.weight",
    }
    assert (
        _remap_key("transformer.visual.blocks.0.norm1.weight", expected)
        == "model.visual.blocks.0.norm1.weight"
    )
    assert (
        _remap_key("transformer.model.layers.0.mlp.up_proj.weight", expected)
        == "model.language_model.layers.0.mlp.up_proj.weight"
    )
    assert _remap_key("transformer.lm_head.weight", expected) == "lm_head.weight"
    assert _remap_key("transformer.nothing.weight", expected) is None
