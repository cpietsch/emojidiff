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


def test_openmoji_encodes_into_omnisvg_tokens_and_decodes_back_exactly() -> None:
    """The training encoder and OmniSVG's released decoder agree, point for point.

    The tokens sit one above the values the training repository's YAML lists; read at
    the YAML's values, a close decodes as an arc. This pins the corrected arithmetic.
    """

    import pytest

    from mojidiff.learning.omnisvg import EXTERNAL, decode_tokens, load_svg_tokenizer
    from mojidiff.learning.omnisvg_encode import (
        BOS,
        CMD_CLOSE,
        CMD_MOVE,
        EOS,
        color_token,
        encode_icon,
        openmoji_to_deepsvg,
    )
    from mojidiff.learning.openmoji_pilot import _select_rows, load_pilot_index

    if not (EXTERNAL / "tokenizer.py").is_file():
        pytest.skip("the OmniSVG inference clone is not present")
    pilot = load_openmoji_pilot_config(_PILOT)
    by_split, _, _ = load_pilot_index(pilot)
    decoder, black = load_svg_tokenizer()
    for row in _select_rows(by_split["primary/train"], 2, 3):
        source = (pilot.raw_root / row.source_path).read_bytes()
        svg, fills = openmoji_to_deepsvg(source)
        tokens = encode_icon(source)
        assert int(tokens[0]) == BOS and int(tokens[-1]) == EOS and int(tokens[1]) == CMD_MOVE
        assert (tokens == CMD_CLOSE).sum() >= 1
        decoded, info = decode_tokens(decoder, black, tokens[1:])
        assert decoded is not None, info
        assert info["paths"] == len(svg.svg_path_groups) == len(fills)
        # Every path's end points come back exactly on the 200-unit grid.
        points = decoder.process_generated_tokens(tokens[None])
        tensors, colours = decoder.raster_svg(points)
        for group, back in zip(svg.svg_path_groups, tensors[0], strict=True):
            expected = group.to_tensor(PAD_VAL=0).round().int().clip(0, 199)[:, 12:14]
            assert expected.shape == back[:, 12:14].shape
            assert bool((expected == back[:, 12:14].int()).all())
        # The decoder reports a colour as its token less the base vocabulary and one;
        # its own black token, 40,012, is the encoder's black at 191,949 read that way.
        assert colours[: len(fills)] == [color_token(fill) - 151_937 for fill in fills]
        assert color_token("#000000") - 151_937 == black
    assert color_token("none") == color_token(None)
    assert color_token("#fff") == color_token("#ffffff")
