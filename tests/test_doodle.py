"""Doodle degradations: deterministic, size-scaled, input untouched, held-out kinds apart."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from mojidiff.learning.doodle import (
    EVAL_KINDS,
    HELDOUT_KINDS,
    TRAIN_KINDS,
    apply_kind,
    degrade,
    degrade_images,
    degrade_split,
    heldout,
    ink_mask,
)


def _icon(size: int) -> np.ndarray:
    """A yellow disc with a black outline and a black eye on white, like an OpenMoji face."""

    rows, cols = np.mgrid[0:size, 0:size]
    centre, radius = (size - 1) / 2, size * 0.35
    distance = np.hypot(rows - centre, cols - centre)
    image = np.full((size, size, 3), 255, dtype=np.uint8)
    image[distance < radius] = (252, 234, 43)
    image[np.abs(distance - radius) < max(1.0, size / 48)] = 0
    image[np.hypot(rows - centre * 0.8, cols - centre * 0.8) < size / 20] = 0
    return image


def test_the_kinds_are_disjoint_and_complete() -> None:
    assert not set(TRAIN_KINDS) & set(HELDOUT_KINDS)
    assert set(EVAL_KINDS) == {"clean", *TRAIN_KINDS, *HELDOUT_KINDS}


@pytest.mark.parametrize("size", [32, 144])
@pytest.mark.parametrize("kind", [*TRAIN_KINDS, *HELDOUT_KINDS])
def test_every_kind_is_deterministic_keeps_the_shape_and_leaves_the_input(
    kind: str, size: int
) -> None:
    image = _icon(size)
    before = image.copy()
    first, params = apply_kind(image, np.random.default_rng(3), kind)
    again, params_again = apply_kind(image, np.random.default_rng(3), kind)
    assert np.array_equal(image, before)
    assert first.shape == image.shape and first.dtype == np.uint8
    assert np.array_equal(first, again) and params == params_again
    assert params["kind"] == kind
    assert not np.array_equal(first, image)


def test_lineart_removes_the_colour_and_keeps_an_outline() -> None:
    image = _icon(144)
    rng = np.random.default_rng(0)
    for _ in range(5):
        out, params = degrade(image, rng, ("lineart",))
        saturation = out.max(axis=2).astype(int) - out.min(axis=2).astype(int)
        assert params["kind"] == "lineart"
        assert saturation.max() < 40  # no yellow left, only ink and paper (plus noise)
        assert ink_mask(out).sum() > 0.25 * ink_mask(image).sum()


def test_the_training_mix_draws_every_family_and_rejects_held_out_kinds() -> None:
    image = _icon(32)
    rng = np.random.default_rng(1)
    seen = {degrade(image, rng)[1]["kind"] for _ in range(60)}
    assert seen == set(TRAIN_KINDS)
    with pytest.raises(ValueError):
        degrade(image, rng, ("pencil",))
    with pytest.raises(ValueError):
        heldout(image, rng, "lineart")


def test_evaluation_sets_are_per_icon_reproducible_and_leave_the_split_alone() -> None:
    from mojidiff.learning.render2svg import SplitData

    images = np.stack([_icon(32), 255 - _icon(32), _icon(32)])
    hexcodes = ["A", "B", "C"]
    out, params = degrade_images(images, hexcodes, ("pencil",), seed=4)
    reordered, _ = degrade_images(images[[2, 0]], ["C", "A"], ("pencil",), seed=4)
    assert np.array_equal(out[0], reordered[1]) and np.array_equal(out[2], reordered[0])
    assert len(params) == 3 and all(p["kind"] == "pencil" for p in params)
    assert np.array_equal(images[1], 255 - _icon(32))

    split = SplitData(
        hexcodes=hexcodes,
        tokens=torch.zeros((3, 4), dtype=torch.long),
        masks=np.zeros((3, 1), dtype=np.uint8),
        images=torch.from_numpy(images.copy()),
        targets=torch.zeros((3, 8, 8, 3), dtype=torch.uint8),
        shape=(4, 2),
    )
    degraded = degrade_split(split, TRAIN_KINDS, seed=0)
    assert degraded.tokens is split.tokens and degraded.targets is split.targets
    assert degraded.masks is split.masks and degraded.hexcodes == split.hexcodes
    assert torch.equal(split.images, torch.from_numpy(images))
    assert not torch.equal(degraded.images, split.images)
    assert torch.equal(degrade_split(split, TRAIN_KINDS, seed=0).images, degraded.images)


def test_doodle_eval_builds_each_set_once_and_downsamples(tmp_path: Path) -> None:
    from mojidiff.learning.doodle_eval import degraded_sets, downsample

    images = np.stack([_icon(32), _icon(32)])
    sets = degraded_sets(images, ["A", "B"], ("clean", "lineart"), 1, tmp_path)
    assert np.array_equal(sets["clean"], images)
    assert (tmp_path / "lineart.npz").exists() and (tmp_path / "lineart.params.json").exists()
    again = degraded_sets(images, ["A", "B"], ("clean", "lineart"), 1, tmp_path)
    assert np.array_equal(again["lineart"], sets["lineart"])
    small = downsample(images[0], 16)
    assert small.shape == (16, 16, 3) and small.dtype == np.uint8
