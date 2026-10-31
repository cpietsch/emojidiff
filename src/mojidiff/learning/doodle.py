"""Doodle degradations of transcriber inputs: a clean render made to look hand-drawn.

The transcriber learns to map these degraded renders to the clean program, so a rough
drawing comes back as a clean OpenMoji-style SVG - where a tracer would copy the mess. Only
the input pixels change; the target tokens are untouched, so the augmentation is the
identity in token space. Every function returns a new array and never writes its input.

Training families (`TRAIN_KINDS`), all with a wobbly outline of changed thickness:
`lineart` (outline only, no colour), `sloppy` (colour misregistered against the outline,
as when colouring outside the lines), `flat` (flat colour under the wobbly outline).
Modifiers on top: grey colour, white erasures, blur and paper noise. Hue changes of the
input alone are left out on purpose: the exact palette permutation (`augment_colour`)
already recolours input and target together, and recolouring only the input would teach the
model to overrule the drawer's colours.

Held-out kinds (`HELDOUT_KINDS`) never appear in training and test generalisation:
`pencil` (grey broken strokes, no colour, paper grain), `hatched` (crayon hatching instead
of flat fills), `affine` (a global rotation and scale) and `thumb` (a 36 px JPEG at quality
20, upscaled again). Pixel parameters are given for 144 px and scale with the image size.
"""

from __future__ import annotations

import dataclasses
import io
import zlib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

if TYPE_CHECKING:
    from mojidiff.learning.render2svg import SplitData

TRAIN_KINDS = ("lineart", "sloppy", "flat")
KIND_WEIGHTS = {"lineart": 0.4, "sloppy": 0.35, "flat": 0.25}
HELDOUT_KINDS = ("pencil", "hatched", "affine", "thumb")
EVAL_KINDS = ("clean", *TRAIN_KINDS, *HELDOUT_KINDS)
WHITE = 255.0
REFERENCE_SIZE = 144


def ink_mask(image: np.ndarray) -> np.ndarray:
    """Soft mask (0 to 1) of the black outline layer: OpenMoji outlines are #000000."""

    darkest = image.max(axis=2).astype(np.float32)
    return np.asarray(np.clip((110.0 - darkest) / 70.0, 0.0, 1.0), dtype=np.float32)


def fill_under_ink(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """The colour layer without outlines: each ink pixel takes its nearest non-ink colour."""

    hard = mask > 0.3
    if not hard.any() or hard.all():
        return image.astype(np.float32)
    _, (rows, cols) = ndi.distance_transform_edt(hard, return_indices=True)
    return np.asarray(image[rows, cols], dtype=np.float32)


def _displacement(
    rng: np.random.Generator, size: int, amplitude: float, cells: int
) -> tuple[np.ndarray, np.ndarray]:
    """A smooth random displacement field: a cells x cells grid, cubic-upsampled."""

    fields = []
    for _ in range(2):
        coarse = rng.normal(size=(cells, cells))
        field = ndi.zoom(coarse, size / cells, order=3)[:size, :size]
        fields.append(field / (np.abs(field).max() + 1e-6) * amplitude)
    return fields[0], fields[1]


def _warp(image: np.ndarray, dy: np.ndarray, dx: np.ndarray, fill: float) -> np.ndarray:
    size = image.shape[0]
    rows, cols = np.mgrid[0:size, 0:size].astype(np.float32)
    coords = [rows + dy, cols + dx]

    def channel(plane: np.ndarray) -> np.ndarray:
        return np.asarray(
            ndi.map_coordinates(
                plane.astype(np.float32), coords, order=1, mode="constant", cval=fill
            )
        )

    if image.ndim == 2:
        return channel(image)
    return np.stack([channel(image[..., c]) for c in range(image.shape[2])], axis=-1)


def _thickness(mask: np.ndarray, steps: int) -> np.ndarray:
    hard = mask > 0.5
    if steps > 0:
        hard = ndi.binary_dilation(hard, iterations=steps)
    elif steps < 0:
        hard = ndi.binary_erosion(hard, iterations=-steps)
    return np.asarray(ndi.gaussian_filter(hard.astype(np.float32), 0.6))


def _to_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def degrade(
    image: np.ndarray, rng: np.random.Generator, kinds: Sequence[str] = TRAIN_KINDS
) -> tuple[np.ndarray, dict[str, Any]]:
    """One training degradation of a (S, S, 3) uint8 render: the image and its parameters.

    The family is drawn from `kinds` with the weights in `KIND_WEIGHTS`."""

    unknown = [kind for kind in kinds if kind not in TRAIN_KINDS]
    if unknown or not kinds:
        raise ValueError(f"training kinds must be among {TRAIN_KINDS}, got {list(kinds)}")
    size = image.shape[0]
    scale = size / REFERENCE_SIZE
    weights = np.array([KIND_WEIGHTS[kind] for kind in kinds])
    kind = str(kinds[int(rng.choice(len(kinds), p=weights / weights.sum()))])
    params: dict[str, Any] = {"kind": kind}
    mask = ink_mask(image)
    colour = fill_under_ink(image, mask)
    amplitude = float(rng.uniform(0.5, 3.0)) * scale
    cells = int(rng.integers(4, 9))
    dy, dx = _displacement(rng, size, amplitude, cells)
    steps = int(rng.choice([-1, 0, 0, 1, 2, 3]))
    steps = int(np.sign(steps) * max(1, round(abs(steps) * scale))) if steps else 0
    params.update(wobble_px=amplitude, wobble_cells=cells, thickness_steps=steps)
    line = _warp(_thickness(mask, steps), dy, dx, fill=0.0)
    if kind == "lineart":
        colour = np.full_like(colour, WHITE)
    else:
        if kind == "sloppy":
            shift = float(rng.uniform(2.0, 5.0)) * scale
            dy2, dx2 = _displacement(rng, size, shift, int(rng.integers(3, 6)))
            colour = _warp(colour, dy2, dx2, fill=WHITE)
            params["misregister_px"] = shift
        if rng.random() < 0.15:
            grey = colour @ np.array([0.299, 0.587, 0.114], np.float32)
            colour = np.repeat(grey[..., None], 3, axis=2)
            params["grey"] = True
    out = colour * (1.0 - line[..., None])
    if rng.random() < 0.25:
        count = int(rng.integers(1, 4))
        rows, cols = np.mgrid[0:size, 0:size]
        margin = max(1, int(16 * scale))
        for _ in range(count):
            cy, cx = rng.integers(margin, max(margin + 1, size - margin), size=2)
            radius = float(rng.uniform(4, 10)) * scale
            out[(rows - cy) ** 2 + (cols - cx) ** 2 < radius * radius] = WHITE
        params["erasures"] = count
    if rng.random() < 0.3:
        sigma = float(rng.uniform(0.6, 1.6)) * scale
        out = np.stack([ndi.gaussian_filter(out[..., c], sigma) for c in range(3)], axis=-1)
        params["blur"] = sigma
    if rng.random() < 0.3:
        noise = float(rng.uniform(2, 8))
        out = out + rng.normal(0, noise, size=out.shape)
        params["noise"] = noise
    return _to_uint8(out), params


def heldout(
    image: np.ndarray, rng: np.random.Generator, kind: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """One held-out degradation (never used in training): the image and its parameters."""

    size = image.shape[0]
    scale = size / REFERENCE_SIZE
    params: dict[str, Any] = {"kind": kind}
    mask = ink_mask(image)
    if kind == "pencil":
        gaps = ndi.gaussian_filter(rng.normal(size=(size, size)), 2.0 * scale)
        keep = (gaps > np.quantile(gaps, 0.25)).astype(np.float32)
        line = mask * keep * 0.7
        out = np.full((size, size, 3), 245.0) * (1 - line[..., None])
        out = out + rng.normal(0, 4, (size, size, 1))
    elif kind == "hatched":
        colour = fill_under_ink(image, mask)
        rows, cols = np.mgrid[0:size, 0:size]
        period = float(rng.integers(4, 7)) * scale
        angle = float(rng.uniform(0, np.pi))
        phase = (cols * np.cos(angle) + rows * np.sin(angle)) * 2 * np.pi / max(period, 2.0)
        stripe = (np.sin(phase) > 0.0).astype(np.float32)[..., None]
        out = (colour * stripe + WHITE * (1 - stripe)) * (1 - mask[..., None])
        params.update(period_px=period, angle=angle)
    elif kind == "affine":
        angle = float(np.deg2rad(rng.uniform(6, 12) * rng.choice([-1, 1])))
        zoom = float(rng.uniform(0.88, 1.12))
        centre = (size - 1) / 2
        matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]) / zoom
        offset = centre - matrix @ np.array([centre, centre])
        out = np.stack(
            [
                ndi.affine_transform(
                    image[..., c].astype(np.float32), matrix, offset=offset, order=1, cval=WHITE
                )
                for c in range(3)
            ],
            axis=-1,
        )
        params.update(angle=angle, zoom=zoom)
    elif kind == "thumb":
        small = max(4, round(36 * scale))
        buffer = io.BytesIO()
        Image.fromarray(image).resize((small, small), Image.Resampling.BILINEAR).save(
            buffer, format="JPEG", quality=20
        )
        buffer.seek(0)
        thumb = Image.open(buffer).convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        out = np.asarray(thumb, dtype=np.float32)
        params.update(thumb_px=small, jpeg_quality=20)
    else:
        raise ValueError(f"held-out kind must be one of {HELDOUT_KINDS}, got {kind!r}")
    return _to_uint8(out), params


def apply_kind(
    image: np.ndarray, rng: np.random.Generator, kind: str
) -> tuple[np.ndarray, dict[str, Any]]:
    """`clean` (a copy), one training family, or one held-out kind."""

    if kind == "clean":
        return image.copy(), {"kind": "clean"}
    if kind in TRAIN_KINDS:
        return degrade(image, rng, (kind,))
    return heldout(image, rng, kind)


def icon_rng(seed: int, hexcode: str, kind: str) -> np.random.Generator:
    """The generator for one icon of an evaluation set: independent of order and count."""

    return np.random.default_rng([seed, zlib.crc32(hexcode.encode()), zlib.crc32(kind.encode())])


def degrade_images(
    images: np.ndarray, hexcodes: Sequence[str], kinds: Sequence[str], seed: int
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """An evaluation set: each (N, S, S, 3) render degraded once, reproducibly per icon.

    One kind is applied as itself (`apply_kind`); several training kinds form a mixed set
    drawn by `degrade`."""

    if len(images) != len(hexcodes):
        raise ValueError("one hexcode per image")
    label = "+".join(kinds)
    out = np.empty_like(images)
    params = []
    for i, (image, hexcode) in enumerate(zip(images, hexcodes, strict=True)):
        rng = icon_rng(seed, hexcode, label)
        if len(kinds) == 1:
            out[i], record = apply_kind(image, rng, kinds[0])
        else:
            out[i], record = degrade(image, rng, kinds)
        params.append(record)
    return out, params


def degrade_split(split: SplitData, kinds: Sequence[str], seed: int) -> SplitData:
    """`split` with degraded input images; tokens, masks and targets are the same objects."""

    import torch

    images, _ = degrade_images(split.images.numpy(), split.hexcodes, kinds, seed)
    return dataclasses.replace(split, images=torch.from_numpy(images))
