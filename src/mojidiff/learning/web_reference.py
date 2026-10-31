"""Ground truth for the browser port of the render-to-SVG decoder (`web/`).

The JavaScript grammar, decode loop and SVG writer in `web/src` must reproduce the Python
code exactly: one wrong legal mask or label changes the decoded program, and nothing in
the browser would say so. This module produces their reference data from the Python code
itself, never from a second implementation:

* `codec_constants`: the codec and sequence-layout constants the JavaScript reads
  (`web/src/codec.json`), the serialisation values milestone 0 exported plus the grammar's
  field limits.
* `decode_walk`: a program decoded the way `render2svg.greedy_decode` decodes a path-order
  model. At every step it records the original position (`path_major_positions`), the
  legal mask given the decoded prefix (`autoregressive.legal_mask`, later positions still
  zero), whether that mask forces its token, and the variant-B decoder-step inputs: the
  `(6, L)` ids rows `[tokens, steps, field, path, axis, role]` and the metric target, taken
  from the teacher-forced functions `position_labels` and `coordinate_roles`.
* `capture_greedy`: a real `greedy_decode` with its `step_inputs` calls recorded, checked
  call by call against the walk of the program it produced, so the walk's forced-token
  schedule and ids are those of the reference decoder, not a reading of it.
* `walk_digest`: SHA-256 of a fixed 65-byte record per step, which the JavaScript rebuilds
  byte for byte. It lets the Node tests check every corpus program at every position with
  a fixture of 64 hex characters per program.

A fixture set holds the corpus programs of chosen splits (`select_splits`) and the model's
decodes of validation icons. Only the validation split is committed (`web/test/fixtures`);
the set with every split lives under the cache (`full_fixture_dir`), since datasets stay out
of the repository (`check_fixture_destination`). Each set records its own counts
(`fixture_counts`), so the tests check either set against itself.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from mojidiff.learning.autoregressive import (
    _CAP_NAMES,
    _FILL_RULE_NAMES,
    _JOIN_NAMES,
    NONE,
    PAD,
    PATH_FIELDS,
    PATH_STRIDE,
    SEGMENT_STRIDE,
    SequenceLayout,
    legal_mask,
    unflatten_program,
)
from mojidiff.learning.render2svg import (
    _CONTROL_SLOTS,
    CACHE_ROOT,
    FIELD_TYPES,
    ROLE_CONTROL,
    ROLE_ENDPOINT,
    ROLE_NONE,
    DecodeStats,
    RenderToProgram,
    _static_tables,
    coordinate_roles,
    greedy_decode,
    path_major_positions,
    position_labels,
    program_legal_masks,
)
from mojidiff.representation.packed import PackedTensorProgram, serialize_packed_svg
from mojidiff.representation.program import SegmentType

FIXTURE_FORMAT = 1
"""Bump when the layout of `codec.json`, the walk record or the digest record changes."""

IDS_ROWS = ("tokens", "steps", "field", "path", "axis", "role")
"""The variant-B decoder step's id rows, in order (reports/browser/m0/export_variants.py)."""

DIGEST_HEX = 32
"""Fixture digests keep the first 128 bits of SHA-256: ample to detect any difference,
half the size of the full hex in a fixture of thousands of programs."""

FIXTURE_SPLITS = ("primary/validation", "primary/test", "primary/train")
"""The corpus splits a fixture set can hold, in the order their programs are written."""

COMMITTED_SPLITS = ("primary/validation",)
"""The corpus splits whose programs may be written inside the repository: the validation
programs, the icons the model's decodes come from. Train and test stay in the cache."""

WEB_FIXTURE_CACHE = CACHE_ROOT / "web-fixtures"


def full_fixture_dir(run_id: str) -> Path:
    """Where the fixture set with every corpus split lives for a checkpoint's run."""

    return WEB_FIXTURE_CACHE / run_id / "full"


def select_splits(names: Sequence[str]) -> tuple[str, ...]:
    """The named splits in `FIXTURE_SPLITS` order, each once; "all" names every split."""

    chosen: set[str] = set()
    for name in names:
        if name == "all":
            chosen.update(FIXTURE_SPLITS)
        elif name in FIXTURE_SPLITS:
            chosen.add(name)
        else:
            raise ValueError(f"unknown split {name!r}: choose from {FIXTURE_SPLITS} or 'all'")
    if not chosen:
        raise ValueError("no split selected")
    return tuple(split for split in FIXTURE_SPLITS if split in chosen)


def check_fixture_destination(out: Path, splits: Sequence[str], repository: Path) -> None:
    """Refuse to write corpus programs of splits beyond `COMMITTED_SPLITS` into `repository`."""

    extra = [split for split in splits if split not in COMMITTED_SPLITS]
    if extra and out.resolve().is_relative_to(repository.resolve()):
        raise ValueError(
            f"{', '.join(extra)} programs stay out of the repository, not in {out}: "
            f"write them under {WEB_FIXTURE_CACHE}"
        )


def fixture_counts(
    programs: Sequence[Mapping[str, Any]], walks: int, schedules: int
) -> dict[str, Any]:
    """What a fixture set records about itself: programs, programs with an SVG, programs per
    source and corpus programs per split, detailed walks and recorded call schedules."""

    sources = Counter(str(program["source"]) for program in programs)
    corpus = Counter(str(p["split"]) for p in programs if p["source"] == "corpus")
    return {
        "programs": len(programs),
        "svg": sum(program["svg_sha256"] is not None for program in programs),
        "sources": dict(sorted(sources.items())),
        "corpus_splits": {split: corpus[split] for split in FIXTURE_SPLITS if split in corpus},
        "walks": walks,
        "schedules": schedules,
    }


def codec_constants(layout: SequenceLayout) -> dict[str, Any]:
    """Everything the JavaScript codec and grammar read, as JSON-ready values."""

    codec = layout.codec
    return {
        "format": FIXTURE_FORMAT,
        # Serialisation constants, as milestone 0's codec/export.py wrote them.
        "max_paths": codec.max_paths,
        "max_segments": codec.max_segments,
        "total_segment_slots": layout.total_segment_slots,
        "coordinate_bins": codec.coordinate_bins,
        "control_coordinate_bins": codec.control_coordinate_bins,
        "control_coordinate_min": codec.control_coordinate_min,
        "control_coordinate_max": codec.control_coordinate_max,
        "palette": list(codec.palette),
        "stroke_widths": list(codec.stroke_widths),
        "dash_patterns": [list(pattern) for pattern in codec.dash_patterns],
        "miter_limits": list(codec.miter_limits),
        "opacities": list(codec.opacities),
        "length": layout.length,
        "vocabulary": layout.vocabulary,
        # Grammar and decoder-step constants.
        "effective_control_coordinate_bins": codec.effective_control_coordinate_bins,
        "path_positions": layout.path_positions,
        "path_stride": PATH_STRIDE,
        "segment_stride": SEGMENT_STRIDE,
        "path_fields": list(PATH_FIELDS),
        "field_types": FIELD_TYPES,
        "fill_rules": list(_FILL_RULE_NAMES),
        "linecaps": list(_CAP_NAMES),
        "linejoins": list(_JOIN_NAMES),
        "control_slots": list(_CONTROL_SLOTS),
        "tokens": {"pad": PAD, "none": NONE},
        "segment_types": {kind.name.lower(): int(kind) for kind in SegmentType},
        "roles": {"none": ROLE_NONE, "endpoint": ROLE_ENDPOINT, "control": ROLE_CONTROL},
        "ids_rows": list(IDS_ROWS),
    }


@dataclass(frozen=True)
class Walk:
    """One program decoded in path-major order, step by step (see the module docstring)."""

    order: np.ndarray  # (L,) original position decoded at each step
    masks: np.ndarray  # (L, V) bool legal mask at each step, given the decoded prefix
    forced: np.ndarray  # (L,) bool: the mask admits exactly one token
    ids: np.ndarray  # (6, L) int64 variant-B rows: tokens, steps, field, path, axis, role
    target: np.ndarray  # (L,) int64 metric target of each step's own position, -1 for none
    tokens: np.ndarray  # (L,) int64 token decoded at each step

    @property
    def model_calls(self) -> int:
        """Calls a batch-1 decode makes: one per step whose mask leaves a choice."""

        return int((~self.forced).sum())


def decode_walk(tokens: Tensor, layout: SequenceLayout) -> Walk:
    """Walk a complete program the way `greedy_decode` decodes a path-order model.

    Raises ValueError when a token is outside its legal mask, i.e. the program is not one
    the grammar can produce, or when the prefix masks disagree with `program_legal_masks`.
    """

    values = tokens.to(torch.long).cpu()
    length, vocabulary = layout.length, layout.vocabulary
    prefix = torch.zeros(length, dtype=torch.long)
    walker = path_major_positions(prefix, layout)
    order = np.zeros(length, dtype=np.int64)
    masks = np.zeros((length, vocabulary), dtype=bool)
    for step in range(length):
        position = next(walker)
        mask = legal_mask(position, prefix, layout).numpy()
        token = int(values[position])
        if not mask[token]:
            raise ValueError(f"token {token} at position {position} is not legal")
        order[step] = position
        masks[step] = mask
        prefix[position] = token
    if sorted(order.tolist()) != list(range(length)):
        raise ValueError("the path-major order is not a permutation of the positions")
    if not np.array_equal(program_legal_masks(values, layout)[order], masks):
        raise ValueError("prefix masks differ from program_legal_masks")

    role, axis = (t[0].numpy() for t in coordinate_roles(values[None], _static_tables(layout)))
    field, path = (t[0].numpy() for t in position_labels(values[None], layout))
    sequence = values.numpy()[order]
    role_at, axis_at = role[order], axis[order]

    def shifted(row: np.ndarray) -> np.ndarray:
        return np.concatenate((np.zeros(1, dtype=np.int64), row[:-1].astype(np.int64)))

    ids = np.stack(
        (
            shifted(sequence),
            np.arange(length, dtype=np.int64),
            field[order].astype(np.int64),
            path[order].astype(np.int64),
            shifted(axis_at),
            shifted(role_at),
        )
    )
    target = np.where(role_at > 0, (role_at - 1) * 2 + axis_at, -1).astype(np.int64)
    return Walk(
        order=order,
        masks=masks,
        forced=masks.sum(axis=1) == 1,
        ids=ids,
        target=target,
        tokens=sequence.astype(np.int64),
    )


def walk_record_dtype(vocabulary: int) -> np.dtype[Any]:
    """The digest's fixed little-endian record per step (65 bytes at vocabulary 418)."""

    return np.dtype(
        [
            ("position", "<u2"),
            ("mask", "u1", ((vocabulary + 7) // 8,)),
            ("forced", "u1"),
            ("token_in", "<u2"),
            ("field", "u1"),
            ("path", "u1"),
            ("axis_in", "u1"),
            ("role_in", "u1"),
            ("target", "i1"),
            ("token", "<u2"),
        ]
    )


def walk_digest(walk: Walk) -> str:
    """SHA-256 (first `DIGEST_HEX` hex digits) of the per-step records, which
    `web/test/helpers.mjs` rebuilds byte for byte.

    Each step: position (u16), the legal mask packed least significant bit first (u8 x
    ceil(V/8)), forced (u8), the ids rows token_in (u16), field, path, axis_in, role_in
    (u8 each; steps is the step index and is left out), target (i8), token (u16).
    """

    length, vocabulary = walk.masks.shape
    records = np.zeros(length, dtype=walk_record_dtype(vocabulary))
    records["position"] = walk.order
    records["mask"] = np.packbits(walk.masks, axis=1, bitorder="little")
    records["forced"] = walk.forced
    records["token_in"] = walk.ids[0]
    records["field"] = walk.ids[2]
    records["path"] = walk.ids[3]
    records["axis_in"] = walk.ids[4]
    records["role_in"] = walk.ids[5]
    records["target"] = walk.target
    records["token"] = walk.tokens
    return hashlib.sha256(records.tobytes()).hexdigest()[:DIGEST_HEX]


def mask_ranges(mask: np.ndarray) -> list[list[int]]:
    """A boolean mask as half-open `[start, stop)` runs of legal tokens."""

    padded = np.concatenate(([False], mask.astype(bool), [False]))
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    return [[int(edges[i]), int(edges[i + 1])] for i in range(0, len(edges), 2)]


def walk_record(walk: Walk, table: dict[bytes, int]) -> dict[str, Any]:
    """A walk as compact JSON: masks become indices into a shared table of distinct masks.

    `table` maps a mask's packed bytes to its index and is extended in place; write its
    masks with `mask_table`.
    """

    indices = []
    for mask in walk.masks:
        key = np.packbits(mask).tobytes()
        indices.append(table.setdefault(key, len(table)))
    return {
        "order": walk.order.tolist(),
        "mask": indices,
        "forced": walk.forced.astype(int).tolist(),
        "token_in": walk.ids[0].tolist(),
        "field": walk.ids[2].tolist(),
        "path": walk.ids[3].tolist(),
        "axis_in": walk.ids[4].tolist(),
        "role_in": walk.ids[5].tolist(),
        "target": walk.target.tolist(),
    }


def mask_table(table: dict[bytes, int], vocabulary: int) -> list[list[list[int]]]:
    """The distinct masks of `walk_record`'s table, in index order, as range lists."""

    ordered = sorted(table.items(), key=lambda item: item[1])
    return [
        mask_ranges(np.unpackbits(np.frombuffer(key, dtype=np.uint8))[:vocabulary].astype(bool))
        for key, _ in ordered
    ]


def program_svg(
    tokens: Tensor, template: PackedTensorProgram, layout: SequenceLayout
) -> str | None:
    """The SVG `serialize_packed_svg` writes for a program, or None if it rejects it."""

    program = unflatten_program(tokens.to(torch.long).cpu(), template, layout)
    try:
        return serialize_packed_svg(program, layout.codec, layout.total_segment_slots).decode()
    except ValueError:
        return None


@dataclass(frozen=True)
class CapturedCall:
    """One decoder call of `greedy_decode`: steps `start..end-1`, rows as variant B feeds them."""

    start: int
    end: int
    ids: np.ndarray  # (6, end - start)
    target: int


def capture_greedy(
    model: RenderToProgram,
    image: Tensor,
    *,
    temperature: float = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, list[CapturedCall], int]:
    """`greedy_decode` of one uint8 HWC render with each decoder call's inputs recorded.

    Returns the program, the calls, and `DecodeStats.model_calls`. The ids are read from
    what `step_inputs` returned (tokens, field, path, metric target) plus the input
    token's role and axis from `coordinate_roles` on the prefix decoded at call time,
    which is what the variant-B graph turns into the metric input features.
    """

    if model.config.order != "path" or not model.config.metric:
        raise ValueError("the browser step is defined for metric, path-order models")
    calls: list[CapturedCall] = []
    original: Callable[..., tuple[Tensor, dict[str, Tensor]]] = model.step_inputs

    def spy(
        tokens: Tensor, order: Tensor | None, start: int, end: int
    ) -> tuple[Tensor, dict[str, Tensor]]:
        inputs, extras = original(tokens, order, start, end)
        if order is None:
            raise ValueError("a path-order model always passes its order")
        role, axis = coordinate_roles(tokens, model.tables())
        role_seq = role.gather(1, order)[0].cpu()
        axis_seq = axis.gather(1, order)[0].cpu()
        steps = torch.arange(start, end)
        previous = (steps - 1).clamp_min(0)
        first = (steps > 0).to(torch.long)
        ids = torch.stack(
            (
                inputs[0].cpu(),
                steps,
                extras["field"][0].cpu(),
                extras["path"][0].cpu(),
                axis_seq[previous] * first,
                role_seq[previous] * first,
            )
        )
        target = int(extras["metric_target"][0, -1])
        calls.append(CapturedCall(start, end, ids.numpy().astype(np.int64), target))
        return inputs, extras

    stats = DecodeStats()
    model.__dict__["step_inputs"] = spy
    try:
        program = greedy_decode(
            model, image[None], temperature=temperature, generator=generator, stats=stats
        )[0]
    finally:
        del model.__dict__["step_inputs"]
    return program, calls, stats.model_calls


def check_calls_against_walk(walk: Walk, calls: list[CapturedCall], model_calls: int) -> None:
    """Raise unless the recorded calls are exactly the walk's schedule and ids.

    Batch 1: a call at every step whose mask leaves a choice, feeding every step since the
    previous call (the forced ones queued in between), ids equal to the walk's rows.
    """

    free = np.flatnonzero(~walk.forced).tolist()
    if model_calls != len(calls) or len(calls) != len(free):
        raise ValueError(f"{len(calls)} calls recorded, {len(free)} free steps, {model_calls}")
    start = 0
    for call, step in zip(calls, free, strict=True):
        if (call.start, call.end) != (start, step + 1):
            raise ValueError(f"call {(call.start, call.end)} expected {(start, step + 1)}")
        if not np.array_equal(call.ids, walk.ids[:, call.start : call.end]):
            raise ValueError(f"ids differ for the call ending at step {step}")
        if call.target != int(walk.target[step]):
            raise ValueError(f"target differs at step {step}")
        start = step + 1


_WORKER: dict[str, Any] = {}


def _init_worker(template: PackedTensorProgram, layout: SequenceLayout) -> None:
    torch.set_num_threads(1)
    _WORKER["template"], _WORKER["layout"] = template, layout


def _entry_in_worker(item: tuple[dict[str, Any], np.ndarray]) -> dict[str, Any]:
    meta, tokens = item
    return program_entry(meta, torch.from_numpy(tokens), _WORKER["template"], _WORKER["layout"])


def program_entry(
    meta: dict[str, Any], tokens: Tensor, template: PackedTensorProgram, layout: SequenceLayout
) -> dict[str, Any]:
    """One program's fixture: its tokens, the digest of its walk, and its SVG's hash.

    `svg_sha256` is the SHA-256 (first `DIGEST_HEX` hex digits) of the UTF-8 SVG
    `serialize_packed_svg` writes, None when it rejects the program; `model_calls` is the
    batch-1 call count of its decode.
    """

    walk = decode_walk(tokens, layout)
    svg = program_svg(tokens, template, layout)
    entry = dict(meta)
    entry.update(
        tokens=tokens.tolist(),
        walk_sha256=walk_digest(walk),
        model_calls=walk.model_calls,
        svg_sha256=None if svg is None else hashlib.sha256(svg.encode()).hexdigest()[:DIGEST_HEX],
        svg_bytes=None if svg is None else len(svg.encode()),
    )
    return entry


def program_entries(
    items: list[tuple[dict[str, Any], np.ndarray]],
    template: PackedTensorProgram,
    layout: SequenceLayout,
    workers: int = 1,
) -> list[dict[str, Any]]:
    """`program_entry` for every (metadata, tokens) item, in order, across processes."""

    if workers <= 1:
        return [program_entry(meta, torch.from_numpy(t), template, layout) for meta, t in items]
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
        max_workers=workers, initializer=_init_worker, initargs=(template, layout)
    ) as pool:
        return list(pool.map(_entry_in_worker, items, chunksize=16))


def token_planes(tokens: np.ndarray) -> bytes:
    """(N, L) tokens below 65,536 as two byte planes, all low bytes then all high bytes.

    Gzip shrinks the planes about a fifth more than interleaved 16-bit values, because
    the high plane is almost all zeros.
    """

    values = np.asarray(tokens)
    if values.min(initial=0) < 0 or values.max(initial=0) > 0xFFFF:
        raise ValueError("tokens must fit in 16 bits")
    values = values.astype(np.uint16)
    return (values & 0xFF).astype(np.uint8).tobytes() + (values >> 8).astype(np.uint8).tobytes()


def write_gz(path: Path, data: bytes) -> int:
    """Deterministic gzip (no timestamp, no file name); returns bytes written."""

    packed = gzip.compress(data, compresslevel=9, mtime=0)
    path.write_bytes(packed)
    return len(packed)


def write_json_gz(path: Path, value: Any) -> int:
    """Deterministic gzip (no timestamp, no file name) of compact JSON; returns bytes written."""

    return write_gz(path, json.dumps(value, separators=(",", ":")).encode())
