"""Greedy decoding through the exported ONNX graphs with ONNX Runtime: the browser's loop.

This is the Python reference for the JavaScript decoder. It runs `encoder.onnx` once per
icon and `decoder_step.onnx` once per free position, under the existing Python grammar:
positions in path-major order (`render2svg.path_major_positions`), each masked by
`autoregressive.legal_mask`; a position with one legal token is not predicted but queued
and fed with the next call, exactly as `render2svg.greedy_decode` and
`fast_decode.GraphDecoder` skip forced tokens. Programs come back in original positions.

The per-step ids (field, path, coordinate role and axis, metric target) are computed on
the CPU from the decoded prefix by `StepRules`, the rules `GraphDecoder` uses; a test
checks them against the batched tensor versions (`render2svg.coordinate_roles` and
`render2svg.position_labels`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor

from mojidiff.learning.autoregressive import PATH_STRIDE, SEGMENT_STRIDE, SequenceLayout, legal_mask
from mojidiff.learning.onnx_export import DECODER_FILE, ENCODER_FILE, ID_ROWS, MANIFEST_FILE
from mojidiff.learning.render2svg import (
    _CONTROL_SLOTS,
    ROLE_CONTROL,
    ROLE_ENDPOINT,
    ROLE_NONE,
    DecodeStats,
    _static_tables,
    path_major_positions,
)


class StepRules:
    """Per-position decoder inputs from earlier tokens only, as plain Python on lists."""

    def __init__(self, layout: SequenceLayout) -> None:
        self.layout = layout
        slot, kind_position, start_axis = _static_tables(layout)
        self.slot: list[int] = slot.tolist()
        self.kind_position: list[int] = kind_position.tolist()
        self.start_axis: list[int] = start_axis.tolist()

    def role(self, position: int, decoded: list[int]) -> tuple[int, int]:
        """(role, axis) of an original position, as `render2svg.coordinate_roles`."""

        axis = self.start_axis[position]
        if axis >= 0:
            return ROLE_ENDPOINT, axis
        slot = self.slot[position]
        if slot < 0:
            return ROLE_NONE, 0
        kind = min(max(decoded[self.kind_position[position]], 0), 4)
        controls = _CONTROL_SLOTS[kind]
        used = controls + 2 if 1 <= kind <= 3 else 0
        if slot >= used:
            return ROLE_NONE, slot % 2
        return (ROLE_CONTROL if slot < controls else ROLE_ENDPOINT), slot % 2

    def labels(self, position: int, decoded: list[int]) -> tuple[int, int]:
        """(field, path) of an original position, as `render2svg.position_labels`."""

        layout = self.layout
        if position < layout.path_positions:
            return position % PATH_STRIDE, position // PATH_STRIDE
        offset = position - layout.path_positions
        slot = offset // SEGMENT_STRIDE
        field = PATH_STRIDE + offset % SEGMENT_STRIDE
        end = 0
        for path in range(layout.codec.max_paths):
            end += decoded[path * PATH_STRIDE]
            if slot < end:
                return field, path
        return field, layout.codec.max_paths

    def target(self, position: int, decoded: list[int]) -> int:
        """Metric target of a position: (role - 1) * 2 + axis, or -1 for no coordinate."""

        role, axis = self.role(position, decoded)
        return (role - 1) * 2 + axis if role != ROLE_NONE else -1

    def ids(self, order: list[int], decoded: list[int], start: int, end: int) -> np.ndarray:
        """(6, end - start) float32 decoder ids for sequence steps start..end-1.

        Step j reads the token of step j - 1 (0 at step 0) with that position's role and
        axis, and is told the field and path of the position it predicts, order[j].
        """

        ids = np.zeros((len(ID_ROWS), end - start), dtype=np.float32)
        for column, step in enumerate(range(start, end)):
            ids[1, column] = step
            ids[2, column], ids[3, column] = self.labels(order[step], decoded)
            if step > 0:
                previous = order[step - 1]
                ids[0, column] = decoded[previous]
                role, axis = self.role(previous, decoded)
                ids[4, column] = axis
                ids[5, column] = role
        return ids


class OnnxDecoder:
    """Greedy decoding of one icon at a time through an exported model directory."""

    def __init__(
        self,
        directory: Path,
        layout: SequenceLayout,
        *,
        threads: int | None = None,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ) -> None:
        import onnxruntime as ort

        self.manifest: dict[str, Any] = json.loads((directory / MANIFEST_FILE).read_text())
        constants = self.manifest["constants"]
        if (constants["vocabulary"], constants["length"]) != (layout.vocabulary, layout.length):
            raise ValueError("the exported model was built for another sequence layout")
        self.layout = layout
        self.rules = StepRules(layout)
        self.image_size = int(constants["image_size"])
        self.cache_prefix = (int(constants["decoder_layers"]), 1, int(constants["heads"]))
        self.head_dim = int(constants["head_dim"])
        options = ort.SessionOptions()
        if threads is not None:
            options.intra_op_num_threads = threads
        self.encoder = ort.InferenceSession(
            str(directory / ENCODER_FILE), options, providers=list(providers)
        )
        self.decoder = ort.InferenceSession(
            str(directory / DECODER_FILE), options, providers=list(providers)
        )

    def encode(self, image: np.ndarray, rows: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Memory keys and values (layers, rows, heads, cells, head_dim) of a uint8 HWC render."""

        if image.shape != (self.image_size, self.image_size, 3):
            raise ValueError(f"expected a {self.image_size}x{self.image_size} RGB render")
        pixels = image[None].astype(np.float32)
        keys, values = self.encoder.run(
            None, {"pixels": pixels, "rows": np.zeros(rows, dtype=np.int64)}
        )
        return keys, values

    def step(
        self,
        ids: np.ndarray,
        target: np.ndarray,
        past_keys: np.ndarray,
        past_values: np.ndarray,
        memory_keys: np.ndarray,
        memory_values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One decoder-step call: logits of the last step and the extended cache."""

        logits, keys, values = self.decoder.run(
            None,
            {
                "ids": ids,
                "target": target,
                "past_keys": past_keys,
                "past_values": past_values,
                "memory_keys": memory_keys,
                "memory_values": memory_values,
            },
        )
        return logits, keys, values

    def empty_cache(self) -> np.ndarray:
        return np.zeros((*self.cache_prefix, 0, self.head_dim), dtype=np.float32)

    def decode(self, image: np.ndarray, stats: DecodeStats | None = None) -> Tensor:
        """Greedy program (length,) in original positions for one uint8 HWC render."""

        layout = self.layout
        memory_keys, memory_values = self.encode(image)
        past_keys, past_values = self.empty_cache(), self.empty_cache()
        decoded = [0] * layout.length
        tensor = torch.zeros(layout.length, dtype=torch.long)
        walker = path_major_positions(tensor, layout)
        order: list[int] = []
        fed = 0
        calls = 0
        for step in range(layout.length):
            position = next(walker)
            order.append(position)
            mask = legal_mask(position, tensor, layout).numpy()
            count = int(mask.sum())
            if count == 0:
                raise ValueError(f"no legal token at step {step}")
            if count == 1:
                token = int(mask.argmax())
            else:
                ids = self.rules.ids(order, decoded, fed, step + 1)[:, None]
                target = np.array([self.rules.target(position, decoded)], dtype=np.float32)
                logits, past_keys, past_values = self.step(
                    ids, target, past_keys, past_values, memory_keys, memory_values
                )
                calls += 1
                fed = step + 1
                token = int(np.where(mask, logits[0], -np.inf).argmax())
            decoded[position] = token
            tensor[position] = token
        if stats is not None:
            stats.model_calls += calls
            stats.positions += layout.length
        return tensor
