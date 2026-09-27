"""Single-icon decoding with a static key/value cache and CUDA graphs.

`render2svg.greedy_decode` is the reference: exact, batched, and dominated at batch 1
by overhead rather than arithmetic - per call it copies the whole decoded sequence to
the GPU, launches about a hundred small kernels, and waits. For the demo, one icon at a
time, this does the same decode with:

* a key/value cache preallocated to the full sequence, written in place;
* the decoder step captured as a CUDA graph for a few fixed chunk sizes, so one call is
  one graph replay; a chunk shorter than its graph is padded with dummy steps that
  write to one scratch cache slot past the end of the sequence, which no real step
  can attend;
* the grammar applied on the CPU to one row of logits copied back per free step.

The per-step extras (field, path, coordinate role) are computed on the CPU from the
decoded prefix with the same rules as the batched code; a test checks both paths agree.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import cast

import torch
import torch.nn.functional as F
from torch import Tensor

from mojidiff.learning.autoregressive import PATH_STRIDE, SEGMENT_STRIDE, legal_mask
from mojidiff.learning.render2svg import (
    _CONTROL_SLOTS,
    ROLE_CONTROL,
    ROLE_ENDPOINT,
    ROLE_NONE,
    DecodeStats,
    RenderToProgram,
    _DecoderBlock,
    _static_tables,
    path_major_positions,
)

CHUNK_SIZES = (1, 2, 4, 8, 16, 32)


class _Buffers:
    def __init__(self, size: int, features: int, vocabulary: int, device: torch.device) -> None:
        self.tokens = torch.zeros((1, size), dtype=torch.long, device=device)
        self.steps = torch.zeros((1, size), dtype=torch.long, device=device)
        self.slots = torch.zeros((1, size), dtype=torch.long, device=device)
        self.field = torch.zeros((1, size), dtype=torch.long, device=device)
        self.path = torch.zeros((1, size), dtype=torch.long, device=device)
        self.value = torch.zeros((1, size), dtype=torch.float32, device=device)
        self.axis = torch.zeros((1, size), dtype=torch.long, device=device)
        self.role = torch.zeros((1, size), dtype=torch.long, device=device)
        self.target = torch.full((1, size), -1, dtype=torch.long, device=device)
        self.logits = torch.zeros((1, vocabulary), dtype=torch.float32, device=device)
        self.last = torch.zeros((), dtype=torch.long, device=device)


class GraphDecoder:
    """Greedy single-icon decoding for one trained model on one CUDA device."""

    def __init__(self, model: RenderToProgram, *, dtype: torch.dtype = torch.bfloat16) -> None:
        self.model = model.eval()
        self.layout = model.layout
        self.config = model.config
        self.device = next(model.parameters()).device
        if self.device.type != "cuda":
            raise ValueError("graph decoding needs a CUDA device")
        self.dtype = dtype
        width = self.config.d_model
        heads = self.config.heads
        length = self.layout.length
        grid = (self.config.image_size // 8) ** 2
        self.heads = heads
        self.head_dim = width // heads
        shape = (1, heads, length + 1, self.head_dim)  # the last slot is scratch
        self.keys = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in model.decoder]
        self.values = [torch.zeros(shape, dtype=dtype, device=self.device) for _ in model.decoder]
        memory_shape = (1, heads, grid, self.head_dim)
        self.memory = [
            (
                torch.zeros(memory_shape, dtype=dtype, device=self.device),
                torch.zeros(memory_shape, dtype=dtype, device=self.device),
            )
            for _ in model.decoder
        ]
        self.key_positions = torch.arange(length + 1, device=self.device)
        tables = _static_tables(self.layout)
        self.slot = tables[0].tolist()
        self.kind_position = tables[1].tolist()
        self.start_axis = tables[2].tolist()
        self.buffers: dict[int, _Buffers] = {}
        self.graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._capture()

    # ------------------------------------------------------------------ graph body

    def _body(self, buffers: _Buffers) -> None:
        model = self.model
        config = self.config
        with torch.autocast("cuda", dtype=self.dtype):
            steps = buffers.steps
            hidden = model.token_embedding(buffers.tokens) + model.position_embedding(steps)
            if config.order == "path":
                hidden = hidden + model.field_embedding(buffers.field)
                hidden = hidden + model.path_embedding(buffers.path)
            if config.metric:
                described = model._describe(buffers.value, buffers.axis, buffers.role)
                keep = ((buffers.role > 0) & (buffers.tokens > 0))[..., None]
                hidden = hidden + model.coordinate_projection(described * keep)
            mask = (self.key_positions[None, :] <= steps[0][:, None])[None, None]
            for index, block in enumerate(model.decoder):
                block = cast(_DecoderBlock, block)
                normed = block.norm_self(hidden)
                key, value = block.self_attention.keys_values(normed)
                self.keys[index].index_copy_(2, buffers.slots[0], key.to(self.dtype))
                self.values[index].index_copy_(2, buffers.slots[0], value.to(self.dtype))
                query = block.self_attention.split(block.self_attention.query(normed))
                attended = F.scaled_dot_product_attention(
                    query, self.keys[index], self.values[index], attn_mask=mask
                )
                size = hidden.shape[1]
                hidden = hidden + block.self_attention.project(
                    attended.transpose(1, 2).reshape(1, size, -1)
                )
                crossed = block.cross_attention.split(
                    block.cross_attention.query(block.norm_cross(hidden))
                )
                attended = F.scaled_dot_product_attention(crossed, *self.memory[index])
                hidden = hidden + block.cross_attention.project(
                    attended.transpose(1, 2).reshape(1, size, -1)
                )
                hidden = hidden + block.feedforward(block.norm_feedforward(hidden))
            last = hidden.index_select(1, buffers.last[None])
            normed = model.norm(last)
            logits = model.head(normed)
            if config.metric:
                target = buffers.target.index_select(1, buffers.last[None])
                query = model.metric_query(normed)
                candidates = model.metric_value(
                    cast(Tensor, model._candidate_features).to(query.dtype)
                )
                scores = torch.einsum("bld,rvd->blrv", query, candidates) / math.sqrt(
                    query.shape[-1]
                )
                chosen = scores.gather(
                    2, target.clamp_min(0)[..., None, None].expand(-1, -1, 1, scores.shape[-1])
                )[:, :, 0]
                logits = logits + chosen * (target >= 0)[..., None].to(chosen.dtype)
            buffers.logits.copy_(logits[:, 0].float())

    def _capture(self) -> None:
        features = 0
        vocabulary = self.layout.vocabulary
        stream = torch.cuda.Stream(self.device)  # type: ignore[no-untyped-call]
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.no_grad(), torch.cuda.stream(stream):
            for size in CHUNK_SIZES:
                buffers = _Buffers(size, features, vocabulary, self.device)
                buffers.steps.copy_(torch.arange(size, device=self.device)[None])
                buffers.slots.copy_(torch.arange(size, device=self.device)[None])
                for _ in range(3):
                    self._body(buffers)
                self.buffers[size] = buffers
        torch.cuda.current_stream(self.device).wait_stream(stream)
        with torch.no_grad():
            for size in CHUNK_SIZES:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._body(self.buffers[size])
                self.graphs[size] = graph

    # ------------------------------------------------------------------ CPU-side rules

    def _role(self, position: int, decoded: list[int]) -> tuple[int, int]:
        """(role, axis) of an original position, from earlier tokens only."""

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

    def _labels(self, position: int, decoded: list[int]) -> tuple[int, int]:
        """(field, path) of an original position, as `position_labels` defines them."""

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

    # ------------------------------------------------------------------ decoding

    @torch.no_grad()
    def decode(self, image: Tensor, stats: DecodeStats | None = None) -> Tensor:
        """Greedy decode of one uint8 HWC render; the program in original positions."""

        model = self.model
        layout = self.layout
        with torch.autocast("cuda", dtype=self.dtype):
            memory = model.encode(image[None].to(self.device))
        for (key, value), (static_key, static_value) in zip(memory, self.memory, strict=True):
            static_key.copy_(key)
            static_value.copy_(value)
        decoded_tensor = torch.zeros(layout.length, dtype=torch.long)
        decoded = [0] * layout.length
        order: list[int] = []
        walker: Iterator[int] = (
            path_major_positions(decoded_tensor, layout)
            if self.config.order == "path"
            else iter(range(layout.length))
        )
        fed = 0
        calls = 0
        for step in range(layout.length):
            position = next(walker)
            order.append(position)
            mask = legal_mask(position, decoded_tensor, layout)
            if int(mask.sum()) == 1:
                token = int(mask.to(torch.long).argmax())
            else:
                logits = self._run(order, decoded, fed, step)
                calls += logits[1]
                fed = step + 1
                scores = logits[0].masked_fill(~mask, float("-inf"))
                token = int(scores.argmax())
            decoded[position] = token
            decoded_tensor[position] = token
        if stats is not None:
            stats.model_calls += calls
            stats.positions += layout.length
        return decoded_tensor[None]

    def _run(self, order: list[int], decoded: list[int], fed: int, step: int) -> tuple[Tensor, int]:
        """Feed steps fed..step through the graphs; the logits predicting `step`."""

        calls = 0
        start = fed
        final: _Buffers | None = None
        while start <= step:
            remaining = step + 1 - start
            size = next((s for s in CHUNK_SIZES if s >= remaining), CHUNK_SIZES[-1])
            real = min(size, remaining)
            buffers = self.buffers[size]
            tokens = [0] * size
            steps = list(range(start, start + size))
            field = [0] * size
            path = [0] * size
            value = [0.0] * size
            axis = [0] * size
            role = [0] * size
            target = [-1] * size
            for index in range(real):
                current = start + index
                previous = order[current - 1] if current > 0 else None
                tokens[index] = decoded[previous] if previous is not None else 0
                here = order[current]
                if self.config.order == "path":
                    field[index], path[index] = self._labels(here, decoded)
                if self.config.metric:
                    if previous is not None:
                        role_in, axis_in = self._role(previous, decoded)
                        token = decoded[previous]
                        index_value = max(token - 1, 0) * 0.25
                        value[index] = index_value - 8.0 if role_in == ROLE_CONTROL else index_value
                        role[index], axis[index] = role_in, axis_in
                    role_out, axis_out = self._role(here, decoded)
                    if role_out != ROLE_NONE:
                        target[index] = (role_out - 1) * 2 + axis_out
            scratch = self.layout.length
            slots = [s if index < real else scratch for index, s in enumerate(steps)]
            steps = [s if index < real else step for index, s in enumerate(steps)]
            buffers.tokens.copy_(torch.tensor([tokens]), non_blocking=True)
            buffers.steps.copy_(torch.tensor([steps]), non_blocking=True)
            buffers.slots.copy_(torch.tensor([slots]), non_blocking=True)
            buffers.field.copy_(torch.tensor([field]), non_blocking=True)
            buffers.path.copy_(torch.tensor([path]), non_blocking=True)
            buffers.value.copy_(torch.tensor([value]), non_blocking=True)
            buffers.axis.copy_(torch.tensor([axis]), non_blocking=True)
            buffers.role.copy_(torch.tensor([role]), non_blocking=True)
            buffers.target.copy_(torch.tensor([target]), non_blocking=True)
            buffers.last.fill_(real - 1)
            self.graphs[size].replay()
            calls += 1
            start += real
            final = buffers
        assert final is not None
        return final.logits[0].cpu(), calls
