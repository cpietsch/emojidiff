"""Gate M, experiment two: a pretrained language model writes OpenMoji as SVG text.

The model is a Qwen3.5 base model, a hybrid of linear and full attention, fine-tuned
with LoRA to continue a caption comment into the icon's SVG in the project's compact
canonical form - the fixed 72-unit box, `<path d fill stroke ...>` elements only,
coordinates on the quarter-unit lattice. The text is what the pretrained prior already
reads, digits and path commands included, so shape knowledge transfers; nothing is
added to the vocabulary. Every output is parsed by the typed codec, so validity and
rendering are measured exactly as everywhere else in this project.

Two engineering facts shape this module. Qwen3.5's vocabulary is 248,320 tokens, so the
logits of a 2,000-token icon are two gigabytes in float32 and do not fit beside the
model on 16 GB; the loss is therefore computed in chunks under activation
checkpointing, which holds one chunk's logits at a time. And the linear-attention
layers need the flash-linear-attention kernels; the reference path ran out of memory
at 1,024 tokens where the kernels train 2B parameters at 0.17 s a step.
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from mojidiff.representation.packed import PackedTensorProgram, serialize_packed_svg
from mojidiff.representation.program import CodecConfig

SVG_OPEN = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">'
SVG_CLOSE = "</svg>"


def compact_svg(program: PackedTensorProgram, codec: CodecConfig, slots: int) -> str:
    """The project's canonical serialization, with the tokens the model need not spend.

    `17.0` becomes `17`, runs of whitespace one space, and the self-closing tag loses
    its space. It is still standard SVG that the project's normalizer reads back.
    """

    return compact_text(serialize_packed_svg(program, codec, slots).decode())


def compact_text(svg: str) -> str:
    svg = re.sub(r"(\d)\.0\b", r"\1", svg)
    svg = re.sub(r"\s+", " ", svg)
    return svg.replace(" />", "/>").replace("> <", "><").strip()


def caption_comment(annotation: str) -> str:
    return f"<!-- {annotation} -->\n"


def training_text(annotation: str, svg: str) -> str:
    return caption_comment(annotation) + svg


def control_prefix(annotation: str) -> str:
    """What the unfine-tuned model is asked to continue: the caption and the open tag."""

    return caption_comment(annotation) + SVG_OPEN


def extract_svg(text: str) -> str | None:
    """The first complete `<svg ...>...</svg>` in a generation, else None."""

    start = text.find("<svg")
    if start < 0:
        return None
    end = text.find(SVG_CLOSE, start)
    if end < 0:
        return None
    return text[start : end + len(SVG_CLOSE)]


# ------------------------------------------------------------------------- model


@dataclass
class Prior:
    model: Any
    tokenizer: Any
    source: str
    revision: str
    parameters: int

    @classmethod
    def load(
        cls, source: str, revision: str, *, device: str = "cuda", lora: dict[str, Any] | None = None
    ) -> Prior:
        import transformers

        config = transformers.AutoConfig.from_pretrained(source, revision=revision)
        text_config = config.get_text_config() if hasattr(config, "get_text_config") else config
        tokenizer = transformers.AutoTokenizer.from_pretrained(source, revision=revision)
        cls_model: Any = transformers.AutoModelForCausalLM
        if getattr(text_config, "model_type", "") in {"qwen3_5", "qwen3_5_text"}:
            cls_model = transformers.Qwen3_5ForCausalLM
        model: Any = cls_model.from_pretrained(
            source,
            revision=revision,
            config=text_config,
            dtype=torch.bfloat16,
            device_map={"": device} if device == "cuda" else None,
        )
        if lora is not None:
            from peft import LoraConfig, get_peft_model

            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()
            model = get_peft_model(
                model,
                LoraConfig(
                    r=int(lora["rank"]),
                    lora_alpha=int(lora["alpha"]),
                    lora_dropout=float(lora.get("dropout", 0.0)),
                    target_modules="all-linear",
                    task_type="CAUSAL_LM",
                ),
            )
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
        return cls(
            model=model,
            tokenizer=tokenizer,
            source=source,
            revision=revision,
            parameters=sum(parameter.numel() for parameter in model.parameters()),
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def decoder(self) -> Any:
        """The transformer body without its head, through any PEFT wrapper."""

        inner = self.model
        while hasattr(inner, "base_model") and not hasattr(inner, "lm_head"):
            inner = inner.base_model
        if hasattr(inner, "model") and hasattr(inner, "lm_head"):
            return inner.model
        if hasattr(inner, "base_model"):
            return inner.base_model.model.model
        raise AttributeError("could not find the decoder body")

    def head(self) -> Any:
        inner = self.model
        while not hasattr(inner, "lm_head"):
            inner = inner.base_model if hasattr(inner, "base_model") else inner.model
        return inner.lm_head

    def suffix_loss(self, input_ids: Tensor, labels: Tensor, chunk: int) -> tuple[Tensor, int]:
        """Cross-entropy over the labelled positions, summed, computed chunk by chunk."""

        hidden = self.decoder()(input_ids=input_ids).last_hidden_state
        return chunked_cross_entropy(hidden, self.head(), labels, chunk)

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        seed: int,
        greedy: bool,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> tuple[str, int, bool]:
        """Continue `prompt`; returns the continuation, its token count, and whether it ended."""

        ids = self.tokenizer(prompt, return_tensors="pt").to(next(self.model.parameters()).device)
        torch.manual_seed(seed)
        stop = _StopOnClose(self.tokenizer, ids["input_ids"].shape[1])
        output = self.model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=not greedy,
            temperature=None if greedy else temperature,
            top_p=None if greedy else top_p,
            top_k=None,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            stopping_criteria=[stop],
        )
        continuation = output[0, ids["input_ids"].shape[1] :]
        text = self.tokenizer.decode(continuation, skip_special_tokens=True)
        ended = SVG_CLOSE in text or bool((continuation == self.tokenizer.eos_token_id).any())
        return text, int(continuation.numel()), ended


class _StopOnClose:
    """Stop once the continuation contains the closing tag; cheap to check every step."""

    def __init__(self, tokenizer: Any, prompt_length: int) -> None:
        self.tokenizer = tokenizer
        self.prompt_length = prompt_length

    def __call__(self, input_ids: Tensor, scores: Tensor, **kwargs: Any) -> Tensor:
        tail = self.tokenizer.decode(input_ids[0, -8:], skip_special_tokens=True)
        done = SVG_CLOSE in tail
        return torch.full((input_ids.shape[0],), done, dtype=torch.bool, device=input_ids.device)


def chunked_cross_entropy(
    hidden: Tensor, head: Any, labels: Tensor, chunk: int
) -> tuple[Tensor, int]:
    """Sum of token cross-entropies at positions where `labels != -100`, one chunk at a time.

    `hidden` is `[1, length, width]` at positions 0..length-1 predicting labels 1..length;
    each chunk's logits exist only inside its checkpointed segment, so the peak is one
    chunk of the vocabulary rather than the whole sequence's.
    """

    shifted = labels[:, 1:]
    states = hidden[:, :-1]
    total = states.new_zeros((), dtype=torch.float32)
    count = int((shifted != -100).sum())

    def piece(segment: Tensor, targets: Tensor) -> Tensor:
        logits = head(segment).float()
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=-100,
            reduction="sum",
        )

    for start in range(0, states.shape[1], chunk):
        targets = shifted[:, start : start + chunk]
        if not bool((targets != -100).any()):
            continue
        segment = states[:, start : start + chunk]
        if torch.is_grad_enabled() and segment.requires_grad:
            total = total + checkpoint(piece, segment, targets, use_reentrant=False)
        else:
            total = total + piece(segment, targets)
    return total, count


# ------------------------------------------------------------------------ examples


@dataclass(frozen=True)
class Example:
    hexcode: str
    annotation: str
    svg: str
    input_ids: tuple[int, ...]
    prompt_tokens: int

    @property
    def length(self) -> int:
        return len(self.input_ids)


def build_example(tokenizer: Any, hexcode: str, annotation: str, svg: str) -> Example:
    prompt = caption_comment(annotation)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + svg, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"the caption's tokens change at the SVG boundary for {hexcode}")
    eos = tokenizer.eos_token_id
    return Example(
        hexcode,
        annotation,
        svg,
        tuple(full_ids) + ((eos,) if eos is not None else ()),
        len(prompt_ids),
    )


def batch_tensors(example: Example, device: str) -> tuple[Tensor, Tensor]:
    """Inputs and labels for one example; the caption's positions carry no loss."""

    ids = torch.tensor([example.input_ids], dtype=torch.long, device=device)
    labels = ids.clone()
    labels[:, : example.prompt_tokens] = -100
    return ids, labels


# ----------------------------------------------------------------------- adapters


def adapter_bytes(model: Any) -> bytes:
    """The LoRA adapter as one deterministic zip, so it can be hashed and restored."""

    import tempfile

    with tempfile.TemporaryDirectory() as directory:
        model.save_pretrained(directory)
        payload = io.BytesIO()
        with zipfile.ZipFile(payload, "w", compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(Path(directory).rglob("*")):
                if path.is_file():
                    info = zipfile.ZipInfo(str(path.relative_to(directory)))
                    info.date_time = (1980, 1, 1, 0, 0, 0)
                    archive.writestr(info, path.read_bytes())
        return payload.getvalue()


def load_adapter(model: Any, payload: bytes) -> Any:
    import tempfile

    from peft import PeftModel

    with tempfile.TemporaryDirectory() as directory:
        with zipfile.ZipFile(io.BytesIO(payload)) as archive:
            archive.extractall(directory)
        return PeftModel.from_pretrained(model, directory)


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def dumps(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()
