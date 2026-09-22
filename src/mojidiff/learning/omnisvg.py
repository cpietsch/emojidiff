"""Gate M, experiment one: OmniSVG 1.1 4B as a pretrained SVG prior.

OmniSVG is a Qwen2.5-VL-3B-Instruct whose vocabulary was extended with drawing tokens:
one token per point on a 200 x 200 grid, one per 12-bit colour, and five commands. It
was trained on two million SVGs and it draws fills only - no strokes. This module loads
its released checkpoint into the current transformers, drives its own tokenizer for the
token-to-SVG decoding, and converts what it draws into the project's codec so that the
same validator, renderer and metrics apply.

Three things about the conversion are deliberate and recorded rather than hidden:

* the drawing is scaled from OmniSVG's 200-unit box to the project's 72-unit box, and
  the quarter-unit lattice then quantises it;
* every fill colour is snapped to the nearest colour of the OpenMoji palette, because
  OmniSVG's colours are 4 bits per channel and cannot land on the palette exactly; the
  snap distance is reported per path;
* a drawing the codec cannot take - too many paths or segments, an empty result - is a
  counted failure with a reason, never a silently dropped sample.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mojidiff.representation.normalizer import NormalizationError, normalize_svg
from mojidiff.representation.packed import (
    PackedTensorProgram,
    pack_tensor_program,
    validate_packed_tensor_program,
)
from mojidiff.representation.program import CodecConfig, CodecError, encode_program

OMNISVG_REPO = "OmniSVG/OmniSVG1.1_4B"
OMNISVG_REVISION = "117d4c4541839e02943554fe71c0a0d54fcef819"
BASE_REPO = "Qwen/Qwen2.5-VL-3B-Instruct"
EXTENDED_VOCABULARY = 197_000
BOS, EOS, PAD = 196_998, 196_999, 151_643
BOX = 200.0
EXTERNAL = Path("/home/dev/workspace/external/OmniSVG")

TRAINING_SYSTEM_PROMPT = "You are an expert SVG code generator."
SYSTEM_PROMPT = (
    "You are an expert SVG code generator. \n"
    "Generate precise, valid SVG path commands that accurately represent the described "
    "scene or object.\n"
    "Focus on capturing key shapes, spatial relationships, and visual composition."
)


IMAGE_INSTRUCTION = "Generate SVG code that accurately represents this image:"
IMAGE_SIZE = 448  # the trainer's and the inference repository's target image size


def image_condition(image: Any) -> Any:
    """A render as the model saw its training images: 448 px RGB on white."""

    from PIL import Image

    rgba = image.convert("RGBA").resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.LANCZOS)
    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
    return Image.alpha_composite(background, rgba).convert("RGB")


def training_instruction(annotation: str) -> str:
    """The text-to-SVG instruction of OmniSVG's training script, verbatim."""

    return f"Generate SVG code for this text description: {annotation}"


def _instruction(prompt: str) -> str:
    return (
        f"Generate an SVG illustration for: {prompt}\n\nRequirements:\n"
        "- Create complete SVG path commands\n"
        "- Include proper coordinates and colors\n"
        "- Maintain visual clarity and composition"
    )


def _stub_moviepy() -> None:
    """deepsvg imports moviepy for video export only; it is not needed to draw."""

    if "moviepy" in sys.modules:
        return
    package = types.ModuleType("moviepy")
    editor = types.ModuleType("moviepy.editor")
    for name in (
        "ImageClip",
        "concatenate_videoclips",
        "VideoClip",
        "ImageSequenceClip",
        "ipython_display",
    ):
        setattr(editor, name, object)
    package.editor = editor  # type: ignore[attr-defined]
    sys.modules["moviepy"] = package
    sys.modules["moviepy.editor"] = editor


def _snapshot(repo: str, revision: str | None = None, patterns: list[str] | None = None) -> Path:
    """The cached snapshot; only the base's config, tokenizer and processor files are needed."""

    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo, revision=revision, allow_patterns=patterns))


@dataclass
class OmniSVG:
    model: Any
    processor: Any
    svg_tokenizer: Any
    black_color_token: int
    checkpoint_sha256: str
    parameters: int

    @classmethod
    def load(cls, device: str = "cuda", *, lora: dict[str, Any] | None = None) -> OmniSVG:
        """The released checkpoint; with `lora`, wrapped for training on its language model.

        The adapters go on the language model's linear projections only - the vision
        tower never sees an input here - and the extended drawing vocabulary is left as
        released, since every OpenMoji token already exists in it.
        """

        import hashlib

        from transformers import AutoConfig, AutoProcessor, Qwen2_5_VLForConditionalGeneration

        base = _snapshot(BASE_REPO, patterns=["*.json", "*.txt", "*.jinja"])
        checkpoint = _snapshot(OMNISVG_REPO, OMNISVG_REVISION) / "pytorch_model.bin"
        digest = hashlib.sha256()
        with checkpoint.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 24), b""):
                digest.update(chunk)
        config = AutoConfig.from_pretrained(base)
        model: Any = Qwen2_5_VLForConditionalGeneration._from_config(  # type: ignore[no-untyped-call]
            config, dtype=torch.bfloat16
        )
        model.resize_token_embeddings(EXTENDED_VOCABULARY, mean_resizing=False)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        expected = set(model.state_dict().keys())
        remapped: dict[str, torch.Tensor] = {}
        unmapped: list[str] = []
        for key, value in state.items():
            target = _remap_key(key, expected)
            if target is None:
                unmapped.append(key)
            else:
                remapped[target] = value
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        if missing or unexpected or unmapped:
            raise RuntimeError(
                f"checkpoint did not load completely: {len(missing)} missing, "
                f"{len(unexpected)} unexpected, {len(unmapped)} unmapped"
            )
        model = model.to(device).eval()
        model.config.bos_token_id, model.config.eos_token_id = BOS, EOS
        model.config.pad_token_id = PAD
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
                    target_modules=LORA_TARGETS_WITH_MERGER
                    if lora.get("merger", False)
                    else LORA_TARGETS,
                    task_type="CAUSAL_LM",
                ),
            )
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.data = parameter.data.float()
        processor: Any = AutoProcessor.from_pretrained(base)  # type: ignore[no-untyped-call]
        svg_tokenizer, black = load_svg_tokenizer()
        return cls(
            model=model,
            processor=processor,
            svg_tokenizer=svg_tokenizer,
            black_color_token=black,
            checkpoint_sha256=digest.hexdigest(),
            parameters=sum(parameter.numel() for parameter in model.parameters()),
        )

    @property
    def trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def _inner(self) -> Any:
        """The conditional-generation model itself, through any PEFT wrapper.

        PEFT wrappers delegate attribute access, so `hasattr` cannot tell the wrapper
        from the model; the wrapper is unwrapped by type.
        """

        from peft import PeftModel

        inner = self.model
        if isinstance(inner, PeftModel):
            inner = inner.base_model.model
        return inner

    def suffix_loss(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        chunk: int,
        *,
        vision: dict[str, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, int]:
        """Summed cross-entropy on the labelled positions, one vocabulary chunk at a time.

        `vision` carries `pixel_values` and `image_grid_thw` for an image-conditioned
        example; the image tokens sit inside `input_ids` where the processor put them.
        """

        from mojidiff.learning.prior import chunked_cross_entropy

        inner = self._inner()
        hidden = inner.model(input_ids=input_ids, **(vision or {})).last_hidden_state
        return chunked_cross_entropy(hidden, inner.lm_head, labels, chunk)

    def prompt_ids(
        self, prompt: str, *, style: str = "release", image: Any = None
    ) -> dict[str, torch.Tensor]:
        """`release`: the inference repository's prompt; `training`: the trainer's;
        `image`: the image-to-SVG prompt both repositories share, with `image` attached."""

        if style == "image":
            if image is None:
                raise ValueError("the image style needs an image")
            from qwen_vl_utils import process_vision_info  # type: ignore[import-untyped]

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": IMAGE_INSTRUCTION},
                        {"type": "image", "image": image_condition(image)},
                    ],
                },
            ]
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, _ = process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=image_inputs, padding=True, return_tensors="pt"
            )
            device = next(self.model.parameters()).device
            return {key: value.to(device) for key, value in inputs.items()}
        if style == "training":
            messages = [
                {"role": "system", "content": TRAINING_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [{"type": "text", "text": training_instruction(prompt)}],
                },
            ]
        elif style == "release":
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": [{"type": "text", "text": _instruction(prompt)}]},
            ]
        else:
            raise ValueError(f"unknown prompt style {style!r}")
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[text], padding=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        return {key: value.to(device) for key, value in inputs.items()}

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        *,
        samples: int,
        max_new_tokens: int,
        seed: int,
        temperature: float = 0.5,
        top_p: float = 0.88,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        style: str = "release",
        greedy: bool = False,
        image: Any = None,
    ) -> list[torch.Tensor]:
        """OmniSVG's own sampling settings for text-to-icon by default; returns the tokens.

        With an image the samples are drawn one at a time, each under its own seed, so
        the vision inputs are never expanded across a batch.
        """

        inputs = self.prompt_ids(prompt, style=style, image=image)
        rounds = (
            [(seed + sample, 1) for sample in range(samples)]
            if image is not None
            else [(seed, samples)]
        )
        drawings: list[torch.Tensor] = []
        length = inputs["input_ids"].shape[1]
        for round_seed, count in rounds:
            torch.manual_seed(round_seed)
            result = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                num_return_sequences=count,
                do_sample=not greedy,
                temperature=None if greedy else temperature,
                top_p=None if greedy else top_p,
                top_k=None if greedy else top_k,
                repetition_penalty=repetition_penalty,
                eos_token_id=EOS,
                pad_token_id=PAD,
                bos_token_id=BOS,
                use_cache=True,
            )
            drawings.extend(row.cpu() for row in result[:, length:])
        return drawings

    def tokens_to_svg(self, tokens: torch.Tensor) -> tuple[str | None, dict[str, Any]]:
        return decode_tokens(self.svg_tokenizer, self.black_color_token, tokens)

    def attach_adapter(self, archive: Path) -> str:
        """Load a saved LoRA archive onto the released model; returns its sha256."""

        from mojidiff.learning.prior import load_adapter, sha256

        payload = archive.read_bytes()
        self.model = load_adapter(self.model, payload).eval()
        return sha256(payload)


LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
# With the merger: the projection from vision patches into the language model's space,
# so that where a drawing starts can be re-grounded in the image.
LORA_TARGETS_WITH_MERGER = (
    r".*(language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"
    r"|visual\.merger\.mlp\.\d+)"
)


def load_svg_tokenizer() -> tuple[Any, int]:
    """The inference repository's token decoder and its black colour token."""

    _stub_moviepy()
    if str(EXTERNAL) not in sys.path:
        sys.path.insert(0, str(EXTERNAL))
    import yaml
    from tokenizer import SVGTokenizer  # type: ignore[import-not-found]

    settings = yaml.safe_load((EXTERNAL / "config.yaml").read_text())
    tokenizer = SVGTokenizer(str(EXTERNAL / "config.yaml"), model_size="4B")
    return tokenizer, int(settings["colors"]["black_color_token"])


def trim_drawing(tokens: torch.Tensor) -> tuple[torch.Tensor, bool]:
    """The generated tokens up to the first EOS, and whether one was generated."""

    ended = bool((tokens == EOS).any())
    trimmed = tokens[: int((tokens == EOS).nonzero()[0, 0])] if ended else tokens
    return trimmed, ended


def decode_tokens(
    svg_tokenizer: Any, black_color_token: int, tokens: torch.Tensor
) -> tuple[str | None, dict[str, Any]]:
    """OmniSVG's decoding: commands, grid points and colours to an SVG string."""

    trimmed, ended = trim_drawing(tokens)
    wrapped = torch.cat((torch.tensor([BOS]), trimmed, torch.tensor([EOS])), dim=0)[None]
    points = svg_tokenizer.process_generated_tokens(wrapped)
    info: dict[str, Any] = {"tokens": int(trimmed.numel()), "ended": ended}
    if len(points) == 0:
        info["failure"] = "no_drawing_tokens"
        return None, info
    tensors, colors = svg_tokenizer.raster_svg(points)
    paths = tensors[0] if tensors and tensors[0] else []
    if not paths:
        info["failure"] = "no_paths"
        return None, info
    while len(colors) < len(paths):
        colors.append(black_color_token)
    try:
        svg = svg_tokenizer.apply_colors_to_svg(paths, colors).to_str()
    except Exception as error:  # noqa: BLE001 - their decoder raises bare exceptions
        info["failure"] = f"decoder_error:{type(error).__name__}"
        return None, info
    info["paths"] = len(paths)
    return str(svg), info


def _remap_key(key: str, expected: set[str]) -> str | None:
    """The checkpoint wraps the model as `transformer.` and uses the older key layout."""

    key = key.removeprefix("transformer.")
    if key in expected:
        return key
    for candidate in ("model." + key, key.replace("model.", "model.language_model.", 1)):
        if candidate in expected:
            return candidate
    return None


# ------------------------------------------------------------------- into the codec


_FILL = re.compile(r'fill="(#[0-9a-fA-F]{6})"')


def _rgb(colour: str) -> np.ndarray:
    return np.array([int(colour[1:3], 16), int(colour[3:5], 16), int(colour[5:7], 16)], dtype=float)


def snap_to_palette(svg: str, palette: tuple[str, ...]) -> tuple[str, dict[str, Any]]:
    """Replace every fill by the nearest palette colour; report how far each moved."""

    table = {colour.lower(): _rgb(colour) for colour in palette}
    distances: list[float] = []

    def replace(match: re.Match[str]) -> str:
        colour = match.group(1).lower()
        if colour in table:
            distances.append(0.0)
            return f'fill="{colour}"'
        rgb = _rgb(colour)
        nearest = min(table, key=lambda name: float(np.linalg.norm(table[name] - rgb)))
        distances.append(float(np.linalg.norm(table[nearest] - rgb)))
        return f'fill="{nearest}"'

    snapped = _FILL.sub(replace, svg)
    return snapped, {
        "fills": len(distances),
        "fills_moved": sum(1 for distance in distances if distance > 0),
        "mean_snap_distance_rgb": float(np.mean(distances)) if distances else 0.0,
    }


def to_project_svg(svg: str, palette: tuple[str, ...]) -> tuple[bytes, dict[str, Any]]:
    """OmniSVG's 200-box, fills-only output as a 72-box SVG in palette colours."""

    body = re.sub(r'\s+filling="\d+"', "", svg)
    body = re.sub(r"<svg[^>]*>", "", body, count=1).replace("</svg>", "")
    # OmniSVG's decoder writes its second colour token as `currentColor`, which its own
    # renderer paints black; the project's normalizer refuses it, so it is black here.
    body = re.sub(r'fill="currentColor"', 'fill="#000000"', body)
    snapped, info = snap_to_palette(body, palette)
    scaled = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 72 72">'
        f'<g transform="scale({72.0 / BOX})">{snapped}</g></svg>'
    )
    return scaled.encode(), info


def parse_into_codec(
    svg: bytes, codec: CodecConfig, total_segment_slots: int
) -> tuple[PackedTensorProgram | None, dict[str, Any]]:
    """The project's own parse: normalise, encode, pack, validate - or say why not."""

    try:
        normalised = normalize_svg(svg)
    except NormalizationError as error:
        return None, {"failure": f"normalize:{error.args[0] if error.args else 'error'}"}
    try:
        dense, report = encode_program(normalised.program, codec)
    except CodecError as error:
        return None, {"failure": f"encode:{str(error)[:80]}"}
    info: dict[str, Any] = {
        "contours": len(normalised.program.contours),
        "dropped_contours": report.dropped_contours,
        "dropped_segments": report.dropped_segments,
        "clamped_coordinates": report.clamped_coordinates,
    }
    if report.dropped_contours or report.dropped_segments or report.clamped_coordinates:
        info["failure"] = "projection_required"
        return None, info
    try:
        packed = pack_tensor_program(dense, codec, total_segment_slots)
        validate_packed_tensor_program(packed, codec, total_segment_slots)
    except CodecError as error:
        info["failure"] = f"pack:{str(error)[:80]}"
        return None, info
    info["active_paths"] = int((packed.path_length > 0).sum())
    info["segments"] = int(packed.path_length.sum())
    return packed, info


def local_snapshot_exists(repo: str) -> bool:
    pattern = os.path.expanduser(
        f"~/.cache/huggingface/hub/models--{repo.replace('/', '--')}/snapshots/*"
    )
    return bool(glob.glob(pattern))
