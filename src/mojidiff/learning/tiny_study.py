"""Deterministic CPU Gate E overfit and checkpoint-resume proof."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import yaml
from PIL import Image, ImageDraw

from mojidiff.learning.geometry import (
    GeometryDenoiser,
    corrupt_factorized_geometry,
    corrupt_path_correlated_geometry,
    corrupt_whole_path_geometry_from_pool,
    geometry_accuracy_by_corruption,
    geometry_loss_and_accuracy,
    packed_batch,
    predict_clean_geometry,
)
from mojidiff.representation.codec_study import _palette, _write_bytes_artifact
from mojidiff.representation.normalizer import normalize_svg
from mojidiff.representation.packed import (
    PackedTensorProgram,
    pack_tensor_program,
    serialize_packed_svg,
    validate_packed_tensor_program,
)
from mojidiff.representation.program import CodecConfig, encode_program
from mojidiff.representation.renderer import RenderLimits, render_typed_svg_isolated
from mojidiff.representation.study import _similarity


class TinyLearningError(RuntimeError):
    """The deterministic tiny learning proof failed or is not reproducible."""


@dataclass(frozen=True)
class TrainCase:
    name: str
    icon_count: int
    corruptions_per_icon: int
    heldout_corruptions_per_icon: int
    steps: int
    resume_step: int | None
    seed: int
    min_train_accuracy: float
    min_heldout_accuracy: float
    max_loss_ratio: float
    resample_each_step: bool
    corruption_kind: str = "factorized_geometry"


@dataclass(frozen=True)
class TinyLearningConfig:
    version: str
    source_revision: str
    raw_root: Path
    fixture: Path
    fixture_sha256: str
    palette_path: Path
    palette_sha256: str
    style_summary: Path
    style_summary_sha256: str
    style_candidate: str
    report_root: Path
    derived_root: Path
    max_paths: int
    max_segments: int
    total_segment_slots: int
    d_model: int
    heads: int
    layers: int
    feedforward: int
    learning_rate: float
    corruption_probability: float
    cases: tuple[TrainCase, ...]
    render_sizes: tuple[int, ...]
    render_timeout_seconds: int


def load_tiny_learning_config(path: Path) -> TinyLearningConfig:
    root = _mapping(yaml.safe_load(path.read_bytes()), "root")
    if root.get("schema_version") != 1:
        raise TinyLearningError("tiny learning schema_version must be 1")
    inputs = _mapping(root.get("inputs"), "inputs")
    codec = _mapping(root.get("codec"), "codec")
    model = _mapping(root.get("model"), "model")
    training = _mapping(root.get("training"), "training")
    render = _mapping(root.get("render"), "render")
    cases = tuple(
        TrainCase(
            name=_string(item, "name"),
            icon_count=_positive_int(item.get("icon_count"), "icon_count"),
            corruptions_per_icon=_positive_int(
                item.get("corruptions_per_icon"), "corruptions_per_icon"
            ),
            heldout_corruptions_per_icon=_positive_int(
                item.get("heldout_corruptions_per_icon"),
                "heldout_corruptions_per_icon",
            ),
            steps=_positive_int(item.get("steps"), "steps"),
            resume_step=(
                None
                if item.get("resume_step") is None
                else _positive_int(item.get("resume_step"), "resume_step")
            ),
            seed=_nonnegative_int(item.get("seed"), "seed"),
            min_train_accuracy=_probability(item.get("min_train_accuracy"), "min_train_accuracy"),
            min_heldout_accuracy=_probability(
                item.get("min_heldout_accuracy"), "min_heldout_accuracy"
            ),
            max_loss_ratio=_probability(item.get("max_loss_ratio"), "max_loss_ratio"),
            resample_each_step=_boolean(
                item.get("resample_each_step", False), "resample_each_step"
            ),
            corruption_kind=_corruption_kind(item.get("corruption_kind", "factorized_geometry")),
        )
        for item in (
            _mapping(value, "training case")
            for value in _sequence(training.get("cases"), "training.cases")
        )
    )
    if not cases or len({case.name for case in cases}) != len(cases):
        raise TinyLearningError("training cases must have unique names")
    if any(case.resume_step is not None and case.resume_step >= case.steps for case in cases):
        raise TinyLearningError("resume step must be smaller than total steps")
    sizes = tuple(
        _positive_int(value, "render size")
        for value in _sequence(render.get("sizes"), "render.sizes")
    )
    result = TinyLearningConfig(
        version=_string(root, "study_version"),
        source_revision=_string(root, "source_revision"),
        raw_root=Path(_string(inputs, "raw_root")),
        fixture=Path(_string(inputs, "fixture")),
        fixture_sha256=_sha256(_string(inputs, "fixture_sha256")),
        palette_path=Path(_string(inputs, "palette")),
        palette_sha256=_sha256(_string(inputs, "palette_sha256")),
        style_summary=Path(_string(inputs, "style_summary")),
        style_summary_sha256=_sha256(_string(inputs, "style_summary_sha256")),
        style_candidate=_string(inputs, "style_candidate"),
        report_root=Path(_string(root, "report_root")),
        derived_root=Path(_string(root, "derived_root")),
        max_paths=_positive_int(codec.get("max_paths"), "max_paths"),
        max_segments=_positive_int(codec.get("max_segments"), "max_segments"),
        total_segment_slots=_positive_int(codec.get("total_segment_slots"), "total_segment_slots"),
        d_model=_positive_int(model.get("d_model"), "d_model"),
        heads=_positive_int(model.get("heads"), "heads"),
        layers=_positive_int(model.get("layers"), "layers"),
        feedforward=_positive_int(model.get("feedforward"), "feedforward"),
        learning_rate=_positive_float(training.get("learning_rate"), "learning_rate"),
        corruption_probability=_probability(
            training.get("corruption_probability"), "corruption_probability"
        ),
        cases=cases,
        render_sizes=sizes,
        render_timeout_seconds=_positive_int(render.get("timeout_seconds"), "timeout_seconds"),
    )
    if result.d_model % result.heads:
        raise TinyLearningError("model.d_model must be divisible by model.heads")
    return result


def run_tiny_learning(config: TinyLearningConfig, config_path: Path) -> dict[str, Any]:
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    _verify(config.fixture, config.fixture_sha256, "fixture")
    _verify(config.palette_path, config.palette_sha256, "palette")
    _verify(config.style_summary, config.style_summary_sha256, "style summary")
    fixture = _mapping(json.loads(config.fixture.read_bytes()), "fixture")
    if fixture.get("source_revision") != config.source_revision:
        raise TinyLearningError("fixture source revision mismatch")
    style = _mapping(json.loads(config.style_summary.read_bytes()), "style summary")
    candidate = _mapping(
        _mapping(style.get("width_vocabulary_analytics"), "width analytics").get(
            config.style_candidate
        ),
        "style candidate",
    )
    widths = tuple(float(value) for value in _sequence(candidate.get("values"), "widths"))
    codec = _codec(config, _palette(config.palette_path), widths)
    rows = [_mapping(row, "fixture row") for row in _sequence(fixture.get("rows"), "rows")]
    clean_programs = [_load_program(row, config, codec) for row in rows]
    device = torch.device("cpu")
    all_metrics: list[dict[str, Any]] = []
    case_summaries: dict[str, Any] = {}
    final_models: dict[str, GeometryDenoiser] = {}
    heldout_sets: dict[str, list[PackedTensorProgram]] = {}

    for case in config.cases:
        if case.icon_count > len(clean_programs):
            raise TinyLearningError(f"case {case.name} requests too many icons")
        clean = clean_programs[: case.icon_count]
        noisy, clean_examples = _corruption_set(
            clean,
            codec,
            config.corruption_probability,
            case.corruptions_per_icon,
            case.seed,
            case.corruption_kind,
            clean,
        )
        heldout, heldout_clean = _corruption_set(
            clean,
            codec,
            config.corruption_probability,
            case.heldout_corruptions_per_icon,
            case.seed + 1_000_000,
            case.corruption_kind,
            clean,
        )
        heldout_sets[case.name] = heldout
        initial_model, initial_optimizer = _new_training(config, codec, case.seed)
        initial = _evaluate(initial_model, heldout, heldout_clean, codec, device)
        single_losses = _train(
            initial_model,
            initial_optimizer,
            noisy,
            clean_examples,
            codec,
            device,
            case.steps,
            case,
            clean,
            config.corruption_probability,
            all_metrics,
        )
        checkpoint_hash: str | None = None
        resume_exact: bool | None = None
        continuous_loss_hash: str | None = None
        resumed_loss_hash: str | None = None
        model = initial_model
        if case.resume_step is not None:
            continuous_hash = _model_hash(initial_model)
            continuous_losses = single_losses
            continuous_loss_hash = _loss_hash(continuous_losses)
            split_model, split_optimizer = _new_training(config, codec, case.seed)
            split_metrics: list[dict[str, Any]] = []
            prefix = _train(
                split_model,
                split_optimizer,
                noisy,
                clean_examples,
                codec,
                device,
                case.resume_step,
                case,
                clean,
                config.corruption_probability,
                split_metrics,
                metric_name=f"{case.name}-resume-prefix",
            )
            checkpoint = _save_checkpoint(split_model, split_optimizer, case.resume_step)
            checkpoint_hash = hashlib.sha256(checkpoint).hexdigest()
            checkpoint_path = config.derived_root / case.name / f"checkpoint-{case.resume_step}.zip"
            _write_bytes_artifact(checkpoint_path, checkpoint)
            resumed_model, resumed_optimizer, loaded_step = _load_checkpoint(
                checkpoint, config, codec
            )
            if loaded_step != case.resume_step:
                raise TinyLearningError("checkpoint step mismatch")
            suffix = _train(
                resumed_model,
                resumed_optimizer,
                noisy,
                clean_examples,
                codec,
                device,
                case.steps - case.resume_step,
                case,
                clean,
                config.corruption_probability,
                split_metrics,
                step_offset=case.resume_step,
                metric_name=f"{case.name}-resume-suffix",
            )
            resume_exact = (
                continuous_hash == _model_hash(resumed_model)
                and continuous_losses == prefix + suffix
            )
            resumed_loss_hash = _loss_hash(prefix + suffix)
            if not resume_exact:
                raise TinyLearningError("checkpoint continuation diverged from continuous run")
            model = resumed_model
        train_final = _evaluate(model, noisy, clean_examples, codec, device)
        final = _evaluate(model, heldout, heldout_clean, codec, device)
        loss_ratio = single_losses[-1] / single_losses[0]
        passes_criteria = (
            train_final["accuracy"] >= case.min_train_accuracy
            and final["accuracy"] >= case.min_heldout_accuracy
            and loss_ratio <= case.max_loss_ratio
            and resume_exact is not False
        )
        final_models[case.name] = model
        case_summaries[case.name] = {
            "config": asdict(case),
            "train_examples": len(noisy),
            "heldout_examples": len(heldout),
            "initial": initial,
            "train_final": train_final,
            "final": final,
            "loss_ratio": loss_ratio,
            "checkpoint_sha256": checkpoint_hash,
            "resume_exact": resume_exact,
            "continuous_loss_sha256": continuous_loss_hash,
            "resumed_loss_sha256": resumed_loss_hash,
            "final_model_sha256": _model_hash(model),
            "passes_predeclared_criteria": passes_criteria,
        }

    render_rows, sheet = _render_evidence(
        config,
        codec,
        rows,
        clean_programs,
        config.cases[-1],
        final_models[config.cases[-1].name],
        heldout_sets[config.cases[-1].name],
        device,
    )
    metrics_payload = _jsonl(all_metrics)
    render_payload = _jsonl(render_rows)
    summary = {
        "schema_version": 1,
        "study_version": config.version,
        "code_identity": _code_identity(),
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "fixture": str(config.fixture),
        "fixture_sha256": config.fixture_sha256,
        "torch_version": str(torch.__version__),
        "device": "cpu",
        "deterministic_algorithms": True,
        "model": {
            "d_model": config.d_model,
            "heads": config.heads,
            "layers": config.layers,
            "feedforward": config.feedforward,
            "parameters": sum(
                parameter.numel() for parameter in final_models[config.cases[-1].name].parameters()
            ),
            "scope": "fixed-topology geometry-only diagnostic",
        },
        "cases": case_summaries,
        "passes_predeclared_gate_candidate": all(
            item["passes_predeclared_criteria"] for item in case_summaries.values()
        ),
        "metrics_sha256": hashlib.sha256(metrics_payload).hexdigest(),
        "render_metrics_sha256": hashlib.sha256(render_payload).hexdigest(),
    }
    _write_bytes_artifact(config.report_root / "metrics.jsonl", metrics_payload)
    _write_bytes_artifact(config.report_root / "render-metrics.jsonl", render_payload)
    _write_bytes_artifact(config.report_root / "trajectory.png", sheet)
    _write_bytes_artifact(config.report_root / "summary.json", _json(summary))
    _write_bytes_artifact(config.report_root / "README.md", _markdown(summary).encode())
    return summary


def _new_training(
    config: TinyLearningConfig, codec: CodecConfig, seed: int
) -> tuple[GeometryDenoiser, torch.optim.Optimizer]:
    torch.manual_seed(seed)
    model = GeometryDenoiser(
        codec,
        config.total_segment_slots,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        feedforward=config.feedforward,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    return model, optimizer


def _train(
    model: GeometryDenoiser,
    optimizer: torch.optim.Optimizer,
    noisy: list[PackedTensorProgram],
    clean: list[PackedTensorProgram],
    codec: CodecConfig,
    device: torch.device,
    steps: int,
    case: TrainCase,
    clean_icons: list[PackedTensorProgram],
    corruption_probability: float,
    metrics: list[dict[str, Any]],
    *,
    step_offset: int = 0,
    metric_name: str | None = None,
) -> list[float]:
    noisy_batch = packed_batch(noisy, device)
    clean_batch = packed_batch(clean, device)
    losses: list[float] = []
    model.train()
    for local_step in range(1, steps + 1):
        step = step_offset + local_step
        if case.resample_each_step:
            step_noisy, step_clean = _corruption_set(
                clean_icons,
                codec,
                corruption_probability,
                case.corruptions_per_icon,
                case.seed + 2_000_000 + step * 100_000,
                case.corruption_kind,
                clean_icons,
            )
            noisy_batch = packed_batch(step_noisy, device)
            clean_batch = packed_batch(step_clean, device)
        optimizer.zero_grad(set_to_none=True)
        loss, counts = geometry_loss_and_accuracy(model(noisy_batch), clean_batch, codec)
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        value = float(loss.detach())
        losses.append(value)
        if step == 1 or step == step_offset + steps or step % 10 == 0:
            metrics.append(
                {
                    "case": metric_name or case.name,
                    "step": step,
                    "loss": value,
                    "train_token_accuracy": counts["correct"] / counts["total"],
                }
            )
    return losses


def _evaluate(
    model: GeometryDenoiser,
    noisy: list[PackedTensorProgram],
    clean: list[PackedTensorProgram],
    codec: CodecConfig,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    correct = 0
    total = 0
    changed_correct = 0
    changed_total = 0
    retained_correct = 0
    retained_total = 0
    losses: list[float] = []
    with torch.no_grad():
        for item, target in zip(noisy, clean, strict=True):
            noisy_batch = packed_batch([item], device)
            clean_batch = packed_batch([target], device)
            logits = model(noisy_batch)
            loss, counts = geometry_loss_and_accuracy(logits, clean_batch, codec)
            split = geometry_accuracy_by_corruption(logits, noisy_batch, clean_batch, codec)
            losses.append(float(loss))
            correct += counts["correct"]
            total += counts["total"]
            changed_correct += split["changed"]["correct"]
            changed_total += split["changed"]["total"]
            retained_correct += split["retained"]["correct"]
            retained_total += split["retained"]["total"]
    return {
        "loss": float(np.mean(losses)),
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "changed_correct": changed_correct,
        "changed_total": changed_total,
        "changed_accuracy": changed_correct / changed_total if changed_total else None,
        "retained_correct": retained_correct,
        "retained_total": retained_total,
        "retained_accuracy": retained_correct / retained_total if retained_total else None,
    }


def _corruption_set(
    clean: list[PackedTensorProgram],
    codec: CodecConfig,
    probability: float,
    repetitions: int,
    seed: int,
    corruption_kind: str,
    donor_pool: list[PackedTensorProgram],
) -> tuple[list[PackedTensorProgram], list[PackedTensorProgram]]:
    noisy: list[PackedTensorProgram] = []
    targets: list[PackedTensorProgram] = []
    for icon_index, program in enumerate(clean):
        for corruption_index in range(repetitions):
            rng = np.random.default_rng(seed + icon_index * 10_000 + corruption_index)
            noisy.append(_corrupt(program, codec, probability, rng, corruption_kind, donor_pool))
            targets.append(program)
    return noisy, targets


def _corrupt(
    program: PackedTensorProgram,
    codec: CodecConfig,
    probability: float,
    rng: np.random.Generator,
    corruption_kind: str,
    donor_pool: list[PackedTensorProgram],
) -> PackedTensorProgram:
    """Apply one predeclared fixed-topology geometry corruption contract."""

    if corruption_kind == "factorized_geometry":
        return corrupt_factorized_geometry(program, codec, probability, rng)
    if corruption_kind == "path_correlated_geometry":
        return corrupt_path_correlated_geometry(program, codec, probability, rng)
    if corruption_kind == "whole_path_geometry":
        return corrupt_whole_path_geometry_from_pool(program, donor_pool, codec, probability, rng)
    raise TinyLearningError(f"unsupported corruption kind: {corruption_kind}")


def _load_program(
    row: dict[str, Any], config: TinyLearningConfig, codec: CodecConfig
) -> PackedTensorProgram:
    relative = Path(_string(row, "source_path"))
    source = (config.raw_root / relative).read_bytes()
    if hashlib.sha256(source).hexdigest() != _string(row, "source_svg_sha256"):
        raise TinyLearningError(f"source hash mismatch: {relative}")
    dense, report = encode_program(normalize_svg(source).program, codec)
    if report.dropped_contours or report.dropped_segments or report.clamped_coordinates:
        raise TinyLearningError(f"fixture requires safety projection: {relative}")
    packed = pack_tensor_program(dense, codec, config.total_segment_slots)
    validate_packed_tensor_program(packed, codec, config.total_segment_slots)
    return packed


def _render_evidence(
    config: TinyLearningConfig,
    codec: CodecConfig,
    rows: list[dict[str, Any]],
    clean: list[PackedTensorProgram],
    case: TrainCase,
    model: GeometryDenoiser,
    noisy: list[PackedTensorProgram],
    device: torch.device,
) -> tuple[list[dict[str, Any]], bytes]:
    render_rows: list[dict[str, Any]] = []
    tiles: list[tuple[str, dict[str, np.ndarray[Any, Any]]]] = []
    limits = RenderLimits(
        max_paths=config.max_paths,
        timeout_seconds=config.render_timeout_seconds,
    )
    model.eval()
    render_noisy = [
        noisy[index * case.heldout_corruptions_per_icon] for index in range(case.icon_count)
    ]
    for row, clean_program, noisy_program in zip(
        rows[: case.icon_count], clean[: case.icon_count], render_noisy, strict=True
    ):
        with torch.no_grad():
            prediction = predict_clean_geometry(
                noisy_program,
                model(packed_batch([noisy_program], device)),
                codec,
            )
        programs = {"x_0": clean_program, "x_t": noisy_program, "x_hat_0": prediction}
        images: dict[str, np.ndarray[Any, Any]] = {}
        for size in config.render_sizes:
            rendered = {
                name: render_typed_svg_isolated(
                    serialize_packed_svg(program, codec, config.total_segment_slots),
                    size,
                    limits,
                )[1]
                for name, program in programs.items()
            }
            images.update({f"{name}-{size}": image for name, image in rendered.items()})
            for name in ("x_t", "x_hat_0"):
                metrics = _similarity(rendered["x_0"], rendered[name])
                render_rows.append(
                    {
                        "case": case.name,
                        "hexcode": row["hexcode"],
                        "size": size,
                        "state": name,
                        **metrics,
                    }
                )
        tiles.append((str(row["hexcode"]), images))
    return render_rows, _contact_sheet(tiles, config.render_sizes)


def _contact_sheet(
    rows: list[tuple[str, dict[str, np.ndarray[Any, Any]]]], sizes: tuple[int, ...]
) -> bytes:
    columns = tuple((state, size) for size in sizes for state in ("x_0", "x_t", "x_hat_0"))
    tile = 88
    label_width = 130
    header = 30
    canvas = Image.new(
        "RGB",
        (label_width + tile * len(columns), header + tile * len(rows)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for column, (state, size) in enumerate(columns):
        draw.text((label_width + column * tile + 4, 8), f"{state} {size}", fill="black")
    for row_index, (hexcode, images) in enumerate(rows):
        y = header + row_index * tile
        draw.text((4, y + 8), hexcode, fill="black")
        for column, (state, size) in enumerate(columns):
            rgba = Image.fromarray(images[f"{state}-{size}"].astype(np.uint8), "RGBA")
            white = Image.new("RGBA", rgba.size, "white")
            white.alpha_composite(rgba)
            canvas.paste(
                white.convert("RGB").resize((72, 72)),
                (label_width + column * tile + 8, y + 8),
            )
    payload = io.BytesIO()
    canvas.save(payload, format="PNG", optimize=True)
    return payload.getvalue()


def _save_checkpoint(model: GeometryDenoiser, optimizer: torch.optim.Optimizer, step: int) -> bytes:
    document = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
    }
    tensors: list[tuple[str, bytes]] = []
    tree = _encode_checkpoint_value(document, tensors)
    manifest = _json({"schema_version": 1, "tree": tree})
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, mode="w", compression=zipfile.ZIP_STORED) as archive:
        _write_checkpoint_member(archive, "manifest.json", manifest)
        for name, tensor_payload in tensors:
            _write_checkpoint_member(archive, name, tensor_payload)
    return payload.getvalue()


def _load_checkpoint(
    payload: bytes, config: TinyLearningConfig, codec: CodecConfig
) -> tuple[GeometryDenoiser, torch.optim.Optimizer, int]:
    document = _decode_checkpoint(payload)
    model, optimizer = _new_training(config, codec, seed=0)
    model.load_state_dict(document["model"])
    optimizer.load_state_dict(document["optimizer"])
    torch.set_rng_state(document["torch_rng_state"])
    return model, optimizer, int(document["step"])


def _encode_checkpoint_value(value: object, tensors: list[tuple[str, bytes]]) -> dict[str, Any]:
    if isinstance(value, torch.Tensor):
        name = f"tensors/{len(tensors):06d}.npy"
        payload = io.BytesIO()
        np.save(payload, value.detach().cpu().numpy(), allow_pickle=False)
        tensors.append((name, payload.getvalue()))
        return {"kind": "tensor", "member": name}
    if isinstance(value, dict):
        items = []
        for key in sorted(value, key=_checkpoint_key_sort):
            items.append(
                {
                    "key": _encode_checkpoint_key(key),
                    "value": _encode_checkpoint_value(value[key], tensors),
                }
            )
        return {"kind": "dict", "items": items}
    if isinstance(value, list):
        return {
            "kind": "list",
            "items": [_encode_checkpoint_value(item, tensors) for item in value],
        }
    if isinstance(value, tuple):
        return {
            "kind": "tuple",
            "items": [_encode_checkpoint_value(item, tensors) for item in value],
        }
    if value is None or isinstance(value, (bool, int, float, str)):
        return {"kind": "scalar", "value": value}
    raise TinyLearningError(f"unsupported checkpoint value: {type(value).__name__}")


def _encode_checkpoint_key(value: object) -> dict[str, int | str]:
    if isinstance(value, bool):
        raise TinyLearningError("boolean checkpoint mapping keys are unsupported")
    if isinstance(value, int):
        return {"kind": "int", "value": value}
    if isinstance(value, str):
        return {"kind": "str", "value": value}
    raise TinyLearningError(f"unsupported checkpoint mapping key: {type(value).__name__}")


def _checkpoint_key_sort(value: object) -> tuple[str, str]:
    encoded = _encode_checkpoint_key(value)
    return str(encoded["kind"]), str(encoded["value"])


def _write_checkpoint_member(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_STORED
    info.create_system = 3
    info.external_attr = 0o600 << 16
    archive.writestr(info, payload)


def _decode_checkpoint(payload: bytes) -> dict[str, Any]:
    if len(payload) > 100_000_000:
        raise TinyLearningError("checkpoint archive exceeds 100 MB")
    try:
        with zipfile.ZipFile(io.BytesIO(payload), mode="r") as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            if len(names) != len(set(names)) or "manifest.json" not in names:
                raise TinyLearningError("checkpoint member names are invalid")
            if any(
                info.compress_type != zipfile.ZIP_STORED
                or info.file_size > 50_000_000
                or not (
                    info.filename == "manifest.json"
                    or (
                        info.filename.startswith("tensors/")
                        and info.filename.endswith(".npy")
                        and info.filename.count("/") == 1
                    )
                )
                for info in infos
            ):
                raise TinyLearningError("checkpoint member violates archive limits")
            if sum(info.file_size for info in infos) > 100_000_000:
                raise TinyLearningError("checkpoint contents exceed 100 MB")
            manifest = _mapping(json.loads(archive.read("manifest.json")), "checkpoint")
            if manifest.get("schema_version") != 1:
                raise TinyLearningError("checkpoint schema_version must be 1")
            document = _decode_checkpoint_value(
                _mapping(manifest.get("tree"), "checkpoint tree"), archive
            )
    except (json.JSONDecodeError, KeyError, ValueError, zipfile.BadZipFile) as error:
        raise TinyLearningError("invalid checkpoint archive") from error
    if not isinstance(document, dict):
        raise TinyLearningError("checkpoint root must be a mapping")
    required = {"step", "model", "optimizer", "torch_rng_state"}
    if set(document) != required:
        raise TinyLearningError("checkpoint root fields are invalid")
    return cast(dict[str, Any], document)


def _decode_checkpoint_value(value: dict[str, Any], archive: zipfile.ZipFile) -> Any:
    kind = value.get("kind")
    if kind == "tensor":
        member = value.get("member")
        if not isinstance(member, str) or member not in archive.namelist():
            raise TinyLearningError("checkpoint tensor member is invalid")
        array = np.load(io.BytesIO(archive.read(member)), allow_pickle=False)
        if not isinstance(array, np.ndarray):
            raise TinyLearningError("checkpoint tensor is not an array")
        return torch.from_numpy(array.copy())
    if kind in {"list", "tuple"}:
        items = _sequence(value.get("items"), "checkpoint sequence")
        decoded = [
            _decode_checkpoint_value(_mapping(item, "checkpoint item"), archive) for item in items
        ]
        return tuple(decoded) if kind == "tuple" else decoded
    if kind == "dict":
        items = _sequence(value.get("items"), "checkpoint mapping")
        result: dict[int | str, Any] = {}
        for item in items:
            entry = _mapping(item, "checkpoint mapping item")
            key = _decode_checkpoint_key(_mapping(entry.get("key"), "checkpoint mapping key"))
            if key in result:
                raise TinyLearningError("duplicate checkpoint mapping key")
            result[key] = _decode_checkpoint_value(
                _mapping(entry.get("value"), "checkpoint mapping value"), archive
            )
        return result
    if kind == "scalar":
        scalar = value.get("value")
        if scalar is None or isinstance(scalar, (bool, int, float, str)):
            return scalar
    raise TinyLearningError("checkpoint value is invalid")


def _decode_checkpoint_key(value: dict[str, Any]) -> int | str:
    kind = value.get("kind")
    result = value.get("value")
    if kind == "int" and isinstance(result, int) and not isinstance(result, bool):
        return result
    if kind == "str" and isinstance(result, str):
        return result
    raise TinyLearningError("checkpoint mapping key is invalid")


def _model_hash(model: GeometryDenoiser) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode())
        array = tensor.detach().cpu().numpy()
        digest.update(str(array.shape).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _loss_hash(losses: list[float]) -> str:
    return hashlib.sha256(np.asarray(losses, dtype="<f8").tobytes()).hexdigest()


def _codec(
    config: TinyLearningConfig, palette: tuple[str, ...], widths: tuple[float, ...]
) -> CodecConfig:
    return CodecConfig(
        max_paths=config.max_paths,
        max_segments=config.max_segments,
        coordinate_bins=289,
        control_coordinate_bins=417,
        control_coordinate_min=-8.0,
        control_coordinate_max=96.0,
        palette=palette,
        stroke_widths=widths,
        dash_patterns=(
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.1598, 5.1598),
            (0.0, 0.0, 3.9396, 3.9396),
            (0.0, 6.7346, 0.0, 0.0, 0.0, 0.0),
            (2.0, 4.0),
            (5.2132, 5.2132),
            (6.1156, 4.5867),
        ),
        miter_limits=(1.5, 2.0, 4.0, 7.0, 10.0),
        opacities=(0.25, 0.4, 0.5, 0.502, 0.6, 0.9969, 0.997, 0.999, 1.0),
        max_serialized_bytes=2_000_000,
    )


def _code_identity() -> dict[str, str]:
    repository = Path(__file__).resolve().parents[3]
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    ).stdout.strip()
    sources = {
        "geometry_sha256": repository / "src/mojidiff/learning/geometry.py",
        "tiny_study_sha256": Path(__file__),
        "packed_sha256": repository / "src/mojidiff/representation/packed.py",
        "program_sha256": repository / "src/mojidiff/representation/program.py",
    }
    return {"git_commit": commit, **{name: _file_sha256(path) for name, path in sources.items()}}


def _markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Tiny fixed-topology geometry learning proof",
        "",
        f"PyTorch {summary['torch_version']} on deterministic CPU; "
        f"{summary['model']['parameters']} parameters.",
        "",
        "| case | train accuracy | held-out accuracy | changed accuracy | retained accuracy | "
        "resume exact | passes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in summary["cases"].items():
        lines.append(
            f"| {name} | {item['train_final']['accuracy']:.4f} | "
            f"{item['final']['accuracy']:.4f} | "
            f"{item['final']['changed_accuracy']:.4f} | "
            f"{item['final']['retained_accuracy']:.4f} | {item['resume_exact']} | "
            f"{item['passes_predeclared_criteria']} |"
        )
    lines.extend(
        [
            "",
            "This is a fixed-topology, geometry-only diagnostic. It is evidence that the "
            "packed pipeline can learn and resume; it is not evidence for the final "
            "corruption process or unconditional generation.",
            "",
        ]
    )
    return "\n".join(lines)


def _verify(path: Path, expected: str, label: str) -> None:
    if _file_sha256(path) != expected:
        raise TinyLearningError(f"{label} hash mismatch")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TinyLearningError(f"{field} must be a mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise TinyLearningError(f"{field} must be a list")
    return value


def _string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise TinyLearningError(f"{field} must be a non-empty string")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TinyLearningError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TinyLearningError(f"{field} must be a nonnegative integer")
    return value


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise TinyLearningError(f"{field} must be positive")
    return float(value)


def _probability(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise TinyLearningError(f"{field} must be within 0..1")
    return float(value)


def _boolean(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise TinyLearningError(f"{field} must be a boolean")
    return value


def _corruption_kind(value: object) -> str:
    if value not in {"factorized_geometry", "path_correlated_geometry", "whole_path_geometry"}:
        raise TinyLearningError("corruption_kind must name a supported fixed-topology contract")
    return value


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise TinyLearningError("hash must be lowercase SHA-256")
    return value


def _json(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()


def _jsonl(rows: list[dict[str, Any]]) -> bytes:
    return "".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="mojidiff-tiny-learning")
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(argv)
    summary = run_tiny_learning(load_tiny_learning_config(args.config), args.config)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
