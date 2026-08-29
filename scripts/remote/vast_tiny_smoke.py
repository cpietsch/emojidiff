#!/usr/bin/env python3
"""Run the bounded Vast tokenizer/render/GPU/checkpoint/artifact smoke contract."""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
import os
import re
import stat
import sys
from dataclasses import fields
from pathlib import Path, PurePosixPath
from typing import Any

_RUN_ID = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_REVISION = re.compile(r"[0-9a-f]{7,64}\Z")
_MAX_CHECKPOINT_BYTES = 16 * 1024 * 1024
_MAX_RESULT_BYTES = 64 * 1024
_MAX_STAGE_MANIFEST_BYTES = 64 * 1024
_MAX_STAGE_MEMBERS = 50_000
_STAGE_MANIFEST = ".mojidiff-stage.json"
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.dont_write_bytecode = True
sys.path.insert(0, str(_SOURCE_ROOT / "src"))


class SmokeError(RuntimeError):
    """The tiny worker smoke contract failed closed."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def _decode_config(encoded: str) -> dict[str, Any]:
    try:
        padding = "=" * (-len(encoded) % 4)
        value = json.loads(base64.urlsafe_b64decode(encoded + padding))
    except (ValueError, UnicodeError) as exc:
        raise SmokeError(f"invalid encoded smoke configuration: {type(exc).__name__}") from exc
    required = {
        "artifact_root",
        "archive_sha256",
        "config_sha256",
        "git_revision",
        "max_steps",
        "max_storage_bytes",
        "run_id",
        "smoke_id",
        "source_root",
        "tree_sha256",
        "workspace_root",
    }
    if not isinstance(value, dict) or set(value) != required:
        raise SmokeError("invalid smoke configuration fields")
    return value


def _safe_id(value: object, field: str) -> str:
    if not isinstance(value, str) or _RUN_ID.fullmatch(value) is None:
        raise SmokeError(f"invalid {field}")
    return value


def _digest(value: object, field: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise SmokeError(f"invalid {field}")
    return value


def _path(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or "\x00" in value or "\n" in value:
        raise SmokeError(f"invalid {field}")
    parsed = PurePosixPath(value)
    if not parsed.is_absolute() or ".." in parsed.parts or parsed.as_posix() != value.rstrip("/"):
        raise SmokeError(f"{field} must be a normalized absolute POSIX path")
    return Path(parsed)


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SmokeError(f"{field} must be a positive integer")
    return value


def _validated_config(encoded: str) -> dict[str, Any]:
    config = _decode_config(encoded)
    run_id = _safe_id(config["run_id"], "run_id")
    _safe_id(config["smoke_id"], "smoke_id")
    for field in ("archive_sha256", "config_sha256", "tree_sha256"):
        _digest(config[field], field)
    revision = config["git_revision"]
    if not isinstance(revision, str) or _GIT_REVISION.fullmatch(revision) is None:
        raise SmokeError("invalid git_revision")
    _positive_int(config["max_steps"], "max_steps")
    storage = _positive_int(config["max_storage_bytes"], "max_storage_bytes")
    minimum_outputs = 2 * _MAX_CHECKPOINT_BYTES + 2 * _MAX_RESULT_BYTES
    if storage < minimum_outputs:
        raise SmokeError("storage cap is too small for both bounded smoke artifact copies")
    workspace = _path(config["workspace_root"], "workspace_root")
    source = _path(config["source_root"], "source_root")
    artifact = _path(config["artifact_root"], "artifact_root")
    if source != workspace / run_id / "source":
        raise SmokeError("source_root does not match the immutable run path")
    for path, field in (
        (workspace, "workspace_root"),
        (source, "source_root"),
        (artifact, "artifact_root"),
    ):
        if path.is_symlink() or not path.is_dir():
            raise SmokeError(f"{field} must be an existing non-symlink directory")
    resolved_workspace = workspace.resolve(strict=False)
    resolved_source = source.resolve(strict=False)
    resolved_artifact = artifact.resolve(strict=False)
    if resolved_workspace != workspace or resolved_source != source:
        raise SmokeError("workspace and staged source paths must not contain symbolic links")
    if resolved_artifact != artifact:
        raise SmokeError("artifact root path must not contain symbolic links")
    if (
        resolved_artifact == resolved_workspace
        or resolved_workspace in resolved_artifact.parents
        or resolved_artifact in resolved_workspace.parents
    ):
        raise SmokeError("artifact root and ephemeral worker workspace must be disjoint")
    if _SOURCE_ROOT != resolved_source:
        raise SmokeError("smoke script is not executing from the expected staged snapshot")
    return config


def _staged_tree_identity(source: Path, maximum: int) -> tuple[str, int]:
    """Recompute the exact staged tree identity before importing project code."""

    if source.is_symlink() or not source.is_dir():
        raise SmokeError("staged source is not a real directory")
    records: list[tuple[str, int, bytes]] = []
    stored_bytes = 0
    members = 0
    for path in sorted(source.rglob("*")):
        members += 1
        if members > _MAX_STAGE_MEMBERS:
            raise SmokeError("staged source contains too many members")
        if path.is_symlink():
            raise SmokeError("staged source contains a symbolic link")
        if path.is_dir():
            continue
        if not path.is_file():
            raise SmokeError("staged source contains a special file")
        name = path.relative_to(source).as_posix()
        if name == _STAGE_MANIFEST:
            continue
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as staged_file:
                while True:
                    chunk = staged_file.read(1024 * 1024)
                    if not chunk:
                        break
                    size += len(chunk)
                    stored_bytes += len(chunk)
                    if stored_bytes > maximum:
                        raise SmokeError("staged source exceeds the configured storage cap")
                    digest.update(chunk)
        except OSError as exc:
            raise SmokeError(f"staged source cannot be read: {type(exc).__name__}") from exc
        records.append((name, size, digest.digest()))
    if not records:
        raise SmokeError("staged source contains no regular files")
    tree = hashlib.sha256(b"mojidiff-tree-v1\0")
    for name, size, digest_bytes in sorted(records):
        tree.update(name.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(size).encode("ascii"))
        tree.update(b"\0")
        tree.update(digest_bytes)
    return tree.hexdigest(), stored_bytes


def _verify_stage(source: Path, config: dict[str, Any]) -> tuple[dict[str, Any], int]:
    expected = {
        "archive_sha256": config["archive_sha256"],
        "config_sha256": config["config_sha256"],
        "git_revision": config["git_revision"],
        "run_id": config["run_id"],
        "schema_version": 1,
        "tree_sha256": config["tree_sha256"],
    }
    manifest_path = source / _STAGE_MANIFEST
    if manifest_path.is_symlink():
        raise SmokeError("stage manifest is a symbolic link")
    try:
        with manifest_path.open("rb") as manifest_file:
            manifest_bytes = manifest_file.read(_MAX_STAGE_MANIFEST_BYTES + 1)
        if len(manifest_bytes) > _MAX_STAGE_MANIFEST_BYTES:
            raise SmokeError("stage manifest exceeds its bounded size")
        observed = json.loads(manifest_bytes)
    except (OSError, ValueError, UnicodeError) as exc:
        raise SmokeError(f"stage manifest is unavailable: {type(exc).__name__}") from exc
    if observed != expected:
        raise SmokeError("stage manifest does not match the requested immutable identity")
    tree_sha256, source_bytes = _staged_tree_identity(
        source, _positive_int(config["max_storage_bytes"], "max_storage_bytes")
    )
    if tree_sha256 != config["tree_sha256"]:
        raise SmokeError("staged source tree hash does not match its immutable identity")
    storage_bytes = source_bytes + len(manifest_bytes)
    if storage_bytes > config["max_storage_bytes"]:
        raise SmokeError("staged source exceeds the configured storage cap")
    return expected, storage_bytes


def _write_or_verify(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise SmokeError(f"refusing to write through artifact symlink: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
    except FileExistsError:
        if path.is_symlink() or not path.is_file():
            raise SmokeError(f"existing artifact is not a regular file: {path}") from None
        try:
            with path.open("rb") as source:
                observed = source.read(len(payload) + 1)
        except OSError as exc:
            raise SmokeError(f"cannot verify existing artifact: {type(exc).__name__}") from exc
        if observed != payload:
            raise SmokeError(f"refusing to overwrite different existing artifact: {path}") from None


def _directory_storage_bytes(root: Path) -> int:
    if root.is_symlink():
        raise SmokeError(f"run artifact root is a symbolic link: {root}")
    if not root.exists():
        return 0
    if not root.is_dir():
        raise SmokeError(f"run artifact root is not a directory: {root}")
    total = 0
    for path in root.rglob("*"):
        try:
            mode = path.lstat().st_mode
        except OSError as exc:
            raise SmokeError(f"cannot inspect run artifact: {type(exc).__name__}") from exc
        if stat.S_ISLNK(mode):
            raise SmokeError(f"run artifacts contain a symbolic link: {path}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise SmokeError(f"run artifacts contain a special file: {path}")
        total += path.stat().st_size
    return total


def _additional_bytes(path: Path, payload: bytes) -> int:
    if path.is_symlink():
        raise SmokeError(f"refusing to verify artifact symlink: {path}")
    if not path.exists():
        return len(payload)
    if not path.is_file():
        raise SmokeError(f"existing artifact is not a regular file: {path}")
    try:
        with path.open("rb") as source:
            observed = source.read(len(payload) + 1)
    except OSError as exc:
        raise SmokeError(f"cannot verify existing artifact: {type(exc).__name__}") from exc
    if observed != payload:
        raise SmokeError(f"refusing to overwrite different existing artifact: {path}")
    return 0


def _codec_and_render_smoke() -> tuple[dict[str, Any], list[float]]:
    try:
        import cairosvg
        import numpy as np

        from mojidiff.representation.program import (
            OPACITY_VOCABULARY,
            CodecConfig,
            FloatContour,
            FloatProgram,
            FloatSegment,
            SegmentType,
            decode_program,
            encode_program,
            serialize_svg,
        )
    except Exception as exc:
        raise SmokeError(f"tokenizer/renderer import failed: {type(exc).__name__}") from exc

    codec = CodecConfig(
        max_paths=1,
        max_segments=4,
        coordinate_bins=289,
        palette=("#000000",),
        stroke_widths=(2.0,),
        dash_patterns=((2.0, 2.0),),
        miter_limits=(4.0,),
        opacities=OPACITY_VOCABULARY,
        max_serialized_bytes=4096,
    )
    source = FloatProgram(
        (
            FloatContour(
                layer=1,
                fill="#000000",
                stroke="#000000",
                stroke_width=2.0,
                linecap="round",
                linejoin="round",
                miter_limit=4.0,
                dash_pattern=(),
                fill_rule="nonzero",
                opacity=0.5,
                fill_opacity=0.6,
                stroke_opacity=0.4,
                start=(8.0, 64.0),
                segments=(
                    FloatSegment(SegmentType.LINE, (36.0, 8.0)),
                    FloatSegment(SegmentType.LINE, (64.0, 64.0)),
                    FloatSegment(SegmentType.CLOSE, ()),
                ),
            ),
        )
    )
    tensor, encoding = encode_program(source, codec)
    decoded = decode_program(tensor, codec)
    reencoded, second = encode_program(decoded, codec)
    for field in fields(tensor):
        if not np.array_equal(getattr(tensor, field.name), getattr(reencoded, field.name)):
            raise SmokeError(f"typed tokenizer round trip changed {field.name}")
    if not encoding.lossless or not second.lossless:
        raise SmokeError("tiny tokenizer fixture unexpectedly projected data")
    svg = serialize_svg(tensor, codec)
    try:
        png = cairosvg.svg2png(bytestring=svg, output_width=18, output_height=18)
    except Exception as exc:
        raise SmokeError(f"tiny renderer failed: {type(exc).__name__}") from exc
    if not png.startswith(b"\x89PNG\r\n\x1a\n"):
        raise SmokeError("tiny renderer did not return a PNG")
    active = int(tensor.path_length[0])
    model_input = [
        float(tensor.path_length[0]) / codec.max_segments,
        float(tensor.opacity[0]) / (len(codec.opacities) + 1),
        float(tensor.fill_opacity[0]) / (len(codec.opacities) + 1),
        float(tensor.stroke_opacity[0]) / (len(codec.opacities) + 1),
        *(float(value) / codec.coordinate_bins for value in tensor.start[0]),
        *(float(value) / len(SegmentType) for value in tensor.segment_type[0, :active]),
        *(
            float(value) / codec.coordinate_bins
            for value in tensor.coordinates[0, :active].reshape(-1)
        ),
    ]
    return (
        {
            "coordinate_bins": codec.coordinate_bins,
            "png_bytes": len(png),
            "png_sha256": hashlib.sha256(png).hexdigest(),
            "svg_bytes": len(svg),
            "svg_sha256": hashlib.sha256(svg).hexdigest(),
            "tensor_fields_stable": True,
        },
        model_input,
    )


def _gpu_checkpoint_smoke(model_input: list[float]) -> tuple[dict[str, Any], bytes]:
    try:
        import torch
    except Exception as exc:
        raise SmokeError(f"PyTorch import failed: {type(exc).__name__}") from exc
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise SmokeError("CUDA GPU is unavailable")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda:0")
    inputs = torch.tensor([model_input], dtype=torch.float32, device=device)
    model = torch.nn.Sequential(
        torch.nn.Linear(inputs.shape[1], 16),
        torch.nn.GELU(),
        torch.nn.Linear(16, inputs.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    prediction = model(inputs)
    loss = torch.nn.functional.mse_loss(prediction, inputs)
    loss.backward()
    gradient_norm = sum(
        float(parameter.grad.detach().float().norm().item())
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    if not gradient_norm > 0:
        raise SmokeError("backward pass produced no finite nonzero gradients")
    optimizer.step()
    torch.cuda.synchronize(device)
    checkpoint = {
        "codec_input_sha256": hashlib.sha256(
            json.dumps(model_input, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "model": {key: value.detach().cpu() for key, value in model.state_dict().items()},
        "optimizer": optimizer.state_dict(),
        "step": 1,
    }
    buffer = io.BytesIO()
    torch.save(checkpoint, buffer)
    payload = buffer.getvalue()
    if not payload or len(payload) > _MAX_CHECKPOINT_BYTES:
        raise SmokeError("checkpoint is empty or exceeds the tiny smoke bound")
    recovered = torch.load(io.BytesIO(payload), map_location="cpu", weights_only=True)
    if recovered.get("step") != 1 or set(recovered.get("model", {})) != set(checkpoint["model"]):
        raise SmokeError("checkpoint read-back did not preserve model state")
    for key, value in checkpoint["model"].items():
        if not torch.equal(value, recovered["model"][key]):
            raise SmokeError(f"checkpoint read-back changed model tensor {key}")
    return (
        {
            "backward_gradient_norm": gradient_norm,
            "checkpoint_bytes": len(payload),
            "checkpoint_sha256": hashlib.sha256(payload).hexdigest(),
            "cuda_version": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
            "loss": float(loss.detach().item()),
            "pytorch": torch.__version__,
            "step": 1,
        },
        payload,
    )


def _run(config: dict[str, Any]) -> dict[str, Any]:
    workspace = _path(config["workspace_root"], "workspace_root")
    source = _path(config["source_root"], "source_root")
    artifact_root = _path(config["artifact_root"], "artifact_root")
    stage, staged_bytes = _verify_stage(source, config)
    codec_result, model_input = _codec_and_render_smoke()
    gpu_result, checkpoint = _gpu_checkpoint_smoke(model_input)
    checkpoint_hash = hashlib.sha256(checkpoint).hexdigest()
    relative = Path(config["run_id"]) / "smoke" / config["smoke_id"]
    local_checkpoint = workspace / relative / f"checkpoint-{checkpoint_hash}.pt"
    durable_checkpoint = artifact_root / relative / f"checkpoint-{checkpoint_hash}.pt"
    result = {
        "artifact": {
            "bytes": len(checkpoint),
            "path": str(durable_checkpoint),
            "sha256": checkpoint_hash,
            "verified_read_back": True,
        },
        "codec_renderer": codec_result,
        "gpu_checkpoint": gpu_result,
        "operation": "smoke",
        "run_id": config["run_id"],
        "schema_version": 1,
        "smoke_id": config["smoke_id"],
        "stage": stage,
    }
    result_bytes = (json.dumps(result, indent=2, sort_keys=True) + "\n").encode("utf-8")
    if len(result_bytes) > _MAX_RESULT_BYTES:
        raise SmokeError("smoke result exceeds its bounded size")
    local_result = workspace / relative / "result.json"
    durable_result = artifact_root / relative / "result.json"
    existing_bytes = (
        staged_bytes
        + _directory_storage_bytes(workspace / config["run_id"] / "smoke")
        + _directory_storage_bytes(artifact_root / config["run_id"])
    )
    additional_bytes = sum(
        _additional_bytes(path, payload)
        for path, payload in (
            (local_checkpoint, checkpoint),
            (durable_checkpoint, checkpoint),
            (local_result, result_bytes),
            (durable_result, result_bytes),
        )
    )
    if existing_bytes + additional_bytes > config["max_storage_bytes"]:
        raise SmokeError("tiny smoke would exceed the configured storage cap")

    _write_or_verify(local_checkpoint, checkpoint)
    _write_or_verify(durable_checkpoint, checkpoint)
    with durable_checkpoint.open("rb") as source_file:
        recovered = source_file.read(len(checkpoint) + 1)
    if hashlib.sha256(recovered).hexdigest() != checkpoint_hash or recovered != checkpoint:
        raise SmokeError("durable artifact write/read/hash round trip failed")
    _write_or_verify(local_result, result_bytes)
    _write_or_verify(durable_result, result_bytes)
    return result


def main() -> int:
    try:
        config = _validated_config(_parse_args().config)
        result = _run(config)
    except SmokeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
