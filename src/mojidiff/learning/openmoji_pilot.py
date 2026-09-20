"""Bounded Gate G pilot over the dominant exact OpenMoji packed bucket.

This pilot deliberately keeps topology and style fixed.  It scales the selected
factorized geometry denoiser from four icons to family-disjoint OpenMoji train and
validation rows, adds structured group/subgroup conditioning, and enforces exact
locked-path editing.  It is not an unconditional-generation result.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Any, cast

import numpy as np
import torch
import yaml
from picosvg.svg import SVG

from mojidiff.learning.geometry import (
    GeometryDenoiser,
    _legal_fields,
    break_even_threshold,
    corrupt_factorized_geometry,
    corrupt_marginal_geometry,
    corrupt_path_correlated_geometry,
    edit_mask_accuracy_by_corruption,
    edit_mask_loss_and_accuracy,
    geometry_accuracy_by_corruption,
    geometry_loss_and_accuracy,
    packed_batch,
    predict_clean_geometry,
    role_token_marginals,
)
from mojidiff.learning.tiny_study import (
    _decode_checkpoint,
    _model_hash,
    _save_checkpoint,
)
from mojidiff.representation.codec_study import _palette, _write_bytes_artifact
from mojidiff.representation.normalizer import normalize_svg
from mojidiff.representation.packed import (
    PackedTensorProgram,
    pack_tensor_program,
    validate_packed_tensor_program,
)
from mojidiff.representation.program import CodecConfig, encode_program

SUMMARY_FILENAME = "summary.json"
"""Name of the compact summary a pilot run writes under its report root."""

TRACE_FILENAME = "validation.jsonl"
"""Name of the periodic held-out trace, written only when `eval_every` is set."""


class OpenMojiPilotError(RuntimeError):
    """The bounded Gate G pilot is invalid or failed closed."""


@dataclass(frozen=True)
class PilotRow:
    source_path: str
    source_svg_sha256: str
    hexcode: str
    split: str
    variant_family_id: str
    group: str
    subgroup: str
    selected_representation: str
    bucket: str


@dataclass(frozen=True)
class SelectionPolicy:
    """Held-out checkpoint selection and early stopping.

    The v1 data-scale predecessor trained a fixed 600-step budget and passed its
    held-out loss minimum at step 240, so its reported checkpoint was already
    overfitting.  When this policy is configured the run keeps the checkpoint with the
    lowest held-out loss and stops once `patience_evals` consecutive evaluations fail
    to improve on it by more than `min_delta`.  Configs without the policy behave
    exactly as before.
    """

    objective: str
    patience_evals: int
    min_delta: float


@dataclass(frozen=True)
class _BestCheckpoint:
    step: int
    loss: float
    payload: bytes
    validation: dict[str, float | int | None]


@dataclass(frozen=True)
class OpenMojiPilotConfig:
    version: str
    source_revision: str
    raw_root: Path
    hybrid: Path
    hybrid_sha256: str
    assignments: Path
    assignments_sha256: str
    palette: Path
    palette_sha256: str
    style_summary: Path
    style_summary_sha256: str
    style_candidate: str
    report_root: Path
    checkpoint_root: Path
    expected_icons: int
    bucket: str
    expected_bucket_icons: int
    max_paths: int
    max_segments: int
    total_segment_slots: int
    d_model: int
    heads: int
    layers: int
    feedforward: int
    train_samples: int
    validation_samples: int
    batch_size: int
    steps: int
    learning_rate: float
    corruption_probability: float
    seed: int
    device: str
    eval_every: int = 0
    selection_policy: SelectionPolicy | None = None
    corruption_probability_max: float | None = None
    """Upper end of a per-example corruption range; `corruption_probability` is the low end.

    The corruption sweep found the denoiser's output near-independent of its input: it
    trained at one fixed level, was never told the level, and learned a prior instead of
    a conditional denoiser. Sampling a level per example and telling the model what it
    is are the two halves of the smallest fix. `None` keeps the single fixed level, so
    every earlier config is untouched.
    """
    noise_level_features: int = 0
    """Sinusoidal conditioning features for the corruption level; 0 disables it."""
    evaluation_corruption_probability: float | None = None
    """Level the held-out evaluation uses. Defaults to `corruption_probability`."""
    corruption_process: str = "factorized"
    """Which fixed-topology corruption process to train and evaluate against.

    Gate F selected `factorized` on four-icon fixtures where held-out recovery was
    memorization. A training-free probe since found path-correlated corruption markedly
    more identifiable - AUC 0.9333 against 0.7698 - so the selection is being re-run at
    corpus scale. Defaults to `factorized`, so every earlier config is unchanged.
    """
    pool_loss_over_fields: bool = False
    """Weight every legal field equally in the loss instead of averaging over groups.

    The group average gives the four single-field QUAD groups 4/13 of the loss against
    40,004 other fields, and makes held-out loss - the selection and early-stopping
    signal - roughly 31% four individual fields. Opt-in so v1 through v7 stay exactly
    reproducible.
    """
    derive_decision_threshold: bool = False
    """Derive the decode threshold from a training-split estimate instead of using argmax.

    Changing a field only pays when the model is confident past `1 / (1 + q)`, where `q`
    is the value head's accuracy on genuinely corrupted fields. Plain argmax uses 0.5
    regardless, which makes the model edit far more than pays. Estimating `q` on TRAIN
    icons keeps the threshold off the data it is reported on.
    """
    value_loss_weight: float = 1.0
    """Scale on the value term of the edit-mask loss.

    The keep and value objectives share an encoder and are unequal at chance - log 2
    nats against log 417, a factor of 8.7 - so an unweighted sum lets the value task
    dominate the representation. Measured cost to the detector: 1.064 of lift.
    """
    metric_coordinates: int = 0
    """Fourier octaves for the metric coordinate encoding; 0 keeps the token tables.

    Coordinates live on a quarter-unit lattice but a token table encodes them as
    unordered categories, so the subtraction the continuity statistic performs has no
    operand the encoder can use. Setting this decodes each token to its view-unit value
    with the role-correct affine map and projects Fourier features of it, replacing the
    categorical tables entirely - which makes the model strictly smaller.
    """
    mask_padding: bool = False
    """Hide typed padding slots from attention; about 29% of the sequence."""
    detection_only: bool = False
    """Train only the keep head, dropping the value term from the loss.

    The value objective is a 289- or 417-class problem against the keep head's 2-class
    one, and they share an encoder, so the encoder is shaped almost entirely by the
    value task. This asks whether the detection signal is learnable at all when nothing
    competes for the representation.
    """
    edit_mask: bool = False
    """Predict a per-field keep-or-change decision and copy the input where it says keep.

    Without it, representing "leave this field alone" costs a full reconstruction through
    a 289- or 417-way softmax, so the identity policy - which outscores every trained
    model so far - is expensive for the model to express. Off by default.
    """
    slot_binding: bool = False
    """Bind each coordinate value to the slot it occupies in its segment.

    Without it the encoder sums six lookups from one shared table, so a segment is an
    unordered bag of its coordinates and the model cannot tell which one held which
    value. Off by default, so every earlier checkpoint still loads.
    """

    @property
    def evaluation_probability(self) -> float:
        if self.evaluation_corruption_probability is not None:
            return self.evaluation_corruption_probability
        return self.corruption_probability


def load_openmoji_pilot_config(path: Path) -> OpenMojiPilotConfig:
    root = _mapping(yaml.safe_load(path.read_bytes()), "root")
    if root.get("schema_version") != 1:
        raise OpenMojiPilotError("openmoji pilot schema_version must be 1")
    inputs = _mapping(root.get("inputs"), "inputs")
    selection = _mapping(root.get("selection"), "selection")
    codec = _mapping(root.get("codec"), "codec")
    model = _mapping(root.get("model"), "model")
    training = _mapping(root.get("training"), "training")
    result = OpenMojiPilotConfig(
        version=_string(root, "study_version"),
        source_revision=_git_revision(_string(root, "source_revision")),
        raw_root=Path(_string(inputs, "raw_root")),
        hybrid=Path(_string(inputs, "hybrid")),
        hybrid_sha256=_sha256(_string(inputs, "hybrid_sha256")),
        assignments=Path(_string(inputs, "assignments")),
        assignments_sha256=_sha256(_string(inputs, "assignments_sha256")),
        palette=Path(_string(inputs, "palette")),
        palette_sha256=_sha256(_string(inputs, "palette_sha256")),
        style_summary=Path(_string(inputs, "style_summary")),
        style_summary_sha256=_sha256(_string(inputs, "style_summary_sha256")),
        style_candidate=_string(inputs, "style_candidate"),
        report_root=Path(_string(root, "report_root")),
        checkpoint_root=Path(_string(root, "checkpoint_root")),
        expected_icons=_positive_int(inputs.get("expected_icons"), "expected_icons"),
        bucket=_string(selection, "bucket"),
        expected_bucket_icons=_positive_int(
            selection.get("expected_bucket_icons"), "expected_bucket_icons"
        ),
        max_paths=_positive_int(codec.get("max_paths"), "max_paths"),
        max_segments=_positive_int(codec.get("max_segments"), "max_segments"),
        total_segment_slots=_positive_int(
            codec.get("total_segment_slots"), "total_segment_slots"
        ),
        d_model=_positive_int(model.get("d_model"), "d_model"),
        heads=_positive_int(model.get("heads"), "heads"),
        layers=_positive_int(model.get("layers"), "layers"),
        feedforward=_positive_int(model.get("feedforward"), "feedforward"),
        train_samples=_positive_int(training.get("train_samples"), "train_samples"),
        validation_samples=_positive_int(
            training.get("validation_samples"), "validation_samples"
        ),
        batch_size=_positive_int(training.get("batch_size"), "batch_size"),
        steps=_positive_int(training.get("steps"), "steps"),
        learning_rate=_positive_float(training.get("learning_rate"), "learning_rate"),
        corruption_probability=_probability(
            training.get("corruption_probability"), "corruption_probability"
        ),
        seed=_nonnegative_int(training.get("seed"), "seed"),
        device=_string(training, "device"),
        eval_every=_nonnegative_int(training.get("eval_every", 0), "eval_every"),
        selection_policy=_selection_policy(training.get("selection_policy")),
        corruption_probability_max=_optional_probability(
            training.get("corruption_probability_max"), "corruption_probability_max"
        ),
        noise_level_features=_nonnegative_int(
            training.get("noise_level_features", 0), "noise_level_features"
        ),
        evaluation_corruption_probability=_optional_probability(
            training.get("evaluation_corruption_probability"),
            "evaluation_corruption_probability",
        ),
        corruption_process=_corruption_process(training.get("corruption_process", "factorized")),
        slot_binding=_flag(model.get("slot_binding", False), "slot_binding"),
        edit_mask=_flag(model.get("edit_mask", False), "edit_mask"),
        detection_only=_flag(training.get("detection_only", False), "detection_only"),
        pool_loss_over_fields=_flag(
            training.get("pool_loss_over_fields", False), "pool_loss_over_fields"
        ),
        mask_padding=_flag(model.get("mask_padding", False), "mask_padding"),
        metric_coordinates=_nonnegative_int(
            model.get("metric_coordinates", 0), "metric_coordinates"
        ),
        value_loss_weight=_positive_float(
            training.get("value_loss_weight", 1.0), "value_loss_weight"
        ),
        derive_decision_threshold=_flag(
            training.get("derive_decision_threshold", False), "derive_decision_threshold"
        ),
    )
    if result.d_model % result.heads:
        raise OpenMojiPilotError("model.d_model must be divisible by model.heads")
    if result.batch_size > result.train_samples:
        raise OpenMojiPilotError("batch_size cannot exceed train_samples")
    if result.device not in {"auto", "cpu", "cuda"}:
        raise OpenMojiPilotError("training.device must be auto, cpu, or cuda")
    if result.eval_every > result.steps:
        raise OpenMojiPilotError("training.eval_every cannot exceed training.steps")
    if result.derive_decision_threshold and not result.edit_mask:
        raise OpenMojiPilotError("derive_decision_threshold requires model.edit_mask")
    if result.detection_only and not result.edit_mask:
        raise OpenMojiPilotError("detection_only requires model.edit_mask")
    if result.selection_policy is not None and not result.eval_every:
        raise OpenMojiPilotError("selection_policy requires a nonzero training.eval_every")
    if (
        result.corruption_probability_max is not None
        and result.corruption_probability_max <= result.corruption_probability
    ):
        raise OpenMojiPilotError(
            "corruption_probability_max must exceed training.corruption_probability"
        )
    return result


CorruptionOperator = Callable[
    [PackedTensorProgram, CodecConfig, float, "np.random.Generator"], PackedTensorProgram
]

CORRUPTION_PROCESSES: dict[str, object] = {
    "factorized": corrupt_factorized_geometry,
    "path_correlated": corrupt_path_correlated_geometry,
    "marginal": corrupt_marginal_geometry,
}
"""Selectable corruption processes.

Whole-path replacement is absent: it needs a donor pool and a coverage audit, and under
single-donor compatibility it replaced 32 of 13,128 held-out fields.

`marginal` exists because uniform replacement leaks. A zero-parameter detector that
knows only the corpus marginal reaches 2.84 precision lift under `factorized` and 0.77
under `marginal`, while the genuine local-continuity signal survives at 1.89 - so
`factorized` makes detection largely a density test rather than a geometric one. It
takes an extra marginal argument, which `corruption_operator` binds.
"""

CORRUPTION_LABELS = {
    "factorized": "factorized_role_uniform_geometry",
    "path_correlated": "path_correlated_geometry_blocks",
    "marginal": "factorized_marginal_respecting_geometry",
}
"""Reported names. `factorized` keeps the string v1 through v6 wrote, so their summaries
stay byte-identical."""


def corruption_operator(
    config: OpenMojiPilotConfig,
    codec: CodecConfig,
    train_programs: list[PackedTensorProgram],
) -> CorruptionOperator:
    """Bind the configured corruption process, estimating a marginal when it needs one.

    The marginal is estimated from the TRAIN programs only, so no held-out information
    reaches the corruption applied to held-out icons.
    """

    if config.corruption_process != "marginal":
        return cast(CorruptionOperator, CORRUPTION_PROCESSES[config.corruption_process])
    marginal = role_token_marginals(train_programs, codec)

    def bound(
        clean: PackedTensorProgram,
        bound_codec: CodecConfig,
        probability: float,
        rng: np.random.Generator,
    ) -> PackedTensorProgram:
        return corrupt_marginal_geometry(clean, bound_codec, probability, rng, marginal)

    return bound


def _corruption_process(value: object) -> str:
    if not isinstance(value, str) or value not in CORRUPTION_PROCESSES:
        raise OpenMojiPilotError(
            f"corruption_process must be one of {sorted(CORRUPTION_PROCESSES)}"
        )
    return value


def _flag(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise OpenMojiPilotError(f"{field} must be a boolean")
    return value


def _optional_probability(value: object, field: str) -> float | None:
    if value is None:
        return None
    return _probability(value, field)


def _selection_policy(value: object) -> SelectionPolicy | None:
    """Parse the optional held-out selection and early-stopping policy."""

    if value is None:
        return None
    mapping = _mapping(value, "selection_policy")
    objective = _string(mapping, "objective")
    if objective != "held_out_loss":
        raise OpenMojiPilotError("selection_policy.objective must be held_out_loss")
    min_delta = mapping.get("min_delta", 0.0)
    if isinstance(min_delta, bool) or not isinstance(min_delta, (int, float)) or min_delta < 0:
        raise OpenMojiPilotError("selection_policy.min_delta must be nonnegative")
    return SelectionPolicy(
        objective=objective,
        patience_evals=_positive_int(mapping.get("patience_evals"), "patience_evals"),
        min_delta=float(min_delta),
    )


def load_pilot_index(
    config: OpenMojiPilotConfig,
) -> tuple[dict[str, tuple[PilotRow, ...]], dict[str, int], dict[str, int]]:
    """Verify and join the immutable hybrid and capacity-assignment ledgers."""

    _verify(config.hybrid, config.hybrid_sha256, "hybrid ledger")
    _verify(config.assignments, config.assignments_sha256, "bucket assignments")
    hybrid_rows = _jsonl(config.hybrid)
    assignment_rows = _jsonl(config.assignments)
    if len(hybrid_rows) != config.expected_icons or len(assignment_rows) != config.expected_icons:
        raise OpenMojiPilotError("input ledgers do not have the expected complete-corpus size")

    assignments: dict[str, str] = {}
    for item in assignment_rows:
        source_path = _safe_source_path(_string(item, "source_path"))
        if source_path in assignments:
            raise OpenMojiPilotError(f"duplicate assignment: {source_path}")
        assignments[source_path] = _string(item, "bucket")

    rows: list[PilotRow] = []
    seen: set[str] = set()
    family_splits: dict[str, set[str]] = {}
    for item in hybrid_rows:
        if item.get("source_revision") != config.source_revision:
            raise OpenMojiPilotError("hybrid source revision mismatch")
        source_path = _safe_source_path(_string(item, "source_path"))
        if source_path in seen or source_path not in assignments:
            raise OpenMojiPilotError(f"hybrid/assignment join is not one-to-one: {source_path}")
        seen.add(source_path)
        split = _string(item, "split")
        if split not in {"primary/train", "primary/validation", "primary/test"}:
            raise OpenMojiPilotError(f"unexpected primary split: {split}")
        family = _string(item, "split_family_cluster")
        family_splits.setdefault(family, set()).add(split)
        representation = _string(item, "selected_representation")
        if representation not in {"semantic", "outlined"}:
            raise OpenMojiPilotError(f"unsupported selected representation: {representation}")
        rows.append(
            PilotRow(
                source_path=source_path,
                source_svg_sha256=_sha256(_string(item, "source_svg_sha256")),
                hexcode=_string(item, "hexcode"),
                split=split,
                variant_family_id=family,
                group=_string(item, "group"),
                subgroup=_string(item, "subgroup"),
                selected_representation=representation,
                bucket=assignments[source_path],
            )
        )
    if set(assignments) != seen:
        raise OpenMojiPilotError("bucket assignments contain rows absent from the hybrid ledger")
    if any(len(splits) != 1 for splits in family_splits.values()):
        raise OpenMojiPilotError("variant-family leakage crosses primary splits")

    selected = [row for row in rows if row.bucket == config.bucket]
    if len(selected) != config.expected_bucket_icons:
        raise OpenMojiPilotError("selected packed bucket count does not match its pinned census")
    groups = {name: index + 1 for index, name in enumerate(sorted({row.group for row in rows}))}
    subgroups = {
        name: index + 1 for index, name in enumerate(sorted({row.subgroup for row in rows}))
    }
    by_split = {
        split: tuple(sorted((row for row in selected if row.split == split), key=_row_key))
        for split in ("primary/train", "primary/validation", "primary/test")
    }
    return by_split, groups, subgroups


def run_openmoji_pilot(config: OpenMojiPilotConfig, config_path: Path) -> dict[str, Any]:
    """Run a bounded one- or few-step pipeline smoke and write compact evidence."""

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config.seed)
    _verify(config.palette, config.palette_sha256, "palette")
    _verify(config.style_summary, config.style_summary_sha256, "style summary")
    by_split, groups, subgroups = load_pilot_index(config)
    train_rows = _select_rows(by_split["primary/train"], config.train_samples, config.seed)
    validation_rows = _select_rows(
        by_split["primary/validation"], config.validation_samples, config.seed + 1
    )
    codec = _selected_codec(config)
    train_programs = [_load_program(row, config, codec) for row in train_rows]
    validation_programs = [_load_program(row, config, codec) for row in validation_rows]
    device = _device(config.device)
    corrupt = corruption_operator(config, codec, train_programs)
    model = _new_model(config, codec, groups, subgroups).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    def _validate() -> dict[str, float | int | None]:
        return _evaluate(
            model,
            validation_rows,
            validation_programs,
            groups,
            subgroups,
            codec,
            config,
            device,
            corrupt,
        )

    # Metric-only tracing. `_evaluate` draws its corruption from an independent numpy
    # generator under `no_grad`, and the denoiser has no dropout, so tracing cannot
    # perturb the training stream. Configs without `eval_every` behave exactly as before.
    trace: list[dict[str, Any]] = []
    untrained: dict[str, float | int | None] | None = None
    if config.eval_every:
        untrained = _validate()
        trace.append({"step": 0, "trained": False, **untrained})

    policy = config.selection_policy
    best: _BestCheckpoint | None = None
    stale_evals = 0
    stopped_early = False

    def _consider(step: int, result: dict[str, float | int | None]) -> bool:
        """Record a held-out evaluation against the selection policy.

        Returns True when `patience_evals` consecutive evaluations have failed to
        improve the objective, which is the caller's signal to stop training.  Step 0
        is deliberately never considered: an untrained model is not a selectable one.
        """

        nonlocal best, stale_evals, stopped_early
        assert policy is not None
        loss = float(cast(float, result["loss"]))
        if best is None or loss < best.loss - policy.min_delta:
            best = _BestCheckpoint(
                step=step,
                loss=loss,
                payload=_save_checkpoint(model, optimizer, step),
                validation=dict(result),
            )
            stale_evals = 0
            return False
        stale_evals += 1
        if stale_evals >= policy.patience_evals:
            stopped_early = True
            return True
        return False

    metrics: list[dict[str, Any]] = []
    completed_steps = 0
    last_traced_step = 0
    for step in range(1, config.steps + 1):
        indices = [
            (step * config.batch_size + offset) % len(train_programs)
            for offset in range(config.batch_size)
        ]
        clean = [train_programs[index] for index in indices]
        generators = [
            np.random.default_rng(config.seed + step * 100_000 + index) for index in indices
        ]
        # Draw the level first, then corrupt with the same generator, so the whole
        # example is a deterministic function of (seed, step, index).
        levels = [sample_corruption_level(config, generator) for generator in generators]
        noisy = [
            corrupt(program, codec, level, generator)
            for program, level, generator in zip(clean, levels, generators, strict=True)
        ]
        condition = _condition(
            [train_rows[index] for index in indices],
            groups,
            subgroups,
            device,
            levels if config.noise_level_features else None,
        )
        clean_batch = packed_batch(clean, device)
        optimizer.zero_grad(set_to_none=True)
        noisy_batch = packed_batch(noisy, device)
        start_logits, coordinate_logits, keep = model.forward_with_edits(noisy_batch, condition)
        if keep is not None:
            loss, counts = edit_mask_loss_and_accuracy(
                (start_logits, coordinate_logits),
                keep,
                noisy_batch,
                clean_batch,
                codec,
                detection_only=config.detection_only,
                pool_over_fields=config.pool_loss_over_fields,
                value_loss_weight=config.value_loss_weight,
            )
        else:
            loss, counts = geometry_loss_and_accuracy(
                (start_logits, coordinate_logits),
                clean_batch,
                codec,
                pool_over_fields=config.pool_loss_over_fields,
            )
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        metrics.append(
            {
                "step": step,
                "loss": float(loss.detach()),
                "train_token_accuracy": counts["correct"] / counts["total"],
            }
        )
        completed_steps = step
        if config.eval_every and step % config.eval_every == 0 and step != config.steps:
            interim = _validate()
            trace.append({"step": step, "trained": True, **interim})
            last_traced_step = step
            if policy is not None and _consider(step, interim):
                break

    final_step_validation = _validate()
    if config.eval_every and last_traced_step != completed_steps:
        trace.append({"step": completed_steps, "trained": True, **final_step_validation})
        if policy is not None:
            _consider(completed_steps, final_step_validation)

    validation = final_step_validation
    selected_step = completed_steps
    if policy is not None:
        if best is None:
            raise OpenMojiPilotError("selection policy recorded no held-out evaluation")
        # Restore the selected parameters in place so the locked-path check, the
        # written checkpoint, and the reported held-out numbers all describe one model.
        selected = _decode_checkpoint(best.payload)
        model.load_state_dict(selected["model"])
        optimizer.load_state_dict(selected["optimizer"])
        selected_step = best.step
        validation = _validate()
        if validation != best.validation:
            raise OpenMojiPilotError("restored selected checkpoint did not reproduce its metrics")

    decision_threshold = 0.5
    value_accuracy_train: float | None = None
    if config.derive_decision_threshold:
        value_accuracy_train = _train_value_accuracy(
            model, train_rows, train_programs, groups, subgroups, codec, config, device, corrupt
        )
        decision_threshold = break_even_threshold(value_accuracy_train)
        validation = _evaluate(
            model,
            validation_rows,
            validation_programs,
            groups,
            subgroups,
            codec,
            config,
            device,
            corrupt,
            decision_threshold,
        )

    lock_verified = _verify_locked_path(
        model,
        validation_rows[0],
        validation_programs[0],
        groups,
        subgroups,
        codec,
        config,
        device,
    )
    checkpoint = (
        best.payload if best is not None else _save_checkpoint(model, optimizer, config.steps)
    )
    restored = _new_model(config, codec, groups, subgroups).to(device)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=config.learning_rate)
    document = _decode_checkpoint(checkpoint)
    restored.load_state_dict(document["model"])
    restored_optimizer.load_state_dict(document["optimizer"])
    checkpoint_round_trip = _model_hash(restored) == _model_hash(model)
    if int(document["step"]) != selected_step or not checkpoint_round_trip:
        raise OpenMojiPilotError("canonical checkpoint did not restore the trained model")

    checkpoint_sha256 = hashlib.sha256(checkpoint).hexdigest()
    metrics_payload = _jsonl_bytes(metrics)
    summary = {
        "schema_version": 1,
        "study_version": config.version,
        "scope": "dominant-bucket fixed-topology geometry pilot; not unconditional generation",
        "config_sha256": _file_sha256(config_path),
        "device": str(device),
        "torch_version": str(torch.__version__),
        "cuda_version": torch.version.cuda,
        "deterministic_algorithms": True,
        "corruption": CORRUPTION_LABELS[config.corruption_process],
        "bucket": config.bucket,
        "bucket_icons": config.expected_bucket_icons,
        "selected_train_rows": [row.source_path for row in train_rows],
        "selected_validation_rows": [row.source_path for row in validation_rows],
        "group_vocabulary_size": len(groups) + 1,
        "subgroup_vocabulary_size": len(subgroups) + 1,
        "model_parameters": parameter_count,
        "steps": config.steps,
        "final_train": metrics[-1],
        "validation": validation,
        "locked_path_exact": lock_verified,
        "checkpoint_bytes": len(checkpoint),
        "checkpoint_sha256": checkpoint_sha256,
        "checkpoint_round_trip": checkpoint_round_trip,
        "metrics_sha256": hashlib.sha256(metrics_payload).hexdigest(),
    }
    if config.eval_every:
        trace_payload = _jsonl_bytes(trace)
        summary["eval_every"] = config.eval_every
        summary["validation_untrained"] = untrained
        summary["validation_trace_sha256"] = hashlib.sha256(trace_payload).hexdigest()
        _write_bytes_artifact(config.report_root / TRACE_FILENAME, trace_payload)
    if config.derive_decision_threshold:
        # Only present when the threshold is derived, so every earlier run's summary is
        # byte-identical to what it wrote.
        summary["decision_threshold"] = decision_threshold
        summary["train_value_accuracy"] = value_accuracy_train
    if policy is not None:
        # `steps` above stays the declared cap; `completed_steps` is what actually ran.
        # `validation` describes the selected checkpoint, not the last trained model.
        summary["selection"] = {
            "objective": policy.objective,
            "patience_evals": policy.patience_evals,
            "min_delta": policy.min_delta,
            "selected_step": selected_step,
            "selected_held_out_loss": None if best is None else best.loss,
            "completed_steps": completed_steps,
            "stopped_early": stopped_early,
            "evals_without_improvement": stale_evals,
        }
        summary["validation_final_step"] = final_step_validation
    _write_bytes_artifact(config.report_root / "metrics.jsonl", metrics_payload)
    _write_bytes_artifact(config.report_root / SUMMARY_FILENAME, _json_bytes(summary))
    _write_bytes_artifact(config.report_root / "README.md", _markdown(summary).encode())
    _write_bytes_artifact(config.checkpoint_root / "checkpoint.zip", checkpoint)
    return summary


def _new_model(
    config: OpenMojiPilotConfig,
    codec: CodecConfig,
    groups: dict[str, int],
    subgroups: dict[str, int],
) -> GeometryDenoiser:
    return GeometryDenoiser(
        codec,
        config.total_segment_slots,
        d_model=config.d_model,
        heads=config.heads,
        layers=config.layers,
        feedforward=config.feedforward,
        group_vocab_size=len(groups) + 1,
        subgroup_vocab_size=len(subgroups) + 1,
        noise_level_features=config.noise_level_features,
        slot_binding=config.slot_binding,
        edit_mask=config.edit_mask,
        mask_padding=config.mask_padding,
        metric_coordinates=config.metric_coordinates,
    )


def sample_corruption_level(config: OpenMojiPilotConfig, rng: np.random.Generator) -> float:
    """The per-example corruption level, or the single fixed one when no range is set."""

    if config.corruption_probability_max is None:
        return config.corruption_probability
    return float(
        rng.uniform(config.corruption_probability, config.corruption_probability_max)
    )


def _evaluate(
    model: GeometryDenoiser,
    rows: tuple[PilotRow, ...],
    clean: list[PackedTensorProgram],
    groups: dict[str, int],
    subgroups: dict[str, int],
    codec: CodecConfig,
    config: OpenMojiPilotConfig,
    device: torch.device,
    corrupt: Callable[
        [PackedTensorProgram, CodecConfig, float, np.random.Generator], PackedTensorProgram
    ],
    threshold: float = 0.5,
) -> dict[str, float | int | None]:
    probability = config.evaluation_probability
    noisy = [
        corrupt(
            program,
            codec,
            probability,
            np.random.default_rng(config.seed + 9_000_000 + index),
        )
        for index, program in enumerate(clean)
    ]
    levels = [probability] * len(clean)
    with torch.no_grad():
        noisy_batch = packed_batch(noisy, device)
        clean_batch = packed_batch(clean, device)
        start_logits, coordinate_logits, keep = model.forward_with_edits(
            noisy_batch,
            _condition(
                rows, groups, subgroups, device, levels if config.noise_level_features else None
            ),
        )
        value_logits = (start_logits, coordinate_logits)
        if keep is not None:
            loss, counts = edit_mask_loss_and_accuracy(
                value_logits,
                keep,
                noisy_batch,
                clean_batch,
                codec,
                detection_only=config.detection_only,
                pool_over_fields=config.pool_loss_over_fields,
                value_loss_weight=config.value_loss_weight,
                threshold=threshold,
            )
            split = edit_mask_accuracy_by_corruption(
                value_logits, keep, noisy_batch, clean_batch, codec, threshold=threshold
            )
        else:
            loss, counts = geometry_loss_and_accuracy(
                value_logits, clean_batch, codec, pool_over_fields=config.pool_loss_over_fields
            )
            split = geometry_accuracy_by_corruption(
                value_logits, noisy_batch, clean_batch, codec
            )
    return {
        "loss": float(loss),
        "accuracy": counts["correct"] / counts["total"],
        "changed_accuracy": (
            split["changed"]["correct"] / split["changed"]["total"]
            if split["changed"]["total"]
            else None
        ),
        "changed_total": split["changed"]["total"],
        "retained_accuracy": (
            split["retained"]["correct"] / split["retained"]["total"]
            if split["retained"]["total"]
            else None
        ),
        "retained_total": split["retained"]["total"],
    }


def _train_value_accuracy(
    model: GeometryDenoiser,
    rows: tuple[PilotRow, ...],
    programs: list[PackedTensorProgram],
    groups: dict[str, int],
    subgroups: dict[str, int],
    codec: CodecConfig,
    config: OpenMojiPilotConfig,
    device: torch.device,
    corrupt: Callable[
        [PackedTensorProgram, CodecConfig, float, np.random.Generator], PackedTensorProgram
    ],
    icons: int = 128,
) -> float:
    """The value head's exact-token accuracy on corrupted TRAIN fields.

    This is the `q` the break-even threshold is derived from. It is measured on training
    icons, under a corruption draw disjoint from the held-out one, so the threshold the
    model decodes with never touches the data it is reported on.
    """

    count = min(icons, len(programs))
    subset = programs[:count]
    subset_rows = rows[:count]
    noisy = [
        corrupt(
            program,
            codec,
            config.evaluation_probability,
            np.random.default_rng(config.seed + 11_000_000 + index),
        )
        for index, program in enumerate(subset)
    ]
    noisy_batch = packed_batch(noisy, device)
    clean_batch = packed_batch(subset, device)
    levels = [config.evaluation_probability] * count
    with torch.no_grad():
        start_logits, coordinate_logits, keep = model.forward_with_edits(
            noisy_batch,
            _condition(
                subset_rows,
                groups,
                subgroups,
                device,
                levels if config.noise_level_features else None,
            ),
        )
    correct = 0
    total = 0
    for value_logits, _, targets, noisy_tokens in _legal_fields(
        (start_logits, coordinate_logits), keep, noisy_batch, clean_batch, codec
    ):
        if not targets.numel():
            continue
        changed = noisy_tokens != targets
        if not bool(changed.any()):
            continue
        predicted = value_logits.argmax(dim=-1) + 1
        correct += int(((predicted == targets) & changed).sum().item())
        total += int(changed.sum().item())
    return correct / total if total else 0.0


def _verify_locked_path(
    model: GeometryDenoiser,
    row: PilotRow,
    clean: PackedTensorProgram,
    groups: dict[str, int],
    subgroups: dict[str, int],
    codec: CodecConfig,
    config: OpenMojiPilotConfig,
    device: torch.device,
) -> bool:
    locks = np.zeros((codec.max_paths,), dtype=np.bool_)
    locks[0] = True
    # Deliberately factorized at probability 1.0 regardless of the training process.
    # This is the structural-safety check, not a training sample: corrupting every
    # unlocked field independently is the strongest test that a locked path survives
    # both corruption and prediction, and path-correlated corruption takes no lock
    # argument.
    noisy = corrupt_factorized_geometry(
        clean,
        codec,
        1.0,
        np.random.default_rng(config.seed + 10_000_000),
        locked_paths=locks,
    )
    with torch.no_grad():
        start_logits, coordinate_logits, keep = model.forward_with_edits(
            packed_batch([noisy], device),
            _condition(
                (row,), groups, subgroups, device, [1.0] if config.noise_level_features else None
            ),
        )
    prediction = predict_clean_geometry(
        noisy, (start_logits, coordinate_logits), codec, locked_paths=locks, keep=keep
    )
    length = int(clean.path_length[0])
    exact = bool(
        np.array_equal(prediction.start[0], clean.start[0])
        and np.array_equal(prediction.coordinates[:length], clean.coordinates[:length])
    )
    if not exact:
        raise OpenMojiPilotError("locked path changed during corruption or prediction")
    validate_packed_tensor_program(prediction, codec, config.total_segment_slots)
    return True


def _condition(
    rows: tuple[PilotRow, ...] | list[PilotRow],
    groups: dict[str, int],
    subgroups: dict[str, int],
    device: torch.device,
    levels: list[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Structured conditioning. `levels` is supplied only when the model expects it,
    because `GeometryDenoiser` rejects a condition whose keys it did not ask for."""

    condition = {
        "group": torch.tensor([groups[row.group] for row in rows], dtype=torch.long, device=device),
        "subgroup": torch.tensor(
            [subgroups[row.subgroup] for row in rows], dtype=torch.long, device=device
        ),
    }
    if levels is not None:
        condition["noise_level"] = torch.tensor(
            levels, dtype=torch.float32, device=device
        )
    return condition


def _load_program(
    row: PilotRow, config: OpenMojiPilotConfig, codec: CodecConfig
) -> PackedTensorProgram:
    source = (config.raw_root / row.source_path).read_bytes()
    if hashlib.sha256(source).hexdigest() != row.source_svg_sha256:
        raise OpenMojiPilotError(f"source hash mismatch: {row.source_path}")
    normalized_source = source if row.selected_representation == "semantic" else _outline(source)
    dense, report = encode_program(normalize_svg(normalized_source).program, codec)
    if report.dropped_contours or report.dropped_segments or report.clamped_coordinates:
        raise OpenMojiPilotError(f"selected bucket requires projection: {row.source_path}")
    packed = pack_tensor_program(dense, codec, config.total_segment_slots)
    validate_packed_tensor_program(packed, codec, config.total_segment_slots)
    return packed


def _selected_codec(config: OpenMojiPilotConfig) -> CodecConfig:
    style = _mapping(json.loads(config.style_summary.read_bytes()), "style summary")
    candidate = _mapping(
        _mapping(style.get("width_vocabulary_analytics"), "width analytics").get(
            config.style_candidate
        ),
        "style candidate",
    )
    widths = tuple(float(value) for value in _sequence(candidate.get("values"), "widths"))
    return CodecConfig(
        max_paths=config.max_paths,
        max_segments=config.max_segments,
        coordinate_bins=289,
        control_coordinate_bins=417,
        control_coordinate_min=-8.0,
        control_coordinate_max=96.0,
        palette=_palette(config.palette),
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


def _select_rows(rows: tuple[PilotRow, ...], count: int, seed: int) -> tuple[PilotRow, ...]:
    if count > len(rows):
        raise OpenMojiPilotError("sample request exceeds available split rows")
    return tuple(
        sorted(
            rows,
            key=lambda row: hashlib.sha256(
                f"{seed}\0{row.source_path}".encode()
            ).digest(),
        )[:count]
    )


def _device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise OpenMojiPilotError("CUDA was required but is unavailable")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def _outline(source: bytes) -> bytes:
    try:
        value = SVG.fromstring(source).topicosvg().tostring()  # type: ignore[no-untyped-call]
    except Exception as exc:
        raise OpenMojiPilotError(f"PicoSVG outline failed: {type(exc).__name__}") from exc
    return value.encode("utf-8") if isinstance(value, str) else value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line:
            raise OpenMojiPilotError(f"blank JSONL row in {path}:{line_number}")
        try:
            rows.append(_mapping(json.loads(line), f"{path}:{line_number}"))
        except json.JSONDecodeError as exc:
            raise OpenMojiPilotError(f"invalid JSONL row in {path}:{line_number}") from exc
    return rows


def _safe_source_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or ".." in path.parts
        or path.suffix != ".svg"
        or path.as_posix() != value
    ):
        raise OpenMojiPilotError(f"unsafe source path: {value}")
    return value


def _row_key(row: PilotRow) -> tuple[str, str]:
    return row.source_path, row.hexcode


def _verify(path: Path, expected: str, label: str) -> None:
    if _file_sha256(path) != expected:
        raise OpenMojiPilotError(f"{label} hash mismatch")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return b"".join(_json_bytes(row) for row in rows)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _markdown(summary: dict[str, Any]) -> str:
    validation = cast(dict[str, Any], summary["validation"])
    return "\n".join(
        [
            "# OpenMoji Gate G dominant-bucket pipeline smoke",
            "",
            f"Device: `{summary['device']}`; parameters: {summary['model_parameters']:,}; "
            f"steps: {summary['steps']}.",
            "",
            f"Validation accuracy: {validation['accuracy']:.4f}; changed-token accuracy: "
            f"{validation['changed_accuracy']:.4f}.",
            "",
            f"Canonical checkpoint round trip: {summary['checkpoint_round_trip']}; "
            f"locked-path exactness: {summary['locked_path_exact']}.",
            "",
            *_selection_markdown(summary),
            "This is a bounded fixed-topology geometry pipeline smoke over the dominant exact "
            "packed bucket. It is not evidence for unconditional generation.",
            "",
        ]
    )


def _selection_markdown(summary: dict[str, Any]) -> list[str]:
    selection = summary.get("selection")
    if not isinstance(selection, dict):
        return []
    return [
        f"Selected by {selection['objective']} at step {selection['selected_step']} of "
        f"{selection['completed_steps']} run (cap {summary['steps']}); early stop: "
        f"{selection['stopped_early']}. Reported held-out numbers describe that "
        "selected checkpoint, not the last trained step.",
        "",
    ]


def _mapping(value: object, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise OpenMojiPilotError(f"{field} must be a mapping")
    return cast(dict[str, Any], value)


def _sequence(value: object, field: str) -> list[Any]:
    if not isinstance(value, list):
        raise OpenMojiPilotError(f"{field} must be a list")
    return value


def _string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value:
        raise OpenMojiPilotError(f"{field} must be a non-empty string")
    return value


def _sha256(value: str) -> str:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise OpenMojiPilotError("expected a lowercase SHA-256 value")
    return value


def _git_revision(value: str) -> str:
    if not 7 <= len(value) <= 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise OpenMojiPilotError("expected a lowercase hexadecimal Git revision")
    return value


def _positive_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise OpenMojiPilotError(f"{field} must be a positive integer")
    return value


def _nonnegative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise OpenMojiPilotError(f"{field} must be a nonnegative integer")
    return value


def _positive_float(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise OpenMojiPilotError(f"{field} must be positive")
    return float(value)


def _probability(value: object, field: str) -> float:
    result = _positive_float(value, field)
    if result > 1:
        raise OpenMojiPilotError(f"{field} must not exceed 1")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report-root", type=Path)
    parser.add_argument("--checkpoint-root", type=Path)
    args = parser.parse_args()
    config = load_openmoji_pilot_config(args.config)
    if (args.report_root is None) != (args.checkpoint_root is None):
        raise OpenMojiPilotError("report and checkpoint root overrides must be supplied together")
    if args.report_root is not None and args.checkpoint_root is not None:
        config = replace(
            config,
            report_root=args.report_root,
            checkpoint_root=args.checkpoint_root,
        )
    print(json.dumps(run_openmoji_pilot(config, args.config), sort_keys=True))


if __name__ == "__main__":
    main()
