from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.assignment_evidence import (
    assignment_evidence_sequence_from_trainable_frames,
)
from objgauss.core.assignment_losses import assignment_loss_v2_breakdown
from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2State,
    predict_assignment_solver_v2,
    validate_assignment_solver_v2_state,
)
from objgauss.core.assignment_solver_v2_eval import (
    ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA,
    assignment_solver_v2_state_from_checkpoint,
    validate_assignment_solver_v2_checkpoint,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.gaussian_decoder_training import (
    ObjectStateGaussianDecoderState,
    initialize_object_state_gaussian_decoder,
    validate_object_state_gaussian_decoder_state,
)
from objgauss.core.gsplat_training_renderer import evaluate_gsplat_training_renderer_loss
from objgauss.core.object_state import (
    ObjectStateProjection,
    object_state_stability_report,
    project_object_states,
)
from objgauss.core.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_GSPLAT,
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
    TrainableKernelFrame,
    image_target_contract_summary,
)
from objgauss.core.training_renderer import (
    TrainingRendererLossResult,
    evaluate_training_renderer_loss,
)

ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA = (
    "objgauss-assignment-v2-render-joint-validation-v1"
)
_STATUS_PASS = "assignment_v2_renderer_joint_validation_pass"
_STATUS_FAIL = "assignment_v2_renderer_joint_validation_fail"


@dataclass(frozen=True)
class AssignmentV2RendererJointValidationReport:
    checkpoint: dict[str, Any]
    solver_state: AssignmentSolverV2State
    frames: tuple[TrainableKernelFrame, ...]
    decoder_state: ObjectStateGaussianDecoderState
    initial_assignments: tuple[np.ndarray, ...]
    final_assignments: tuple[np.ndarray, ...]
    projections: tuple[ObjectStateProjection, ...]
    initial_renderer_api: TrainingRendererLossResult
    final_renderer_api: TrainingRendererLossResult
    initial_assignment_loss: dict[str, Any]
    final_assignment_loss: dict[str, Any]
    object_state_eval: dict[str, Any]
    checkpoint_roundtrip: dict[str, Any]
    image_renderer: str
    vram_reserve_gb: int
    schema: str = ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA

    @property
    def passed(self) -> bool:
        return (
            self.final_total_loss < self.initial_total_loss
            and self.final_renderer_api.image_render_loss
            < self.initial_renderer_api.image_render_loss
            and self.object_state_eval["status"] == "objectstate_eval_pass"
            and bool(self.checkpoint_roundtrip["pass"])
        )

    @property
    def initial_total_loss(self) -> float:
        return float(
            self.initial_renderer_api.image_render_loss
            + _supervised_loss(self.initial_assignment_loss)
        )

    @property
    def final_total_loss(self) -> float:
        return float(
            self.final_renderer_api.image_render_loss
            + _supervised_loss(self.final_assignment_loss)
        )

    def as_dict(self) -> dict[str, Any]:
        initial_loss = _loss_record(
            iteration=0,
            image_render_loss=self.initial_renderer_api.image_render_loss,
            object_loss=_supervised_loss(self.initial_assignment_loss),
        )
        final_loss = _loss_record(
            iteration=1,
            image_render_loss=self.final_renderer_api.image_render_loss,
            object_loss=_supervised_loss(self.final_assignment_loss),
        )
        payload = {
            "schema": self.schema,
            "kind": "assignment_v2_renderer_joint_validation",
            "status": _STATUS_PASS if self.passed else _STATUS_FAIL,
            "checkpoint_schema": self.checkpoint["schema"],
            "training_schema": self.checkpoint["source"]["training_schema"],
            "decoder_schema": self.decoder_state.schema,
            "slots": int(self.solver_state.config.slots),
            "frame_count": len(self.frames),
            "image_renderer": self.image_renderer,
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "loss_decreased": bool(final_loss["total_loss"] < initial_loss["total_loss"]),
            "image_render_loss_decreased": bool(
                final_loss["image_render_loss"] < initial_loss["image_render_loss"]
            ),
            "object_loss_decreased": bool(final_loss["object_loss"] < initial_loss["object_loss"]),
            "assignment_loss": {
                "initial": self.initial_assignment_loss,
                "final": self.final_assignment_loss,
            },
            "renderer_api": self.final_renderer_api.as_dict(),
            "baseline_renderer_api": self.initial_renderer_api.as_dict(),
            "object_state_eval": self.object_state_eval,
            "checkpoint_roundtrip": self.checkpoint_roundtrip,
            "image_target_contract": image_target_contract_summary(
                tuple(frame.image_target for frame in self.frames)
            ),
            "source": {
                "assignment_checkpoint_schema": self.checkpoint["schema"],
                "assignment_checkpoint_source": self.checkpoint["source"]["source"],
                "sampled_gaussians": int(self.frames[0].positions.shape[0])
                if self.frames
                else 0,
                "target_source": "trainable_frame_target_assignment",
            },
            "solver": {
                "family": self.solver_state.config.solver_family,
                "step": int(self.solver_state.step),
                "slots": int(self.solver_state.config.slots),
                "feature_dim": int(self.solver_state.config.feature_dim),
                "temperature": float(self.solver_state.config.temperature),
                "source": self.solver_state.source,
            },
            "decoder": {
                "step": int(self.decoder_state.step),
                "slots": int(self.decoder_state.slots),
                "source": self.decoder_state.source,
                "available_fields": list(
                    self.decoder_state.as_dict()["available_fields"]
                ),
                "frozen_fields": list(self.decoder_state.as_dict()["frozen_fields"]),
            },
            "trained_fields": [],
            "frozen_fields": [
                "solver.feature_centers",
                "solver.position_centers",
                "solver.slot_bias",
                "decoder.object_colors",
                "means",
                "quats",
                "scales",
                "opacities",
                "cameras",
                "dynamic_k",
            ],
            "gpu_policy": {
                "uses_gpu": self.image_renderer == TRAINING_IMAGE_RENDERER_GSPLAT,
                "vram_reserve_gb": int(self.vram_reserve_gb),
            },
            "identity_gate": {
                "renderer_loss_does_not_override_identity_gate": True,
                "required_upstream_gate": "ASSIGNMENT-SOLVER-V2-EVAL-001",
            },
            "non_goals": {
                "trains_optimizer": False,
                "unfreezes_gaussian_geometry": False,
                "mutates_dynamic_k": False,
                "uses_rollout_model": False,
                "uses_replay_buffer": False,
            },
        }
        return validate_assignment_v2_renderer_joint_summary(payload)


def evaluate_assignment_v2_renderer_joint(
    frames: Sequence[TrainableKernelFrame],
    checkpoint: dict[str, Any],
    *,
    decoder_state: ObjectStateGaussianDecoderState | None = None,
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    vram_reserve_gb: int = 1,
) -> AssignmentV2RendererJointValidationReport:
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")
    if vram_reserve_gb < 0:
        raise ValueError("vram_reserve_gb must be >= 0")
    checked_checkpoint = validate_assignment_solver_v2_checkpoint(checkpoint)
    checked_frames = _validate_frames(frames)
    solver_state = validate_assignment_solver_v2_state(
        assignment_solver_v2_state_from_checkpoint(checked_checkpoint)
    )
    if not all(frame.image_target is not None for frame in checked_frames):
        raise ValueError("assignment v2 renderer validation requires image targets")
    if not all(frame.target_assignment is not None for frame in checked_frames):
        raise ValueError("assignment v2 renderer validation requires target assignments")

    evidence_batches = assignment_evidence_sequence_from_trainable_frames(
        checked_frames,
        source="assignment_v2_renderer_joint_validation",
    )
    final_assignments = tuple(
        predict_assignment_solver_v2(batch, solver_state).assignment
        for batch in evidence_batches
    )
    initial_assignments = tuple(
        np.full_like(assignment, 1.0 / assignment.shape[1], dtype=np.float32)
        for assignment in final_assignments
    )
    targets = tuple(frame.target_assignment for frame in checked_frames)
    resolved_decoder = _decoder_state_for_frames(
        checked_frames,
        solver_state=solver_state,
        decoder_state=decoder_state,
    )
    initial_renderer = _evaluate_renderer(
        checked_frames,
        initial_assignments,
        resolved_decoder,
        image_renderer=image_renderer,
    )
    final_renderer = _evaluate_renderer(
        checked_frames,
        final_assignments,
        resolved_decoder,
        image_renderer=image_renderer,
    )
    projections = tuple(
        project_object_states(
            _frame_cloud(frame),
            assignment,
            evidence_features=frame.features,
        )
        for frame, assignment in zip(checked_frames, final_assignments, strict=True)
    )
    initial_assignment_loss = assignment_loss_v2_breakdown(
        initial_assignments,
        target_assignments=targets,
        supervised_weight=1.0,
    ).as_dict()
    final_assignment_loss = assignment_loss_v2_breakdown(
        final_assignments,
        target_assignments=targets,
        supervised_weight=1.0,
    ).as_dict()
    object_state_eval = _object_state_eval(checked_frames, projections)
    checkpoint_roundtrip = _checkpoint_roundtrip_summary(
        checked_frames,
        checkpoint=checked_checkpoint,
        expected=final_assignments,
    )
    return AssignmentV2RendererJointValidationReport(
        checkpoint=checked_checkpoint,
        solver_state=solver_state,
        frames=checked_frames,
        decoder_state=resolved_decoder,
        initial_assignments=initial_assignments,
        final_assignments=final_assignments,
        projections=projections,
        initial_renderer_api=initial_renderer,
        final_renderer_api=final_renderer,
        initial_assignment_loss=initial_assignment_loss,
        final_assignment_loss=final_assignment_loss,
        object_state_eval=object_state_eval,
        checkpoint_roundtrip=checkpoint_roundtrip,
        image_renderer=image_renderer,
        vram_reserve_gb=int(vram_reserve_gb),
    )


def validate_assignment_v2_renderer_joint_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("assignment v2 renderer joint summary must be a dict")
    if payload.get("schema") != ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA:
        raise ValueError(f"unsupported assignment v2 renderer schema: {payload.get('schema')}")
    if payload.get("kind") != "assignment_v2_renderer_joint_validation":
        raise ValueError("assignment v2 renderer joint kind is unsupported")
    if payload.get("checkpoint_schema") != ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA:
        raise ValueError("assignment v2 renderer joint summary must reference a v2 checkpoint")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_FAIL}:
        raise ValueError("assignment v2 renderer joint status is unsupported")
    for key in (
        "initial_loss",
        "final_loss",
        "renderer_api",
        "baseline_renderer_api",
        "object_state_eval",
        "checkpoint_roundtrip",
        "image_target_contract",
        "source",
        "identity_gate",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"assignment v2 renderer joint summary missing {key}")
    initial = _validate_loss_payload(payload["initial_loss"], "initial_loss")
    final = _validate_loss_payload(payload["final_loss"], "final_loss")
    expected_pass = (
        final["total_loss"] < initial["total_loss"]
        and final["image_render_loss"] < initial["image_render_loss"]
        and payload["object_state_eval"].get("status") == "objectstate_eval_pass"
        and payload["checkpoint_roundtrip"].get("pass") is True
    )
    expected_status = _STATUS_PASS if expected_pass else _STATUS_FAIL
    if payload["status"] != expected_status:
        raise ValueError("assignment v2 renderer joint status must match gates")
    if payload["identity_gate"].get("renderer_loss_does_not_override_identity_gate") is not True:
        raise ValueError("assignment v2 renderer joint must not override identity gate")
    non_goals = payload["non_goals"]
    if (
        non_goals.get("trains_optimizer")
        or non_goals.get("unfreezes_gaussian_geometry")
        or non_goals.get("mutates_dynamic_k")
        or non_goals.get("uses_rollout_model")
        or non_goals.get("uses_replay_buffer")
    ):
        raise ValueError("assignment v2 renderer joint violates non-goals")
    return payload


def _validate_frames(frames: Sequence[TrainableKernelFrame]) -> tuple[TrainableKernelFrame, ...]:
    checked = tuple(frames)
    if not checked:
        raise ValueError("frames must contain at least one frame")
    feature_dim = int(np.asarray(checked[0].features).shape[1])
    for index, frame in enumerate(checked):
        positions = np.asarray(frame.positions, dtype=np.float32)
        features = np.asarray(frame.features, dtype=np.float32)
        rgb = np.asarray(frame.target_rgb, dtype=np.float32)
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise ValueError(f"frames[{index}].positions must be N x 3")
        if features.ndim != 2 or features.shape[0] != positions.shape[0]:
            raise ValueError(f"frames[{index}].features rows must match positions")
        if features.shape[1] != feature_dim:
            raise ValueError("all frames must share feature_dim")
        if rgb.shape != (positions.shape[0], 3):
            raise ValueError(f"frames[{index}].target_rgb must be N x 3")
    return checked


def _decoder_state_for_frames(
    frames: Sequence[TrainableKernelFrame],
    *,
    solver_state: AssignmentSolverV2State,
    decoder_state: ObjectStateGaussianDecoderState | None,
) -> ObjectStateGaussianDecoderState:
    if decoder_state is not None:
        checked = validate_object_state_gaussian_decoder_state(decoder_state)
        if checked.slots != solver_state.config.slots:
            raise ValueError("decoder slots must match assignment solver v2 slots")
        return checked
    slots = int(solver_state.config.slots)
    colors = np.zeros((slots, 3), dtype=np.float32)
    mass = np.zeros(slots, dtype=np.float32)
    for frame in frames:
        target = np.asarray(frame.target_assignment, dtype=np.float32)
        rgb = np.asarray(frame.target_rgb, dtype=np.float32)
        colors += target.T @ rgb
        mass += target.sum(axis=0)
    active = mass > 1e-8
    colors[active] = colors[active] / mass[active, None]
    if not np.all(active):
        fallback = initialize_object_state_gaussian_decoder(slots=slots, seed=0)
        colors[~active] = fallback.object_colors[~active]
    return validate_object_state_gaussian_decoder_state(
        ObjectStateGaussianDecoderState(
            object_colors=colors,
            source="target_rgb_slot_mean_assignment_v2_renderer_validation",
        )
    )


def _evaluate_renderer(
    frames: Sequence[TrainableKernelFrame],
    assignments: Sequence[np.ndarray],
    decoder_state: ObjectStateGaussianDecoderState,
    *,
    image_renderer: str,
) -> TrainingRendererLossResult:
    if image_renderer == TRAINING_IMAGE_RENDERER_POINT:
        return evaluate_training_renderer_loss(
            frames,
            assignments,
            decoder_state.object_colors,
            decoder_opacity_logits=decoder_state.object_opacity_logits,
            decoder_scale_log_offsets=decoder_state.object_scale_log_offsets,
        )
    return evaluate_gsplat_training_renderer_loss(
        frames,
        assignments,
        decoder_state.object_colors,
        decoder_opacity_logits=decoder_state.object_opacity_logits,
        decoder_scale_log_offsets=decoder_state.object_scale_log_offsets,
    )


def _object_state_eval(
    frames: Sequence[TrainableKernelFrame],
    projections: Sequence[ObjectStateProjection],
) -> dict[str, Any]:
    reports = []
    for frame, projection in zip(frames, projections, strict=True):
        target = np.asarray(frame.target_assignment, dtype=np.float32)
        labels = np.argmax(target, axis=1)
        report = object_state_stability_report(
            projection,
            purity_labels=labels,
            assignment_confidence_floor=0.4,
            purity_floor=0.8,
        )
        reports.append(report)
    mean_entropy = float(np.mean([report.mean_normalized_entropy for report in reports]))
    confidence = float(np.mean([report.assignment_confidence for report in reports]))
    max_dominant = float(max(report.dominant_slot_mass_fraction for report in reports))
    min_purity = min(
        (
            report.object_purity
            for report in reports
            if report.object_purity is not None
        ),
        default=None,
    )
    slot_collapse = any(report.slot_collapse for report in reports)
    pass_gate = (
        mean_entropy <= 0.6
        and confidence >= 0.4
        and not slot_collapse
        and min_purity is not None
        and min_purity >= 0.8
    )
    return {
        "status": "objectstate_eval_pass" if pass_gate else "objectstate_eval_fail",
        "mean_normalized_entropy": mean_entropy,
        "assignment_confidence": confidence,
        "max_dominant_slot_mass_fraction": max_dominant,
        "slot_collapse": bool(slot_collapse),
        "object_purity": None if min_purity is None else float(min_purity),
        "frame_count": len(reports),
        "frames": [
            {
                "mean_normalized_entropy": float(report.mean_normalized_entropy),
                "assignment_confidence": float(report.assignment_confidence),
                "effective_slots": float(report.effective_slots),
                "slot_collapse": bool(report.slot_collapse),
                "object_purity": None
                if report.object_purity is None
                else float(report.object_purity),
                "diagnostics": list(report.diagnostics),
            }
            for report in reports
        ],
    }


def _checkpoint_roundtrip_summary(
    frames: Sequence[TrainableKernelFrame],
    *,
    checkpoint: dict[str, Any],
    expected: Sequence[np.ndarray],
) -> dict[str, Any]:
    restored = assignment_solver_v2_state_from_checkpoint(checkpoint)
    evidence = assignment_evidence_sequence_from_trainable_frames(
        frames,
        source="assignment_v2_renderer_joint_roundtrip",
    )
    max_delta = 0.0
    for batch, expected_assignment in zip(evidence, expected, strict=True):
        restored_assignment = predict_assignment_solver_v2(batch, restored).assignment
        delta = float(np.max(np.abs(restored_assignment - expected_assignment)))
        max_delta = max(max_delta, delta)
    return {
        "pass": max_delta <= 1e-5,
        "max_assignment_delta": float(max_delta),
        "state_step": int(restored.step),
    }


def _loss_record(
    *,
    iteration: int,
    image_render_loss: float,
    object_loss: float,
) -> dict[str, Any]:
    image = float(image_render_loss)
    obj = float(object_loss)
    total = image + obj
    return {
        "iteration": int(iteration),
        "total_loss": total,
        "image_render_loss": image,
        "object_loss": obj,
        "entropy_loss": 0.0,
        "balance_loss": 0.0,
        "temporal_loss": 0.0,
    }


def _supervised_loss(payload: dict[str, Any]) -> float:
    losses = payload.get("losses")
    if not isinstance(losses, dict):
        raise ValueError("assignment loss payload missing losses")
    return float(losses["supervised"])


def _validate_loss_payload(payload: dict[str, Any], label: str) -> dict[str, float]:
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a dict")
    required = (
        "total_loss",
        "image_render_loss",
        "object_loss",
        "entropy_loss",
        "balance_loss",
        "temporal_loss",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"{label} missing {', '.join(missing)}")
    return {key: float(payload[key]) for key in required}


def _frame_cloud(frame: TrainableKernelFrame) -> GaussianCloud:
    positions = np.asarray(frame.positions, dtype=np.float32)
    vertices = np.zeros(
        positions.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ],
    )
    vertices["x"] = positions[:, 0]
    vertices["y"] = positions[:, 1]
    vertices["z"] = positions[:, 2]
    return GaussianCloud(vertices=vertices, source_format="assignment_v2_renderer_validation")
