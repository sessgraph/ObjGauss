from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from objgauss.core.assignment_evidence import (
    assignment_evidence_sequence_from_trainable_frames,
)
from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2TrainingResult,
    initialize_assignment_solver_v2,
    train_assignment_solver_v2,
)
from objgauss.core.assignment_solver_v2_eval import (
    ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA,
    assignment_solver_v2_checkpoint,
)
from objgauss.core.assignment_v2_renderer_validation import (
    AssignmentV2RendererJointValidationReport,
    evaluate_assignment_v2_renderer_joint,
    validate_assignment_v2_renderer_joint_summary,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.renderer_loss import (
    RendererLossBoundaryReport,
    renderer_loss_boundary_report,
)
from objgauss.core.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
    TrainableKernelFrame,
    TrainableKernelSample,
    bind_image_targets_to_frames,
    trainable_kernel_sample_from_cloud,
)

REAL_SAMPLE_V2_SMOKE_SCHEMA = "objgauss-real-sample-v2-smoke-v1"
_STATUS_PASS = "real_sample_v2_smoke_pass"
_STATUS_FAIL = "real_sample_v2_smoke_fail"
_OBJECT_ID_TARGET_SOURCE = "object_id_one_hot_targets"
_BOUNDARY_READY_STATUSES = {
    "assignment_v2_renderer_joint_validation_ready",
    "full_3dgs_assignment_v2_renderer_joint_validation_ready",
}


@dataclass(frozen=True)
class RealSampleV2SmokeReport:
    sample: TrainableKernelSample
    training_result: AssignmentSolverV2TrainingResult
    checkpoint: dict[str, Any]
    renderer_joint: AssignmentV2RendererJointValidationReport
    renderer_boundary: RendererLossBoundaryReport
    sample_source: str
    object_id_field: str
    schema: str = REAL_SAMPLE_V2_SMOKE_SCHEMA

    @property
    def passed(self) -> bool:
        renderer_summary = self.renderer_joint.as_dict()
        boundary_summary = self.renderer_boundary.as_dict()
        return (
            self.training_result.final_loss.total_loss
            < self.training_result.initial_loss.total_loss
            and renderer_summary["status"]
            == "assignment_v2_renderer_joint_validation_pass"
            and boundary_summary["status"] in _BOUNDARY_READY_STATUSES
            and self.sample.target_source == _OBJECT_ID_TARGET_SOURCE
            and bool(renderer_summary["checkpoint_roundtrip"]["pass"])
        )

    def as_dict(self) -> dict[str, Any]:
        training_summary = self.training_result.as_dict()
        renderer_summary = self.renderer_joint.as_dict()
        boundary_summary = self.renderer_boundary.as_dict()
        sample_summary = self.sample.as_dict()
        payload = {
            "schema": self.schema,
            "kind": "real_sample_v2_smoke",
            "status": _STATUS_PASS if self.passed else _STATUS_FAIL,
            "sample": {
                **sample_summary,
                "source": self.sample_source,
                "source_kind": _sample_source_kind(self.sample_source),
                "object_id_field": self.object_id_field,
                "target_assignment_required": True,
            },
            "training_schema": training_summary["schema"],
            "checkpoint_schema": self.checkpoint["schema"],
            "renderer_joint_schema": renderer_summary["schema"],
            "renderer_boundary_schema": boundary_summary["schema"],
            "training_loss": {
                "initial_total_loss": float(self.training_result.initial_loss.total_loss),
                "final_total_loss": float(self.training_result.final_loss.total_loss),
                "loss_decreased": bool(
                    self.training_result.final_loss.total_loss
                    < self.training_result.initial_loss.total_loss
                ),
                "initial_supervised_loss": float(
                    self.training_result.initial_loss.supervised_loss
                ),
                "final_supervised_loss": float(
                    self.training_result.final_loss.supervised_loss
                ),
                "supervised_loss_decreased": bool(
                    self.training_result.final_loss.supervised_loss
                    < self.training_result.initial_loss.supervised_loss
                ),
            },
            "gates": {
                "object_id_targets_bound": self.sample.target_source
                == _OBJECT_ID_TARGET_SOURCE,
                "training_loss_decreased": bool(
                    self.training_result.final_loss.total_loss
                    < self.training_result.initial_loss.total_loss
                ),
                "renderer_joint_passed": renderer_summary["status"]
                == "assignment_v2_renderer_joint_validation_pass",
                "renderer_boundary_ready": boundary_summary["status"]
                in _BOUNDARY_READY_STATUSES,
                "checkpoint_roundtrip_passed": bool(
                    renderer_summary["checkpoint_roundtrip"]["pass"]
                ),
            },
            "truth_contract": {
                "target_source": _OBJECT_ID_TARGET_SOURCE,
                "object_id_labels_are_training_targets": True,
                "semantic_ground_truth_claimed": False,
                "fixture_oracle_claimed": False,
            },
            "renderer_joint": renderer_summary,
            "renderer_boundary": boundary_summary,
            "checkpoint": {
                "schema": self.checkpoint["schema"],
                "source": self.checkpoint["source"],
                "solver_step": int(self.training_result.final_state.step),
            },
            "training": training_summary,
            "non_goals": {
                "uses_fixture_oracle": False,
                "claims_semantic_ground_truth": False,
                "uses_gpu": False,
                "unfreezes_gaussian_geometry": False,
                "mutates_dynamic_k": False,
                "uses_rollout_model": False,
                "uses_replay_buffer": False,
            },
        }
        return validate_real_sample_v2_smoke_summary(payload)


def real_sample_v2_smoke_from_cloud(
    cloud: GaussianCloud,
    *,
    sample_source: str = "memory://gaussian-cloud",
    object_id_field: str = "object_id",
    slots: int | None = None,
    frame_count: int = 2,
    max_points: int | None = 64,
    temporal_offset: float = 0.01,
    image_width: int = 16,
    image_height: int = 16,
    point_radius: int = 1,
    visibility_policy: str = "covered_pixels",
    seed: int = 0,
    iterations: int = 80,
    learning_rate: float = 0.35,
    cluster_weight: float = 0.0,
    entropy_weight: float = 0.0,
    balance_weight: float = 0.0,
    supervised_weight: float = 1.0,
    solver_temperature: float = 1.0,
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    vram_reserve_gb: int = 1,
) -> RealSampleV2SmokeReport:
    sample = trainable_kernel_sample_from_cloud(
        cloud,
        slots=slots,
        frame_count=frame_count,
        max_points=max_points,
        object_id_field=object_id_field,
        temporal_offset=temporal_offset,
        bind_image_targets=True,
        image_width=image_width,
        image_height=image_height,
        point_radius=point_radius,
        visibility_policy=visibility_policy,
        seed=seed,
    )
    return evaluate_real_sample_v2_smoke(
        sample,
        sample_source=sample_source,
        object_id_field=object_id_field,
        image_width=image_width,
        image_height=image_height,
        point_radius=point_radius,
        visibility_policy=visibility_policy,
        seed=seed,
        iterations=iterations,
        learning_rate=learning_rate,
        cluster_weight=cluster_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        supervised_weight=supervised_weight,
        solver_temperature=solver_temperature,
        image_renderer=image_renderer,
        vram_reserve_gb=vram_reserve_gb,
    )


def evaluate_real_sample_v2_smoke(
    sample: TrainableKernelSample,
    *,
    sample_source: str = "memory://trainable-kernel-sample",
    object_id_field: str = "object_id",
    image_width: int = 16,
    image_height: int = 16,
    point_radius: int = 1,
    visibility_policy: str = "covered_pixels",
    seed: int = 0,
    iterations: int = 80,
    learning_rate: float = 0.35,
    cluster_weight: float = 0.0,
    entropy_weight: float = 0.0,
    balance_weight: float = 0.0,
    supervised_weight: float = 1.0,
    solver_temperature: float = 1.0,
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    vram_reserve_gb: int = 1,
) -> RealSampleV2SmokeReport:
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if supervised_weight <= 0:
        raise ValueError("supervised_weight must be > 0")
    if solver_temperature <= 0:
        raise ValueError("solver_temperature must be > 0")
    checked_sample = _validate_real_sample(sample)
    if not all(frame.image_target is not None for frame in checked_sample.frames):
        checked_sample = _sample_with_bound_image_targets(
            checked_sample,
            width=image_width,
            height=image_height,
            point_radius=point_radius,
            visibility_policy=visibility_policy,
        )
    batches = assignment_evidence_sequence_from_trainable_frames(
        checked_sample.frames,
        source="real_sample_v2_smoke",
    )
    initial_state = initialize_assignment_solver_v2(
        slots=checked_sample.slots,
        feature_dim=batches[0].feature_dim,
        position_dim=batches[0].positions.shape[1],
        temperature=solver_temperature,
        seed=seed,
    )
    training_result = train_assignment_solver_v2(
        batches,
        initial_state=initial_state,
        iterations=iterations,
        learning_rate=learning_rate,
        cluster_weight=cluster_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        supervised_weight=supervised_weight,
        seed=seed,
        record_every=max(1, iterations // 4),
    )
    checkpoint = assignment_solver_v2_checkpoint(
        training_result,
        source="real_sample_v2_smoke",
    )
    renderer_joint = evaluate_assignment_v2_renderer_joint(
        checked_sample.frames,
        checkpoint,
        image_renderer=image_renderer,
        vram_reserve_gb=vram_reserve_gb,
    )
    renderer_boundary = renderer_loss_boundary_report(renderer_joint.as_dict())
    return RealSampleV2SmokeReport(
        sample=checked_sample,
        training_result=training_result,
        checkpoint=checkpoint,
        renderer_joint=renderer_joint,
        renderer_boundary=renderer_boundary,
        sample_source=str(sample_source),
        object_id_field=str(object_id_field),
    )


def validate_real_sample_v2_smoke_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("real sample v2 smoke summary must be a dict")
    if payload.get("schema") != REAL_SAMPLE_V2_SMOKE_SCHEMA:
        raise ValueError(f"unsupported real sample v2 smoke schema: {payload.get('schema')}")
    if payload.get("kind") != "real_sample_v2_smoke":
        raise ValueError("real sample v2 smoke kind is unsupported")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_FAIL}:
        raise ValueError("real sample v2 smoke status is unsupported")
    for key in (
        "sample",
        "training_loss",
        "gates",
        "truth_contract",
        "renderer_joint",
        "renderer_boundary",
        "checkpoint",
        "training",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"real sample v2 smoke summary missing {key}")
    sample = payload["sample"]
    if not isinstance(sample, dict):
        raise ValueError("real sample v2 smoke sample must be a dict")
    if sample.get("target_source") != _OBJECT_ID_TARGET_SOURCE:
        raise ValueError("real sample v2 smoke requires object_id one-hot targets")
    if not isinstance(sample.get("object_id_mapping"), dict) or not sample["object_id_mapping"]:
        raise ValueError("real sample v2 smoke requires a non-empty object_id mapping")
    if sample.get("target_assignment_required") is not True:
        raise ValueError("real sample v2 smoke must require target assignments")
    if payload.get("checkpoint_schema") != ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA:
        raise ValueError("real sample v2 smoke must reference a v2 checkpoint")
    renderer_summary = validate_assignment_v2_renderer_joint_summary(
        payload["renderer_joint"]
    )
    boundary = payload["renderer_boundary"]
    if not isinstance(boundary, dict):
        raise ValueError("real sample v2 smoke renderer_boundary must be a dict")
    loss = payload["training_loss"]
    if not isinstance(loss, dict):
        raise ValueError("real sample v2 smoke training_loss must be a dict")
    loss_decreased = float(loss["final_total_loss"]) < float(loss["initial_total_loss"])
    if bool(loss["loss_decreased"]) != loss_decreased:
        raise ValueError("real sample v2 smoke loss_decreased must match totals")
    truth_contract = payload["truth_contract"]
    if truth_contract.get("semantic_ground_truth_claimed") is not False:
        raise ValueError("real sample v2 smoke must not claim semantic ground truth")
    if truth_contract.get("fixture_oracle_claimed") is not False:
        raise ValueError("real sample v2 smoke must not claim fixture oracle truth")
    gates = payload["gates"]
    if not isinstance(gates, dict):
        raise ValueError("real sample v2 smoke gates must be a dict")
    expected_pass = (
        gates.get("object_id_targets_bound") is True
        and gates.get("training_loss_decreased") is True
        and gates.get("renderer_joint_passed") is True
        and gates.get("renderer_boundary_ready") is True
        and gates.get("checkpoint_roundtrip_passed") is True
    )
    if gates.get("training_loss_decreased") != loss_decreased:
        raise ValueError("real sample v2 smoke training gate must match training loss")
    if gates.get("renderer_joint_passed") != (
        renderer_summary["status"] == "assignment_v2_renderer_joint_validation_pass"
    ):
        raise ValueError("real sample v2 smoke renderer gate must match renderer summary")
    if gates.get("renderer_boundary_ready") != (
        boundary.get("status") in _BOUNDARY_READY_STATUSES
    ):
        raise ValueError("real sample v2 smoke boundary gate must match boundary summary")
    expected_status = _STATUS_PASS if expected_pass else _STATUS_FAIL
    if payload["status"] != expected_status:
        raise ValueError("real sample v2 smoke status must match gates")
    non_goals = payload["non_goals"]
    if (
        non_goals.get("uses_fixture_oracle")
        or non_goals.get("claims_semantic_ground_truth")
        or non_goals.get("uses_gpu")
        or non_goals.get("unfreezes_gaussian_geometry")
        or non_goals.get("mutates_dynamic_k")
        or non_goals.get("uses_rollout_model")
        or non_goals.get("uses_replay_buffer")
    ):
        raise ValueError("real sample v2 smoke summary violates non-goals")
    return payload


def _validate_real_sample(sample: TrainableKernelSample) -> TrainableKernelSample:
    if not isinstance(sample, TrainableKernelSample):
        raise TypeError("sample must be a TrainableKernelSample")
    if int(sample.slots) < 1:
        raise ValueError("sample slots must be >= 1")
    if int(sample.source_count) < 1:
        raise ValueError("sample source_count must be >= 1")
    if int(sample.sampled_count) < 1:
        raise ValueError("sample sampled_count must be >= 1")
    if sample.target_source != _OBJECT_ID_TARGET_SOURCE:
        raise ValueError("real sample v2 smoke requires object_id_one_hot_targets")
    if not sample.object_id_mapping:
        raise ValueError("real sample v2 smoke requires object_id_mapping")
    frames = tuple(sample.frames)
    if not frames:
        raise ValueError("sample frames must contain at least one frame")
    for index, frame in enumerate(frames):
        _validate_smoke_frame(frame, index=index, slots=sample.slots)
    return TrainableKernelSample(
        frames=frames,
        slots=int(sample.slots),
        source_count=int(sample.source_count),
        sampled_count=int(sample.sampled_count),
        target_source=str(sample.target_source),
        object_id_mapping={int(key): int(value) for key, value in sample.object_id_mapping.items()},
    )


def _validate_smoke_frame(
    frame: TrainableKernelFrame,
    *,
    index: int,
    slots: int,
) -> None:
    positions = np.asarray(frame.positions, dtype=np.float32)
    features = np.asarray(frame.features, dtype=np.float32)
    rgb = np.asarray(frame.target_rgb, dtype=np.float32)
    target = None if frame.target_assignment is None else np.asarray(frame.target_assignment, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3 or positions.shape[0] == 0:
        raise ValueError(f"sample.frames[{index}].positions must be non-empty N x 3")
    if features.ndim != 2 or features.shape[0] != positions.shape[0] or features.shape[1] == 0:
        raise ValueError(f"sample.frames[{index}].features must match positions")
    if rgb.shape != (positions.shape[0], 3):
        raise ValueError(f"sample.frames[{index}].target_rgb must be N x 3")
    if target is None:
        raise ValueError(f"sample.frames[{index}] must bind target_assignment")
    if target.shape != (positions.shape[0], int(slots)):
        raise ValueError(f"sample.frames[{index}].target_assignment must be N x slots")
    if not np.allclose(target.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"sample.frames[{index}].target_assignment rows must sum to 1")


def _sample_with_bound_image_targets(
    sample: TrainableKernelSample,
    *,
    width: int,
    height: int,
    point_radius: int,
    visibility_policy: str,
) -> TrainableKernelSample:
    frames = bind_image_targets_to_frames(
        sample.frames,
        width=width,
        height=height,
        point_radius=point_radius,
        visibility_policy=visibility_policy,
        source="real_sample_v2_point_splat_debug",
    )
    return TrainableKernelSample(
        frames=frames,
        slots=sample.slots,
        source_count=sample.source_count,
        sampled_count=sample.sampled_count,
        target_source=sample.target_source,
        object_id_mapping=dict(sample.object_id_mapping),
    )


def _sample_source_kind(source: str) -> str:
    if source.startswith("public/samples/") or "/public/samples/" in source:
        return "public_sample"
    if source.startswith("fixture://"):
        return "fixture"
    if source.startswith("memory://"):
        return "memory"
    return "external_or_local_path"
