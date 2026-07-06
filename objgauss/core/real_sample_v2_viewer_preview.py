from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.assignment_evidence import (
    assignment_evidence_sequence_from_trainable_frames,
)
from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2Prediction,
    AssignmentSolverV2State,
    predict_assignment_solver_v2,
)
from objgauss.core.assignment_solver_v2_eval import (
    ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA,
    assignment_solver_v2_state_from_checkpoint,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io_ply import append_or_replace_property
from objgauss.core.object_state import (
    ObjectStabilityReport,
    ObjectStateProjection,
    object_state_stability_report,
    project_object_states,
)
from objgauss.core.objects import apply_object_colors
from objgauss.core.real_sample_v2_model_handoff import (
    RealSampleV2ModelHandoffReport,
    real_sample_v2_model_handoff_from_cloud,
    validate_real_sample_v2_model_handoff_summary,
)
from objgauss.core.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
    TrainableKernelSample,
    trainable_kernel_sample_from_cloud,
)

REAL_SAMPLE_V2_VIEWER_PREVIEW_SCHEMA = "objgauss-real-sample-v2-viewer-preview-v1"
REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT = 2.0
REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT = 1.0
_STATUS_PASS = "real_sample_v2_viewer_preview_pass"
_STATUS_FAIL = "real_sample_v2_viewer_preview_fail"
_QUALITY_PASS = "full_cloud_objectstate_preview_quality_pass"
_QUALITY_DIAGNOSTIC = "full_cloud_objectstate_preview_quality_diagnostic"


@dataclass(frozen=True)
class RealSampleV2ViewerPreviewReport:
    handoff: RealSampleV2ModelHandoffReport
    sample: TrainableKernelSample
    prediction: AssignmentSolverV2Prediction
    projection: ObjectStateProjection
    stability: ObjectStabilityReport
    projected_cloud: GaussianCloud
    sample_source: str
    object_id_field: str
    baseline_feature_weight: float
    baseline_position_weight: float
    promoted_feature_weight: float
    promoted_position_weight: float
    viewer_path: str | None = None
    schema: str = REAL_SAMPLE_V2_VIEWER_PREVIEW_SCHEMA

    @property
    def passed(self) -> bool:
        handoff_summary = self.handoff.as_dict()
        return (
            handoff_summary["status"] == "real_sample_v2_model_handoff_pass"
            and self.projected_cloud.count == self.sample.source_count
            and self.prediction.evidence_count == self.sample.source_count
        )

    def as_dict(self) -> dict[str, Any]:
        handoff_summary = self.handoff.as_dict()
        validate_real_sample_v2_model_handoff_summary(handoff_summary)
        checkpoint_state = assignment_solver_v2_state_from_checkpoint(self.handoff.checkpoint)
        target_slots = _target_slots(self.sample)
        direct_match = (
            0.0
            if target_slots.size == 0
            else float(np.mean(self.projection.derived_object_ids == target_slots))
        )
        hard_segmentation = _hard_segmentation_summary(
            target_slots,
            self.projection.derived_object_ids,
        )
        payload = {
            "schema": self.schema,
            "kind": "real_sample_v2_viewer_preview",
            "status": _STATUS_PASS if self.passed else _STATUS_FAIL,
            "source": {
                "input": self.sample_source,
                "source_gaussians": int(self.sample.source_count),
                "object_id_field": self.object_id_field,
                "target_source": self.sample.target_source,
                "object_id_mapping": {
                    str(object_id): int(slot)
                    for object_id, slot in self.sample.object_id_mapping.items()
                },
            },
            "handoff": {
                "schema": handoff_summary["schema"],
                "status": handoff_summary["status"],
                "recommended_solver_temperature": handoff_summary[
                    "recommended_solver_temperature"
                ],
                "sampled_gaussians": handoff_summary["sample"]["sampled_count"],
                "restore_renderer_joint_status": handoff_summary["restore_validation"][
                    "renderer_joint_status"
                ],
                "restore_object_state_status": handoff_summary["restore_validation"][
                    "object_state_status"
                ],
            },
            "checkpoint": {
                "schema": self.handoff.checkpoint["schema"],
                "solver_step": int(checkpoint_state.step),
                "solver_temperature": float(checkpoint_state.config.temperature),
                "slots": int(checkpoint_state.config.slots),
                "feature_dim": int(checkpoint_state.config.feature_dim),
                "state_schema": checkpoint_state.schema,
            },
            "assignment_weight_policy": {
                "family": "assignment_v2_cost_weight_promotion",
                "promotion_source": "REAL-SAMPLE-V2-WEAK-BOUNDARY-OPT-001",
                "baseline_feature_weight": float(self.baseline_feature_weight),
                "baseline_position_weight": float(self.baseline_position_weight),
                "promoted_feature_weight": float(self.promoted_feature_weight),
                "promoted_position_weight": float(self.promoted_position_weight),
                "applied": (
                    float(self.baseline_feature_weight) != float(self.promoted_feature_weight)
                    or float(self.baseline_position_weight) != float(self.promoted_position_weight)
                ),
                "uses_target_labels_for_prediction": False,
                "mutates_checkpoint": False,
            },
            "projection": {
                "source": "checkpoint_restored_full_cloud_assignment",
                "projected_gaussians": int(self.prediction.evidence_count),
                "exported_gaussians": int(self.projected_cloud.count),
                "export_object_id": "argmax_assignment_slot",
                "preserved_target_fields": [
                    "target_object_id",
                    "target_slot",
                ],
                "debug_fields": [
                    "assignment_confidence",
                    "assignment_entropy",
                ],
                "predicted_object_count": int(
                    np.unique(self.projection.derived_object_ids).shape[0]
                ),
                "slot_mass": [float(value) for value in self.stability.slot_mass],
                "slot_mass_fraction": [
                    float(value) for value in self.stability.slot_mass_fraction
                ],
                "hard_segmentation": hard_segmentation,
            },
            "quality": {
                "status": (
                    _QUALITY_PASS
                    if not self.stability.diagnostics
                    else _QUALITY_DIAGNOSTIC
                ),
                "mean_normalized_entropy": float(self.stability.mean_normalized_entropy),
                "assignment_confidence": float(self.stability.assignment_confidence),
                "object_purity": (
                    None
                    if self.stability.object_purity is None
                    else float(self.stability.object_purity)
                ),
                "direct_slot_match": direct_match,
                "diagnostics": list(self.stability.diagnostics),
            },
            "viewer": {
                "route_param": "ply",
                "viewer_path": self.viewer_path,
                "debug_route": f"/?ply={self.viewer_path}" if self.viewer_path else None,
                "load_mode": "url-object-aware-ply",
            },
            "output_policy": {
                "preview_ply": "write to /tmp or ignored outputs; do not commit generated preview PLY",
                "summary": "write to /tmp or ignored outputs for viewer handoff",
                "checkpoint": "read from handoff or regenerate; do not commit training checkpoints",
            },
            "non_goals": {
                "uses_gpu": False,
                "unfreezes_gaussian_geometry": False,
                "unfreezes_camera": False,
                "mutates_dynamic_k": False,
                "uses_rollout_model": False,
                "uses_replay_buffer": False,
                "uses_diffusion": False,
                "claims_public_demo_release": False,
            },
        }
        return validate_real_sample_v2_viewer_preview_summary(payload)


def real_sample_v2_viewer_preview_from_cloud(
    cloud: GaussianCloud,
    *,
    sample_source: str = "memory://gaussian-cloud",
    object_id_field: str = "object_id",
    slots: int | None = None,
    frame_count: int = 2,
    max_points: int | None = 24,
    temporal_offset: float = 0.01,
    image_width: int = 12,
    image_height: int = 12,
    point_radius: int = 1,
    visibility_policy: str = "covered_pixels",
    seed: int = 4,
    iterations: int = 100,
    learning_rate: float = 0.4,
    temperature_candidates: Sequence[float] = (1.0, 0.75, 0.5, 0.35, 0.25),
    baseline_temperature: float = 1.0,
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    vram_reserve_gb: int = 1,
    assignment_feature_weight: float | None = None,
    assignment_position_weight: float | None = None,
    rewrite_sh: bool = False,
    viewer_path: str | None = None,
) -> RealSampleV2ViewerPreviewReport:
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")
    handoff = real_sample_v2_model_handoff_from_cloud(
        cloud,
        sample_source=sample_source,
        object_id_field=object_id_field,
        slots=slots,
        frame_count=frame_count,
        max_points=max_points,
        temporal_offset=temporal_offset,
        image_width=image_width,
        image_height=image_height,
        point_radius=point_radius,
        visibility_policy=visibility_policy,
        seed=seed,
        iterations=iterations,
        learning_rate=learning_rate,
        temperature_candidates=temperature_candidates,
        baseline_temperature=baseline_temperature,
        image_renderer=image_renderer,
        vram_reserve_gb=vram_reserve_gb,
    )
    return real_sample_v2_viewer_preview_from_handoff(
        cloud,
        handoff,
        sample_source=sample_source,
        object_id_field=object_id_field,
        slots=slots,
        seed=seed,
        assignment_feature_weight=assignment_feature_weight,
        assignment_position_weight=assignment_position_weight,
        rewrite_sh=rewrite_sh,
        viewer_path=viewer_path,
    )


def real_sample_v2_viewer_preview_from_handoff(
    cloud: GaussianCloud,
    handoff: RealSampleV2ModelHandoffReport,
    *,
    sample_source: str = "memory://gaussian-cloud",
    object_id_field: str = "object_id",
    slots: int | None = None,
    seed: int = 4,
    assignment_feature_weight: float | None = None,
    assignment_position_weight: float | None = None,
    rewrite_sh: bool = False,
    viewer_path: str | None = None,
) -> RealSampleV2ViewerPreviewReport:
    handoff_summary = handoff.as_dict()
    validate_real_sample_v2_model_handoff_summary(handoff_summary)
    if handoff_summary["status"] != "real_sample_v2_model_handoff_pass":
        raise ValueError("viewer preview requires a passing real sample v2 model handoff")
    sample = trainable_kernel_sample_from_cloud(
        cloud,
        slots=slots,
        frame_count=1,
        max_points=None,
        object_id_field=object_id_field,
        temporal_offset=0.0,
        bind_image_targets=False,
        seed=seed,
    )
    baseline_state = assignment_solver_v2_state_from_checkpoint(handoff.checkpoint)
    state = _assignment_state_with_weights(
        baseline_state,
        feature_weight=assignment_feature_weight,
        position_weight=assignment_position_weight,
    )
    evidence = assignment_evidence_sequence_from_trainable_frames(
        sample.frames,
        source="real_sample_v2_viewer_preview",
    )[0]
    prediction = predict_assignment_solver_v2(evidence, state)
    frame = sample.frames[0]
    projection = project_object_states(
        cloud,
        prediction.assignment,
        evidence_features=frame.features,
    )
    stability = object_state_stability_report(
        projection,
        purity_labels=_target_slots(sample),
    )
    projected_cloud = _projected_viewer_cloud(
        cloud,
        sample=sample,
        prediction=prediction,
        projection=projection,
        object_id_field=object_id_field,
        rewrite_sh=rewrite_sh,
    )
    return RealSampleV2ViewerPreviewReport(
        handoff=handoff,
        sample=sample,
        prediction=prediction,
        projection=projection,
        stability=stability,
        projected_cloud=projected_cloud,
        sample_source=str(sample_source),
        object_id_field=str(object_id_field),
        baseline_feature_weight=float(baseline_state.config.feature_weight),
        baseline_position_weight=float(baseline_state.config.position_weight),
        promoted_feature_weight=float(state.config.feature_weight),
        promoted_position_weight=float(state.config.position_weight),
        viewer_path=viewer_path,
    )


def validate_real_sample_v2_viewer_preview_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("real sample v2 viewer preview summary must be a dict")
    if payload.get("schema") != REAL_SAMPLE_V2_VIEWER_PREVIEW_SCHEMA:
        raise ValueError(f"unsupported real sample v2 viewer preview schema: {payload.get('schema')}")
    if payload.get("kind") != "real_sample_v2_viewer_preview":
        raise ValueError("real sample v2 viewer preview kind is unsupported")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_FAIL}:
        raise ValueError("real sample v2 viewer preview status is unsupported")
    for key in (
        "source",
        "handoff",
        "checkpoint",
        "assignment_weight_policy",
        "projection",
        "quality",
        "viewer",
        "output_policy",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"real sample v2 viewer preview summary missing {key}")
    if payload["checkpoint"].get("schema") != ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA:
        raise ValueError("real sample v2 viewer preview must reference assignment v2 checkpoint")
    weight_policy = payload["assignment_weight_policy"]
    if weight_policy.get("family") != "assignment_v2_cost_weight_promotion":
        raise ValueError("viewer preview assignment weight policy is unsupported")
    for key in (
        "baseline_feature_weight",
        "baseline_position_weight",
        "promoted_feature_weight",
        "promoted_position_weight",
    ):
        if float(weight_policy[key]) < 0.0:
            raise ValueError(f"viewer preview {key} must be >= 0")
    if weight_policy.get("uses_target_labels_for_prediction") is not False:
        raise ValueError("viewer preview promotion must not use target labels for prediction")
    if weight_policy.get("mutates_checkpoint") is not False:
        raise ValueError("viewer preview promotion must not mutate checkpoint")
    source_count = int(payload["source"]["source_gaussians"])
    projection = payload["projection"]
    if int(projection["projected_gaussians"]) != source_count:
        raise ValueError("viewer preview projection must cover the full source cloud")
    if int(projection["exported_gaussians"]) != source_count:
        raise ValueError("viewer preview export must cover the full source cloud")
    if projection.get("export_object_id") != "argmax_assignment_slot":
        raise ValueError("viewer preview must export object_id from argmax assignment")
    hard_segmentation = projection.get("hard_segmentation")
    if not isinstance(hard_segmentation, dict):
        raise ValueError("viewer preview projection missing hard_segmentation")
    if int(hard_segmentation["gaussian_count"]) != source_count:
        raise ValueError("viewer preview hard segmentation must cover the full source cloud")
    if int(hard_segmentation["mixed_gaussians"]) < 0:
        raise ValueError("viewer preview mixed_gaussians must be >= 0")
    quality = payload["quality"]
    if quality.get("status") not in {_QUALITY_PASS, _QUALITY_DIAGNOSTIC}:
        raise ValueError("viewer preview quality status is unsupported")
    for metric in ("mean_normalized_entropy", "assignment_confidence", "direct_slot_match"):
        value = float(quality[metric])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"viewer preview quality {metric} must be in [0, 1]")
    if quality["object_purity"] is not None:
        purity = float(quality["object_purity"])
        if not 0.0 <= purity <= 1.0:
            raise ValueError("viewer preview object_purity must be in [0, 1]")
    viewer = payload["viewer"]
    if viewer.get("route_param") != "ply":
        raise ValueError("viewer preview route param must be ply")
    non_goals = payload["non_goals"]
    if (
        non_goals.get("uses_gpu")
        or non_goals.get("unfreezes_gaussian_geometry")
        or non_goals.get("unfreezes_camera")
        or non_goals.get("mutates_dynamic_k")
        or non_goals.get("uses_rollout_model")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("claims_public_demo_release")
    ):
        raise ValueError("real sample v2 viewer preview summary violates non-goals")
    return payload


def _assignment_state_with_weights(
    state: AssignmentSolverV2State,
    *,
    feature_weight: float | None,
    position_weight: float | None,
) -> AssignmentSolverV2State:
    next_feature_weight = (
        float(state.config.feature_weight)
        if feature_weight is None
        else float(feature_weight)
    )
    next_position_weight = (
        float(state.config.position_weight)
        if position_weight is None
        else float(position_weight)
    )
    if next_feature_weight < 0.0 or next_position_weight < 0.0:
        raise ValueError("viewer preview assignment weights must be >= 0")
    if (
        next_feature_weight == float(state.config.feature_weight)
        and next_position_weight == float(state.config.position_weight)
    ):
        return state
    config = AssignmentSolverV2Config(
        slots=state.config.slots,
        feature_dim=state.config.feature_dim,
        position_dim=state.config.position_dim,
        temperature=state.config.temperature,
        feature_weight=next_feature_weight,
        position_weight=next_position_weight,
        solver_family=state.config.solver_family,
        cost_terms=state.config.cost_terms,
        balance_policy=state.config.balance_policy,
        temporal_policy=state.config.temporal_policy,
        matching_policy=state.config.matching_policy,
    )
    return AssignmentSolverV2State(
        config=config,
        feature_centers=state.feature_centers,
        position_centers=state.position_centers,
        slot_bias=state.slot_bias,
        step=state.step,
        source="real_sample_v2_viewer_preview_weighted_promotion",
        schema=state.schema,
    )


def _projected_viewer_cloud(
    cloud: GaussianCloud,
    *,
    sample: TrainableKernelSample,
    prediction: AssignmentSolverV2Prediction,
    projection: ObjectStateProjection,
    object_id_field: str,
    rewrite_sh: bool,
) -> GaussianCloud:
    if prediction.evidence_count != cloud.count:
        raise ValueError("viewer preview prediction must cover the source cloud")
    if projection.derived_object_ids.shape[0] != cloud.count:
        raise ValueError("viewer preview projection must cover the source cloud")
    if object_id_field not in cloud.fields:
        raise ValueError(f"PLY vertex data has no {object_id_field!r} property")

    vertices = cloud.vertices
    target_object_ids = np.asarray(vertices[object_id_field], dtype=np.int32)
    confidence = np.max(prediction.assignment, axis=1).astype(np.float32, copy=False)
    entropy = _normalized_row_entropy(prediction.assignment).astype(np.float32, copy=False)
    vertices = append_or_replace_property(
        vertices,
        "target_object_id",
        target_object_ids,
        np.int32,
    )
    vertices = append_or_replace_property(
        vertices,
        "target_slot",
        _target_slots(sample),
        np.int32,
    )
    vertices = append_or_replace_property(
        vertices,
        "assignment_confidence",
        confidence,
        np.float32,
    )
    vertices = append_or_replace_property(
        vertices,
        "assignment_entropy",
        entropy,
        np.float32,
    )
    vertices = append_or_replace_property(
        vertices,
        "object_id",
        projection.derived_object_ids.astype(np.int32, copy=False),
        np.int32,
    )
    return apply_object_colors(
        cloud.with_vertices(vertices),
        object_id_field="object_id",
        rewrite_sh=rewrite_sh,
    )


def _hard_segmentation_summary(
    target_slots: np.ndarray,
    predicted_slots: np.ndarray,
) -> dict[str, Any]:
    target = np.asarray(target_slots, dtype=np.int32)
    predicted = np.asarray(predicted_slots, dtype=np.int32)
    if target.shape[0] != predicted.shape[0]:
        raise ValueError("target and predicted slots must have the same length")
    object_counts = _slot_counts(predicted, key="object_id")
    target_counts = _slot_counts(target, key="target_slot")
    mixed = 0
    for object_id in sorted(set(int(value) for value in predicted.tolist())):
        mask = predicted == object_id
        if not np.any(mask):
            continue
        target_values, counts = np.unique(target[mask], return_counts=True)
        mixed += int(np.sum(counts) - np.max(counts))
    return {
        "gaussian_count": int(predicted.shape[0]),
        "object_id_counts": object_counts,
        "target_slot_counts": target_counts,
        "mixed_gaussians": int(mixed),
    }


def _slot_counts(values: np.ndarray, *, key: str) -> list[dict[str, int]]:
    slots, counts = np.unique(np.asarray(values, dtype=np.int32), return_counts=True)
    return [
        {
            key: int(slot),
            "count": int(count),
        }
        for slot, count in zip(slots, counts, strict=True)
    ]


def _target_slots(sample: TrainableKernelSample) -> np.ndarray:
    frame = sample.frames[0]
    if frame.target_assignment is None:
        raise ValueError("viewer preview requires object_id target assignments")
    return np.argmax(frame.target_assignment, axis=1).astype(np.int32, copy=False)


def _normalized_row_entropy(assignment: np.ndarray) -> np.ndarray:
    matrix = np.asarray(assignment, dtype=np.float32)
    if matrix.shape[1] <= 1:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    clipped = np.clip(matrix, 1e-8, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return (entropy / np.log(matrix.shape[1])).astype(np.float32, copy=False)
