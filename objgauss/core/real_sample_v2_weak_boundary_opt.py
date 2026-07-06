from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.assignment_evidence import (
    assignment_evidence_sequence_from_trainable_frames,
)
from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
    predict_assignment_solver_v2,
)
from objgauss.core.assignment_solver_v2_eval import (
    assignment_solver_v2_state_from_checkpoint,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io_ply import append_or_replace_property
from objgauss.core.object_state import project_object_states
from objgauss.core.real_sample_v2_full_cloud_purity import (
    real_sample_v2_full_cloud_purity_from_cloud,
)
from objgauss.core.real_sample_v2_segmentation_quality import (
    RealSampleV2SegmentationQualityReport,
    real_sample_v2_segmentation_quality_from_projected_cloud,
    real_sample_v2_segmentation_quality_from_purity_report,
    validate_real_sample_v2_segmentation_quality_summary,
)
from objgauss.core.real_sample_v2_viewer_preview import (
    _projected_viewer_cloud,
)
from objgauss.core.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
)

REAL_SAMPLE_V2_WEAK_BOUNDARY_OPT_SCHEMA = "objgauss-real-sample-v2-weak-boundary-opt-v1"
_STATUS_PASS = "real_sample_v2_weak_boundary_opt_pass"
_STATUS_DIAGNOSTIC = "real_sample_v2_weak_boundary_opt_diagnostic"


@dataclass(frozen=True)
class RealSampleV2WeakBoundaryOptReport:
    baseline_quality: RealSampleV2SegmentationQualityReport
    candidate_quality: RealSampleV2SegmentationQualityReport
    baseline_predicted_object_ids: np.ndarray
    candidate_predicted_object_ids: np.ndarray
    baseline_confidence: np.ndarray
    baseline_entropy: np.ndarray
    candidate_feature_weight: float
    candidate_position_weight: float
    sample_source: str
    max_points: int
    solver_temperature: float
    viewer_path: str | None = None
    schema: str = REAL_SAMPLE_V2_WEAK_BOUNDARY_OPT_SCHEMA

    @property
    def candidate_cloud(self) -> GaussianCloud:
        return self.candidate_quality.projected_cloud

    @property
    def passed(self) -> bool:
        delta = _quality_delta(
            self.baseline_quality.as_dict(),
            self.candidate_quality.as_dict(),
        )
        return (
            self.candidate_quality.as_dict()["status"]
            == "real_sample_v2_segmentation_quality_pass"
            and int(delta["mixed_gaussians_delta"]) < 0
            and float(delta["direct_slot_match_delta"]) > 0.0
        )

    def as_dict(self) -> dict[str, Any]:
        baseline = validate_real_sample_v2_segmentation_quality_summary(
            self.baseline_quality.as_dict()
        )
        candidate = validate_real_sample_v2_segmentation_quality_summary(
            self.candidate_quality.as_dict()
        )
        delta = _quality_delta(baseline, candidate)
        changed = _changed_gaussians(
            self.baseline_predicted_object_ids,
            self.candidate_predicted_object_ids,
        )
        payload = {
            "schema": self.schema,
            "kind": "real_sample_v2_weak_boundary_opt",
            "status": _STATUS_PASS if self.passed else _STATUS_DIAGNOSTIC,
            "source": {
                "input": self.sample_source,
                "source_gaussians": int(self.candidate_cloud.count),
            },
            "fixed_target": {
                "max_points": int(self.max_points),
                "solver_temperature": float(self.solver_temperature),
                "coverage_scan": "disabled",
                "temperature_sharpening_scan": "disabled",
                "export_object_id": "argmax_assignment_slot",
            },
            "candidate_policy": {
                "family": "assignment_v2_cost_weight_normalization",
                "feature_weight": float(self.candidate_feature_weight),
                "position_weight": float(self.candidate_position_weight),
                "baseline_feature_weight": 1.0,
                "baseline_position_weight": 1.0,
                "uses_target_labels_for_prediction": False,
                "mutates_checkpoint": False,
            },
            "baseline": baseline,
            "candidate": candidate,
            "quality_delta": delta,
            "changed_gaussians": changed,
            "export_fields": {
                "candidate_fields": [
                    "baseline_object_id",
                    "baseline_assignment_confidence",
                    "baseline_assignment_entropy",
                    "weak_boundary_candidate",
                    "boundary_changed",
                ],
                "candidate_object_id": "candidate_argmax_assignment_slot",
                "baseline_object_id": "baseline_argmax_assignment_slot",
            },
            "recommendation": _recommendation(delta, changed, candidate),
            "viewer": {
                "route_param": "ply",
                "viewer_path": self.viewer_path,
                "debug_route": f"/?ply={self.viewer_path}" if self.viewer_path else None,
                "load_mode": "url-object-aware-ply",
            },
            "output_policy": {
                "preview_ply": "write candidate PLY to /tmp or ignored outputs; do not commit generated preview PLY",
                "summary": "write to /tmp or ignored outputs for weak-boundary diagnostics",
                "screenshots": "write browser evidence to /tmp only",
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
        return validate_real_sample_v2_weak_boundary_opt_summary(payload)


def real_sample_v2_weak_boundary_opt_from_cloud(
    cloud: GaussianCloud,
    *,
    sample_source: str = "memory://gaussian-cloud",
    object_id_field: str = "object_id",
    slots: int | None = None,
    max_points: int = 128,
    solver_temperature: float = 0.35,
    candidate_feature_weight: float = 2.0,
    candidate_position_weight: float = 1.0,
    frame_count: int = 2,
    temporal_offset: float = 0.01,
    image_width: int = 12,
    image_height: int = 12,
    point_radius: int = 1,
    visibility_policy: str = "covered_pixels",
    seed: int = 4,
    iterations: int = 100,
    learning_rate: float = 0.4,
    baseline_temperature: float = 1.0,
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    vram_reserve_gb: int = 1,
    rewrite_sh: bool = False,
    viewer_path: str | None = None,
) -> RealSampleV2WeakBoundaryOptReport:
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")
    if max_points < 1:
        raise ValueError("max_points must be >= 1")
    if solver_temperature <= 0.0:
        raise ValueError("solver_temperature must be > 0")
    if candidate_feature_weight < 0.0 or candidate_position_weight < 0.0:
        raise ValueError("candidate weights must be >= 0")
    coverage = real_sample_v2_full_cloud_purity_from_cloud(
        cloud,
        sample_source=sample_source,
        object_id_field=object_id_field,
        slots=slots,
        max_point_candidates=(int(max_points),),
        frame_count=frame_count,
        temporal_offset=temporal_offset,
        image_width=image_width,
        image_height=image_height,
        point_radius=point_radius,
        visibility_policy=visibility_policy,
        seed=seed,
        iterations=iterations,
        learning_rate=learning_rate,
        temperature_candidates=(float(solver_temperature),),
        baseline_temperature=baseline_temperature,
        image_renderer=image_renderer,
        vram_reserve_gb=vram_reserve_gb,
        rewrite_sh=rewrite_sh,
        viewer_path=viewer_path,
    )
    baseline_quality = real_sample_v2_segmentation_quality_from_purity_report(
        coverage,
        sample_source=sample_source,
        viewer_path=viewer_path,
    )
    baseline_viewer = coverage.best_candidate
    candidate_cloud, candidate_prediction = _candidate_projected_cloud(
        cloud,
        baseline_viewer=baseline_viewer,
        feature_weight=candidate_feature_weight,
        position_weight=candidate_position_weight,
        object_id_field=object_id_field,
        rewrite_sh=rewrite_sh,
    )
    candidate_cloud = _with_boundary_fields(
        candidate_cloud,
        baseline_predicted_object_ids=baseline_viewer.projection.derived_object_ids,
        candidate_predicted_object_ids=np.argmax(candidate_prediction.assignment, axis=1).astype(
            np.int32,
            copy=False,
        ),
        baseline_confidence=np.max(baseline_viewer.prediction.assignment, axis=1),
        baseline_entropy=_normalized_row_entropy(baseline_viewer.prediction.assignment),
    )
    candidate_quality = real_sample_v2_segmentation_quality_from_projected_cloud(
        candidate_cloud,
        sample_source=sample_source,
        max_points=max_points,
        solver_temperature=solver_temperature,
        viewer_path=viewer_path,
    )
    return RealSampleV2WeakBoundaryOptReport(
        baseline_quality=baseline_quality,
        candidate_quality=candidate_quality,
        baseline_predicted_object_ids=baseline_viewer.projection.derived_object_ids.astype(
            np.int32,
            copy=False,
        ),
        candidate_predicted_object_ids=np.argmax(candidate_prediction.assignment, axis=1).astype(
            np.int32,
            copy=False,
        ),
        baseline_confidence=np.max(baseline_viewer.prediction.assignment, axis=1).astype(
            np.float32,
            copy=False,
        ),
        baseline_entropy=_normalized_row_entropy(baseline_viewer.prediction.assignment).astype(
            np.float32,
            copy=False,
        ),
        candidate_feature_weight=float(candidate_feature_weight),
        candidate_position_weight=float(candidate_position_weight),
        sample_source=str(sample_source),
        max_points=int(max_points),
        solver_temperature=float(solver_temperature),
        viewer_path=viewer_path,
    )


def validate_real_sample_v2_weak_boundary_opt_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("real sample v2 weak-boundary opt summary must be a dict")
    if payload.get("schema") != REAL_SAMPLE_V2_WEAK_BOUNDARY_OPT_SCHEMA:
        raise ValueError(f"unsupported weak-boundary opt schema: {payload.get('schema')}")
    if payload.get("kind") != "real_sample_v2_weak_boundary_opt":
        raise ValueError("weak-boundary opt kind is unsupported")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_DIAGNOSTIC}:
        raise ValueError("weak-boundary opt status is unsupported")
    for key in (
        "source",
        "fixed_target",
        "candidate_policy",
        "baseline",
        "candidate",
        "quality_delta",
        "changed_gaussians",
        "export_fields",
        "recommendation",
        "viewer",
        "output_policy",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"weak-boundary opt summary missing {key}")
    validate_real_sample_v2_segmentation_quality_summary(payload["baseline"])
    validate_real_sample_v2_segmentation_quality_summary(payload["candidate"])
    fixed = payload["fixed_target"]
    if int(fixed["max_points"]) < 1:
        raise ValueError("weak-boundary max_points must be >= 1")
    if float(fixed["solver_temperature"]) <= 0.0:
        raise ValueError("weak-boundary solver_temperature must be > 0")
    if fixed.get("coverage_scan") != "disabled":
        raise ValueError("weak-boundary opt must keep coverage scan disabled")
    if fixed.get("temperature_sharpening_scan") != "disabled":
        raise ValueError("weak-boundary opt must keep sharpening scan disabled")
    policy = payload["candidate_policy"]
    if policy.get("family") != "assignment_v2_cost_weight_normalization":
        raise ValueError("weak-boundary candidate policy is unsupported")
    if float(policy["feature_weight"]) < 0.0 or float(policy["position_weight"]) < 0.0:
        raise ValueError("weak-boundary candidate weights must be >= 0")
    if policy.get("uses_target_labels_for_prediction") is not False:
        raise ValueError("weak-boundary opt must not use target labels for prediction")
    changed = payload["changed_gaussians"]
    if int(changed["changed_count"]) < 0:
        raise ValueError("weak-boundary changed_count must be >= 0")
    delta = payload["quality_delta"]
    float(delta["direct_slot_match_delta"])
    int(delta["mixed_gaussians_delta"])
    recommendation = payload["recommendation"]
    if recommendation.get("requires_geometry_unfreeze") is not False:
        raise ValueError("weak-boundary opt must not recommend geometry unfreeze")
    if recommendation.get("requires_diffusion_replay_or_rollout") is not False:
        raise ValueError("weak-boundary opt must not recommend diffusion/replay/rollout")
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
        raise ValueError("weak-boundary opt summary violates non-goals")
    return payload


def _candidate_projected_cloud(
    cloud: GaussianCloud,
    *,
    baseline_viewer: Any,
    feature_weight: float,
    position_weight: float,
    object_id_field: str,
    rewrite_sh: bool,
) -> tuple[GaussianCloud, Any]:
    state = assignment_solver_v2_state_from_checkpoint(baseline_viewer.handoff.checkpoint)
    config = AssignmentSolverV2Config(
        slots=state.config.slots,
        feature_dim=state.config.feature_dim,
        position_dim=state.config.position_dim,
        temperature=state.config.temperature,
        feature_weight=float(feature_weight),
        position_weight=float(position_weight),
    )
    candidate_state = AssignmentSolverV2State(
        config=config,
        feature_centers=state.feature_centers,
        position_centers=state.position_centers,
        slot_bias=state.slot_bias,
        step=state.step,
        source="real_sample_v2_weak_boundary_cost_weight_candidate",
    )
    evidence = assignment_evidence_sequence_from_trainable_frames(
        baseline_viewer.sample.frames,
        source="real_sample_v2_weak_boundary_opt",
    )[0]
    prediction = predict_assignment_solver_v2(evidence, candidate_state)
    frame = baseline_viewer.sample.frames[0]
    projection = project_object_states(
        cloud,
        prediction.assignment,
        evidence_features=frame.features,
    )
    projected_cloud = _projected_viewer_cloud(
        cloud,
        sample=baseline_viewer.sample,
        prediction=prediction,
        projection=projection,
        object_id_field=object_id_field,
        rewrite_sh=rewrite_sh,
    )
    return projected_cloud, prediction


def _with_boundary_fields(
    cloud: GaussianCloud,
    *,
    baseline_predicted_object_ids: np.ndarray,
    candidate_predicted_object_ids: np.ndarray,
    baseline_confidence: np.ndarray,
    baseline_entropy: np.ndarray,
) -> GaussianCloud:
    if cloud.count != baseline_predicted_object_ids.shape[0]:
        raise ValueError("baseline object ids must match cloud count")
    changed = baseline_predicted_object_ids != candidate_predicted_object_ids
    weak_boundary = (
        changed
        | (np.asarray(baseline_confidence) < 0.65)
        | (np.asarray(baseline_entropy) > 0.5)
    )
    vertices = cloud.vertices
    vertices = append_or_replace_property(
        vertices,
        "baseline_object_id",
        baseline_predicted_object_ids.astype(np.int32, copy=False),
        np.int32,
    )
    vertices = append_or_replace_property(
        vertices,
        "baseline_assignment_confidence",
        np.asarray(baseline_confidence, dtype=np.float32),
        np.float32,
    )
    vertices = append_or_replace_property(
        vertices,
        "baseline_assignment_entropy",
        np.asarray(baseline_entropy, dtype=np.float32),
        np.float32,
    )
    vertices = append_or_replace_property(
        vertices,
        "weak_boundary_candidate",
        weak_boundary.astype(np.uint8, copy=False),
        np.uint8,
    )
    vertices = append_or_replace_property(
        vertices,
        "boundary_changed",
        changed.astype(np.uint8, copy=False),
        np.uint8,
    )
    return cloud.with_vertices(vertices)


def _quality_delta(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    baseline_quality = baseline["global_quality"]
    candidate_quality = candidate["global_quality"]
    return {
        "direct_slot_match_delta": (
            float(candidate_quality["direct_slot_match"])
            - float(baseline_quality["direct_slot_match"])
        ),
        "hard_argmax_object_purity_delta": (
            float(candidate_quality["hard_argmax_object_purity"])
            - float(baseline_quality["hard_argmax_object_purity"])
        ),
        "min_predicted_object_purity_delta": (
            float(candidate_quality["min_predicted_object_purity"])
            - float(baseline_quality["min_predicted_object_purity"])
        ),
        "min_target_recall_delta": (
            float(candidate_quality["min_target_recall"])
            - float(baseline_quality["min_target_recall"])
        ),
        "mixed_gaussians_delta": (
            int(candidate_quality["mixed_gaussians"])
            - int(baseline_quality["mixed_gaussians"])
        ),
        "baseline_mixed_gaussians": int(baseline_quality["mixed_gaussians"]),
        "candidate_mixed_gaussians": int(candidate_quality["mixed_gaussians"]),
    }


def _changed_gaussians(
    baseline: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, Any]:
    if baseline.shape[0] != candidate.shape[0]:
        raise ValueError("baseline and candidate predictions must have the same length")
    changed = baseline != candidate
    pairs: dict[tuple[int, int], int] = {}
    for left, right in zip(baseline[changed], candidate[changed], strict=True):
        key = (int(left), int(right))
        pairs[key] = pairs.get(key, 0) + 1
    return {
        "changed_count": int(np.sum(changed)),
        "changed_fraction": float(np.mean(changed)) if changed.size else 0.0,
        "pairs": [
            {
                "baseline_object_id": int(left),
                "candidate_object_id": int(right),
                "count": int(count),
            }
            for (left, right), count in sorted(pairs.items())
        ],
    }


def _recommendation(
    delta: dict[str, Any],
    changed: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    improved = int(delta["mixed_gaussians_delta"]) < 0 and float(delta["direct_slot_match_delta"]) > 0.0
    if improved:
        decision = "promote_cost_weight_normalization_candidate"
        action = "use_feature_weight_boost_for_next_viewer_preview"
        export_policy = "keep_boundary_fields_for_audit"
    else:
        decision = "keep_baseline_and_export_uncertainty"
        action = "do_not_promote_weight_candidate"
        export_policy = "mark_weak_boundary_without_reassigning"
    return {
        "decision": decision,
        "action": action,
        "export_policy": export_policy,
        "candidate_status": candidate["status"],
        "changed_gaussians": int(changed["changed_count"]),
        "requires_more_coverage": False,
        "requires_temperature_sharpening": False,
        "requires_geometry_unfreeze": False,
        "requires_diffusion_replay_or_rollout": False,
    }


def _normalized_row_entropy(assignment: np.ndarray) -> np.ndarray:
    matrix = np.asarray(assignment, dtype=np.float32)
    if matrix.shape[1] <= 1:
        return np.zeros(matrix.shape[0], dtype=np.float32)
    clipped = np.clip(matrix, 1e-8, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return (entropy / np.log(matrix.shape[1])).astype(np.float32, copy=False)
