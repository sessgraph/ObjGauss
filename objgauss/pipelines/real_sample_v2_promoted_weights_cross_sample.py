from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io_ply import append_or_replace_property
from objgauss.pipelines.real_sample_v2_model_handoff import (
    real_sample_v2_model_handoff_from_cloud,
)
from objgauss.pipelines.real_sample_v2_viewer_preview import (
    REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT,
    REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT,
    RealSampleV2ViewerPreviewReport,
    real_sample_v2_viewer_preview_from_handoff,
    validate_real_sample_v2_viewer_preview_summary,
)
from objgauss.pipelines.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
)

REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA = (
    "objgauss-real-sample-v2-promoted-weights-cross-sample-v1"
)

__all__ = (
    "REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA",
    "RealSampleV2PromotedWeightsCrossSampleReport",
    "real_sample_v2_promoted_weights_cross_sample_from_cloud",
    "validate_real_sample_v2_promoted_weights_cross_sample_summary",
)
_STATUS_PASS = "real_sample_v2_promoted_weights_cross_sample_pass"
_STATUS_DIAGNOSTIC = "real_sample_v2_promoted_weights_cross_sample_diagnostic"


@dataclass(frozen=True)
class RealSampleV2PromotedWeightsCrossSampleReport:
    baseline_preview: RealSampleV2ViewerPreviewReport
    promoted_preview: RealSampleV2ViewerPreviewReport
    sample_source: str
    object_id_field: str
    max_points: int
    solver_temperature: float
    baseline_feature_weight: float
    baseline_position_weight: float
    promoted_feature_weight: float
    promoted_position_weight: float
    reference_sample: str = "public/samples/lego_alpha_v1_objects.ply"
    viewer_path: str | None = None
    schema: str = REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA

    @property
    def promoted_cloud(self) -> GaussianCloud:
        return _with_cross_sample_fields(
            self.promoted_preview.projected_cloud,
            baseline_preview=self.baseline_preview,
            promoted_preview=self.promoted_preview,
        )

    @property
    def passed(self) -> bool:
        baseline = validate_real_sample_v2_viewer_preview_summary(
            self.baseline_preview.as_dict()
        )
        promoted = validate_real_sample_v2_viewer_preview_summary(
            self.promoted_preview.as_dict()
        )
        return _passes_cross_sample_gate(baseline, promoted)

    def as_dict(self) -> dict[str, Any]:
        baseline = validate_real_sample_v2_viewer_preview_summary(
            self.baseline_preview.as_dict()
        )
        promoted = validate_real_sample_v2_viewer_preview_summary(
            self.promoted_preview.as_dict()
        )
        delta = _quality_delta(baseline, promoted)
        changed = _changed_gaussians(self.baseline_preview, self.promoted_preview)
        passed = _passes_cross_sample_gate(baseline, promoted)
        payload = {
            "schema": self.schema,
            "kind": "real_sample_v2_promoted_weights_cross_sample",
            "status": _STATUS_PASS if passed else _STATUS_DIAGNOSTIC,
            "source": {
                "input": self.sample_source,
                "reference_sample": self.reference_sample,
                "source_gaussians": int(self.promoted_preview.projected_cloud.count),
                "object_id_field": self.object_id_field,
                "sample_selection": "second_compatible_public_or_local_object_aware_ply",
                "compatible": True,
            },
            "fixed_target": {
                "max_points": int(self.max_points),
                "solver_temperature": float(self.solver_temperature),
                "coverage_scan": "disabled",
                "temperature_sharpening_scan": "disabled",
                "export_object_id": "argmax_assignment_slot",
            },
            "promotion_policy": {
                "family": "assignment_v2_cost_weight_cross_sample_check",
                "promotion_source": "REAL-SAMPLE-V2-WEAK-BOUNDARY-OPT-001",
                "baseline_feature_weight": float(self.baseline_feature_weight),
                "baseline_position_weight": float(self.baseline_position_weight),
                "promoted_feature_weight": float(self.promoted_feature_weight),
                "promoted_position_weight": float(self.promoted_position_weight),
                "uses_target_labels_for_prediction": False,
                "mutates_checkpoint": False,
            },
            "baseline": baseline,
            "promoted": promoted,
            "quality_delta": delta,
            "changed_gaussians": changed,
            "cross_sample_gate": {
                "hard_mixed_gaussians_non_regression": (
                    int(delta["mixed_gaussians_delta"]) <= 0
                ),
                "direct_slot_match_non_regression": (
                    float(delta["direct_slot_match_delta"]) >= 0.0
                ),
                "soft_purity_non_regression": (
                    delta["object_purity_delta"] is None
                    or float(delta["object_purity_delta"]) >= 0.0
                ),
                "predicted_object_count_stable": (
                    int(delta["predicted_object_count_delta"]) == 0
                ),
                "baseline_viewer_preview_status": baseline["status"],
                "promoted_viewer_preview_status": promoted["status"],
                "result": "pass" if passed else "diagnostic",
            },
            "export_fields": {
                "promoted_preview_fields": [
                    "baseline_object_id",
                    "baseline_assignment_confidence",
                    "baseline_assignment_entropy",
                    "promotion_changed",
                    "promotion_hard_fix",
                    "promotion_hard_regression",
                ],
                "promoted_object_id": "promoted_argmax_assignment_slot",
                "baseline_object_id": "baseline_argmax_assignment_slot",
            },
            "recommendation": _recommendation(passed, delta, changed),
            "viewer": {
                "route_param": "ply",
                "viewer_path": self.viewer_path,
                "debug_route": f"/?ply={self.viewer_path}" if self.viewer_path else None,
                "load_mode": "url-object-aware-ply",
            },
            "output_policy": {
                "preview_ply": "write promoted PLY to /tmp or ignored outputs; do not commit generated preview PLY",
                "summary": "write to /tmp or ignored outputs for cross-sample diagnostics",
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
        return validate_real_sample_v2_promoted_weights_cross_sample_summary(payload)


def real_sample_v2_promoted_weights_cross_sample_from_cloud(
    cloud: GaussianCloud,
    *,
    sample_source: str = "memory://gaussian-cloud",
    object_id_field: str = "object_id",
    slots: int | None = None,
    max_points: int = 128,
    solver_temperature: float = 0.35,
    baseline_feature_weight: float = 1.0,
    baseline_position_weight: float = 1.0,
    promoted_feature_weight: float = REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT,
    promoted_position_weight: float = REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT,
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
    reference_sample: str = "public/samples/lego_alpha_v1_objects.ply",
) -> RealSampleV2PromotedWeightsCrossSampleReport:
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")
    if object_id_field not in cloud.fields:
        raise ValueError(f"PLY vertex data has no {object_id_field!r} property")
    if max_points < 1:
        raise ValueError("max_points must be >= 1")
    if solver_temperature <= 0.0:
        raise ValueError("solver_temperature must be > 0")
    for name, value in (
        ("baseline_feature_weight", baseline_feature_weight),
        ("baseline_position_weight", baseline_position_weight),
        ("promoted_feature_weight", promoted_feature_weight),
        ("promoted_position_weight", promoted_position_weight),
    ):
        if float(value) < 0.0:
            raise ValueError(f"{name} must be >= 0")

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
        temperature_candidates=(float(solver_temperature),),
        baseline_temperature=baseline_temperature,
        image_renderer=image_renderer,
        vram_reserve_gb=vram_reserve_gb,
    )
    baseline_preview = real_sample_v2_viewer_preview_from_handoff(
        cloud,
        handoff,
        sample_source=sample_source,
        object_id_field=object_id_field,
        slots=slots,
        seed=seed,
        assignment_feature_weight=baseline_feature_weight,
        assignment_position_weight=baseline_position_weight,
        rewrite_sh=rewrite_sh,
        viewer_path=None,
    )
    promoted_preview = real_sample_v2_viewer_preview_from_handoff(
        cloud,
        handoff,
        sample_source=sample_source,
        object_id_field=object_id_field,
        slots=slots,
        seed=seed,
        assignment_feature_weight=promoted_feature_weight,
        assignment_position_weight=promoted_position_weight,
        rewrite_sh=rewrite_sh,
        viewer_path=viewer_path,
    )
    return RealSampleV2PromotedWeightsCrossSampleReport(
        baseline_preview=baseline_preview,
        promoted_preview=promoted_preview,
        sample_source=str(sample_source),
        object_id_field=str(object_id_field),
        max_points=int(max_points),
        solver_temperature=float(solver_temperature),
        baseline_feature_weight=float(baseline_feature_weight),
        baseline_position_weight=float(baseline_position_weight),
        promoted_feature_weight=float(promoted_feature_weight),
        promoted_position_weight=float(promoted_position_weight),
        reference_sample=str(reference_sample),
        viewer_path=viewer_path,
    )


def validate_real_sample_v2_promoted_weights_cross_sample_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("real sample v2 promoted weights cross-sample summary must be a dict")
    if payload.get("schema") != REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA:
        raise ValueError(f"unsupported promoted weights cross-sample schema: {payload.get('schema')}")
    if payload.get("kind") != "real_sample_v2_promoted_weights_cross_sample":
        raise ValueError("promoted weights cross-sample kind is unsupported")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_DIAGNOSTIC}:
        raise ValueError("promoted weights cross-sample status is unsupported")
    for key in (
        "source",
        "fixed_target",
        "promotion_policy",
        "baseline",
        "promoted",
        "quality_delta",
        "changed_gaussians",
        "cross_sample_gate",
        "export_fields",
        "recommendation",
        "viewer",
        "output_policy",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"promoted weights cross-sample summary missing {key}")

    baseline = validate_real_sample_v2_viewer_preview_summary(payload["baseline"])
    promoted = validate_real_sample_v2_viewer_preview_summary(payload["promoted"])
    source = payload["source"]
    source_count = int(source["source_gaussians"])
    if source_count < 1:
        raise ValueError("promoted weights cross-sample source_gaussians must be positive")
    if source.get("compatible") is not True:
        raise ValueError("promoted weights cross-sample requires a compatible sample")
    if int(baseline["source"]["source_gaussians"]) != source_count:
        raise ValueError("baseline source count must match cross-sample source count")
    if int(promoted["source"]["source_gaussians"]) != source_count:
        raise ValueError("promoted source count must match cross-sample source count")

    fixed = payload["fixed_target"]
    if int(fixed["max_points"]) < 1:
        raise ValueError("promoted weights cross-sample max_points must be >= 1")
    if float(fixed["solver_temperature"]) <= 0.0:
        raise ValueError("promoted weights cross-sample solver_temperature must be > 0")
    if fixed.get("coverage_scan") != "disabled":
        raise ValueError("promoted weights cross-sample must keep coverage scan disabled")
    if fixed.get("temperature_sharpening_scan") != "disabled":
        raise ValueError("promoted weights cross-sample must keep sharpening scan disabled")

    policy = payload["promotion_policy"]
    if policy.get("family") != "assignment_v2_cost_weight_cross_sample_check":
        raise ValueError("promoted weights cross-sample policy is unsupported")
    for key in (
        "baseline_feature_weight",
        "baseline_position_weight",
        "promoted_feature_weight",
        "promoted_position_weight",
    ):
        if float(policy[key]) < 0.0:
            raise ValueError(f"promoted weights cross-sample {key} must be >= 0")
    if policy.get("uses_target_labels_for_prediction") is not False:
        raise ValueError("promoted weights cross-sample must not use target labels for prediction")
    if policy.get("mutates_checkpoint") is not False:
        raise ValueError("promoted weights cross-sample must not mutate checkpoints")

    delta = payload["quality_delta"]
    int(delta["mixed_gaussians_delta"])
    float(delta["direct_slot_match_delta"])
    float(delta["assignment_confidence_delta"])
    float(delta["mean_normalized_entropy_delta"])
    int(delta["predicted_object_count_delta"])
    if delta["object_purity_delta"] is not None:
        float(delta["object_purity_delta"])
    changed = payload["changed_gaussians"]
    if int(changed["changed_count"]) < 0:
        raise ValueError("promoted weights cross-sample changed_count must be >= 0")
    if int(changed["hard_regression_count"]) < 0 or int(changed["hard_fix_count"]) < 0:
        raise ValueError("promoted weights cross-sample change counts must be >= 0")

    gate = payload["cross_sample_gate"]
    expected_pass = _passes_cross_sample_gate(baseline, promoted)
    if gate.get("result") != ("pass" if expected_pass else "diagnostic"):
        raise ValueError("promoted weights cross-sample gate result is inconsistent")
    if payload["status"] == _STATUS_PASS and not expected_pass:
        raise ValueError("passing promoted weights cross-sample summary violates gate")

    recommendation = payload["recommendation"]
    if recommendation.get("requires_geometry_unfreeze") is not False:
        raise ValueError("promoted weights cross-sample must not recommend geometry unfreeze")
    if recommendation.get("requires_diffusion_replay_or_rollout") is not False:
        raise ValueError("promoted weights cross-sample must not recommend diffusion/replay/rollout")

    viewer = payload["viewer"]
    if viewer.get("route_param") != "ply":
        raise ValueError("promoted weights cross-sample viewer route param must be ply")
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
        raise ValueError("promoted weights cross-sample summary violates non-goals")
    return payload


def _passes_cross_sample_gate(
    baseline: dict[str, Any],
    promoted: dict[str, Any],
) -> bool:
    delta = _quality_delta(baseline, promoted)
    soft_purity_delta = delta["object_purity_delta"]
    return (
        baseline["status"] == "real_sample_v2_viewer_preview_pass"
        and promoted["status"] == "real_sample_v2_viewer_preview_pass"
        and promoted["quality"]["status"] == "full_cloud_objectstate_preview_quality_pass"
        and int(delta["mixed_gaussians_delta"]) <= 0
        and float(delta["direct_slot_match_delta"]) >= 0.0
        and int(delta["predicted_object_count_delta"]) == 0
        and (soft_purity_delta is None or float(soft_purity_delta) >= 0.0)
    )


def _quality_delta(
    baseline: dict[str, Any],
    promoted: dict[str, Any],
) -> dict[str, Any]:
    baseline_quality = baseline["quality"]
    promoted_quality = promoted["quality"]
    baseline_hard = baseline["projection"]["hard_segmentation"]
    promoted_hard = promoted["projection"]["hard_segmentation"]
    baseline_purity = baseline_quality["object_purity"]
    promoted_purity = promoted_quality["object_purity"]
    return {
        "mixed_gaussians_delta": int(promoted_hard["mixed_gaussians"])
        - int(baseline_hard["mixed_gaussians"]),
        "direct_slot_match_delta": float(promoted_quality["direct_slot_match"])
        - float(baseline_quality["direct_slot_match"]),
        "object_purity_delta": (
            None
            if baseline_purity is None or promoted_purity is None
            else float(promoted_purity) - float(baseline_purity)
        ),
        "assignment_confidence_delta": float(promoted_quality["assignment_confidence"])
        - float(baseline_quality["assignment_confidence"]),
        "mean_normalized_entropy_delta": float(promoted_quality["mean_normalized_entropy"])
        - float(baseline_quality["mean_normalized_entropy"]),
        "predicted_object_count_delta": int(promoted["projection"]["predicted_object_count"])
        - int(baseline["projection"]["predicted_object_count"]),
    }


def _changed_gaussians(
    baseline_preview: RealSampleV2ViewerPreviewReport,
    promoted_preview: RealSampleV2ViewerPreviewReport,
) -> dict[str, Any]:
    baseline_ids = baseline_preview.projection.derived_object_ids.astype(np.int32, copy=False)
    promoted_ids = promoted_preview.projection.derived_object_ids.astype(np.int32, copy=False)
    target_slots = np.asarray(
        promoted_preview.projected_cloud.vertices["target_slot"],
        dtype=np.int32,
    )
    if baseline_ids.shape != promoted_ids.shape or baseline_ids.shape != target_slots.shape:
        raise ValueError("baseline, promoted, and target slots must have matching shapes")
    changed = baseline_ids != promoted_ids
    hard_fix = changed & (baseline_ids != target_slots) & (promoted_ids == target_slots)
    hard_regression = changed & (baseline_ids == target_slots) & (promoted_ids != target_slots)
    pairs = []
    if np.any(changed):
        pair_values, counts = np.unique(
            np.column_stack([baseline_ids[changed], promoted_ids[changed]]),
            axis=0,
            return_counts=True,
        )
        pairs = [
            {
                "baseline_object_id": int(pair[0]),
                "promoted_object_id": int(pair[1]),
                "count": int(count),
            }
            for pair, count in zip(pair_values, counts, strict=True)
        ]
    return {
        "changed_count": int(np.sum(changed)),
        "changed_fraction": float(np.mean(changed)) if changed.size else 0.0,
        "hard_fix_count": int(np.sum(hard_fix)),
        "hard_regression_count": int(np.sum(hard_regression)),
        "unchanged_count": int(changed.shape[0] - np.sum(changed)),
        "pairs": pairs,
    }


def _with_cross_sample_fields(
    cloud: GaussianCloud,
    *,
    baseline_preview: RealSampleV2ViewerPreviewReport,
    promoted_preview: RealSampleV2ViewerPreviewReport,
) -> GaussianCloud:
    vertices = cloud.vertices
    baseline_vertices = baseline_preview.projected_cloud.vertices
    baseline_ids = baseline_preview.projection.derived_object_ids.astype(np.int32, copy=False)
    promoted_ids = promoted_preview.projection.derived_object_ids.astype(np.int32, copy=False)
    target_slots = np.asarray(vertices["target_slot"], dtype=np.int32)
    if cloud.count != baseline_ids.shape[0] or cloud.count != promoted_ids.shape[0]:
        raise ValueError("cross-sample exported fields must match cloud count")
    changed = baseline_ids != promoted_ids
    hard_fix = changed & (baseline_ids != target_slots) & (promoted_ids == target_slots)
    hard_regression = changed & (baseline_ids == target_slots) & (promoted_ids != target_slots)
    vertices = append_or_replace_property(
        vertices,
        "baseline_object_id",
        baseline_ids,
        np.int32,
    )
    vertices = append_or_replace_property(
        vertices,
        "baseline_assignment_confidence",
        np.asarray(baseline_vertices["assignment_confidence"], dtype=np.float32),
        np.float32,
    )
    vertices = append_or_replace_property(
        vertices,
        "baseline_assignment_entropy",
        np.asarray(baseline_vertices["assignment_entropy"], dtype=np.float32),
        np.float32,
    )
    vertices = append_or_replace_property(
        vertices,
        "promotion_changed",
        changed.astype(np.uint8, copy=False),
        np.uint8,
    )
    vertices = append_or_replace_property(
        vertices,
        "promotion_hard_fix",
        hard_fix.astype(np.uint8, copy=False),
        np.uint8,
    )
    vertices = append_or_replace_property(
        vertices,
        "promotion_hard_regression",
        hard_regression.astype(np.uint8, copy=False),
        np.uint8,
    )
    return cloud.with_vertices(vertices)


def _recommendation(
    passed: bool,
    delta: dict[str, Any],
    changed: dict[str, Any],
) -> dict[str, Any]:
    hard_regressed = (
        int(delta["mixed_gaussians_delta"]) > 0
        or float(delta["direct_slot_match_delta"]) < 0.0
        or int(changed["hard_regression_count"]) > int(changed["hard_fix_count"])
    )
    soft_improved = (
        (delta["object_purity_delta"] is not None and float(delta["object_purity_delta"]) > 0.0)
        or float(delta["assignment_confidence_delta"]) > 0.0
    )
    if passed:
        decision = "promoted_weights_cross_sample_non_regression_pass"
        action = "keep_promoted_weights_and_add_more_cross_sample_rows"
        global_default = "eligible_after_additional_cross_sample_rows"
    elif hard_regressed and soft_improved:
        decision = "hold_promoted_weights_for_global_default"
        action = "keep_as_sample_specific_viewer_preview_candidate_and_add_evidence_normalization_gate"
        global_default = "not_yet_stable_across_samples"
    else:
        decision = "do_not_promote_weights_globally"
        action = "revisit_assignment_weight_policy_before_more_viewer_defaults"
        global_default = "blocked_by_cross_sample_diagnostic"
    return {
        "decision": decision,
        "action": action,
        "global_default": global_default,
        "requires_more_cross_sample_rows": True,
        "requires_more_coverage": False,
        "requires_temperature_sharpening": False,
        "requires_geometry_unfreeze": False,
        "requires_diffusion_replay_or_rollout": False,
    }
