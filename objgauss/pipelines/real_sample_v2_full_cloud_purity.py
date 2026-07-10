from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

from objgauss.core.gaussian import GaussianCloud
from objgauss.pipelines.real_sample_v2_viewer_preview import (
    RealSampleV2ViewerPreviewReport,
    real_sample_v2_viewer_preview_from_cloud,
    validate_real_sample_v2_viewer_preview_summary,
)
from objgauss.pipelines.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
)

REAL_SAMPLE_V2_FULL_CLOUD_PURITY_SCHEMA = "objgauss-real-sample-v2-full-cloud-purity-v1"

__all__ = (
    "REAL_SAMPLE_V2_FULL_CLOUD_PURITY_SCHEMA",
    "RealSampleV2FullCloudPurityReport",
    "real_sample_v2_full_cloud_purity_from_cloud",
    "validate_real_sample_v2_full_cloud_purity_summary",
)
_STATUS_PASS = "real_sample_v2_full_cloud_purity_pass"
_STATUS_FAIL = "real_sample_v2_full_cloud_purity_fail"
_QUALITY_PASS = "full_cloud_objectstate_preview_quality_pass"
_QUALITY_DIAGNOSTIC = "full_cloud_objectstate_preview_quality_diagnostic"


@dataclass(frozen=True)
class RealSampleV2FullCloudPurityReport:
    candidates: tuple[RealSampleV2ViewerPreviewReport, ...]
    max_point_candidates: tuple[int, ...]
    sample_source: str
    object_id_field: str
    schema: str = REAL_SAMPLE_V2_FULL_CLOUD_PURITY_SCHEMA

    @property
    def best_index(self) -> int:
        records = [
            _candidate_record(report, max_points)
            for report, max_points in zip(self.candidates, self.max_point_candidates, strict=True)
        ]
        return max(range(len(records)), key=lambda index: _candidate_rank(records[index]))

    @property
    def best_candidate(self) -> RealSampleV2ViewerPreviewReport:
        return self.candidates[self.best_index]

    @property
    def passed(self) -> bool:
        best = _candidate_record(
            self.best_candidate,
            self.max_point_candidates[self.best_index],
        )
        return (
            self.best_candidate.passed
            and best["quality"]["status"] == _QUALITY_PASS
        )

    def as_dict(self) -> dict[str, Any]:
        if not self.candidates:
            raise ValueError("full-cloud purity report requires at least one candidate")
        records = [
            _candidate_record(report, max_points)
            for report, max_points in zip(self.candidates, self.max_point_candidates, strict=True)
        ]
        best_index = max(range(len(records)), key=lambda index: _candidate_rank(records[index]))
        baseline = records[0]
        best = records[best_index]
        payload = {
            "schema": self.schema,
            "kind": "real_sample_v2_full_cloud_purity",
            "status": _STATUS_PASS if self.passed else _STATUS_FAIL,
            "source": {
                "input": self.sample_source,
                "source_gaussians": int(best["source_gaussians"]),
                "object_id_field": self.object_id_field,
                "target_source": best["target_source"],
                "object_id_mapping": dict(best["object_id_mapping"]),
            },
            "segmentation_target": {
                "target": "object_id_one_hot_segmentation",
                "scan_axis": "max_points",
                "max_point_candidates": list(self.max_point_candidates),
                "selected_max_points": int(best["max_points"]),
                "selected_solver_temperature": float(best["solver_temperature"]),
                "selection_policy": (
                    "highest_full_cloud_object_purity_then_direct_slot_match_then_confidence"
                ),
                "export_object_id": "argmax_assignment_slot",
            },
            "baseline_candidate": baseline,
            "best_candidate": best,
            "candidate_count": len(records),
            "coverage_sweep": records,
            "quality_delta": _quality_delta(baseline, best),
            "recommendation": _recommendation(baseline, best),
            "viewer": {
                "route_param": "ply",
                "viewer_path": best["viewer_path"],
                "debug_route": best["debug_route"],
                "load_mode": "url-object-aware-ply",
            },
            "output_policy": {
                "preview_ply": "write best candidate to /tmp or ignored outputs; do not commit generated preview PLY",
                "summary": "write to /tmp or ignored outputs for segmentation target diagnostics",
                "checkpoint": "regenerate or read handoff checkpoint; do not commit training checkpoints",
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
        return validate_real_sample_v2_full_cloud_purity_summary(payload)


def real_sample_v2_full_cloud_purity_from_cloud(
    cloud: GaussianCloud,
    *,
    sample_source: str = "memory://gaussian-cloud",
    object_id_field: str = "object_id",
    slots: int | None = None,
    max_point_candidates: Sequence[int] = (24, 64, 128),
    frame_count: int = 2,
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
    rewrite_sh: bool = False,
    viewer_path: str | None = None,
) -> RealSampleV2FullCloudPurityReport:
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")
    resolved_max_points = _max_point_candidates(max_point_candidates)
    reports = tuple(
        real_sample_v2_viewer_preview_from_cloud(
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
            rewrite_sh=rewrite_sh,
            viewer_path=viewer_path,
        )
        for max_points in resolved_max_points
    )
    return RealSampleV2FullCloudPurityReport(
        candidates=reports,
        max_point_candidates=resolved_max_points,
        sample_source=str(sample_source),
        object_id_field=str(object_id_field),
    )


def validate_real_sample_v2_full_cloud_purity_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("real sample v2 full-cloud purity summary must be a dict")
    if payload.get("schema") != REAL_SAMPLE_V2_FULL_CLOUD_PURITY_SCHEMA:
        raise ValueError(f"unsupported real sample v2 full-cloud purity schema: {payload.get('schema')}")
    if payload.get("kind") != "real_sample_v2_full_cloud_purity":
        raise ValueError("real sample v2 full-cloud purity kind is unsupported")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_FAIL}:
        raise ValueError("real sample v2 full-cloud purity status is unsupported")
    for key in (
        "source",
        "segmentation_target",
        "baseline_candidate",
        "best_candidate",
        "coverage_sweep",
        "quality_delta",
        "recommendation",
        "viewer",
        "output_policy",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"real sample v2 full-cloud purity summary missing {key}")
    candidates = payload["coverage_sweep"]
    if not isinstance(candidates, list) or not candidates:
        raise ValueError("full-cloud purity summary requires candidate records")
    for candidate in candidates:
        _validate_candidate_record(candidate)
    selected = int(payload["segmentation_target"]["selected_max_points"])
    if selected not in {int(candidate["max_points"]) for candidate in candidates}:
        raise ValueError("selected max_points must be present in coverage sweep")
    if payload["segmentation_target"].get("export_object_id") != "argmax_assignment_slot":
        raise ValueError("full-cloud purity export object_id must be argmax assignment slot")
    best = payload["best_candidate"]
    if int(best["max_points"]) != selected:
        raise ValueError("best candidate must match selected max_points")
    if payload["status"] == _STATUS_PASS and best["quality"]["status"] != _QUALITY_PASS:
        raise ValueError("passing full-cloud purity summary requires best quality pass")
    recommendation = payload["recommendation"]
    if recommendation.get("requires_geometry_unfreeze") is not False:
        raise ValueError("full-cloud purity recommendation must not require geometry unfreeze")
    if recommendation.get("requires_diffusion_replay_or_rollout") is not False:
        raise ValueError("full-cloud purity recommendation must not require diffusion/replay/rollout")
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
        raise ValueError("real sample v2 full-cloud purity summary violates non-goals")
    return payload


def _max_point_candidates(values: Sequence[int]) -> tuple[int, ...]:
    if not values:
        raise ValueError("max_point_candidates must contain at least one value")
    output: list[int] = []
    for value in values:
        points = int(value)
        if points < 1:
            raise ValueError("max_point_candidates must be >= 1")
        if points not in output:
            output.append(points)
    return tuple(output)


def _candidate_record(
    report: RealSampleV2ViewerPreviewReport,
    max_points: int,
) -> dict[str, Any]:
    summary = validate_real_sample_v2_viewer_preview_summary(report.as_dict())
    quality = summary["quality"]
    return {
        "max_points": int(max_points),
        "sampled_gaussians": int(summary["handoff"]["sampled_gaussians"]),
        "source_gaussians": int(summary["source"]["source_gaussians"]),
        "target_source": summary["source"]["target_source"],
        "object_id_mapping": dict(summary["source"]["object_id_mapping"]),
        "solver_temperature": float(summary["handoff"]["recommended_solver_temperature"]),
        "handoff_status": summary["handoff"]["status"],
        "projected_gaussians": int(summary["projection"]["projected_gaussians"]),
        "predicted_object_count": int(summary["projection"]["predicted_object_count"]),
        "quality": {
            "status": quality["status"],
            "mean_normalized_entropy": float(quality["mean_normalized_entropy"]),
            "assignment_confidence": float(quality["assignment_confidence"]),
            "object_purity": (
                None
                if quality["object_purity"] is None
                else float(quality["object_purity"])
            ),
            "direct_slot_match": float(quality["direct_slot_match"]),
            "diagnostics": list(quality["diagnostics"]),
        },
        "viewer_path": summary["viewer"]["viewer_path"],
        "debug_route": summary["viewer"]["debug_route"],
    }


def _validate_candidate_record(candidate: dict[str, Any]) -> None:
    if not isinstance(candidate, dict):
        raise TypeError("full-cloud purity candidate must be a dict")
    for key in (
        "max_points",
        "sampled_gaussians",
        "source_gaussians",
        "solver_temperature",
        "projected_gaussians",
        "predicted_object_count",
        "quality",
    ):
        if key not in candidate:
            raise ValueError(f"full-cloud purity candidate missing {key}")
    if int(candidate["max_points"]) < 1:
        raise ValueError("candidate max_points must be >= 1")
    if int(candidate["sampled_gaussians"]) != int(candidate["max_points"]):
        raise ValueError("candidate sampled_gaussians must match max_points")
    if int(candidate["projected_gaussians"]) != int(candidate["source_gaussians"]):
        raise ValueError("candidate must project the full source cloud")
    quality = candidate["quality"]
    if quality.get("status") not in {_QUALITY_PASS, _QUALITY_DIAGNOSTIC}:
        raise ValueError("candidate quality status is unsupported")
    for metric in ("mean_normalized_entropy", "assignment_confidence", "direct_slot_match"):
        value = float(quality[metric])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"candidate {metric} must be in [0, 1]")
    if quality["object_purity"] is not None:
        purity = float(quality["object_purity"])
        if not 0.0 <= purity <= 1.0:
            raise ValueError("candidate object_purity must be in [0, 1]")


def _candidate_rank(candidate: dict[str, Any]) -> tuple[float, float, float, float, float]:
    quality = candidate["quality"]
    purity = -1.0 if quality["object_purity"] is None else float(quality["object_purity"])
    return (
        1.0 if quality["status"] == _QUALITY_PASS else 0.0,
        purity,
        float(quality["direct_slot_match"]),
        float(quality["assignment_confidence"]),
        -float(quality["mean_normalized_entropy"]),
    )


def _quality_delta(baseline: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_points_delta": int(best["max_points"]) - int(baseline["max_points"]),
        "entropy_delta": _delta(
            best["quality"]["mean_normalized_entropy"],
            baseline["quality"]["mean_normalized_entropy"],
        ),
        "confidence_delta": _delta(
            best["quality"]["assignment_confidence"],
            baseline["quality"]["assignment_confidence"],
        ),
        "purity_delta": _nullable_delta(
            best["quality"]["object_purity"],
            baseline["quality"]["object_purity"],
        ),
        "direct_slot_match_delta": _delta(
            best["quality"]["direct_slot_match"],
            baseline["quality"]["direct_slot_match"],
        ),
    }


def _recommendation(baseline: dict[str, Any], best: dict[str, Any]) -> dict[str, Any]:
    baseline_pass = baseline["quality"]["status"] == _QUALITY_PASS
    best_pass = best["quality"]["status"] == _QUALITY_PASS
    if best_pass and not baseline_pass:
        decision = "increase_segmentation_target_coverage"
        action = "set_max_points"
        evidence_normalization = "not_required_for_current_sample"
    elif best_pass:
        decision = "segmentation_target_coverage_sufficient"
        action = "keep_current_max_points"
        evidence_normalization = "not_required_for_current_sample"
    else:
        decision = "coverage_scan_insufficient"
        action = "diagnose_evidence_normalization"
        evidence_normalization = "required_next"
    return {
        "decision": decision,
        "action": action,
        "max_points": int(best["max_points"]),
        "solver_temperature": float(best["solver_temperature"]),
        "selection_policy": "highest_full_cloud_object_purity_then_direct_slot_match_then_confidence",
        "evidence_normalization": evidence_normalization,
        "requires_geometry_unfreeze": False,
        "requires_diffusion_replay_or_rollout": False,
    }


def _delta(left: object, right: object) -> float:
    return float(left) - float(right)


def _nullable_delta(left: object, right: object) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)
