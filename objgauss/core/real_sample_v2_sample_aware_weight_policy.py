from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io_ply import append_or_replace_property
from objgauss.core.real_sample_v2_model_handoff import real_sample_v2_model_handoff_from_cloud
from objgauss.core.real_sample_v2_viewer_preview import (
    REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT,
    REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT,
    RealSampleV2ViewerPreviewReport,
    real_sample_v2_viewer_preview_from_handoff,
    validate_real_sample_v2_viewer_preview_summary,
)
from objgauss.core.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
)

REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA = (
    "objgauss-real-sample-v2-sample-aware-weight-policy-v1"
)
_STATUS_PASS = "real_sample_v2_sample_aware_weight_policy_pass"
_STATUS_FAIL = "real_sample_v2_sample_aware_weight_policy_fail"


@dataclass(frozen=True)
class _WeightCandidate:
    name: str
    feature_weight: float
    position_weight: float
    preview: RealSampleV2ViewerPreviewReport
    normalization: dict[str, Any] | None = None


@dataclass(frozen=True)
class RealSampleV2SampleAwareWeightPolicyReport:
    candidates: tuple[_WeightCandidate, ...]
    sample_source: str
    object_id_field: str
    max_points: int
    solver_temperature: float
    viewer_path: str | None = None
    schema: str = REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA

    @property
    def selected_candidate(self) -> _WeightCandidate:
        records = _candidate_records(self.candidates)
        selected_name = _select_candidate_name(records)
        for candidate in self.candidates:
            if candidate.name == selected_name:
                return candidate
        raise ValueError("selected sample-aware candidate is not available")

    @property
    def selected_cloud(self) -> GaussianCloud:
        selected = self.selected_candidate
        baseline = self.candidates[0]
        selected_index = next(
            index
            for index, candidate in enumerate(self.candidates)
            if candidate.name == selected.name
        )
        return _with_sample_aware_fields(
            selected.preview.projected_cloud,
            baseline_preview=baseline.preview,
            selected_preview=selected.preview,
            selected_index=selected_index,
        )

    @property
    def passed(self) -> bool:
        records = _candidate_records(self.candidates)
        return any(bool(record["sample_policy_gate"]["eligible_for_sample"]) for record in records)

    def as_dict(self) -> dict[str, Any]:
        records = _candidate_records(self.candidates)
        selected_name = _select_candidate_name(records)
        selected_record = next(record for record in records if record["candidate"]["name"] == selected_name)
        evidence_gate = _evidence_normalization_gate(records)
        payload = {
            "schema": self.schema,
            "kind": "real_sample_v2_sample_aware_weight_policy",
            "status": _STATUS_PASS if self.passed else _STATUS_FAIL,
            "source": {
                "input": self.sample_source,
                "source_gaussians": int(self.selected_candidate.preview.projected_cloud.count),
                "object_id_field": self.object_id_field,
                "compatible": True,
            },
            "fixed_target": {
                "max_points": int(self.max_points),
                "solver_temperature": float(self.solver_temperature),
                "coverage_scan": "disabled",
                "temperature_sharpening_scan": "disabled",
                "export_object_id": "argmax_assignment_slot",
            },
            "policy": {
                "family": "sample_aware_assignment_weight_policy_v1",
                "selection_rule": "prefer_hard_boundary_non_regression_over_bounded_normalization_over_baseline",
                "baseline_candidate": self.candidates[0].name,
                "candidate_count": len(records),
                "uses_target_labels_for_prediction": False,
                "uses_target_labels_for_gate": True,
                "mutates_checkpoint": False,
                "bounded_evidence_normalization": _bounded_normalization_policy(records),
            },
            "candidates": records,
            "selected_policy": {
                "candidate_name": selected_record["candidate"]["name"],
                "feature_weight": selected_record["candidate"]["feature_weight"],
                "position_weight": selected_record["candidate"]["position_weight"],
                "selection_reason": selected_record["sample_policy_gate"]["decision"],
                "selected_for_viewer_export": True,
                "global_default": _selected_global_default_label(evidence_gate),
            },
            "evidence_normalization_gate": evidence_gate,
            "viewer": {
                "route_param": "ply",
                "viewer_path": self.viewer_path,
                "debug_route": f"/?ply={self.viewer_path}" if self.viewer_path else None,
                "load_mode": "url-object-aware-ply",
            },
            "output_policy": {
                "preview_ply": "write selected sample-aware PLY to /tmp or ignored outputs; do not commit generated preview PLY",
                "summary": "write to /tmp or ignored outputs for sample-aware policy diagnostics",
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
        return validate_real_sample_v2_sample_aware_weight_policy_summary(payload)


def real_sample_v2_sample_aware_weight_policy_from_cloud(
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
) -> RealSampleV2SampleAwareWeightPolicyReport:
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
        viewer_path=viewer_path,
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
    bounded_feature_weight, bounded_position_weight, normalization = _bounded_normalized_weights(
        baseline_preview=baseline_preview,
        promoted_preview=promoted_preview,
        baseline_feature_weight=float(baseline_feature_weight),
        baseline_position_weight=float(baseline_position_weight),
        promoted_feature_weight=float(promoted_feature_weight),
        promoted_position_weight=float(promoted_position_weight),
    )
    bounded_preview = real_sample_v2_viewer_preview_from_handoff(
        cloud,
        handoff,
        sample_source=sample_source,
        object_id_field=object_id_field,
        slots=slots,
        seed=seed,
        assignment_feature_weight=bounded_feature_weight,
        assignment_position_weight=bounded_position_weight,
        rewrite_sh=rewrite_sh,
        viewer_path=viewer_path,
    )
    return RealSampleV2SampleAwareWeightPolicyReport(
        candidates=(
            _WeightCandidate(
                name="baseline",
                feature_weight=float(baseline_feature_weight),
                position_weight=float(baseline_position_weight),
                preview=baseline_preview,
            ),
            _WeightCandidate(
                name="promoted",
                feature_weight=float(promoted_feature_weight),
                position_weight=float(promoted_position_weight),
                preview=promoted_preview,
            ),
            _WeightCandidate(
                name="bounded-normalized",
                feature_weight=float(bounded_feature_weight),
                position_weight=float(bounded_position_weight),
                preview=bounded_preview,
                normalization=normalization,
            ),
        ),
        sample_source=str(sample_source),
        object_id_field=str(object_id_field),
        max_points=int(max_points),
        solver_temperature=float(solver_temperature),
        viewer_path=viewer_path,
    )


def validate_real_sample_v2_sample_aware_weight_policy_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("real sample v2 sample-aware weight policy summary must be a dict")
    if payload.get("schema") != REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA:
        raise ValueError(f"unsupported sample-aware weight policy schema: {payload.get('schema')}")
    if payload.get("kind") != "real_sample_v2_sample_aware_weight_policy":
        raise ValueError("sample-aware weight policy kind is unsupported")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_FAIL}:
        raise ValueError("sample-aware weight policy status is unsupported")
    for key in (
        "source",
        "fixed_target",
        "policy",
        "candidates",
        "selected_policy",
        "evidence_normalization_gate",
        "viewer",
        "output_policy",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"sample-aware weight policy summary missing {key}")
    source = payload["source"]
    source_count = int(source["source_gaussians"])
    if source_count < 1:
        raise ValueError("sample-aware weight policy requires source gaussians")
    if source.get("compatible") is not True:
        raise ValueError("sample-aware weight policy requires a compatible sample")
    fixed = payload["fixed_target"]
    if int(fixed["max_points"]) < 1:
        raise ValueError("sample-aware weight policy max_points must be >= 1")
    if float(fixed["solver_temperature"]) <= 0.0:
        raise ValueError("sample-aware weight policy solver_temperature must be > 0")
    if fixed.get("coverage_scan") != "disabled":
        raise ValueError("sample-aware weight policy must keep coverage scan disabled")
    if fixed.get("temperature_sharpening_scan") != "disabled":
        raise ValueError("sample-aware weight policy must keep sharpening scan disabled")
    if fixed.get("export_object_id") != "argmax_assignment_slot":
        raise ValueError("sample-aware weight policy must export argmax assignment slot")

    policy = payload["policy"]
    if policy.get("family") != "sample_aware_assignment_weight_policy_v1":
        raise ValueError("sample-aware weight policy family is unsupported")
    if policy.get("uses_target_labels_for_prediction") is not False:
        raise ValueError("sample-aware policy must not use target labels for prediction")
    if policy.get("uses_target_labels_for_gate") is not True:
        raise ValueError("sample-aware policy gate must be explicit about target-label evaluation")
    if policy.get("mutates_checkpoint") is not False:
        raise ValueError("sample-aware policy must not mutate checkpoint")
    normalization_policy = policy.get("bounded_evidence_normalization")
    if not isinstance(normalization_policy, dict):
        raise ValueError("sample-aware policy must describe bounded evidence normalization")
    if normalization_policy.get("schema") != "objgauss-bounded-evidence-normalization-v1":
        raise ValueError("sample-aware bounded evidence normalization schema is unsupported")
    if normalization_policy.get("uses_target_labels_for_prediction") is not False:
        raise ValueError("bounded evidence normalization must not use target labels for prediction")
    if normalization_policy.get("uses_target_labels_for_gate") is not True:
        raise ValueError("bounded evidence normalization gate must be explicit about target labels")
    if not 0.0 <= float(normalization_policy["feature_weight_blend"]) <= 1.0:
        raise ValueError("bounded evidence normalization blend must be in [0, 1]")

    candidates = payload["candidates"]
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("sample-aware weight policy requires at least baseline and candidate")
    candidate_names = set()
    eligible = False
    for candidate in candidates:
        _validate_candidate_record(candidate, source_count)
        candidate_names.add(candidate["candidate"]["name"])
        eligible = eligible or bool(candidate["sample_policy_gate"]["eligible_for_sample"])
    selected = payload["selected_policy"]
    if selected.get("candidate_name") not in candidate_names:
        raise ValueError("selected sample-aware candidate is not in candidates")
    selected_record = next(
        candidate for candidate in candidates if candidate["candidate"]["name"] == selected["candidate_name"]
    )
    if selected_record["sample_policy_gate"]["eligible_for_sample"] is not True:
        raise ValueError("selected sample-aware candidate must be eligible")
    if payload["status"] == _STATUS_PASS and not eligible:
        raise ValueError("passing sample-aware policy requires an eligible candidate")

    gate = payload["evidence_normalization_gate"]
    if gate.get("status") not in {
        "not_required_for_selected_policy",
        "satisfied_by_bounded_normalization",
        "required_before_global_weight_promotion",
    }:
        raise ValueError("sample-aware evidence normalization gate status is unsupported")
    if bool(gate["requires_evidence_normalization"]) != (
        gate["status"] == "required_before_global_weight_promotion"
    ):
        raise ValueError("sample-aware evidence normalization gate is inconsistent")
    if gate.get("requires_geometry_unfreeze") is not False:
        raise ValueError("sample-aware gate must not require geometry unfreeze")
    if gate.get("requires_diffusion_replay_or_rollout") is not False:
        raise ValueError("sample-aware gate must not require diffusion/replay/rollout")

    viewer = payload["viewer"]
    if viewer.get("route_param") != "ply":
        raise ValueError("sample-aware weight policy viewer route param must be ply")
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
        raise ValueError("sample-aware weight policy summary violates non-goals")
    return payload


def _validate_candidate_record(candidate: dict[str, Any], source_count: int) -> None:
    preview = validate_real_sample_v2_viewer_preview_summary(candidate["preview"])
    if int(preview["source"]["source_gaussians"]) != source_count:
        raise ValueError("sample-aware candidate source count mismatch")
    policy = candidate["candidate"]
    if float(policy["feature_weight"]) < 0.0 or float(policy["position_weight"]) < 0.0:
        raise ValueError("sample-aware candidate weights must be >= 0")
    metrics = candidate["metrics"]
    if int(metrics["mixed_gaussians"]) < 0:
        raise ValueError("sample-aware candidate mixed_gaussians must be >= 0")
    for metric in ("direct_slot_match", "assignment_confidence", "mean_normalized_entropy"):
        value = float(metrics[metric])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"sample-aware candidate {metric} must be in [0, 1]")
    if metrics["object_purity"] is not None:
        purity = float(metrics["object_purity"])
        if not 0.0 <= purity <= 1.0:
            raise ValueError("sample-aware candidate object_purity must be in [0, 1]")
    delta = candidate["delta_vs_baseline"]
    int(delta["mixed_gaussians_delta"])
    float(delta["direct_slot_match_delta"])
    if delta["object_purity_delta"] is not None:
        float(delta["object_purity_delta"])
    changed = candidate["changed_gaussians"]
    if int(changed["changed_count"]) < 0:
        raise ValueError("sample-aware candidate changed_count must be >= 0")
    if int(changed["hard_fix_count"]) < 0 or int(changed["hard_regression_count"]) < 0:
        raise ValueError("sample-aware candidate hard change counts must be >= 0")
    normalization = candidate.get("bounded_evidence_normalization")
    if normalization is not None:
        if normalization.get("schema") != "objgauss-bounded-evidence-normalization-v1":
            raise ValueError("sample-aware candidate normalization schema is unsupported")
        if not 0.0 <= float(normalization["feature_weight_blend"]) <= 1.0:
            raise ValueError("sample-aware candidate normalization blend must be in [0, 1]")
        if not 0.0 <= float(normalization["soft_evidence_blend"]) <= 1.0:
            raise ValueError("sample-aware candidate soft evidence blend must be in [0, 1]")


def _candidate_records(candidates: tuple[_WeightCandidate, ...]) -> list[dict[str, Any]]:
    if len(candidates) < 2:
        raise ValueError("sample-aware policy requires at least two candidates")
    baseline_summary = validate_real_sample_v2_viewer_preview_summary(
        candidates[0].preview.as_dict()
    )
    records = []
    for index, candidate in enumerate(candidates):
        summary = validate_real_sample_v2_viewer_preview_summary(candidate.preview.as_dict())
        delta = _quality_delta(baseline_summary, summary)
        changed = _changed_gaussians(candidates[0].preview, candidate.preview)
        gate = _sample_policy_gate(
            baseline=baseline_summary,
            candidate=summary,
            delta=delta,
            changed=changed,
            is_baseline=index == 0,
        )
        if candidate.normalization:
            blend = float(candidate.normalization["feature_weight_blend"])
            if blend == 0.0:
                gate = {
                    **gate,
                    "eligible_for_sample": False,
                    "decision": "bounded_evidence_normalization_noop_baseline_fallback",
                }
            elif gate["eligible_for_sample"]:
                gate = {
                    **gate,
                    "decision": "bounded_evidence_normalization_non_regression",
                }
        record = {
            "candidate": {
                "name": candidate.name,
                "role": "baseline" if index == 0 else "candidate",
                "feature_weight": float(candidate.feature_weight),
                "position_weight": float(candidate.position_weight),
            },
            "preview": summary,
            "metrics": _metrics(summary),
            "delta_vs_baseline": delta,
            "changed_gaussians": changed,
            "sample_policy_gate": gate,
        }
        if candidate.normalization is not None:
            record["bounded_evidence_normalization"] = candidate.normalization
        records.append(record)
    return records


def _sample_policy_gate(
    *,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    delta: dict[str, Any],
    changed: dict[str, Any],
    is_baseline: bool,
) -> dict[str, Any]:
    quality_pass = candidate["quality"]["status"] == "full_cloud_objectstate_preview_quality_pass"
    status_pass = candidate["status"] == "real_sample_v2_viewer_preview_pass"
    if is_baseline:
        eligible = status_pass and quality_pass
        decision = "baseline_safe_fallback" if eligible else "baseline_unavailable"
    else:
        hard_regression_free = int(changed["hard_regression_count"]) == 0
        eligible = (
            baseline["status"] == "real_sample_v2_viewer_preview_pass"
            and status_pass
            and quality_pass
            and hard_regression_free
            and int(delta["mixed_gaussians_delta"]) <= 0
            and float(delta["direct_slot_match_delta"]) >= 0.0
            and int(delta["predicted_object_count_delta"]) == 0
            and (
                delta["object_purity_delta"] is None
                or float(delta["object_purity_delta"]) >= 0.0
            )
        )
        decision = (
            "candidate_hard_boundary_non_regression"
            if eligible
            else "candidate_blocked_by_sample_aware_gate"
        )
    return {
        "eligible_for_sample": bool(eligible),
        "decision": decision,
        "hard_mixed_gaussians_non_regression": int(delta["mixed_gaussians_delta"]) <= 0,
        "direct_slot_match_non_regression": float(delta["direct_slot_match_delta"]) >= 0.0,
        "hard_regression_free": int(changed["hard_regression_count"]) == 0,
        "soft_purity_non_regression": (
            delta["object_purity_delta"] is None
            or float(delta["object_purity_delta"]) >= 0.0
        ),
        "predicted_object_count_stable": int(delta["predicted_object_count_delta"]) == 0,
        "hard_fix_count": int(changed["hard_fix_count"]),
        "hard_regression_count": int(changed["hard_regression_count"]),
    }


def _select_candidate_name(records: list[dict[str, Any]]) -> str:
    eligible = [
        record for record in records if record["sample_policy_gate"]["eligible_for_sample"]
    ]
    if not eligible:
        raise ValueError("no sample-aware candidate passed the gate")

    def score(record: dict[str, Any]) -> tuple[float, int, float, float, float, int]:
        metrics = record["metrics"]
        purity = -1.0 if metrics["object_purity"] is None else float(metrics["object_purity"])
        policy_rank = {
            "promoted": 2,
            "bounded-normalized": 1,
            "baseline": 0,
        }.get(str(record["candidate"]["name"]), 0)
        return (
            float(metrics["direct_slot_match"]),
            -int(metrics["mixed_gaussians"]),
            purity,
            float(metrics["assignment_confidence"]),
            -float(metrics["mean_normalized_entropy"]),
            policy_rank,
        )

    selected = max(eligible, key=score)
    return str(selected["candidate"]["name"])


def _evidence_normalization_gate(records: list[dict[str, Any]]) -> dict[str, Any]:
    blocked_soft_sharpening = []
    bounded_normalized = None
    for record in records:
        if record["candidate"]["name"] == "bounded-normalized":
            bounded_normalized = record
        if record["candidate"]["role"] == "baseline":
            continue
        delta = record["delta_vs_baseline"]
        gate = record["sample_policy_gate"]
        soft_improved = (
            (delta["object_purity_delta"] is not None and float(delta["object_purity_delta"]) > 0.0)
            or float(delta["assignment_confidence_delta"]) > 0.0
        )
        hard_regressed = (
            int(delta["mixed_gaussians_delta"]) > 0
            or float(delta["direct_slot_match_delta"]) < 0.0
            or int(gate["hard_regression_count"]) > 0
        )
        if soft_improved and hard_regressed and not gate["eligible_for_sample"]:
            blocked_soft_sharpening.append(record["candidate"]["name"])
    bounded_satisfied = (
        bool(blocked_soft_sharpening)
        and bounded_normalized is not None
        and bool(bounded_normalized["sample_policy_gate"]["eligible_for_sample"])
    )
    required = bool(blocked_soft_sharpening) and not bounded_satisfied
    return {
        "status": (
            "required_before_global_weight_promotion"
            if required
            else (
                "satisfied_by_bounded_normalization"
                if bounded_satisfied
                else "not_required_for_selected_policy"
            )
        ),
        "requires_evidence_normalization": required,
        "blocked_soft_sharpening_candidates": blocked_soft_sharpening,
        "bounded_normalized_candidate": (
            None
            if bounded_normalized is None
            else {
                "name": bounded_normalized["candidate"]["name"],
                "eligible_for_sample": bool(bounded_normalized["sample_policy_gate"]["eligible_for_sample"]),
                "feature_weight": float(bounded_normalized["candidate"]["feature_weight"]),
                "position_weight": float(bounded_normalized["candidate"]["position_weight"]),
                "feature_weight_blend": float(
                    bounded_normalized.get("bounded_evidence_normalization", {}).get(
                        "feature_weight_blend",
                        0.0,
                    )
                ),
                "hard_regression_count": int(
                    bounded_normalized["sample_policy_gate"]["hard_regression_count"]
                ),
            }
        ),
        "decision": (
            "block_soft_sharpening_without_hard_boundary_non_regression"
            if required
            else (
                "bounded_normalization_satisfies_soft_sharpening_gate"
                if bounded_satisfied
                else "selected_policy_passes_sample_gate"
            )
        ),
        "action": (
            "add_evidence_normalization_or_sample_specific_weight_policy_before_global_default"
            if required
            else (
                "use_bounded_normalized_candidate_for_this_sample"
                if bounded_satisfied
                else "keep_selected_policy_for_this_sample"
            )
        ),
        "requires_geometry_unfreeze": False,
        "requires_diffusion_replay_or_rollout": False,
    }


def _bounded_normalization_policy(records: list[dict[str, Any]]) -> dict[str, Any]:
    normalized = next(
        (record for record in records if record["candidate"]["name"] == "bounded-normalized"),
        None,
    )
    normalization = normalized.get("bounded_evidence_normalization", {}) if normalized else {}
    return {
        "schema": "objgauss-bounded-evidence-normalization-v1",
        "enabled": normalized is not None,
        "candidate_name": "bounded-normalized",
        "source_candidate": "promoted",
        "baseline_candidate": "baseline",
        "feature_weight_blend": float(normalization.get("feature_weight_blend", 0.0)),
        "position_weight_blend": float(normalization.get("position_weight_blend", 0.0)),
        "uses_target_labels_for_prediction": False,
        "uses_target_labels_for_gate": True,
        "mutates_checkpoint": False,
    }


def _selected_global_default_label(evidence_gate: dict[str, Any]) -> str:
    status = evidence_gate.get("status")
    if status == "required_before_global_weight_promotion":
        return "sample_specific_only"
    if status == "satisfied_by_bounded_normalization":
        return "sample_specific_bounded_normalization"
    return "eligible_after_more_cross_sample_rows"


def _bounded_normalized_weights(
    *,
    baseline_preview: RealSampleV2ViewerPreviewReport,
    promoted_preview: RealSampleV2ViewerPreviewReport,
    baseline_feature_weight: float,
    baseline_position_weight: float,
    promoted_feature_weight: float,
    promoted_position_weight: float,
) -> tuple[float, float, dict[str, Any]]:
    baseline_summary = validate_real_sample_v2_viewer_preview_summary(
        baseline_preview.as_dict()
    )
    promoted_summary = validate_real_sample_v2_viewer_preview_summary(
        promoted_preview.as_dict()
    )
    delta = _quality_delta(baseline_summary, promoted_summary)
    changed = _changed_gaussians(baseline_preview, promoted_preview)
    hard_fix = int(changed["hard_fix_count"])
    hard_regression = int(changed["hard_regression_count"])
    if hard_regression <= 0:
        hard_safety_blend = 1.0
        reason = "promoted_has_no_hard_regression"
    elif hard_regression >= hard_fix:
        hard_safety_blend = 0.0
        reason = "hard_regression_not_bounded_by_hard_fix"
    else:
        hard_safety_blend = max(
            0.0,
            min(1.0, (hard_fix - hard_regression) / max(hard_fix, 1)),
        )
        reason = "partial_soft_sharpening_with_hard_regression_budget"
    confidence_gain = max(0.0, float(delta["assignment_confidence_delta"]))
    entropy_reduction = max(0.0, -float(delta["mean_normalized_entropy_delta"]))
    purity_gain = (
        0.0
        if delta["object_purity_delta"] is None
        else max(0.0, float(delta["object_purity_delta"]))
    )
    bounded_confidence_gain = min(1.0, confidence_gain)
    bounded_entropy_gain = min(1.0, entropy_reduction)
    bounded_purity_gain = min(1.0, purity_gain)
    soft_evidence_blend = float(
        np.mean([bounded_confidence_gain, bounded_entropy_gain, bounded_purity_gain])
    )
    blend = hard_safety_blend
    feature_weight = baseline_feature_weight + blend * (
        promoted_feature_weight - baseline_feature_weight
    )
    position_weight = baseline_position_weight + blend * (
        promoted_position_weight - baseline_position_weight
    )
    normalization = {
        "schema": "objgauss-bounded-evidence-normalization-v1",
        "source_candidate": "promoted",
        "baseline_candidate": "baseline",
        "reason": reason,
        "feature_weight_blend": float(blend),
        "position_weight_blend": float(blend),
        "hard_safety_blend": float(hard_safety_blend),
        "soft_evidence_blend": soft_evidence_blend,
        "baseline_feature_weight": float(baseline_feature_weight),
        "baseline_position_weight": float(baseline_position_weight),
        "promoted_feature_weight": float(promoted_feature_weight),
        "promoted_position_weight": float(promoted_position_weight),
        "normalized_feature_weight": float(feature_weight),
        "normalized_position_weight": float(position_weight),
        "promoted_hard_fix_count": hard_fix,
        "promoted_hard_regression_count": hard_regression,
        "promoted_mixed_gaussians_delta": int(delta["mixed_gaussians_delta"]),
        "promoted_direct_slot_match_delta": float(delta["direct_slot_match_delta"]),
        "promoted_assignment_confidence_delta": float(delta["assignment_confidence_delta"]),
        "promoted_entropy_delta": float(delta["mean_normalized_entropy_delta"]),
        "bounded_confidence_gain": bounded_confidence_gain,
        "bounded_entropy_reduction": bounded_entropy_gain,
        "bounded_purity_gain": bounded_purity_gain,
        "uses_target_labels_for_prediction": False,
        "uses_target_labels_for_gate": True,
        "mutates_checkpoint": False,
    }
    return float(feature_weight), float(position_weight), normalization


def _metrics(summary: dict[str, Any]) -> dict[str, Any]:
    hard = summary["projection"]["hard_segmentation"]
    quality = summary["quality"]
    return {
        "mixed_gaussians": int(hard["mixed_gaussians"]),
        "direct_slot_match": float(quality["direct_slot_match"]),
        "object_purity": (
            None if quality["object_purity"] is None else float(quality["object_purity"])
        ),
        "assignment_confidence": float(quality["assignment_confidence"]),
        "mean_normalized_entropy": float(quality["mean_normalized_entropy"]),
        "predicted_object_count": int(summary["projection"]["predicted_object_count"]),
        "object_id_counts": list(hard["object_id_counts"]),
    }


def _quality_delta(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    baseline_metrics = _metrics(baseline)
    candidate_metrics = _metrics(candidate)
    baseline_purity = baseline_metrics["object_purity"]
    candidate_purity = candidate_metrics["object_purity"]
    return {
        "mixed_gaussians_delta": int(candidate_metrics["mixed_gaussians"])
        - int(baseline_metrics["mixed_gaussians"]),
        "direct_slot_match_delta": float(candidate_metrics["direct_slot_match"])
        - float(baseline_metrics["direct_slot_match"]),
        "object_purity_delta": (
            None
            if baseline_purity is None or candidate_purity is None
            else float(candidate_purity) - float(baseline_purity)
        ),
        "assignment_confidence_delta": float(candidate_metrics["assignment_confidence"])
        - float(baseline_metrics["assignment_confidence"]),
        "mean_normalized_entropy_delta": float(candidate_metrics["mean_normalized_entropy"])
        - float(baseline_metrics["mean_normalized_entropy"]),
        "predicted_object_count_delta": int(candidate_metrics["predicted_object_count"])
        - int(baseline_metrics["predicted_object_count"]),
    }


def _changed_gaussians(
    baseline_preview: RealSampleV2ViewerPreviewReport,
    candidate_preview: RealSampleV2ViewerPreviewReport,
) -> dict[str, Any]:
    baseline_ids = baseline_preview.projection.derived_object_ids.astype(np.int32, copy=False)
    candidate_ids = candidate_preview.projection.derived_object_ids.astype(np.int32, copy=False)
    target_slots = np.asarray(candidate_preview.projected_cloud.vertices["target_slot"], dtype=np.int32)
    if baseline_ids.shape != candidate_ids.shape or baseline_ids.shape != target_slots.shape:
        raise ValueError("baseline, candidate, and target slots must have matching shapes")
    changed = baseline_ids != candidate_ids
    hard_fix = changed & (baseline_ids != target_slots) & (candidate_ids == target_slots)
    hard_regression = changed & (baseline_ids == target_slots) & (candidate_ids != target_slots)
    pairs = []
    if np.any(changed):
        pair_values, counts = np.unique(
            np.column_stack([baseline_ids[changed], candidate_ids[changed]]),
            axis=0,
            return_counts=True,
        )
        pairs = [
            {
                "baseline_object_id": int(pair[0]),
                "candidate_object_id": int(pair[1]),
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


def _with_sample_aware_fields(
    cloud: GaussianCloud,
    *,
    baseline_preview: RealSampleV2ViewerPreviewReport,
    selected_preview: RealSampleV2ViewerPreviewReport,
    selected_index: int,
) -> GaussianCloud:
    vertices = cloud.vertices
    baseline_vertices = baseline_preview.projected_cloud.vertices
    baseline_ids = baseline_preview.projection.derived_object_ids.astype(np.int32, copy=False)
    selected_ids = selected_preview.projection.derived_object_ids.astype(np.int32, copy=False)
    target_slots = np.asarray(vertices["target_slot"], dtype=np.int32)
    if cloud.count != baseline_ids.shape[0] or cloud.count != selected_ids.shape[0]:
        raise ValueError("sample-aware exported fields must match cloud count")
    changed = baseline_ids != selected_ids
    hard_fix = changed & (baseline_ids != target_slots) & (selected_ids == target_slots)
    hard_regression = changed & (baseline_ids == target_slots) & (selected_ids != target_slots)
    vertices = append_or_replace_property(
        vertices,
        "sample_aware_baseline_object_id",
        baseline_ids,
        np.int32,
    )
    vertices = append_or_replace_property(
        vertices,
        "sample_aware_baseline_confidence",
        np.asarray(baseline_vertices["assignment_confidence"], dtype=np.float32),
        np.float32,
    )
    vertices = append_or_replace_property(
        vertices,
        "sample_aware_baseline_entropy",
        np.asarray(baseline_vertices["assignment_entropy"], dtype=np.float32),
        np.float32,
    )
    vertices = append_or_replace_property(
        vertices,
        "sample_aware_selected_index",
        np.full(cloud.count, int(selected_index), dtype=np.int32),
        np.int32,
    )
    vertices = append_or_replace_property(
        vertices,
        "sample_aware_changed",
        changed.astype(np.uint8, copy=False),
        np.uint8,
    )
    vertices = append_or_replace_property(
        vertices,
        "sample_aware_hard_fix",
        hard_fix.astype(np.uint8, copy=False),
        np.uint8,
    )
    vertices = append_or_replace_property(
        vertices,
        "sample_aware_hard_regression",
        hard_regression.astype(np.uint8, copy=False),
        np.uint8,
    )
    return cloud.with_vertices(vertices)
