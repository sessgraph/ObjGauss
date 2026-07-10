from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from objgauss.core.gaussian import GaussianCloud
from objgauss.pipelines.real_sample_v2_sample_aware_weight_policy import (
    REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA,
    RealSampleV2SampleAwareWeightPolicyReport,
    real_sample_v2_sample_aware_weight_policy_from_cloud,
    validate_real_sample_v2_sample_aware_weight_policy_summary,
)
from objgauss.pipelines.real_sample_v2_viewer_preview import (
    REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT,
    REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT,
)
from objgauss.pipelines.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
)

REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA = (
    "objgauss-real-sample-v2-bounded-normalization-cross-sample-v1"
)

__all__ = (
    "REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA",
    "RealSampleV2BoundedNormalizationCrossSampleInput",
    "RealSampleV2BoundedNormalizationCrossSampleReport",
    "real_sample_v2_bounded_normalization_cross_sample_from_clouds",
    "validate_real_sample_v2_bounded_normalization_cross_sample_summary",
)
_STATUS_PASS = "real_sample_v2_bounded_normalization_cross_sample_pass"
_STATUS_DIAGNOSTIC = "real_sample_v2_bounded_normalization_cross_sample_diagnostic"


@dataclass(frozen=True)
class RealSampleV2BoundedNormalizationCrossSampleInput:
    sample_id: str
    cloud: GaussianCloud
    sample_source: str = "memory://gaussian-cloud"
    object_id_field: str = "object_id"
    slots: int | None = None
    viewer_path: str | None = None


@dataclass(frozen=True)
class _BoundedNormalizationSampleReport:
    sample_id: str
    sample_source: str
    report: RealSampleV2SampleAwareWeightPolicyReport


@dataclass(frozen=True)
class RealSampleV2BoundedNormalizationCrossSampleReport:
    samples: tuple[_BoundedNormalizationSampleReport, ...]
    min_samples: int
    schema: str = REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA

    @property
    def passed(self) -> bool:
        rows = [_sample_row(sample) for sample in self.samples]
        return _aggregate(rows, min_samples=self.min_samples)["result"] == "pass"

    def as_dict(self) -> dict[str, Any]:
        rows = [_sample_row(sample) for sample in self.samples]
        aggregate = _aggregate(rows, min_samples=self.min_samples)
        payload = {
            "schema": self.schema,
            "kind": "real_sample_v2_bounded_normalization_cross_sample",
            "status": _STATUS_PASS if aggregate["result"] == "pass" else _STATUS_DIAGNOSTIC,
            "sample_count": len(rows),
            "min_samples": int(self.min_samples),
            "policy": {
                "family": "bounded_normalization_cross_sample_gate_v1",
                "source_policy_schema": REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA,
                "sample_policy": "sample_aware_assignment_weight_policy_v1",
                "selection_rule": "each_sample_selects_promoted_or_bounded_normalized_or_baseline",
                "hard_gate": "selected_policy_hard_regression_count_must_be_zero",
                "uses_target_labels_for_prediction": False,
                "uses_target_labels_for_gate": True,
                "mutates_checkpoint": False,
            },
            "rows": rows,
            "aggregate": aggregate,
            "recommendation": _recommendation(aggregate),
            "output_policy": {
                "summary": "write cross-sample summary to /tmp or ignored outputs",
                "preview_ply": "not written by aggregate report; use per-sample policy command for selected PLY exports",
                "screenshots": "write browser evidence to /tmp only when a viewer PR consumes selected outputs",
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
        return validate_real_sample_v2_bounded_normalization_cross_sample_summary(payload)


def real_sample_v2_bounded_normalization_cross_sample_from_clouds(
    samples: Sequence[RealSampleV2BoundedNormalizationCrossSampleInput],
    *,
    min_samples: int = 2,
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
) -> RealSampleV2BoundedNormalizationCrossSampleReport:
    if min_samples < 1:
        raise ValueError("min_samples must be >= 1")
    if len(samples) < 1:
        raise ValueError("at least one cross-sample input is required")
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")

    sample_reports = []
    seen_ids: set[str] = set()
    for index, sample in enumerate(samples):
        sample_id = str(sample.sample_id).strip()
        if not sample_id:
            raise ValueError("sample_id must not be empty")
        if sample_id in seen_ids:
            raise ValueError(f"duplicate sample_id: {sample_id}")
        seen_ids.add(sample_id)
        report = real_sample_v2_sample_aware_weight_policy_from_cloud(
            sample.cloud,
            sample_source=sample.sample_source,
            object_id_field=sample.object_id_field,
            slots=sample.slots,
            max_points=max_points,
            solver_temperature=solver_temperature,
            baseline_feature_weight=baseline_feature_weight,
            baseline_position_weight=baseline_position_weight,
            promoted_feature_weight=promoted_feature_weight,
            promoted_position_weight=promoted_position_weight,
            frame_count=frame_count,
            temporal_offset=temporal_offset,
            image_width=image_width,
            image_height=image_height,
            point_radius=point_radius,
            visibility_policy=visibility_policy,
            seed=seed,
            iterations=iterations,
            learning_rate=learning_rate,
            baseline_temperature=baseline_temperature,
            image_renderer=image_renderer,
            vram_reserve_gb=vram_reserve_gb,
            rewrite_sh=rewrite_sh,
            viewer_path=sample.viewer_path,
        )
        sample_reports.append(
            _BoundedNormalizationSampleReport(
                sample_id=sample_id,
                sample_source=str(sample.sample_source or f"sample://{index}"),
                report=report,
            )
        )
    return RealSampleV2BoundedNormalizationCrossSampleReport(
        samples=tuple(sample_reports),
        min_samples=int(min_samples),
    )


def validate_real_sample_v2_bounded_normalization_cross_sample_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("bounded normalization cross-sample summary must be a dict")
    if payload.get("schema") != REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA:
        raise ValueError(
            f"unsupported bounded normalization cross-sample schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "real_sample_v2_bounded_normalization_cross_sample":
        raise ValueError("bounded normalization cross-sample kind is unsupported")
    if payload.get("status") not in {_STATUS_PASS, _STATUS_DIAGNOSTIC}:
        raise ValueError("bounded normalization cross-sample status is unsupported")

    for key in (
        "sample_count",
        "min_samples",
        "policy",
        "rows",
        "aggregate",
        "recommendation",
        "output_policy",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"bounded normalization cross-sample summary missing {key}")

    min_samples = int(payload["min_samples"])
    if min_samples < 1:
        raise ValueError("bounded normalization cross-sample min_samples must be >= 1")
    rows = payload["rows"]
    if not isinstance(rows, list) or len(rows) != int(payload["sample_count"]):
        raise ValueError("bounded normalization cross-sample rows must match sample_count")
    sample_ids = set()
    for row in rows:
        _validate_row(row)
        sample_id = row["sample_id"]
        if sample_id in sample_ids:
            raise ValueError(f"duplicate bounded normalization sample_id: {sample_id}")
        sample_ids.add(sample_id)

    policy = payload["policy"]
    if policy.get("family") != "bounded_normalization_cross_sample_gate_v1":
        raise ValueError("bounded normalization cross-sample policy is unsupported")
    if policy.get("source_policy_schema") != REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA:
        raise ValueError("bounded normalization cross-sample source policy schema is unsupported")
    if policy.get("uses_target_labels_for_prediction") is not False:
        raise ValueError("bounded normalization cross-sample must not predict with target labels")
    if policy.get("uses_target_labels_for_gate") is not True:
        raise ValueError("bounded normalization cross-sample must declare target-label gate use")
    if policy.get("mutates_checkpoint") is not False:
        raise ValueError("bounded normalization cross-sample must not mutate checkpoints")

    aggregate = payload["aggregate"]
    expected_aggregate = _aggregate(rows, min_samples=min_samples)
    for key in (
        "result",
        "all_sample_policies_pass",
        "selected_hard_regression_count",
        "selected_hard_regression_samples",
        "blocked_promoted_sample_count",
        "blocked_promoted_samples",
    ):
        if aggregate.get(key) != expected_aggregate[key]:
            raise ValueError(f"bounded normalization aggregate {key} is inconsistent")
    if payload["status"] != (_STATUS_PASS if expected_aggregate["result"] == "pass" else _STATUS_DIAGNOSTIC):
        raise ValueError("bounded normalization cross-sample status is inconsistent")

    recommendation = payload["recommendation"]
    if recommendation.get("requires_geometry_unfreeze") is not False:
        raise ValueError("bounded normalization cross-sample must not recommend geometry unfreeze")
    if recommendation.get("requires_diffusion_replay_or_rollout") is not False:
        raise ValueError("bounded normalization cross-sample must not recommend diffusion/replay/rollout")

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
        raise ValueError("bounded normalization cross-sample summary violates non-goals")
    return payload


def _sample_row(sample: _BoundedNormalizationSampleReport) -> dict[str, Any]:
    summary = validate_real_sample_v2_sample_aware_weight_policy_summary(
        sample.report.as_dict()
    )
    selected_name = str(summary["selected_policy"]["candidate_name"])
    selected = _candidate_by_name(summary, selected_name)
    promoted = _candidate_by_name(summary, "promoted")
    bounded = _candidate_by_name(summary, "bounded-normalized")
    return {
        "sample_id": sample.sample_id,
        "sample_policy_schema": summary["schema"],
        "status": summary["status"],
        "source": {
            "input": sample.sample_source,
            "source_gaussians": int(summary["source"]["source_gaussians"]),
            "object_id_field": summary["source"]["object_id_field"],
            "compatible": summary["source"]["compatible"],
        },
        "fixed_target": dict(summary["fixed_target"]),
        "selected_policy": dict(summary["selected_policy"]),
        "selected_metrics": dict(selected["metrics"]),
        "selected_delta_vs_baseline": dict(selected["delta_vs_baseline"]),
        "selected_changed_gaussians": _compact_changed_gaussians(selected),
        "selected_gate": dict(selected["sample_policy_gate"]),
        "promoted_candidate": _compact_candidate(promoted),
        "bounded_normalized_candidate": _compact_candidate(bounded),
        "evidence_normalization_status": summary["evidence_normalization_gate"]["status"],
        "evidence_normalization_action": summary["evidence_normalization_gate"]["action"],
    }


def _validate_row(row: dict[str, Any]) -> None:
    if not isinstance(row.get("sample_id"), str) or not row["sample_id"]:
        raise ValueError("bounded normalization row sample_id must be a non-empty string")
    if row.get("sample_policy_schema") != REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA:
        raise ValueError("bounded normalization row sample policy schema is unsupported")
    if row.get("status") not in {
        "real_sample_v2_sample_aware_weight_policy_pass",
        "real_sample_v2_sample_aware_weight_policy_fail",
    }:
        raise ValueError("bounded normalization row status is unsupported")
    source = row["source"]
    if int(source["source_gaussians"]) < 1:
        raise ValueError("bounded normalization row source_gaussians must be positive")
    if source.get("compatible") is not True:
        raise ValueError("bounded normalization row source must be compatible")
    fixed = row["fixed_target"]
    if int(fixed["max_points"]) < 1:
        raise ValueError("bounded normalization row max_points must be positive")
    if float(fixed["solver_temperature"]) <= 0.0:
        raise ValueError("bounded normalization row solver_temperature must be positive")
    if fixed.get("coverage_scan") != "disabled":
        raise ValueError("bounded normalization row must keep coverage scan disabled")
    if fixed.get("temperature_sharpening_scan") != "disabled":
        raise ValueError("bounded normalization row must keep sharpening scan disabled")

    selected = row["selected_policy"]
    if selected.get("candidate_name") not in {"baseline", "promoted", "bounded-normalized"}:
        raise ValueError("bounded normalization row selected candidate is unsupported")
    if selected.get("selected_for_viewer_export") is not True:
        raise ValueError("bounded normalization row selected candidate must be exportable")
    _validate_metrics(row["selected_metrics"])
    _validate_delta(row["selected_delta_vs_baseline"])
    _validate_changed(row["selected_changed_gaussians"])
    _validate_gate(row["selected_gate"])
    if int(row["selected_changed_gaussians"]["hard_regression_count"]) != int(
        row["selected_gate"]["hard_regression_count"]
    ):
        raise ValueError("bounded normalization selected hard regression counts mismatch")
    _validate_compact_candidate(row["promoted_candidate"])
    _validate_compact_candidate(row["bounded_normalized_candidate"])
    if row["evidence_normalization_status"] not in {
        "not_required_for_selected_policy",
        "satisfied_by_bounded_normalization",
        "required_before_global_weight_promotion",
    }:
        raise ValueError("bounded normalization row evidence status is unsupported")


def _validate_metrics(metrics: dict[str, Any]) -> None:
    if int(metrics["mixed_gaussians"]) < 0:
        raise ValueError("bounded normalization mixed_gaussians must be non-negative")
    if int(metrics["predicted_object_count"]) < 1:
        raise ValueError("bounded normalization predicted_object_count must be positive")
    for key in ("direct_slot_match", "assignment_confidence", "mean_normalized_entropy"):
        value = float(metrics[key])
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"bounded normalization metric {key} must be in [0, 1]")
    if metrics["object_purity"] is not None:
        value = float(metrics["object_purity"])
        if not 0.0 <= value <= 1.0:
            raise ValueError("bounded normalization object_purity must be in [0, 1]")


def _validate_delta(delta: dict[str, Any]) -> None:
    int(delta["mixed_gaussians_delta"])
    float(delta["direct_slot_match_delta"])
    float(delta["assignment_confidence_delta"])
    float(delta["mean_normalized_entropy_delta"])
    int(delta["predicted_object_count_delta"])
    if delta["object_purity_delta"] is not None:
        float(delta["object_purity_delta"])


def _validate_changed(changed: dict[str, Any]) -> None:
    if int(changed["changed_count"]) < 0:
        raise ValueError("bounded normalization changed_count must be non-negative")
    if int(changed["hard_fix_count"]) < 0 or int(changed["hard_regression_count"]) < 0:
        raise ValueError("bounded normalization hard change counts must be non-negative")


def _validate_gate(gate: dict[str, Any]) -> None:
    if not isinstance(gate.get("eligible_for_sample"), bool):
        raise ValueError("bounded normalization gate eligibility must be boolean")
    _validate_changed(
        {
            "changed_count": 0,
            "hard_fix_count": gate["hard_fix_count"],
            "hard_regression_count": gate["hard_regression_count"],
        }
    )


def _validate_compact_candidate(candidate: dict[str, Any]) -> None:
    if candidate["name"] not in {"promoted", "bounded-normalized"}:
        raise ValueError("bounded normalization compact candidate name is unsupported")
    if float(candidate["feature_weight"]) < 0.0 or float(candidate["position_weight"]) < 0.0:
        raise ValueError("bounded normalization compact candidate weights must be non-negative")
    _validate_metrics(candidate["metrics"])
    _validate_delta(candidate["delta_vs_baseline"])
    _validate_changed(candidate["changed_gaussians"])
    _validate_gate(candidate["sample_policy_gate"])


def _candidate_by_name(summary: dict[str, Any], name: str) -> dict[str, Any]:
    for candidate in summary["candidates"]:
        if candidate["candidate"]["name"] == name:
            return candidate
    raise ValueError(f"sample-aware summary has no {name!r} candidate")


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": candidate["candidate"]["name"],
        "feature_weight": float(candidate["candidate"]["feature_weight"]),
        "position_weight": float(candidate["candidate"]["position_weight"]),
        "metrics": dict(candidate["metrics"]),
        "delta_vs_baseline": dict(candidate["delta_vs_baseline"]),
        "changed_gaussians": _compact_changed_gaussians(candidate),
        "sample_policy_gate": dict(candidate["sample_policy_gate"]),
        "bounded_evidence_normalization": candidate.get("bounded_evidence_normalization"),
    }


def _compact_changed_gaussians(candidate: dict[str, Any]) -> dict[str, Any]:
    changed = candidate["changed_gaussians"]
    return {
        "changed_count": int(changed["changed_count"]),
        "changed_fraction": float(changed["changed_fraction"]),
        "hard_fix_count": int(changed["hard_fix_count"]),
        "hard_regression_count": int(changed["hard_regression_count"]),
        "unchanged_count": int(changed["unchanged_count"]),
    }


def _aggregate(rows: list[dict[str, Any]], *, min_samples: int) -> dict[str, Any]:
    selected_policy_counts = Counter(
        str(row["selected_policy"]["candidate_name"]) for row in rows
    )
    evidence_status_counts = Counter(
        str(row["evidence_normalization_status"]) for row in rows
    )
    selected_hard_regression_samples = [
        row["sample_id"]
        for row in rows
        if int(row["selected_changed_gaussians"]["hard_regression_count"]) > 0
    ]
    blocked_promoted_samples = [
        row["sample_id"]
        for row in rows
        if row["promoted_candidate"]["sample_policy_gate"]["eligible_for_sample"] is False
    ]
    all_sample_policies_pass = all(
        row["status"] == "real_sample_v2_sample_aware_weight_policy_pass"
        for row in rows
    )
    selected_hard_regression_count = sum(
        int(row["selected_changed_gaussians"]["hard_regression_count"]) for row in rows
    )
    enough_samples = len(rows) >= int(min_samples)
    result = (
        "pass"
        if enough_samples and all_sample_policies_pass and selected_hard_regression_count == 0
        else "diagnostic"
    )
    return {
        "result": result,
        "sample_count": len(rows),
        "min_samples": int(min_samples),
        "enough_samples": enough_samples,
        "all_sample_policies_pass": all_sample_policies_pass,
        "selected_policy_counts": dict(sorted(selected_policy_counts.items())),
        "evidence_normalization_status_counts": dict(sorted(evidence_status_counts.items())),
        "selected_hard_regression_count": int(selected_hard_regression_count),
        "selected_hard_regression_samples": selected_hard_regression_samples,
        "blocked_promoted_sample_count": len(blocked_promoted_samples),
        "blocked_promoted_samples": blocked_promoted_samples,
    }


def _recommendation(aggregate: dict[str, Any]) -> dict[str, Any]:
    if aggregate["result"] == "pass":
        if int(aggregate["blocked_promoted_sample_count"]) > 0:
            decision = "sample_aware_bounded_normalization_cross_sample_pass"
            action = "keep_sample_aware_policy_and_add_more_small_real_public_samples"
            global_default = "sample_aware_policy_not_single_weight_default"
        else:
            decision = "single_weight_policy_cross_sample_non_regression_pass"
            action = "add_more_cross_sample_rows_before_global_default"
            global_default = "eligible_after_more_cross_sample_rows"
    else:
        decision = "bounded_normalization_cross_sample_diagnostic"
        action = "inspect_selected_policy_regressions_before_viewer_defaults"
        global_default = "blocked_by_cross_sample_gate"
    return {
        "decision": decision,
        "action": action,
        "global_default": global_default,
        "requires_more_cross_sample_rows": True,
        "requires_geometry_unfreeze": False,
        "requires_diffusion_replay_or_rollout": False,
        "requires_public_demo_claim_change": False,
    }
