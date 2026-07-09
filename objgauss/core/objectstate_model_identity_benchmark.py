from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.assignment_solver_v2 import AssignmentSolverV2State
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.objectstate_model_identity_gate import (
    OBJECTSTATE_MODEL_IDENTITY_BASELINES,
    OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA,
    ObjectStateModelIdentityGateThresholds,
    objectstate_model_identity_gate_summary,
    validate_objectstate_model_identity_gate_summary,
)

OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA = (
    "objgauss-objectstate-model-identity-benchmark-v1"
)
OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS = (
    "viewpoint", "dropout", "occlusion", "appearance", "spatial"
)
_IDENTITY_METRIC_FLOAT_KEYS = (
    "identity_retrieval_at_1", "identity_margin", "slot_swap_rate",
    "objectstate_drift", "assignment_consistency", "occlusion_recovery",
)
_IDENTITY_METRIC_INT_KEYS = (
    "identity_retrieval_evaluated_count", "identity_retrieval_correct_count",
    "slot_swap_count", "occlusion_recovery_check_count", "pairwise_distance_count",
)
_CLAIM_POLICY_KEYS = (
    "uses_existing_identity_gate", "uses_permutation_aware_identity_matching",
    "physical_identity_labels_are_evaluation_only",
    "assignment_matrix_is_single_source_of_truth", "tests_identity_under_perturbations",
    "does_not_claim_identity_ablation", "does_not_claim_temporal_assignment",
    "does_not_claim_reality_gate_pass", "does_not_claim_world_model",
)
_NON_GOAL_KEYS = (
    "trains_model", "uses_renderer_loss", "uses_temporal_loss",
    "uses_hungarian_dependency", "uses_transformer", "uses_slot_attention",
    "uses_replay_buffer", "uses_diffusion", "uses_dynamics_model",
    "runs_long_training", "mutates_viewer_defaults", "ingests_real_capture",
)


@dataclass(frozen=True)
class ObjectStateModelIdentityBenchmarkThresholds:
    identity_retrieval_lift_vs_xyz_min: float = 0.0
    identity_margin_min: float = 0.0
    occlusion_recovery_lift_vs_random_min: float = 0.0
    slot_swap_rate_max: float = 1.0

    def as_dict(self) -> dict[str, float]:
        payload = {
            "identity_retrieval_lift_vs_xyz_min": float(self.identity_retrieval_lift_vs_xyz_min),
            "identity_margin_min": float(self.identity_margin_min),
            "occlusion_recovery_lift_vs_random_min": float(self.occlusion_recovery_lift_vs_random_min),
            "slot_swap_rate_max": float(self.slot_swap_rate_max),
        }
        for key in (
            "identity_retrieval_lift_vs_xyz_min",
            "identity_margin_min",
            "occlusion_recovery_lift_vs_random_min",
        ):
            if payload[key] < 0.0:
                raise ValueError(f"{key} must be >= 0")
        if not 0.0 <= payload["slot_swap_rate_max"] <= 1.0:
            raise ValueError("slot_swap_rate_max must be in [0,1]")
        return payload


@dataclass(frozen=True)
class ObjectStateModelIdentityBenchmarkScenario:
    scenario_id: str
    perturbation_kind: str
    frame0_cloud: GaussianCloud
    frame0_identity_labels: np.ndarray
    frame1_cloud: GaussianCloud
    frame1_identity_labels: np.ndarray
    frame0_id: str = "t0"
    frame1_id: str = "t1"
    frame0_features: np.ndarray | None = None
    frame1_features: np.ndarray | None = None
    description: str = ""


def objectstate_model_identity_benchmark_summary(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
    solver_state: AssignmentSolverV2State,
    *,
    output_dir: str | Path,
    sample_id: str = "model-identity-benchmark-001",
    required_perturbations: Sequence[str] = OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS,
    thresholds: ObjectStateModelIdentityBenchmarkThresholds | None = None,
    identity_gate_thresholds: ObjectStateModelIdentityGateThresholds | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    scenario_list = _validate_scenarios(scenarios)
    required = _validate_perturbations(required_perturbations, label="required_perturbations")
    checked_thresholds = thresholds or ObjectStateModelIdentityBenchmarkThresholds()
    threshold_payload = checked_thresholds.as_dict()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    scenario_results = []
    for index, scenario in enumerate(scenario_list):
        scenario_dir = output_root / f"{index:02d}-{_safe_name(scenario.scenario_id)}"
        gate_summary = objectstate_model_identity_gate_summary(
            scenario.frame0_cloud,
            scenario.frame0_identity_labels,
            scenario.frame1_cloud,
            scenario.frame1_identity_labels,
            solver_state,
            output_dir=scenario_dir,
            sample_id=f"{sample_id}:{scenario.scenario_id}",
            frame0_id=scenario.frame0_id,
            frame1_id=scenario.frame1_id,
            frame0_features=scenario.frame0_features,
            frame1_features=scenario.frame1_features,
            thresholds=identity_gate_thresholds,
            seed=int(seed) + index * 29,
        )
        scenario_results.append(_scenario_result(scenario, gate_summary))

    baselines = _aggregate_baselines(scenario_results)
    coverage = _coverage(scenario_results, required)
    breakdown = _perturbation_breakdown(scenario_results)
    long_training_gate = _long_training_gate(
        baselines,
        coverage,
        thresholds=threshold_payload,
    )
    summary_path = output_root / "identity-benchmark-summary.json"
    payload = {
        "schema": OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA,
        "kind": "objectstate_model_identity_benchmark",
        "status": (
            "objectstate_model_identity_benchmark_candidate_ready"
            if long_training_gate["status"] == "candidate_ready"
            else "objectstate_model_identity_benchmark_blocked"
        ),
        "sample_id": str(sample_id),
        "identity_gate_schema": OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA,
        "num_scenarios": int(len(scenario_results)),
        "num_pairs": int(
            sum(
                int(item["metrics"]["identity_retrieval_evaluated_count"])
                for item in scenario_results
            )
        ),
        "required_perturbations": list(required),
        "perturbation_coverage": coverage,
        "thresholds": threshold_payload,
        "baselines": baselines,
        "perturbation_breakdown": breakdown,
        "scenario_results": scenario_results,
        "long_training_gate": long_training_gate,
        "artifact_refs": {
            "identity_benchmark_summary": str(summary_path),
            "scenario_summaries": [
                str(item["summary_path"]) for item in scenario_results
            ],
        },
        "claim_policy": {key: True for key in _CLAIM_POLICY_KEYS},
        "non_goals": {key: False for key in _NON_GOAL_KEYS},
    }
    checked = validate_objectstate_model_identity_benchmark_summary(payload)
    summary_path.write_text(json.dumps(checked, indent=2, sort_keys=True), encoding="utf-8")
    checked["summary_path"] = str(summary_path)
    return validate_objectstate_model_identity_benchmark_summary(checked)


def validate_objectstate_model_identity_benchmark_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("model identity benchmark summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA:
        raise ValueError(
            f"unsupported model identity benchmark schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_model_identity_benchmark":
        raise ValueError("model identity benchmark kind is unsupported")
    if payload.get("status") not in {
        "objectstate_model_identity_benchmark_candidate_ready",
        "objectstate_model_identity_benchmark_blocked",
    }:
        raise ValueError("model identity benchmark status is unsupported")
    if payload.get("identity_gate_schema") != OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA:
        raise ValueError("model identity benchmark must reference identity gate schema")
    if int(payload.get("num_scenarios", 0)) < 1:
        raise ValueError("model identity benchmark requires at least one scenario")
    if int(payload.get("num_pairs", 0)) < 1:
        raise ValueError("model identity benchmark requires at least one identity pair")
    required = _validate_perturbations(
        payload.get("required_perturbations", ()),
        label="required_perturbations",
    )
    coverage = _mapping(payload, "perturbation_coverage")
    for kind in required:
        if kind not in coverage or not isinstance(coverage[kind], bool):
            raise ValueError("model identity benchmark coverage must report required perturbations")
    baselines = _mapping(payload, "baselines")
    for name in OBJECTSTATE_MODEL_IDENTITY_BASELINES:
        _baseline_aggregate(baselines, name)
    breakdown = _mapping(payload, "perturbation_breakdown")
    for kind in OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS:
        if kind not in breakdown:
            raise ValueError(f"model identity benchmark missing perturbation {kind}")
        _mapping(breakdown[kind], "baselines")
    scenario_results = payload.get("scenario_results")
    if not isinstance(scenario_results, list) or not scenario_results:
        raise ValueError("model identity benchmark requires scenario_results")
    if len(scenario_results) != int(payload["num_scenarios"]):
        raise ValueError("model identity benchmark num_scenarios mismatch")
    for item in scenario_results:
        _validate_scenario_result(item)
    gate = _mapping(payload, "long_training_gate")
    if gate.get("status") not in {"blocked", "candidate_ready"}:
        raise ValueError("model identity benchmark long_training_gate status is unsupported")
    expected_status = (
        "objectstate_model_identity_benchmark_candidate_ready"
        if gate["status"] == "candidate_ready"
        else "objectstate_model_identity_benchmark_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("model identity benchmark status must match long_training_gate")
    checks = _mapping(gate, "checks")
    if not checks or any(not isinstance(value, bool) for value in checks.values()):
        raise ValueError("model identity benchmark checks must be booleans")
    reasons = gate.get("reasons")
    if not isinstance(reasons, list):
        raise ValueError("model identity benchmark reasons must be a list")
    if gate["status"] == "candidate_ready" and reasons:
        raise ValueError("candidate_ready benchmark cannot have blocker reasons")
    if gate["status"] == "blocked" and not reasons:
        raise ValueError("blocked benchmark must explain reasons")
    claim_policy = _mapping(payload, "claim_policy")
    if any(not claim_policy.get(key) for key in _CLAIM_POLICY_KEYS):
        raise ValueError("model identity benchmark must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("model identity benchmark cannot claim non-goals")
    if "summary_path" in payload and not isinstance(payload["summary_path"], str):
        raise ValueError("model identity benchmark summary_path must be a string")
    return dict(payload)


def _validate_scenarios(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> tuple[ObjectStateModelIdentityBenchmarkScenario, ...]:
    if not scenarios:
        raise ValueError("model identity benchmark requires at least one scenario")
    validated = []
    seen = set()
    for scenario in scenarios:
        if not isinstance(scenario, ObjectStateModelIdentityBenchmarkScenario):
            raise TypeError("scenarios must be ObjectStateModelIdentityBenchmarkScenario")
        scenario_id = str(scenario.scenario_id)
        if not scenario_id:
            raise ValueError("scenario_id must be non-empty")
        if scenario_id in seen:
            raise ValueError(f"duplicate model identity benchmark scenario_id: {scenario_id}")
        if scenario.perturbation_kind not in OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS:
            raise ValueError(f"unsupported identity perturbation: {scenario.perturbation_kind}")
        validated.append(scenario)
        seen.add(scenario_id)
    return tuple(validated)


def _validate_perturbations(value: Sequence[str], *, label: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{label} must be a sequence")
    if not value:
        raise ValueError(f"{label} must not be empty")
    seen = set()
    result = []
    for item in value:
        kind = str(item)
        if kind in seen:
            raise ValueError(f"duplicate perturbation kind: {kind}")
        if kind not in OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS:
            raise ValueError(f"unsupported identity perturbation: {kind}")
        result.append(kind)
        seen.add(kind)
    return tuple(result)


def _scenario_result(
    scenario: ObjectStateModelIdentityBenchmarkScenario,
    gate_summary: Mapping[str, Any],
) -> dict[str, Any]:
    checked = validate_objectstate_model_identity_gate_summary(gate_summary)
    baselines = {
        name: {"metrics": _plain_metrics(checked["baselines"][name]["metrics"])}
        for name in OBJECTSTATE_MODEL_IDENTITY_BASELINES
    }
    return {
        "scenario_id": str(scenario.scenario_id),
        "perturbation_kind": str(scenario.perturbation_kind),
        "description": str(scenario.description),
        "status": checked["status"],
        "gate_status": checked["gate_status"],
        "frames": checked["frames"],
        "metrics": _plain_metrics(checked["metrics"]),
        "baselines": baselines,
        "baseline_comparison": checked["baseline_comparison"],
        "summary_path": checked["summary_path"],
        "artifact_refs": checked["artifact_refs"],
    }


def _aggregate_baselines(
    scenario_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        name: _aggregate_metric_rows(
            [item["baselines"][name]["metrics"] for item in scenario_results]
        )
        for name in OBJECTSTATE_MODEL_IDENTITY_BASELINES
    }


def _perturbation_breakdown(
    scenario_results: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    breakdown = {}
    for kind in OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS:
        matches = [
            item for item in scenario_results if item["perturbation_kind"] == kind
        ]
        breakdown[kind] = {
            "status": "covered" if matches else "missing",
            "num_scenarios": int(len(matches)),
            "num_pairs": int(
                sum(int(item["metrics"]["identity_retrieval_evaluated_count"]) for item in matches)
            ),
            "scenario_ids": [str(item["scenario_id"]) for item in matches],
            "baselines": _aggregate_baselines(matches),
        }
    return breakdown


def _aggregate_metric_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "num_scenarios": 0,
            "num_pairs": 0,
            "metrics": _zero_metrics(),
        }
    evaluated = sum(int(row["identity_retrieval_evaluated_count"]) for row in rows)
    correct = sum(int(row["identity_retrieval_correct_count"]) for row in rows)
    occlusion_checks = sum(int(row["occlusion_recovery_check_count"]) for row in rows)
    slot_swaps = sum(int(row["slot_swap_count"]) for row in rows)
    pairwise = sum(int(row["pairwise_distance_count"]) for row in rows)
    return {
        "num_scenarios": int(len(rows)),
        "num_pairs": int(evaluated),
        "metrics": {
            "identity_retrieval_at_1": float(correct / evaluated) if evaluated else 0.0,
            "identity_margin": _weighted_mean(rows, "identity_margin", "identity_retrieval_evaluated_count"),
            "slot_swap_rate": float(slot_swaps / evaluated) if evaluated else 0.0,
            "objectstate_drift": _weighted_mean(
                rows,
                "objectstate_drift",
                "identity_retrieval_evaluated_count",
            ),
            "assignment_consistency": float(
                np.mean([float(row["assignment_consistency"]) for row in rows])
            ),
            "occlusion_recovery": (
                _weighted_mean(rows, "occlusion_recovery", "occlusion_recovery_check_count")
                if occlusion_checks
                else 0.0
            ),
            "identity_retrieval_evaluated_count": int(evaluated),
            "identity_retrieval_correct_count": int(correct),
            "slot_swap_count": int(slot_swaps),
            "occlusion_recovery_check_count": int(occlusion_checks),
            "pairwise_distance_count": int(pairwise),
        },
    }


def _coverage(
    scenario_results: Sequence[Mapping[str, Any]],
    required: Sequence[str],
) -> dict[str, bool]:
    covered = {str(item["perturbation_kind"]) for item in scenario_results}
    return {kind: kind in covered for kind in required}


def _long_training_gate(
    baselines: Mapping[str, Mapping[str, Any]],
    coverage: Mapping[str, bool],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    candidate = _baseline_aggregate(baselines, "assignment_solver_v2")["metrics"]
    xyz = _baseline_aggregate(baselines, "xyz_centroid")["metrics"]
    random = _baseline_aggregate(baselines, "random_assignment")["metrics"]
    oracle = _baseline_aggregate(baselines, "oracle_target_assignment")["metrics"]
    retrieval_lift = float(
        candidate["identity_retrieval_at_1"] - xyz["identity_retrieval_at_1"]
    )
    occlusion_lift = float(
        candidate["occlusion_recovery"] - random["occlusion_recovery"]
    )
    slot_swap_rate = float(candidate["slot_swap_rate"])
    checks = {
        "required_perturbations_covered": all(bool(value) for value in coverage.values()),
        "assignment_solver_v2_retrieval_above_xyz_centroid": retrieval_lift
        > float(thresholds["identity_retrieval_lift_vs_xyz_min"]),
        "assignment_solver_v2_identity_margin_positive": float(candidate["identity_margin"])
        > float(thresholds["identity_margin_min"]),
        "assignment_solver_v2_occlusion_recovery_above_random": occlusion_lift
        > float(thresholds["occlusion_recovery_lift_vs_random_min"]),
        "assignment_solver_v2_slot_swap_rate_reported_and_bounded": np.isfinite(slot_swap_rate)
        and slot_swap_rate <= float(thresholds["slot_swap_rate_max"]),
        "oracle_target_assignment_remains_upper_bound": float(
            oracle["identity_retrieval_at_1"]
        )
        + 1e-9
        >= float(candidate["identity_retrieval_at_1"]),
    }
    reasons = [name for name, passed in checks.items() if not bool(passed)]
    return {
        "status": "candidate_ready" if not reasons else "blocked",
        "checks": checks,
        "reasons": reasons,
        "thresholds": dict(thresholds),
        "comparison": {
            "candidate_retrieval_lift_vs_xyz_centroid": retrieval_lift,
            "candidate_occlusion_recovery_lift_vs_random": occlusion_lift,
            "candidate_retrieval_gap_to_oracle": float(
                oracle["identity_retrieval_at_1"] - candidate["identity_retrieval_at_1"]
            ),
            "candidate_slot_swap_rate": slot_swap_rate,
        },
    }


def _validate_scenario_result(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("model identity benchmark scenario result must be a mapping")
    if not payload.get("scenario_id"):
        raise ValueError("model identity benchmark scenario result requires scenario_id")
    if payload.get("perturbation_kind") not in OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS:
        raise ValueError("model identity benchmark scenario result has unsupported perturbation")
    if payload.get("status") not in {
        "objectstate_model_identity_gate_pass",
        "objectstate_model_identity_gate_fail",
    }:
        raise ValueError("model identity benchmark scenario result has unsupported status")
    metrics = _mapping(payload, "metrics")
    _finite(metrics.get("identity_retrieval_at_1"), "scenario.metrics.identity_retrieval_at_1")
    baselines = _mapping(payload, "baselines")
    for name in OBJECTSTATE_MODEL_IDENTITY_BASELINES:
        _mapping(baselines, name)
    summary_path = payload.get("summary_path")
    if not isinstance(summary_path, str) or not summary_path:
        raise ValueError("model identity benchmark scenario result requires summary_path")


def _baseline_aggregate(
    baselines: Mapping[str, Any],
    name: str,
) -> Mapping[str, Any]:
    value = baselines.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"model identity benchmark missing baseline {name}")
    metrics = value.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError(f"model identity benchmark baseline {name} missing metrics")
    for key in _IDENTITY_METRIC_FLOAT_KEYS:
        _finite(metrics.get(key), f"baselines.{name}.{key}")
    return value


def _plain_metrics(metrics: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{key: float(metrics[key]) for key in _IDENTITY_METRIC_FLOAT_KEYS},
        **{key: int(metrics[key]) for key in _IDENTITY_METRIC_INT_KEYS},
    }


def _zero_metrics() -> dict[str, Any]:
    return {
        **{key: 0.0 for key in _IDENTITY_METRIC_FLOAT_KEYS},
        **{key: 0 for key in _IDENTITY_METRIC_INT_KEYS},
    }


def _weighted_mean(
    rows: Sequence[Mapping[str, Any]],
    metric_key: str,
    weight_key: str,
) -> float:
    total_weight = sum(int(row[weight_key]) for row in rows)
    if total_weight <= 0:
        return 0.0
    total = sum(float(row[metric_key]) * int(row[weight_key]) for row in rows)
    return float(total / total_weight)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"model identity benchmark requires {key}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _safe_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in {"-", "_"} else "-" for char in str(value))
    return safe.strip("-") or "scenario"
