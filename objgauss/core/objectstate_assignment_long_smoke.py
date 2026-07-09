from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.assignment_evidence import AssignmentEvidenceBatch
from objgauss.core.assignment_solver_v2 import (
    ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
    ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA,
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
    assignment_solver_v2_state_from_dict,
    train_assignment_solver_v2,
    validate_assignment_solver_v2_training_summary,
)
from objgauss.core.features import positions
from objgauss.core.objectstate_assignment_long_smoke_contract import (
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SUMMARY_SCHEMA,
    ObjectStateAssignmentLongSmokeContractThresholds,
    objectstate_assignment_long_smoke_contract_summary,
    validate_objectstate_assignment_long_smoke_contract_summary,
)
from objgauss.core.objectstate_model_identity_benchmark import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS,
    ObjectStateModelIdentityBenchmarkScenario,
    objectstate_model_identity_benchmark_summary,
    validate_objectstate_model_identity_benchmark_summary,
)
from objgauss.core.objectstate_model_identity_benchmark_report import (
    objectstate_model_identity_benchmark_report_difficulty_by_scenario,
    objectstate_model_identity_benchmark_report_scenarios,
)

OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA = (
    "objgauss-objectstate-assignment-long-smoke-v1"
)
_METRIC_KEYS = (
    "identity_retrieval_at_1",
    "identity_margin",
    "slot_swap_rate",
    "objectstate_drift",
    "assignment_consistency",
    "occlusion_recovery",
)
_GAP_KEYS = (
    "identity_retrieval_at_1",
    "occlusion_recovery",
    "slot_swap_rate",
)
_SUCCESS_CHECKS = (
    "held_out_identity_retrieval_at_1_not_decrease",
    "identity_margin_improves",
    "occlusion_recovery_not_decrease",
    "generalization_gap_not_expand",
    "slot_swap_rate_interpretable",
    "checkpoint_roundtrip",
    "duration_within_limit",
)
_CLAIM_POLICY_KEYS = (
    "semantic_policy_only",
    "uses_passed_teacher_evidence_leakage_audit",
    "bounded_duration",
    "fixed_seed",
    "checkpoint_roundtrip_verified",
    "before_after_identity_benchmark_ran",
    "held_out_generalization_evaluated",
    "does_not_claim_real_data_identity_pass",
    "does_not_claim_world_model",
)
_NON_GOAL_KEYS = (
    "runs_teacher_model",
    "downloads_teacher_weights",
    "uses_gpu",
    "uses_renderer_loss",
    "uses_temporal_loss",
    "uses_dynamics",
    "uses_diffusion",
    "uses_replay_buffer",
    "ingests_real_capture",
    "mutates_viewer_defaults",
)


def objectstate_assignment_long_smoke_summary(
    output_dir: str | Path,
    *,
    sample_id: str = "objectstate-assignment-long-smoke-001",
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario] | None = None,
    teacher_evidence_leakage_audit: Mapping[str, Any] | None,
    thresholds: ObjectStateAssignmentLongSmokeContractThresholds | None = None,
    iterations: int = 120,
    learning_rate: float = 0.4,
    seed: int = 0,
) -> dict[str, Any]:
    threshold_config = thresholds or ObjectStateAssignmentLongSmokeContractThresholds()
    threshold_payload = threshold_config.as_dict()
    contract = objectstate_assignment_long_smoke_contract_summary(
        sample_id=f"{sample_id}:contract",
        teacher_evidence_leakage_audit=teacher_evidence_leakage_audit,
        thresholds=threshold_config,
    )
    if contract["readiness_gate"]["long_smoke_contract_ready"] is not True:
        raise ValueError("assignment long smoke requires ready long-smoke contract")
    if iterations < 1:
        raise ValueError("assignment long smoke iterations must be >= 1")
    if iterations > 600:
        raise ValueError("assignment long smoke iterations must stay <= 600")
    if learning_rate <= 0.0:
        raise ValueError("assignment long smoke learning_rate must be > 0")

    scenario_list = tuple(
        scenarios or objectstate_model_identity_benchmark_report_scenarios()
    )
    _validate_report_ladder(scenario_list)
    difficulty = objectstate_model_identity_benchmark_report_difficulty_by_scenario()
    train_scenarios = tuple(
        scenario
        for scenario in scenario_list
        if difficulty[scenario.scenario_id] in {"easy", "medium"}
    )
    held_out_scenarios = tuple(
        scenario
        for scenario in scenario_list
        if difficulty[scenario.scenario_id] == "hard"
    )
    if not train_scenarios or not held_out_scenarios:
        raise ValueError("assignment long smoke requires train and held-out scenarios")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = output_root / "assignment-long-smoke-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    started = time.perf_counter()
    initial_state = _initial_semantic_state(scenario_list)
    training = train_assignment_solver_v2(
        _training_batches(train_scenarios),
        initial_state=initial_state,
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        cluster_weight=0.0,
        entropy_weight=0.0,
        balance_weight=0.0,
        supervised_weight=1.0,
        seed=int(seed),
        record_every=max(1, int(iterations) // 4),
    )
    training_summary = validate_assignment_solver_v2_training_summary(
        training.as_dict(include_state_arrays=False)
    )

    all_before = _identity_benchmark(
        scenario_list,
        training.initial_state,
        artifact_root / "all-before",
        sample_id=f"{sample_id}:all-before",
        seed=seed + 11,
    )
    all_after = _identity_benchmark(
        scenario_list,
        training.final_state,
        artifact_root / "all-after",
        sample_id=f"{sample_id}:all-after",
        seed=seed + 22,
    )
    train_before = _identity_benchmark(
        train_scenarios,
        training.initial_state,
        artifact_root / "train-before",
        sample_id=f"{sample_id}:train-before",
        seed=seed + 33,
    )
    train_after = _identity_benchmark(
        train_scenarios,
        training.final_state,
        artifact_root / "train-after",
        sample_id=f"{sample_id}:train-after",
        seed=seed + 44,
    )
    held_out_before = _identity_benchmark(
        held_out_scenarios,
        training.initial_state,
        artifact_root / "held-out-before",
        sample_id=f"{sample_id}:held-out-before",
        seed=seed + 55,
    )
    held_out_after = _identity_benchmark(
        held_out_scenarios,
        training.final_state,
        artifact_root / "held-out-after",
        sample_id=f"{sample_id}:held-out-after",
        seed=seed + 66,
    )

    checkpoint_path = output_root / "assignment-long-smoke-final-state.json"
    checkpoint = training.final_state.as_dict(include_arrays=True)
    checkpoint_path.write_text(
        json.dumps(checkpoint, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    restored = assignment_solver_v2_state_from_dict(checkpoint)
    restored_held_out_after = _identity_benchmark(
        held_out_scenarios,
        restored,
        artifact_root / "held-out-after-restored",
        sample_id=f"{sample_id}:held-out-after-restored",
        seed=seed + 77,
    )
    roundtrip = _checkpoint_roundtrip(held_out_after, restored_held_out_after)

    observed_duration = float(time.perf_counter() - started)
    train_metrics_before = _solver_metrics(train_before)
    train_metrics_after = _solver_metrics(train_after)
    held_metrics_before = _solver_metrics(held_out_before)
    held_metrics_after = _solver_metrics(held_out_after)
    before_gap = _generalization_gap(train_metrics_before, held_metrics_before)
    after_gap = _generalization_gap(train_metrics_after, held_metrics_after)
    success = _success_checks(
        held_metrics_before,
        held_metrics_after,
        before_gap=before_gap,
        after_gap=after_gap,
        roundtrip_ok=roundtrip["roundtrip_ok"],
        observed_duration_seconds=observed_duration,
        thresholds=threshold_payload,
    )
    status = (
        "objectstate_assignment_long_smoke_pass"
        if all(check["passed"] for check in success.values())
        else "objectstate_assignment_long_smoke_reviewable"
    )
    summary_path = output_root / "assignment-long-smoke-summary.json"
    payload = {
        "schema": OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA,
        "kind": "objectstate_assignment_long_smoke",
        "status": status,
        "sample_id": str(sample_id),
        "contract_schema": OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SUMMARY_SCHEMA,
        "solver_state_schema": ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
        "training_schema": ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA,
        "contract": contract,
        "run_config": {
            "evidence_policy": "semantic",
            "iterations": int(iterations),
            "learning_rate": float(learning_rate),
            "seed": int(seed),
            "max_duration_seconds": int(threshold_payload["max_duration_seconds"]),
            "train_difficulties": ["easy", "medium"],
            "held_out_difficulties": ["hard"],
        },
        "duration": {
            "observed_seconds": observed_duration,
            "max_duration_seconds": int(threshold_payload["max_duration_seconds"]),
            "within_limit": observed_duration
            <= float(threshold_payload["max_duration_seconds"]),
        },
        "training": {
            "summary": training_summary,
            "loss_decreased": bool(
                training.final_loss.total_loss < training.initial_loss.total_loss
            ),
            "supervised_loss_decreased": bool(
                training.final_loss.supervised_loss
                < training.initial_loss.supervised_loss
            ),
        },
        "identity_benchmarks": {
            "all_before": _benchmark_digest(all_before),
            "all_after": _benchmark_digest(all_after),
            "train_before": _benchmark_digest(train_before),
            "train_after": _benchmark_digest(train_after),
            "held_out_before": _benchmark_digest(held_out_before),
            "held_out_after": _benchmark_digest(held_out_after),
        },
        "metrics": {
            "train_before": train_metrics_before,
            "train_after": train_metrics_after,
            "held_out_before": held_metrics_before,
            "held_out_after": held_metrics_after,
            "generalization_gap_before": before_gap,
            "generalization_gap_after": after_gap,
        },
        "success_checks": success,
        "checkpoint": {
            "path": str(checkpoint_path),
            "schema": checkpoint["schema"],
            **roundtrip,
        },
        "artifact_refs": {
            "summary": str(summary_path),
            "artifact_root": str(artifact_root),
            "checkpoint": str(checkpoint_path),
            "before_identity_benchmark_summary": all_before["summary_path"],
            "after_identity_benchmark_summary": all_after["summary_path"],
            "held_out_before_identity_benchmark_summary": held_out_before[
                "summary_path"
            ],
            "held_out_after_identity_benchmark_summary": held_out_after[
                "summary_path"
            ],
            "held_out_after_restored_identity_benchmark_summary": restored_held_out_after[
                "summary_path"
            ],
        },
        "next_stage_gate": {
            "temporal_assignment_contract_allowed": status
            == "objectstate_assignment_long_smoke_pass",
            "status": (
                "pass"
                if status == "objectstate_assignment_long_smoke_pass"
                else "reviewable"
            ),
            "blocked_reasons": [
                name for name, check in success.items() if not bool(check["passed"])
            ],
            "next_recommended_pr": (
                "OBJECTSTATE-TEMPORAL-ASSIGNMENT-CONTRACT-001"
                if status == "objectstate_assignment_long_smoke_pass"
                else None
            ),
        },
        "claim_policy": {key: True for key in _CLAIM_POLICY_KEYS},
        "non_goals": {key: False for key in _NON_GOAL_KEYS},
    }
    checked = validate_objectstate_assignment_long_smoke_summary(payload)
    summary_path.write_text(
        json.dumps(checked, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checked["summary_path"] = str(summary_path)
    return validate_objectstate_assignment_long_smoke_summary(checked)


def validate_objectstate_assignment_long_smoke_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("assignment long smoke summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA:
        raise ValueError(
            f"unsupported assignment long smoke schema: {payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_assignment_long_smoke":
        raise ValueError("assignment long smoke kind is unsupported")
    if payload.get("status") not in {
        "objectstate_assignment_long_smoke_pass",
        "objectstate_assignment_long_smoke_reviewable",
    }:
        raise ValueError("assignment long smoke status is unsupported")
    if (
        payload.get("contract_schema")
        != OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SUMMARY_SCHEMA
    ):
        raise ValueError("assignment long smoke contract schema mismatch")
    validate_objectstate_assignment_long_smoke_contract_summary(
        _mapping(payload, "contract")
    )
    config = _mapping(payload, "run_config")
    if config.get("evidence_policy") != "semantic":
        raise ValueError("assignment long smoke must use semantic evidence policy")
    if _positive_int(config.get("iterations"), "iterations") > 600:
        raise ValueError("assignment long smoke iterations must stay <= 600")
    _positive_number(config.get("learning_rate"), "learning_rate")
    duration = _mapping(payload, "duration")
    if duration.get("within_limit") is not True:
        raise ValueError("assignment long smoke duration must be within limit")
    if float(duration.get("observed_seconds", 0.0)) > float(
        duration["max_duration_seconds"]
    ):
        raise ValueError("assignment long smoke observed duration exceeds limit")
    training = _mapping(payload, "training")
    validate_assignment_solver_v2_training_summary(
        dict(_mapping(training, "summary"))
    )
    if not isinstance(training.get("loss_decreased"), bool):
        raise ValueError("assignment long smoke loss_decreased must be bool")
    if not isinstance(training.get("supervised_loss_decreased"), bool):
        raise ValueError(
            "assignment long smoke supervised_loss_decreased must be bool"
        )
    benchmarks = _mapping(payload, "identity_benchmarks")
    for key in (
        "all_before",
        "all_after",
        "train_before",
        "train_after",
        "held_out_before",
        "held_out_after",
    ):
        _validate_benchmark_digest(_mapping(benchmarks, key))
    metrics = _mapping(payload, "metrics")
    for key in ("train_before", "train_after", "held_out_before", "held_out_after"):
        _validate_metric_digest(_mapping(metrics, key))
    _validate_gap(_mapping(metrics, "generalization_gap_before"))
    _validate_gap(_mapping(metrics, "generalization_gap_after"))
    success = _mapping(payload, "success_checks")
    if set(success) != set(_SUCCESS_CHECKS):
        raise ValueError("assignment long smoke success checks mismatch")
    for key in _SUCCESS_CHECKS:
        _validate_success_check(_mapping(success, key), key)
    all_pass = all(bool(check["passed"]) for check in success.values())
    if payload["status"] == "objectstate_assignment_long_smoke_pass" and not all_pass:
        raise ValueError("assignment long smoke pass requires all checks to pass")
    if payload["status"] == "objectstate_assignment_long_smoke_reviewable" and all_pass:
        raise ValueError("assignment long smoke reviewable contradicts passing checks")
    checkpoint = _mapping(payload, "checkpoint")
    if checkpoint.get("roundtrip_ok") is not True:
        raise ValueError("assignment long smoke requires checkpoint roundtrip")
    artifacts = _mapping(payload, "artifact_refs")
    for key in (
        "summary",
        "artifact_root",
        "checkpoint",
        "before_identity_benchmark_summary",
        "after_identity_benchmark_summary",
        "held_out_before_identity_benchmark_summary",
        "held_out_after_identity_benchmark_summary",
        "held_out_after_restored_identity_benchmark_summary",
    ):
        if not isinstance(artifacts.get(key), str) or not artifacts[key]:
            raise ValueError(f"assignment long smoke missing artifact ref {key}")
    gate = _mapping(payload, "next_stage_gate")
    if gate.get("status") not in {"pass", "reviewable"}:
        raise ValueError("assignment long smoke next stage gate status unsupported")
    if gate.get("temporal_assignment_contract_allowed") is not all_pass:
        raise ValueError("assignment long smoke next stage gate contradicts checks")
    claim_policy = _mapping(payload, "claim_policy")
    if any(not bool(claim_policy.get(key)) for key in _CLAIM_POLICY_KEYS):
        raise ValueError("assignment long smoke must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(non_goals.get(key)) for key in _NON_GOAL_KEYS):
        raise ValueError("assignment long smoke cannot claim non-goals")
    if "summary_path" in payload and not isinstance(payload["summary_path"], str):
        raise ValueError("assignment long smoke summary_path must be a string")
    return dict(payload)


def _training_batches(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> tuple[AssignmentEvidenceBatch, ...]:
    batches = []
    for index, scenario in enumerate(scenarios):
        labels = np.asarray(scenario.frame0_identity_labels, dtype=np.int64)
        target = np.eye(int(labels.max()) + 1, dtype=np.float32)[labels]
        batches.append(
            AssignmentEvidenceBatch(
                positions=positions(scenario.frame0_cloud),
                features=_feature_matrix(
                    scenario.frame0_features,
                    rows=scenario.frame0_cloud.count,
                    label=f"{scenario.scenario_id}.frame0_features",
                ),
                frame_index=index,
                target_assignment=target,
                source=f"assignment-long-smoke:{scenario.scenario_id}:frame0",
            )
        )
    return tuple(batches)


def _initial_semantic_state(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> AssignmentSolverV2State:
    reference = scenarios[0]
    features = _feature_matrix(
        reference.frame0_features,
        rows=reference.frame0_cloud.count,
        label="reference.frame0_features",
    )
    labels = np.asarray(reference.frame0_identity_labels, dtype=np.int64)
    slots = int(labels.max()) + 1
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(
            slots=slots,
            feature_dim=int(features.shape[1]),
            temperature=0.5,
            feature_weight=1.0,
            position_weight=0.0,
        ),
        feature_centers=np.zeros((slots, features.shape[1]), dtype=np.float32),
        position_centers=np.zeros((slots, 3), dtype=np.float32),
        slot_bias=np.zeros(slots, dtype=np.float32),
        source="assignment_long_smoke_semantic_collapsed_init",
    )


def _identity_benchmark(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
    state: AssignmentSolverV2State,
    output_dir: Path,
    *,
    sample_id: str,
    seed: int,
) -> dict[str, Any]:
    required = tuple(
        kind
        for kind in OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS
        if any(scenario.perturbation_kind == kind for scenario in scenarios)
    )
    return validate_objectstate_model_identity_benchmark_summary(
        objectstate_model_identity_benchmark_summary(
            scenarios,
            state,
            output_dir=output_dir,
            sample_id=sample_id,
            evidence_policy="semantic",
            evidence_policy_source="assignment_long_smoke_teacher_evidence",
            native_gaussian_evidence_only=False,
            uses_semantic_evidence=True,
            required_perturbations=required,
            seed=seed,
        )
    )


def _solver_metrics(benchmark: Mapping[str, Any]) -> dict[str, float]:
    return _metric_digest(benchmark["baselines"]["assignment_solver_v2"]["metrics"])


def _metric_digest(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in _METRIC_KEYS}


def _benchmark_digest(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_objectstate_model_identity_benchmark_summary(benchmark)
    return {
        "schema": checked["schema"],
        "status": checked["status"],
        "sample_id": checked["sample_id"],
        "num_scenarios": int(checked["num_scenarios"]),
        "num_pairs": int(checked["num_pairs"]),
        "metrics": _solver_metrics(checked),
        "long_training_gate": checked["long_training_gate"],
        "summary_path": checked["summary_path"],
    }


def _generalization_gap(
    train_metrics: Mapping[str, float],
    held_out_metrics: Mapping[str, float],
) -> dict[str, float]:
    return {
        key: abs(float(train_metrics[key]) - float(held_out_metrics[key]))
        for key in _GAP_KEYS
    }


def _success_checks(
    before: Mapping[str, float],
    after: Mapping[str, float],
    *,
    before_gap: Mapping[str, float],
    after_gap: Mapping[str, float],
    roundtrip_ok: bool,
    observed_duration_seconds: float,
    thresholds: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    retrieval_delta = float(
        after["identity_retrieval_at_1"] - before["identity_retrieval_at_1"]
    )
    margin_delta = float(after["identity_margin"] - before["identity_margin"])
    occlusion_delta = float(after["occlusion_recovery"] - before["occlusion_recovery"])
    before_gap_value = _max_gap(before_gap)
    after_gap_value = _max_gap(after_gap)
    gap_delta = float(after_gap_value - before_gap_value)
    slot_swap_rate = float(after["slot_swap_rate"])
    duration_limit = float(thresholds["max_duration_seconds"])

    return {
        "held_out_identity_retrieval_at_1_not_decrease": _check(
            "held_out_identity_retrieval_at_1_not_decrease",
            metric="identity_retrieval_at_1",
            passed=retrieval_delta
            >= -float(thresholds["max_identity_retrieval_drop"]),
            before=float(before["identity_retrieval_at_1"]),
            after=float(after["identity_retrieval_at_1"]),
            delta=retrieval_delta,
        ),
        "identity_margin_improves": _check(
            "identity_margin_improves",
            metric="identity_margin",
            passed=margin_delta > float(thresholds["min_identity_margin_delta"]),
            before=float(before["identity_margin"]),
            after=float(after["identity_margin"]),
            delta=margin_delta,
        ),
        "occlusion_recovery_not_decrease": _check(
            "occlusion_recovery_not_decrease",
            metric="occlusion_recovery",
            passed=occlusion_delta
            >= -float(thresholds["max_occlusion_recovery_drop"]),
            before=float(before["occlusion_recovery"]),
            after=float(after["occlusion_recovery"]),
            delta=occlusion_delta,
        ),
        "generalization_gap_not_expand": _check(
            "generalization_gap_not_expand",
            metric="generalization_gap",
            passed=gap_delta <= float(thresholds["max_generalization_gap_increase"]),
            before=before_gap_value,
            after=after_gap_value,
            delta=gap_delta,
        ),
        "slot_swap_rate_interpretable": _check(
            "slot_swap_rate_interpretable",
            metric="slot_swap_rate",
            passed=np.isfinite(slot_swap_rate)
            and slot_swap_rate <= float(thresholds["max_slot_swap_rate"]),
            before=float(before["slot_swap_rate"]),
            after=slot_swap_rate,
            delta=float(slot_swap_rate - before["slot_swap_rate"]),
        ),
        "checkpoint_roundtrip": _check(
            "checkpoint_roundtrip",
            metric="checkpoint_roundtrip_ok",
            passed=bool(roundtrip_ok),
            before=1.0,
            after=1.0 if roundtrip_ok else 0.0,
            delta=0.0 if roundtrip_ok else -1.0,
        ),
        "duration_within_limit": _check(
            "duration_within_limit",
            metric="observed_duration_seconds",
            passed=float(observed_duration_seconds) <= duration_limit,
            before=duration_limit,
            after=float(observed_duration_seconds),
            delta=float(observed_duration_seconds) - duration_limit,
        ),
    }


def _check(
    name: str,
    *,
    metric: str,
    passed: bool,
    before: float,
    after: float,
    delta: float,
) -> dict[str, Any]:
    return {
        "name": name,
        "metric": metric,
        "status": "pass" if bool(passed) else "fail",
        "passed": bool(passed),
        "before": float(before),
        "after": float(after),
        "delta": float(delta),
    }


def _checkpoint_roundtrip(
    reference: Mapping[str, Any],
    restored: Mapping[str, Any],
) -> dict[str, Any]:
    before = _solver_metrics(reference)
    after = _solver_metrics(restored)
    deltas = {key: abs(float(before[key]) - float(after[key])) for key in _METRIC_KEYS}
    tolerance = 1e-5
    return {
        "roundtrip_ok": all(delta <= tolerance for delta in deltas.values()),
        "metric_deltas": deltas,
        "tolerance": tolerance,
    }


def _validate_report_ladder(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> None:
    if len(scenarios) != 15:
        raise ValueError("assignment long smoke requires the 15 report scenarios")
    expected = set(objectstate_model_identity_benchmark_report_difficulty_by_scenario())
    scenario_ids = {str(scenario.scenario_id) for scenario in scenarios}
    if scenario_ids != expected:
        raise ValueError("assignment long smoke scenarios must match the report ladder")
    perturbations = {str(scenario.perturbation_kind) for scenario in scenarios}
    if perturbations != set(OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS):
        raise ValueError("assignment long smoke scenarios must cover all perturbations")
    for scenario in scenarios:
        _feature_matrix(
            scenario.frame0_features,
            rows=scenario.frame0_cloud.count,
            label=f"{scenario.scenario_id}.frame0_features",
        )
        _feature_matrix(
            scenario.frame1_features,
            rows=scenario.frame1_cloud.count,
            label=f"{scenario.scenario_id}.frame1_features",
        )


def _feature_matrix(value: np.ndarray | None, *, rows: int, label: str) -> np.ndarray:
    if value is None:
        raise ValueError(f"assignment long smoke requires {label}")
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"assignment long smoke {label} must be 2D")
    if array.shape[0] != rows:
        raise ValueError(f"assignment long smoke {label} row count mismatch")
    if array.shape[1] < 1:
        raise ValueError(f"assignment long smoke {label} requires feature columns")
    if not np.isfinite(array).all():
        raise ValueError(f"assignment long smoke {label} must be finite")
    return array.astype(np.float32, copy=False)


def _validate_benchmark_digest(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload.get("schema"), str) or not payload["schema"]:
        raise ValueError("assignment long smoke benchmark digest requires schema")
    _validate_metric_digest(_mapping(payload, "metrics"))
    if not isinstance(payload.get("summary_path"), str) or not payload["summary_path"]:
        raise ValueError("assignment long smoke benchmark digest requires summary_path")


def _validate_metric_digest(payload: Mapping[str, Any]) -> None:
    for key in _METRIC_KEYS:
        _finite(payload.get(key), key)


def _validate_gap(payload: Mapping[str, Any]) -> None:
    for key in _GAP_KEYS:
        _finite(payload.get(key), f"generalization_gap.{key}")


def _validate_success_check(payload: Mapping[str, Any], name: str) -> None:
    if payload.get("name") != name:
        raise ValueError(f"assignment long smoke success check {name} has wrong name")
    if payload.get("status") not in {"pass", "fail"}:
        raise ValueError(f"assignment long smoke success check {name} status unsupported")
    if not isinstance(payload.get("passed"), bool):
        raise ValueError(
            f"assignment long smoke success check {name} passed must be bool"
        )
    if (payload["status"] == "pass") is not bool(payload["passed"]):
        raise ValueError(f"assignment long smoke success check {name} status mismatch")
    if not isinstance(payload.get("metric"), str) or not payload["metric"]:
        raise ValueError(
            f"assignment long smoke success check {name} requires metric"
        )
    for metric_key in ("before", "after", "delta"):
        _finite(payload.get(metric_key), f"success_checks.{name}.{metric_key}")


def _max_gap(payload: Mapping[str, float]) -> float:
    return max(abs(float(payload[key])) for key in _GAP_KEYS)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"assignment long smoke requires {key}")
    return value


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"assignment long smoke {label} must be a positive integer")
    return int(value)


def _positive_number(value: Any, label: str) -> float:
    number = _finite(value, label)
    if number <= 0.0:
        raise ValueError(f"assignment long smoke {label} must be > 0")
    return number


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"assignment long smoke {label} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"assignment long smoke {label} must be finite")
    return number
