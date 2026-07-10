from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.assignment_evidence import AssignmentEvidenceBatch
from objgauss.core.assignment_solver_v2 import (
    ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
    assignment_solver_v2_state_from_dict,
    predict_assignment_solver_v2,
)
from objgauss.core.features import positions
from objgauss.pipelines.objectstate_assignment_long_smoke import (
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA,
    validate_objectstate_assignment_long_smoke_summary,
)
from objgauss.evaluation.objectstate_model_identity_benchmark import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS,
    ObjectStateModelIdentityBenchmarkScenario,
    objectstate_model_identity_benchmark_summary,
    validate_objectstate_model_identity_benchmark_summary,
)
from objgauss.evaluation.objectstate_model_identity_benchmark_report import (
    objectstate_model_identity_benchmark_report_scenarios,
)
from objgauss.pipelines.objectstate_temporal_assignment_contract import (
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA,
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_METRICS,
    objectstate_temporal_assignment_contract_summary,
    validate_objectstate_temporal_assignment_contract_summary,
)

OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA = (
    "objgauss-objectstate-temporal-assignment-v1"
)
_IDENTITY_METRICS = (
    "identity_retrieval_at_1",
    "identity_margin",
    "slot_swap_rate",
    "objectstate_drift",
    "assignment_consistency",
    "occlusion_recovery",
)
_SUCCESS_CHECKS = (
    "temporal_assignment_consistency",
    "identity_retrieval_at_1",
    "identity_margin",
    "slot_swap_rate",
    "occlusion_recovery",
    "track_fragmentation_rate",
    "checkpoint_roundtrip",
)
_CLAIM_POLICY_KEYS = (
    "semantic_policy_only",
    "uses_passed_assignment_long_smoke",
    "uses_ready_temporal_assignment_contract",
    "assignment_matrix_is_single_source_of_truth",
    "physical_identity_labels_are_evaluation_only",
    "writes_slot_match_manifest",
    "does_not_enable_solver_temporal_policy",
    "does_not_claim_real_data_identity_pass",
    "does_not_claim_world_model",
)
_NON_GOAL_KEYS = (
    "runs_temporal_training",
    "enables_assignment_solver_temporal_policy",
    "uses_renderer_loss",
    "uses_dynamics",
    "uses_diffusion",
    "uses_replay_buffer",
    "uses_gpu",
    "downloads_teacher_weights",
    "ingests_real_capture",
    "mutates_viewer_defaults",
)


def objectstate_temporal_assignment_summary(
    output_dir: str | Path,
    *,
    assignment_long_smoke_summary: Mapping[str, Any],
    temporal_assignment_contract: Mapping[str, Any] | None = None,
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario] | None = None,
    sample_id: str = "objectstate-temporal-assignment-001",
    seed: int = 0,
) -> dict[str, Any]:
    long_smoke = validate_objectstate_assignment_long_smoke_summary(
        assignment_long_smoke_summary
    )
    contract = (
        objectstate_temporal_assignment_contract_summary(
            sample_id=f"{sample_id}:contract",
            assignment_long_smoke_summary=long_smoke,
        )
        if temporal_assignment_contract is None
        else validate_objectstate_temporal_assignment_contract_summary(
            temporal_assignment_contract
        )
    )
    if contract["readiness_gate"]["temporal_assignment_contract_ready"] is not True:
        raise ValueError("temporal assignment requires ready temporal assignment contract")
    if long_smoke["status"] != "objectstate_assignment_long_smoke_pass":
        raise ValueError("temporal assignment requires passed assignment long smoke")

    scenario_list = tuple(
        scenarios or objectstate_model_identity_benchmark_report_scenarios()
    )
    _validate_scenarios(scenario_list)
    checkpoint_path = Path(long_smoke["artifact_refs"]["checkpoint"])
    if not checkpoint_path.exists():
        raise ValueError("temporal assignment requires existing long-smoke checkpoint")
    state = assignment_solver_v2_state_from_dict(
        json.loads(checkpoint_path.read_text(encoding="utf-8"))
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = output_root / "temporal-assignment-artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)

    benchmark = validate_objectstate_model_identity_benchmark_summary(
        objectstate_model_identity_benchmark_summary(
            scenario_list,
            state,
            output_dir=artifact_root / "identity-benchmark",
            sample_id=f"{sample_id}:identity-benchmark",
            evidence_policy="semantic",
            evidence_policy_source="temporal_assignment_long_smoke_checkpoint",
            native_gaussian_evidence_only=False,
            uses_semantic_evidence=True,
            seed=int(seed) + 17,
        )
    )
    slot_manifest = _slot_match_manifest(scenario_list, state)
    temporal_metrics = _temporal_metrics(slot_manifest)
    identity_metrics = _identity_metrics(benchmark)
    metric_summary = {
        **temporal_metrics,
        "identity_retrieval_at_1": identity_metrics["identity_retrieval_at_1"],
        "identity_margin": identity_metrics["identity_margin"],
        "slot_swap_rate": temporal_metrics["slot_swap_rate"],
        "occlusion_recovery": identity_metrics["occlusion_recovery"],
        "checkpoint_roundtrip": bool(long_smoke["checkpoint"]["roundtrip_ok"]),
    }
    success = _success_checks(
        metric_summary,
        baseline_metrics=long_smoke["metrics"]["held_out_after"],
        contract=contract,
    )
    status = (
        "objectstate_temporal_assignment_pass"
        if all(check["passed"] for check in success.values())
        else "objectstate_temporal_assignment_reviewable"
    )
    summary_path = output_root / "temporal-assignment-summary.json"
    slot_manifest_path = artifact_root / "temporal-slot-match-manifest.json"
    slot_manifest_path.write_text(
        json.dumps(slot_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    payload = {
        "schema": OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA,
        "kind": "objectstate_temporal_assignment",
        "status": status,
        "sample_id": str(sample_id),
        "contract_schema": OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA,
        "long_smoke_schema": OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA,
        "solver_state_schema": ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
        "contract": contract,
        "long_smoke": _long_smoke_digest(long_smoke),
        "run_config": {
            "evidence_policy": "semantic",
            "seed": int(seed),
            "scenario_count": int(len(scenario_list)),
            "temporal_policy": state.config.temporal_policy,
            "matching_policy": state.config.matching_policy,
            "training": "not_run",
        },
        "identity_benchmark": _benchmark_digest(benchmark),
        "slot_match_manifest": {
            "path": str(slot_manifest_path),
            "scenario_count": int(slot_manifest["scenario_count"]),
            "identity_pair_count": int(slot_manifest["identity_pair_count"]),
        },
        "metrics": metric_summary,
        "success_checks": success,
        "next_stage_gate": {
            "controlled_capture_allowed": status == "objectstate_temporal_assignment_pass",
            "status": (
                "pass" if status == "objectstate_temporal_assignment_pass" else "reviewable"
            ),
            "blocked_reasons": [
                name for name, check in success.items() if not bool(check["passed"])
            ],
            "next_recommended_pr": (
                "OBJECTSTATE-CONTROLLED-CAPTURE-001"
                if status == "objectstate_temporal_assignment_pass"
                else None
            ),
        },
        "artifact_refs": {
            "summary": str(summary_path),
            "artifact_root": str(artifact_root),
            "slot_match_manifest": str(slot_manifest_path),
            "identity_benchmark_summary": benchmark["summary_path"],
            "long_smoke_summary": long_smoke.get("summary_path"),
            "long_smoke_checkpoint": str(checkpoint_path),
        },
        "claim_policy": {key: True for key in _CLAIM_POLICY_KEYS},
        "non_goals": {key: False for key in _NON_GOAL_KEYS},
    }
    checked = validate_objectstate_temporal_assignment_summary(payload)
    summary_path.write_text(
        json.dumps(checked, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checked["summary_path"] = str(summary_path)
    return validate_objectstate_temporal_assignment_summary(checked)


def validate_objectstate_temporal_assignment_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("temporal assignment summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA:
        raise ValueError(f"unsupported temporal assignment schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_temporal_assignment":
        raise ValueError("temporal assignment kind is unsupported")
    if payload.get("status") not in {
        "objectstate_temporal_assignment_pass",
        "objectstate_temporal_assignment_reviewable",
    }:
        raise ValueError("temporal assignment status is unsupported")
    if payload.get("contract_schema") != OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA:
        raise ValueError("temporal assignment contract schema mismatch")
    if payload.get("long_smoke_schema") != OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA:
        raise ValueError("temporal assignment long-smoke schema mismatch")
    validate_objectstate_temporal_assignment_contract_summary(_mapping(payload, "contract"))
    long_smoke = _mapping(payload, "long_smoke")
    if long_smoke.get("status") != "objectstate_assignment_long_smoke_pass":
        raise ValueError("temporal assignment requires passed long smoke")
    config = _mapping(payload, "run_config")
    if config.get("evidence_policy") != "semantic":
        raise ValueError("temporal assignment must use semantic evidence policy")
    if config.get("training") != "not_run":
        raise ValueError("temporal assignment smoke must not run temporal training")
    if config.get("temporal_policy") != "disabled":
        raise ValueError("temporal assignment smoke must keep solver temporal policy disabled")
    benchmark = _mapping(payload, "identity_benchmark")
    _validate_benchmark_digest(benchmark)
    slot_manifest = _mapping(payload, "slot_match_manifest")
    if int(slot_manifest.get("scenario_count", 0)) < 1:
        raise ValueError("temporal assignment slot manifest requires scenarios")
    if int(slot_manifest.get("identity_pair_count", 0)) < 1:
        raise ValueError("temporal assignment slot manifest requires identity pairs")
    metrics = _mapping(payload, "metrics")
    for metric in OBJECTSTATE_TEMPORAL_ASSIGNMENT_METRICS:
        if metric == "checkpoint_roundtrip":
            if metrics.get(metric) is not True:
                raise ValueError("temporal assignment checkpoint roundtrip must be true")
        else:
            _finite(metrics.get(metric), f"metrics.{metric}")
    success = _mapping(payload, "success_checks")
    if set(success) != set(_SUCCESS_CHECKS):
        raise ValueError("temporal assignment success checks mismatch")
    for name in _SUCCESS_CHECKS:
        _validate_success_check(_mapping(success, name), name)
    all_pass = all(bool(check["passed"]) for check in success.values())
    if payload["status"] == "objectstate_temporal_assignment_pass" and not all_pass:
        raise ValueError("temporal assignment pass requires all checks to pass")
    if payload["status"] == "objectstate_temporal_assignment_reviewable" and all_pass:
        raise ValueError("temporal assignment reviewable contradicts passing checks")
    gate = _mapping(payload, "next_stage_gate")
    if gate.get("controlled_capture_allowed") is not all_pass:
        raise ValueError("temporal assignment next stage gate contradicts checks")
    artifacts = _mapping(payload, "artifact_refs")
    for key in (
        "summary",
        "artifact_root",
        "slot_match_manifest",
        "identity_benchmark_summary",
        "long_smoke_checkpoint",
    ):
        if not isinstance(artifacts.get(key), str) or not artifacts[key]:
            raise ValueError(f"temporal assignment missing artifact ref {key}")
    claim_policy = _mapping(payload, "claim_policy")
    if any(not bool(claim_policy.get(key)) for key in _CLAIM_POLICY_KEYS):
        raise ValueError("temporal assignment must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(non_goals.get(key)) for key in _NON_GOAL_KEYS):
        raise ValueError("temporal assignment cannot claim non-goals")
    if "summary_path" in payload and not isinstance(payload["summary_path"], str):
        raise ValueError("temporal assignment summary_path must be a string")
    return dict(payload)


def _slot_match_manifest(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
    state,
) -> dict[str, Any]:
    rows = []
    identity_pair_count = 0
    consistent_count = 0
    confidence_values = []
    for scenario in scenarios:
        pred0 = predict_assignment_solver_v2(
            _evidence_batch(
                scenario.frame0_cloud,
                scenario.frame0_features,
                frame_index=0,
                source=f"{scenario.scenario_id}:t0",
            ),
            state,
        )
        pred1 = predict_assignment_solver_v2(
            _evidence_batch(
                scenario.frame1_cloud,
                scenario.frame1_features,
                frame_index=1,
                source=f"{scenario.scenario_id}:t1",
            ),
            state,
        )
        hard0 = np.argmax(pred0.assignment, axis=1)
        hard1 = np.argmax(pred1.assignment, axis=1)
        labels0 = np.asarray(scenario.frame0_identity_labels, dtype=np.int64)
        labels1 = np.asarray(scenario.frame1_identity_labels, dtype=np.int64)
        identities = sorted(set(labels0.tolist()).intersection(set(labels1.tolist())))
        matches = []
        for identity in identities:
            slot0 = _majority_slot(hard0[labels0 == identity])
            slot1 = _majority_slot(hard1[labels1 == identity])
            consistent = slot0 == slot1
            identity_pair_count += 1
            consistent_count += int(consistent)
            matches.append(
                {
                    "physical_identity": int(identity),
                    "frame0_slot": int(slot0),
                    "frame1_slot": int(slot1),
                    "consistent": bool(consistent),
                }
            )
        confidence_values.extend(np.asarray(pred0.confidence, dtype=np.float32).tolist())
        confidence_values.extend(np.asarray(pred1.confidence, dtype=np.float32).tolist())
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "perturbation_kind": scenario.perturbation_kind,
                "identity_matches": matches,
                "temporal_assignment_consistency": (
                    float(sum(1 for item in matches if item["consistent"]) / len(matches))
                    if matches
                    else 0.0
                ),
            }
        )
    return {
        "schema": "objgauss-objectstate-temporal-slot-match-manifest-v1",
        "scenario_count": int(len(rows)),
        "identity_pair_count": int(identity_pair_count),
        "consistent_pair_count": int(consistent_count),
        "mean_assignment_confidence": float(np.mean(confidence_values)) if confidence_values else 0.0,
        "scenario_results": rows,
        "claim_policy": {
            "physical_identity_labels_are_evaluation_only": True,
            "hard_slots_are_derived_from_assignment": True,
        },
    }


def _temporal_metrics(manifest: Mapping[str, Any]) -> dict[str, float]:
    pairs = int(manifest["identity_pair_count"])
    consistent = int(manifest["consistent_pair_count"])
    consistency = float(consistent / pairs) if pairs else 0.0
    return {
        "temporal_assignment_consistency": consistency,
        "track_fragmentation_rate": float(1.0 - consistency),
        "slot_swap_rate": float(1.0 - consistency),
        "mean_assignment_confidence": float(manifest["mean_assignment_confidence"]),
    }


def _success_checks(
    metrics: Mapping[str, Any],
    *,
    baseline_metrics: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    thresholds = contract["success_metrics"]
    occlusion_delta = float(metrics["occlusion_recovery"]) - float(
        baseline_metrics["occlusion_recovery"]
    )
    return {
        "temporal_assignment_consistency": _check(
            "temporal_assignment_consistency",
            metric="temporal_assignment_consistency",
            passed=float(metrics["temporal_assignment_consistency"])
            >= float(thresholds["temporal_assignment_consistency"]["threshold"]),
            observed=float(metrics["temporal_assignment_consistency"]),
            threshold=float(thresholds["temporal_assignment_consistency"]["threshold"]),
        ),
        "identity_retrieval_at_1": _check(
            "identity_retrieval_at_1",
            metric="identity_retrieval_at_1",
            passed=float(metrics["identity_retrieval_at_1"])
            >= float(thresholds["identity_retrieval_at_1"]["threshold"]),
            observed=float(metrics["identity_retrieval_at_1"]),
            threshold=float(thresholds["identity_retrieval_at_1"]["threshold"]),
        ),
        "identity_margin": _check(
            "identity_margin",
            metric="identity_margin",
            passed=float(metrics["identity_margin"])
            > float(thresholds["identity_margin"]["threshold"]),
            observed=float(metrics["identity_margin"]),
            threshold=float(thresholds["identity_margin"]["threshold"]),
        ),
        "slot_swap_rate": _check(
            "slot_swap_rate",
            metric="slot_swap_rate",
            passed=float(metrics["slot_swap_rate"])
            <= float(thresholds["slot_swap_rate"]["threshold"]),
            observed=float(metrics["slot_swap_rate"]),
            threshold=float(thresholds["slot_swap_rate"]["threshold"]),
        ),
        "occlusion_recovery": _check(
            "occlusion_recovery",
            metric="occlusion_recovery",
            passed=occlusion_delta
            >= -float(thresholds["occlusion_recovery"]["threshold"]),
            observed=float(metrics["occlusion_recovery"]),
            threshold=float(thresholds["occlusion_recovery"]["threshold"]),
            delta=occlusion_delta,
        ),
        "track_fragmentation_rate": _check(
            "track_fragmentation_rate",
            metric="track_fragmentation_rate",
            passed=float(metrics["track_fragmentation_rate"])
            <= float(thresholds["track_fragmentation_rate"]["threshold"]),
            observed=float(metrics["track_fragmentation_rate"]),
            threshold=float(thresholds["track_fragmentation_rate"]["threshold"]),
        ),
        "checkpoint_roundtrip": {
            "name": "checkpoint_roundtrip",
            "metric": "checkpoint_roundtrip",
            "status": "pass" if bool(metrics["checkpoint_roundtrip"]) else "fail",
            "passed": bool(metrics["checkpoint_roundtrip"]),
            "observed": bool(metrics["checkpoint_roundtrip"]),
            "threshold": True,
        },
    }


def _check(
    name: str,
    *,
    metric: str,
    passed: bool,
    observed: float,
    threshold: float,
    delta: float | None = None,
) -> dict[str, Any]:
    payload = {
        "name": name,
        "metric": metric,
        "status": "pass" if bool(passed) else "fail",
        "passed": bool(passed),
        "observed": float(observed),
        "threshold": float(threshold),
    }
    if delta is not None:
        payload["delta"] = float(delta)
    return payload


def _evidence_batch(
    cloud,
    features: np.ndarray | None,
    *,
    frame_index: int,
    source: str,
) -> AssignmentEvidenceBatch:
    feature_matrix = _feature_matrix(features, rows=cloud.count, label=source)
    return AssignmentEvidenceBatch(
        positions=positions(cloud),
        features=feature_matrix,
        frame_index=int(frame_index),
        source=source,
    )


def _identity_metrics(benchmark: Mapping[str, Any]) -> dict[str, float]:
    metrics = benchmark["baselines"]["assignment_solver_v2"]["metrics"]
    return {key: float(metrics[key]) for key in _IDENTITY_METRICS}


def _benchmark_digest(benchmark: Mapping[str, Any]) -> dict[str, Any]:
    checked = validate_objectstate_model_identity_benchmark_summary(benchmark)
    return {
        "schema": checked["schema"],
        "status": checked["status"],
        "sample_id": checked["sample_id"],
        "num_scenarios": int(checked["num_scenarios"]),
        "num_pairs": int(checked["num_pairs"]),
        "metrics": _identity_metrics(checked),
        "summary_path": checked["summary_path"],
    }


def _long_smoke_digest(long_smoke: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": long_smoke["schema"],
        "status": long_smoke["status"],
        "sample_id": long_smoke["sample_id"],
        "summary_path": long_smoke.get("summary_path"),
        "checkpoint_roundtrip_ok": bool(long_smoke["checkpoint"]["roundtrip_ok"]),
        "temporal_assignment_contract_allowed": bool(
            long_smoke["next_stage_gate"]["temporal_assignment_contract_allowed"]
        ),
    }


def _validate_scenarios(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> None:
    if len(scenarios) != 15:
        raise ValueError("temporal assignment requires the 15 report scenarios")
    perturbations = {str(scenario.perturbation_kind) for scenario in scenarios}
    if perturbations != set(OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS):
        raise ValueError("temporal assignment scenarios must cover all perturbations")
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
        raise ValueError(f"temporal assignment requires {label}")
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"temporal assignment {label} must be 2D")
    if array.shape[0] != rows:
        raise ValueError(f"temporal assignment {label} row count mismatch")
    if array.shape[1] < 1:
        raise ValueError(f"temporal assignment {label} requires feature columns")
    if not np.isfinite(array).all():
        raise ValueError(f"temporal assignment {label} must be finite")
    return array.astype(np.float32, copy=False)


def _majority_slot(values: np.ndarray) -> int:
    if values.size == 0:
        raise ValueError("temporal assignment cannot match empty identity slot")
    counts = np.bincount(np.asarray(values, dtype=np.int64))
    return int(np.argmax(counts))


def _validate_benchmark_digest(payload: Mapping[str, Any]) -> None:
    if not isinstance(payload.get("summary_path"), str) or not payload["summary_path"]:
        raise ValueError("temporal assignment benchmark digest requires summary_path")
    metrics = _mapping(payload, "metrics")
    for key in _IDENTITY_METRICS:
        _finite(metrics.get(key), f"identity_benchmark.{key}")


def _validate_success_check(payload: Mapping[str, Any], name: str) -> None:
    if payload.get("name") != name:
        raise ValueError(f"temporal assignment success check {name} has wrong name")
    if payload.get("status") not in {"pass", "fail"}:
        raise ValueError(f"temporal assignment success check {name} status unsupported")
    if not isinstance(payload.get("passed"), bool):
        raise ValueError(f"temporal assignment success check {name} passed must be bool")
    if name == "checkpoint_roundtrip":
        if not isinstance(payload.get("observed"), bool) or payload.get("threshold") is not True:
            raise ValueError("temporal assignment checkpoint success check malformed")
        return
    _finite(payload.get("observed"), f"success_checks.{name}.observed")
    _finite(payload.get("threshold"), f"success_checks.{name}.threshold")


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"temporal assignment requires {key}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"temporal assignment {label} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"temporal assignment {label} must be finite")
    return number


__all__ = (
    "OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA",
    "objectstate_temporal_assignment_summary",
    "validate_objectstate_temporal_assignment_summary",
)
