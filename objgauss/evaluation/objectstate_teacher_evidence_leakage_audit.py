from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
)
from objgauss.evaluation.objectstate_model_identity_benchmark import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS,
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA,
    ObjectStateModelIdentityBenchmarkScenario,
    objectstate_model_identity_benchmark_summary,
    validate_objectstate_model_identity_benchmark_summary,
)
from objgauss.evaluation.objectstate_model_identity_benchmark_report import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES,
    objectstate_model_identity_benchmark_report_difficulty_by_scenario,
    objectstate_model_identity_benchmark_report_scenarios,
)
from objgauss.datasets.objectstate_teacher_evidence import (
    TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES,
    TeacherEvidenceBatch,
    teacher_evidence_batch_summary,
)

OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA = (
    "objgauss-objectstate-teacher-evidence-leakage-audit-v1"
)
TEACHER_EVIDENCE_LEAKAGE_AUDIT_CHECKS = (
    "physical_label_ban",
    "semantic_feature_shuffle",
    "random_semantic_baseline",
    "train_test_semantic_source_split",
)
_CLAIM_POLICY_KEYS = (
    "audits_teacher_evidence_not_ground_truth_identity",
    "uses_identity_benchmark_report_ladder",
    "semantic_shuffle_is_negative_path",
    "random_semantic_is_negative_baseline",
    "training_requires_inference_time_source_split",
    "does_not_claim_long_training_ready",
    "does_not_claim_real_data_identity_pass",
    "does_not_claim_world_model",
)
_NON_GOAL_KEYS = (
    "runs_teacher_model",
    "downloads_teacher_weights",
    "trains_model",
    "runs_long_smoke",
    "uses_renderer_loss",
    "uses_temporal_loss",
    "ingests_real_capture",
    "mutates_viewer_defaults",
)
_METRIC_KEYS = (
    "identity_retrieval_at_1",
    "identity_margin",
    "slot_swap_rate",
    "objectstate_drift",
    "assignment_consistency",
    "occlusion_recovery",
)


@dataclass(frozen=True)
class TeacherEvidenceLeakageAuditThresholds:
    semantic_shuffle_retrieval_drop_min: float = 0.20
    semantic_shuffle_margin_drop_min: float = 0.01
    random_semantic_lift_vs_reference_max: float = 0.20

    def as_dict(self) -> dict[str, float]:
        payload = {
            "semantic_shuffle_retrieval_drop_min": float(
                self.semantic_shuffle_retrieval_drop_min
            ),
            "semantic_shuffle_margin_drop_min": float(
                self.semantic_shuffle_margin_drop_min
            ),
            "random_semantic_lift_vs_reference_max": float(
                self.random_semantic_lift_vs_reference_max
            ),
        }
        for key, value in payload.items():
            if value < 0.0 or not np.isfinite(value):
                raise ValueError(f"{key} must be finite and >= 0")
        return payload


def objectstate_teacher_evidence_leakage_audit_summary(
    output_dir: str | Path,
    *,
    artifact_dir: str | Path | None = None,
    sample_id: str = "objectstate-teacher-evidence-leakage-audit-001",
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario] | None = None,
    teacher_batches: Sequence[TeacherEvidenceBatch] | None = None,
    thresholds: TeacherEvidenceLeakageAuditThresholds | None = None,
    seed: int = 0,
) -> dict[str, Any]:
    scenario_list = tuple(scenarios or objectstate_model_identity_benchmark_report_scenarios())
    _validate_report_ladder(scenario_list)
    batch_list = tuple(teacher_batches) if teacher_batches is not None else (
        _default_synthetic_teacher_batch(scenario_list),
    )
    audited_scenarios, audited_evidence = _bind_teacher_evidence(
        scenario_list,
        batch_list,
    )
    threshold_payload = (thresholds or TeacherEvidenceLeakageAuditThresholds()).as_dict()

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    artifact_root = (
        Path(artifact_dir)
        if artifact_dir is not None
        else output_root / "teacher-evidence-leakage-audit-artifacts"
    )
    artifact_root.mkdir(parents=True, exist_ok=True)

    baseline = _semantic_benchmark(
        audited_scenarios,
        artifact_root / "semantic-reference",
        sample_id=f"{sample_id}:semantic-reference",
        evidence_policy="semantic_reference",
        evidence_policy_source="teacher_evidence_leakage_audit_reference",
        seed=seed,
    )
    shuffled = _semantic_benchmark(
        _shuffle_frame1_semantic_features(audited_scenarios),
        artifact_root / "semantic-feature-shuffle",
        sample_id=f"{sample_id}:semantic-feature-shuffle",
        evidence_policy="semantic_feature_shuffle",
        evidence_policy_source="teacher_evidence_leakage_audit_negative_path",
        seed=seed + 101,
    )
    random_semantic = _semantic_benchmark(
        _random_semantic_features(audited_scenarios, seed=seed + 202),
        artifact_root / "random-semantic-baseline",
        sample_id=f"{sample_id}:random-semantic-baseline",
        evidence_policy="random_semantic_baseline",
        evidence_policy_source="teacher_evidence_leakage_audit_negative_path",
        seed=seed + 303,
    )

    batch_reports = [_batch_report(batch) for batch in batch_list]
    checks = {
        "physical_label_ban": _physical_label_ban_check(
            batch_reports,
            audited_scenarios,
        ),
        "semantic_feature_shuffle": _semantic_shuffle_check(
            baseline,
            shuffled,
            thresholds=threshold_payload,
        ),
        "random_semantic_baseline": _random_semantic_check(
            random_semantic,
            thresholds=threshold_payload,
        ),
        "train_test_semantic_source_split": _train_test_source_split_check(batch_reports),
    }
    blocked = [name for name, check in checks.items() if not bool(check["passed"])]
    summary_path = output_root / "teacher-evidence-leakage-audit-summary.json"
    payload = {
        "schema": OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA,
        "kind": "objectstate_teacher_evidence_leakage_audit",
        "status": (
            "objectstate_teacher_evidence_leakage_audit_pass"
            if not blocked
            else "objectstate_teacher_evidence_leakage_audit_blocked"
        ),
        "sample_id": str(sample_id),
        "benchmark_schema": OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA,
        "scenario_count": int(len(scenario_list)),
        "identity_pair_count": int(baseline["num_pairs"]),
        "difficulty_levels": list(OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES),
        "perturbation_kinds": list(OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS),
        "thresholds": threshold_payload,
        "teacher_evidence_batches": batch_reports,
        "audited_evidence": audited_evidence,
        "benchmark_metrics": {
            "semantic_reference": _benchmark_metrics(baseline),
            "semantic_feature_shuffle": _benchmark_metrics(shuffled),
            "random_semantic_baseline": _benchmark_metrics(random_semantic),
        },
        "audit_checks": checks,
        "training_gate": {
            "semantic_teacher_evidence_training_allowed": not blocked,
            "status": "cleared" if not blocked else "blocked",
            "blocked_checks": blocked,
            "next_allowed_pr": (
                "OBJECTSTATE-ASSIGNMENT-LONG-SMOKE-CONTRACT-001"
                if not blocked
                else None
            ),
        },
        "artifact_refs": {
            "teacher_evidence_leakage_audit_summary": str(summary_path),
            "teacher_evidence_leakage_audit_artifacts": str(artifact_root),
            "semantic_reference_summary": baseline["summary_path"],
            "semantic_feature_shuffle_summary": shuffled["summary_path"],
            "random_semantic_baseline_summary": random_semantic["summary_path"],
        },
        "claim_policy": {key: True for key in _CLAIM_POLICY_KEYS},
        "non_goals": {key: False for key in _NON_GOAL_KEYS},
    }
    checked = validate_objectstate_teacher_evidence_leakage_audit_summary(payload)
    summary_path.write_text(
        json.dumps(checked, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    checked["summary_path"] = str(summary_path)
    return validate_objectstate_teacher_evidence_leakage_audit_summary(checked)


def teacher_evidence_scenarios_from_audit(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
    teacher_batches: Sequence[TeacherEvidenceBatch],
    leakage_audit: Mapping[str, Any],
) -> tuple[
    tuple[ObjectStateModelIdentityBenchmarkScenario, ...],
    dict[str, Any],
]:
    """Bind the exact feature content cleared by a leakage audit.

    The audit summary intentionally does not inline feature matrices.  A caller
    that wants to train or evaluate must therefore supply the matrices again;
    this function rejects any content, ordering, source, or sample-id change.
    """

    checked = validate_objectstate_teacher_evidence_leakage_audit_summary(
        leakage_audit
    )
    if checked["status"] != "objectstate_teacher_evidence_leakage_audit_pass":
        raise ValueError("teacher evidence requires a passed leakage audit")
    batch_list = tuple(teacher_batches)
    bound, digest = _bind_teacher_evidence(tuple(scenarios), batch_list)
    batch_reports = [_batch_report(batch) for batch in batch_list]
    if (
        digest != checked["audited_evidence"]
        or _content_sha256({"batch_reports": batch_reports})
        != _content_sha256(
            {"batch_reports": checked["teacher_evidence_batches"]}
        )
    ):
        raise ValueError(
            "teacher evidence content does not match the passed leakage audit"
        )
    return bound, digest


def validate_objectstate_teacher_evidence_leakage_audit_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("teacher evidence leakage audit summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA:
        raise ValueError(
            "unsupported teacher evidence leakage audit schema: "
            f"{payload.get('schema')}"
        )
    if payload.get("kind") != "objectstate_teacher_evidence_leakage_audit":
        raise ValueError("teacher evidence leakage audit kind is unsupported")
    if payload.get("status") not in {
        "objectstate_teacher_evidence_leakage_audit_pass",
        "objectstate_teacher_evidence_leakage_audit_blocked",
    }:
        raise ValueError("teacher evidence leakage audit status is unsupported")
    if payload.get("benchmark_schema") != OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA:
        raise ValueError("teacher evidence leakage audit must reference benchmark schema")
    if int(payload.get("scenario_count", 0)) != 15:
        raise ValueError("teacher evidence leakage audit requires the 15-scenario ladder")
    if int(payload.get("identity_pair_count", 0)) < 1:
        raise ValueError("teacher evidence leakage audit requires identity pairs")
    if tuple(payload.get("difficulty_levels", ())) != (
        OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES
    ):
        raise ValueError("teacher evidence leakage audit difficulty levels mismatch")
    if tuple(payload.get("perturbation_kinds", ())) != (
        OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS
    ):
        raise ValueError("teacher evidence leakage audit perturbation kinds mismatch")
    thresholds = _mapping(payload, "thresholds")
    for key in (
        "semantic_shuffle_retrieval_drop_min",
        "semantic_shuffle_margin_drop_min",
        "random_semantic_lift_vs_reference_max",
    ):
        _finite(thresholds.get(key), f"thresholds.{key}")
    batches = payload.get("teacher_evidence_batches")
    if not isinstance(batches, list) or not batches:
        raise ValueError("teacher evidence leakage audit requires batch reports")
    for report in batches:
        _validate_batch_report(report)
    _validate_audited_evidence(
        _mapping(payload, "audited_evidence"),
        expected_scenarios=int(payload["scenario_count"]),
        expected_batches=len(batches),
    )
    metrics = _mapping(payload, "benchmark_metrics")
    for key in ("semantic_reference", "semantic_feature_shuffle", "random_semantic_baseline"):
        _validate_metric_digest(_mapping(metrics, key))
    checks = _mapping(payload, "audit_checks")
    if set(checks) != set(TEACHER_EVIDENCE_LEAKAGE_AUDIT_CHECKS):
        raise ValueError("teacher evidence leakage audit checks mismatch")
    blocked = []
    for name in TEACHER_EVIDENCE_LEAKAGE_AUDIT_CHECKS:
        check = _validate_check(checks[name], name)
        if not bool(check["passed"]):
            blocked.append(name)
    gate = _mapping(payload, "training_gate")
    if gate.get("status") not in {"cleared", "blocked"}:
        raise ValueError("teacher evidence leakage audit training gate status unsupported")
    if gate.get("semantic_teacher_evidence_training_allowed") is not (not blocked):
        raise ValueError("teacher evidence leakage audit training gate contradicts checks")
    if gate.get("blocked_checks") != blocked:
        raise ValueError("teacher evidence leakage audit blocked checks mismatch")
    expected_status = (
        "objectstate_teacher_evidence_leakage_audit_pass"
        if not blocked
        else "objectstate_teacher_evidence_leakage_audit_blocked"
    )
    if payload["status"] != expected_status:
        raise ValueError("teacher evidence leakage audit status contradicts checks")
    artifact_refs = _mapping(payload, "artifact_refs")
    for key in (
        "teacher_evidence_leakage_audit_summary",
        "teacher_evidence_leakage_audit_artifacts",
        "semantic_reference_summary",
        "semantic_feature_shuffle_summary",
        "random_semantic_baseline_summary",
    ):
        if not isinstance(artifact_refs.get(key), str) or not artifact_refs[key]:
            raise ValueError(f"teacher evidence leakage audit missing artifact ref {key}")
    claim_policy = _mapping(payload, "claim_policy")
    if any(not bool(claim_policy.get(key)) for key in _CLAIM_POLICY_KEYS):
        raise ValueError("teacher evidence leakage audit must preserve claim policy")
    non_goals = _mapping(payload, "non_goals")
    if any(bool(non_goals.get(key)) for key in _NON_GOAL_KEYS):
        raise ValueError("teacher evidence leakage audit cannot claim non-goals")
    if "summary_path" in payload and not isinstance(payload["summary_path"], str):
        raise ValueError("teacher evidence leakage audit summary_path must be a string")
    return dict(payload)


def _bind_teacher_evidence(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
    batches: Sequence[TeacherEvidenceBatch],
) -> tuple[
    tuple[ObjectStateModelIdentityBenchmarkScenario, ...],
    dict[str, Any],
]:
    if not batches:
        raise ValueError("teacher evidence leakage audit requires teacher batches")
    raw_matrices = []
    feature_dim = None
    batch_digests = []
    for index, batch in enumerate(batches):
        if not isinstance(batch, TeacherEvidenceBatch):
            raise TypeError("teacher evidence batches must be TeacherEvidenceBatch")
        matrix = _feature_matrix(
            batch.feature_matrix,
            f"teacher_batches[{index}].feature_matrix",
            rows=len(batch.gaussian_ids),
        )
        if feature_dim is None:
            feature_dim = int(matrix.shape[1])
        elif matrix.shape[1] != feature_dim:
            raise ValueError("teacher evidence batches must share feature_dim")
        raw_matrices.append(matrix)
        batch_digests.append(
            {
                "batch_index": index,
                "sample_id": str(batch.sample_id),
                "source": str(batch.source),
                "row_count": int(matrix.shape[0]),
                "feature_dim": int(matrix.shape[1]),
                "feature_sha256": _feature_sha256(matrix),
            }
        )

    concatenated = np.concatenate(raw_matrices, axis=0)
    expected_rows = sum(
        scenario.frame0_cloud.count + scenario.frame1_cloud.count
        for scenario in scenarios
    )
    if concatenated.shape[0] != expected_rows:
        raise ValueError(
            "teacher evidence rows must equal all scenario frame rows "
            f"({concatenated.shape[0]} != {expected_rows})"
        )

    offset = 0
    bound = []
    frame_digests = []
    for scenario in scenarios:
        frame_features = []
        for frame_name, row_count in (
            ("frame0", scenario.frame0_cloud.count),
            ("frame1", scenario.frame1_cloud.count),
        ):
            end = offset + int(row_count)
            matrix = concatenated[offset:end].astype(np.float32, copy=True)
            frame_features.append(matrix)
            frame_digests.append(
                {
                    "scenario_id": str(scenario.scenario_id),
                    "frame": frame_name,
                    "row_start": int(offset),
                    "row_end": int(end),
                    "row_count": int(row_count),
                    "feature_sha256": _feature_sha256(matrix),
                }
            )
            offset = end
        bound.append(
            replace(
                scenario,
                frame0_features=frame_features[0],
                frame1_features=frame_features[1],
            )
        )

    content = {
        "binding_policy": "scenario_order_frame0_then_frame1",
        "scenario_ids": [str(scenario.scenario_id) for scenario in scenarios],
        "scenario_count": len(scenarios),
        "frame_count": len(frame_digests),
        "batch_count": len(batches),
        "row_count": int(concatenated.shape[0]),
        "feature_dim": int(concatenated.shape[1]),
        "batch_digests": batch_digests,
        "frame_digests": frame_digests,
    }
    content["content_sha256"] = _content_sha256(content)
    return tuple(bound), content


def _semantic_benchmark(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
    output_dir: Path,
    *,
    sample_id: str,
    evidence_policy: str,
    evidence_policy_source: str,
    seed: int,
) -> dict[str, Any]:
    summary = objectstate_model_identity_benchmark_summary(
        scenarios,
        _semantic_solver_state(scenarios),
        output_dir=output_dir,
        sample_id=sample_id,
        evidence_policy=evidence_policy,
        evidence_policy_source=evidence_policy_source,
        native_gaussian_evidence_only=False,
        uses_semantic_evidence=True,
        seed=seed,
    )
    return validate_objectstate_model_identity_benchmark_summary(summary)


def _semantic_solver_state(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> AssignmentSolverV2State:
    reference = scenarios[0]
    features = _feature_matrix(
        reference.frame0_features,
        "reference.frame0_features",
        rows=reference.frame0_cloud.count,
    )
    labels = np.asarray(reference.frame0_identity_labels, dtype=np.int64)
    identities = tuple(sorted(int(item) for item in set(labels.tolist())))
    centers = []
    for identity in identities:
        mask = labels == identity
        if not np.any(mask):
            raise ValueError("teacher evidence leakage audit missing identity features")
        centers.append(features[mask].mean(axis=0))
    feature_centers = np.stack(centers).astype(np.float32, copy=False)
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(
            slots=int(len(identities)),
            feature_dim=int(feature_centers.shape[1]),
            temperature=0.15,
            feature_weight=1.0,
            position_weight=0.0,
        ),
        feature_centers=feature_centers,
        position_centers=np.zeros((len(identities), 3), dtype=np.float32),
        slot_bias=np.zeros(len(identities), dtype=np.float32),
        source="teacher_evidence_leakage_audit_semantic_reference",
    )


def _shuffle_frame1_semantic_features(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> tuple[ObjectStateModelIdentityBenchmarkScenario, ...]:
    return tuple(
        replace(
            scenario,
            frame1_features=_object_feature_shift(
                scenario.frame1_features,
                scenario.frame1_identity_labels,
            ),
            description=f"{scenario.description}; audit=semantic_feature_shuffle",
        )
        for scenario in scenarios
    )


def _object_feature_shift(features: np.ndarray | None, labels: np.ndarray) -> np.ndarray:
    matrix = _feature_matrix(features, "semantic_features", rows=len(labels))
    labels_array = np.asarray(labels, dtype=np.int64)
    identities = tuple(sorted(int(item) for item in set(labels_array.tolist())))
    if len(identities) < 2:
        raise ValueError("semantic feature shuffle requires at least two identities")
    shifted = matrix.copy()
    for index, identity in enumerate(identities):
        source_identity = identities[(index + 1) % len(identities)]
        target_indices = np.flatnonzero(labels_array == identity)
        source_indices = np.flatnonzero(labels_array == source_identity)
        if target_indices.size != source_indices.size:
            raise ValueError(
                "semantic feature shuffle requires equal per-identity row counts"
            )
        shifted[target_indices] = matrix[source_indices]
    return shifted.astype(np.float32, copy=False)


def _random_semantic_features(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
    *,
    seed: int,
) -> tuple[ObjectStateModelIdentityBenchmarkScenario, ...]:
    rng = np.random.default_rng(int(seed))
    randomized = []
    for scenario in scenarios:
        feature_dim = int(
            _feature_matrix(
                scenario.frame0_features,
                "frame0_features",
                rows=scenario.frame0_cloud.count,
            ).shape[1]
        )
        randomized.append(
            replace(
                scenario,
                frame0_features=rng.normal(
                    size=(scenario.frame0_cloud.count, feature_dim),
                ).astype(np.float32),
                frame1_features=rng.normal(
                    size=(scenario.frame1_cloud.count, feature_dim),
                ).astype(np.float32),
                description=f"{scenario.description}; audit=random_semantic_baseline",
            )
        )
    return tuple(randomized)


def _default_synthetic_teacher_batch(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> TeacherEvidenceBatch:
    matrices = []
    gaussian_ids = []
    for scenario in scenarios:
        for frame_name, features, row_count in (
            (
                "frame0",
                scenario.frame0_features,
                scenario.frame0_cloud.count,
            ),
            (
                "frame1",
                scenario.frame1_features,
                scenario.frame1_cloud.count,
            ),
        ):
            matrix = _feature_matrix(
                features,
                f"{scenario.scenario_id}.{frame_name}_features",
                rows=row_count,
            )
            matrices.append(matrix)
            gaussian_ids.extend(
                f"{scenario.scenario_id}:{frame_name}:g{index:06d}"
                for index in range(row_count)
            )
    feature_matrix = np.concatenate(matrices, axis=0)
    return TeacherEvidenceBatch(
        sample_id="identity-report-ladder:synthetic-semantic-reference",
        gaussian_ids=tuple(gaussian_ids),
        feature_matrix=feature_matrix,
        source="synthetic_semantic",
        allowed_for_training=False,
        allowed_for_evaluation=True,
        leakage_risk="medium",
        provenance={
            "producer": "objectstate_model_identity_benchmark_report_scenarios",
            "feature_space": "synthetic_report_one_hot_semantic_fixture",
            "input_refs": [
                f"fixture://{scenario.scenario_id}/{frame_name}"
                for scenario in scenarios
                for frame_name in ("frame0", "frame1")
            ],
            "generation_method": "deterministic_controlled_fixture_features",
            "train_test_semantic_source_split": {
                "train_source": "synthetic_report_reference",
                "test_source": "synthetic_report_reference",
                "direct_object_id_embedding_shared": True,
                "policy": "fixture_only_not_training_evidence",
            },
        },
    )


def _batch_report(batch: Any) -> dict[str, Any]:
    sample_id = getattr(batch, "sample_id", "<invalid>")
    source = getattr(batch, "source", "<invalid>")
    try:
        summary = teacher_evidence_batch_summary(batch)
    except (TypeError, ValueError) as exc:
        return {
            "sample_id": str(sample_id),
            "source": str(source),
            "validation_status": "fail",
            "error": str(exc),
            "summary": None,
        }
    matrix = np.asarray(batch.feature_matrix, dtype=np.float32)
    return {
        "sample_id": str(summary["sample_id"]),
        "source": str(summary["source"]),
        "validation_status": "pass",
        "error": None,
        "summary": summary,
        "feature_sha256": _feature_sha256(matrix),
    }


def _physical_label_ban_check(
    batch_reports: Sequence[Mapping[str, Any]],
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> dict[str, Any]:
    failures = [
        report for report in batch_reports if report.get("validation_status") != "pass"
    ]
    content_failures = _direct_one_hot_identity_findings(scenarios)
    reasons = [
        f"{report.get('sample_id')}: {report.get('error')}"
        for report in failures
    ]
    reasons.extend(
        f"{finding['scenario_id']}:{finding['frame']}:"
        "direct_one_hot_identity_embedding"
        for finding in content_failures
    )
    return _check(
        "physical_label_ban",
        passed=not reasons,
        metrics={
            "batch_count": len(batch_reports),
            "failed_batch_count": len(failures),
            "direct_one_hot_identity_embedding_count": len(content_failures),
        },
        reasons=reasons,
        details={"content_leakage_findings": content_failures},
    )


def _direct_one_hot_identity_findings(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> list[dict[str, Any]]:
    findings = []
    for scenario in scenarios:
        for frame_name, features, labels in (
            (
                "frame0",
                scenario.frame0_features,
                scenario.frame0_identity_labels,
            ),
            (
                "frame1",
                scenario.frame1_features,
                scenario.frame1_identity_labels,
            ),
        ):
            matrix = _feature_matrix(
                features,
                f"{scenario.scenario_id}.{frame_name}_features",
                rows=len(labels),
            )
            label_array = np.asarray(labels, dtype=np.int64)
            if _is_direct_one_hot_identity_embedding(matrix, label_array):
                findings.append(
                    {
                        "scenario_id": str(scenario.scenario_id),
                        "frame": frame_name,
                        "feature_dim": int(matrix.shape[1]),
                        "identity_count": int(np.unique(label_array).size),
                        "reason": "one_hot_active_column_is_bijective_with_identity_label",
                    }
                )
    return findings


def _is_direct_one_hot_identity_embedding(
    features: np.ndarray,
    labels: np.ndarray,
) -> bool:
    if features.shape[0] != labels.shape[0] or features.shape[1] < 2:
        return False
    near_zero = np.isclose(features, 0.0, atol=1e-6)
    near_one = np.isclose(features, 1.0, atol=1e-6)
    if not np.all(near_zero | near_one):
        return False
    active = np.sum(near_one, axis=1)
    if not np.all(active == 1):
        return False
    active_column = np.argmax(features, axis=1)
    label_to_column = {}
    for label in np.unique(labels):
        columns = np.unique(active_column[labels == label])
        if columns.size != 1:
            return False
        label_to_column[int(label)] = int(columns[0])
    return len(set(label_to_column.values())) == len(label_to_column)


def _semantic_shuffle_check(
    baseline: Mapping[str, Any],
    shuffled: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    baseline_metrics = _benchmark_metrics(baseline)
    shuffled_metrics = _benchmark_metrics(shuffled)
    retrieval_drop = float(
        baseline_metrics["identity_retrieval_at_1"]
        - shuffled_metrics["identity_retrieval_at_1"]
    )
    margin_drop = float(
        baseline_metrics["identity_margin"] - shuffled_metrics["identity_margin"]
    )
    checks = {
        "retrieval_drop_sufficient": retrieval_drop
        >= float(thresholds["semantic_shuffle_retrieval_drop_min"]),
        "margin_drop_sufficient": margin_drop
        >= float(thresholds["semantic_shuffle_margin_drop_min"]),
    }
    return _check(
        "semantic_feature_shuffle",
        passed=all(checks.values()),
        metrics={
            "semantic_reference_retrieval_at_1": baseline_metrics["identity_retrieval_at_1"],
            "shuffle_retrieval_at_1": shuffled_metrics["identity_retrieval_at_1"],
            "retrieval_drop": retrieval_drop,
            "semantic_reference_identity_margin": baseline_metrics["identity_margin"],
            "shuffle_identity_margin": shuffled_metrics["identity_margin"],
            "margin_drop": margin_drop,
        },
        reasons=[name for name, passed in checks.items() if not passed],
    )


def _random_semantic_check(
    random_semantic: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float],
) -> dict[str, Any]:
    solver = _benchmark_metrics(random_semantic)
    random_assignment = _baseline_metrics(random_semantic, "random_assignment")
    xyz = _baseline_metrics(random_semantic, "xyz_centroid")
    reference = max(
        float(random_assignment["identity_retrieval_at_1"]),
        float(xyz["identity_retrieval_at_1"]),
    )
    lift = float(solver["identity_retrieval_at_1"] - reference)
    passed = lift <= float(thresholds["random_semantic_lift_vs_reference_max"])
    return _check(
        "random_semantic_baseline",
        passed=passed,
        metrics={
            "random_semantic_retrieval_at_1": solver["identity_retrieval_at_1"],
            "random_assignment_retrieval_at_1": random_assignment["identity_retrieval_at_1"],
            "xyz_centroid_retrieval_at_1": xyz["identity_retrieval_at_1"],
            "reference_retrieval_at_1": reference,
            "random_semantic_lift_vs_reference": lift,
        },
        reasons=[] if passed else ["random_semantic_beats_random_or_xyz_reference"],
    )


def _train_test_source_split_check(
    batch_reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    valid = [
        report["summary"]
        for report in batch_reports
        if report.get("validation_status") == "pass" and report.get("summary") is not None
    ]
    training = [
        summary for summary in valid if summary["permissions"]["allowed_for_training"]
    ]
    reasons = []
    split_reports = []
    if not training:
        reasons.append("no_training_allowed_teacher_evidence_batch")
    for summary in training:
        provenance = summary["provenance"]
        split = provenance.get("train_test_semantic_source_split")
        if not isinstance(split, Mapping):
            reasons.append(f"{summary['sample_id']}:missing_train_test_semantic_source_split")
            continue
        train_source = split.get("train_source")
        test_source = split.get("test_source")
        direct_shared = split.get("direct_object_id_embedding_shared")
        split_reports.append({
            "sample_id": summary["sample_id"],
            "source": summary["source"],
            "train_source": train_source,
            "test_source": test_source,
            "direct_object_id_embedding_shared": direct_shared,
        })
        if summary["source"] not in TEACHER_EVIDENCE_INFERENCE_TIME_SOURCES:
            reasons.append(f"{summary['sample_id']}:source_not_inference_time")
        if not isinstance(train_source, str) or not train_source:
            reasons.append(f"{summary['sample_id']}:missing_train_source")
        if not isinstance(test_source, str) or not test_source:
            reasons.append(f"{summary['sample_id']}:missing_test_source")
        if direct_shared is not False:
            reasons.append(f"{summary['sample_id']}:direct_object_id_embedding_shared")
    return _check(
        "train_test_semantic_source_split",
        passed=not reasons,
        metrics={
            "valid_batch_count": len(valid),
            "training_batch_count": len(training),
            "split_report_count": len(split_reports),
        },
        reasons=reasons,
        details={"split_reports": split_reports},
    )


def _benchmark_metrics(benchmark: Mapping[str, Any]) -> dict[str, float]:
    return _metric_digest(_baseline_metrics(benchmark, "assignment_solver_v2"))


def _baseline_metrics(benchmark: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    return benchmark["baselines"][name]["metrics"]


def _metric_digest(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {key: float(metrics[key]) for key in _METRIC_KEYS}


def _check(
    name: str,
    *,
    passed: bool,
    metrics: Mapping[str, Any],
    reasons: Sequence[str],
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "name": name,
        "status": "pass" if passed else "fail",
        "passed": bool(passed),
        "metrics": dict(metrics),
        "reasons": [str(item) for item in reasons],
        "details": {} if details is None else dict(details),
    }


def _validate_check(payload: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError(f"teacher evidence leakage audit check {name} must be a mapping")
    if payload.get("name") != name:
        raise ValueError(f"teacher evidence leakage audit check {name} has wrong name")
    if payload.get("status") not in {"pass", "fail"}:
        raise ValueError(f"teacher evidence leakage audit check {name} status unsupported")
    if not isinstance(payload.get("passed"), bool):
        raise ValueError(f"teacher evidence leakage audit check {name} passed must be bool")
    if (payload["status"] == "pass") is not bool(payload["passed"]):
        raise ValueError(f"teacher evidence leakage audit check {name} status mismatch")
    if not isinstance(payload.get("metrics"), Mapping):
        raise ValueError(f"teacher evidence leakage audit check {name} requires metrics")
    if not isinstance(payload.get("reasons"), list):
        raise ValueError(f"teacher evidence leakage audit check {name} requires reasons")
    if payload["passed"] and payload["reasons"]:
        raise ValueError(f"teacher evidence leakage audit check {name} pass cannot have reasons")
    if not payload["passed"] and not payload["reasons"]:
        raise ValueError(f"teacher evidence leakage audit check {name} fail requires reasons")
    if not isinstance(payload.get("details"), Mapping):
        raise ValueError(f"teacher evidence leakage audit check {name} requires details")
    return payload


def _validate_batch_report(payload: Any) -> None:
    if not isinstance(payload, Mapping):
        raise ValueError("teacher evidence leakage audit batch report must be a mapping")
    if not isinstance(payload.get("sample_id"), str) or not payload["sample_id"]:
        raise ValueError("teacher evidence leakage audit batch report requires sample_id")
    if payload.get("validation_status") not in {"pass", "fail"}:
        raise ValueError("teacher evidence leakage audit batch validation status unsupported")
    if payload["validation_status"] == "pass":
        if payload.get("error") is not None:
            raise ValueError("teacher evidence leakage audit passing batch cannot have error")
        if not isinstance(payload.get("summary"), Mapping):
            raise ValueError("teacher evidence leakage audit passing batch requires summary")
        if not _is_sha256(payload.get("feature_sha256")):
            raise ValueError(
                "teacher evidence leakage audit passing batch requires feature_sha256"
            )
    else:
        if not isinstance(payload.get("error"), str) or not payload["error"]:
            raise ValueError("teacher evidence leakage audit failing batch requires error")
        if payload.get("summary") is not None:
            raise ValueError("teacher evidence leakage audit failing batch cannot have summary")


def _validate_audited_evidence(
    payload: Mapping[str, Any],
    *,
    expected_scenarios: int,
    expected_batches: int,
) -> None:
    if payload.get("binding_policy") != "scenario_order_frame0_then_frame1":
        raise ValueError("teacher evidence leakage audit binding policy unsupported")
    scenario_ids = payload.get("scenario_ids")
    if (
        not isinstance(scenario_ids, list)
        or len(scenario_ids) != expected_scenarios
        or any(not isinstance(item, str) or not item for item in scenario_ids)
    ):
        raise ValueError("teacher evidence leakage audit scenario ids mismatch")
    if int(payload.get("scenario_count", 0)) != expected_scenarios:
        raise ValueError("teacher evidence leakage audit scenario count mismatch")
    if int(payload.get("frame_count", 0)) != expected_scenarios * 2:
        raise ValueError("teacher evidence leakage audit frame count mismatch")
    if int(payload.get("batch_count", 0)) != expected_batches:
        raise ValueError("teacher evidence leakage audit batch count mismatch")
    if int(payload.get("row_count", 0)) < 1 or int(payload.get("feature_dim", 0)) < 1:
        raise ValueError("teacher evidence leakage audit content shape is invalid")
    batch_digests = payload.get("batch_digests")
    if not isinstance(batch_digests, list) or len(batch_digests) != expected_batches:
        raise ValueError("teacher evidence leakage audit batch digests mismatch")
    for index, digest in enumerate(batch_digests):
        if not isinstance(digest, Mapping) or digest.get("batch_index") != index:
            raise ValueError("teacher evidence leakage audit batch digest order mismatch")
        if not isinstance(digest.get("sample_id"), str) or not digest["sample_id"]:
            raise ValueError("teacher evidence leakage audit batch digest sample_id missing")
        if not isinstance(digest.get("source"), str) or not digest["source"]:
            raise ValueError("teacher evidence leakage audit batch digest source missing")
        if int(digest.get("row_count", 0)) < 1:
            raise ValueError("teacher evidence leakage audit batch digest rows invalid")
        if int(digest.get("feature_dim", 0)) != int(payload["feature_dim"]):
            raise ValueError("teacher evidence leakage audit batch feature_dim mismatch")
        if not _is_sha256(digest.get("feature_sha256")):
            raise ValueError("teacher evidence leakage audit batch digest sha256 invalid")
    frame_digests = payload.get("frame_digests")
    if not isinstance(frame_digests, list) or len(frame_digests) != expected_scenarios * 2:
        raise ValueError("teacher evidence leakage audit frame digests mismatch")
    expected_offset = 0
    for digest in frame_digests:
        if not isinstance(digest, Mapping):
            raise ValueError("teacher evidence leakage audit frame digest must be mapping")
        if digest.get("scenario_id") not in scenario_ids:
            raise ValueError("teacher evidence leakage audit frame scenario mismatch")
        if digest.get("frame") not in {"frame0", "frame1"}:
            raise ValueError("teacher evidence leakage audit frame name unsupported")
        if int(digest.get("row_start", -1)) != expected_offset:
            raise ValueError("teacher evidence leakage audit frame offsets are not contiguous")
        row_count = int(digest.get("row_count", 0))
        expected_offset += row_count
        if row_count < 1 or int(digest.get("row_end", -1)) != expected_offset:
            raise ValueError("teacher evidence leakage audit frame range invalid")
        if not _is_sha256(digest.get("feature_sha256")):
            raise ValueError("teacher evidence leakage audit frame digest sha256 invalid")
    if expected_offset != int(payload["row_count"]):
        raise ValueError("teacher evidence leakage audit frame rows mismatch")
    if not _is_sha256(payload.get("content_sha256")):
        raise ValueError("teacher evidence leakage audit content sha256 invalid")
    without_sha = dict(payload)
    without_sha.pop("content_sha256", None)
    if payload["content_sha256"] != _content_sha256(without_sha):
        raise ValueError("teacher evidence leakage audit content sha256 mismatch")


def _validate_metric_digest(payload: Mapping[str, Any]) -> None:
    for key in _METRIC_KEYS:
        _finite(payload.get(key), key)


def _validate_report_ladder(
    scenarios: Sequence[ObjectStateModelIdentityBenchmarkScenario],
) -> None:
    if len(scenarios) != 15:
        raise ValueError("teacher evidence leakage audit requires the 15 report scenarios")
    expected_ids = set(objectstate_model_identity_benchmark_report_difficulty_by_scenario())
    scenario_ids = {str(scenario.scenario_id) for scenario in scenarios}
    if scenario_ids != expected_ids:
        raise ValueError("teacher evidence leakage audit scenarios must match the report ladder")
    for scenario in scenarios:
        _feature_matrix(
            scenario.frame0_features,
            "scenario.frame0_features",
            rows=scenario.frame0_cloud.count,
        )
        _feature_matrix(
            scenario.frame1_features,
            "scenario.frame1_features",
            rows=scenario.frame1_cloud.count,
        )


def _feature_matrix(value: np.ndarray | None, label: str, *, rows: int) -> np.ndarray:
    if value is None:
        raise ValueError(f"teacher evidence leakage audit requires {label}")
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"teacher evidence leakage audit {label} must be 2D")
    if array.shape[0] != rows:
        raise ValueError(f"teacher evidence leakage audit {label} row count mismatch")
    if array.shape[1] < 1:
        raise ValueError(f"teacher evidence leakage audit {label} requires features")
    if not np.isfinite(array).all():
        raise ValueError(f"teacher evidence leakage audit {label} must be finite")
    return array.astype(np.float32, copy=False)


def _feature_sha256(features: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(features, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("utf-8"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _content_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != 64:
        return False
    return all(character in "0123456789abcdef" for character in value)


def _mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"teacher evidence leakage audit requires {key}")
    return value


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"teacher evidence leakage audit {label} must be a number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"teacher evidence leakage audit {label} must be finite")
    return number


__all__ = (
    "OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA",
    "TEACHER_EVIDENCE_LEAKAGE_AUDIT_CHECKS",
    "TeacherEvidenceLeakageAuditThresholds",
    "objectstate_teacher_evidence_leakage_audit_summary",
    "teacher_evidence_scenarios_from_audit",
    "validate_objectstate_teacher_evidence_leakage_audit_summary",
)
