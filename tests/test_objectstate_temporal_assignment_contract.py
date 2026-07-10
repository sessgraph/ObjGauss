from __future__ import annotations

import numpy as np
import pytest

from objgauss.pipelines.objectstate_assignment_long_smoke import (
    objectstate_assignment_long_smoke_summary,
)
from objgauss.evaluation.objectstate_model_identity_benchmark_report import (
    objectstate_model_identity_benchmark_report_scenarios,
)
from objgauss.datasets.objectstate_teacher_evidence import TeacherEvidenceBatch
from objgauss.evaluation.objectstate_teacher_evidence_leakage_audit import (
    objectstate_teacher_evidence_leakage_audit_summary,
)
from objgauss.pipelines.objectstate_temporal_assignment_contract import (
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SCHEMA,
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA,
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_INPUTS,
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_LOSS_TERMS,
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_METRICS,
    ObjectStateTemporalAssignmentContractThresholds,
    objectstate_temporal_assignment_contract_summary,
    validate_objectstate_temporal_assignment_contract_summary,
)


def test_temporal_assignment_contract_blocks_without_long_smoke():
    summary = objectstate_temporal_assignment_contract_summary()

    assert summary["schema"] == OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA
    assert summary["contract_schema"] == OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SCHEMA
    assert summary["status"] == "objectstate_temporal_assignment_contract_blocked"
    assert summary["evidence_policy"]["policy"] == "semantic"
    assert summary["preconditions"]["assignment_long_smoke"]["status"] == "missing"
    assert summary["readiness_gate"]["temporal_assignment_contract_ready"] is False
    assert "assignment_long_smoke_missing" in summary["readiness_gate"]["blocked_reasons"]
    assert summary["readiness_gate"]["next_allowed_pr"] is None
    assert tuple(summary["temporal_inputs"]["required_inputs"]) == (
        OBJECTSTATE_TEMPORAL_ASSIGNMENT_INPUTS
    )
    assert tuple(summary["loss_contract"]["allowed_loss_terms"]) == (
        OBJECTSTATE_TEMPORAL_ASSIGNMENT_LOSS_TERMS
    )
    assert set(summary["success_metrics"]) == set(OBJECTSTATE_TEMPORAL_ASSIGNMENT_METRICS)
    assert summary["training_constraints"]["renderer_loss_allowed"] is False
    assert summary["training_constraints"]["dynamics_allowed"] is False
    assert summary["non_goals"]["runs_temporal_training"] is False
    assert validate_objectstate_temporal_assignment_contract_summary(summary) == summary


def test_temporal_assignment_contract_ready_after_passed_long_smoke(tmp_path):
    long_smoke = _passed_long_smoke(tmp_path)
    assert long_smoke["status"] == "objectstate_assignment_long_smoke_pass"

    summary = objectstate_temporal_assignment_contract_summary(
        sample_id="temporal-assignment-contract-pass",
        assignment_long_smoke_summary=long_smoke,
        thresholds=ObjectStateTemporalAssignmentContractThresholds(
            max_duration_seconds=300,
            min_temporal_assignment_consistency=0.90,
            max_slot_swap_rate=0.20,
        ),
    )

    assert summary["status"] == "objectstate_temporal_assignment_contract_ready"
    assert summary["duration_policy"]["max_duration_seconds"] == 300
    assert summary["preconditions"]["assignment_long_smoke_passed"] is True
    assert summary["preconditions"]["temporal_contract_unlocked"] is True
    assert summary["readiness_gate"]["temporal_assignment_contract_ready"] is True
    assert summary["readiness_gate"]["blocked_reasons"] == []
    assert summary["readiness_gate"]["next_allowed_pr"] == (
        "OBJECTSTATE-TEMPORAL-ASSIGNMENT-001"
    )
    assert summary["success_metrics"]["temporal_assignment_consistency"][
        "threshold"
    ] == pytest.approx(0.90)
    assert summary["success_metrics"]["slot_swap_rate"]["threshold"] == pytest.approx(0.20)
    assert summary["required_run_artifacts"]["slot_match_manifest"] is True
    assert validate_objectstate_temporal_assignment_contract_summary(summary) == summary


def test_temporal_assignment_contract_rejects_native_policy():
    with pytest.raises(ValueError, match="policy=semantic"):
        objectstate_temporal_assignment_contract_summary(evidence_policy="xyz_rgb")


def _passed_long_smoke(tmp_path):
    scenarios = objectstate_model_identity_benchmark_report_scenarios()
    teacher_batch = _training_allowed_teacher_batch(scenarios)
    leakage_audit = objectstate_teacher_evidence_leakage_audit_summary(
        tmp_path / "leakage-audit",
        scenarios=scenarios,
        teacher_batches=(teacher_batch,),
        seed=23,
    )
    return objectstate_assignment_long_smoke_summary(
        tmp_path / "long-smoke",
        scenarios=scenarios,
        teacher_evidence_leakage_audit=leakage_audit,
        teacher_evidence_batches=(teacher_batch,),
        sample_id="temporal-assignment-contract-long-smoke",
        iterations=120,
        learning_rate=0.4,
        seed=31,
    )


def _training_allowed_teacher_batch(scenarios) -> TeacherEvidenceBatch:
    projection = np.asarray(
        [
            [0.88, 0.04, 0.03, 0.02],
            [0.03, 0.87, 0.05, 0.02],
            [0.04, 0.02, 0.89, 0.03],
            [0.02, 0.05, 0.03, 0.86],
        ],
        dtype=np.float32,
    )
    feature_matrix = np.concatenate(
        [
            np.asarray(features, dtype=np.float32) @ projection
            for scenario in scenarios
            for features in (scenario.frame0_features, scenario.frame1_features)
        ],
        axis=0,
    )
    return TeacherEvidenceBatch(
        sample_id="temporal-assignment-contract-dino-split",
        gaussian_ids=tuple(f"g{index:06d}" for index in range(feature_matrix.shape[0])),
        feature_matrix=feature_matrix,
        source="dino_v2",
        confidence=0.92,
        uncertainty=0.08,
        allowed_for_training=True,
        allowed_for_evaluation=True,
        leakage_risk="low",
        provenance={
            "producer": "temporal-assignment-contract-test",
            "feature_space": "dino_v2_patch_embedding",
            "input_refs": ["fixture://viewpoint-easy/frame0"],
            "generation_method": "inference_time_teacher_embedding",
            "train_test_semantic_source_split": {
                "train_source": "dino_v2:train-scenes",
                "test_source": "dino_v2:heldout-scenes",
                "direct_object_id_embedding_shared": False,
                "policy": "model_weights_shared_without_object_id_embedding",
            },
        },
    )
