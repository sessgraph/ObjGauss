from __future__ import annotations

import numpy as np
import pytest

from objgauss.pipelines.objectstate_assignment_long_smoke_contract import (
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SCHEMA,
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SUMMARY_SCHEMA,
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SUCCESS_CRITERIA,
    ObjectStateAssignmentLongSmokeContractThresholds,
    objectstate_assignment_long_smoke_contract_summary,
    validate_objectstate_assignment_long_smoke_contract_summary,
)
from objgauss.evaluation.objectstate_model_identity_benchmark_report import (
    objectstate_model_identity_benchmark_report_scenarios,
)
from objgauss.datasets.objectstate_teacher_evidence import TeacherEvidenceBatch
from objgauss.evaluation.objectstate_teacher_evidence_leakage_audit import (
    objectstate_teacher_evidence_leakage_audit_summary,
)


def test_assignment_long_smoke_contract_blocks_without_leakage_audit():
    summary = objectstate_assignment_long_smoke_contract_summary()

    assert summary["schema"] == OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SUMMARY_SCHEMA
    assert summary["contract_schema"] == OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SCHEMA
    assert summary["status"] == "objectstate_assignment_long_smoke_contract_blocked"
    assert summary["evidence_policy"]["policy"] == "semantic"
    assert summary["duration_policy"]["max_duration_seconds"] == 600
    assert summary["readiness_gate"]["long_smoke_contract_ready"] is False
    assert "teacher_evidence_leakage_audit_missing" in summary["readiness_gate"]["blocked_reasons"]
    assert summary["readiness_gate"]["next_allowed_pr"] is None
    assert set(summary["success_criteria"]) == set(
        OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SUCCESS_CRITERIA
    )
    assert summary["success_criteria"]["identity_margin_improves"]["comparison"].startswith(
        "after > before"
    )
    assert summary["training_constraints"]["renderer_loss_allowed"] is False
    assert summary["training_constraints"]["temporal_loss_allowed"] is False
    assert summary["non_goals"]["runs_long_smoke"] is False
    assert validate_objectstate_assignment_long_smoke_contract_summary(summary) == summary


def test_assignment_long_smoke_contract_ready_after_passed_teacher_leakage_audit(tmp_path):
    leakage_audit = objectstate_teacher_evidence_leakage_audit_summary(
        tmp_path / "leakage-audit",
        teacher_batches=(_training_allowed_teacher_batch(),),
        seed=23,
    )
    assert leakage_audit["status"] == "objectstate_teacher_evidence_leakage_audit_pass"

    summary = objectstate_assignment_long_smoke_contract_summary(
        sample_id="assignment-long-smoke-contract-pass",
        teacher_evidence_leakage_audit=leakage_audit,
        thresholds=ObjectStateAssignmentLongSmokeContractThresholds(
            max_duration_seconds=300,
            min_identity_margin_delta=0.001,
            max_slot_swap_rate=0.25,
        ),
    )

    assert summary["status"] == "objectstate_assignment_long_smoke_contract_ready"
    assert summary["duration_policy"]["max_duration_seconds"] == 300
    assert summary["preconditions"]["teacher_evidence_leakage_audit_passed"] is True
    assert summary["preconditions"]["semantic_teacher_evidence_training_allowed"] is True
    assert summary["readiness_gate"]["long_smoke_contract_ready"] is True
    assert summary["readiness_gate"]["blocked_reasons"] == []
    assert summary["readiness_gate"]["next_allowed_pr"] == (
        "OBJECTSTATE-ASSIGNMENT-LONG-SMOKE-001"
    )
    assert summary["success_criteria"]["identity_margin_improves"]["min_delta"] == pytest.approx(
        0.001
    )
    assert summary["success_criteria"]["slot_swap_rate_interpretable"]["max_value"] == pytest.approx(
        0.25
    )
    assert summary["required_run_artifacts"]["before_identity_benchmark_summary"] is True
    assert validate_objectstate_assignment_long_smoke_contract_summary(summary) == summary


def test_assignment_long_smoke_contract_rejects_native_policy():
    with pytest.raises(ValueError, match="policy=semantic"):
        objectstate_assignment_long_smoke_contract_summary(evidence_policy="xyz_rgb_opacity")


def _training_allowed_teacher_batch() -> TeacherEvidenceBatch:
    scenarios = objectstate_model_identity_benchmark_report_scenarios()
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
        sample_id="assignment-long-smoke-dino-split",
        gaussian_ids=tuple(f"g{index:06d}" for index in range(feature_matrix.shape[0])),
        feature_matrix=feature_matrix,
        source="dino_v2",
        allowed_for_training=True,
        allowed_for_evaluation=True,
        leakage_risk="low",
        provenance={
            "producer": "assignment-long-smoke-contract-test",
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
