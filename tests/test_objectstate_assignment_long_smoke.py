from __future__ import annotations

import json

import numpy as np
import pytest

from objgauss.core.objectstate_assignment_long_smoke import (
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA,
    objectstate_assignment_long_smoke_summary,
    validate_objectstate_assignment_long_smoke_summary,
)
from objgauss.core.objectstate_model_identity_benchmark_report import (
    objectstate_model_identity_benchmark_report_scenarios,
)
from objgauss.core.objectstate_teacher_evidence import TeacherEvidenceBatch
from objgauss.core.objectstate_teacher_evidence_leakage_audit import (
    objectstate_teacher_evidence_leakage_audit_summary,
)


def test_assignment_long_smoke_runs_bounded_semantic_before_after(tmp_path):
    leakage_audit = _passed_leakage_audit(tmp_path)
    summary = objectstate_assignment_long_smoke_summary(
        tmp_path / "long-smoke",
        teacher_evidence_leakage_audit=leakage_audit,
        sample_id="assignment-long-smoke-test",
        iterations=120,
        learning_rate=0.4,
        seed=31,
    )

    assert summary["schema"] == OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA
    assert summary["status"] == "objectstate_assignment_long_smoke_pass"
    assert summary["contract"]["readiness_gate"]["long_smoke_contract_ready"] is True
    assert summary["run_config"]["evidence_policy"] == "semantic"
    assert summary["run_config"]["iterations"] == 120
    assert summary["duration"]["within_limit"] is True
    assert summary["training"]["loss_decreased"] is True
    assert summary["training"]["supervised_loss_decreased"] is True

    held_before = summary["metrics"]["held_out_before"]
    held_after = summary["metrics"]["held_out_after"]
    assert held_after["identity_retrieval_at_1"] >= held_before["identity_retrieval_at_1"]
    assert held_after["identity_margin"] > held_before["identity_margin"]
    assert held_after["occlusion_recovery"] >= held_before["occlusion_recovery"]

    assert all(check["passed"] for check in summary["success_checks"].values())
    assert summary["checkpoint"]["roundtrip_ok"] is True
    assert summary["next_stage_gate"]["temporal_assignment_contract_allowed"] is True
    assert summary["next_stage_gate"]["next_recommended_pr"] == (
        "OBJECTSTATE-TEMPORAL-ASSIGNMENT-CONTRACT-001"
    )
    assert summary["claim_policy"]["uses_passed_teacher_evidence_leakage_audit"] is True
    assert summary["non_goals"]["uses_temporal_loss"] is False

    refs = summary["artifact_refs"]
    assert json.loads(
        (tmp_path / "long-smoke" / "assignment-long-smoke-summary.json").read_text(
            encoding="utf-8"
        )
    )["schema"] == OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA
    assert refs["summary"] == str(
        tmp_path / "long-smoke" / "assignment-long-smoke-summary.json"
    )
    assert (tmp_path / "long-smoke" / "assignment-long-smoke-final-state.json").exists()
    assert refs["held_out_after_restored_identity_benchmark_summary"]

    assert validate_objectstate_assignment_long_smoke_summary(summary) == summary


def test_assignment_long_smoke_requires_ready_contract(tmp_path):
    with pytest.raises(ValueError, match="ready long-smoke contract"):
        objectstate_assignment_long_smoke_summary(
            tmp_path,
            teacher_evidence_leakage_audit=None,
        )


def test_assignment_long_smoke_rejects_unbounded_iteration_count(tmp_path):
    leakage_audit = _passed_leakage_audit(tmp_path)

    with pytest.raises(ValueError, match="iterations must stay <= 600"):
        objectstate_assignment_long_smoke_summary(
            tmp_path / "long-smoke",
            teacher_evidence_leakage_audit=leakage_audit,
            iterations=601,
        )


def _passed_leakage_audit(tmp_path):
    return objectstate_teacher_evidence_leakage_audit_summary(
        tmp_path / "leakage-audit",
        teacher_batches=(_training_allowed_teacher_batch(),),
        seed=23,
    )


def _training_allowed_teacher_batch() -> TeacherEvidenceBatch:
    scenarios = objectstate_model_identity_benchmark_report_scenarios()
    first = scenarios[0]
    feature_matrix = np.asarray(first.frame0_features, dtype=np.float32)
    return TeacherEvidenceBatch(
        sample_id="assignment-long-smoke-dino-split",
        gaussian_ids=tuple(f"g{index:06d}" for index in range(feature_matrix.shape[0])),
        feature_matrix=feature_matrix,
        source="dino_v2",
        confidence=0.92,
        uncertainty=0.08,
        allowed_for_training=True,
        allowed_for_evaluation=True,
        leakage_risk="low",
        provenance={
            "producer": "assignment-long-smoke-test",
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
