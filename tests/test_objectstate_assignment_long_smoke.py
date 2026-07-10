from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from objgauss.pipelines.objectstate_assignment_long_smoke import (
    OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA,
    objectstate_assignment_long_smoke_summary,
    validate_objectstate_assignment_long_smoke_summary,
)
from objgauss.evaluation.objectstate_model_identity_benchmark_report import (
    objectstate_model_identity_benchmark_report_scenarios,
)
from objgauss.datasets.objectstate_teacher_evidence import TeacherEvidenceBatch
from objgauss.evaluation.objectstate_teacher_evidence_leakage_audit import (
    objectstate_teacher_evidence_leakage_audit_summary,
)


def test_assignment_long_smoke_runs_bounded_semantic_before_after(tmp_path):
    scenarios = objectstate_model_identity_benchmark_report_scenarios()
    teacher_batch = _training_allowed_teacher_batch(scenarios)
    leakage_audit = _passed_leakage_audit(
        tmp_path,
        scenarios=scenarios,
        teacher_batch=teacher_batch,
    )
    summary = objectstate_assignment_long_smoke_summary(
        tmp_path / "long-smoke",
        scenarios=scenarios,
        teacher_evidence_leakage_audit=leakage_audit,
        teacher_evidence_batches=(teacher_batch,),
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
    assert summary["claim_policy"]["uses_exact_audited_teacher_evidence"] is True
    assert summary["teacher_evidence"]["matches_passed_leakage_audit"] is True
    assert summary["teacher_evidence"]["used_for_assignment_training"] is True
    assert summary["teacher_evidence"]["used_for_identity_evaluation"] is True
    assert summary["teacher_evidence"]["feature_dim"] == 4
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
    scenarios = objectstate_model_identity_benchmark_report_scenarios()
    teacher_batch = _training_allowed_teacher_batch(scenarios)
    leakage_audit = _passed_leakage_audit(
        tmp_path,
        scenarios=scenarios,
        teacher_batch=teacher_batch,
    )

    with pytest.raises(ValueError, match="iterations must stay <= 600"):
        objectstate_assignment_long_smoke_summary(
            tmp_path / "long-smoke",
            scenarios=scenarios,
            teacher_evidence_leakage_audit=leakage_audit,
            teacher_evidence_batches=(teacher_batch,),
            iterations=601,
        )


def test_assignment_long_smoke_rejects_teacher_content_changed_after_audit(tmp_path):
    scenarios = objectstate_model_identity_benchmark_report_scenarios()
    teacher_batch = _training_allowed_teacher_batch(scenarios)
    leakage_audit = _passed_leakage_audit(
        tmp_path,
        scenarios=scenarios,
        teacher_batch=teacher_batch,
    )
    changed = replace(
        teacher_batch,
        feature_matrix=np.asarray(teacher_batch.feature_matrix).copy(),
    )
    changed.feature_matrix[0, 0] += 0.25

    with pytest.raises(ValueError, match="content does not match"):
        objectstate_assignment_long_smoke_summary(
            tmp_path / "long-smoke-changed",
            scenarios=scenarios,
            teacher_evidence_leakage_audit=leakage_audit,
            teacher_evidence_batches=(changed,),
        )

    permission_changed = replace(teacher_batch, allowed_for_training=False)
    with pytest.raises(ValueError, match="content does not match"):
        objectstate_assignment_long_smoke_summary(
            tmp_path / "long-smoke-permission-changed",
            scenarios=scenarios,
            teacher_evidence_leakage_audit=leakage_audit,
            teacher_evidence_batches=(permission_changed,),
        )


def _passed_leakage_audit(tmp_path, *, scenarios, teacher_batch):
    return objectstate_teacher_evidence_leakage_audit_summary(
        tmp_path / "leakage-audit",
        scenarios=scenarios,
        teacher_batches=(teacher_batch,),
        seed=23,
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
