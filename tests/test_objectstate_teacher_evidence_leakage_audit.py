from __future__ import annotations

from dataclasses import replace

import numpy as np

from objgauss.evaluation.objectstate_model_identity_benchmark_report import (
    objectstate_model_identity_benchmark_report_scenarios,
)
from objgauss.datasets.objectstate_teacher_evidence import TeacherEvidenceBatch
from objgauss.evaluation.objectstate_teacher_evidence_leakage_audit import (
    OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA,
    TEACHER_EVIDENCE_LEAKAGE_AUDIT_CHECKS,
    objectstate_teacher_evidence_leakage_audit_summary,
    validate_objectstate_teacher_evidence_leakage_audit_summary,
)


def test_teacher_evidence_leakage_audit_blocks_default_synthetic_training_source(tmp_path):
    summary = objectstate_teacher_evidence_leakage_audit_summary(
        tmp_path,
        sample_id="teacher-leakage-default-synthetic",
        seed=13,
    )

    assert summary["schema"] == OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA
    assert summary["status"] == "objectstate_teacher_evidence_leakage_audit_blocked"
    assert set(summary["audit_checks"]) == set(TEACHER_EVIDENCE_LEAKAGE_AUDIT_CHECKS)
    assert validate_objectstate_teacher_evidence_leakage_audit_summary(summary) == summary

    checks = summary["audit_checks"]
    assert checks["physical_label_ban"]["passed"] is False
    assert checks["physical_label_ban"]["metrics"][
        "direct_one_hot_identity_embedding_count"
    ] == 30
    assert checks["semantic_feature_shuffle"]["passed"] is True
    assert checks["random_semantic_baseline"]["passed"] is True
    assert checks["train_test_semantic_source_split"]["passed"] is False
    assert "no_training_allowed_teacher_evidence_batch" in (
        checks["train_test_semantic_source_split"]["reasons"]
    )
    assert summary["training_gate"]["semantic_teacher_evidence_training_allowed"] is False
    assert summary["teacher_evidence_batches"][0]["summary"]["source"] == "synthetic_semantic"
    assert (
        summary["teacher_evidence_batches"][0]["summary"]["permissions"]["allowed_for_training"]
        is False
    )

    shuffle = checks["semantic_feature_shuffle"]["metrics"]
    assert shuffle["retrieval_drop"] > 0.0
    random_baseline = checks["random_semantic_baseline"]["metrics"]
    assert random_baseline["random_semantic_lift_vs_reference"] <= 0.20


def test_teacher_evidence_leakage_audit_passes_inference_time_split_policy(tmp_path):
    scenarios = objectstate_model_identity_benchmark_report_scenarios()
    batch = _teacher_corpus_batch(
        scenarios,
        sample_id="teacher-leakage-dino-split",
    )

    summary = objectstate_teacher_evidence_leakage_audit_summary(
        tmp_path,
        sample_id="teacher-leakage-dino-pass",
        teacher_batches=(batch,),
        seed=17,
    )

    assert summary["status"] == "objectstate_teacher_evidence_leakage_audit_pass"
    assert summary["training_gate"]["status"] == "cleared"
    assert summary["training_gate"]["semantic_teacher_evidence_training_allowed"] is True
    for check in TEACHER_EVIDENCE_LEAKAGE_AUDIT_CHECKS:
        assert summary["audit_checks"][check]["passed"] is True
    split = summary["audit_checks"]["train_test_semantic_source_split"]["details"][
        "split_reports"
    ][0]
    assert split["direct_object_id_embedding_shared"] is False
    assert split["train_source"] == "dino_v2:train-scenes"
    assert summary["audited_evidence"]["feature_dim"] == 4
    assert summary["audited_evidence"]["frame_count"] == 30


def test_teacher_evidence_leakage_audit_blocks_disguised_one_hot_identity(
    tmp_path,
):
    scenarios = objectstate_model_identity_benchmark_report_scenarios()
    batch = _teacher_corpus_batch(
        scenarios,
        sample_id="teacher-leakage-disguised-one-hot",
        use_one_hot=True,
    )

    summary = objectstate_teacher_evidence_leakage_audit_summary(
        tmp_path,
        teacher_batches=(batch,),
        seed=18,
    )

    assert summary["status"] == "objectstate_teacher_evidence_leakage_audit_blocked"
    check = summary["audit_checks"]["physical_label_ban"]
    assert check["passed"] is False
    assert check["metrics"]["direct_one_hot_identity_embedding_count"] == 30
    assert all(
        finding["reason"]
        == "one_hot_active_column_is_bijective_with_identity_label"
        for finding in check["details"]["content_leakage_findings"]
    )
    assert summary["audit_checks"]["semantic_feature_shuffle"]["passed"] is True
    assert summary["audit_checks"]["random_semantic_baseline"]["passed"] is True


def test_teacher_evidence_leakage_audit_reports_forbidden_provenance(tmp_path):
    scenarios = objectstate_model_identity_benchmark_report_scenarios()
    batch = _teacher_corpus_batch(
        scenarios,
        sample_id="teacher-leakage-forbidden-provenance",
    )
    batch = replace(
        batch,
        provenance={**batch.provenance, "target_assignment": "forbidden"},
    )

    summary = objectstate_teacher_evidence_leakage_audit_summary(
        tmp_path,
        sample_id="teacher-leakage-forbidden-blocked",
        teacher_batches=(batch,),
        seed=19,
    )

    assert summary["status"] == "objectstate_teacher_evidence_leakage_audit_blocked"
    assert summary["audit_checks"]["physical_label_ban"]["passed"] is False
    assert "target_assignment" in summary["teacher_evidence_batches"][0]["error"]
    assert summary["training_gate"]["blocked_checks"] == [
        "physical_label_ban",
        "train_test_semantic_source_split",
    ]


def _teacher_corpus_batch(
    scenarios,
    *,
    sample_id: str,
    use_one_hot: bool = False,
) -> TeacherEvidenceBatch:
    projection = np.asarray(
        [
            [0.88, 0.04, 0.03, 0.02],
            [0.03, 0.87, 0.05, 0.02],
            [0.04, 0.02, 0.89, 0.03],
            [0.02, 0.05, 0.03, 0.86],
        ],
        dtype=np.float32,
    )
    matrices = []
    for scenario in scenarios:
        for features in (scenario.frame0_features, scenario.frame1_features):
            matrix = np.asarray(features, dtype=np.float32)
            matrices.append(matrix if use_one_hot else matrix @ projection)
    feature_matrix = np.concatenate(matrices, axis=0)
    return TeacherEvidenceBatch(
        sample_id=sample_id,
        gaussian_ids=tuple(f"g{index:06d}" for index in range(feature_matrix.shape[0])),
        feature_matrix=feature_matrix,
        source="dino_v2",
        confidence=0.92,
        uncertainty=0.08,
        allowed_for_training=True,
        allowed_for_evaluation=True,
        leakage_risk="low",
        provenance={
            "producer": "test-dino-teacher",
            "feature_space": "dino_v2_patch_embedding",
            "input_refs": ["fixture://identity-report-ladder"],
            "generation_method": "inference_time_teacher_embedding",
            "train_test_semantic_source_split": {
                "train_source": "dino_v2:train-scenes",
                "test_source": "dino_v2:heldout-scenes",
                "direct_object_id_embedding_shared": False,
                "policy": "model_weights_shared_without_object_id_embedding",
            },
        },
    )
