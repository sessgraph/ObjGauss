from __future__ import annotations

import numpy as np
import pytest

from objgauss.datasets.objectstate_teacher_evidence import (
    OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA,
    OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SCHEMA,
    OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SUMMARY_SCHEMA,
    TEACHER_EVIDENCE_FORBIDDEN_PROVENANCE_KEYS,
    TEACHER_EVIDENCE_SOURCES,
    TeacherEvidenceBatch,
    objectstate_teacher_evidence_contract_summary,
    teacher_evidence_batch_summary,
    validate_objectstate_teacher_evidence_contract_summary,
    validate_teacher_evidence_batch,
    validate_teacher_evidence_batch_summary,
)


def test_teacher_evidence_batch_summarizes_inference_time_features():
    batch = _batch(source="dino_v2", leakage_risk="low", allowed_for_training=True)

    checked = validate_teacher_evidence_batch(batch)
    summary = teacher_evidence_batch_summary(batch)

    assert checked.schema == OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA
    assert summary["schema"] == OBJECTSTATE_TEACHER_EVIDENCE_BATCH_SCHEMA
    assert summary["sample_id"] == "teacher-evidence-test"
    assert summary["source"] == "dino_v2"
    assert summary["evidence_count"] == 3
    assert summary["feature_dim"] == 4
    assert summary["feature_matrix"]["inline_values_stored"] is False
    assert summary["permissions"]["allowed_for_training"] is True
    assert summary["permissions"]["allowed_for_evaluation"] is True
    assert summary["leakage"]["risk"] == "low"
    assert summary["leakage"]["source_is_inference_time"] is True
    assert summary["leakage"]["requires_leakage_audit"] is False
    assert summary["claim_policy"]["teacher_evidence_is_not_ground_truth_identity"] is True
    assert validate_teacher_evidence_batch_summary(summary) == summary


def test_teacher_evidence_rejects_gt_leakage_provenance_keys():
    with pytest.raises(ValueError, match="forbidden GT leakage keys"):
        validate_teacher_evidence_batch(
            _batch(
                provenance={
                    "producer": "fixture",
                    "feature_space": "identity-one-hot",
                    "input_refs": ["fixture://frame0"],
                    "generation_method": "copied_from_gt",
                    "target_assignment": "forbidden",
                },
            )
        )

    assert "target_assignment" in TEACHER_EVIDENCE_FORBIDDEN_PROVENANCE_KEYS


def test_teacher_evidence_training_requires_low_risk_inference_source():
    with pytest.raises(ValueError, match="leakage_risk none or low"):
        validate_teacher_evidence_batch(
            _batch(source="dino_v2", leakage_risk="medium", allowed_for_training=True)
        )

    with pytest.raises(ValueError, match="inference-time source"):
        validate_teacher_evidence_batch(
            _batch(
                source="synthetic_semantic",
                leakage_risk="low",
                allowed_for_training=True,
            )
        )

    evaluation_only = validate_teacher_evidence_batch(
        _batch(
            source="synthetic_semantic",
            leakage_risk="medium",
            allowed_for_training=False,
            allowed_for_evaluation=True,
        )
    )
    assert evaluation_only.allowed_for_training is False
    assert evaluation_only.allowed_for_evaluation is True


def test_teacher_evidence_contract_summary_lists_required_leakage_rules():
    summary = objectstate_teacher_evidence_contract_summary(
        _batch(source="clip", leakage_risk="none", allowed_for_training=True)
    )

    assert summary["schema"] == OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SUMMARY_SCHEMA
    assert summary["contract_schema"] == OBJECTSTATE_TEACHER_EVIDENCE_CONTRACT_SCHEMA
    assert set(summary["allowed_sources"]) == set(TEACHER_EVIDENCE_SOURCES)
    assert "physical_label_ban" in summary["next_required_audits"]
    assert "random_semantic_baseline" in summary["next_required_audits"]
    assert "target_assignment" in summary["forbidden_provenance_keys"]
    assert summary["contract_rules"]["teacher_evidence_is_not_ground_truth_identity"] is True
    assert summary["example_batch_summary"]["source"] == "clip"
    assert validate_objectstate_teacher_evidence_contract_summary(summary) == summary


def _batch(
    *,
    source: str = "synthetic_semantic",
    leakage_risk: str = "medium",
    allowed_for_training: bool = False,
    allowed_for_evaluation: bool = True,
    provenance: dict[str, object] | None = None,
) -> TeacherEvidenceBatch:
    return TeacherEvidenceBatch(
        sample_id="teacher-evidence-test",
        gaussian_ids=("g0", "g1", "g2"),
        feature_matrix=np.asarray(
            [
                [1.0, 0.0, 0.0, 0.1],
                [0.0, 1.0, 0.0, 0.2],
                [0.0, 0.0, 1.0, 0.3],
            ],
            dtype=np.float32,
        ),
        evidence_policy="semantic",
        source=source,
        confidence=np.asarray([0.9, 0.8, 0.85], dtype=np.float32),
        uncertainty=0.1,
        provenance=provenance
        or {
            "producer": "unit-test",
            "feature_space": "teacher-embedding-v1",
            "input_refs": ["fixture://rgb/frame0"],
            "generation_method": "inference_time_teacher_embedding",
        },
        allowed_for_training=allowed_for_training,
        allowed_for_evaluation=allowed_for_evaluation,
        leakage_risk=leakage_risk,
    )
