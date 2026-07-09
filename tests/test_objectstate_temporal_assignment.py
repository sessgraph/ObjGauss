from __future__ import annotations

import json

import numpy as np
import pytest

from objgauss.core.objectstate_assignment_long_smoke import (
    objectstate_assignment_long_smoke_summary,
)
from objgauss.core.objectstate_model_identity_benchmark_report import (
    objectstate_model_identity_benchmark_report_scenarios,
)
from objgauss.core.objectstate_teacher_evidence import TeacherEvidenceBatch
from objgauss.core.objectstate_teacher_evidence_leakage_audit import (
    objectstate_teacher_evidence_leakage_audit_summary,
)
from objgauss.core.objectstate_temporal_assignment import (
    OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA,
    objectstate_temporal_assignment_summary,
    validate_objectstate_temporal_assignment_summary,
)
from objgauss.core.objectstate_temporal_assignment_contract import (
    objectstate_temporal_assignment_contract_summary,
)


def test_temporal_assignment_writes_slot_manifest_and_passes(tmp_path):
    long_smoke = _passed_long_smoke(tmp_path)
    contract = objectstate_temporal_assignment_contract_summary(
        assignment_long_smoke_summary=long_smoke,
    )
    summary = objectstate_temporal_assignment_summary(
        tmp_path / "temporal-assignment",
        assignment_long_smoke_summary=long_smoke,
        temporal_assignment_contract=contract,
        sample_id="temporal-assignment-test",
        seed=41,
    )

    assert summary["schema"] == OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA
    assert summary["status"] == "objectstate_temporal_assignment_pass"
    assert summary["contract"]["readiness_gate"]["temporal_assignment_contract_ready"] is True
    assert summary["run_config"]["evidence_policy"] == "semantic"
    assert summary["run_config"]["temporal_policy"] == "disabled"
    assert summary["run_config"]["training"] == "not_run"
    assert summary["metrics"]["temporal_assignment_consistency"] == pytest.approx(1.0)
    assert summary["metrics"]["track_fragmentation_rate"] == pytest.approx(0.0)
    assert summary["metrics"]["slot_swap_rate"] == pytest.approx(0.0)
    assert summary["metrics"]["identity_retrieval_at_1"] == pytest.approx(1.0)
    assert summary["metrics"]["checkpoint_roundtrip"] is True
    assert all(check["passed"] for check in summary["success_checks"].values())
    assert summary["next_stage_gate"]["controlled_capture_allowed"] is True
    assert summary["next_stage_gate"]["next_recommended_pr"] == (
        "OBJECTSTATE-CONTROLLED-CAPTURE-001"
    )

    manifest_path = tmp_path / (
        "temporal-assignment/temporal-assignment-artifacts/"
        "temporal-slot-match-manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["scenario_count"] == 15
    assert manifest["identity_pair_count"] == 60
    assert manifest["consistent_pair_count"] == 60
    assert summary["artifact_refs"]["slot_match_manifest"] == str(manifest_path)

    assert validate_objectstate_temporal_assignment_summary(summary) == summary


def test_temporal_assignment_requires_ready_contract(tmp_path):
    long_smoke = _passed_long_smoke(tmp_path)
    blocked_contract = objectstate_temporal_assignment_contract_summary()

    with pytest.raises(ValueError, match="ready temporal assignment contract"):
        objectstate_temporal_assignment_summary(
            tmp_path / "temporal-assignment",
            assignment_long_smoke_summary=long_smoke,
            temporal_assignment_contract=blocked_contract,
        )


def _passed_long_smoke(tmp_path):
    leakage_audit = objectstate_teacher_evidence_leakage_audit_summary(
        tmp_path / "leakage-audit",
        teacher_batches=(_training_allowed_teacher_batch(),),
        seed=23,
    )
    return objectstate_assignment_long_smoke_summary(
        tmp_path / "long-smoke",
        teacher_evidence_leakage_audit=leakage_audit,
        sample_id="temporal-assignment-long-smoke",
        iterations=120,
        learning_rate=0.4,
        seed=31,
    )


def _training_allowed_teacher_batch() -> TeacherEvidenceBatch:
    scenarios = objectstate_model_identity_benchmark_report_scenarios()
    first = scenarios[0]
    feature_matrix = np.asarray(first.frame0_features, dtype=np.float32)
    return TeacherEvidenceBatch(
        sample_id="temporal-assignment-dino-split",
        gaussian_ids=tuple(f"g{index:06d}" for index in range(feature_matrix.shape[0])),
        feature_matrix=feature_matrix,
        source="dino_v2",
        confidence=0.92,
        uncertainty=0.08,
        allowed_for_training=True,
        allowed_for_evaluation=True,
        leakage_risk="low",
        provenance={
            "producer": "temporal-assignment-test",
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
