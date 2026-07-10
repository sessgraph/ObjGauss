from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from objgauss.pipelines.objectstate_assignment_long_smoke import (
    objectstate_assignment_long_smoke_summary,
)
from objgauss.datasets.objectstate_controlled_capture_environment import (
    objectstate_controlled_capture_environment,
)
from objgauss.pipelines.objectstate_controlled_capture_handoff import (
    OBJECTSTATE_CONTROLLED_CAPTURE_HANDOFF_SCHEMA,
    objectstate_controlled_capture_handoff_summary,
    validate_objectstate_controlled_capture_handoff_summary,
)
from objgauss.evaluation.objectstate_model_identity_benchmark_report import (
    objectstate_model_identity_benchmark_report_scenarios,
)
from objgauss.datasets.objectstate_teacher_evidence import TeacherEvidenceBatch
from objgauss.evaluation.objectstate_teacher_evidence_leakage_audit import (
    objectstate_teacher_evidence_leakage_audit_summary,
)
from objgauss.pipelines.objectstate_temporal_assignment import (
    objectstate_temporal_assignment_summary,
)
from objgauss.pipelines.objectstate_temporal_assignment_contract import (
    objectstate_temporal_assignment_contract_summary,
)


def test_controlled_capture_handoff_blocks_without_capture_environment(tmp_path):
    temporal = _passed_temporal_assignment(tmp_path)

    summary = objectstate_controlled_capture_handoff_summary(
        temporal_assignment_summary=temporal,
        sample_id="controlled-capture-handoff-blocked",
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_HANDOFF_SCHEMA
    assert summary["status"] == "objectstate_controlled_capture_handoff_blocked"
    assert summary["readiness"]["temporal_assignment_passed"] is True
    assert summary["readiness"]["controlled_capture_environment_ready"] is False
    assert summary["readiness"]["controlled_capture_collection_ready"] is False
    assert summary["readiness"]["controlled_evidence_handoff_ready"] is False
    assert "capture_environment_summary_missing" in summary["hard_blockers"]
    assert summary["next_actions"] == [
        "run audit-controlled-capture-environment on the capture host"
    ]
    assert summary["claim_policy"]["routes_to_existing_controlled_capture_toolchain"] is True
    assert summary["non_goals"]["captures_video"] is False
    assert validate_objectstate_controlled_capture_handoff_summary(summary) == summary


def test_controlled_capture_handoff_allows_collection_after_environment_ready(tmp_path):
    temporal = _passed_temporal_assignment(tmp_path)
    environment = _ready_capture_environment(tmp_path)

    summary = objectstate_controlled_capture_handoff_summary(
        temporal_assignment_summary=temporal,
        capture_environment_summary=environment,
        sample_id="controlled-capture-handoff-collection-ready",
    )

    assert summary["status"] == "objectstate_controlled_capture_collection_ready"
    assert summary["readiness"]["temporal_assignment_passed"] is True
    assert summary["readiness"]["controlled_capture_environment_ready"] is True
    assert summary["readiness"]["controlled_capture_collection_ready"] is True
    assert summary["readiness"]["controlled_evidence_handoff_ready"] is False
    assert "capture_bundle_readiness_summary_missing" in summary["hard_blockers"]
    assert summary["next_actions"] == [
        "initialize a local controlled capture bundle skeleton",
        "capture RGB / Gaussian evidence and fill objects, frames and annotations CSVs",
        "run audit-controlled-capture-bundle-readiness",
    ]
    assert "audit-controlled-capture-bundle-readiness" in (
        summary["handoff_routes"]["bundle_readiness_command"]
    )
    assert validate_objectstate_controlled_capture_handoff_summary(summary) == summary


def _ready_capture_environment(tmp_path):
    dev_root = tmp_path / "dev"
    dev_root.mkdir()
    (dev_root / "video0").write_text("", encoding="utf-8")

    def resolver(command: str) -> str:
        return f"/usr/bin/{command}"

    def importer(module: str):
        if module == "cv2":
            return SimpleNamespace(__version__="fixture")
        raise ImportError(module)

    return objectstate_controlled_capture_environment(
        dev_root=dev_root,
        command_resolver=resolver,
        importer=importer,
    )


def _passed_temporal_assignment(tmp_path):
    long_smoke = _passed_long_smoke(tmp_path)
    contract = objectstate_temporal_assignment_contract_summary(
        assignment_long_smoke_summary=long_smoke,
    )
    return objectstate_temporal_assignment_summary(
        tmp_path / "temporal-assignment",
        assignment_long_smoke_summary=long_smoke,
        temporal_assignment_contract=contract,
        sample_id="controlled-capture-handoff-temporal",
        seed=41,
    )


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
        sample_id="controlled-capture-handoff-long-smoke",
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
        sample_id="controlled-capture-handoff-dino-split",
        gaussian_ids=tuple(f"g{index:06d}" for index in range(feature_matrix.shape[0])),
        feature_matrix=feature_matrix,
        source="dino_v2",
        confidence=0.92,
        uncertainty=0.08,
        allowed_for_training=True,
        allowed_for_evaluation=True,
        leakage_risk="low",
        provenance={
            "producer": "controlled-capture-handoff-test",
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
