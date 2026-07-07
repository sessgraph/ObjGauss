from __future__ import annotations

import pytest

from objgauss.core.objectstate_reality_gate import ObjectStateRealityGateReport
from objgauss.core.objectstate_reality_public_rows import (
    OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA,
    ObjectStateRealityPublicArtifact,
    default_objectstate_reality_public_artifacts,
    evaluate_public_artifact_reality_gate,
    objectstate_reality_public_rows_summary,
    objectstate_reality_rows_from_public_artifacts,
    validate_objectstate_reality_public_rows_summary,
)


def test_public_artifact_rows_build_blocked_identity_prediction_intervention_rows():
    artifacts = default_objectstate_reality_public_artifacts()
    rows = objectstate_reality_rows_from_public_artifacts(artifacts)

    assert {artifact.sample_id for artifact in artifacts} == {
        "real-sample-v2-sample-aware-lego",
        "polyhaven-chair",
        "nike-real-splat-demo",
        "plush",
    }
    assert len(rows) == len(artifacts) * 3
    assert {row.status for row in rows} == {"blocked"}
    assert {row.evidence_kind for row in rows} == {
        "identity",
        "prediction",
        "intervention",
    }
    lego_identity = next(
        row
        for row in rows
        if row.sample_id == "real-sample-v2-sample-aware-lego"
        and row.evidence_kind == "identity"
    )
    assert "objgauss-real-sample-v2-sample-aware-lego.ply" in " ".join(
        lego_identity.artifact_refs
    )
    assert "object_id labels are renderer addresses" in lego_identity.block_reason
    assert lego_identity.has_identity_gt is False
    assert lego_identity.has_timestamp is False


def test_public_artifact_reality_gate_records_rows_as_blocked_not_pass():
    report = evaluate_public_artifact_reality_gate()
    payload = report.as_dict()

    assert isinstance(report, ObjectStateRealityGateReport)
    assert payload["status"] == "objectstate_reality_gate_fail"
    assert payload["row_count"] == 12
    assert payload["blocked_row_count"] == 12
    assert payload["pass_row_count"] == 0
    assert payload["fail_row_count"] == 0
    assert payload["metrics"]["controlled_real_or_public_row_count"] == 12
    assert payload["hard_gates"]["real_or_public_rows_present"] is True
    assert payload["hard_gates"]["identity_pass_rows_present"] is False
    assert payload["hard_gates"]["prediction_pass_rows_present"] is False
    assert payload["hard_gates"]["intervention_pass_rows_present"] is False
    assert payload["claim_policy"]["blocked_rows_are_not_pass_rows"] is True


def test_public_artifact_rows_summary_keeps_claim_boundaries():
    summary = objectstate_reality_public_rows_summary()

    assert summary["schema"] == OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA
    assert summary["artifact_count"] == 4
    assert summary["row_count"] == 12
    assert summary["blocked_row_count"] == 12
    assert summary["pass_row_count"] == 0
    assert summary["gate"]["status"] == "objectstate_reality_gate_fail"
    assert summary["claim_policy"] == {
        "public_artifact_rows_are_reality_evidence_candidates": True,
        "current_rows_are_blocked_not_pass": True,
        "object_id_is_not_identity_ground_truth": True,
        "does_not_claim_real_world_state_variable": True,
    }
    assert summary["non_goals"]["writes_public_samples"] is False
    assert "missing action events" in summary["blocked_rows_markdown"]
    assert validate_objectstate_reality_public_rows_summary(summary) is summary


def test_public_artifact_rows_reject_non_public_source_kind():
    with pytest.raises(ValueError, match="source_kind=public_replay"):
        objectstate_reality_rows_from_public_artifacts(
            (
                ObjectStateRealityPublicArtifact(
                    sample_id="bad-open-world",
                    object_category="room",
                    scenario="unsupported",
                    artifact_refs=("public/samples/bad.ply",),
                    license="unknown",
                    source_kind="open_world_real",
                ),
            )
        )
