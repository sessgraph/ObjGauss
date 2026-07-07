from __future__ import annotations

import pytest

from objgauss.core.objectstate_reality_gate import (
    OBJECTSTATE_REALITY_GATE_SCHEMA,
    OBJECTSTATE_REALITY_ROW_SCHEMA,
    ObjectStateRealityGateReport,
    ObjectStateRealityRow,
    evaluate_objectstate_reality_gate,
    objectstate_reality_blocked_rows_markdown,
    validate_objectstate_reality_gate_summary,
)


def test_objectstate_reality_gate_passes_controlled_real_public_rows():
    report = evaluate_objectstate_reality_gate(
        (
            _identity_row(),
            _prediction_row(),
            _intervention_row(),
            _blocked_public_row(),
        ),
        synthetic_smoke_passed=True,
    )
    payload = report.as_dict()

    assert isinstance(report, ObjectStateRealityGateReport)
    assert payload["schema"] == OBJECTSTATE_REALITY_GATE_SCHEMA
    assert payload["row_schema"] == OBJECTSTATE_REALITY_ROW_SCHEMA
    assert payload["status"] == "objectstate_reality_gate_pass"
    assert payload["pass_row_count"] == 3
    assert payload["blocked_row_count"] == 1
    assert payload["metrics"]["controlled_real_or_public_row_count"] == 4
    assert payload["metrics"]["controlled_real_identity_collapse"] is False
    assert payload["metrics"]["controlled_real_fragmentation_rate"] == 0.0
    assert payload["metrics"]["controlled_real_swap_rate"] == 0.0
    assert payload["metrics"]["short_horizon_prediction_gap_vs_history_model"] == 0.02
    assert payload["metrics"]["intervention_counterfactual_outcome_accuracy"] == 1.0
    assert payload["hard_blockers"] == []
    assert payload["claim_policy"]["does_not_claim_world_model"] is True
    assert payload["non_goals"]["uses_replay_buffer"] is False
    assert payload["blocked_rows"][0]["status"] == "blocked"
    assert "missing 6DoF" in objectstate_reality_blocked_rows_markdown(report)
    assert validate_objectstate_reality_gate_summary(payload) is payload


def test_objectstate_reality_gate_fails_when_real_rows_are_only_blocked():
    report = evaluate_objectstate_reality_gate(
        (_blocked_public_row(),),
        synthetic_smoke_passed=True,
    )
    payload = report.as_dict()

    assert payload["status"] == "objectstate_reality_gate_fail"
    assert payload["pass_row_count"] == 0
    assert payload["blocked_row_count"] == 1
    assert payload["hard_gates"]["real_or_public_rows_present"] is True
    assert payload["hard_gates"]["identity_pass_rows_present"] is False
    assert payload["hard_gates"]["prediction_pass_rows_present"] is False
    assert payload["hard_gates"]["intervention_pass_rows_present"] is False
    assert "identity_pass_rows_present" in payload["hard_blockers"]
    assert validate_objectstate_reality_gate_summary(payload) is payload


def test_objectstate_reality_gate_fails_on_reported_identity_collapse():
    row = _identity_row(
        status="fail",
        metrics={
            "idf1": 0.4,
            "fragmentation_rate": 0.7,
            "swap_rate": 0.2,
            "identity_collapse": True,
        },
        failure_reason="identity_collapse_after_reappearance",
    )
    report = evaluate_objectstate_reality_gate(
        (row, _prediction_row(), _intervention_row()),
        synthetic_smoke_passed=True,
    )
    payload = report.as_dict()

    assert payload["status"] == "objectstate_reality_gate_fail"
    assert payload["metrics"]["controlled_real_identity_collapse"] is True
    assert payload["hard_gates"]["controlled_real_identity_collapse_absent"] is False
    assert payload["hard_gates"]["failed_rows_absent"] is False
    assert "controlled_real_identity_collapse_absent" in payload["hard_blockers"]
    assert "failed_rows_absent" in payload["hard_blockers"]
    assert validate_objectstate_reality_gate_summary(payload) is payload


def test_objectstate_reality_gate_rejects_open_world_pass_rows():
    with pytest.raises(ValueError, match="open_world_real rows cannot be marked pass"):
        evaluate_objectstate_reality_gate(
            (
                _identity_row(
                    row_id="open-world-pass",
                    sample_id="open-kitchen-001",
                    source_kind="open_world_real",
                ),
            ),
            synthetic_smoke_passed=True,
        )


def test_objectstate_reality_gate_rejects_non_blocked_rows_without_required_gt():
    with pytest.raises(ValueError, match="intervention reality rows require pose and action"):
        evaluate_objectstate_reality_gate(
            (
                _intervention_row(
                    has_action_gt=False,
                ),
            ),
            synthetic_smoke_passed=True,
        )


def _identity_row(**overrides):
    data = {
        "row_id": "cup-identity-001",
        "sample_id": "controlled-tabletop-cup-001",
        "source_kind": "controlled_real",
        "evidence_kind": "identity",
        "status": "pass",
        "object_category": "cup",
        "scenario": "occlusion_reappearance",
        "observation_modalities": ("rgb", "gaussian"),
        "artifact_refs": ("datasets/controlled-tabletop-cup-001/manifest.json",),
        "metrics": {
            "idf1": 1.0,
            "fragmentation_rate": 0.0,
            "swap_rate": 0.0,
            "identity_collapse": False,
        },
        "has_identity_gt": True,
        "has_pose_gt": True,
        "has_action_gt": False,
        "has_timestamp": True,
        "license": "local-research",
    }
    data.update(overrides)
    return ObjectStateRealityRow(**data)


def _prediction_row(**overrides):
    data = {
        "row_id": "cup-prediction-001",
        "sample_id": "controlled-tabletop-cup-001",
        "source_kind": "controlled_real",
        "evidence_kind": "prediction",
        "status": "pass",
        "object_category": "cup",
        "scenario": "short_horizon_push_replay",
        "observation_modalities": ("rgb", "gaussian"),
        "artifact_refs": ("datasets/controlled-tabletop-cup-001/pose-tracks.json",),
        "metrics": {
            "state_ade": 0.06,
            "history_ade": 0.04,
            "prediction_gap_vs_history_model": 0.02,
        },
        "has_identity_gt": True,
        "has_pose_gt": True,
        "has_action_gt": False,
        "has_timestamp": True,
        "license": "local-research",
    }
    data.update(overrides)
    return ObjectStateRealityRow(**data)


def _intervention_row(**overrides):
    data = {
        "row_id": "cup-intervention-001",
        "sample_id": "controlled-tabletop-cup-001",
        "source_kind": "controlled_real",
        "evidence_kind": "intervention",
        "status": "pass",
        "object_category": "cup",
        "scenario": "push_left_counterfactual",
        "observation_modalities": ("rgb", "gaussian"),
        "artifact_refs": ("datasets/controlled-tabletop-cup-001/actions.json",),
        "metrics": {
            "action_conditioned_ade": 0.03,
            "counterfactual_outcome_accuracy": 1.0,
            "wrong_direction_rate": 0.0,
        },
        "has_identity_gt": True,
        "has_pose_gt": True,
        "has_action_gt": True,
        "has_timestamp": True,
        "license": "local-research",
    }
    data.update(overrides)
    return ObjectStateRealityRow(**data)


def _blocked_public_row(**overrides):
    data = {
        "row_id": "lego-action-blocked-001",
        "sample_id": "nerf-synthetic-lego-public",
        "source_kind": "public_replay",
        "evidence_kind": "intervention",
        "status": "blocked",
        "object_category": "lego",
        "scenario": "public_static_replay_without_action_gt",
        "observation_modalities": ("rgb", "gaussian"),
        "artifact_refs": ("public/samples/objgauss-real-sample-v2.ply",),
        "metrics": {},
        "has_identity_gt": True,
        "has_pose_gt": False,
        "has_action_gt": False,
        "has_timestamp": True,
        "license": "nerf-synthetic",
        "block_reason": "missing 6DoF pose and action ground truth",
    }
    data.update(overrides)
    return ObjectStateRealityRow(**data)
