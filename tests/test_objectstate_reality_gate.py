from __future__ import annotations

import pytest

from objgauss.evaluation.objectstate_reality_gate import (
    OBJECTSTATE_REALITY_GATE_SCHEMA,
    OBJECTSTATE_REALITY_ROW_SCHEMA,
    ObjectStateRealityGateReport,
    ObjectStateRealityGateThresholds,
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
    assert payload["metrics"][
        "short_horizon_prediction_gap_vs_history_model"
    ] == pytest.approx(-0.02)
    assert payload["metrics"]["intervention_counterfactual_outcome_accuracy"] == 1.0
    assert payload["declaration_diagnostics"]["caller_status_mismatch_count"] == 0
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


def test_objectstate_reality_gate_derives_open_world_declared_pass_as_fail():
    report = evaluate_objectstate_reality_gate(
        (
            _identity_row(
                row_id="open-world-pass",
                sample_id="open-kitchen-001",
                source_kind="open_world_real",
            ),
        ),
        synthetic_smoke_passed=True,
        thresholds=ObjectStateRealityGateThresholds(
            require_prediction_pass_row=False,
            require_intervention_pass_row=False,
        ),
    )
    payload = report.as_dict()

    assert payload["rows"][0]["status"] == "fail"
    assert "open_world_real_not_eligible_for_pass" in payload["rows"][0][
        "failure_reason"
    ]
    assert payload["hard_gates"]["no_open_world_pass_rows"] is True
    assert payload["declaration_diagnostics"]["caller_status_mismatch_count"] == 1


def test_objectstate_reality_gate_derives_forged_pass_as_fail():
    report = evaluate_objectstate_reality_gate(
        (
            _identity_row(),
            _prediction_row(
                metrics={
                    "state_ade": 0.06,
                    "history_ade": 0.04,
                    "prediction_gap_vs_history_model": -100.0,
                }
            ),
            _intervention_row(),
        ),
        synthetic_smoke_passed=True,
    )
    payload = report.as_dict()

    prediction = next(
        row for row in payload["rows"] if row["evidence_kind"] == "prediction"
    )
    assert prediction["status"] == "fail"
    assert prediction["metrics"]["prediction_gap_vs_history_model"] == pytest.approx(
        0.02
    )
    assert "state_ade_above_maximum" in prediction["failure_reason"]
    assert "state_does_not_strictly_beat_history" in prediction["failure_reason"]
    assert payload["declaration_diagnostics"]["caller_status_mismatch_count"] == 1
    assert payload["declaration_diagnostics"]["derived_metric_mismatch_count"] == 1


def test_objectstate_reality_gate_derives_forged_fail_as_pass():
    report = evaluate_objectstate_reality_gate(
        (
            _identity_row(
                status="fail",
                failure_reason="caller claimed failure despite passing primitives",
            ),
            _prediction_row(),
            _intervention_row(),
        ),
        synthetic_smoke_passed=True,
    )
    payload = report.as_dict()

    identity = next(
        row for row in payload["rows"] if row["evidence_kind"] == "identity"
    )
    assert identity["status"] == "pass"
    assert identity["failure_reason"] is None
    assert payload["status"] == "objectstate_reality_gate_pass"
    assert payload["declaration_diagnostics"]["caller_status_mismatch_count"] == 1


def test_objectstate_reality_gate_does_not_upgrade_preassociated_identity_metrics():
    report = evaluate_objectstate_reality_gate(
        (
            _identity_row(
                status="fail",
                failure_reason="legacy predictions were preassociated by GT object id",
                metrics={
                    "idf1": 1.0,
                    "fragmentation_rate": 0.0,
                    "swap_rate": 0.0,
                    "identity_collapse": False,
                    "raw_prediction_observations": False,
                },
            ),
        ),
        synthetic_smoke_passed=True,
        thresholds=ObjectStateRealityGateThresholds(
            require_prediction_pass_row=False,
            require_intervention_pass_row=False,
        ),
    )

    assert report.rows[0].status == "fail"
    assert "raw_prediction_observations_required" in report.rows[0].failure_reason
    assert report.declaration_diagnostics["caller_status_mismatch_count"] == 0


def test_objectstate_reality_gate_treats_missing_raw_identity_flag_as_legacy_fail():
    metrics = dict(_identity_row().metrics)
    metrics.pop("raw_prediction_observations")
    report = evaluate_objectstate_reality_gate(
        (_identity_row(metrics=metrics),),
        synthetic_smoke_passed=True,
        thresholds=ObjectStateRealityGateThresholds(
            require_prediction_pass_row=False,
            require_intervention_pass_row=False,
        ),
    )

    assert report.rows[0].status == "fail"
    assert report.rows[0].metrics["raw_prediction_observations"] is False
    assert "raw_prediction_observations_required" in report.rows[0].failure_reason


def test_objectstate_reality_gate_rejects_non_boolean_raw_identity_flag():
    metrics = dict(_identity_row().metrics)
    metrics["raw_prediction_observations"] = 1.0
    with pytest.raises(
        TypeError,
        match="metric raw_prediction_observations must be bool",
    ):
        evaluate_objectstate_reality_gate(
            (_identity_row(metrics=metrics),),
            synthetic_smoke_passed=True,
        )


def test_objectstate_reality_gate_recomputes_forged_derived_metrics():
    report = evaluate_objectstate_reality_gate(
        (
            _identity_row(),
            _prediction_row(
                metrics={
                    "state_ade": 0.03,
                    "history_ade": 0.05,
                    "prediction_gap_vs_history_model": 123.0,
                }
            ),
            _intervention_row(
                metrics={
                    "action_conditioned_ade": 0.03,
                    "no_action_ade": 0.10,
                    "intervention_gain": -50.0,
                    "counterfactual_outcome_accuracy": 1.0,
                    "wrong_direction_rate": 0.0,
                }
            ),
        ),
        synthetic_smoke_passed=True,
    )
    payload = report.as_dict()

    prediction = payload["pass_rows"][1]
    intervention = payload["pass_rows"][2]
    assert prediction["metrics"]["prediction_gap_vs_history_model"] == pytest.approx(
        -0.02
    )
    assert intervention["metrics"]["intervention_gain"] == pytest.approx(0.07)
    assert payload["declaration_diagnostics"]["derived_metric_mismatch_count"] == 2


def test_objectstate_reality_gate_equal_history_baseline_is_fail():
    report = evaluate_objectstate_reality_gate(
        (
            _prediction_row(
                metrics={
                    "state_ade": 0.0,
                    "history_ade": 0.0,
                    "prediction_gap_vs_history_model": -1.0,
                }
            ),
        ),
        synthetic_smoke_passed=True,
        thresholds=ObjectStateRealityGateThresholds(
            require_identity_pass_row=False,
            require_intervention_pass_row=False,
        ),
    )

    assert report.rows[0].status == "fail"
    assert report.rows[0].metrics["prediction_gap_vs_history_model"] == 0.0
    assert "state_does_not_strictly_beat_history" in report.rows[0].failure_reason


def test_objectstate_reality_gate_requires_positive_intervention_gain():
    report = evaluate_objectstate_reality_gate(
        (
            _intervention_row(
                metrics={
                    "action_conditioned_ade": 0.03,
                    "no_action_ade": 0.03,
                    "intervention_gain": 10.0,
                    "counterfactual_outcome_accuracy": 1.0,
                    "wrong_direction_rate": 0.0,
                }
            ),
        ),
        synthetic_smoke_passed=True,
        thresholds=ObjectStateRealityGateThresholds(
            require_identity_pass_row=False,
            require_prediction_pass_row=False,
        ),
    )

    assert report.rows[0].status == "fail"
    assert report.rows[0].metrics["intervention_gain"] == 0.0
    assert "intervention_gain_not_positive" in report.rows[0].failure_reason


def test_objectstate_reality_gate_rejects_non_blocked_missing_primitive_metric():
    with pytest.raises(ValueError, match="prediction reality row missing metric history_ade"):
        evaluate_objectstate_reality_gate(
            (
                _prediction_row(
                    metrics={"state_ade": 0.01},
                ),
            ),
            synthetic_smoke_passed=True,
        )

    with pytest.raises(
        ValueError,
        match="intervention reality row missing metric no_action_ade",
    ):
        evaluate_objectstate_reality_gate(
            (
                _intervention_row(
                    metrics={
                        "action_conditioned_ade": 0.01,
                        "counterfactual_outcome_accuracy": 1.0,
                        "wrong_direction_rate": 0.0,
                    },
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
            "raw_prediction_observations": True,
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
            "state_ade": 0.03,
            "history_ade": 0.05,
            "prediction_gap_vs_history_model": -0.02,
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
            "no_action_ade": 0.1,
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
