from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
)
from objgauss.core.objectstate_real_intervention_rows import (
    OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA,
    objectstate_real_intervention_rows_from_bundle,
    objectstate_real_intervention_rows_summary,
    validate_objectstate_real_intervention_rows_summary,
)


def test_real_intervention_rows_enter_intervention_only_reality_gate():
    summary = objectstate_real_intervention_rows_summary(_intervention_bundle())

    assert summary["schema"] == OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA
    assert summary["status"] == "objectstate_real_intervention_rows_pass"
    assert summary["row_counts"]["intervention_rows"] == 1
    assert summary["row_counts"]["intervention_pass_rows"] == 1
    assert summary["row_counts"]["intervention_fail_rows"] == 0
    assert summary["row_counts"]["intervention_blocked_rows"] == 0
    assert summary["row_counts"]["referenced_action_transition_pairs"] == 1
    assert summary["metrics"]["action_transition_coverage_rate"] == 1.0
    assert summary["metrics"]["mean_action_conditioned_ade"] == 0.01
    assert summary["metrics"]["mean_no_action_ade"] == 0.12
    assert summary["metrics"]["mean_intervention_gain"] == 0.11
    assert summary["metrics"]["mean_counterfactual_outcome_accuracy"] == 1.0
    assert summary["metrics"]["mean_wrong_direction_rate"] == 0.0
    assert summary["metrics"]["mean_identity_consistency_rate"] == 1.0
    assert summary["intervention_rows"][0]["status"] == "pass"
    assert summary["intervention_rows"][0]["metrics"]["intervention_gain"] == 0.11
    assert summary["intervention_gate"]["status"] == "objectstate_reality_gate_pass"
    assert summary["intervention_gate"]["thresholds"]["require_identity_pass_row"] is False
    assert summary["intervention_gate"]["thresholds"]["require_prediction_pass_row"] is False
    assert summary["claim_policy"]["action_transition_overlap_required"] is True
    assert summary["claim_policy"]["identity_stability_required_across_transition"] is True
    assert validate_objectstate_real_intervention_rows_summary(summary) == summary

    rows = objectstate_real_intervention_rows_from_bundle(_intervention_bundle())
    assert len(rows) == 1
    assert rows[0].status == "pass"


def test_real_intervention_rows_keep_evidence_incomplete_as_blocked_not_fail():
    bundle = _intervention_bundle(
        {
            "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
            "row_id": "cup-intervention-incomplete-001",
            "evidence_kind": "intervention",
            "accounting_status": "evidence_incomplete",
            "object_id": "cup-001",
            "metrics": {},
            "artifact_refs": ["outputs/captures/cup/intervention.csv"],
            "gt_requirements": {
                "identity": True,
                "pose": True,
                "action": True,
                "timestamp": True,
            },
            "reason": "missing action-conditioned candidate predictions",
        }
    )

    summary = objectstate_real_intervention_rows_summary(bundle)

    assert summary["status"] == "objectstate_real_intervention_rows_incomplete"
    assert summary["row_counts"]["intervention_pass_rows"] == 0
    assert summary["row_counts"]["intervention_fail_rows"] == 0
    assert summary["row_counts"]["intervention_blocked_rows"] == 1
    assert summary["metrics"]["intervention_accounting_status_counts"][
        "evidence_incomplete"
    ] == 1
    assert summary["intervention_rows"][0]["status"] == "blocked"
    assert "evidence_incomplete" in summary["intervention_rows"][0]["block_reason"]
    assert summary["intervention_gate"]["status"] == "objectstate_reality_gate_fail"
    assert "intervention_pass_rows_present" in summary["intervention_gate"]["hard_blockers"]


def test_real_intervention_rows_fail_explicit_intervention_fail_accounting():
    bundle = _intervention_bundle(
        {
            "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
            "row_id": "cup-intervention-fail-001",
            "evidence_kind": "intervention",
            "accounting_status": "fail",
            "object_id": "cup-001",
            "action_id": "push-left-001",
            "transition_id": "transition-cup-000-001",
            "metrics": {
                "action_conditioned_ade": 0.18,
                "no_action_ade": 0.12,
                "counterfactual_outcome_accuracy": 0.0,
                "wrong_direction_rate": 1.0,
                "identity_consistency_rate": 1.0,
            },
            "artifact_refs": ["outputs/captures/cup/intervention-eval.json"],
            "gt_requirements": {
                "identity": True,
                "pose": True,
                "action": True,
                "timestamp": True,
            },
            "reason": "action-conditioned predictor moved in the wrong direction",
        }
    )

    summary = objectstate_real_intervention_rows_summary(bundle)

    assert summary["status"] == "objectstate_real_intervention_rows_fail"
    assert summary["row_counts"]["intervention_fail_rows"] == 1
    assert summary["intervention_rows"][0]["status"] == "fail"
    assert summary["intervention_rows"][0]["failure_reason"] == (
        "action-conditioned predictor moved in the wrong direction"
    )
    assert summary["intervention_rows"][0]["metrics"]["intervention_gain"] == -0.06
    assert summary["intervention_gate"]["metrics"][
        "intervention_counterfactual_outcome_accuracy"
    ] == 0.0
    assert "failed_rows_absent" in summary["intervention_gate"]["hard_blockers"]


def test_real_intervention_rows_require_stable_identity_across_transition():
    bundle = _intervention_bundle()
    bundle["identity_link_rows"][1]["physical_identity_id"] = "cup-physical-002"

    with pytest.raises(ValueError, match="stable identity across transition"):
        objectstate_real_intervention_rows_summary(bundle)


def test_real_intervention_rows_summary_is_incomplete_without_intervention_accounting():
    bundle = _intervention_bundle()
    bundle["gate_accounting_rows"] = []

    summary = objectstate_real_intervention_rows_summary(bundle)

    assert summary["status"] == "objectstate_real_intervention_rows_incomplete"
    assert summary["intervention_rows"] == []
    assert summary["intervention_gate"] is None
    assert "missing intervention accounting rows" in summary["hard_blockers"]


def test_real_intervention_rows_cli_writes_summary_rows_and_blocked_markdown(tmp_path, capsys):
    bundle_path = tmp_path / "bundle.json"
    summary_path = tmp_path / "intervention-summary.json"
    rows_path = tmp_path / "intervention-rows.json"
    blocked_path = tmp_path / "blocked.md"
    bundle_path.write_text(json.dumps(_intervention_bundle()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "real-intervention-rows",
                str(bundle_path),
                "--summary-output",
                str(summary_path),
                "--rows-output",
                str(rows_path),
                "--blocked-rows-output",
                str(blocked_path),
                "--require-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    rows = json.loads(rows_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA}" in stdout
    assert "intervention_gate_status=objectstate_reality_gate_pass" in stdout
    assert "intervention_pass_rows=1" in stdout
    assert "action_transition_coverage_rate=1.000000" in stdout
    assert summary["status"] == "objectstate_real_intervention_rows_pass"
    assert rows[0]["status"] == "pass"
    assert blocked_path.read_text(encoding="utf-8") == "No blocked real intervention rows.\n"


def _intervention_bundle(accounting_row=None):
    return {
        "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "kind": "objectstate_real_evidence_bundle",
        "sample": {
            "sample_id": "controlled-tabletop-cup-001",
            "scene_id": "tabletop-cup-box",
            "sequence_id": "push-left-001",
            "source_dataset": "local-controlled-tabletop",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "push_left_counterfactual",
            "gt_provenance": "external-motion-capture",
            "license": "local-research",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": ["outputs/captures/cup/capture-manifest.json"],
        },
        "row_schemas": {
            "observation": "objgauss-objectstate-real-observation-row-v1",
            "object_pose": "objgauss-objectstate-real-object-pose-row-v1",
            "identity_link": "objgauss-objectstate-real-identity-link-row-v1",
            "action_interval": "objgauss-objectstate-real-action-interval-row-v1",
            "state_transition": "objgauss-objectstate-real-state-transition-row-v1",
            "gate_accounting": "objgauss-objectstate-real-gate-accounting-row-v1",
        },
        "observation_rows": [
            _observation("obs-000", "f000", 0.0),
            _observation("obs-001", "f001", 0.1),
        ],
        "object_pose_rows": [
            _pose("pose-000", "f000", 0.0, [0.0, 0.0, 0.0]),
            _pose("pose-001", "f001", 0.1, [-0.1, 0.0, 0.0]),
        ],
        "identity_link_rows": [
            _identity("identity-000", "f000", 0.0),
            _identity("identity-001", "f001", 0.1),
        ],
        "action_interval_rows": [
            {
                "schema": "objgauss-objectstate-real-action-interval-row-v1",
                "row_id": "action-row-001",
                "action_id": "push-left-001",
                "action_type": "push_left",
                "object_id": "cup-001",
                "action_start_ts": 0.02,
                "action_end_ts": 0.08,
                "action_vector": [-0.1, 0.0, 0.0],
                "actor": "hand-001",
                "gt_provenance": "external-motion-capture",
            }
        ],
        "state_transition_rows": [
            {
                "schema": "objgauss-objectstate-real-state-transition-row-v1",
                "row_id": "transition-row-001",
                "transition_id": "transition-cup-000-001",
                "object_id": "cup-001",
                "source_frame_id": "f000",
                "target_frame_id": "f001",
                "source_timestamp": 0.0,
                "target_timestamp": 0.1,
                "source_pose_row_id": "pose-000",
                "target_pose_row_id": "pose-001",
                "gt_provenance": "external-motion-capture",
            }
        ],
        "gate_accounting_rows": [
            accounting_row
            or {
                "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
                "row_id": "cup-intervention-pass-001",
                "evidence_kind": "intervention",
                "accounting_status": "pass",
                "object_id": "cup-001",
                "action_id": "push-left-001",
                "transition_id": "transition-cup-000-001",
                "metrics": {
                    "action_conditioned_ade": 0.01,
                    "no_action_ade": 0.12,
                    "counterfactual_outcome_accuracy": 1.0,
                    "wrong_direction_rate": 0.0,
                    "identity_consistency_rate": 1.0,
                },
                "artifact_refs": ["outputs/captures/cup/intervention-eval.json"],
                "gt_requirements": {
                    "identity": True,
                    "pose": True,
                    "action": True,
                    "timestamp": True,
                },
            }
        ],
    }


def _observation(row_id, frame_id, timestamp):
    return {
        "schema": "objgauss-objectstate-real-observation-row-v1",
        "row_id": row_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "camera_id": "cam-001",
        "observation": {
            "rgb": f"rgb/{frame_id}.png",
            "gaussian": f"gaussians/{frame_id}.ply",
        },
    }


def _pose(row_id, frame_id, timestamp, position):
    return {
        "schema": "objgauss-objectstate-real-object-pose-row-v1",
        "row_id": row_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "camera_id": "cam-001",
        "object_id": "cup-001",
        "object_pose_6dof": {
            "position": position,
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
        "object_visibility": 1.0,
    }


def _identity(row_id, frame_id, timestamp):
    return {
        "schema": "objgauss-objectstate-real-identity-link-row-v1",
        "row_id": row_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "object_id": "cup-001",
        "physical_identity_id": "cup-physical-001",
        "gt_provenance": "external-motion-capture",
        "confidence": 1.0,
    }
