from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA,
    objectstate_real_evidence_bundle_summary,
    read_objectstate_real_evidence_bundle,
    validate_objectstate_real_evidence_bundle,
    validate_objectstate_real_evidence_bundle_summary,
)


def test_real_evidence_bundle_summary_reports_state_variable_readiness(tmp_path):
    bundle = _real_bundle()

    checked = validate_objectstate_real_evidence_bundle(bundle)
    summary = objectstate_real_evidence_bundle_summary(bundle)

    assert checked["schema"] == OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA
    assert summary["schema"] == OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA
    assert summary["status"] == "objectstate_real_evidence_bundle_ready"
    assert summary["readiness"]["state_variable_evidence_ready"] is True
    assert summary["readiness"]["intervention_accounting_ready"] is True
    assert summary["metrics"]["observation_row_count"] == 2
    assert summary["metrics"]["object_pose_row_count"] == 2
    assert summary["metrics"]["identity_link_row_count"] == 2
    assert summary["metrics"]["action_interval_row_count"] == 1
    assert summary["metrics"]["state_transition_row_count"] == 1
    assert summary["metrics"]["gate_accounting_row_count"] == 3
    assert summary["metrics"]["action_transition_overlap_count"] == 1
    assert summary["metrics"]["action_transition_coverage_rate"] == 1.0
    assert summary["evidence_accounts"]["static_scene_evidence"] == {
        "available": True,
        "usable_for_state_variable_gate": False,
    }
    assert summary["evidence_accounts"]["state_variable_evidence"]["available"] is True
    assert summary["claim_policy"]["evidence_incomplete_is_not_model_fail"] is True
    assert summary["non_goals"]["creates_reality_rows"] is False
    assert validate_objectstate_real_evidence_bundle_summary(summary) == summary

    path = tmp_path / "real-evidence-bundle.json"
    path.write_text(json.dumps(bundle), encoding="utf-8")
    assert read_objectstate_real_evidence_bundle(path)["sample"]["sample_id"] == (
        "controlled-tabletop-cup-001"
    )


def test_real_evidence_bundle_rejects_intervention_accounting_without_overlap():
    bundle = _real_bundle()
    bundle["action_interval_rows"][0]["action_start_ts"] = 1.0
    bundle["action_interval_rows"][0]["action_end_ts"] = 1.1

    with pytest.raises(ValueError, match="action interval must overlap"):
        validate_objectstate_real_evidence_bundle(bundle)


def test_real_evidence_bundle_rejects_transition_wrong_pose_row_id():
    bundle = _real_bundle()
    bundle["state_transition_rows"][0]["source_pose_row_id"] = "pose-missing"

    with pytest.raises(ValueError, match="source must reference object pose row"):
        validate_objectstate_real_evidence_bundle(bundle)


def test_real_evidence_bundle_keeps_static_scene_separate_from_state_variable():
    bundle = _real_bundle()
    bundle["object_pose_rows"] = []
    bundle["identity_link_rows"] = []
    bundle["action_interval_rows"] = []
    bundle["state_transition_rows"] = []
    bundle["gate_accounting_rows"] = []

    summary = objectstate_real_evidence_bundle_summary(bundle)

    assert summary["status"] == "objectstate_real_evidence_bundle_incomplete"
    assert summary["readiness"]["observation_rows_present"] is True
    assert summary["readiness"]["state_variable_evidence_ready"] is False
    assert summary["readiness"]["intervention_accounting_ready"] is False
    assert summary["evidence_accounts"]["static_scene_evidence"]["available"] is True
    assert summary["evidence_accounts"]["state_variable_evidence"]["available"] is False
    assert "missing object pose rows" in summary["hard_blockers"]


def test_real_evidence_bundle_allows_incomplete_intervention_accounting():
    bundle = _real_bundle()
    bundle["action_interval_rows"] = []
    bundle["gate_accounting_rows"][-1] = {
        "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
        "row_id": "cup-intervention-incomplete-001",
        "evidence_kind": "intervention",
        "accounting_status": "evidence_incomplete",
        "metrics": {},
        "artifact_refs": ["outputs/captures/cup/actions.csv"],
        "gt_requirements": {
            "identity": True,
            "pose": True,
            "action": True,
            "timestamp": True,
        },
        "reason": "missing usable action interval",
    }

    summary = objectstate_real_evidence_bundle_summary(bundle)

    assert summary["status"] == "objectstate_real_evidence_bundle_ready"
    assert summary["metrics"]["gate_accounting_status_counts"][
        "evidence_incomplete"
    ] == 1
    assert summary["readiness"]["intervention_accounting_ready"] is False


def test_validate_real_evidence_bundle_cli(tmp_path, capsys):
    bundle_path = tmp_path / "real-evidence-bundle.json"
    summary_path = tmp_path / "real-evidence-summary.json"
    bundle_path.write_text(json.dumps(_real_bundle()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "validate-real-evidence-bundle",
                str(bundle_path),
                "--summary-output",
                str(summary_path),
                "--require-ready",
                "--require-intervention-accounting-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA}" in stdout
    assert "state_variable_evidence_ready=true" in stdout
    assert "intervention_accounting_ready=true" in stdout
    assert "action_transition_coverage_rate=1.000000" in stdout
    assert summary["schema"] == OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SUMMARY_SCHEMA


def _real_bundle():
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
            "scenario": "push_left",
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
            _accounting(
                "cup-identity-accounting-001",
                "identity",
                metrics={
                    "idf1": 1.0,
                    "fragmentation_rate": 0.0,
                    "swap_rate": 0.0,
                },
            ),
            _accounting(
                "cup-prediction-accounting-001",
                "prediction",
                transition_id="transition-cup-000-001",
                metrics={
                    "state_ade": 0.02,
                    "history_ade": 0.03,
                    "state_vs_history_error_ratio": 0.66,
                },
            ),
            _accounting(
                "cup-intervention-accounting-001",
                "intervention",
                action_id="push-left-001",
                transition_id="transition-cup-000-001",
                metrics={
                    "action_conditioned_ade": 0.01,
                    "no_action_ade": 0.12,
                    "intervention_gain": 0.11,
                    "counterfactual_outcome_accuracy": 1.0,
                    "wrong_direction_rate": 0.0,
                    "identity_consistency_rate": 1.0,
                },
            ),
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


def _accounting(
    row_id,
    evidence_kind,
    *,
    action_id=None,
    transition_id=None,
    metrics=None,
):
    row = {
        "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
        "row_id": row_id,
        "evidence_kind": evidence_kind,
        "accounting_status": "pass",
        "object_id": "cup-001",
        "metrics": metrics or {},
        "artifact_refs": ["outputs/captures/cup/reality-summary.json"],
        "gt_requirements": {
            "identity": True,
            "pose": evidence_kind in {"prediction", "intervention"},
            "action": evidence_kind == "intervention",
            "timestamp": True,
        },
    }
    if action_id is not None:
        row["action_id"] = action_id
    if transition_id is not None:
        row["transition_id"] = transition_id
    return row
