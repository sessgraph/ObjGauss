from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
)
from objgauss.core.objectstate_real_identity_rows import (
    OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA,
    objectstate_real_identity_rows_from_bundle,
    objectstate_real_identity_rows_summary,
    validate_objectstate_real_identity_rows_summary,
)


def test_real_identity_rows_enter_identity_only_reality_gate():
    summary = objectstate_real_identity_rows_summary(_identity_bundle())

    assert summary["schema"] == OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA
    assert summary["status"] == "objectstate_real_identity_rows_pass"
    assert summary["row_counts"]["identity_rows"] == 1
    assert summary["row_counts"]["identity_pass_rows"] == 1
    assert summary["row_counts"]["identity_fail_rows"] == 0
    assert summary["row_counts"]["identity_blocked_rows"] == 0
    assert summary["metrics"]["physical_identity_count"] == 1
    assert summary["metrics"]["identity_frame_count"] == 2
    assert summary["identity_rows"][0]["status"] == "pass"
    assert summary["identity_rows"][0]["metrics"]["idf1"] == 1.0
    assert summary["identity_rows"][0]["metrics"]["identity_link_count"] == 2.0
    assert summary["identity_gate"]["status"] == "objectstate_reality_gate_pass"
    assert summary["identity_gate"]["thresholds"]["require_prediction_pass_row"] is False
    assert summary["identity_gate"]["thresholds"]["require_intervention_pass_row"] is False
    assert summary["claim_policy"]["prediction_rows_out_of_scope"] is True
    assert validate_objectstate_real_identity_rows_summary(summary) == summary

    rows = objectstate_real_identity_rows_from_bundle(_identity_bundle())
    assert len(rows) == 1
    assert rows[0].status == "pass"


def test_real_identity_rows_keep_evidence_incomplete_as_blocked_not_fail():
    bundle = _identity_bundle(
        {
            "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
            "row_id": "cup-identity-incomplete-001",
            "evidence_kind": "identity",
            "accounting_status": "evidence_incomplete",
            "object_id": "cup-001",
            "metrics": {},
            "artifact_refs": ["outputs/captures/cup/identity.csv"],
            "gt_requirements": {
                "identity": True,
                "pose": False,
                "action": False,
                "timestamp": True,
            },
            "reason": "missing candidate identity metrics",
        }
    )

    summary = objectstate_real_identity_rows_summary(bundle)

    assert summary["status"] == "objectstate_real_identity_rows_incomplete"
    assert summary["row_counts"]["identity_pass_rows"] == 0
    assert summary["row_counts"]["identity_fail_rows"] == 0
    assert summary["row_counts"]["identity_blocked_rows"] == 1
    assert summary["metrics"]["identity_accounting_status_counts"][
        "evidence_incomplete"
    ] == 1
    assert summary["identity_rows"][0]["status"] == "blocked"
    assert "evidence_incomplete" in summary["identity_rows"][0]["block_reason"]
    assert summary["identity_gate"]["status"] == "objectstate_reality_gate_fail"
    assert "identity_pass_rows_present" in summary["identity_gate"]["hard_blockers"]


def test_real_identity_rows_fail_explicit_identity_fail_accounting():
    bundle = _identity_bundle(
        {
            "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
            "row_id": "cup-identity-fail-001",
            "evidence_kind": "identity",
            "accounting_status": "fail",
            "object_id": "cup-001",
            "metrics": {
                "idf1": 0.2,
                "fragmentation_rate": 0.8,
                "swap_rate": 0.1,
                "identity_collapse": True,
            },
            "artifact_refs": ["outputs/captures/cup/identity-eval.json"],
            "gt_requirements": {
                "identity": True,
                "pose": False,
                "action": False,
                "timestamp": True,
            },
            "reason": "identity collapsed after occlusion",
        }
    )

    summary = objectstate_real_identity_rows_summary(bundle)

    assert summary["status"] == "objectstate_real_identity_rows_fail"
    assert summary["row_counts"]["identity_fail_rows"] == 1
    assert summary["identity_rows"][0]["status"] == "fail"
    assert summary["identity_rows"][0]["failure_reason"] == (
        "identity collapsed after occlusion"
    )
    assert summary["identity_gate"]["metrics"]["controlled_real_identity_collapse"] is True
    assert "failed_rows_absent" in summary["identity_gate"]["hard_blockers"]


def test_real_identity_rows_summary_is_incomplete_without_identity_accounting():
    bundle = _identity_bundle()
    bundle["gate_accounting_rows"] = []

    summary = objectstate_real_identity_rows_summary(bundle)

    assert summary["status"] == "objectstate_real_identity_rows_incomplete"
    assert summary["identity_rows"] == []
    assert summary["identity_gate"] is None
    assert "missing identity accounting rows" in summary["hard_blockers"]


def test_real_identity_rows_cli_writes_summary_rows_and_blocked_markdown(tmp_path, capsys):
    bundle_path = tmp_path / "bundle.json"
    summary_path = tmp_path / "identity-summary.json"
    rows_path = tmp_path / "identity-rows.json"
    blocked_path = tmp_path / "blocked.md"
    bundle_path.write_text(json.dumps(_identity_bundle()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "real-identity-rows",
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

    assert f"schema={OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA}" in stdout
    assert "identity_gate_status=objectstate_reality_gate_pass" in stdout
    assert "identity_pass_rows=1" in stdout
    assert summary["status"] == "objectstate_real_identity_rows_pass"
    assert rows[0]["status"] == "pass"
    assert blocked_path.read_text(encoding="utf-8") == "No blocked real identity rows.\n"


def _identity_bundle(accounting_row=None):
    return {
        "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "kind": "objectstate_real_evidence_bundle",
        "sample": {
            "sample_id": "controlled-tabletop-cup-001",
            "scene_id": "tabletop-cup-box",
            "sequence_id": "identity-occlusion-001",
            "source_dataset": "local-controlled-tabletop",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "occlusion_reappearance",
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
            _pose("pose-000", "f000", 0.0),
            _pose("pose-001", "f001", 0.1),
        ],
        "identity_link_rows": [
            _identity("identity-000", "f000", 0.0),
            _identity("identity-001", "f001", 0.1),
        ],
        "action_interval_rows": [],
        "state_transition_rows": [],
        "gate_accounting_rows": [
            accounting_row
            or {
                "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
                "row_id": "cup-identity-pass-001",
                "evidence_kind": "identity",
                "accounting_status": "pass",
                "object_id": "cup-001",
                "metrics": {
                    "idf1": 1.0,
                    "fragmentation_rate": 0.0,
                    "swap_rate": 0.0,
                    "identity_collapse": False,
                },
                "artifact_refs": ["outputs/captures/cup/identity-eval.json"],
                "gt_requirements": {
                    "identity": True,
                    "pose": False,
                    "action": False,
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


def _pose(row_id, frame_id, timestamp):
    return {
        "schema": "objgauss-objectstate-real-object-pose-row-v1",
        "row_id": row_id,
        "frame_id": frame_id,
        "timestamp": timestamp,
        "camera_id": "cam-001",
        "object_id": "cup-001",
        "object_pose_6dof": {
            "position": [0.0, 0.0, 0.0],
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
