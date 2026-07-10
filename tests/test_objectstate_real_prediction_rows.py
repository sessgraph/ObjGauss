from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_real_evidence_bundle import (
    OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
)
from objgauss.evaluation.objectstate_real_prediction_rows import (
    OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA,
    objectstate_real_prediction_rows_from_bundle,
    objectstate_real_prediction_rows_summary,
    validate_objectstate_real_prediction_rows_summary,
)


def test_real_prediction_rows_enter_prediction_only_reality_gate():
    summary = objectstate_real_prediction_rows_summary(_prediction_bundle())

    assert summary["schema"] == OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA
    assert summary["status"] == "objectstate_real_prediction_rows_pass"
    assert summary["row_counts"]["prediction_rows"] == 1
    assert summary["row_counts"]["prediction_pass_rows"] == 1
    assert summary["row_counts"]["prediction_fail_rows"] == 0
    assert summary["row_counts"]["prediction_blocked_rows"] == 0
    assert summary["row_counts"]["state_transition_rows"] == 1
    assert summary["row_counts"]["referenced_transition_rows"] == 1
    assert summary["metrics"]["pose_transition_coverage"] == 1.0
    assert summary["metrics"]["mean_state_ade"] == 0.02
    assert summary["metrics"]["mean_history_ade"] == 0.04
    assert summary["metrics"]["mean_state_vs_history_error_ratio"] == 0.5
    assert summary["metrics"]["mean_prediction_gap_vs_history_model"] == -0.02
    assert summary["prediction_rows"][0]["status"] == "pass"
    assert summary["prediction_rows"][0]["metrics"]["state_ade"] == 0.02
    assert summary["prediction_rows"][0]["metrics"]["prediction_gap_vs_history_model"] == -0.02
    assert summary["prediction_gate"]["status"] == "objectstate_reality_gate_pass"
    assert summary["prediction_gate"]["thresholds"]["require_identity_pass_row"] is False
    assert summary["prediction_gate"]["thresholds"]["require_intervention_pass_row"] is False
    assert summary["claim_policy"]["history_baseline_comparison_required"] is True
    assert validate_objectstate_real_prediction_rows_summary(summary) == summary

    rows = objectstate_real_prediction_rows_from_bundle(_prediction_bundle())
    assert len(rows) == 1
    assert rows[0].status == "pass"


def test_real_prediction_rows_keep_evidence_incomplete_as_blocked_not_fail():
    bundle = _prediction_bundle(
        {
            "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
            "row_id": "cup-prediction-incomplete-001",
            "evidence_kind": "prediction",
            "accounting_status": "evidence_incomplete",
            "object_id": "cup-001",
            "metrics": {},
            "artifact_refs": ["outputs/captures/cup/prediction.csv"],
            "gt_requirements": {
                "identity": True,
                "pose": True,
                "action": False,
                "timestamp": True,
            },
            "reason": "missing candidate future-pose predictions",
        }
    )

    summary = objectstate_real_prediction_rows_summary(bundle)

    assert summary["status"] == "objectstate_real_prediction_rows_incomplete"
    assert summary["row_counts"]["prediction_pass_rows"] == 0
    assert summary["row_counts"]["prediction_fail_rows"] == 0
    assert summary["row_counts"]["prediction_blocked_rows"] == 1
    assert summary["metrics"]["prediction_accounting_status_counts"][
        "evidence_incomplete"
    ] == 1
    assert summary["prediction_rows"][0]["status"] == "blocked"
    assert "evidence_incomplete" in summary["prediction_rows"][0]["block_reason"]
    assert summary["prediction_gate"]["status"] == "objectstate_reality_gate_fail"
    assert "prediction_pass_rows_present" in summary["prediction_gate"]["hard_blockers"]


def test_real_prediction_rows_fail_explicit_prediction_fail_accounting():
    bundle = _prediction_bundle(
        {
            "schema": "objgauss-objectstate-real-gate-accounting-row-v1",
            "row_id": "cup-prediction-fail-001",
            "evidence_kind": "prediction",
            "accounting_status": "fail",
            "object_id": "cup-001",
            "transition_id": "transition-cup-000-001",
            "metrics": {
                "state_ade": 0.08,
                "history_ade": 0.04,
                "state_vs_history_error_ratio": 2.0,
            },
            "artifact_refs": ["outputs/captures/cup/prediction-eval.json"],
            "gt_requirements": {
                "identity": True,
                "pose": True,
                "action": False,
                "timestamp": True,
            },
            "reason": "objectstate predictor worse than history baseline",
        }
    )

    summary = objectstate_real_prediction_rows_summary(bundle)

    assert summary["status"] == "objectstate_real_prediction_rows_fail"
    assert summary["row_counts"]["prediction_fail_rows"] == 1
    assert summary["prediction_rows"][0]["status"] == "fail"
    assert summary["prediction_rows"][0]["failure_reason"].startswith(
        "derived prediction gate failed:"
    )
    assert "state_does_not_strictly_beat_history" in summary["prediction_rows"][0][
        "failure_reason"
    ]
    assert summary["prediction_rows"][0]["metrics"]["prediction_gap_vs_history_model"] == 0.04
    assert summary["prediction_gate"]["metrics"][
        "short_horizon_prediction_gap_vs_history_model"
    ] == 0.04
    assert "failed_rows_absent" in summary["prediction_gate"]["hard_blockers"]


def test_real_prediction_rows_recompute_forged_pass_and_gap():
    bundle = _prediction_bundle()
    bundle["gate_accounting_rows"][0]["metrics"].update(
        {
            "state_ade": 0.08,
            "history_ade": 0.04,
            "prediction_gap_vs_history_model": -999.0,
        }
    )

    summary = objectstate_real_prediction_rows_summary(bundle)

    assert summary["status"] == "objectstate_real_prediction_rows_fail"
    assert summary["row_counts"]["prediction_pass_rows"] == 0
    assert summary["row_counts"]["prediction_fail_rows"] == 1
    assert summary["metrics"]["reality_status_counts"] == {"fail": 1}
    assert summary["metrics"]["mean_prediction_gap_vs_history_model"] == 0.04
    assert summary["prediction_rows"] == summary["prediction_gate"]["rows"]
    assert summary["prediction_rows"][0]["status"] == "fail"
    assert (
        summary["prediction_rows"][0]["metrics"][
            "prediction_gap_vs_history_model"
        ]
        == 0.04
    )
    diagnostics = summary["prediction_gate"]["declaration_diagnostics"]
    assert diagnostics["caller_status_mismatch_count"] == 1
    assert diagnostics["derived_metric_mismatch_count"] == 1


def test_real_prediction_rows_require_transition_for_pass_fail_accounting():
    bundle = _prediction_bundle()
    del bundle["gate_accounting_rows"][0]["transition_id"]

    with pytest.raises(ValueError, match="require transition_id"):
        objectstate_real_prediction_rows_summary(bundle)


def test_real_prediction_rows_summary_is_incomplete_without_prediction_accounting():
    bundle = _prediction_bundle()
    bundle["gate_accounting_rows"] = []

    summary = objectstate_real_prediction_rows_summary(bundle)

    assert summary["status"] == "objectstate_real_prediction_rows_incomplete"
    assert summary["prediction_rows"] == []
    assert summary["prediction_gate"] is None
    assert "missing prediction accounting rows" in summary["hard_blockers"]


def test_real_prediction_rows_cli_writes_summary_rows_and_blocked_markdown(tmp_path, capsys):
    bundle_path = tmp_path / "bundle.json"
    summary_path = tmp_path / "prediction-summary.json"
    rows_path = tmp_path / "prediction-rows.json"
    blocked_path = tmp_path / "blocked.md"
    bundle_path.write_text(json.dumps(_prediction_bundle()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "real-prediction-rows",
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

    assert f"schema={OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA}" in stdout
    assert "prediction_gate_status=objectstate_reality_gate_pass" in stdout
    assert "prediction_pass_rows=1" in stdout
    assert "pose_transition_coverage=1.000000" in stdout
    assert summary["status"] == "objectstate_real_prediction_rows_pass"
    assert rows[0]["status"] == "pass"
    assert blocked_path.read_text(encoding="utf-8") == "No blocked real prediction rows.\n"


def _prediction_bundle(accounting_row=None):
    return {
        "schema": OBJECTSTATE_REAL_EVIDENCE_BUNDLE_SCHEMA,
        "kind": "objectstate_real_evidence_bundle",
        "sample": {
            "sample_id": "controlled-tabletop-cup-001",
            "scene_id": "tabletop-cup-box",
            "sequence_id": "prediction-push-001",
            "source_dataset": "local-controlled-tabletop",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "short_horizon_push_replay",
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
            _pose("pose-001", "f001", 0.1, [0.1, 0.0, 0.0]),
        ],
        "identity_link_rows": [
            _identity("identity-000", "f000", 0.0),
            _identity("identity-001", "f001", 0.1),
        ],
        "action_interval_rows": [],
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
                "row_id": "cup-prediction-pass-001",
                "evidence_kind": "prediction",
                "accounting_status": "pass",
                "object_id": "cup-001",
                "transition_id": "transition-cup-000-001",
                "metrics": {
                    "state_ade": 0.02,
                    "history_ade": 0.04,
                },
                "artifact_refs": ["outputs/captures/cup/prediction-eval.json"],
                "gt_requirements": {
                    "identity": True,
                    "pose": True,
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
