from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_controlled_real_evidence_bundle import (
    objectstate_controlled_real_evidence_bundle_adapter_summary,
)
from objgauss.core.objectstate_controlled_real_readiness_audit import (
    OBJECTSTATE_CONTROLLED_REAL_READINESS_AUDIT_SCHEMA,
    objectstate_controlled_real_readiness_audit,
    objectstate_controlled_real_readiness_breakdown_csv,
    objectstate_controlled_real_readiness_markdown,
    validate_objectstate_controlled_real_readiness_audit,
)


def test_controlled_real_readiness_audit_reports_ready_inputs_and_missing_metrics():
    bundle = _real_bundle()

    summary = objectstate_controlled_real_readiness_audit(bundle)

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REAL_READINESS_AUDIT_SCHEMA
    assert summary["status"] == "objectstate_controlled_real_readiness_ready"
    assert summary["row_count"] == 3
    assert summary["identity_ready_rows"] == 1
    assert summary["prediction_ready_rows"] == 1
    assert summary["intervention_ready_rows"] == 1
    assert summary["evidence_incomplete_rows"] == 3
    assert summary["blocked_rows"] == 3
    assert summary["blocked_reasons"]["missing_evaluator_metrics"] == 3
    assert summary["blocked_reasons"]["missing_action_vector"] == 0
    assert summary["readiness"]["all_gate_inputs_ready"] is True
    assert validate_objectstate_controlled_real_readiness_audit(summary) == summary

    markdown = objectstate_controlled_real_readiness_markdown(summary)
    csv_text = objectstate_controlled_real_readiness_breakdown_csv(summary)

    assert "# Controlled Real Readiness Audit" in markdown
    assert "controlled-real-accounting:controlled-tabletop-cup-001:identity" in csv_text
    assert "missing_evaluator_metrics" in csv_text


def test_controlled_real_readiness_audit_blocks_intervention_without_action():
    bundle = _real_bundle()
    bundle["action_interval_rows"] = []
    for row in bundle["gate_accounting_rows"]:
        if row["evidence_kind"] == "intervention":
            row.pop("action_id", None)
            row["reason"] = "missing controlled action interval"

    summary = objectstate_controlled_real_readiness_audit(bundle)

    assert summary["status"] == "objectstate_controlled_real_readiness_incomplete"
    assert summary["identity_ready_rows"] == 1
    assert summary["prediction_ready_rows"] == 1
    assert summary["intervention_ready_rows"] == 0
    assert summary["blocked_reasons"]["missing_action_vector"] == 1
    assert summary["blocked_reasons"]["action_interval_no_transition_overlap"] == 0
    assert summary["blocked_reasons"]["missing_evaluator_metrics"] == 2


def test_controlled_real_readiness_audit_cli_writes_outputs(tmp_path, capsys):
    bundle_path = tmp_path / "real-bundle.json"
    summary_path = tmp_path / "controlled-real-readiness-summary.json"
    report_path = tmp_path / "controlled-real-readiness-report.md"
    breakdown_path = tmp_path / "controlled-real-readiness-breakdown.csv"
    bundle_path.write_text(json.dumps(_real_bundle()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "audit-controlled-real-readiness",
                str(bundle_path),
                "--summary-output",
                str(summary_path),
                "--report-output",
                str(report_path),
                "--breakdown-output",
                str(breakdown_path),
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_REAL_READINESS_AUDIT_SCHEMA}" in stdout
    assert "identity_ready_rows=1" in stdout
    assert "blocked_reason.missing_evaluator_metrics=3" in stdout
    assert summary["status"] == "objectstate_controlled_real_readiness_ready"
    assert report_path.read_text(encoding="utf-8").startswith(
        "# Controlled Real Readiness Audit"
    )
    assert breakdown_path.read_text(encoding="utf-8").splitlines()[0] == (
        "row_id,evidence_kind,accounting_status,evaluator_ready,blocked,"
        "blocked_reasons,object_id,action_id,transition_id"
    )


def _real_bundle():
    return objectstate_controlled_real_evidence_bundle_adapter_summary(
        _capture_manifest(),
        source_summary_ref="outputs/captures/cup/capture-manifest.json",
    )["bundle"]


def _capture_manifest():
    return {
        "schema": "objgauss-objectstate-controlled-capture-manifest-v1",
        "sample": {
            "sample_id": "controlled-tabletop-cup-001",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "push_left",
            "fps": 30.0,
            "capture_device": "cam-001",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": ["outputs/captures/cup/capture-manifest.json"],
            "license": "local-research",
        },
        "objects": [{"object_id": "cup-001", "category": "cup"}],
        "actions": [
            {
                "action_id": "push-left-001",
                "action_type": "push_left",
                "object_id": "cup-001",
                "start_timestamp": 0.02,
                "end_timestamp": 0.08,
                "actor": "hand-001",
                "vector": [-0.1, 0.0, 0.0],
            }
        ],
        "frames": [
            _frame("f000", 0.0, [0.0, 0.0, 0.0], action_id="push-left-001"),
            _frame("f001", 0.1, [-0.1, 0.0, 0.0], action_id="push-left-001"),
        ],
    }


def _frame(frame_id, timestamp, position, *, action_id):
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "action_id": action_id,
        "observation": {
            "rgb": f"rgb/{frame_id}.png",
            "gaussian": f"gaussians/{frame_id}.ply",
        },
        "condition": {
            "view_id": "front",
            "lighting_id": "normal",
            "camera_pose": {
                "position": [0.0, 0.0, 1.0],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        "objects": [
            {
                "object_id": "cup-001",
                "visible": True,
                "occlusion_fraction": 0.0,
                "pose": {
                    "position": position,
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
    }
