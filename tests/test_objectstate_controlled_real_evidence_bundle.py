from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_real_evidence_bundle import (
    OBJECTSTATE_CONTROLLED_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA,
    objectstate_controlled_real_evidence_bundle_adapter_summary,
    validate_objectstate_controlled_real_evidence_bundle_adapter_summary,
)


def test_controlled_real_evidence_bundle_adapter_maps_capture_rows():
    summary = objectstate_controlled_real_evidence_bundle_adapter_summary(
        _capture_manifest(),
        source_summary_ref="outputs/captures/cup/capture-manifest.json",
    )
    bundle = summary["bundle"]
    bundle_summary = summary["bundle_summary"]

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA
    assert summary["status"] == "objectstate_controlled_real_evidence_bundle_adapter_ready"
    assert summary["row_counts"] == {
        "observation_rows": 2,
        "object_pose_rows": 2,
        "identity_link_rows": 2,
        "action_interval_rows": 1,
        "state_transition_rows": 1,
        "gate_accounting_rows": 3,
    }
    assert summary["accounting_status_counts"] == {
        "pass": 0,
        "fail": 0,
        "evidence_incomplete": 3,
        "unsupported": 0,
    }
    assert summary["readiness"]["default_accounting_is_evidence_incomplete"] is True
    assert summary["readiness"]["intervention_pass_not_created_without_metrics"] is True
    assert bundle["sample"]["source_kind"] == "controlled_real"
    assert bundle["gate_accounting_rows"][1]["transition_id"].startswith(
        "controlled-transition:"
    )
    assert bundle["gate_accounting_rows"][2]["action_id"] == "push-left-001"
    assert bundle_summary["readiness"]["state_variable_evidence_ready"] is True
    assert bundle_summary["readiness"]["intervention_accounting_ready"] is False
    assert bundle_summary["metrics"]["action_transition_coverage_rate"] == 1.0
    assert validate_objectstate_controlled_real_evidence_bundle_adapter_summary(
        summary
    ) == summary


def test_controlled_real_evidence_bundle_adapter_keeps_missing_pose_incomplete():
    manifest = _capture_manifest()
    for frame in manifest["frames"]:
        for item in frame["objects"]:
            item.pop("pose")

    summary = objectstate_controlled_real_evidence_bundle_adapter_summary(manifest)

    assert summary["status"] == (
        "objectstate_controlled_real_evidence_bundle_adapter_incomplete"
    )
    assert summary["row_counts"]["object_pose_rows"] == 0
    assert summary["bundle_summary"]["readiness"]["state_variable_evidence_ready"] is False
    assert "missing object pose rows" in summary["issues"]
    assert summary["accounting_status_counts"]["evidence_incomplete"] == 3


def test_controlled_real_evidence_bundle_cli_writes_bundle_and_summary(tmp_path, capsys):
    capture_path = tmp_path / "capture-manifest.json"
    bundle_path = tmp_path / "real-evidence-bundle.json"
    summary_path = tmp_path / "controlled-real-bundle-summary.json"
    capture_path.write_text(json.dumps(_capture_manifest()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "controlled-real-evidence-bundle",
                str(capture_path),
                "--bundle-output",
                str(bundle_path),
                "--summary-output",
                str(summary_path),
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA}" in stdout
    assert "accounting.evidence_incomplete=3" in stdout
    assert "readiness.real_bundle_ready=true" in stdout
    assert summary["bundle"]["schema"] == "objgauss-objectstate-real-evidence-bundle-v1"
    assert bundle["gate_accounting_rows"][0]["accounting_status"] == (
        "evidence_incomplete"
    )


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
        "objects": [
            {
                "object_id": "cup-001",
                "category": "cup",
                "instance_label": "red cup",
            }
        ],
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
