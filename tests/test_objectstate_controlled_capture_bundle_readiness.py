from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_capture_bundle_readiness import (
    OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA,
    objectstate_controlled_capture_bundle_readiness,
    validate_objectstate_controlled_capture_bundle_readiness_summary,
)
from objgauss.datasets.objectstate_controlled_capture_template import (
    write_objectstate_controlled_capture_bundle_template,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n"
PLY_BYTES = (
    b"ply\n"
    b"format ascii 1.0\n"
    b"element vertex 1\n"
    b"property float x\n"
    b"property float y\n"
    b"property float z\n"
    b"end_header\n"
    b"0 0 0\n"
)


def test_controlled_capture_bundle_readiness_reports_skeleton_blockers(tmp_path):
    write_objectstate_controlled_capture_bundle_template(
        tmp_path,
        sample_id="controlled-tabletop-cup-box-001",
    )

    summary = objectstate_controlled_capture_bundle_readiness(tmp_path)

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA
    assert summary["status"] == "objectstate_controlled_capture_bundle_readiness_blocked"
    assert summary["readiness"]["layout_ready"] is True
    assert summary["readiness"]["sample_metadata_ready"] is True
    assert summary["readiness"]["csv_headers_ready"] is True
    assert summary["readiness"]["object_rows_present"] is False
    assert summary["readiness"]["frame_rows_present"] is False
    assert summary["readiness"]["annotation_rows_present"] is False
    assert summary["readiness"]["capture_import_ready"] is False
    assert summary["readiness"]["capture_files_ready"] is False
    assert summary["readiness"]["capture_bundle_ready"] is False
    assert "objects.csv requires at least one physical object row" in summary["hard_blockers"]
    assert "capture file audit cannot run until capture import is ready" in summary["hard_blockers"]
    assert any("declare physical objects" in action for action in summary["next_actions"])
    assert validate_objectstate_controlled_capture_bundle_readiness_summary(summary) == summary


def test_controlled_capture_bundle_readiness_passes_ready_capture_with_candidate(
    tmp_path,
):
    _write_ready_bundle(tmp_path)
    candidate_path = tmp_path / "objectstates.json"
    candidate_path.write_text('{"schema":"fixture"}\n', encoding="utf-8")

    summary = objectstate_controlled_capture_bundle_readiness(
        tmp_path,
        candidate_artifact=candidate_path,
        require_candidate_artifact=True,
        require_intervention_ready=True,
        hash_files=True,
    )

    assert summary["status"] == "objectstate_controlled_capture_bundle_readiness_ready"
    assert summary["row_counts"] == {
        "objects_csv": 2,
        "frames_csv": 3,
        "annotations_csv": 6,
        "actions_csv": 1,
    }
    assert summary["readiness"]["capture_bundle_ready"] is True
    assert summary["readiness"]["identity_bundle_handoff_ready"] is True
    assert summary["readiness"]["candidate_artifact_ready"] is True
    assert summary["readiness"]["intervention_stage_ready"] is True
    assert summary["readiness"]["intervention_action_gt_ready"] is True
    assert summary["intervention_action_gt"]["metrics"] == {
        "action_count": 1,
        "nonzero_vector_action_count": 1,
        "usable_action_transition_count": 1,
    }
    assert summary["intervention_action_gt"]["usable_action_ids"] == ["push-left-001"]
    assert summary["identity_scenario"]["ready"] is True
    assert (
        summary["capture_file_audit"]["status"]
        == "objectstate_controlled_capture_file_audit_pass"
    )
    assert len(summary["capture_file_audit"]["file_records"]["rgb"][0]["sha256"]) == 64
    assert summary["hard_blockers"] == []
    assert summary["import_summary"]["capture_summary"]["readiness"] == {
        "identity_stage_ready": True,
        "prediction_stage_ready": True,
        "intervention_stage_ready": True,
        "real_gaussian_reconstruction_present": True,
    }


def test_object_state_audit_controlled_capture_bundle_readiness_cli(
    tmp_path,
    capsys,
):
    _write_ready_bundle(tmp_path)
    candidate_path = tmp_path / "objectstates.json"
    candidate_path.write_text('{"schema":"fixture"}\n', encoding="utf-8")
    summary_path = tmp_path / "readiness-summary.json"

    assert (
        main(
            [
                "object-state",
                "audit-controlled-capture-bundle-readiness",
                str(tmp_path),
                "--candidate-artifact",
                str(candidate_path),
                "--require-candidate-artifact",
                "--require-intervention-ready",
                "--hash-files",
                "--summary-output",
                str(summary_path),
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA}" in stdout
    assert "readiness_status=objectstate_controlled_capture_bundle_readiness_ready" in stdout
    assert "capture_bundle_ready=true" in stdout
    assert "identity_bundle_handoff_ready=true" in stdout
    assert "intervention_action_gt_ready=true" in stdout
    assert "candidate_artifact_ready=true" in stdout
    assert "hard_blockers=0" in stdout
    assert summary["status"] == "objectstate_controlled_capture_bundle_readiness_ready"


def test_controlled_capture_bundle_readiness_blocks_weak_action_gt(tmp_path):
    _write_ready_bundle(tmp_path)
    (tmp_path / "actions.csv").write_text(
        "\n".join(
            (
                "action_id,action_type,object_id,start_timestamp,end_timestamp,actor,target_object_id,vector_x,vector_y,vector_z",
                "push-left-001,push_left,cup-001,0.033333,0.066667,scripted-hand,,,,",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    candidate_path = tmp_path / "objectstates.json"
    candidate_path.write_text('{"schema":"fixture"}\n', encoding="utf-8")

    summary = objectstate_controlled_capture_bundle_readiness(
        tmp_path,
        candidate_artifact=candidate_path,
        require_candidate_artifact=True,
        require_intervention_ready=True,
    )

    assert summary["status"] == "objectstate_controlled_capture_bundle_readiness_blocked"
    assert summary["readiness"]["intervention_stage_ready"] is True
    assert summary["readiness"]["intervention_action_gt_ready"] is False
    assert summary["readiness"]["capture_bundle_ready"] is False
    assert summary["intervention_action_gt"]["metrics"] == {
        "action_count": 1,
        "nonzero_vector_action_count": 0,
        "usable_action_transition_count": 0,
    }
    assert any("requires a non-zero vector" in item for item in summary["hard_blockers"])
    assert any("non-zero vectors" in item for item in summary["next_actions"])


def _write_ready_bundle(root) -> None:
    write_objectstate_controlled_capture_bundle_template(
        root,
        sample_id="controlled-tabletop-cup-box-001",
        object_category="cup_box",
        scenario="cross_view_occlusion_reappearance",
        objects=[
            {"object_id": "cup-001", "category": "cup", "instance_label": "blue cup"},
            {"object_id": "box-001", "category": "box", "instance_label": "red box"},
        ],
    )
    (root / "frames.csv").write_text(
        "\n".join(
            (
                "frame_id,timestamp,rgb,gaussian,action_id,view_id,lighting_id,camera_x,camera_y,camera_z,camera_qx,camera_qy,camera_qz,camera_qw",
                "frame-000000,0.000000,rgb/000000.png,gaussians/000000.ply,,front,bright,0.00,0.0,0.0,0,0,0,1",
                "frame-000001,0.033333,rgb/000001.png,gaussians/000001.ply,push-left-001,front,dim,0.02,0.0,0.0,0,0,0,1",
                "frame-000002,0.066667,rgb/000002.png,gaussians/000002.ply,,right,dim,0.04,0.0,0.0,0,0,0,1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "annotations.csv").write_text(
        "\n".join(
            (
                "frame_id,object_id,visible,occlusion_fraction,x,y,z,qx,qy,qz,qw",
                "frame-000000,cup-001,true,0.0,0.10,0.20,0.30,0,0,0,1",
                "frame-000000,box-001,true,0.0,0.40,0.20,0.30,0,0,0,1",
                "frame-000001,cup-001,false,0.8,0.11,0.20,0.30,0,0,0,1",
                "frame-000001,box-001,true,0.0,0.41,0.20,0.30,0,0,0,1",
                "frame-000002,cup-001,true,0.0,0.12,0.20,0.30,0,0,0,1",
                "frame-000002,box-001,true,0.0,0.42,0.20,0.30,0,0,0,1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "actions.csv").write_text(
        "\n".join(
            (
                "action_id,action_type,object_id,start_timestamp,end_timestamp,actor,target_object_id,vector_x,vector_y,vector_z",
                "push-left-001,push_left,cup-001,0.033333,0.066667,scripted-hand,,-0.02,0.0,0.0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    for index in range(3):
        (root / "rgb" / f"{index:06d}.png").write_bytes(PNG_BYTES)
        (root / "gaussians" / f"{index:06d}.ply").write_bytes(PLY_BYTES)
