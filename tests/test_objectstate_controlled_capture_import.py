from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA,
    validate_objectstate_controlled_capture_manifest,
)
from objgauss.core.objectstate_controlled_capture_import import (
    OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_ACCEPTANCE_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA,
    objectstate_controlled_capture_bundle_acceptance_summary,
    objectstate_controlled_capture_import_summary,
    objectstate_controlled_capture_manifest_from_bundle,
    validate_objectstate_controlled_capture_bundle_acceptance_summary,
    validate_objectstate_controlled_capture_import_summary,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
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


def test_controlled_capture_bundle_import_builds_stage_ready_manifest(tmp_path):
    _write_bundle(tmp_path)

    summary = objectstate_controlled_capture_import_summary(tmp_path)
    manifest = summary["manifest"]
    capture_summary = summary["capture_summary"]

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA
    assert summary["row_counts"] == {
        "objects": 2,
        "frames": 3,
        "annotations": 6,
        "actions": 1,
    }
    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA
    assert manifest["sample"]["sample_id"] == "controlled-tabletop-cup-box-001"
    assert manifest["objects"][0]["dimensions_m"] == [0.08, 0.08, 0.1]
    assert manifest["actions"][0]["action_id"] == "push-left-001"
    assert manifest["frames"][0]["condition"]["view_id"] == "front"
    assert manifest["frames"][2]["condition"]["camera_pose"]["position"] == [
        0.04,
        0.0,
        0.0,
    ]
    assert capture_summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA
    assert capture_summary["readiness"] == {
        "identity_stage_ready": True,
        "prediction_stage_ready": True,
        "intervention_stage_ready": True,
        "real_gaussian_reconstruction_present": True,
    }
    assert validate_objectstate_controlled_capture_manifest(manifest) is not None
    assert validate_objectstate_controlled_capture_import_summary(summary) == summary


def test_controlled_capture_bundle_import_cli_writes_outputs(tmp_path, capsys):
    _write_bundle(tmp_path)
    manifest_path = tmp_path / "capture-manifest.json"
    summary_path = tmp_path / "import-summary.json"
    controlled_real_path = tmp_path / "controlled-real-seed.json"

    assert (
        main(
            [
                "object-state",
                "import-controlled-capture-bundle",
                str(tmp_path),
                "--output",
                str(manifest_path),
                "--summary-output",
                str(summary_path),
                "--controlled-real-output",
                str(controlled_real_path),
                "--require-identity-ready",
                "--require-prediction-ready",
                "--require-intervention-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    controlled_real = json.loads(controlled_real_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA}" in stdout
    assert "identity_stage_ready=true" in stdout
    assert "prediction_stage_ready=true" in stdout
    assert "intervention_stage_ready=true" in stdout
    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA
    assert controlled_real["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert controlled_real["ground_truth"] == {
        "identity": True,
        "pose": True,
        "action": True,
        "timestamp": True,
    }


def test_controlled_capture_bundle_acceptance_requires_file_audit_pass(tmp_path):
    _write_bundle(tmp_path, include_frame_files=True)

    summary = objectstate_controlled_capture_bundle_acceptance_summary(
        tmp_path,
        hash_files=True,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_ACCEPTANCE_SCHEMA
    assert summary["status"] == (
        "objectstate_controlled_capture_bundle_acceptance_pass"
    )
    assert summary["acceptance_gates"] == {
        "identity_stage_ready": True,
        "prediction_stage_ready": True,
        "intervention_stage_ready": True,
        "capture_file_audit_pass": True,
    }
    assert (
        summary["capture_file_audit"]["status"]
        == "objectstate_controlled_capture_file_audit_pass"
    )
    assert len(summary["capture_file_audit"]["file_records"]["rgb"][0]["sha256"]) == 64
    assert validate_objectstate_controlled_capture_bundle_acceptance_summary(summary) == summary


def test_controlled_capture_bundle_acceptance_fails_missing_frame_files(tmp_path):
    _write_bundle(tmp_path, include_frame_files=False)

    summary = objectstate_controlled_capture_bundle_acceptance_summary(tmp_path)

    assert summary["status"] == (
        "objectstate_controlled_capture_bundle_acceptance_fail"
    )
    assert summary["acceptance_gates"]["identity_stage_ready"] is True
    assert summary["acceptance_gates"]["capture_file_audit_pass"] is False
    assert summary["capture_file_audit"]["missing_files"]


def test_controlled_capture_bundle_acceptance_cli_writes_outputs(tmp_path, capsys):
    _write_bundle(tmp_path, include_frame_files=True)
    manifest_path = tmp_path / "accepted-capture.json"
    summary_path = tmp_path / "acceptance-summary.json"
    import_path = tmp_path / "import-summary.json"
    audit_path = tmp_path / "file-audit.json"
    missing_path = tmp_path / "missing-files.md"
    controlled_real_path = tmp_path / "controlled-real-seed.json"

    assert (
        main(
            [
                "object-state",
                "accept-controlled-capture-bundle",
                str(tmp_path),
                "--output",
                str(manifest_path),
                "--summary-output",
                str(summary_path),
                "--import-summary-output",
                str(import_path),
                "--file-audit-output",
                str(audit_path),
                "--missing-files-output",
                str(missing_path),
                "--controlled-real-output",
                str(controlled_real_path),
                "--require-prediction-ready",
                "--require-intervention-ready",
                "--hash-files",
                "--require-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    import_summary = json.loads(import_path.read_text(encoding="utf-8"))
    file_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    controlled_real = json.loads(controlled_real_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_ACCEPTANCE_SCHEMA}" in stdout
    assert "acceptance_status=objectstate_controlled_capture_bundle_acceptance_pass" in stdout
    assert "capture_bundle_files_ready=true" in stdout
    assert "missing_files=0" in stdout
    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_ACCEPTANCE_SCHEMA
    assert import_summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA
    assert file_audit["status"] == "objectstate_controlled_capture_file_audit_pass"
    assert controlled_real["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert "no missing files" in missing_path.read_text(encoding="utf-8")


def test_controlled_capture_bundle_import_rejects_annotation_without_frame(tmp_path):
    _write_bundle(tmp_path)
    with (tmp_path / "annotations.csv").open("a", encoding="utf-8") as handle:
        handle.write(
            "missing-frame,cup-001,true,0.0,0.2,0.2,0.3,0,0,0,1\n"
        )

    with pytest.raises(ValueError, match="unknown frame_id"):
        objectstate_controlled_capture_manifest_from_bundle(tmp_path)


def test_controlled_capture_bundle_import_rejects_partial_pose(tmp_path):
    _write_bundle(tmp_path)
    rows = (tmp_path / "annotations.csv").read_text(encoding="utf-8").splitlines()
    rows[1] = "frame-000000,cup-001,true,0.0,0.1,0.2,0.3,0,0,0,"
    (tmp_path / "annotations.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="annotation pose requires all columns"):
        objectstate_controlled_capture_manifest_from_bundle(tmp_path)


def _write_bundle(root, *, include_frame_files: bool = False) -> None:
    (root / "sample.json").write_text(
        json.dumps(
            {
                "sample_id": "controlled-tabletop-cup-box-001",
                "source_kind": "controlled_real",
                "object_category": "cup_box",
                "scenario": "cross_view_occlusion_reappearance",
                "fps": 30.0,
                "capture_device": "fixture-camera",
                "observation_modalities": ["rgb", "gaussian"],
                "artifact_refs": [
                    "capture-manifest.json",
                    "rgb/",
                    "gaussians/",
                ],
                "license": "local controlled capture; not public release",
            }
        ),
        encoding="utf-8",
    )
    (root / "objects.csv").write_text(
        "\n".join(
            (
                "object_id,category,instance_label,dimension_x_m,dimension_y_m,dimension_z_m",
                "cup-001,cup,blue cup,0.08,0.08,0.10",
                "box-001,box,red box,0.12,0.10,0.08",
            )
        )
        + "\n",
        encoding="utf-8",
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
    if include_frame_files:
        (root / "rgb").mkdir(parents=True, exist_ok=True)
        (root / "gaussians").mkdir(parents=True, exist_ok=True)
        for index in range(3):
            (root / "rgb" / f"{index:06d}.png").write_bytes(PNG_BYTES)
            (root / "gaussians" / f"{index:06d}.ply").write_bytes(PLY_BYTES)
