from __future__ import annotations

import csv
import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA,
    OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA,
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SUMMARY_SCHEMA,
    objectstate_bop_capture_acceptance_summary,
    objectstate_bop_capture_adapter_summary,
    objectstate_bop_capture_condition_sidecar_summary,
    objectstate_bop_capture_manifest_from_scene,
    validate_objectstate_bop_capture_acceptance_summary,
    validate_objectstate_bop_capture_adapter_summary,
    validate_objectstate_bop_capture_condition_sidecar,
    validate_objectstate_bop_capture_condition_sidecar_summary,
)
from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
)
from objgauss.core.objectstate_controlled_reality_candidate_template import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
    finalize_objectstate_controlled_prediction_candidate_template,
    validate_objectstate_controlled_prediction_candidate_finalize_summary,
    validate_objectstate_controlled_reality_candidate_template_summary,
    write_objectstate_controlled_reality_candidate_templates_from_manifest,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    validate_objectstate_controlled_prediction_candidates,
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


def test_bop_capture_adapter_builds_identity_prediction_ready_manifest(tmp_path):
    _write_bop_scene(tmp_path)

    summary = objectstate_bop_capture_adapter_summary(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        dataset_id="bop-ycbv",
        max_frames=3,
    )
    manifest = summary["manifest"]
    capture_summary = summary["capture_summary"]

    assert summary["schema"] == OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA
    assert summary["row_counts"] == {
        "objects": 2,
        "frames": 3,
        "annotations": 6,
        "actions": 0,
    }
    assert summary["selected_frame_ids"] == [0, 1, 2]
    assert summary["readiness"]["identity_stage_ready"] is True
    assert summary["readiness"]["prediction_stage_ready"] is True
    assert summary["readiness"]["intervention_stage_ready"] is False
    assert summary["readiness"]["real_gaussian_reconstruction_present"] is False
    assert "local per-frame Gaussian reconstruction" in " ".join(
        summary["hard_blockers"]
    )
    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA
    assert manifest["sample"]["sample_id"] == "bop-ycbv-scene-000001"
    assert manifest["objects"][0]["object_id"] == "bop-ycbv-obj-000001"
    assert manifest["frames"][0]["observation"]["rgb"] == "rgb/000000.png"
    assert manifest["frames"][0]["objects"][0]["pose"]["position"] == [
        0.01,
        0.02,
        0.03,
    ]
    assert manifest["frames"][1]["objects"][0]["occlusion_fraction"] == pytest.approx(
        0.2
    )
    assert manifest["frames"][0]["objects"][0]["pose"]["rotation_xyzw"] == [
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    assert capture_summary["ground_truth"] == {
        "identity": True,
        "pose": True,
        "action": False,
        "timestamp": True,
    }
    assert (
        summary["controlled_real_manifest_seed"]["schema"]
        == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    )
    assert validate_objectstate_bop_capture_adapter_summary(summary) == summary


def test_bop_capture_adapter_manifest_helper_returns_manifest(tmp_path):
    _write_bop_scene(tmp_path)

    manifest = objectstate_bop_capture_manifest_from_scene(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
    )

    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA
    assert len(manifest["frames"]) == 3


def test_bop_capture_adapter_merges_condition_sidecar(tmp_path):
    _write_bop_scene(tmp_path)
    sidecar_path = tmp_path / "bop-condition-sidecar.json"
    sidecar_payload = _condition_sidecar_payload()
    _write_json(sidecar_path, sidecar_payload)

    summary = objectstate_bop_capture_adapter_summary(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        condition_sidecar=sidecar_path,
    )
    manifest = summary["manifest"]

    assert summary["condition_sidecar_schema"] == (
        OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA
    )
    assert summary["condition_sidecar"] == (
        validate_objectstate_bop_capture_condition_sidecar(sidecar_payload)
    )
    assert summary["source"]["source_files"]["condition_sidecar"] == str(sidecar_path)
    assert manifest["frames"][0]["condition"] == {
        "view_id": "front",
        "lighting_id": "bright",
        "camera_pose": {
            "position": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    assert manifest["frames"][1]["condition"]["view_id"] == "front"
    assert manifest["frames"][1]["condition"]["lighting_id"] == "dim"
    assert manifest["frames"][2]["condition"]["view_id"] == "right"


def test_bop_condition_sidecar_summary_reports_template_blocker(tmp_path):
    _write_bop_scene(tmp_path)

    summary = objectstate_bop_capture_condition_sidecar_summary(tmp_path)

    assert summary["schema"] == OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SUMMARY_SCHEMA
    assert (
        summary["status"]
        == "objectstate_bop_capture_condition_sidecar_needs_metadata"
    )
    assert summary["selected_frame_ids"] == [0, 1, 2]
    assert summary["row_counts"] == {
        "selected_frames": 3,
        "csv_condition_rows": 0,
        "sidecar_frames": 3,
    }
    assert summary["readiness"]["selected_frames_covered"] is True
    assert summary["readiness"]["condition_csv_loaded"] is False
    assert summary["readiness"]["min_view_conditions_met"] is True
    assert summary["readiness"]["min_lighting_conditions_met"] is False
    assert summary["readiness"]["camera_motion_present"] is False
    assert summary["readiness"]["identity_scenario_metadata_ready"] is False
    assert summary["coverage"]["lighting_ids"] == ["bop-default"]
    assert summary["coverage"]["camera_pose_count"] == 0
    assert summary["condition_csv_template"] == [
        {
            "frame_id": 0,
            "view_id": "bop-camera-frame-000000",
            "lighting_id": "bop-default",
            "camera_x": "",
            "camera_y": "",
            "camera_z": "",
            "camera_qx": "",
            "camera_qy": "",
            "camera_qz": "",
            "camera_qw": "",
        },
        {
            "frame_id": 1,
            "view_id": "bop-camera-frame-000001",
            "lighting_id": "bop-default",
            "camera_x": "",
            "camera_y": "",
            "camera_z": "",
            "camera_qx": "",
            "camera_qy": "",
            "camera_qz": "",
            "camera_qw": "",
        },
        {
            "frame_id": 2,
            "view_id": "bop-camera-frame-000002",
            "lighting_id": "bop-default",
            "camera_x": "",
            "camera_y": "",
            "camera_z": "",
            "camera_qx": "",
            "camera_qy": "",
            "camera_qz": "",
            "camera_qw": "",
        },
    ]
    assert "condition CSV was not provided" in " ".join(summary["issues"])
    assert (
        validate_objectstate_bop_capture_condition_sidecar_summary(summary)
        == summary
    )


def test_bop_condition_sidecar_cli_writes_identity_ready_sidecar(tmp_path, capsys):
    _write_bop_scene(tmp_path)
    csv_path = tmp_path / "bop-conditions.csv"
    template_path = tmp_path / "bop-conditions.template.csv"
    sidecar_path = tmp_path / "bop-condition-sidecar.json"
    summary_path = tmp_path / "bop-condition-sidecar-summary.json"
    _write_condition_csv(csv_path)

    assert (
        main(
            [
                "object-state",
                "init-bop-condition-sidecar",
                str(tmp_path),
                "--condition-csv",
                str(csv_path),
                "--output",
                str(sidecar_path),
                "--summary-output",
                str(summary_path),
                "--condition-csv-template-output",
                str(template_path),
                "--require-identity-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with template_path.open("r", encoding="utf-8", newline="") as handle:
        template_rows = list(csv.DictReader(handle))

    assert f"schema={OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SUMMARY_SCHEMA}" in stdout
    assert (
        "bop_condition_sidecar_status="
        "objectstate_bop_capture_condition_sidecar_identity_ready"
    ) in stdout
    assert "readiness.identity_scenario_metadata_ready=true" in stdout
    assert f"condition_csv_template={template_path}" in stdout
    assert sidecar["schema"] == OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA
    assert sidecar["frames"]["2"]["view_id"] == "right"
    assert sidecar["frames"]["2"]["lighting_id"] == "dim"
    assert sidecar["frames"]["2"]["camera_pose"]["position"] == [0.04, 0.0, 0.0]
    assert (
        summary["status"]
        == "objectstate_bop_capture_condition_sidecar_identity_ready"
    )
    assert summary["coverage"]["lighting_condition_count"] == 2
    assert summary["coverage"]["max_camera_translation_m"] == pytest.approx(0.04)
    assert template_rows[2] == {
        "frame_id": "2",
        "view_id": "right",
        "lighting_id": "dim",
        "camera_x": "0.04",
        "camera_y": "0",
        "camera_z": "0",
        "camera_qx": "0",
        "camera_qy": "0",
        "camera_qz": "0",
        "camera_qw": "1",
    }


def test_bop_capture_adapter_cli_accepts_condition_sidecar(tmp_path, capsys):
    _write_bop_scene(tmp_path)
    sidecar_path = tmp_path / "bop-condition-sidecar.json"
    _write_json(sidecar_path, _condition_sidecar_payload())
    manifest_path = tmp_path / "capture-manifest.json"

    assert (
        main(
            [
                "object-state",
                "import-bop-capture-scene",
                str(tmp_path),
                "--sample-id",
                "bop-ycbv-scene-000001",
                "--condition-sidecar",
                str(sidecar_path),
                "--output",
                str(manifest_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert "bop_adapter_status=objectstate_bop_capture_adapter_ready" in stdout
    assert manifest["frames"][0]["condition"]["view_id"] == "front"
    assert manifest["frames"][1]["condition"]["lighting_id"] == "dim"
    assert manifest["frames"][2]["condition"]["camera_pose"]["position"] == [
        0.04,
        0.0,
        0.0,
    ]


def test_bop_capture_adapter_cli_writes_outputs(tmp_path, capsys):
    _write_bop_scene(tmp_path)
    manifest_path = tmp_path / "capture-manifest.json"
    summary_path = tmp_path / "bop-adapter-summary.json"
    controlled_real_path = tmp_path / "controlled-real-seed.json"

    assert (
        main(
            [
                "object-state",
                "import-bop-capture-scene",
                str(tmp_path),
                "--sample-id",
                "bop-ycbv-scene-000001",
                "--dataset-id",
                "bop-ycbv",
                "--output",
                str(manifest_path),
                "--summary-output",
                str(summary_path),
                "--controlled-real-output",
                str(controlled_real_path),
                "--require-identity-ready",
                "--require-prediction-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    controlled_real = json.loads(controlled_real_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA}" in stdout
    assert "bop_adapter_status=objectstate_bop_capture_adapter_ready" in stdout
    assert "identity_stage_ready=true" in stdout
    assert "prediction_stage_ready=true" in stdout
    assert "real_gaussian_reconstruction_present=false" in stdout
    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA
    assert summary["schema"] == OBJECTSTATE_BOP_CAPTURE_ADAPTER_SCHEMA
    assert controlled_real["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert controlled_real["ground_truth"] == {
        "identity": True,
        "pose": True,
        "action": False,
        "timestamp": True,
    }


def test_bop_capture_acceptance_passes_rgb_only_file_audit(tmp_path):
    _write_bop_scene(tmp_path)

    summary = objectstate_bop_capture_acceptance_summary(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        dataset_id="bop-ycbv",
        hash_files=True,
    )

    assert summary["schema"] == OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA
    assert summary["status"] == "objectstate_bop_capture_acceptance_pass"
    assert summary["readiness"]["capture_file_audit_pass"] is True
    assert summary["readiness"]["rgb_files_present"] is True
    assert summary["readiness"]["gaussian_files_present"] is True
    assert summary["readiness"]["phase1_gaussian_evidence_required"] is False
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is False
    assert summary["requirements"]["gaussian_refs_included"] is False
    assert summary["file_audit"]["file_counts"]["rgb"]["valid"] == 3
    assert summary["file_audit"]["file_counts"]["gaussian"] == {
        "referenced": 0,
        "existing": 0,
        "valid": 0,
        "missing": 0,
    }
    assert "rerun with require_gaussian_files" in " ".join(summary["hard_blockers"])
    assert validate_objectstate_bop_capture_acceptance_summary(summary) == summary


def test_bop_capture_acceptance_requires_gaussian_files_when_requested(tmp_path):
    _write_bop_scene(tmp_path)

    summary = objectstate_bop_capture_acceptance_summary(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        require_gaussian_files=True,
    )

    assert summary["status"] == "objectstate_bop_capture_acceptance_fail"
    assert summary["requirements"]["gaussian_refs_included"] is True
    assert summary["readiness"]["capture_file_audit_pass"] is False
    assert summary["readiness"]["gaussian_files_present"] is False
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is False
    assert len(summary["file_audit"]["missing_files"]) == 3
    assert "invalid or missing Gaussian files" in " ".join(summary["hard_blockers"])


def test_bop_capture_acceptance_passes_with_gaussian_files(tmp_path):
    _write_bop_scene(tmp_path)
    _write_gaussian_frames(tmp_path)

    summary = objectstate_bop_capture_acceptance_summary(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        require_gaussian_files=True,
    )

    assert summary["status"] == "objectstate_bop_capture_acceptance_pass"
    assert summary["readiness"]["gaussian_files_present"] is True
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is True
    assert summary["file_audit"]["file_counts"]["gaussian"]["valid"] == 3
    assert "gaussian" in summary["manifest"]["sample"]["observation_modalities"]
    assert "local per-frame Gaussian reconstruction" not in " ".join(
        summary["hard_blockers"]
    )


def test_bop_capture_acceptance_cli_writes_file_audit_outputs(tmp_path, capsys):
    _write_bop_scene(tmp_path)
    _write_gaussian_frames(tmp_path)
    manifest_path = tmp_path / "accepted-capture-manifest.json"
    summary_path = tmp_path / "bop-acceptance-summary.json"
    file_audit_path = tmp_path / "bop-file-audit.json"
    missing_files_path = tmp_path / "bop-missing-files.md"
    controlled_real_path = tmp_path / "controlled-real-seed.json"

    assert (
        main(
            [
                "object-state",
                "accept-bop-capture-scene",
                str(tmp_path),
                "--sample-id",
                "bop-ycbv-scene-000001",
                "--output",
                str(manifest_path),
                "--summary-output",
                str(summary_path),
                "--file-audit-output",
                str(file_audit_path),
                "--missing-files-output",
                str(missing_files_path),
                "--controlled-real-output",
                str(controlled_real_path),
                "--require-gaussian-files",
                "--hash-files",
                "--require-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    file_audit = json.loads(file_audit_path.read_text(encoding="utf-8"))
    controlled_real = json.loads(controlled_real_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA}" in stdout
    assert "bop_acceptance_status=objectstate_bop_capture_acceptance_pass" in stdout
    assert "capture_file_audit_pass=true" in stdout
    assert "phase1_gaussian_evidence_ready=true" in stdout
    assert "missing_files=0" in stdout
    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA
    assert summary["schema"] == OBJECTSTATE_BOP_CAPTURE_ACCEPTANCE_SCHEMA
    assert file_audit["status"] == "objectstate_controlled_capture_file_audit_pass"
    assert controlled_real["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert "no missing files" in missing_files_path.read_text(encoding="utf-8")


def test_bop_acceptance_manifest_initializes_prediction_candidates(tmp_path):
    _write_bop_scene(tmp_path)
    _write_gaussian_frames(tmp_path)
    capture_manifest_path = tmp_path / "capture-manifest.json"
    template_dir = tmp_path / "reality-candidates"

    acceptance = objectstate_bop_capture_acceptance_summary(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        require_gaussian_files=True,
    )
    _write_json(capture_manifest_path, acceptance["manifest"])

    summary = write_objectstate_controlled_reality_candidate_templates_from_manifest(
        capture_manifest_path,
        output_dir=template_dir,
        candidate_id="bop-ycbv-candidate",
        candidate_source="bop candidate fixture",
        artifact_ref="outputs/captures/bop-ycbv-scene-000001/objectstates.json",
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA
    assert summary["source"]["kind"] == "capture_manifest"
    assert summary["sample"]["sample_id"] == "bop-ycbv-scene-000001"
    assert summary["readiness"]["capture_prediction_stage_ready"] is True
    assert summary["readiness"]["capture_intervention_stage_ready"] is False
    assert summary["row_counts"]["prediction_drafts"] == 4
    assert summary["row_counts"]["intervention_drafts"] == 0
    assert "no intervention draft rows were generated" in summary["issues"]
    assert "finalize_prediction_candidates" in summary["next_commands"]
    assert "finalize_candidates" not in summary["next_commands"]
    assert validate_objectstate_controlled_reality_candidate_template_summary(summary) == summary


def test_bop_prediction_template_can_finalize_without_intervention_rows(tmp_path):
    _write_bop_scene(tmp_path)
    _write_gaussian_frames(tmp_path)
    capture_manifest_path = tmp_path / "capture-manifest.json"
    template_dir = tmp_path / "reality-candidates"
    output_dir = tmp_path / "prediction-candidates"

    acceptance = objectstate_bop_capture_acceptance_summary(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        require_gaussian_files=True,
    )
    _write_json(capture_manifest_path, acceptance["manifest"])
    write_objectstate_controlled_reality_candidate_templates_from_manifest(
        capture_manifest_path,
        output_dir=template_dir,
        candidate_id="bop-ycbv-candidate",
        candidate_source="bop candidate fixture",
        artifact_ref="outputs/captures/bop-ycbv-scene-000001/objectstates.json",
    )
    prediction_template_path = template_dir / "prediction-candidates.template.json"
    prediction_template = _read_json(prediction_template_path)
    for index, row in enumerate(prediction_template["predictions"]):
        value = float(index) * 0.01
        row["predicted_position"] = [value, 0.0, 0.03]
        row["history_baseline_position"] = [value + 0.01, 0.0, 0.03]
    _write_json(prediction_template_path, prediction_template)

    summary = finalize_objectstate_controlled_prediction_candidate_template(
        prediction_template_path,
        output_dir=output_dir,
        capture_manifest=capture_manifest_path,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA
    assert summary["row_counts"]["prediction_candidates"] == 4
    assert validate_objectstate_controlled_prediction_candidate_finalize_summary(summary) == summary
    prediction_candidates = _read_json(output_dir / "prediction-candidates.json")
    assert validate_objectstate_controlled_prediction_candidates(
        prediction_candidates
    )["sample_id"] == "bop-ycbv-scene-000001"


def test_bop_capture_adapter_rejects_duplicate_obj_ids(tmp_path):
    _write_bop_scene(tmp_path)
    scene_gt = json.loads((tmp_path / "scene_gt.json").read_text(encoding="utf-8"))
    scene_gt["0"].append(scene_gt["0"][0])
    (tmp_path / "scene_gt.json").write_text(
        json.dumps(scene_gt),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate obj_id=1"):
        objectstate_bop_capture_adapter_summary(
            tmp_path,
            sample_id="bop-ycbv-scene-000001",
        )


def test_bop_capture_adapter_requires_rgb_files(tmp_path):
    _write_bop_scene(tmp_path)
    (tmp_path / "rgb" / "000001.png").unlink()

    with pytest.raises(FileNotFoundError, match="could not find RGB file"):
        objectstate_bop_capture_adapter_summary(
            tmp_path,
            sample_id="bop-ycbv-scene-000001",
        )


def _write_bop_scene(root) -> None:
    (root / "rgb").mkdir()
    for frame_id in range(3):
        (root / "rgb" / f"{frame_id:06d}.png").write_bytes(PNG_BYTES)
    scene_camera = {
        str(frame_id): {
            "cam_K": [572.4, 0.0, 325.2, 0.0, 573.5, 242.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        for frame_id in range(3)
    }
    identity_rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    scene_gt = {}
    scene_gt_info = {}
    for frame_id in range(3):
        scene_gt[str(frame_id)] = [
            {
                "obj_id": 1,
                "cam_R_m2c": identity_rotation,
                "cam_t_m2c": [10.0 + frame_id, 20.0, 30.0],
            },
            {
                "obj_id": 2,
                "cam_R_m2c": identity_rotation,
                "cam_t_m2c": [40.0 + frame_id, 50.0, 60.0],
            },
        ]
        scene_gt_info[str(frame_id)] = [
            {
                "bbox_obj": [10, 20, 30, 40],
                "bbox_visib": [10, 20, 30, 40],
                "px_count_all": 1000,
                "px_count_valid": 1000,
                "px_count_visib": 1000 - frame_id * 200,
                "visib_fract": 1.0 - frame_id * 0.2,
            },
            {
                "bbox_obj": [50, 60, 30, 40],
                "bbox_visib": [50, 60, 30, 40],
                "px_count_all": 900,
                "px_count_valid": 900,
                "px_count_visib": 900,
                "visib_fract": 1.0,
            },
        ]
    (root / "scene_camera.json").write_text(json.dumps(scene_camera), encoding="utf-8")
    (root / "scene_gt.json").write_text(json.dumps(scene_gt), encoding="utf-8")
    (root / "scene_gt_info.json").write_text(
        json.dumps(scene_gt_info),
        encoding="utf-8",
    )


def _write_gaussian_frames(root) -> None:
    (root / "gaussians").mkdir()
    for frame_id in range(3):
        (root / "gaussians" / f"{frame_id:06d}.ply").write_bytes(PLY_BYTES)


def _condition_sidecar_payload():
    return {
        "schema": OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
        "kind": "objectstate_bop_capture_condition_sidecar",
        "frames": {
            "0": {
                "view_id": "front",
                "lighting_id": "bright",
                "camera_pose": {
                    "position": [0.0, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "1": {
                "view_id": "front",
                "lighting_id": "dim",
                "camera_pose": {
                    "position": [0.02, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "000002": {
                "view_id": "right",
                "lighting_id": "dim",
                "camera_pose": {
                    "position": [0.04, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
        },
        "condition_policy": {
            "sidecar_only": True,
            "does_not_create_ground_truth": True,
            "does_not_infer_from_pixels": True,
        },
    }


def _write_condition_csv(path) -> None:
    path.write_text(
        "\n".join(
            [
                "frame_id,view_id,lighting_id,camera_x,camera_y,camera_z,camera_qx,camera_qy,camera_qz,camera_qw",
                "0,front,bright,0.0,0.0,0.0,0.0,0.0,0.0,1.0",
                "1,front,dim,0.02,0.0,0.0,0.0,0.0,0.0,1.0",
                "2,right,dim,0.04,0.0,0.0,0.0,0.0,0.0,1.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
