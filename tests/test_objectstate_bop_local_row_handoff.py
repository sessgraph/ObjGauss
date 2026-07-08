from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_bop_candidate_artifact_template import (
    finalize_objectstate_bop_candidate_artifact_template,
    write_objectstate_bop_candidate_artifact_template,
)
from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
)
from objgauss.core.objectstate_bop_local_row_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
    objectstate_bop_local_row_handoff,
    validate_objectstate_bop_local_row_handoff_summary,
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


def test_bop_local_row_handoff_writes_identity_prediction_ledger(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "local-row-package"
    sidecar_path = scene_root / "bop-condition-sidecar.json"
    artifact_path = _write_finalized_artifact(scene_root, output_root)
    _write_json(sidecar_path, _condition_sidecar_payload())

    summary = objectstate_bop_local_row_handoff(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_artifact=artifact_path,
        condition_sidecar=sidecar_path,
        identity_candidate_id="fixture-bop-identity",
        prediction_candidate_id="fixture-bop-prediction",
        max_centroid_distance=0.01,
    )

    assert summary["schema"] == OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA
    assert summary["status"] == "objectstate_bop_local_row_handoff_reviewable"
    assert validate_objectstate_bop_local_row_handoff_summary(summary) == summary
    assert summary["reviewability_gates"] == {
        "identity_handoff_reviewable": True,
        "prediction_handoff_reviewable": True,
        "phase1_evidence_ledger_identity_reviewable": True,
        "phase1_evidence_ledger_prediction_reviewable": True,
        "same_sample_scope": True,
    }
    assert summary["pass_gates"] == {
        "identity_handoff_pass": True,
        "prediction_eval_pass": True,
    }
    assert summary["phase1_evidence_ledger_summary"]["maturity"] == (
        "identity_prediction_reviewable"
    )
    assert (
        summary["phase1_evidence_ledger_summary"]["phase1_evidence_gates"][
            "intervention_evidence_reviewable"
        ]
        is False
    )
    assert (output_root / "identity-handoff" / "identity-eval-summary.json").is_file()
    assert (
        output_root
        / "reality-candidates"
        / "prediction-evidence-package-summary.json"
    ).is_file()
    ledger = _read_json(output_root / "phase1-evidence-ledger.json")
    assert ledger["maturity"] == "identity_prediction_reviewable"
    assert summary["phase1_evidence_ledger_summary"] == ledger


def test_bop_local_row_handoff_keeps_identity_scenario_blocker_visible(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "local-row-package"
    artifact_path = _write_finalized_artifact(scene_root, output_root)

    summary = objectstate_bop_local_row_handoff(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_artifact=artifact_path,
        max_centroid_distance=0.01,
    )

    assert summary["status"] == "objectstate_bop_local_row_handoff_incomplete"
    assert summary["reviewability_gates"]["prediction_handoff_reviewable"] is True
    assert summary["reviewability_gates"]["identity_handoff_reviewable"] is False
    assert (
        summary["reviewability_gates"][
            "phase1_evidence_ledger_prediction_reviewable"
        ]
        is True
    )
    assert (
        summary["reviewability_gates"]["phase1_evidence_ledger_identity_reviewable"]
        is False
    )
    assert summary["phase1_evidence_ledger_summary"]["maturity"] == (
        "prediction_reviewable"
    )
    assert "identity handoff is not reviewable" in summary["issues"]
    assert (
        summary["identity_handoff"]["identity_handoff"]["identity_scenario_audit"][
            "readiness"
        ]["camera_motion_present"]
        is False
    )


def test_object_state_bop_local_row_handoff_cli(tmp_path, capsys):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "local-row-package"
    sidecar_path = scene_root / "bop-condition-sidecar.json"
    summary_path = tmp_path / "bop-local-row-handoff-summary.json"
    artifact_path = _write_finalized_artifact(scene_root, output_root)
    _write_json(sidecar_path, _condition_sidecar_payload())

    assert (
        main(
            [
                "object-state",
                "bop-local-row-handoff",
                str(scene_root),
                "--output-root",
                str(output_root),
                "--sample-id",
                "bop-ycbv-scene-000001",
                "--candidate-artifact",
                str(artifact_path),
                "--condition-sidecar",
                str(sidecar_path),
                "--identity-candidate-id",
                "cli-bop-identity",
                "--prediction-candidate-id",
                "cli-bop-prediction",
                "--max-centroid-distance",
                "0.01",
                "--summary-output",
                str(summary_path),
                "--require-reviewable",
                "--require-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_path)

    assert f"schema={OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA}" in stdout
    assert "bop_local_row_handoff_status=objectstate_bop_local_row_handoff_reviewable" in stdout
    assert "reviewability.identity_handoff_reviewable=true" in stdout
    assert "reviewability.prediction_handoff_reviewable=true" in stdout
    assert "pass.identity_handoff_pass=true" in stdout
    assert "pass.prediction_eval_pass=true" in stdout
    assert "phase1_evidence_ledger_maturity=identity_prediction_reviewable" in stdout
    assert "identity_evidence_package_summary=" in stdout
    assert "prediction_evidence_package_summary=" in stdout
    assert summary["status"] == "objectstate_bop_local_row_handoff_reviewable"


def _write_finalized_artifact(scene_root, output_root):
    template_path = output_root / "objectstates.template.json"
    artifact_path = output_root / "objectstates.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    write_objectstate_bop_candidate_artifact_template(
        scene_root,
        output=template_path,
        sample_id="bop-ycbv-scene-000001",
        target_artifact_path=artifact_path,
        candidate_id="bop-local-row-candidate",
        candidate_source="unit-test-filled-template",
    )
    _fill_template(template_path)
    finalize_objectstate_bop_candidate_artifact_template(
        template_path,
        reconstruction_noise_robustness=0.97,
        reconstruction_noise_variant_count=2,
    )
    return artifact_path


def _write_bop_scene(root) -> None:
    (root / "rgb").mkdir(parents=True)
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
    visibility_by_frame = (1.0, 0.2, 1.0)
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
                "px_count_visib": int(1000 * visibility_by_frame[frame_id]),
                "visib_fract": visibility_by_frame[frame_id],
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
    _write_json(root / "scene_camera.json", scene_camera)
    _write_json(root / "scene_gt.json", scene_gt)
    _write_json(root / "scene_gt_info.json", scene_gt_info)


def _write_gaussian_frames(root) -> None:
    (root / "gaussians").mkdir()
    for frame_id in range(3):
        (root / "gaussians" / f"{frame_id:06d}.ply").write_bytes(PLY_BYTES)


def _fill_template(path) -> None:
    template = _read_json(path)
    for frame in template["object_state_frames"]:
        frame_index = frame["frame_index"]
        for slot_index, placeholder in enumerate(frame["state_placeholders"]):
            if placeholder["object_id"].endswith("000001"):
                gt = [0.01 + 0.001 * frame_index, 0.02, 0.03]
            else:
                gt = [0.04 + 0.001 * frame_index, 0.05, 0.06]
            centroid = [value + 0.0002 for value in gt]
            placeholder["state_id"] = slot_index
            placeholder["centroid"] = centroid
            placeholder["bbox"] = [
                [value - 0.001 for value in centroid],
                [value + 0.001 for value in centroid],
            ]
            placeholder["confidence"] = 0.92
            placeholder["note"] = "Filled from unit-test model output."
    _write_json(path, template)


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


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
