from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.pipelines.objectstate_bop_baseline_local_row_handoff import (
    OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA,
    objectstate_bop_baseline_local_row_handoff,
    validate_objectstate_bop_baseline_local_row_handoff_summary,
)
from objgauss.datasets.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n"


def test_bop_baseline_local_row_handoff_writes_reviewable_negative_evidence(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "baseline-local-row"
    sidecar_path = scene_root / "bop-condition-sidecar.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    _write_json(sidecar_path, _condition_sidecar_payload())

    summary = objectstate_bop_baseline_local_row_handoff(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        condition_sidecar=sidecar_path,
    )

    assert summary["schema"] == OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA
    assert validate_objectstate_bop_baseline_local_row_handoff_summary(summary) == summary
    assert summary["status"] == (
        "objectstate_bop_baseline_local_row_handoff_reviewable"
    )
    assert summary["candidate_artifact"] == str(output_root / "objectstates.json")
    assert summary["row_counts"]["baseline_frames"] == 3
    assert summary["row_counts"]["baseline_states"] == 3
    assert summary["row_counts"]["baseline_total_gaussians"] == 6
    assert summary["reviewability_gates"] == {
        "baseline_candidate_written": True,
        "baseline_candidate_ready_for_identity_handoff": True,
        "local_row_identity_handoff_reviewable": True,
        "local_row_prediction_handoff_reviewable": True,
        "phase1_evidence_ledger_identity_reviewable": True,
        "phase1_evidence_ledger_prediction_reviewable": True,
    }
    assert summary["pass_gates"]["identity_handoff_pass"] is False
    assert summary["pass_gates"]["prediction_eval_pass"] is True
    assert summary["baseline_candidate"]["baseline_policy"][
        "does_not_use_bop_pose_gt_for_prediction"
    ] is True
    identity_metrics = summary["local_row_handoff"]["identity_handoff"][
        "identity_handoff"
    ]["identity_eval"]["metrics"]
    assert identity_metrics["raw_prediction_observations"] is True
    assert identity_metrics["identity_collapse"] is False
    assert identity_metrics["missing_prediction_count"] == 3
    assert identity_metrics["reconstruction_noise_evidence_present"] is False
    assert (output_root / "objectstates.json").is_file()
    assert (output_root / "phase1-evidence-ledger.json").is_file()


def test_bop_baseline_local_row_handoff_cli(tmp_path, capsys):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "baseline-local-row"
    sidecar_path = scene_root / "bop-condition-sidecar.json"
    summary_path = tmp_path / "baseline-local-row-summary.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    _write_json(sidecar_path, _condition_sidecar_payload())

    assert (
        main(
            [
                "object-state",
                "bop-baseline-local-row-handoff",
                str(scene_root),
                "--output-root",
                str(output_root),
                "--sample-id",
                "bop-ycbv-scene-000001",
                "--condition-sidecar",
                str(sidecar_path),
                "--summary-output",
                str(summary_path),
                "--require-reviewable",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_path)

    assert f"schema={OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA}" in stdout
    assert "bop_baseline_local_row_handoff_status=" in stdout
    assert "reviewability.baseline_candidate_written=true" in stdout
    assert "reviewability.local_row_identity_handoff_reviewable=true" in stdout
    assert "pass.identity_handoff_pass=false" in stdout
    assert "baseline_total_gaussians=6" in stdout
    assert "candidate_artifact=" in stdout
    assert summary["status"] == (
        "objectstate_bop_baseline_local_row_handoff_reviewable"
    )


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
        x0 = 0.0 + float(frame_id)
        x1 = 2.0 + float(frame_id)
        body = (
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
            f"{x0} 0 0\n"
            f"{x1} 0 0\n"
        )
        (root / "gaussians" / f"{frame_id:06d}.ply").write_text(
            body,
            encoding="ascii",
        )


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
