from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.datasets.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
)
from objgauss.pipelines.objectstate_bop_phase1_local_row_readiness import (
    OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA,
    objectstate_bop_phase1_local_row_readiness,
    validate_objectstate_bop_phase1_local_row_readiness_summary,
)
from objgauss.pipelines.objectstate_bop_prediction_baseline_handoff import (
    objectstate_bop_prediction_baseline_handoff,
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


def test_bop_phase1_local_row_readiness_reports_missing_scene(tmp_path):
    summary = objectstate_bop_phase1_local_row_readiness(
        tmp_path / "missing-scene",
        output_root=tmp_path / "phase1-row",
        sample_id="bop-ycbv-scene-000001",
    )

    assert summary["schema"] == OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA
    assert summary["status"] == "objectstate_bop_phase1_local_row_readiness_blocked"
    assert summary["blocking_stage"] == "local_bop_scene"
    assert summary["readiness"]["bop_acceptance_available"] is False
    assert summary["routes"]["identity"]["status"] == (
        "objectstate_bop_identity_route_audit_blocked"
    )
    assert summary["routes"]["prediction"]["status"] == (
        "objectstate_bop_phase1_route_audit_blocked"
    )
    assert any("place a local BOP scene" in item for item in summary["next_actions"])
    assert validate_objectstate_bop_phase1_local_row_readiness_summary(summary) == summary


def test_bop_phase1_local_row_readiness_requires_candidate_after_gaussians(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "phase1-row"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)

    summary = objectstate_bop_phase1_local_row_readiness(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
    )

    assert (
        summary["status"]
        == "objectstate_bop_phase1_local_row_readiness_candidate_required"
    )
    assert summary["blocking_stage"] == "candidate_artifact"
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is True
    assert summary["readiness"]["prediction_route_ready_for_handoff"] is True
    assert summary["readiness"]["candidate_artifact_binding_ready"] is False
    assert any("create or export a trainable ObjectState artifact" in item for item in summary["next_actions"])


def test_bop_phase1_local_row_readiness_hints_rgbd_export_when_depth_exists(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "phase1-row"
    _write_bop_scene(scene_root)
    _write_depth_frames(scene_root)

    summary = objectstate_bop_phase1_local_row_readiness(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
    )

    hint = summary["rgbd_gaussian_export_hint"]
    assert summary["status"] == "objectstate_bop_phase1_local_row_readiness_blocked"
    assert summary["blocking_stage"] == "bop_acceptance"
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is False
    assert hint["selected_frames"] == 3
    assert hint["depth_files_present"] == 3
    assert hint["missing_depth_files"] == 0
    assert hint["missing_gaussian_files"] == 3
    assert hint["rgbd_export_candidate"] is True
    assert "export-bop-rgbd-gaussian-evidence" in hint["recommended_command"]
    assert any("RGB-D Gaussian evidence export" in item for item in summary["next_actions"])
    assert any("rgbd_export_hint" in item for item in summary["issues"])


def test_bop_phase1_local_row_readiness_reports_identity_scenario_blocker(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "phase1-row"
    artifact_path = output_root / "objectstates.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    _write_json(artifact_path, _trainable_artifact(frame_count=3))

    summary = objectstate_bop_phase1_local_row_readiness(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_artifact=artifact_path,
    )

    assert (
        summary["status"]
        == "objectstate_bop_phase1_local_row_readiness_identity_scenario_blocked"
    )
    assert summary["blocking_stage"] == "identity_scenario_metadata"
    assert summary["readiness"]["candidate_artifact_binding_ready"] is True
    assert summary["readiness"]["identity_scenario_metadata_ready"] is False
    assert summary["readiness"]["prediction_route_ready_for_handoff"] is True
    assert any("do not relax the identity scenario gate" in item for item in summary["next_actions"])


def test_bop_phase1_local_row_readiness_accepts_condition_sidecar(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "phase1-row"
    artifact_path = output_root / "objectstates.json"
    sidecar_path = scene_root / "bop-condition-sidecar.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    _write_json(artifact_path, _trainable_artifact(frame_count=3))
    _write_json(sidecar_path, _condition_sidecar_payload())

    summary = objectstate_bop_phase1_local_row_readiness(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_artifact=artifact_path,
        condition_sidecar=sidecar_path,
    )

    assert (
        summary["status"]
        == "objectstate_bop_phase1_local_row_readiness_identity_prediction_handoff_ready"
    )
    assert summary["blocking_stage"] == "handoff_ready"
    assert summary["readiness"]["identity_route_ready_for_handoff"] is True
    assert summary["readiness"]["prediction_route_ready_for_handoff"] is True
    assert summary["readiness"]["identity_scenario_metadata_ready"] is True
    assert (
        summary["routes"]["identity"]["identity_scenario_metadata_audit"]["status"]
        == "objectstate_bop_identity_route_scenario_metadata_ready"
    )
    assert (
        summary["routes"]["identity"]["acceptance"]["manifest"]["frames"][2][
            "condition"
        ]["view_id"]
        == "right"
    )


def test_bop_phase1_local_row_readiness_reports_prediction_reviewable(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "phase1-row"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    objectstate_bop_prediction_baseline_handoff(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_id="fixture-bop-baseline",
        candidate_source="unit-test BOP baseline",
        artifact_ref="outputs/captures/bop-ycbv-scene-000001/objectstates.json",
    )

    summary = objectstate_bop_phase1_local_row_readiness(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
    )

    assert (
        summary["status"]
        == "objectstate_bop_phase1_local_row_readiness_prediction_reviewable"
    )
    assert summary["blocking_stage"] == "prediction_reviewable_identity_pending"
    assert summary["readiness"]["prediction_evidence_reviewable"] is True
    assert summary["readiness"]["identity_evidence_reviewable"] is False
    assert summary["readiness"]["phase1_has_any_reviewable_evidence"] is True
    assert any("resolve identity route blockers" in item for item in summary["next_actions"])


def test_object_state_audit_bop_phase1_local_row_cli(tmp_path, capsys):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "phase1-row"
    artifact_path = output_root / "objectstates.json"
    summary_path = tmp_path / "bop-phase1-local-row-summary.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    _write_json(artifact_path, _trainable_artifact(frame_count=3))

    assert (
        main(
            [
                "object-state",
                "audit-bop-phase1-local-row",
                str(scene_root),
                "--output-root",
                str(output_root),
                "--sample-id",
                "bop-ycbv-scene-000001",
                "--candidate-artifact",
                str(artifact_path),
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA}" in stdout
    assert "bop_phase1_local_row_status=objectstate_bop_phase1_local_row_readiness_identity_scenario_blocked" in stdout
    assert "blocking_stage=identity_scenario_metadata" in stdout
    assert "identity_route_status=objectstate_bop_identity_route_audit_blocked" in stdout
    assert "prediction_route_status=objectstate_bop_phase1_route_audit_handoff_ready" in stdout
    assert "readiness.candidate_artifact_binding_ready=true" in stdout
    assert "readiness.identity_scenario_metadata_ready=false" in stdout
    assert "rgbd_export_candidate=false" in stdout
    assert summary["blocking_stage"] == "identity_scenario_metadata"


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
            {"visib_fract": visibility_by_frame[frame_id]},
            {"visib_fract": 1.0},
        ]
    _write_json(root / "scene_camera.json", scene_camera)
    _write_json(root / "scene_gt.json", scene_gt)
    _write_json(root / "scene_gt_info.json", scene_gt_info)


def _write_gaussian_frames(root) -> None:
    (root / "gaussians").mkdir()
    for frame_id in range(3):
        (root / "gaussians" / f"{frame_id:06d}.ply").write_bytes(PLY_BYTES)


def _write_depth_frames(root) -> None:
    (root / "depth").mkdir()
    for frame_id in range(3):
        (root / "depth" / f"{frame_id:06d}.png").write_bytes(PNG_BYTES)


def _trainable_artifact(*, frame_count: int):
    object_states = []
    assignments = []
    for frame_index in range(frame_count):
        object_states.append(
            {
                "frame_index": frame_index,
                "states": [
                    _state(0, [0.1 + 0.01 * frame_index, 0.2, 0.3]),
                    _state(1, [0.4 + 0.01 * frame_index, 0.2, 0.3]),
                ],
                "derived_object_ids": [0, 1],
            }
        )
        assignments.append(
            {
                "frame_index": frame_index,
                "shape": [2, 2],
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
            }
        )
    return {
        "schema": "objgauss-trainable-kernel-model-artifact-v1",
        "kind": "trainable_kernel_mvp_model",
        "label": "fixture-bop-objectstates",
        "source": {
            "input": "outputs/captures/bop-ycbv-scene-000001/objectstates.json",
            "sample": None,
        },
        "training": {
            "schema": "objgauss-v1-trainable-kernel-mvp-v1",
            "frame_count": frame_count,
        },
        "renderer_api": {},
        "learned_parameters": {"decoder_colors": []},
        "assignments": assignments,
        "object_states": object_states,
        "artifact_policy": {
            "git_policy": "do_not_commit_training_outputs_by_default",
        },
    }


def _state(state_id: int, centroid: list[float]):
    return {
        "id": state_id,
        "slot_mass": 1.0,
        "confidence": 0.92,
        "mass_fraction": 0.5,
        "assignment_entropy": 0.0,
        "normalized_assignment_entropy": 0.0,
        "centroid": centroid,
        "bbox": [
            [centroid[0] - 0.01, centroid[1] - 0.01, centroid[2] - 0.01],
            [centroid[0] + 0.01, centroid[1] + 0.01, centroid[2] + 0.01],
        ],
        "feature": [centroid[0], centroid[1], centroid[2]],
        "status": "active",
        "diagnostics": [],
    }


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


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
