from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
)
from objgauss.core.objectstate_bop_identity_route_audit import (
    OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA,
    objectstate_bop_identity_route_audit,
    validate_objectstate_bop_identity_route_audit_summary,
)
from objgauss.core.objectstate_phase1_evidence_ledger import (
    objectstate_phase1_evidence_ledger,
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


def test_bop_identity_route_audit_reports_missing_scene(tmp_path):
    summary = objectstate_bop_identity_route_audit(
        tmp_path / "missing-scene",
        output_root=tmp_path / "identity-package",
        sample_id="bop-ycbv-scene-000001",
    )

    assert summary["schema"] == OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA
    assert summary["status"] == "objectstate_bop_identity_route_audit_blocked"
    assert summary["readiness"]["bop_acceptance_available"] is False
    assert summary["readiness"]["route_ready_for_identity_handoff"] is False
    assert any("local BOP scene cannot be accepted" in item for item in summary["hard_blockers"])
    assert any("place a local BOP scene" in item for item in summary["next_actions"])
    assert validate_objectstate_bop_identity_route_audit_summary(summary) == summary


def test_bop_identity_route_audit_blocks_without_identity_scenario_metadata(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "identity-package"
    artifact_path = output_root / "objectstates.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    _write_json(artifact_path, _trainable_artifact(frame_count=3))

    summary = objectstate_bop_identity_route_audit(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_artifact=artifact_path,
    )

    assert summary["status"] == "objectstate_bop_identity_route_audit_blocked"
    assert summary["readiness"]["bop_acceptance_pass"] is True
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is True
    assert summary["readiness"]["candidate_artifact_valid"] is True
    assert summary["readiness"]["candidate_artifact_binding_ready"] is True
    assert summary["readiness"]["identity_scenario_metadata_ready"] is False
    assert summary["readiness"]["route_ready_for_identity_handoff"] is False
    assert (
        summary["identity_scenario_metadata_audit"]["readiness"][
            "occlusion_reappearance_present"
        ]
        is True
    )
    assert (
        summary["identity_scenario_metadata_audit"]["readiness"][
            "min_lighting_conditions_met"
        ]
        is False
    )
    assert (
        summary["identity_scenario_metadata_audit"]["readiness"][
            "camera_motion_present"
        ]
        is False
    )
    assert any("do not relax the identity scenario gate" in item for item in summary["next_actions"])


def test_bop_identity_route_audit_accepts_condition_sidecar_for_handoff(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "identity-package"
    artifact_path = output_root / "objectstates.json"
    sidecar_path = scene_root / "bop-condition-sidecar.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    _write_json(artifact_path, _trainable_artifact(frame_count=3))
    _write_json(sidecar_path, _condition_sidecar_payload())

    summary = objectstate_bop_identity_route_audit(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_artifact=artifact_path,
        condition_sidecar=sidecar_path,
    )

    assert (
        summary["status"]
        == "objectstate_bop_identity_route_audit_handoff_ready"
    )
    assert summary["readiness"]["identity_scenario_metadata_ready"] is True
    assert summary["readiness"]["route_ready_for_identity_handoff"] is True
    assert summary["identity_scenario_metadata_audit"]["readiness"] == {
        "min_frame_count_met": True,
        "occlusion_reappearance_present": True,
        "min_view_conditions_met": True,
        "min_lighting_conditions_met": True,
        "camera_motion_present": True,
    }
    assert (
        summary["identity_scenario_metadata_audit"]["scenario_coverage"][
            "max_camera_translation_m"
        ]
        >= 0.04
    )
    assert (
        summary["acceptance"]["manifest"]["frames"][2]["condition"]["view_id"]
        == "right"
    )


def test_bop_identity_route_audit_reports_existing_identity_evidence(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "identity-package"
    identity_dir = output_root / "identity-handoff"
    artifact_path = output_root / "objectstates.json"
    identity_summary_path = identity_dir / "identity-evidence-package-summary.json"
    ledger_path = output_root / "phase1-evidence-ledger.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    _write_json(artifact_path, _trainable_artifact(frame_count=3))
    _write_json(identity_summary_path, _identity_summary(row_status="pass"))
    _write_json(
        ledger_path,
        objectstate_phase1_evidence_ledger(identity_summaries=(identity_summary_path,)),
    )

    summary = objectstate_bop_identity_route_audit(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_artifact=artifact_path,
    )

    assert (
        summary["status"]
        == "objectstate_bop_identity_route_audit_identity_reviewable"
    )
    assert summary["readiness"]["identity_evidence_package_reviewable"] is True
    assert summary["readiness"]["phase1_evidence_ledger_identity_reviewable"] is True
    assert summary["readiness"]["route_has_reviewable_identity_evidence"] is True
    assert (
        summary["records"]["phase1_evidence_ledger"]["payload"]["maturity"]
        == "identity_reviewable"
    )
    assert any("prediction and intervention gates remain separate" in item for item in summary["hard_blockers"])


def test_object_state_audit_bop_identity_route_cli(tmp_path, capsys):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "identity-package"
    artifact_path = output_root / "objectstates.json"
    summary_path = tmp_path / "bop-identity-route-summary.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    _write_json(artifact_path, _trainable_artifact(frame_count=3))

    assert (
        main(
            [
                "object-state",
                "audit-bop-identity-route",
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

    assert f"schema={OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA}" in stdout
    assert "bop_identity_route_status=objectstate_bop_identity_route_audit_blocked" in stdout
    assert "readiness.candidate_artifact_binding_ready=true" in stdout
    assert "readiness.identity_scenario_metadata_ready=false" in stdout
    assert "identity_scenario.camera_motion_present=false" in stdout
    assert "next_action=" in stdout
    assert summary["status"] == "objectstate_bop_identity_route_audit_blocked"


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


def _identity_summary(*, row_status: str):
    gates = {
        "required_files_present": True,
        "required_json_schemas_valid": True,
        "sample_ids_consistent": True,
        "capture_file_audit_pass": True,
        "candidate_artifact_file_audit_pass": True,
        "candidate_artifact_ref_match": True,
        "identity_scenario_audit_pass": True,
        "identity_predictions_present": True,
        "identity_eval_present": True,
        "identity_row_present": True,
        "identity_row_is_pass_or_fail": True,
        "controlled_real_output_matches_eval": True,
        "standalone_outputs_match_handoff": True,
        "identity_only_gate_does_not_require_prediction_or_intervention": True,
    }
    return {
        "schema": "objgauss-objectstate-controlled-identity-evidence-package-v1",
        "kind": "objectstate_controlled_identity_evidence_package",
        "status": "objectstate_controlled_identity_evidence_package_reviewable",
        "package_root": "/tmp/identity-package",
        "sample_id": "bop-ycbv-scene-000001",
        "files": [_file_record("handoff_summary")],
        "sample_consistency": {
            "consistent": True,
            "sample_id": "bop-ycbv-scene-000001",
            "unique_sample_ids": ["bop-ycbv-scene-000001"],
            "values": {"handoff_summary": "bop-ycbv-scene-000001"},
        },
        "identity": {
            "identity_eval_present": True,
            "identity_eval_status": (
                "objectstate_controlled_identity_eval_pass"
                if row_status == "pass"
                else "objectstate_controlled_identity_eval_fail"
            ),
            "identity_prediction_count": 6,
            "identity_row_present": True,
            "identity_row_status": row_status,
        },
        "evidence": {},
        "handoff_consistency": {
            "controlled_real_matches_eval": True,
            "standalone_outputs_match_handoff": True,
            "issues": [],
        },
        "reviewability_gates": gates,
        "issues": [],
        "claim_policy": _identity_claim_policy(),
        "non_goals": _non_goals(),
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


def _file_record(key: str):
    return {
        "key": key,
        "path": f"/tmp/{key}.json",
        "required": True,
        "kind": "json",
        "exists": True,
        "is_file": True,
        "size_bytes": 128,
        "schema": "fixture-schema",
        "expected_schema": "fixture-schema",
        "schema_ok": True,
        "validator_ok": True,
        "sample_id": "bop-ycbv-scene-000001",
        "status": "fixture-status",
        "issues": [],
    }


def _identity_claim_policy():
    return {
        "read_only_audit": True,
        "checks_local_identity_evidence_package": True,
        "requires_real_gaussian_file_acceptance": True,
        "requires_candidate_artifact_audit": True,
        "requires_identity_scenario_audit": True,
        "reviewable_allows_identity_pass_or_fail": True,
        "does_not_create_ground_truth": True,
        "does_not_run_identity_eval": True,
        "does_not_claim_metric_pass": True,
        "does_not_claim_prediction_or_intervention_gate": True,
        "does_not_claim_world_model": True,
    }


def _non_goals():
    return {
        "captures_video": False,
        "creates_ground_truth": False,
        "reconstructs_gaussians": False,
        "runs_tracking_model": False,
        "runs_identity_model": False,
        "runs_prediction_model": False,
        "runs_intervention_model": False,
        "trains_gaussian_model": False,
        "trains_dynamics_model": False,
        "writes_public_samples": False,
        "uses_replay_buffer": False,
        "uses_diffusion": False,
        "mutates_viewer_defaults": False,
    }


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
