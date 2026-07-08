from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.core.objectstate_controlled_identity_handoff import (
    OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
    objectstate_controlled_identity_handoff,
    validate_objectstate_controlled_identity_handoff_summary,
)
from objgauss.core.trainable_artifact import TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA

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


def test_controlled_identity_handoff_runs_identity_only_reality_gate(tmp_path):
    _write_capture_bundle_files(tmp_path)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)

    summary = objectstate_controlled_identity_handoff(
        _capture_manifest(),
        artifact,
        candidate_id="stable-objectstate-slots",
        artifact_refs=(str(artifact_path),),
        max_centroid_distance=0.05,
        capture_root=tmp_path,
        hash_files=True,
        candidate_artifact_path=artifact_path,
        hash_candidate_artifact=True,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA
    assert summary["status"] == "objectstate_controlled_identity_handoff_pass"
    assert (
        summary["capture_file_audit"]["status"]
        == "objectstate_controlled_capture_file_audit_pass"
    )
    assert summary["capture_file_audit"]["readiness"]["frame_formats_valid"] is True
    assert (
        summary["candidate_artifact_file_audit"]["status"]
        == "objectstate_controlled_candidate_artifact_file_audit_pass"
    )
    assert len(summary["capture_file_audit"]["file_records"]["rgb"][0]["sha256"]) == 64
    assert len(summary["candidate_artifact_file_audit"]["file_record"]["sha256"]) == 64
    assert summary["candidate_artifact_ref_match"]["matches"] is True
    assert (
        summary["identity_scenario_audit"]["status"]
        == "objectstate_controlled_identity_scenario_audit_pass"
    )
    assert summary["identity_scenario_audit"]["readiness"] == {
        "min_frame_count_met": True,
        "occlusion_reappearance_present": True,
        "min_view_conditions_met": True,
        "min_lighting_conditions_met": True,
        "camera_motion_present": True,
    }
    assert summary["identity_scenario_audit"]["scenario_coverage"] == {
        "view_ids": ["front", "right"],
        "view_condition_count": 2,
        "lighting_ids": ["bright", "dim"],
        "lighting_condition_count": 2,
        "camera_pose_count": 3,
        "max_camera_translation_m": 0.04,
    }
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert summary["controlled_real_manifest"]["evidence_rows"][0]["status"] == "pass"
    assert summary["controlled_real_manifest"]["evidence_rows"][1]["status"] == "blocked"
    assert summary["controlled_real_manifest"]["evidence_rows"][2]["status"] == "blocked"
    gate = summary["controlled_real_summary"]["gate"]
    assert gate["status"] == "objectstate_reality_gate_pass"
    assert gate["hard_blockers"] == []
    assert summary["controlled_real_summary"]["blocked_row_count"] == 2
    assert validate_objectstate_controlled_identity_handoff_summary(summary) is summary


def test_controlled_identity_handoff_surfaces_failed_identity_gate(tmp_path):
    _write_capture_bundle_files(tmp_path)
    artifact = _trainable_artifact(slot_ids_by_frame=((0, 1), (1, 0), (1, 0)))
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)

    summary = objectstate_controlled_identity_handoff(
        _capture_manifest(),
        artifact,
        candidate_id="fragmented-objectstate-slots",
        artifact_refs=(str(artifact_path),),
        capture_root=tmp_path,
        candidate_artifact_path=artifact_path,
    )

    assert summary["status"] == "objectstate_controlled_identity_handoff_fail"
    assert (
        summary["capture_file_audit"]["status"]
        == "objectstate_controlled_capture_file_audit_pass"
    )
    assert (
        summary["candidate_artifact_file_audit"]["status"]
        == "objectstate_controlled_candidate_artifact_file_audit_pass"
    )
    assert summary["candidate_artifact_ref_match"]["matches"] is True
    assert (
        summary["identity_scenario_audit"]["status"]
        == "objectstate_controlled_identity_scenario_audit_pass"
    )
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_fail"
    assert summary["controlled_real_manifest"]["evidence_rows"][0]["status"] == "fail"
    assert summary["controlled_real_summary"]["gate"]["status"] == "objectstate_reality_gate_fail"
    assert "failed_rows_absent" in summary["controlled_real_summary"]["gate"]["hard_blockers"]


def test_controlled_identity_handoff_requires_capture_file_audit_pass(tmp_path):
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)

    summary = objectstate_controlled_identity_handoff(
        _capture_manifest(),
        artifact,
        candidate_id="stable-objectstate-slots",
        artifact_refs=(str(artifact_path),),
        capture_root=tmp_path,
        candidate_artifact_path=artifact_path,
    )

    assert summary["status"] == "objectstate_controlled_identity_handoff_fail"
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert (
        summary["capture_file_audit"]["status"]
        == "objectstate_controlled_capture_file_audit_fail"
    )
    assert (
        summary["candidate_artifact_file_audit"]["status"]
        == "objectstate_controlled_candidate_artifact_file_audit_pass"
    )
    assert summary["candidate_artifact_ref_match"]["matches"] is True
    assert (
        summary["identity_scenario_audit"]["status"]
        == "objectstate_controlled_identity_scenario_audit_pass"
    )
    assert summary["capture_file_audit"]["missing_files"]
    assert summary["controlled_real_summary"]["gate"]["status"] == "objectstate_reality_gate_pass"


def test_controlled_identity_handoff_rejects_text_placeholder_frame_files(tmp_path):
    _write_capture_bundle_files(tmp_path, valid_formats=False)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)

    summary = objectstate_controlled_identity_handoff(
        _capture_manifest(),
        artifact,
        candidate_id="stable-objectstate-slots",
        artifact_refs=(str(artifact_path),),
        capture_root=tmp_path,
        candidate_artifact_path=artifact_path,
    )

    assert summary["status"] == "objectstate_controlled_identity_handoff_fail"
    assert (
        summary["capture_file_audit"]["status"]
        == "objectstate_controlled_capture_file_audit_fail"
    )
    assert summary["capture_file_audit"]["readiness"]["frame_formats_valid"] is False
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert summary["controlled_real_summary"]["gate"]["status"] == "objectstate_reality_gate_pass"


def test_controlled_identity_handoff_requires_candidate_artifact_file_audit_pass(
    tmp_path,
):
    _write_capture_bundle_files(tmp_path)

    summary = objectstate_controlled_identity_handoff(
        _capture_manifest(),
        _trainable_artifact(),
        candidate_id="stable-objectstate-slots",
        capture_root=tmp_path,
    )

    assert summary["status"] == "objectstate_controlled_identity_handoff_fail"
    assert (
        summary["capture_file_audit"]["status"]
        == "objectstate_controlled_capture_file_audit_pass"
    )
    assert (
        summary["candidate_artifact_file_audit"]["status"]
        == "objectstate_controlled_candidate_artifact_file_audit_fail"
    )
    assert (
        summary["candidate_artifact_file_audit"]["file_record"]["missing_reason"]
        == "candidate artifact path not provided"
    )
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert summary["controlled_real_summary"]["gate"]["status"] == "objectstate_reality_gate_pass"


def test_controlled_identity_handoff_requires_candidate_artifact_ref_match(
    tmp_path,
):
    _write_capture_bundle_files(tmp_path)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)

    summary = objectstate_controlled_identity_handoff(
        _capture_manifest(),
        artifact,
        candidate_id="stable-objectstate-slots",
        artifact_refs=("outputs/controlled-real/mismatched-objectstates.json",),
        capture_root=tmp_path,
        candidate_artifact_path=artifact_path,
    )

    assert summary["status"] == "objectstate_controlled_identity_handoff_fail"
    assert (
        summary["candidate_artifact_file_audit"]["status"]
        == "objectstate_controlled_candidate_artifact_file_audit_pass"
    )
    assert summary["candidate_artifact_ref_match"]["matches"] is False
    assert (
        summary["candidate_artifact_ref_match"]["missing_reason"]
        == "audited candidate artifact path not in artifact_refs"
    )
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert summary["controlled_real_summary"]["gate"]["status"] == "objectstate_reality_gate_pass"


def test_controlled_identity_handoff_requires_identity_scenario_challenge(
    tmp_path,
):
    _write_capture_bundle_files(tmp_path)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)

    summary = objectstate_controlled_identity_handoff(
        _capture_manifest(include_occlusion=False),
        artifact,
        candidate_id="stable-objectstate-slots",
        artifact_refs=(str(artifact_path),),
        capture_root=tmp_path,
        candidate_artifact_path=artifact_path,
    )

    assert summary["status"] == "objectstate_controlled_identity_handoff_fail"
    assert (
        summary["identity_scenario_audit"]["status"]
        == "objectstate_controlled_identity_scenario_audit_fail"
    )
    assert summary["identity_scenario_audit"]["readiness"] == {
        "min_frame_count_met": True,
        "occlusion_reappearance_present": False,
        "min_view_conditions_met": True,
        "min_lighting_conditions_met": True,
        "camera_motion_present": True,
    }
    assert "clear-visible-before" in summary["identity_scenario_audit"]["issues"][0]
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert summary["controlled_real_summary"]["gate"]["status"] == "objectstate_reality_gate_pass"


def test_controlled_identity_handoff_requires_clear_visible_reappearance(
    tmp_path,
):
    _write_capture_bundle_files(tmp_path)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)
    capture = _capture_manifest(include_occlusion=False)
    for frame in capture["frames"]:
        for item in frame["objects"]:
            item["visible"] = True
            item["occlusion_fraction"] = 0.75

    summary = objectstate_controlled_identity_handoff(
        capture,
        artifact,
        candidate_id="stable-objectstate-slots",
        artifact_refs=(str(artifact_path),),
        capture_root=tmp_path,
        candidate_artifact_path=artifact_path,
    )

    assert summary["status"] == "objectstate_controlled_identity_handoff_fail"
    assert (
        summary["identity_scenario_audit"]["status"]
        == "objectstate_controlled_identity_scenario_audit_fail"
    )
    assert summary["identity_scenario_audit"]["readiness"] == {
        "min_frame_count_met": True,
        "occlusion_reappearance_present": False,
        "min_view_conditions_met": True,
        "min_lighting_conditions_met": True,
        "camera_motion_present": True,
    }
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert summary["controlled_real_summary"]["gate"]["status"] == "objectstate_reality_gate_pass"


def test_controlled_identity_handoff_requires_real_identity_condition_coverage(
    tmp_path,
):
    _write_capture_bundle_files(tmp_path)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)

    summary = objectstate_controlled_identity_handoff(
        _capture_manifest(include_conditions=False),
        artifact,
        candidate_id="stable-objectstate-slots",
        artifact_refs=(str(artifact_path),),
        capture_root=tmp_path,
        candidate_artifact_path=artifact_path,
    )

    assert summary["status"] == "objectstate_controlled_identity_handoff_fail"
    assert (
        summary["identity_scenario_audit"]["status"]
        == "objectstate_controlled_identity_scenario_audit_fail"
    )
    assert summary["identity_scenario_audit"]["readiness"] == {
        "min_frame_count_met": True,
        "occlusion_reappearance_present": True,
        "min_view_conditions_met": False,
        "min_lighting_conditions_met": False,
        "camera_motion_present": False,
    }
    assert summary["identity_scenario_audit"]["scenario_coverage"] == {
        "view_ids": [],
        "view_condition_count": 0,
        "lighting_ids": [],
        "lighting_condition_count": 0,
        "camera_pose_count": 0,
        "max_camera_translation_m": 0.0,
    }
    assert "frame.condition.view_id" in summary["identity_scenario_audit"]["issues"][0]
    assert "frame.condition.lighting_id" in summary["identity_scenario_audit"]["issues"][1]
    assert "frame.condition.camera_pose" in summary["identity_scenario_audit"]["issues"][2]
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert summary["controlled_real_summary"]["gate"]["status"] == "objectstate_reality_gate_pass"


def test_object_state_controlled_identity_handoff_cli_writes_artifacts(tmp_path, capsys):
    _write_capture_bundle_files(tmp_path)
    capture_path = tmp_path / "capture.json"
    artifact_path = tmp_path / "objectstates.json"
    output_dir = tmp_path / "handoff"
    capture_path.write_text(json.dumps(_capture_manifest()), encoding="utf-8")
    artifact_path.write_text(json.dumps(_trainable_artifact()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "controlled-identity-handoff",
                str(capture_path),
                str(artifact_path),
                "--output-dir",
                str(output_dir),
                "--candidate-id",
                "cli-objectstate-slots",
                "--max-centroid-distance",
                "0.05",
                "--hash-files",
                "--hash-candidate-artifact",
                "--require-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    handoff = json.loads((output_dir / "handoff-summary.json").read_text(encoding="utf-8"))
    capture_file_audit = json.loads(
        (output_dir / "capture-file-audit.json").read_text(encoding="utf-8")
    )
    candidate_artifact_file_audit = json.loads(
        (output_dir / "candidate-artifact-file-audit.json").read_text(
            encoding="utf-8"
        )
    )
    identity_scenario_audit = json.loads(
        (output_dir / "identity-scenario-audit.json").read_text(encoding="utf-8")
    )
    capture_missing = (output_dir / "capture-missing-files.md").read_text(
        encoding="utf-8"
    )
    predictions = json.loads((output_dir / "identity-predictions.json").read_text(encoding="utf-8"))
    identity_eval = json.loads((output_dir / "identity-eval-summary.json").read_text(encoding="utf-8"))
    controlled_real = json.loads((output_dir / "controlled-real.json").read_text(encoding="utf-8"))
    controlled_real_summary = json.loads((output_dir / "controlled-real-summary.json").read_text(encoding="utf-8"))
    blocked_rows = (output_dir / "blocked-rows.md").read_text(encoding="utf-8")

    assert f"schema={OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA}" in stdout
    assert "handoff_status=objectstate_controlled_identity_handoff_pass" in stdout
    assert (
        "capture_file_audit_status=objectstate_controlled_capture_file_audit_pass"
        in stdout
    )
    assert (
        "candidate_artifact_file_audit_status="
        "objectstate_controlled_candidate_artifact_file_audit_pass"
        in stdout
    )
    assert "candidate_artifact_ref_match=true" in stdout
    assert (
        "identity_scenario_audit_status="
        "objectstate_controlled_identity_scenario_audit_pass"
        in stdout
    )
    assert "identity_scenario_view_conditions=2" in stdout
    assert "identity_scenario_lighting_conditions=2" in stdout
    assert "identity_scenario_max_camera_translation_m=0.040000" in stdout
    assert "identity_gate_status=objectstate_reality_gate_pass" in stdout
    assert capture_file_audit["status"] == "objectstate_controlled_capture_file_audit_pass"
    assert (
        candidate_artifact_file_audit["status"]
        == "objectstate_controlled_candidate_artifact_file_audit_pass"
    )
    assert len(capture_file_audit["file_records"]["rgb"][0]["sha256"]) == 64
    assert len(candidate_artifact_file_audit["file_record"]["sha256"]) == 64
    assert handoff["candidate_artifact_ref_match"]["matches"] is True
    assert identity_scenario_audit["status"] == (
        "objectstate_controlled_identity_scenario_audit_pass"
    )
    assert identity_scenario_audit["scenario_coverage"]["view_condition_count"] == 2
    assert identity_scenario_audit["scenario_coverage"]["lighting_condition_count"] == 2
    assert identity_scenario_audit["scenario_coverage"]["camera_pose_count"] == 3
    assert "no missing files" in capture_missing
    assert predictions["candidate"]["candidate_id"] == "cli-objectstate-slots"
    assert identity_eval["status"] == "objectstate_controlled_identity_eval_pass"
    assert controlled_real["evidence_rows"][0]["status"] == "pass"
    assert controlled_real_summary["blocked_row_count"] == 2
    assert "prediction" in blocked_rows
    assert handoff["status"] == "objectstate_controlled_identity_handoff_pass"
    assert handoff["capture_file_audit"] == capture_file_audit
    assert handoff["candidate_artifact_file_audit"] == candidate_artifact_file_audit
    assert handoff["identity_scenario_audit"] == identity_scenario_audit


def _capture_manifest(*, include_occlusion: bool = True, include_conditions: bool = True):
    frames = []
    for frame_index, timestamp in enumerate((0.0, 0.033333, 0.066667)):
        frame_objects = []
        for object_id, x in (("cup-001", 0.1), ("box-001", 0.4)):
            frame_objects.append(
                {
                    "object_id": object_id,
                    "visible": (frame_index != 1) if include_occlusion else True,
                    "occlusion_fraction": (
                        0.75 if include_occlusion and frame_index == 1 else 0.0
                    ),
                    "pose": {
                        "position": [x + 0.01 * frame_index, 0.2, 0.3],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            )
        frame = {
            "frame_id": f"frame-{frame_index:06d}",
            "timestamp": timestamp,
            "observation": {
                "rgb": f"rgb/{frame_index:06d}.png",
                "gaussian": f"gaussians/{frame_index:06d}.ply",
            },
            "objects": frame_objects,
        }
        if include_conditions:
            frame["condition"] = {
                "view_id": "front" if frame_index < 2 else "right",
                "lighting_id": "bright" if frame_index == 0 else "dim",
                "camera_pose": {
                    "position": [0.02 * frame_index, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        frames.append(frame)
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "controlled-tabletop-cup-box-identity-001",
            "source_kind": "controlled_real",
            "object_category": "cup_box",
            "scenario": "cross_view_occlusion_reappearance",
            "fps": 30.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/controlled-real/cup-box-identity-001/capture.json",
                "outputs/controlled-real/cup-box-identity-001/rgb/",
                "outputs/controlled-real/cup-box-identity-001/gaussians/",
            ],
            "license": "local controlled capture; not public release",
        },
        "objects": [
            {"object_id": "cup-001", "category": "cup", "instance_label": "blue cup"},
            {"object_id": "box-001", "category": "box", "instance_label": "red box"},
        ],
        "actions": [],
        "frames": frames,
    }


def _write_capture_bundle_files(root, *, valid_formats: bool = True) -> None:
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "gaussians").mkdir(parents=True, exist_ok=True)
    for frame_index in range(3):
        rgb_bytes = PNG_BYTES if valid_formats else f"rgb-{frame_index}".encode()
        ply_bytes = PLY_BYTES if valid_formats else f"ply-{frame_index}".encode()
        (root / "rgb" / f"{frame_index:06d}.png").write_bytes(rgb_bytes)
        (root / "gaussians" / f"{frame_index:06d}.ply").write_bytes(ply_bytes)


def _write_candidate_artifact_file(root, artifact):
    path = root / "objectstates.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _trainable_artifact(
    *,
    slot_ids_by_frame: tuple[tuple[int, int], ...] = ((0, 1), (0, 1), (0, 1)),
):
    object_states = []
    assignments = []
    for frame_index, slot_ids in enumerate(slot_ids_by_frame):
        cup_slot, box_slot = slot_ids
        cup_x = 0.1 + 0.01 * frame_index
        box_x = 0.4 + 0.01 * frame_index
        object_states.append(
            {
                "frame_index": frame_index,
                "states": [
                    _state(cup_slot, [cup_x, 0.2, 0.3]),
                    _state(box_slot, [box_x, 0.2, 0.3]),
                ],
                "derived_object_ids": [cup_slot, box_slot],
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
        "schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "kind": "trainable_kernel_mvp_model",
        "label": "fixture-trainable-objectstates",
        "source": {
            "input": "outputs/controlled-real/cup-box-identity-001/objectstates.json",
            "sample": None,
        },
        "training": {
            "schema": "objgauss-v1-trainable-kernel-mvp-v1",
            "frame_count": len(object_states),
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
