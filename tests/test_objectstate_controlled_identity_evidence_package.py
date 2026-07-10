from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.pipelines.objectstate_controlled_identity_evidence_package import (
    OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA,
    objectstate_controlled_identity_evidence_package,
    validate_objectstate_controlled_identity_evidence_package_summary,
)
from objgauss.pipelines.objectstate_controlled_identity_handoff import (
    objectstate_controlled_identity_handoff,
)
from objgauss.pipelines.trainable_artifact import TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA

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


def test_controlled_identity_evidence_package_is_reviewable(tmp_path):
    package_dir = _write_identity_package_from_handoff(tmp_path)

    summary = objectstate_controlled_identity_evidence_package(package_dir)

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA
    assert (
        summary["status"]
        == "objectstate_controlled_identity_evidence_package_reviewable"
    )
    assert summary["sample_id"] == "controlled-tabletop-cup-box-identity-001"
    assert summary["identity"]["identity_eval_status"] == (
        "objectstate_controlled_identity_eval_pass"
    )
    assert summary["identity"]["identity_prediction_count"] == 6
    assert summary["identity"]["identity_row_status"] == "pass"
    assert summary["evidence"]["capture_file_audit_pass"] is True
    assert summary["evidence"]["candidate_artifact_file_audit_pass"] is True
    assert summary["evidence"]["candidate_artifact_ref_match"] is True
    assert summary["evidence"]["identity_scenario_audit_pass"] is True
    assert all(summary["reviewability_gates"].values())
    assert summary["issues"] == []
    assert (
        validate_objectstate_controlled_identity_evidence_package_summary(summary)
        == summary
    )


def test_controlled_identity_evidence_package_allows_failed_identity_row_for_review(
    tmp_path,
):
    package_dir = _write_identity_package_from_handoff(
        tmp_path,
        slot_ids_by_frame=((0, 1), (1, 0), (1, 0)),
    )

    summary = objectstate_controlled_identity_evidence_package(package_dir)

    assert (
        summary["status"]
        == "objectstate_controlled_identity_evidence_package_reviewable"
    )
    assert summary["identity"]["identity_eval_status"] == (
        "objectstate_controlled_identity_eval_fail"
    )
    assert summary["identity"]["identity_row_status"] == "fail"
    assert summary["reviewability_gates"]["identity_row_is_pass_or_fail"] is True
    assert summary["reviewability_gates"]["identity_only_gate_does_not_require_prediction_or_intervention"] is True
    assert all(summary["reviewability_gates"].values())
    assert summary["issues"] == []


def test_controlled_identity_evidence_package_reports_missing_identity_eval(tmp_path):
    package_dir = _write_identity_package_from_handoff(tmp_path)
    (package_dir / "identity-eval-summary.json").unlink()

    summary = objectstate_controlled_identity_evidence_package(package_dir)

    assert (
        summary["status"]
        == "objectstate_controlled_identity_evidence_package_incomplete"
    )
    assert summary["reviewability_gates"]["required_files_present"] is False
    assert summary["reviewability_gates"]["identity_eval_present"] is False
    assert any("identity_eval_summary" in issue for issue in summary["issues"])


def test_object_state_audit_controlled_identity_evidence_package_cli(
    tmp_path,
    capsys,
):
    package_dir = _write_identity_package_from_handoff(tmp_path)
    summary_path = package_dir / "identity-evidence-package-summary.json"

    assert (
        main(
            [
                "object-state",
                "audit-controlled-identity-evidence-package",
                str(package_dir),
                "--summary-output",
                str(summary_path),
                "--require-reviewable",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA}" in stdout
    assert "reviewable=true" in stdout
    assert "capture_file_audit_pass=true" in stdout
    assert "candidate_artifact_file_audit_pass=true" in stdout
    assert "candidate_artifact_ref_match=true" in stdout
    assert "identity_scenario_audit_pass=true" in stdout
    assert "identity_eval_status=objectstate_controlled_identity_eval_pass" in stdout
    assert "identity_prediction_count=6" in stdout
    assert "identity_row_status=pass" in stdout
    assert "issue_count=0" in stdout
    assert (
        summary["status"]
        == "objectstate_controlled_identity_evidence_package_reviewable"
    )


def _write_identity_package_from_handoff(
    root,
    *,
    slot_ids_by_frame: tuple[tuple[int, int], ...] = ((0, 1), (0, 1), (0, 1)),
):
    _write_capture_bundle_files(root)
    capture = _capture_manifest()
    artifact = _trainable_artifact(slot_ids_by_frame=slot_ids_by_frame)
    artifact_path = _write_candidate_artifact_file(root, artifact)
    handoff = objectstate_controlled_identity_handoff(
        capture,
        artifact,
        candidate_id="stable-objectstate-slots",
        artifact_refs=(str(artifact_path),),
        max_centroid_distance=0.05,
        capture_root=root,
        hash_files=True,
        candidate_artifact_path=artifact_path,
        hash_candidate_artifact=True,
    )
    package_dir = root / "identity-package"
    _write_json(package_dir / "capture-manifest.json", capture)
    _write_json(package_dir / "capture-file-audit.json", handoff["capture_file_audit"])
    (package_dir / "capture-missing-files.md").write_text(
        handoff["capture_file_audit"]["missing_files_markdown"],
        encoding="utf-8",
    )
    _write_json(
        package_dir / "candidate-artifact-file-audit.json",
        handoff["candidate_artifact_file_audit"],
    )
    _write_json(
        package_dir / "identity-scenario-audit.json",
        handoff["identity_scenario_audit"],
    )
    _write_json(package_dir / "identity-predictions.json", handoff["identity_predictions"])
    _write_json(package_dir / "identity-eval-summary.json", handoff["identity_eval"])
    _write_json(package_dir / "controlled-real.json", handoff["controlled_real_manifest"])
    _write_json(
        package_dir / "controlled-real-summary.json",
        handoff["controlled_real_summary"],
    )
    (package_dir / "blocked-rows.md").write_text(
        handoff["controlled_real_summary"]["blocked_rows_markdown"],
        encoding="utf-8",
    )
    _write_json(package_dir / "handoff-summary.json", handoff)
    return package_dir


def _capture_manifest():
    frames = []
    for frame_index, timestamp in enumerate((0.0, 0.033333, 0.066667)):
        frame_objects = []
        for object_id, x in (("cup-001", 0.1), ("box-001", 0.4)):
            frame_objects.append(
                {
                    "object_id": object_id,
                    "visible": frame_index != 1,
                    "occlusion_fraction": 0.75 if frame_index == 1 else 0.0,
                    "pose": {
                        "position": [x + 0.01 * frame_index, 0.2, 0.3],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            )
        frames.append(
            {
                "frame_id": f"frame-{frame_index:06d}",
                "timestamp": timestamp,
                "observation": {
                    "rgb": f"rgb/{frame_index:06d}.png",
                    "gaussian": f"gaussians/{frame_index:06d}.ply",
                },
                "objects": frame_objects,
                "condition": {
                    "view_id": "front" if frame_index < 2 else "right",
                    "lighting_id": "bright" if frame_index == 0 else "dim",
                    "camera_pose": {
                        "position": [0.02 * frame_index, 0.0, 0.0],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                },
            }
        )
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


def _write_capture_bundle_files(root) -> None:
    (root / "rgb").mkdir(parents=True, exist_ok=True)
    (root / "gaussians").mkdir(parents=True, exist_ok=True)
    for frame_index in range(3):
        (root / "rgb" / f"{frame_index:06d}.png").write_bytes(PNG_BYTES)
        (root / "gaussians" / f"{frame_index:06d}.ply").write_bytes(PLY_BYTES)


def _write_candidate_artifact_file(root, artifact):
    path = root / "objectstates.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _trainable_artifact(
    *,
    slot_ids_by_frame: tuple[tuple[int, int], ...],
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
        "identity_evidence": {
            "reconstruction_noise_robustness": 1.0,
            "reconstruction_noise_variant_count": 2,
            "source": "fixture repeated Gaussian reconstruction noise variants",
        },
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


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
