from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.pipelines.objectstate_controlled_identity_bundle_handoff import (
    OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA,
    objectstate_controlled_identity_bundle_handoff,
    validate_objectstate_controlled_identity_bundle_handoff_summary,
)
from objgauss.pipelines.objectstate_controlled_identity_evidence_package import (
    OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA,
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


def test_controlled_identity_bundle_handoff_runs_acceptance_and_handoff(tmp_path):
    _write_bundle(tmp_path, include_frame_files=True)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)

    summary = objectstate_controlled_identity_bundle_handoff(
        tmp_path,
        artifact,
        candidate_id="stable-objectstate-slots",
        max_centroid_distance=0.05,
        candidate_artifact_path=artifact_path,
        hash_files=True,
        hash_candidate_artifact=True,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA
    assert summary["status"] == "objectstate_controlled_identity_bundle_handoff_pass"
    assert summary["handoff_gates"] == {
        "capture_bundle_acceptance_pass": True,
        "identity_handoff_pass": True,
    }
    assert (
        summary["capture_bundle_acceptance"]["status"]
        == "objectstate_controlled_capture_bundle_acceptance_pass"
    )
    assert (
        summary["identity_handoff"]["status"]
        == "objectstate_controlled_identity_handoff_pass"
    )
    assert summary["identity_handoff"]["candidate_artifact_ref_match"]["matches"] is True
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert summary["controlled_real_manifest"]["evidence_rows"][0]["status"] == "pass"
    assert summary["controlled_real_summary"]["blocked_row_count"] == 2
    assert len(
        summary["capture_bundle_acceptance"]["capture_file_audit"]["file_records"]["rgb"][
            0
        ]["sha256"]
    ) == 64
    assert (
        len(
            summary["identity_handoff"]["candidate_artifact_file_audit"][
                "file_record"
            ]["sha256"]
        )
        == 64
    )
    assert validate_objectstate_controlled_identity_bundle_handoff_summary(summary) == summary


def test_controlled_identity_bundle_handoff_fails_without_real_bundle_files(tmp_path):
    _write_bundle(tmp_path, include_frame_files=False)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)

    summary = objectstate_controlled_identity_bundle_handoff(
        tmp_path,
        artifact,
        candidate_id="stable-objectstate-slots",
        max_centroid_distance=0.05,
        candidate_artifact_path=artifact_path,
    )

    assert summary["status"] == "objectstate_controlled_identity_bundle_handoff_fail"
    assert summary["handoff_gates"] == {
        "capture_bundle_acceptance_pass": False,
        "identity_handoff_pass": False,
    }
    assert (
        summary["capture_bundle_acceptance"]["status"]
        == "objectstate_controlled_capture_bundle_acceptance_fail"
    )
    assert (
        summary["identity_handoff"]["capture_file_audit"]["status"]
        == "objectstate_controlled_capture_file_audit_fail"
    )
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert "controlled capture bundle acceptance did not pass" in summary["issues"]
    assert summary["capture_bundle_acceptance"]["capture_file_audit"]["missing_files"]


def test_object_state_controlled_identity_bundle_handoff_cli_writes_artifacts(
    tmp_path,
    capsys,
):
    _write_bundle(tmp_path, include_frame_files=True)
    artifact_path = tmp_path / "objectstates.json"
    output_dir = tmp_path / "bundle-handoff"
    artifact_path.write_text(json.dumps(_trainable_artifact()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "controlled-identity-bundle-handoff",
                str(tmp_path),
                str(artifact_path),
                "--output-dir",
                str(output_dir),
                "--candidate-id",
                "cli-bundle-objectstate-slots",
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
    bundle_handoff = json.loads(
        (output_dir / "bundle-handoff-summary.json").read_text(encoding="utf-8")
    )
    handoff = json.loads((output_dir / "handoff-summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output_dir / "capture-manifest.json").read_text(encoding="utf-8"))
    acceptance = json.loads(
        (output_dir / "bundle-acceptance-summary.json").read_text(encoding="utf-8")
    )
    import_summary = json.loads(
        (output_dir / "bundle-import-summary.json").read_text(encoding="utf-8")
    )
    bundle_file_audit = json.loads(
        (output_dir / "bundle-file-audit.json").read_text(encoding="utf-8")
    )
    controlled_real_seed = json.loads(
        (output_dir / "controlled-real-seed.json").read_text(encoding="utf-8")
    )
    predictions = json.loads(
        (output_dir / "identity-predictions.json").read_text(encoding="utf-8")
    )
    identity_eval = json.loads(
        (output_dir / "identity-eval-summary.json").read_text(encoding="utf-8")
    )
    controlled_real = json.loads(
        (output_dir / "controlled-real.json").read_text(encoding="utf-8")
    )
    controlled_real_summary = json.loads(
        (output_dir / "controlled-real-summary.json").read_text(encoding="utf-8")
    )
    identity_evidence_package = json.loads(
        (output_dir / "identity-evidence-package-summary.json").read_text(
            encoding="utf-8"
        )
    )

    assert f"schema={OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA}" in stdout
    assert (
        "bundle_handoff_status=objectstate_controlled_identity_bundle_handoff_pass"
        in stdout
    )
    assert (
        "acceptance_status=objectstate_controlled_capture_bundle_acceptance_pass"
        in stdout
    )
    assert "handoff_status=objectstate_controlled_identity_handoff_pass" in stdout
    assert "bundle_file_audit_status=objectstate_controlled_capture_file_audit_pass" in stdout
    assert (
        "candidate_artifact_file_audit_status="
        "objectstate_controlled_candidate_artifact_file_audit_pass"
        in stdout
    )
    assert "candidate_artifact_ref_match=true" in stdout
    assert "identity_gate_status=objectstate_reality_gate_pass" in stdout
    assert "track_retrieval_recall_at_1=1.000000" in stdout
    assert "long_term_drift_rate=0.000000" in stdout
    assert "reconstruction_noise_robustness=1.000000" in stdout
    assert (
        "identity_evidence_package_status="
        "objectstate_controlled_identity_evidence_package_reviewable"
        in stdout
    )
    assert "identity_evidence_package_reviewable=true" in stdout

    assert bundle_handoff["status"] == "objectstate_controlled_identity_bundle_handoff_pass"
    assert handoff["status"] == "objectstate_controlled_identity_handoff_pass"
    assert manifest == import_summary["manifest"]
    assert acceptance == bundle_handoff["capture_bundle_acceptance"]
    assert bundle_file_audit == acceptance["capture_file_audit"]
    assert predictions["candidate"]["candidate_id"] == "cli-bundle-objectstate-slots"
    assert identity_eval["status"] == "objectstate_controlled_identity_eval_pass"
    assert controlled_real["evidence_rows"][0]["status"] == "pass"
    assert controlled_real_seed["ground_truth"] == {
        "identity": True,
        "pose": True,
        "action": True,
        "timestamp": True,
    }
    assert controlled_real_summary["blocked_row_count"] == 2
    assert (
        identity_evidence_package["schema"]
        == OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA
    )
    assert identity_evidence_package["status"] == (
        "objectstate_controlled_identity_evidence_package_reviewable"
    )
    assert identity_evidence_package["identity"]["identity_row_status"] == "pass"
    assert identity_evidence_package["issues"] == []
    assert "no missing files" in (output_dir / "bundle-missing-files.md").read_text(
        encoding="utf-8"
    )
    assert "prediction" in (output_dir / "blocked-rows.md").read_text(
        encoding="utf-8"
    )


def _write_bundle(root, *, include_frame_files: bool) -> None:
    (root / "sample.json").write_text(
        json.dumps(
            {
                "sample_id": "controlled-tabletop-cup-box-identity-001",
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
                "object_id,category,instance_label",
                "cup-001,cup,blue cup",
                "box-001,box,red box",
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


def _write_candidate_artifact_file(root, artifact):
    path = root / "objectstates.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def _trainable_artifact():
    object_states = []
    assignments = []
    for frame_index in range(3):
        cup_x = 0.1 + 0.01 * frame_index
        box_x = 0.4 + 0.01 * frame_index
        object_states.append(
            {
                "frame_index": frame_index,
                "states": [
                    _state(0, [cup_x, 0.2, 0.3]),
                    _state(1, [box_x, 0.2, 0.3]),
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
