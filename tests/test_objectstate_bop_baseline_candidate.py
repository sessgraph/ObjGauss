from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_bop_baseline_candidate import (
    OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA,
    validate_objectstate_bop_baseline_candidate_summary,
    write_objectstate_bop_gaussian_centroid_baseline_candidate,
)
from objgauss.core.objectstate_bop_capture_adapter import (
    objectstate_bop_capture_condition_sidecar_summary,
)
from objgauss.core.objectstate_bop_local_row_batch_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
)
from objgauss.core.objectstate_bop_phase1_authoring_progress import (
    objectstate_bop_phase1_authoring_progress,
)
from objgauss.core.objectstate_bop_phase1_sample_workspace import (
    objectstate_bop_phase1_sample_workspaces,
)
from objgauss.core.trainable_artifact import (
    TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
    validate_trainable_kernel_model_artifact,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n"


def test_bop_gaussian_centroid_baseline_candidate_writes_artifact(tmp_path):
    _write_bop_scene(tmp_path, frame_count=2)
    _write_gaussian_frames(tmp_path, frame_count=2)
    output = tmp_path / "objectstates.json"

    summary = write_objectstate_bop_gaussian_centroid_baseline_candidate(
        tmp_path,
        output=output,
        sample_id="bop-ycbv-test-000001",
    )

    assert summary["schema"] == OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA
    assert summary["status"] == "objectstate_bop_baseline_candidate_written"
    assert validate_objectstate_bop_baseline_candidate_summary(summary) == summary
    assert summary["row_counts"] == {
        "frames": 2,
        "states": 2,
        "total_gaussians": 4,
    }
    artifact = _read_json(output)
    assert artifact["schema"] == TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA
    assert validate_trainable_kernel_model_artifact(artifact) is True
    assert artifact["artifact_policy"]["baseline_candidate_only"] is True
    assert artifact["object_states"][0]["states"][0]["centroid"] == [1.0, 0.0, 0.0]
    assert artifact["object_states"][1]["states"][0]["centroid"] == [2.0, 0.0, 0.0]
    assert artifact["object_states"][0]["states"][0]["bbox"] == [
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
    ]
    assert any("bop-identity-handoff" in command for command in summary["next_commands"])


def test_bop_gaussian_centroid_baseline_candidate_cli(tmp_path, capsys):
    _write_bop_scene(tmp_path, frame_count=2)
    _write_gaussian_frames(tmp_path, frame_count=2)
    output = tmp_path / "objectstates.json"
    summary_output = tmp_path / "baseline-summary.json"

    assert (
        main(
            [
                "object-state",
                "generate-bop-objectstate-baseline-candidate",
                str(tmp_path),
                "--output",
                str(output),
                "--sample-id",
                "bop-ycbv-test-000001",
                "--summary-output",
                str(summary_output),
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_output)
    assert f"schema={OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA}" in stdout
    assert "bop_baseline_candidate_status=objectstate_bop_baseline_candidate_written" in stdout
    assert "total_gaussians=4" in stdout
    assert summary["readiness"]["ready_for_identity_handoff"] is True


def test_bop_baseline_candidate_unblocks_authoring_progress_candidate_path(tmp_path):
    scene_root = tmp_path / "bop" / "ycbv" / "test" / "000001"
    batch_spec = tmp_path / "workspace" / "bop-local-row-batch.json"
    artifact_root = tmp_path / "workspace" / "artifacts" / "bop-ycbv-test-000001"
    sidecar_path = artifact_root / "bop-condition-sidecar.json"
    candidate_path = artifact_root / "objectstates.json"
    _write_bop_scene(scene_root, frame_count=2)
    _write_gaussian_frames(scene_root, frame_count=2)
    _write_batch_spec(
        batch_spec,
        scene_root="../bop/ycbv/test/000001",
        candidate_artifact="artifacts/bop-ycbv-test-000001/objectstates.json",
        condition_sidecar="artifacts/bop-ycbv-test-000001/bop-condition-sidecar.json",
        max_frames=2,
    )
    objectstate_bop_phase1_sample_workspaces(batch_spec)
    _write_json(
        sidecar_path,
        objectstate_bop_capture_condition_sidecar_summary(
            scene_root,
            max_frames=2,
        )["sidecar"],
    )

    before = objectstate_bop_phase1_authoring_progress(batch_spec)
    assert before["readiness"]["all_samples_ready_for_batch_readiness_input"] is False
    assert any(
        "generate-bop-objectstate-baseline-candidate" in command
        for command in before["sample_records"][0]["next_commands"]
    )

    write_objectstate_bop_gaussian_centroid_baseline_candidate(
        scene_root,
        output=candidate_path,
        sample_id="bop-ycbv-test-000001",
        max_frames=2,
    )
    after = objectstate_bop_phase1_authoring_progress(batch_spec)

    assert after["status"] == (
        "objectstate_bop_phase1_authoring_ready_for_batch_readiness"
    )
    assert after["readiness"]["all_samples_ready_for_batch_readiness_input"] is True
    assert after["row_counts"]["target_candidate_artifacts_valid"] == 1


def _write_batch_spec(
    path,
    *,
    scene_root: str,
    candidate_artifact: str,
    condition_sidecar: str,
    max_frames: int,
) -> None:
    payload = {
        "schema": OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
        "kind": "objectstate_bop_local_row_batch_spec",
        "batch": {
            "batch_id": "unit-test-batch",
            "output_root": "handoff",
        },
        "defaults": {
            "dataset_id": "bop-ycbv",
            "object_category": "bop_objects",
            "scenario": "bop_pose_sequence",
            "fps": 30.0,
            "license_text": (
                "BOP dataset terms; verify source dataset license before redistribution"
            ),
            "rgb_dir": "rgb",
            "depth_dir": "depth",
            "gaussian_dir": "gaussians",
            "max_frames": max_frames,
            "frame_step": 1,
        },
        "samples": [
            {
                "sample_id": "bop-ycbv-test-000001",
                "scene_root": scene_root,
                "candidate_artifact": candidate_artifact,
                "condition_sidecar": condition_sidecar,
                "output_root": "samples/bop-ycbv-test-000001",
            }
        ],
        "claim_policy": {
            "local_only": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_claim_world_model": True,
        },
    }
    _write_json(path, payload)


def _write_bop_scene(root, *, frame_count: int) -> None:
    (root / "rgb").mkdir(parents=True)
    for frame_id in range(frame_count):
        (root / "rgb" / f"{frame_id:06d}.png").write_bytes(PNG_BYTES)
    scene_camera = {
        str(frame_id): {
            "cam_K": [572.4, 0.0, 325.2, 0.0, 573.5, 242.0, 0.0, 0.0, 1.0],
            "depth_scale": 1.0,
        }
        for frame_id in range(frame_count)
    }
    identity_rotation = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    scene_gt = {}
    scene_gt_info = {}
    for frame_id in range(frame_count):
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
            {"visib_fract": 1.0 - frame_id * 0.1},
            {"visib_fract": 1.0},
        ]
    _write_json(root / "scene_camera.json", scene_camera)
    _write_json(root / "scene_gt.json", scene_gt)
    _write_json(root / "scene_gt_info.json", scene_gt_info)


def _write_gaussian_frames(root, *, frame_count: int) -> None:
    (root / "gaussians").mkdir(exist_ok=True)
    for frame_id in range(frame_count):
        offset = float(frame_id)
        body = (
            "ply\n"
            "format ascii 1.0\n"
            "element vertex 2\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
            f"{offset} 0 0\n"
            f"{offset + 2.0} 0 0\n"
        )
        (root / "gaussians" / f"{frame_id:06d}.ply").write_text(
            body,
            encoding="ascii",
        )


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
