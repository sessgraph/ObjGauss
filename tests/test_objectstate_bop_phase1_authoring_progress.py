from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_bop_capture_adapter import (
    objectstate_bop_capture_condition_sidecar_summary,
)
from objgauss.core.objectstate_bop_local_row_batch_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
)
from objgauss.core.objectstate_bop_phase1_authoring_progress import (
    OBJECTSTATE_BOP_PHASE1_AUTHORING_PROGRESS_SCHEMA,
    objectstate_bop_phase1_authoring_progress,
    validate_objectstate_bop_phase1_authoring_progress_summary,
)
from objgauss.core.objectstate_bop_phase1_sample_workspace import (
    objectstate_bop_phase1_sample_workspaces,
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


def test_bop_phase1_authoring_progress_reports_missing_targets(tmp_path):
    scene_root = tmp_path / "bop" / "ycbv" / "test" / "000001"
    batch_spec = tmp_path / "workspace" / "bop-local-row-batch.json"
    _write_bop_scene(scene_root, frame_count=3, write_depth=True)
    _write_batch_spec(
        batch_spec,
        scene_root="../bop/ycbv/test/000001",
        candidate_artifact="artifacts/bop-ycbv-test-000001/objectstates.json",
        condition_sidecar="artifacts/bop-ycbv-test-000001/bop-condition-sidecar.json",
    )
    objectstate_bop_phase1_sample_workspaces(batch_spec)

    summary = objectstate_bop_phase1_authoring_progress(batch_spec)

    assert summary["schema"] == OBJECTSTATE_BOP_PHASE1_AUTHORING_PROGRESS_SCHEMA
    assert summary["status"] == "objectstate_bop_phase1_authoring_in_progress"
    assert validate_objectstate_bop_phase1_authoring_progress_summary(summary) == summary
    assert summary["row_counts"]["samples"] == 1
    assert summary["row_counts"]["sample_workspace_helpers_present"] == 1
    assert summary["row_counts"]["target_condition_sidecars_present"] == 0
    assert summary["row_counts"]["target_candidate_artifacts_present"] == 0
    assert summary["row_counts"]["expected_gaussian_files"] == 3
    assert summary["row_counts"]["missing_gaussian_files"] == 3
    record = summary["sample_records"][0]
    assert record["readiness"]["depth_export_candidate"] is True
    assert record["readiness"]["ready_for_batch_readiness_input"] is False
    assert any(
        "target condition sidecar is not present yet" in issue
        for issue in record["issues"]
    )
    assert any(
        "export-bop-rgbd-gaussian-evidence" in command
        for command in record["next_commands"]
    )


def test_bop_phase1_authoring_progress_ready_for_batch_readiness(tmp_path):
    scene_root = tmp_path / "bop" / "ycbv" / "test" / "000001"
    batch_spec = tmp_path / "workspace" / "bop-local-row-batch.json"
    artifact_root = tmp_path / "workspace" / "artifacts" / "bop-ycbv-test-000001"
    sidecar_path = artifact_root / "bop-condition-sidecar.json"
    artifact_path = artifact_root / "objectstates.json"
    _write_bop_scene(scene_root, frame_count=3, write_depth=True)
    _write_batch_spec(
        batch_spec,
        scene_root="../bop/ycbv/test/000001",
        candidate_artifact="artifacts/bop-ycbv-test-000001/objectstates.json",
        condition_sidecar="artifacts/bop-ycbv-test-000001/bop-condition-sidecar.json",
    )
    objectstate_bop_phase1_sample_workspaces(batch_spec)
    _write_json(
        sidecar_path,
        objectstate_bop_capture_condition_sidecar_summary(scene_root)["sidecar"],
    )
    _write_gaussian_frames(scene_root, frame_count=3)
    _write_json(artifact_path, _trainable_artifact(frame_count=3))

    summary = objectstate_bop_phase1_authoring_progress(batch_spec)

    assert summary["status"] == (
        "objectstate_bop_phase1_authoring_ready_for_batch_readiness"
    )
    assert summary["readiness"]["all_samples_ready_for_batch_readiness_input"] is True
    assert summary["row_counts"]["target_condition_sidecars_valid"] == 1
    assert summary["row_counts"]["present_gaussian_files"] == 3
    assert summary["row_counts"]["missing_gaussian_files"] == 0
    assert summary["row_counts"]["target_candidate_artifacts_valid"] == 1
    assert "yes" in summary["sample_table_markdown"]
    assert any(
        "audit-bop-local-row-batch-readiness" in command
        for command in summary["next_commands"]
    )


def test_bop_phase1_authoring_progress_cli_outputs_summary_and_table(
    tmp_path,
    capsys,
):
    scene_root = tmp_path / "bop" / "ycbv" / "test" / "000001"
    batch_spec = tmp_path / "workspace" / "bop-local-row-batch.json"
    summary_output = tmp_path / "workspace" / "authoring-progress.json"
    table_output = tmp_path / "workspace" / "authoring-progress.md"
    _write_bop_scene(scene_root, frame_count=3, write_depth=True)
    _write_batch_spec(
        batch_spec,
        scene_root="../bop/ycbv/test/000001",
        candidate_artifact="artifacts/bop-ycbv-test-000001/objectstates.json",
        condition_sidecar="artifacts/bop-ycbv-test-000001/bop-condition-sidecar.json",
    )
    objectstate_bop_phase1_sample_workspaces(batch_spec)

    assert (
        main(
            [
                "object-state",
                "audit-bop-phase1-authoring-progress",
                str(batch_spec),
                "--summary-output",
                str(summary_output),
                "--sample-table-output",
                str(table_output),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_output)
    table = table_output.read_text(encoding="utf-8")
    assert f"schema={OBJECTSTATE_BOP_PHASE1_AUTHORING_PROGRESS_SCHEMA}" in stdout
    assert "bop_phase1_authoring_progress_status=" in stdout
    assert "missing_gaussian_files=3" in stdout
    assert "sample_table=" in stdout
    assert summary["row_counts"]["samples"] == 1
    assert "| sample_id |" in table


def test_bop_phase1_authoring_progress_cli_require_ready_fails(tmp_path):
    scene_root = tmp_path / "bop" / "ycbv" / "test" / "000001"
    batch_spec = tmp_path / "workspace" / "bop-local-row-batch.json"
    _write_bop_scene(scene_root, frame_count=3, write_depth=True)
    _write_batch_spec(
        batch_spec,
        scene_root="../bop/ycbv/test/000001",
        candidate_artifact="artifacts/bop-ycbv-test-000001/objectstates.json",
        condition_sidecar="artifacts/bop-ycbv-test-000001/bop-condition-sidecar.json",
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "object-state",
                "audit-bop-phase1-authoring-progress",
                str(batch_spec),
                "--require-ready-for-batch-readiness",
            ]
        )

    assert exc.value.code == 2


def _write_batch_spec(
    path,
    *,
    scene_root: str,
    candidate_artifact: str,
    condition_sidecar: str,
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
            "max_frames": 3,
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


def _write_bop_scene(root, *, frame_count: int, write_depth: bool) -> None:
    (root / "rgb").mkdir(parents=True)
    for frame_id in range(frame_count):
        (root / "rgb" / f"{frame_id:06d}.png").write_bytes(PNG_BYTES)
    if write_depth:
        (root / "depth").mkdir()
        for frame_id in range(frame_count):
            (root / "depth" / f"{frame_id:06d}.png").write_bytes(PNG_BYTES)
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
    (root / "gaussians").mkdir()
    for frame_id in range(frame_count):
        (root / "gaussians" / f"{frame_id:06d}.ply").write_bytes(PLY_BYTES)


def _trainable_artifact(*, frame_count: int) -> dict:
    return {
        "schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "kind": "trainable_kernel_mvp_model",
        "label": "unit-test-bop-candidate",
        "source": {"sample_id": "bop-ycbv-test-000001"},
        "training": {
            "schema": "objgauss-v1-trainable-kernel-mvp-v1",
            "frame_count": frame_count,
            "source": "unit-test",
        },
        "renderer_api": {},
        "learned_parameters": {"decoder_colors": []},
        "assignments": [
            {
                "frame_index": frame_id,
                "shape": [2, 2],
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
            }
            for frame_id in range(frame_count)
        ],
        "object_states": [
            {
                "frame_index": frame_id,
                "states": [],
                "derived_object_ids": [],
            }
            for frame_id in range(frame_count)
        ],
        "artifact_policy": {"candidate_packaging_only": True},
    }


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))
