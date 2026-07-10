from __future__ import annotations

import csv
import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_bop_local_row_batch_spec import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
)
from objgauss.datasets.objectstate_bop_phase1_sample_workspace import (
    OBJECTSTATE_BOP_PHASE1_SAMPLE_WORKSPACES_SCHEMA,
    objectstate_bop_phase1_sample_workspaces,
    validate_objectstate_bop_phase1_sample_workspaces_summary,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n"


def test_bop_phase1_sample_workspaces_write_authoring_helpers(tmp_path):
    scene_root = tmp_path / "bop" / "ycbv" / "test" / "000001"
    batch_spec = tmp_path / "workspace" / "bop-local-row-batch.json"
    artifact_root = tmp_path / "workspace" / "artifacts" / "bop-ycbv-test-000001"
    _write_bop_scene(scene_root, frame_count=3, write_rgb=True)
    _write_batch_spec(
        batch_spec,
        scene_root="../bop/ycbv/test/000001",
        candidate_artifact="artifacts/bop-ycbv-test-000001/objectstates.json",
        condition_sidecar="artifacts/bop-ycbv-test-000001/bop-condition-sidecar.json",
    )

    summary = objectstate_bop_phase1_sample_workspaces(batch_spec)

    assert summary["schema"] == OBJECTSTATE_BOP_PHASE1_SAMPLE_WORKSPACES_SCHEMA
    assert summary["status"] == "objectstate_bop_phase1_sample_workspaces_initialized"
    assert validate_objectstate_bop_phase1_sample_workspaces_summary(summary) == summary
    assert summary["row_counts"] == {
        "samples": 1,
        "scene_roots_present": 1,
        "condition_csv_templates_written": 1,
        "condition_sidecar_drafts_written": 1,
        "target_condition_sidecars_present": 0,
        "target_candidate_artifacts_present": 0,
    }
    assert summary["readiness"] == {
        "sample_count_nonzero": True,
        "all_scene_roots_present": True,
        "all_condition_templates_written": True,
        "authoring_workspace_initialized": True,
        "sample_workspaces_ready_to_author": True,
    }

    rows = _read_csv(artifact_root / "bop-conditions.template.csv")
    assert rows[0]["frame_id"] == "0"
    assert rows[0]["view_id"] == "bop-camera-frame-000000"
    assert rows[0]["lighting_id"] == "bop-default"
    sidecar_draft = _read_json(artifact_root / "bop-condition-sidecar.draft.json")
    assert sidecar_draft["schema"] == "objgauss-objectstate-bop-capture-condition-sidecar-v1"
    readme = (artifact_root / "README.md").read_text(encoding="utf-8")
    assert "init-bop-condition-sidecar" in readme
    assert "init-bop-objectstate-artifact-template" in readme
    assert not (artifact_root / "bop-condition-sidecar.json").exists()
    assert not (artifact_root / "objectstates.json").exists()
    assert any(
        "target condition sidecar is not present yet" in issue
        for issue in summary["issues"]
    )
    assert any(
        "target ObjectState candidate artifact is not present yet" in issue
        for issue in summary["issues"]
    )


def test_bop_phase1_sample_workspaces_report_missing_scene(tmp_path):
    batch_spec = tmp_path / "workspace" / "bop-local-row-batch.json"
    _write_batch_spec(
        batch_spec,
        scene_root="../missing-scene",
        candidate_artifact="artifacts/sample/objectstates.json",
        condition_sidecar="artifacts/sample/bop-condition-sidecar.json",
    )

    summary = objectstate_bop_phase1_sample_workspaces(batch_spec)

    assert summary["status"] == "objectstate_bop_phase1_sample_workspaces_blocked"
    assert summary["readiness"]["all_scene_roots_present"] is False
    assert summary["row_counts"]["condition_csv_templates_written"] == 0
    assert any("scene roots are missing" in blocker for blocker in summary["hard_blockers"])
    assert any("scene root missing" in issue for issue in summary["issues"])


def test_bop_phase1_sample_workspaces_cli(tmp_path, capsys):
    scene_root = tmp_path / "bop" / "ycbv" / "test" / "000001"
    batch_spec = tmp_path / "workspace" / "bop-local-row-batch.json"
    summary_output = tmp_path / "workspace" / "sample-workspaces-summary.json"
    _write_bop_scene(scene_root, frame_count=3, write_rgb=True)
    _write_batch_spec(
        batch_spec,
        scene_root="../bop/ycbv/test/000001",
        candidate_artifact="artifacts/bop-ycbv-test-000001/objectstates.json",
        condition_sidecar="artifacts/bop-ycbv-test-000001/bop-condition-sidecar.json",
    )

    assert (
        main(
            [
                "object-state",
                "init-bop-phase1-sample-workspaces",
                str(batch_spec),
                "--summary-output",
                str(summary_output),
                "--require-ready-to-author",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_output)
    assert f"schema={OBJECTSTATE_BOP_PHASE1_SAMPLE_WORKSPACES_SCHEMA}" in stdout
    assert (
        "bop_phase1_sample_workspaces_status="
        "objectstate_bop_phase1_sample_workspaces_initialized"
    ) in stdout
    assert "sample_workspace=bop-ycbv-test-000001:" in stdout
    assert summary["row_counts"]["samples"] == 1


def test_bop_phase1_sample_workspaces_cli_require_ready_fails(tmp_path):
    batch_spec = tmp_path / "workspace" / "bop-local-row-batch.json"
    _write_batch_spec(
        batch_spec,
        scene_root="../missing-scene",
        candidate_artifact="artifacts/sample/objectstates.json",
        condition_sidecar="artifacts/sample/bop-condition-sidecar.json",
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "object-state",
                "init-bop-phase1-sample-workspaces",
                str(batch_spec),
                "--require-ready-to-author",
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


def _write_bop_scene(root, *, frame_count: int, write_rgb: bool) -> None:
    if write_rgb:
        (root / "rgb").mkdir(parents=True)
        for frame_id in range(frame_count):
            (root / "rgb" / f"{frame_id:06d}.png").write_bytes(PNG_BYTES)
    root.mkdir(parents=True, exist_ok=True)
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


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
