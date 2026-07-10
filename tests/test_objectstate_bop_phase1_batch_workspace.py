from __future__ import annotations

import csv
import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_bop_phase1_batch_workspace import (
    OBJECTSTATE_BOP_PHASE1_BATCH_WORKSPACE_SCHEMA,
    objectstate_bop_phase1_batch_workspace,
    validate_objectstate_bop_phase1_batch_workspace_summary,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n"


def test_bop_phase1_batch_workspace_writes_authoring_files(tmp_path):
    dataset_root = tmp_path / "bop" / "ycbv"
    ready_scene = dataset_root / "test" / "000001"
    blocked_scene = dataset_root / "test" / "000002"
    workspace = tmp_path / "workspace"
    _write_bop_scene(ready_scene, frame_count=3, write_rgb=True)
    _write_bop_scene(blocked_scene, frame_count=1, write_rgb=True)

    summary = objectstate_bop_phase1_batch_workspace(
        dataset_root,
        workspace_root=workspace,
        dataset_id="bop-ycbv",
        max_frames=3,
    )

    assert summary["schema"] == OBJECTSTATE_BOP_PHASE1_BATCH_WORKSPACE_SCHEMA
    assert summary["status"] == "objectstate_bop_phase1_batch_workspace_authored"
    assert validate_objectstate_bop_phase1_batch_workspace_summary(summary) == summary
    assert summary["row_counts"] == {
        "selector_candidates": 2,
        "ready_selector_candidates": 1,
        "samples_csv_rows": 1,
    }
    assert summary["readiness"] == {
        "selector_ready": True,
        "samples_csv_has_rows": True,
        "batch_spec_written": True,
        "batch_spec_inputs_ready": False,
        "workspace_reviewable": True,
    }

    rows = _read_csv(workspace / "samples.csv")
    assert rows == [
        {
            "sample_id": "bop-ycbv-test-000001",
            "scene_root": "../bop/ycbv/test/000001",
            "candidate_artifact": "artifacts/bop-ycbv-test-000001/objectstates.json",
            "condition_sidecar": (
                "artifacts/bop-ycbv-test-000001/bop-condition-sidecar.json"
            ),
            "output_root": "samples/bop-ycbv-test-000001",
            "dataset_id": "bop-ycbv",
            "object_category": "bop_objects",
            "scenario": "bop_pose_sequence",
            "max_frames": "3",
            "frame_step": "1",
        }
    ]
    batch_spec = _read_json(workspace / "bop-local-row-batch.json")
    assert batch_spec["batch"] == {
        "batch_id": "bop-phase1-local-row-batch",
        "output_root": "handoff",
    }
    assert batch_spec["samples"][0]["sample_id"] == "bop-ycbv-test-000001"
    assert batch_spec["samples"][0]["candidate_artifact"] == (
        "artifacts/bop-ycbv-test-000001/objectstates.json"
    )
    assert (workspace / "selector-summary.json").is_file()
    assert (workspace / "batch-spec-authoring-summary.json").is_file()
    assert (workspace / "README.md").read_text(encoding="utf-8").startswith(
        "# ObjGauss BOP Phase 1 Batch Workspace"
    )
    assert any("candidate_artifact" in issue for issue in summary["issues"])
    assert any(
        "audit-bop-local-row-batch-readiness" in command
        for command in summary["next_commands"]
    )


def test_bop_phase1_batch_workspace_reports_no_ready_scene(tmp_path):
    dataset_root = tmp_path / "bop" / "ycbv"
    workspace = tmp_path / "workspace"
    _write_bop_scene(dataset_root / "test" / "000001", frame_count=1, write_rgb=True)

    summary = objectstate_bop_phase1_batch_workspace(
        dataset_root,
        workspace_root=workspace,
        min_frames=3,
    )

    assert summary["status"] == "objectstate_bop_phase1_batch_workspace_blocked"
    assert summary["readiness"]["workspace_reviewable"] is False
    assert summary["row_counts"]["samples_csv_rows"] == 0
    assert (workspace / "samples.csv").is_file()
    assert not (workspace / "bop-local-row-batch.json").exists()
    assert any("no selector-ready" in blocker for blocker in summary["hard_blockers"])


def test_bop_phase1_batch_workspace_cli(tmp_path, capsys):
    dataset_root = tmp_path / "bop" / "ycbv"
    workspace = tmp_path / "workspace"
    summary_output = workspace / "workspace-summary.json"
    _write_bop_scene(dataset_root / "test" / "000001", frame_count=3, write_rgb=True)

    assert (
        main(
            [
                "object-state",
                "init-bop-phase1-batch-workspace",
                str(dataset_root),
                "--workspace-root",
                str(workspace),
                "--summary-output",
                str(summary_output),
                "--max-frames",
                "3",
                "--require-authored",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_output)
    assert f"schema={OBJECTSTATE_BOP_PHASE1_BATCH_WORKSPACE_SCHEMA}" in stdout
    assert (
        "bop_phase1_batch_workspace_status="
        "objectstate_bop_phase1_batch_workspace_authored"
    ) in stdout
    assert f"workspace_root={workspace}" in stdout
    assert "next_command=" in stdout
    assert summary["row_counts"]["samples_csv_rows"] == 1


def test_bop_phase1_batch_workspace_cli_require_authored_fails(tmp_path):
    dataset_root = tmp_path / "missing-bop"
    workspace = tmp_path / "workspace"

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "object-state",
                "init-bop-phase1-batch-workspace",
                str(dataset_root),
                "--workspace-root",
                str(workspace),
                "--require-authored",
            ]
        )
    assert exc.value.code == 2


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
