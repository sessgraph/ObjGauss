from __future__ import annotations

import csv
import json

from objgauss.cli import main
from objgauss.core.objectstate_bop_phase1_subset_selector import (
    OBJECTSTATE_BOP_PHASE1_SUBSET_SELECTOR_SCHEMA,
    objectstate_bop_phase1_subset_selector,
    validate_objectstate_bop_phase1_subset_selector_summary,
)

PNG_BYTES = b"\x89PNG\r\n\x1a\n"


def test_bop_phase1_subset_selector_reports_missing_root(tmp_path):
    summary = objectstate_bop_phase1_subset_selector(tmp_path / "missing-bop")

    assert summary["schema"] == OBJECTSTATE_BOP_PHASE1_SUBSET_SELECTOR_SCHEMA
    assert summary["status"] == "objectstate_bop_phase1_subset_selector_blocked"
    assert summary["readiness"] == {
        "dataset_root_exists": False,
        "scene_candidates_found": False,
        "recommended_scene_ready": False,
    }
    assert summary["row_counts"] == {
        "scene_candidates": 0,
        "ready_candidates": 0,
    }
    assert summary["recommended"] is None
    assert "BOP dataset root does not exist" in " ".join(summary["hard_blockers"])
    assert (
        validate_objectstate_bop_phase1_subset_selector_summary(summary)
        == summary
    )


def test_bop_phase1_subset_selector_recommends_nested_scene(tmp_path):
    dataset_root = tmp_path / "bop" / "ycbv"
    ready_scene = dataset_root / "test" / "000001"
    blocked_scene = dataset_root / "test" / "000002"
    _write_bop_scene(ready_scene, frame_count=3, write_rgb=True)
    _write_bop_scene(blocked_scene, frame_count=1, write_rgb=True)

    summary = objectstate_bop_phase1_subset_selector(
        dataset_root,
        output_root=tmp_path / "captures" / "bop-ycbv-test-000001",
        dataset_id="bop-ycbv",
        max_frames=3,
    )

    assert summary["status"] == "objectstate_bop_phase1_subset_selector_ready"
    assert summary["row_counts"] == {
        "scene_candidates": 2,
        "ready_candidates": 1,
    }
    assert summary["recommended"]["scene_root"] == str(ready_scene)
    assert summary["recommended"]["sample_id"] == "bop-ycbv-test-000001"
    assert summary["recommended"]["metrics"] == {
        "frames": 3,
        "objects": 2,
        "annotations": 6,
        "persistent_objects": 2,
    }
    assert summary["candidates"][0]["status"] == "bop_phase1_subset_candidate_ready"
    assert summary["candidates"][1]["status"] == "bop_phase1_subset_candidate_blocked"
    assert "init-bop-condition-sidecar" in summary["next_commands"][0]
    assert "accept-bop-capture-scene" in summary["next_commands"][1]
    assert "audit-bop-phase1-local-row" in summary["next_commands"][2]
    assert validate_objectstate_bop_phase1_subset_selector_summary(summary) == summary


def test_bop_phase1_subset_selector_cli_writes_summary(tmp_path, capsys):
    dataset_root = tmp_path / "bop" / "ycbv"
    scene_root = dataset_root / "test" / "000001"
    summary_path = tmp_path / "bop-subset-selector-summary.json"
    _write_bop_scene(scene_root, frame_count=3, write_rgb=True)

    assert (
        main(
            [
                "object-state",
                "select-bop-phase1-subset",
                str(dataset_root),
                "--summary-output",
                str(summary_path),
                "--dataset-id",
                "bop-ycbv",
                "--output-root",
                str(tmp_path / "captures" / "bop-ycbv-test-000001"),
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_BOP_PHASE1_SUBSET_SELECTOR_SCHEMA}" in stdout
    assert (
        "bop_phase1_subset_selector_status="
        "objectstate_bop_phase1_subset_selector_ready"
    ) in stdout
    assert f"recommended_scene_root={scene_root}" in stdout
    assert "next_command=" in stdout
    assert summary["recommended"]["sample_id"] == "bop-ycbv-test-000001"


def test_bop_phase1_subset_selector_cli_writes_batch_samples_csv(tmp_path, capsys):
    dataset_root = tmp_path / "bop" / "ycbv"
    ready_scene = dataset_root / "test" / "000001"
    blocked_scene = dataset_root / "test" / "000002"
    template_csv = tmp_path / "batch" / "samples.csv"
    artifact_root = tmp_path / "captures"
    batch_spec = tmp_path / "batch" / "batch.json"
    _write_bop_scene(ready_scene, frame_count=3, write_rgb=True)
    _write_bop_scene(blocked_scene, frame_count=1, write_rgb=True)
    _write_json(
        artifact_root / "bop-ycbv-test-000001" / "objectstates.json",
        {"schema": "unit-test-candidate"},
    )
    _write_json(
        artifact_root / "bop-ycbv-test-000001" / "bop-condition-sidecar.json",
        {"schema": "unit-test-sidecar"},
    )

    assert (
        main(
            [
                "object-state",
                "select-bop-phase1-subset",
                str(dataset_root),
                "--dataset-id",
                "bop-ycbv",
                "--batch-samples-csv-template-output",
                str(template_csv),
                "--batch-sample-artifact-root",
                str(artifact_root),
                "--max-frames",
                "3",
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    rows = _read_csv(template_csv)

    assert f"batch_samples_csv_template={template_csv}" in stdout
    assert "batch_samples_csv_template_rows=1" in stdout
    assert rows == [
        {
            "sample_id": "bop-ycbv-test-000001",
            "scene_root": "../bop/ycbv/test/000001",
            "candidate_artifact": "../captures/bop-ycbv-test-000001/objectstates.json",
            "condition_sidecar": "../captures/bop-ycbv-test-000001/bop-condition-sidecar.json",
            "output_root": "samples/bop-ycbv-test-000001",
            "dataset_id": "bop-ycbv",
            "object_category": "bop_objects",
            "scenario": "bop_pose_sequence",
            "max_frames": "3",
            "frame_step": "1",
        }
    ]
    assert (
        main(
            [
                "object-state",
                "init-bop-local-row-batch-spec",
                "--samples-csv",
                str(template_csv),
                "--output",
                str(batch_spec),
                "--require-inputs",
            ]
        )
        == 0
    )
    spec = json.loads(batch_spec.read_text(encoding="utf-8"))
    assert spec["samples"][0]["sample_id"] == "bop-ycbv-test-000001"


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


def _read_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
