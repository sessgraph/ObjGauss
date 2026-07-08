from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_bop_candidate_artifact_template import (
    finalize_objectstate_bop_candidate_artifact_template,
    write_objectstate_bop_candidate_artifact_template,
)
from objgauss.core.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
)
from objgauss.core.objectstate_bop_local_row_batch_handoff import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
)
from objgauss.core.objectstate_bop_local_row_batch_readiness import (
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_READINESS_SCHEMA,
    objectstate_bop_local_row_batch_readiness,
    validate_objectstate_bop_local_row_batch_readiness_summary,
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


def test_bop_local_row_batch_readiness_reports_ready_batch(tmp_path):
    spec_path = _write_batch_spec(tmp_path, sample_count=3)

    summary = objectstate_bop_local_row_batch_readiness(
        spec_path,
        min_reviewable_samples=3,
        min_scene_or_category_coverage=3,
    )

    assert summary["schema"] == OBJECTSTATE_BOP_LOCAL_ROW_BATCH_READINESS_SCHEMA
    assert summary["status"] == "objectstate_bop_local_row_batch_readiness_ready"
    assert validate_objectstate_bop_local_row_batch_readiness_summary(summary) == summary
    assert summary["sample_summary"]["samples"] == 3
    assert summary["sample_summary"]["ready_for_local_row_handoff_samples"] == 3
    assert summary["sample_summary"]["ready_or_reviewable_samples"] == 3
    assert summary["coverage"]["object_category_count"] == 3
    assert summary["readiness_gates"]["all_samples_ready_or_reviewable"] is True
    assert summary["readiness_gates"]["min_reviewable_samples_met"] is True
    assert summary["readiness_gates"]["scene_or_category_coverage_met"] is True
    assert any(
        "bop-local-row-batch-handoff" in action
        for action in summary["next_actions"]
    )
    assert "bop-ycbv-scene-000001" in summary["sample_table_markdown"]


def test_bop_local_row_batch_readiness_blocks_missing_candidate(tmp_path):
    spec_path = _write_batch_spec(tmp_path, sample_count=1, missing_candidate=True)

    summary = objectstate_bop_local_row_batch_readiness(
        spec_path,
        min_reviewable_samples=1,
        min_scene_or_category_coverage=1,
    )

    assert summary["status"] == "objectstate_bop_local_row_batch_readiness_blocked"
    assert summary["sample_summary"]["ready_or_reviewable_samples"] == 0
    assert summary["readiness_gates"]["all_samples_candidate_artifact_binding_ready"] is False
    assert summary["readiness_gates"]["all_samples_ready_or_reviewable"] is False
    assert any("candidate artifact" in blocker for blocker in summary["hard_blockers"])
    assert any(
        "candidate_artifact_binding_ready" in issue for issue in summary["issues"]
    )


def test_bop_local_row_batch_readiness_cli(tmp_path, capsys):
    spec_path = _write_batch_spec(tmp_path, sample_count=1)
    summary_output = tmp_path / "batch-readiness.json"
    table_output = tmp_path / "batch-readiness.md"

    assert (
        main(
            [
                "object-state",
                "audit-bop-local-row-batch-readiness",
                str(spec_path),
                "--min-reviewable-samples",
                "1",
                "--min-scene-or-category-coverage",
                "1",
                "--summary-output",
                str(summary_output),
                "--table-output",
                str(table_output),
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_output)

    assert f"schema={OBJECTSTATE_BOP_LOCAL_ROW_BATCH_READINESS_SCHEMA}" in stdout
    assert "bop_local_row_batch_readiness_status=objectstate_bop_local_row_batch_readiness_ready" in stdout
    assert "readiness.all_samples_ready_or_reviewable=true" in stdout
    assert "sample=bop-ycbv-scene-000001" in stdout
    assert summary["status"] == "objectstate_bop_local_row_batch_readiness_ready"
    assert "bop-ycbv-scene-000001" in table_output.read_text(encoding="utf-8")


def _write_batch_spec(
    tmp_path,
    *,
    sample_count: int,
    missing_candidate: bool = False,
):
    samples = []
    for index in range(1, sample_count + 1):
        sample_id = f"bop-ycbv-scene-{index:06d}"
        scene_root = tmp_path / f"bop-scene-{index}"
        if missing_candidate:
            _write_bop_scene(scene_root)
            _write_gaussian_frames(scene_root)
            artifact_path = tmp_path / "artifacts" / sample_id / "missing.json"
        else:
            artifact_dir = tmp_path / "artifacts" / sample_id
            artifact_path = _write_finalized_artifact(
                scene_root,
                artifact_dir,
                sample_id=sample_id,
            )
        sidecar_path = scene_root / "bop-condition-sidecar.json"
        _write_json(sidecar_path, _condition_sidecar_payload())
        samples.append(
            {
                "sample_id": sample_id,
                "scene_root": scene_root.name,
                "candidate_artifact": str(artifact_path.relative_to(tmp_path)),
                "condition_sidecar": str(sidecar_path.relative_to(tmp_path)),
                "object_category": f"bop-category-{index}",
            }
        )
    spec = _batch_spec_payload(sample_count=0)
    spec["samples"] = samples
    spec_path = tmp_path / "bop-local-row-batch.json"
    _write_json(spec_path, spec)
    return spec_path


def _batch_spec_payload(*, sample_count: int):
    return {
        "schema": OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
        "kind": "objectstate_bop_local_row_batch_spec",
        "batch": {
            "batch_id": "fixture-bop-local-row-batch",
            "output_root": "batch-output",
        },
        "defaults": {
            "dataset_id": "bop-ycbv",
            "scenario": "bop_pose_sequence",
        },
        "samples": [
            {
                "sample_id": f"bop-ycbv-scene-{index:06d}",
                "scene_root": f"bop-scene-{index}",
                "candidate_artifact": f"artifacts/{index}/objectstates.json",
            }
            for index in range(1, sample_count + 1)
        ],
        "claim_policy": {
            "local_only": True,
            "does_not_create_ground_truth": True,
            "does_not_reconstruct_gaussians": True,
            "does_not_train_model": True,
            "does_not_claim_world_model": True,
        },
    }


def _write_finalized_artifact(scene_root, artifact_dir, *, sample_id: str):
    template_path = artifact_dir / "objectstates.template.json"
    artifact_path = artifact_dir / "objectstates.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    write_objectstate_bop_candidate_artifact_template(
        scene_root,
        output=template_path,
        sample_id=sample_id,
        target_artifact_path=artifact_path,
        candidate_id=f"{sample_id}-candidate",
        candidate_source="unit-test-filled-template",
    )
    _fill_template(template_path)
    finalize_objectstate_bop_candidate_artifact_template(
        template_path,
        reconstruction_noise_robustness=0.97,
        reconstruction_noise_variant_count=2,
    )
    return artifact_path


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
            {
                "bbox_obj": [10, 20, 30, 40],
                "bbox_visib": [10, 20, 30, 40],
                "px_count_all": 1000,
                "px_count_valid": 1000,
                "px_count_visib": int(1000 * visibility_by_frame[frame_id]),
                "visib_fract": visibility_by_frame[frame_id],
            },
            {
                "bbox_obj": [50, 60, 30, 40],
                "bbox_visib": [50, 60, 30, 40],
                "px_count_all": 900,
                "px_count_valid": 900,
                "px_count_visib": 900,
                "visib_fract": 1.0,
            },
        ]
    _write_json(root / "scene_camera.json", scene_camera)
    _write_json(root / "scene_gt.json", scene_gt)
    _write_json(root / "scene_gt_info.json", scene_gt_info)


def _write_gaussian_frames(root) -> None:
    (root / "gaussians").mkdir()
    for frame_id in range(3):
        (root / "gaussians" / f"{frame_id:06d}.ply").write_bytes(PLY_BYTES)


def _fill_template(path) -> None:
    template = _read_json(path)
    for frame in template["object_state_frames"]:
        frame_index = frame["frame_index"]
        for slot_index, placeholder in enumerate(frame["state_placeholders"]):
            if placeholder["object_id"].endswith("000001"):
                gt = [0.01 + 0.001 * frame_index, 0.02, 0.03]
            else:
                gt = [0.04 + 0.001 * frame_index, 0.05, 0.06]
            centroid = [value + 0.0002 for value in gt]
            placeholder["state_id"] = slot_index
            placeholder["centroid"] = centroid
            placeholder["bbox"] = [
                [value - 0.001 for value in centroid],
                [value + 0.001 for value in centroid],
            ]
            placeholder["confidence"] = 0.92
            placeholder["note"] = "Filled from unit-test model output."
    _write_json(path, template)


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


def _read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)
