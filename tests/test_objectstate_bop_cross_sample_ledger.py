from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.pipelines.objectstate_bop_candidate_artifact_template import (
    finalize_objectstate_bop_candidate_artifact_template,
    write_objectstate_bop_candidate_artifact_template,
)
from objgauss.datasets.objectstate_bop_capture_adapter import (
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
)
from objgauss.pipelines.objectstate_bop_cross_sample_ledger import (
    OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA,
    objectstate_bop_cross_sample_ledger,
    validate_objectstate_bop_cross_sample_ledger_summary,
)
from objgauss.pipelines.objectstate_bop_local_row_handoff import (
    objectstate_bop_local_row_handoff,
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


def test_bop_cross_sample_ledger_reaches_candidate_ready(tmp_path):
    summary_paths = [
        _write_local_row_summary(
            tmp_path,
            sample_id=f"bop-ycbv-scene-{index:06d}",
            object_category=f"bop-category-{index}",
            scene_index=index,
        )
        for index in range(1, 4)
    ]

    summary = objectstate_bop_cross_sample_ledger(
        local_row_summaries=summary_paths,
        min_reviewable_samples=3,
        min_scene_or_category_coverage=3,
    )

    assert summary["schema"] == OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA
    assert summary["status"] == "objectstate_bop_cross_sample_ledger_reviewable"
    assert summary["maturity"] == "candidate_cross_sample_reviewable"
    assert validate_objectstate_bop_cross_sample_ledger_summary(summary) == summary
    assert summary["candidate_gate"]["candidate_cross_sample_ready"] is True
    assert summary["sample_summary"]["summary_count"] == 3
    assert summary["sample_summary"]["reviewable_sample_count"] == 3
    assert (
        summary["sample_summary"]["identity_prediction_reviewable_sample_count"] == 3
    )
    assert summary["sample_summary"]["identity_pass_sample_count"] == 3
    assert summary["sample_summary"]["prediction_pass_sample_count"] == 3
    assert summary["sample_summary"]["object_category_count"] == 3
    assert summary["candidate_gate"]["gates"] == {
        "min_reviewable_samples_met": True,
        "min_identity_prediction_reviewable_samples_met": True,
        "scene_or_category_coverage_met": True,
        "blocked_rows_separated_from_pass_rows": True,
        "intervention_not_claimed": True,
        "does_not_claim_world_model": True,
    }
    assert summary["issues"] == []
    assert "| bop-ycbv-scene-000001 | true | true | pass | pass | true | true |" in (
        summary["sample_table_markdown"]
    )


def test_bop_cross_sample_ledger_keeps_threshold_gap_visible(tmp_path):
    summary_path = _write_local_row_summary(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        object_category="bop-category-1",
        scene_index=1,
    )

    summary = objectstate_bop_cross_sample_ledger(
        local_row_summaries=(summary_path,),
        min_reviewable_samples=3,
        min_scene_or_category_coverage=3,
    )

    assert summary["status"] == "objectstate_bop_cross_sample_ledger_reviewable"
    assert summary["maturity"] == "identity_prediction_reviewable"
    assert summary["candidate_gate"]["candidate_cross_sample_ready"] is False
    assert (
        summary["candidate_gate"]["gates"]["min_reviewable_samples_met"]
        is False
    )
    assert (
        summary["candidate_gate"]["gates"][
            "min_identity_prediction_reviewable_samples_met"
        ]
        is False
    )
    assert any(
        "candidate gate incomplete: min_reviewable_samples_met" in issue
        for issue in summary["issues"]
    )


def test_bop_cross_sample_ledger_discovers_cli_outputs(tmp_path, capsys):
    summary_path = _write_local_row_summary(
        tmp_path,
        sample_id="bop-ycbv-scene-000001",
        object_category="bop-category-1",
        scene_index=1,
    )
    summary_output = tmp_path / "cross-sample-ledger.json"
    table_output = tmp_path / "cross-sample-table.md"

    assert (
        main(
            [
                "object-state",
                "audit-bop-cross-sample-ledger",
                "--discover-root",
                str(summary_path.parent.parent),
                "--max-depth",
                "3",
                "--min-reviewable-samples",
                "1",
                "--min-scene-or-category-coverage",
                "1",
                "--summary-output",
                str(summary_output),
                "--table-output",
                str(table_output),
                "--require-reviewable",
                "--require-candidate-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_output)
    table = table_output.read_text(encoding="utf-8")

    assert f"schema={OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA}" in stdout
    assert "bop_cross_sample_ledger_status=objectstate_bop_cross_sample_ledger_reviewable" in stdout
    assert "candidate_cross_sample_ready=true" in stdout
    assert "discovered_local_row_summaries=1" in stdout
    assert "candidate_gate.does_not_claim_world_model=true" in stdout
    assert summary["maturity"] == "candidate_cross_sample_reviewable"
    assert "bop-ycbv-scene-000001" in table


def test_bop_cross_sample_ledger_reports_missing_summary(tmp_path):
    missing_path = tmp_path / "missing" / "bop-local-row-handoff-summary.json"

    summary = objectstate_bop_cross_sample_ledger(
        local_row_summaries=(missing_path,),
        min_reviewable_samples=1,
        min_scene_or_category_coverage=1,
    )

    assert summary["status"] == "objectstate_bop_cross_sample_ledger_incomplete"
    assert summary["audit_gates"]["all_files_present"] is False
    assert any("summary file is missing" in issue for issue in summary["issues"])


def _write_local_row_summary(
    tmp_path,
    *,
    sample_id: str,
    object_category: str,
    scene_index: int,
):
    scene_root = tmp_path / f"bop-scene-{scene_index}"
    output_root = tmp_path / f"local-row-package-{scene_index}"
    sidecar_path = scene_root / "bop-condition-sidecar.json"
    artifact_path = _write_finalized_artifact(
        scene_root,
        output_root,
        sample_id=sample_id,
    )
    _write_json(sidecar_path, _condition_sidecar_payload())
    summary = objectstate_bop_local_row_handoff(
        scene_root,
        output_root=output_root,
        sample_id=sample_id,
        candidate_artifact=artifact_path,
        object_category=object_category,
        condition_sidecar=sidecar_path,
        identity_candidate_id=f"{sample_id}-identity",
        prediction_candidate_id=f"{sample_id}-prediction",
        max_centroid_distance=0.01,
    )
    summary_path = output_root / "bop-local-row-handoff-summary.json"
    _write_json(summary_path, summary)
    return summary_path


def _write_finalized_artifact(scene_root, output_root, *, sample_id: str):
    template_path = output_root / "objectstates.template.json"
    artifact_path = output_root / "objectstates.json"
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
    return json.loads(path.read_text(encoding="utf-8"))
