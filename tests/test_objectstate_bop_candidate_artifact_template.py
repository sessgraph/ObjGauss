from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_bop_candidate_artifact_template import (
    OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA,
    OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA,
    validate_objectstate_bop_candidate_artifact_template,
    validate_objectstate_bop_candidate_artifact_template_summary,
    write_objectstate_bop_candidate_artifact_template,
)
from objgauss.core.objectstate_bop_identity_route_audit import (
    objectstate_bop_identity_route_audit,
)
from objgauss.core.trainable_artifact import (
    TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
    validate_trainable_kernel_model_artifact,
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


def test_bop_candidate_artifact_template_is_draft_only(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output = tmp_path / "objectstates.template.json"
    target = tmp_path / "objectstates.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)

    summary = write_objectstate_bop_candidate_artifact_template(
        scene_root,
        output=output,
        sample_id="bop-ycbv-scene-000001",
        target_artifact_path=target,
        candidate_id="candidate-draft",
        candidate_source="unit-test-model-output",
    )
    template = _read_json(output)

    assert summary["schema"] == OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA
    assert summary["status"] == "objectstate_bop_candidate_artifact_template_ready"
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is True
    assert summary["readiness"]["draft_not_valid_for_identity_route"] is True
    assert summary["row_counts"] == {"frames": 3, "state_placeholders": 6}
    assert summary["target_artifact"] == str(target)
    assert validate_objectstate_bop_candidate_artifact_template_summary(summary) == summary

    assert template["schema"] == OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA
    assert template["template_status"] == "draft_not_valid_for_identity_route"
    assert template["target_artifact_schema"] == TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA
    assert template["target_artifact_schema"] != template["schema"]
    assert template["artifact_policy"]["pose_gt_values_omitted"] is True
    assert template["artifact_policy"]["fill_from_model_outputs_not_ground_truth"] is True
    assert template["object_state_frames"][0]["frame_index"] == 0
    assert template["object_state_frames"][0]["gaussian_ref"] == "gaussians/000000.ply"
    assert template["object_state_frames"][0]["object_ids"] == [
        "bop-ycbv-obj-000001",
        "bop-ycbv-obj-000002",
    ]
    first_placeholder = template["object_state_frames"][0]["state_placeholders"][0]
    assert first_placeholder["centroid"].startswith("TODO model-predicted")
    assert first_placeholder["bbox"].startswith("TODO model-predicted")
    serialized = json.dumps(template)
    assert "cam_t_m2c" not in serialized
    assert "target_position" not in serialized
    assert "0.01" not in serialized
    assert validate_objectstate_bop_candidate_artifact_template(template) == template
    with pytest.raises(ValueError, match="unsupported trainable model artifact schema"):
        validate_trainable_kernel_model_artifact(template)


def test_bop_candidate_artifact_template_blocks_identity_route(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "identity-package"
    template_path = output_root / "objectstates.template.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    write_objectstate_bop_candidate_artifact_template(
        scene_root,
        output=template_path,
        sample_id="bop-ycbv-scene-000001",
        target_artifact_path=output_root / "objectstates.json",
    )

    summary = objectstate_bop_identity_route_audit(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_artifact=template_path,
    )

    assert summary["readiness"]["candidate_artifact_present"] is True
    assert summary["readiness"]["candidate_artifact_valid"] is False
    assert summary["readiness"]["candidate_artifact_binding_ready"] is False
    assert summary["readiness"]["route_ready_for_identity_handoff"] is False
    candidate = summary["records"]["candidate_artifact"]
    assert candidate["schema"] == OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA
    assert any("validator failed" in issue for issue in candidate["issues"])


def test_bop_candidate_artifact_template_records_missing_gaussian_readiness(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output = tmp_path / "objectstates.template.json"
    _write_bop_scene(scene_root)

    summary = write_objectstate_bop_candidate_artifact_template(
        scene_root,
        output=output,
        sample_id="bop-ycbv-scene-000001",
    )
    template = _read_json(output)

    assert summary["acceptance"]["status"] == "objectstate_bop_capture_acceptance_fail"
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is False
    assert summary["readiness"]["template_written"] is True
    assert template["object_state_frames"][2]["gaussian_ref"] == "gaussians/000002.ply"


def test_bop_candidate_artifact_template_refuses_overwrite(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output = tmp_path / "objectstates.template.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="BOP candidate artifact template exists"):
        write_objectstate_bop_candidate_artifact_template(
            scene_root,
            output=output,
            sample_id="bop-ycbv-scene-000001",
        )


def test_bop_candidate_artifact_template_cli_writes_outputs(tmp_path, capsys):
    scene_root = tmp_path / "bop-scene"
    output = tmp_path / "objectstates.template.json"
    target = tmp_path / "objectstates.json"
    summary_path = tmp_path / "objectstates-template-summary.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)

    assert (
        main(
            [
                "object-state",
                "init-bop-objectstate-artifact-template",
                str(scene_root),
                "--output",
                str(output),
                "--summary-output",
                str(summary_path),
                "--sample-id",
                "bop-ycbv-scene-000001",
                "--target-artifact-path",
                str(target),
                "--candidate-id",
                "candidate-cli",
                "--candidate-source",
                "cli fixture model",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_path)
    template = _read_json(output)

    assert f"schema={OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA}" in stdout
    assert "bop_objectstate_artifact_template_status=objectstate_bop_candidate_artifact_template_ready" in stdout
    assert "readiness.phase1_gaussian_evidence_ready=true" in stdout
    assert "template_not_valid_for_identity_route=true" in stdout
    assert "next_command=uv run objgauss object-state audit-bop-phase1-local-row" in stdout
    assert summary["output"] == str(output)
    assert summary["target_artifact"] == str(target)
    assert template["candidate"] == {
        "candidate_id": "candidate-cli",
        "source": "cli fixture model",
    }


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
                "px_count_visib": 1000 - frame_id * 200,
                "visib_fract": 1.0 - frame_id * 0.2,
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


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
