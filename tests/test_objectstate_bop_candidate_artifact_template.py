from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.pipelines.objectstate_bop_candidate_artifact_template import (
    OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_FINALIZE_SCHEMA,
    OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA,
    OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA,
    finalize_objectstate_bop_candidate_artifact_template,
    validate_objectstate_bop_candidate_artifact_finalize_summary,
    validate_objectstate_bop_candidate_artifact_template,
    validate_objectstate_bop_candidate_artifact_template_summary,
    write_objectstate_bop_candidate_artifact_template,
)
from objgauss.pipelines.objectstate_bop_identity_route_audit import (
    objectstate_bop_identity_route_audit,
)
from objgauss.pipelines.trainable_artifact import (
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
    assert first_placeholder["persistent_id"].startswith("TODO integer candidate")
    assert first_placeholder["slot"].startswith("TODO integer renderer")
    assert first_placeholder["state_id"] is None
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


def test_bop_candidate_artifact_finalizer_outputs_identity_route_artifact(tmp_path):
    scene_root = tmp_path / "bop-scene"
    output_root = tmp_path / "identity-package"
    template_path = output_root / "objectstates.template.json"
    artifact_path = output_root / "objectstates.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    write_objectstate_bop_candidate_artifact_template(
        scene_root,
        output=template_path,
        sample_id="bop-ycbv-scene-000001",
        target_artifact_path=artifact_path,
        candidate_id="bop-filled-candidate",
        candidate_source="unit-test-filled-template",
    )
    _fill_template(template_path)

    summary = finalize_objectstate_bop_candidate_artifact_template(
        template_path,
        reconstruction_noise_robustness=0.97,
        reconstruction_noise_variant_count=2,
    )
    artifact = _read_json(artifact_path)

    assert summary["schema"] == OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_FINALIZE_SCHEMA
    assert summary["status"] == "objectstate_bop_candidate_artifact_finalized"
    assert summary["readiness"]["phase1_gaussian_evidence_ready"] is True
    assert summary["readiness"]["target_artifact_valid"] is True
    assert summary["readiness"]["identity_evidence_present"] is True
    assert summary["row_counts"] == {"frames": 3, "states": 6}
    assert summary["pose_gt_leakage_guard"]["min_distance_to_pose_gt"] > 0.0
    assert validate_objectstate_bop_candidate_artifact_finalize_summary(summary) == summary

    assert artifact["schema"] == TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA
    assert artifact["label"] == "bop-filled-candidate"
    assert artifact["source"]["input"] == str(artifact_path)
    assert artifact["source"]["template"] == str(template_path)
    assert artifact["training"]["frame_count"] == 3
    assert artifact["training"]["optimization_steps"] == 0
    assert artifact["artifact_policy"]["candidate_packaging_only"] is True
    assert artifact["artifact_policy"]["not_a_training_run"] is True
    assert artifact["identity_evidence"] == {
        "reconstruction_noise_robustness": 0.97,
        "reconstruction_noise_variant_count": 2,
    }
    first_state = artifact["object_states"][0]["states"][0]
    assert first_state["id"] == first_state["persistent_id"] == 100
    assert first_state["slot"] == first_state["object_id"] == 0
    assert artifact["object_states"][0]["derived_object_ids"] == [0, 1]
    assert artifact["assignments"][0]["matrix"] == [[1.0, 0.0], [0.0, 1.0]]
    assert validate_trainable_kernel_model_artifact(artifact) is True

    route = objectstate_bop_identity_route_audit(
        scene_root,
        output_root=output_root,
        sample_id="bop-ycbv-scene-000001",
        candidate_artifact=artifact_path,
    )
    assert route["readiness"]["candidate_artifact_valid"] is True
    assert route["readiness"]["candidate_artifact_binding_ready"] is True
    assert route["readiness"]["route_ready_for_identity_handoff"] is False


def test_bop_candidate_artifact_finalizer_cli_writes_outputs(tmp_path, capsys):
    scene_root = tmp_path / "bop-scene"
    template_path = tmp_path / "objectstates.template.json"
    artifact_path = tmp_path / "objectstates.json"
    summary_path = tmp_path / "objectstates-finalize-summary.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    write_objectstate_bop_candidate_artifact_template(
        scene_root,
        output=template_path,
        sample_id="bop-ycbv-scene-000001",
        target_artifact_path=artifact_path,
        candidate_id="bop-cli-finalized",
    )
    _fill_template(template_path)

    assert (
        main(
            [
                "object-state",
                "finalize-bop-objectstate-artifact-template",
                str(template_path),
                "--summary-output",
                str(summary_path),
                "--reconstruction-noise-robustness",
                "0.96",
                "--reconstruction-noise-variant-count",
                "2",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_path)
    artifact = _read_json(artifact_path)

    assert f"schema={OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_FINALIZE_SCHEMA}" in stdout
    assert "bop_objectstate_artifact_finalize_status=objectstate_bop_candidate_artifact_finalized" in stdout
    assert "readiness.target_artifact_valid=true" in stdout
    assert "readiness.identity_evidence_present=true" in stdout
    assert "next_command=uv run objgauss object-state audit-bop-phase1-local-row" in stdout
    assert summary["output"] == str(artifact_path)
    assert artifact["identity_evidence"]["reconstruction_noise_robustness"] == 0.96


def test_bop_candidate_artifact_finalizer_rejects_todo_values(tmp_path):
    scene_root = tmp_path / "bop-scene"
    template_path = tmp_path / "objectstates.template.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    write_objectstate_bop_candidate_artifact_template(
        scene_root,
        output=template_path,
        sample_id="bop-ycbv-scene-000001",
    )

    with pytest.raises(ValueError, match="TODO"):
        finalize_objectstate_bop_candidate_artifact_template(template_path)


def test_bop_candidate_artifact_finalizer_rejects_pose_gt_centroid_leakage(tmp_path):
    scene_root = tmp_path / "bop-scene"
    template_path = tmp_path / "objectstates.template.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    write_objectstate_bop_candidate_artifact_template(
        scene_root,
        output=template_path,
        sample_id="bop-ycbv-scene-000001",
    )
    _fill_template(template_path, exact_gt=True)

    with pytest.raises(ValueError, match="pose GT centroid leakage"):
        finalize_objectstate_bop_candidate_artifact_template(template_path)


def test_bop_candidate_artifact_finalizer_preserves_legacy_state_id_fallback(tmp_path):
    scene_root = tmp_path / "bop-scene"
    template_path = tmp_path / "objectstates.template.json"
    artifact_path = tmp_path / "objectstates.json"
    _write_bop_scene(scene_root)
    _write_gaussian_frames(scene_root)
    write_objectstate_bop_candidate_artifact_template(
        scene_root,
        output=template_path,
        target_artifact_path=artifact_path,
        sample_id="bop-ycbv-scene-000001",
    )
    _fill_template(template_path, legacy_addresses=True)

    finalize_objectstate_bop_candidate_artifact_template(template_path)
    artifact = _read_json(artifact_path)

    state = artifact["object_states"][0]["states"][0]
    assert state["id"] == state["persistent_id"] == 0
    assert state["slot"] == state["object_id"] == 0
    assert "legacy_state_id_used_for_identity_and_renderer_slot" in state["diagnostics"]
    assert artifact["object_states"][0]["derived_object_ids"] == [0, 1]


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


def _fill_template(
    path,
    *,
    exact_gt: bool = False,
    legacy_addresses: bool = False,
) -> None:
    template = _read_json(path)
    for frame in template["object_state_frames"]:
        frame_index = frame["frame_index"]
        for slot_index, placeholder in enumerate(frame["state_placeholders"]):
            if placeholder["object_id"].endswith("000001"):
                gt = [0.01 + 0.001 * frame_index, 0.02, 0.03]
            else:
                gt = [0.04 + 0.001 * frame_index, 0.05, 0.06]
            centroid = gt if exact_gt else [value + 0.0002 for value in gt]
            if legacy_addresses:
                placeholder["state_id"] = slot_index
            else:
                placeholder["persistent_id"] = 100 + slot_index
                placeholder["slot"] = slot_index
            placeholder["centroid"] = centroid
            placeholder["bbox"] = [
                [value - 0.001 for value in centroid],
                [value + 0.001 for value in centroid],
            ]
            placeholder["confidence"] = 0.91
            placeholder["note"] = "Filled from unit-test model output."
    _write_json(path, template)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
