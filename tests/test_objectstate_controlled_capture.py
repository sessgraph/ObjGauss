from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA,
    objectstate_controlled_capture_summary,
    objectstate_controlled_real_manifest_from_capture_manifest,
    read_objectstate_controlled_capture_manifest,
    validate_objectstate_controlled_capture_manifest,
    validate_objectstate_controlled_capture_summary,
)
from objgauss.core.objectstate_controlled_real_rows import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    validate_objectstate_controlled_real_manifest,
)


def test_controlled_capture_manifest_summarizes_full_stage_readiness():
    manifest = _capture_manifest(include_gaussian=True, include_pose=True, include_action=True)

    summary = objectstate_controlled_capture_summary(manifest)
    seed = objectstate_controlled_real_manifest_from_capture_manifest(manifest)

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA
    assert summary["frame_count"] == 3
    assert summary["object_count"] == 1
    assert summary["action_count"] == 1
    assert summary["ground_truth"] == {
        "identity": True,
        "pose": True,
        "action": True,
        "timestamp": True,
    }
    assert summary["readiness"] == {
        "identity_stage_ready": True,
        "prediction_stage_ready": True,
        "intervention_stage_ready": True,
        "real_gaussian_reconstruction_present": True,
    }
    assert summary["issues"] == []
    assert summary["controlled_real_manifest_seed"] == seed
    assert seed["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert seed["ground_truth"] == summary["ground_truth"]
    assert [row["status"] for row in seed["evidence_rows"]] == [
        "blocked",
        "blocked",
        "blocked",
    ]
    assert "missing candidate identity metrics" in seed["evidence_rows"][0]["block_reason"]
    assert validate_objectstate_controlled_real_manifest(seed) is not None
    assert validate_objectstate_controlled_capture_summary(summary) is summary


def test_controlled_capture_manifest_keeps_incomplete_capture_blocked():
    summary = objectstate_controlled_capture_summary(
        _capture_manifest(include_gaussian=False, include_pose=False, include_action=False)
    )
    seed = summary["controlled_real_manifest_seed"]

    assert summary["readiness"]["identity_stage_ready"] is True
    assert summary["readiness"]["prediction_stage_ready"] is False
    assert summary["readiness"]["intervention_stage_ready"] is False
    assert summary["readiness"]["real_gaussian_reconstruction_present"] is False
    assert "not all frames reference reconstructed Gaussian evidence" in summary["issues"]
    assert "6DoF pose GT is incomplete" in summary["issues"]
    assert "action GT is missing" in summary["issues"]
    assert "missing candidate identity metrics" in seed["evidence_rows"][0]["block_reason"]
    assert "not prediction-ready" in seed["evidence_rows"][1]["block_reason"]
    assert "not intervention-ready" in seed["evidence_rows"][2]["block_reason"]


def test_controlled_capture_manifest_reads_json_file(tmp_path):
    manifest_path = tmp_path / "capture-manifest.json"
    manifest_path.write_text(json.dumps(_capture_manifest()), encoding="utf-8")

    manifest = read_objectstate_controlled_capture_manifest(manifest_path)

    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA
    assert manifest["sample"]["sample_id"] == "controlled-tabletop-cup-capture-001"


def test_controlled_capture_manifest_preserves_frame_conditions():
    manifest = _capture_manifest(include_conditions=True)

    checked = validate_objectstate_controlled_capture_manifest(manifest)

    assert checked["frames"][0]["condition"] == {
        "view_id": "front",
        "lighting_id": "bright",
        "camera_pose": {
            "position": [0.0, 0.0, 0.0],
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
    assert checked["frames"][2]["condition"]["view_id"] == "right"
    assert checked["frames"][2]["condition"]["lighting_id"] == "dim"


def test_controlled_capture_manifest_rejects_non_monotonic_timestamps():
    manifest = _capture_manifest()
    manifest["frames"][1]["timestamp"] = manifest["frames"][0]["timestamp"]

    with pytest.raises(ValueError, match="timestamps must be strictly increasing"):
        validate_objectstate_controlled_capture_manifest(manifest)


def test_controlled_capture_manifest_rejects_duplicate_object_in_frame():
    manifest = _capture_manifest()
    manifest["frames"][0]["objects"].append(dict(manifest["frames"][0]["objects"][0]))

    with pytest.raises(ValueError, match="duplicate object_id"):
        validate_objectstate_controlled_capture_manifest(manifest)


def test_controlled_capture_manifest_rejects_malformed_pose():
    manifest = _capture_manifest(include_pose=True)
    manifest["frames"][0]["objects"][0]["pose"]["rotation_xyzw"] = [0.0, 0.0, 0.0]

    with pytest.raises(ValueError, match="pose.rotation_xyzw must have length 4"):
        validate_objectstate_controlled_capture_manifest(manifest)


def test_controlled_capture_manifest_rejects_empty_frame_condition():
    manifest = _capture_manifest()
    manifest["frames"][0]["condition"] = {}

    with pytest.raises(ValueError, match="frame.condition must include"):
        validate_objectstate_controlled_capture_manifest(manifest)


def test_object_state_validate_controlled_capture_cli_writes_summary_and_seed(
    tmp_path,
    capsys,
):
    manifest_path = tmp_path / "capture-manifest.json"
    summary_path = tmp_path / "capture-summary.json"
    seed_path = tmp_path / "controlled-real-seed.json"
    manifest_path.write_text(
        json.dumps(_capture_manifest(include_gaussian=True, include_pose=True)),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "object-state",
                "validate-controlled-capture",
                str(manifest_path),
                "--require-identity-ready",
                "--require-prediction-ready",
                "--summary-output",
                str(summary_path),
                "--controlled-real-output",
                str(seed_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    seed = json.loads(seed_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA}" in stdout
    assert "identity_stage_ready=true" in stdout
    assert "prediction_stage_ready=true" in stdout
    assert "intervention_stage_ready=false" in stdout
    assert "real_gaussian_reconstruction_present=true" in stdout
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA
    assert summary["readiness"]["prediction_stage_ready"] is True
    assert seed["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert [row["status"] for row in seed["evidence_rows"]] == [
        "blocked",
        "blocked",
        "blocked",
    ]


def _capture_manifest(
    *,
    include_gaussian: bool = True,
    include_pose: bool = True,
    include_action: bool = False,
    include_conditions: bool = False,
):
    frames = []
    for index, timestamp in enumerate((0.0, 0.033333, 0.066667)):
        observation = {"rgb": f"rgb/{index:06d}.png"}
        if include_gaussian:
            observation["gaussian"] = f"gaussians/{index:06d}.ply"
        frame_object = {
            "object_id": "cup-001",
            "visible": index != 1,
            "occlusion_fraction": 0.75 if index == 1 else 0.0,
        }
        if include_pose:
            frame_object["pose"] = {
                "position": [0.1 + index * 0.01, 0.2, 0.3],
                "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
            }
        frame = {
            "frame_id": f"frame-{index:06d}",
            "timestamp": timestamp,
            "observation": observation,
            "objects": [frame_object],
        }
        if include_conditions:
            frame["condition"] = {
                "view_id": "front" if index < 2 else "right",
                "lighting_id": "bright" if index == 0 else "dim",
                "camera_pose": {
                    "position": [0.02 * index, 0.0, 0.0],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        if include_action and index == 1:
            frame["action_id"] = "push-left-001"
        frames.append(frame)
    actions = []
    if include_action:
        actions.append(
            {
                "action_id": "push-left-001",
                "action_type": "push_left",
                "object_id": "cup-001",
                "start_timestamp": 0.033333,
                "end_timestamp": 0.066667,
                "actor": "scripted-hand",
                "vector": [-0.02, 0.0, 0.0],
            }
        )
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "controlled-tabletop-cup-capture-001",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "cross_view_occlusion_reappearance",
            "fps": 30.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/controlled-real/cup-capture-001/capture-manifest.json",
                "outputs/controlled-real/cup-capture-001/rgb/",
                "outputs/controlled-real/cup-capture-001/gaussians/",
            ],
            "license": "local controlled capture; not public release",
        },
        "objects": [
            {
                "object_id": "cup-001",
                "category": "cup",
                "instance_label": "blue cup",
                "dimensions_m": [0.08, 0.08, 0.1],
            }
        ],
        "actions": actions,
        "frames": frames,
    }
