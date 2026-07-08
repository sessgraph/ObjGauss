from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.core.objectstate_transition_dataset import (
    OBJECTSTATE_TRANSITION_DATASET_SCHEMA,
    OBJECTSTATE_TRANSITION_ROW_SCHEMA,
    objectstate_transition_dataset_from_capture_manifest,
    validate_objectstate_transition_dataset,
    write_objectstate_transition_dataset,
)


def test_transition_dataset_compiles_object_experience_rows():
    dataset = objectstate_transition_dataset_from_capture_manifest(
        _capture_manifest(include_action=True),
        require_action_transition=True,
    )

    assert dataset["schema"] == OBJECTSTATE_TRANSITION_DATASET_SCHEMA
    assert dataset["row_schema"] == OBJECTSTATE_TRANSITION_ROW_SCHEMA
    assert dataset["sample"]["sample_id"] == "transition-cup-box-001"
    assert dataset["row_counts"] == {
        "objects": 2,
        "object_episodes": 2,
        "frames": 3,
        "source_actions": 1,
        "transitions": 4,
        "action_conditioned_transitions": 2,
        "no_action_transitions": 2,
    }
    assert dataset["readiness"]["object_episode_ready"] is True
    assert dataset["readiness"]["pose_transition_ready"] is True
    assert dataset["readiness"]["action_conditioned_transition_ready"] is True
    assert dataset["readiness"]["real_gaussian_refs_present"] is True
    assert dataset["claim_policy"]["object_level_transition_dataset"] is True
    assert dataset["claim_policy"]["does_not_train_dynamics_model"] is True
    assert all(value is False for value in dataset["non_goals"].values())
    assert validate_objectstate_transition_dataset(dataset) == dataset

    rows = {
        (row["object_id"], row["source_frame_id"], row["target_frame_id"]): row
        for row in dataset["transitions"]
    }
    cup_action = rows[("cup-001", "frame-000000", "frame-000001")]
    assert cup_action["schema"] == OBJECTSTATE_TRANSITION_ROW_SCHEMA
    assert cup_action["has_action"] is True
    assert cup_action["action_ids"] == ["push-cup-001"]
    assert cup_action["state_t"]["pose"]["position"] == [0.1, 0.2, 0.3]
    assert cup_action["state_t1"]["pose"]["position"] == [0.11, 0.2, 0.3]
    assert cup_action["delta_t"] == pytest.approx(0.5)

    box_no_action = rows[("box-001", "frame-000001", "frame-000002")]
    assert box_no_action["has_action"] is False
    assert box_no_action["action_context"] == []


def test_transition_dataset_requires_action_transition_when_requested():
    with pytest.raises(ValueError, match="requires at least one action transition"):
        objectstate_transition_dataset_from_capture_manifest(
            _capture_manifest(include_action=False),
            require_action_transition=True,
        )


def test_transition_dataset_writer_and_cli(tmp_path, capsys):
    capture_path = tmp_path / "capture-manifest.json"
    output = tmp_path / "objectstate-transitions.json"
    summary_output = tmp_path / "objectstate-transitions-summary.json"
    capture_path.write_text(
        json.dumps(_capture_manifest(include_action=True), indent=2) + "\n",
        encoding="utf-8",
    )

    direct = write_objectstate_transition_dataset(
        capture_path,
        output,
        require_action_transition=True,
    )
    assert output.is_file()
    assert direct["output"] == str(output)
    assert direct["row_counts"]["action_conditioned_transitions"] == 2

    cli_output = tmp_path / "objectstate-transitions-cli.json"
    assert (
        main(
            [
                "object-state",
                "compile-objectstate-transitions",
                str(capture_path),
                "--output",
                str(cli_output),
                "--summary-output",
                str(summary_output),
                "--require-action-transition",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_output.read_text(encoding="utf-8"))
    assert f"schema={OBJECTSTATE_TRANSITION_DATASET_SCHEMA}" in stdout
    assert "sample_id=transition-cup-box-001" in stdout
    assert "transitions=4" in stdout
    assert "action_conditioned_transitions=2" in stdout
    assert "action_conditioned_transition_ready=true" in stdout
    assert summary["schema"] == OBJECTSTATE_TRANSITION_DATASET_SCHEMA
    assert cli_output.is_file()


def _capture_manifest(*, include_action: bool) -> dict:
    frames = []
    for frame_index, timestamp in enumerate((0.0, 0.5, 1.0)):
        frame = {
            "frame_id": f"frame-{frame_index:06d}",
            "timestamp": timestamp,
            "observation": {
                "rgb": f"rgb/{frame_index:06d}.png",
                "gaussian": f"gaussians/{frame_index:06d}.ply",
            },
            "condition": {
                "view_id": "front" if frame_index < 2 else "side",
                "lighting_id": "lab",
                "camera_pose": {
                    "position": [0.0, -1.0, 0.8],
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            },
            "objects": [
                _frame_object("cup-001", [0.1 + 0.01 * frame_index, 0.2, 0.3]),
                _frame_object("box-001", [0.4, 0.2 + 0.01 * frame_index, 0.3]),
            ],
        }
        if include_action and frame_index == 0:
            frame["action_id"] = "push-cup-001"
        frames.append(frame)
    actions = []
    if include_action:
        actions.append(
            {
                "action_id": "push-cup-001",
                "action_type": "push_left",
                "object_id": "cup-001",
                "start_timestamp": 0.0,
                "end_timestamp": 0.5,
                "actor": "hand-001",
                "vector": [-0.02, 0.0, 0.0],
            }
        )
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "transition-cup-box-001",
            "source_kind": "controlled_real",
            "object_category": "cup_box",
            "scenario": "object_transition_dataset",
            "fps": 2.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/captures/transition-cup-box-001/capture-manifest.json",
                "outputs/captures/transition-cup-box-001/rgb/",
                "outputs/captures/transition-cup-box-001/gaussians/",
            ],
            "license": "local controlled capture; not public release",
        },
        "objects": [
            {"object_id": "cup-001", "category": "cup", "instance_label": "blue cup"},
            {"object_id": "box-001", "category": "box", "instance_label": "red box"},
        ],
        "actions": actions,
        "frames": frames,
    }


def _frame_object(object_id: str, position: list[float]) -> dict:
    return {
        "object_id": object_id,
        "visible": True,
        "occlusion_fraction": 0.0,
        "pose": {
            "position": position,
            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
        },
    }
