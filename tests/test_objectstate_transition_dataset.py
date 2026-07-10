from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.datasets.objectstate_transition_dataset import (
    OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA,
    OBJECTSTATE_TRANSITION_DATASET_SCHEMA,
    OBJECTSTATE_TRANSITION_ROW_SCHEMA,
    objectstate_transition_dataset_audit,
    objectstate_transition_dataset_audit_from_path,
    objectstate_transition_dataset_from_capture_manifest,
    validate_objectstate_transition_dataset,
    validate_objectstate_transition_dataset_audit,
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


def test_transition_dataset_audit_reports_ready_dataset():
    dataset = objectstate_transition_dataset_from_capture_manifest(
        _capture_manifest(include_action=True),
        require_action_transition=True,
    )

    audit = objectstate_transition_dataset_audit(
        dataset,
        min_object_episodes=2,
        min_transitions=4,
        min_action_conditioned_transitions=2,
        min_horizon_seconds=1.0,
        require_action_transition=True,
        require_gaussian_refs=True,
    )

    assert audit["schema"] == OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA
    assert audit["status"] == "objectstate_transition_dataset_audit_ready"
    assert audit["sample"]["sample_id"] == "transition-cup-box-001"
    assert audit["metrics"]["object_episode_count"] == 2
    assert audit["metrics"]["transition_count"] == 4
    assert audit["metrics"]["action_conditioned_transition_count"] == 2
    assert audit["metrics"]["action_transition_fraction"] == pytest.approx(0.5)
    assert audit["metrics"]["object_horizon_seconds"]["min_seconds"] == pytest.approx(1.0)
    assert audit["readiness"]["transition_dataset_ready"] is True
    assert audit["hard_blockers"] == []
    assert audit["claim_policy"]["does_not_create_replay_buffer"] is True
    assert audit["claim_policy"]["does_not_claim_metric_pass"] is True
    assert all(value is False for value in audit["non_goals"].values())
    assert validate_objectstate_transition_dataset_audit(audit) == audit


def test_transition_dataset_audit_blocks_missing_action_transitions():
    dataset = objectstate_transition_dataset_from_capture_manifest(
        _capture_manifest(include_action=False),
    )

    audit = objectstate_transition_dataset_audit(
        dataset,
        require_action_transition=True,
    )

    assert audit["status"] == "objectstate_transition_dataset_audit_blocked"
    assert audit["readiness"]["action_transition_count_ready"] is False
    assert audit["readiness"]["transition_dataset_ready"] is False
    assert any(
        "action_conditioned_transition_count 0 < required 1" in blocker
        for blocker in audit["hard_blockers"]
    )


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

    audit_output = tmp_path / "objectstate-transitions-audit.json"
    assert (
        main(
            [
                "object-state",
                "audit-objectstate-transition-dataset",
                str(cli_output),
                "--min-object-episodes",
                "2",
                "--min-transitions",
                "4",
                "--min-action-conditioned-transitions",
                "2",
                "--min-horizon-seconds",
                "1.0",
                "--require-action-transition",
                "--require-gaussian-refs",
                "--summary-output",
                str(audit_output),
                "--require-ready",
            ]
        )
        == 0
    )

    audit_summary = json.loads(audit_output.read_text(encoding="utf-8"))
    audit_stdout = capsys.readouterr().out
    direct_audit = objectstate_transition_dataset_audit_from_path(
        cli_output,
        min_object_episodes=2,
        min_transitions=4,
        min_action_conditioned_transitions=2,
        min_horizon_seconds=1.0,
        require_action_transition=True,
        require_gaussian_refs=True,
    )
    assert f"schema={OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA}" in audit_stdout
    assert "status=objectstate_transition_dataset_audit_ready" in audit_stdout
    assert "transition_dataset_ready=true" in audit_stdout
    assert validate_objectstate_transition_dataset_audit(audit_summary) == audit_summary
    assert direct_audit["schema"] == OBJECTSTATE_TRANSITION_DATASET_AUDIT_SCHEMA


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
