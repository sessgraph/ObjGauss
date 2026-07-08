from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    evaluate_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_candidates,
)
from objgauss.core.objectstate_transition_dataset import (
    objectstate_transition_dataset_from_capture_manifest,
    write_objectstate_transition_dataset,
)
from objgauss.core.objectstate_transition_prediction_candidates import (
    OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA,
    objectstate_transition_prediction_candidates,
    validate_objectstate_transition_prediction_candidates_summary,
)


def test_transition_prediction_candidates_export_constant_velocity():
    capture = _capture_manifest(include_action=False)
    dataset = objectstate_transition_dataset_from_capture_manifest(capture)

    summary = objectstate_transition_prediction_candidates(
        dataset,
        policy="constant_velocity",
        candidate_id="transition-cv-fixture",
        artifact_ref="outputs/captures/transition/prediction-candidates.json",
        confidence=0.8,
    )

    assert summary["schema"] == OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA
    assert validate_objectstate_transition_prediction_candidates_summary(summary) == summary
    assert summary["row_counts"] == {
        "prediction_candidates": 4,
        "constant_velocity_rows": 2,
        "action_delta_rows": 0,
        "hold_rows": 2,
        "action_conditioned_rows": 0,
        "no_action_rows": 4,
    }
    assert summary["policy"]["uses_target_pose_values"] is False
    assert summary["claim_policy"]["does_not_read_target_pose_values_for_prediction"] is True

    candidates = summary["prediction_candidates"]
    validate_objectstate_controlled_prediction_candidates(candidates)
    rows = {
        (row["source_frame_id"], row["target_frame_id"], row["object_id"]): row
        for row in candidates["predictions"]
    }
    assert rows[("frame-000000", "frame-000001", "cup-001")][
        "predicted_position"
    ] == pytest.approx([0.1, 0.2, 0.3])
    assert rows[("frame-000001", "frame-000002", "cup-001")][
        "predicted_position"
    ] == pytest.approx([0.12, 0.2, 0.3])
    assert rows[("frame-000001", "frame-000002", "cup-001")][
        "history_baseline_position"
    ] == pytest.approx([0.11, 0.2, 0.3])

    prediction_eval = evaluate_objectstate_controlled_prediction_candidates(
        capture,
        candidates,
    )
    assert prediction_eval["status"] == "objectstate_controlled_prediction_eval_pass"
    assert prediction_eval["metrics"]["state_ade"] == pytest.approx(0.005)
    assert prediction_eval["metrics"]["history_ade"] == pytest.approx(0.01)


def test_transition_prediction_candidates_do_not_read_target_pose_values():
    dataset = objectstate_transition_dataset_from_capture_manifest(
        _capture_manifest(include_action=False)
    )
    mutated_dataset = json.loads(json.dumps(dataset))
    for row in mutated_dataset["transitions"]:
        if row["target_frame_id"] == "frame-000002":
            row["state_t1"]["pose"]["position"] = [9.0, 9.0, 9.0]

    original = objectstate_transition_prediction_candidates(
        dataset,
        policy="constant_velocity",
    )
    mutated = objectstate_transition_prediction_candidates(
        mutated_dataset,
        policy="constant_velocity",
    )

    original_rows = _prediction_rows_by_key(original["prediction_candidates"])
    mutated_rows = _prediction_rows_by_key(mutated["prediction_candidates"])
    assert original_rows.keys() == mutated_rows.keys()
    for key in original_rows:
        assert mutated_rows[key]["predicted_position"] == pytest.approx(
            original_rows[key]["predicted_position"]
        )
        assert mutated_rows[key]["history_baseline_position"] == pytest.approx(
            original_rows[key]["history_baseline_position"]
        )


def test_transition_prediction_candidates_export_action_delta_rows():
    dataset = objectstate_transition_dataset_from_capture_manifest(
        _capture_manifest(include_action=True),
        require_action_transition=True,
    )

    summary = objectstate_transition_prediction_candidates(
        dataset,
        policy="action_delta",
        candidate_id="transition-action-delta-fixture",
        require_action_transition=True,
    )

    rows = _prediction_rows_by_key(summary["prediction_candidates"])
    assert summary["row_counts"]["action_delta_rows"] == 2
    assert summary["row_counts"]["action_conditioned_rows"] == 2
    assert rows[("frame-000000", "frame-000001", "cup-001")][
        "predicted_position"
    ] == pytest.approx([0.08, 0.2, 0.3])
    assert rows[("frame-000001", "frame-000002", "box-001")][
        "predicted_position"
    ] == pytest.approx([0.4, 0.21, 0.3])


def test_export_transition_prediction_candidates_cli(tmp_path, capsys):
    capture_path = tmp_path / "capture-manifest.json"
    transition_path = tmp_path / "objectstate-transitions.json"
    candidates_path = tmp_path / "prediction-candidates.json"
    summary_path = tmp_path / "transition-prediction-summary.json"
    capture_path.write_text(
        json.dumps(_capture_manifest(include_action=True), indent=2) + "\n",
        encoding="utf-8",
    )
    write_objectstate_transition_dataset(
        capture_path,
        transition_path,
        require_action_transition=True,
    )

    assert (
        main(
            [
                "object-state",
                "export-transition-prediction-candidates",
                str(transition_path),
                "--output",
                str(candidates_path),
                "--policy",
                "action_delta",
                "--candidate-id",
                "cli-transition-action-delta",
                "--candidate-source",
                "cli transition action-delta baseline",
                "--artifact-ref",
                "outputs/captures/transition/prediction-candidates.json",
                "--confidence",
                "0.7",
                "--require-action-transition",
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA}" in stdout
    assert "sample_id=transition-cup-box-001" in stdout
    assert "candidate_id=cli-transition-action-delta" in stdout
    assert "policy=action_delta" in stdout
    assert "prediction_candidate_count=4" in stdout
    assert "action_delta_rows=2" in stdout
    assert "uses_target_pose_values=false" in stdout
    assert "does_not_read_target_pose_values_for_prediction=true" in stdout
    assert summary["schema"] == OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA
    assert candidates["schema"] == "objgauss-objectstate-controlled-prediction-candidates-v1"
    assert candidates["candidate"]["candidate_id"] == "cli-transition-action-delta"


def _prediction_rows_by_key(payload):
    return {
        (row["source_frame_id"], row["target_frame_id"], row["object_id"]): row
        for row in payload["predictions"]
    }


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
            "scenario": "transition_prediction_candidates",
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
