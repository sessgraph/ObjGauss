from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.evaluation.objectstate_controlled_intervention_eval import (
    evaluate_objectstate_controlled_intervention_candidates,
    validate_objectstate_controlled_intervention_candidates,
)
from objgauss.datasets.objectstate_transition_dataset import (
    objectstate_transition_dataset_from_capture_manifest,
    write_objectstate_transition_dataset,
)
from objgauss.pipelines.objectstate_transition_intervention_candidates import (
    OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA,
    objectstate_transition_intervention_candidates,
    validate_objectstate_transition_intervention_candidates_summary,
)


def test_transition_intervention_candidates_export_action_delta():
    capture = _capture_manifest()
    dataset = objectstate_transition_dataset_from_capture_manifest(
        capture,
        require_action_transition=True,
    )

    summary = objectstate_transition_intervention_candidates(
        dataset,
        policy="action_delta",
        candidate_id="transition-action-fixture",
        artifact_ref="outputs/captures/action/intervention-candidates.json",
        confidence=0.8,
    )

    assert summary["schema"] == OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA
    assert validate_objectstate_transition_intervention_candidates_summary(summary) == summary
    assert summary["row_counts"] == {
        "intervention_candidates": 1,
        "action_delta_rows": 1,
        "hold_action_rows": 0,
        "skipped_action_contexts": 1,
    }
    assert summary["policy"]["uses_target_pose_values"] is False
    assert summary["claim_policy"]["does_not_read_target_pose_values_for_prediction"] is True

    candidates = summary["intervention_candidates"]
    validate_objectstate_controlled_intervention_candidates(candidates)
    row = candidates["interventions"][0]
    assert row["source_frame_id"] == "frame-000000"
    assert row["target_frame_id"] == "frame-000001"
    assert row["action_conditioned_position"] == pytest.approx([0.0, 0.2, 0.3])
    assert row["no_action_baseline_position"] == pytest.approx([0.1, 0.2, 0.3])

    intervention_eval = evaluate_objectstate_controlled_intervention_candidates(
        capture,
        candidates,
    )
    assert intervention_eval["status"] == "objectstate_controlled_intervention_eval_pass"
    assert intervention_eval["metrics"]["action_conditioned_ade"] == pytest.approx(0.0)
    assert intervention_eval["metrics"]["no_action_ade"] == pytest.approx(0.1)
    assert intervention_eval["metrics"]["wrong_direction_rate"] == 0.0


def test_transition_intervention_candidates_do_not_read_target_pose_values():
    dataset = objectstate_transition_dataset_from_capture_manifest(
        _capture_manifest(),
        require_action_transition=True,
    )
    mutated_dataset = json.loads(json.dumps(dataset))
    mutated_dataset["transitions"][0]["state_t1"]["pose"]["position"] = [9.0, 9.0, 9.0]

    original = objectstate_transition_intervention_candidates(
        dataset,
        policy="action_delta",
    )
    mutated = objectstate_transition_intervention_candidates(
        mutated_dataset,
        policy="action_delta",
    )

    original_row = original["intervention_candidates"]["interventions"][0]
    mutated_row = mutated["intervention_candidates"]["interventions"][0]
    assert mutated_row["action_conditioned_position"] == pytest.approx(
        original_row["action_conditioned_position"]
    )
    assert mutated_row["no_action_baseline_position"] == pytest.approx(
        original_row["no_action_baseline_position"]
    )


def test_transition_intervention_candidates_hold_action_negative_candidate():
    dataset = objectstate_transition_dataset_from_capture_manifest(
        _capture_manifest(),
        require_action_transition=True,
    )

    summary = objectstate_transition_intervention_candidates(
        dataset,
        policy="hold_action",
    )

    assert summary["row_counts"]["hold_action_rows"] == 1
    row = summary["intervention_candidates"]["interventions"][0]
    assert row["action_conditioned_position"] == pytest.approx([0.1, 0.2, 0.3])
    intervention_eval = evaluate_objectstate_controlled_intervention_candidates(
        _capture_manifest(),
        summary["intervention_candidates"],
    )
    assert intervention_eval["status"] == "objectstate_controlled_intervention_eval_fail"
    assert intervention_eval["metrics"]["wrong_direction_rate"] == 1.0


def test_export_transition_intervention_candidates_cli(tmp_path, capsys):
    capture_path = tmp_path / "capture-manifest.json"
    transition_path = tmp_path / "objectstate-transitions.json"
    candidates_path = tmp_path / "intervention-candidates.json"
    summary_path = tmp_path / "transition-intervention-summary.json"
    capture_path.write_text(
        json.dumps(_capture_manifest(), indent=2) + "\n",
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
                "export-transition-intervention-candidates",
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
                "outputs/captures/action/intervention-candidates.json",
                "--confidence",
                "0.7",
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidates = json.loads(candidates_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA}" in stdout
    assert "sample_id=transition-action-cup-001" in stdout
    assert "candidate_id=cli-transition-action-delta" in stdout
    assert "policy=action_delta" in stdout
    assert "intervention_candidate_count=1" in stdout
    assert "action_delta_rows=1" in stdout
    assert "uses_target_pose_values=false" in stdout
    assert "does_not_read_target_pose_values_for_prediction=true" in stdout
    assert summary["schema"] == OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA
    assert candidates["schema"] == "objgauss-objectstate-controlled-intervention-candidates-v1"
    assert candidates["candidate"]["candidate_id"] == "cli-transition-action-delta"


def _capture_manifest() -> dict:
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "transition-action-cup-001",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "transition_intervention_candidates",
            "fps": 2.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/captures/transition-action-cup-001/capture-manifest.json",
                "outputs/captures/transition-action-cup-001/rgb/",
                "outputs/captures/transition-action-cup-001/gaussians/",
            ],
            "license": "local controlled capture; not public release",
        },
        "objects": [
            {"object_id": "cup-001", "category": "cup", "instance_label": "blue cup"},
        ],
        "actions": [
            {
                "action_id": "push-left-001",
                "action_type": "push_left",
                "object_id": "cup-001",
                "start_timestamp": 0.0,
                "end_timestamp": 0.5,
                "actor": "fixture-human",
                "vector": [-0.1, 0.0, 0.0],
            }
        ],
        "frames": [
            _frame("frame-000000", 0.0, [0.1, 0.2, 0.3], action_id="push-left-001"),
            _frame("frame-000001", 0.5, [0.0, 0.2, 0.3]),
            _frame("frame-000002", 1.0, [0.0, 0.2, 0.3]),
        ],
    }


def _frame(
    frame_id: str,
    timestamp: float,
    position: list[float],
    *,
    action_id: str | None = None,
) -> dict:
    frame = {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "observation": {
            "rgb": f"rgb/{frame_id}.png",
            "gaussian": f"gaussians/{frame_id}.ply",
        },
        "objects": [
            {
                "object_id": "cup-001",
                "visible": True,
                "occlusion_fraction": 0.0,
                "pose": {
                    "position": position,
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
        ],
    }
    if action_id is not None:
        frame["action_id"] = action_id
    return frame
