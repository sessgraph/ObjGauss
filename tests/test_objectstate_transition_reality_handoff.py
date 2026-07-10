from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.datasets.objectstate_transition_dataset import (
    objectstate_transition_dataset_from_capture_manifest,
    write_objectstate_transition_dataset,
)
from objgauss.pipelines.objectstate_transition_reality_handoff import (
    OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA,
    objectstate_transition_reality_handoff,
    validate_objectstate_transition_reality_handoff_summary,
    write_objectstate_transition_reality_handoff,
)


def test_transition_reality_handoff_runs_prediction_and_intervention_rows():
    capture = _capture_manifest()
    dataset = objectstate_transition_dataset_from_capture_manifest(
        capture,
        require_action_transition=True,
    )

    summary = objectstate_transition_reality_handoff(capture, dataset)

    assert summary["schema"] == OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA
    assert summary["status"] == "objectstate_transition_reality_handoff_pass"
    assert validate_objectstate_transition_reality_handoff_summary(summary) == summary
    assert summary["handoff_gates"] == {
        "intervention_action_gt_ready": True,
        "transition_dataset_ready": True,
        "action_transition_ready": True,
        "prediction_eval_pass": True,
        "intervention_eval_pass": True,
        "prediction_generator_target_pose_guard": True,
        "intervention_generator_target_pose_guard": True,
        "partial_reality_gate_pass": True,
    }
    assert summary["prediction_eval"]["status"] == (
        "objectstate_controlled_prediction_eval_pass"
    )
    assert summary["intervention_eval"]["status"] == (
        "objectstate_controlled_intervention_eval_pass"
    )
    assert summary["prediction_eval"]["metrics"]["state_ade"] == pytest.approx(0.0)
    assert summary["intervention_eval"]["metrics"]["action_conditioned_ade"] == (
        pytest.approx(0.0)
    )
    rows = {
        row["evidence_kind"]: row
        for row in summary["controlled_real_manifest"]["evidence_rows"]
    }
    assert rows["identity"]["status"] == "blocked"
    assert rows["prediction"]["status"] == "pass"
    assert rows["intervention"]["status"] == "pass"
    assert summary["controlled_real_summary"]["pass_row_count"] == 2
    assert summary["controlled_real_summary"]["blocked_row_count"] == 1
    assert summary["claim_policy"]["baseline_candidates_not_learned_model"] is True
    assert summary["claim_policy"]["intervention_action_gt_preflight_required"] is True
    assert summary["claim_policy"]["does_not_evaluate_identity"] is True
    assert summary["claim_policy"]["does_not_claim_world_model"] is True


def test_transition_reality_handoff_blocks_weak_intervention_action_gt():
    capture = _capture_manifest()
    capture["actions"][0]["vector"] = [0.0, 0.0, 0.0]
    dataset = objectstate_transition_dataset_from_capture_manifest(
        capture,
        require_action_transition=True,
    )

    with pytest.raises(
        ValueError,
        match="requires intervention action GT readiness.*non-zero vector",
    ):
        objectstate_transition_reality_handoff(capture, dataset)


def test_transition_reality_handoff_records_failed_intervention_candidate():
    capture = _capture_manifest()
    dataset = objectstate_transition_dataset_from_capture_manifest(
        capture,
        require_action_transition=True,
    )

    summary = objectstate_transition_reality_handoff(
        capture,
        dataset,
        intervention_policy="hold_action",
    )

    assert summary["status"] == "objectstate_transition_reality_handoff_fail"
    assert summary["handoff_gates"]["prediction_eval_pass"] is True
    assert summary["handoff_gates"]["intervention_eval_pass"] is False
    assert summary["handoff_gates"]["partial_reality_gate_pass"] is False
    assert any("controlled intervention eval did not pass" in item for item in summary["issues"])
    rows = {
        row["evidence_kind"]: row
        for row in summary["controlled_real_manifest"]["evidence_rows"]
    }
    assert rows["intervention"]["status"] == "fail"
    assert "controlled intervention metrics failed" in rows["intervention"][
        "failure_reason"
    ]
    assert validate_objectstate_transition_reality_handoff_summary(summary) == summary


def test_transition_reality_handoff_writer_and_cli(tmp_path, capsys):
    capture_path = tmp_path / "capture-manifest.json"
    transition_path = tmp_path / "objectstate-transitions.json"
    output_dir = tmp_path / "transition-reality-handoff"
    cli_output_dir = tmp_path / "transition-reality-handoff-cli"
    capture_path.write_text(
        json.dumps(_capture_manifest(), indent=2) + "\n",
        encoding="utf-8",
    )
    write_objectstate_transition_dataset(
        capture_path,
        transition_path,
        require_action_transition=True,
    )

    direct = write_objectstate_transition_reality_handoff(
        capture_path,
        transition_path,
        output_dir,
    )

    assert direct["status"] == "objectstate_transition_reality_handoff_pass"
    assert (output_dir / "prediction-candidates.json").is_file()
    assert (output_dir / "intervention-candidates.json").is_file()
    assert (output_dir / "prediction-eval-summary.json").is_file()
    assert (output_dir / "intervention-eval-summary.json").is_file()
    assert (output_dir / "transition-reality-handoff-summary.json").is_file()
    assert direct["files"]["handoff_summary"] == str(
        output_dir / "transition-reality-handoff-summary.json"
    )

    assert (
        main(
            [
                "object-state",
                "transition-reality-handoff",
                str(capture_path),
                str(transition_path),
                "--output-dir",
                str(cli_output_dir),
                "--prediction-candidate-id",
                "cli-transition-prediction",
                "--intervention-candidate-id",
                "cli-transition-intervention",
                "--confidence",
                "0.7",
                "--require-gaussian-refs",
                "--require-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(
        (cli_output_dir / "transition-reality-handoff-summary.json").read_text(
            encoding="utf-8"
        )
    )
    prediction_candidates = json.loads(
        (cli_output_dir / "prediction-candidates.json").read_text(encoding="utf-8")
    )
    intervention_candidates = json.loads(
        (cli_output_dir / "intervention-candidates.json").read_text(encoding="utf-8")
    )
    assert f"schema={OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA}" in stdout
    assert "sample_id=transition-action-cup-001" in stdout
    assert "transition_reality_handoff_status=objectstate_transition_reality_handoff_pass" in stdout
    assert "intervention_action_gt_ready=true" in stdout
    assert "prediction_eval_status=objectstate_controlled_prediction_eval_pass" in stdout
    assert "intervention_eval_status=objectstate_controlled_intervention_eval_pass" in stdout
    assert "partial_reality_gate_status=objectstate_reality_gate_pass" in stdout
    assert "pass_rows=2" in stdout
    assert "blocked_rows=1" in stdout
    assert summary["schema"] == OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA
    assert summary["candidate"]["prediction_candidate_id"] == "cli-transition-prediction"
    assert summary["candidate"]["intervention_candidate_id"] == "cli-transition-intervention"
    assert prediction_candidates["schema"] == (
        "objgauss-objectstate-controlled-prediction-candidates-v1"
    )
    assert intervention_candidates["schema"] == (
        "objgauss-objectstate-controlled-intervention-candidates-v1"
    )


def _capture_manifest() -> dict:
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "transition-action-cup-001",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "transition_reality_handoff",
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
            _frame("frame-000002", 1.0, [-0.1, 0.2, 0.3]),
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
