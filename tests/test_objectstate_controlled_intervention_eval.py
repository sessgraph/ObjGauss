from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.evaluation.objectstate_controlled_intervention_eval import (
    OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
    OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA,
    evaluate_objectstate_controlled_intervention_candidates,
    read_objectstate_controlled_intervention_candidates,
    validate_objectstate_controlled_intervention_candidates,
    validate_objectstate_controlled_intervention_eval_summary,
)
from objgauss.datasets.objectstate_controlled_real_manifest import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
)
from objgauss.evaluation.objectstate_controlled_real_rows import (
    evaluate_controlled_real_manifest_reality_gate,
)
from objgauss.evaluation.objectstate_reality_gate import ObjectStateRealityGateThresholds


def test_controlled_intervention_eval_outputs_pass_row_for_action_prediction():
    summary = evaluate_objectstate_controlled_intervention_candidates(
        _capture_manifest(),
        _intervention_candidates(),
    )
    manifest = summary["controlled_real_manifest"]

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA
    assert summary["status"] == "objectstate_controlled_intervention_eval_pass"
    assert summary["metrics"]["intervention_count"] == 1
    assert summary["metrics"]["action_conditioned_ade"] == pytest.approx(0.0)
    assert summary["metrics"]["no_action_ade"] == pytest.approx(0.1)
    assert summary["metrics"]["intervention_gain"] == pytest.approx(0.1)
    assert summary["metrics"]["counterfactual_outcome_accuracy"] == 1.0
    assert summary["metrics"]["wrong_direction_rate"] == 0.0
    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert manifest["evidence_rows"][2]["evidence_kind"] == "intervention"
    assert manifest["evidence_rows"][2]["status"] == "pass"
    assert (
        manifest["evidence_rows"][2]["metrics"]["action_conditioned_ade"]
        == pytest.approx(0.0)
    )

    gate = evaluate_controlled_real_manifest_reality_gate(
        manifest,
        thresholds=ObjectStateRealityGateThresholds(
            require_identity_pass_row=False,
            require_prediction_pass_row=False,
        ),
    )
    assert gate.as_dict()["status"] == "objectstate_reality_gate_pass"
    assert validate_objectstate_controlled_intervention_eval_summary(summary) is summary


def test_controlled_intervention_eval_fails_wrong_direction():
    candidates = _intervention_candidates()
    candidates["interventions"][0]["action_conditioned_position"] = [0.2, 0.2, 0.3]

    summary = evaluate_objectstate_controlled_intervention_candidates(
        _capture_manifest(),
        candidates,
    )
    intervention_row = summary["controlled_real_manifest"]["evidence_rows"][2]

    assert summary["status"] == "objectstate_controlled_intervention_eval_fail"
    assert summary["metrics"]["wrong_direction_rate"] == 1.0
    assert summary["pass_gates"]["wrong_direction_rate_at_or_below_threshold"] is False
    assert intervention_row["status"] == "fail"
    assert "wrong_direction_rate_at_or_below_threshold" in intervention_row["failure_reason"]


def test_controlled_intervention_eval_rejects_unknown_action():
    candidates = _intervention_candidates()
    candidates["interventions"][0]["action_id"] = "missing-action"

    with pytest.raises(ValueError, match="unknown action_id"):
        evaluate_objectstate_controlled_intervention_candidates(
            _capture_manifest(),
            candidates,
        )


def test_controlled_intervention_eval_requires_action_vector():
    capture = _capture_manifest()
    capture["actions"][0].pop("vector")

    with pytest.raises(ValueError, match="requires vector"):
        evaluate_objectstate_controlled_intervention_candidates(
            capture,
            _intervention_candidates(),
        )


def test_controlled_intervention_candidates_read_json_file(tmp_path):
    candidates_path = tmp_path / "intervention-candidates.json"
    candidates_path.write_text(json.dumps(_intervention_candidates()), encoding="utf-8")

    candidates = read_objectstate_controlled_intervention_candidates(candidates_path)

    assert candidates["schema"] == OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA
    assert candidates["sample_id"] == "controlled-tabletop-cup-action-001"
    assert validate_objectstate_controlled_intervention_candidates(candidates) == candidates


def test_object_state_eval_controlled_intervention_cli_writes_summary_and_manifest(
    tmp_path,
    capsys,
):
    capture_path = tmp_path / "capture.json"
    candidates_path = tmp_path / "intervention-candidates.json"
    summary_path = tmp_path / "intervention-summary.json"
    controlled_real_path = tmp_path / "controlled-real.json"
    capture_path.write_text(json.dumps(_capture_manifest()), encoding="utf-8")
    candidates_path.write_text(json.dumps(_intervention_candidates()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "eval-controlled-intervention",
                str(capture_path),
                str(candidates_path),
                "--require-pass",
                "--summary-output",
                str(summary_path),
                "--controlled-real-output",
                str(controlled_real_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    controlled_real = json.loads(controlled_real_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA}" in stdout
    assert "intervention_eval_status=objectstate_controlled_intervention_eval_pass" in stdout
    assert "action_conditioned_ade=0.000000" in stdout
    assert "counterfactual_outcome_accuracy=1.000000" in stdout
    assert "wrong_direction_rate=0.000000" in stdout
    assert summary["status"] == "objectstate_controlled_intervention_eval_pass"
    assert controlled_real["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert controlled_real["evidence_rows"][2]["status"] == "pass"


def _intervention_candidates():
    return {
        "schema": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
        "sample_id": "controlled-tabletop-cup-action-001",
        "candidate": {
            "candidate_id": "candidate-objectstate-intervention-v0",
            "source": "fixture action-conditioned future pose predictions",
            "artifact_refs": [
                "outputs/controlled-real/cup-action-001/interventions.json"
            ],
        },
        "interventions": [
            {
                "source_frame_id": "frame-000000",
                "target_frame_id": "frame-000002",
                "object_id": "cup-001",
                "action_id": "push-left-001",
                "action_conditioned_position": [0.0, 0.2, 0.3],
                "no_action_baseline_position": [0.1, 0.2, 0.3],
                "confidence": 0.95,
            }
        ],
    }


def _capture_manifest():
    frames = []
    for frame_id, timestamp, x, action_id in (
        ("frame-000000", 0.0, 0.1, None),
        ("frame-000001", 0.5, 0.05, "push-left-001"),
        ("frame-000002", 1.0, 0.0, None),
    ):
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
                        "position": [x, 0.2, 0.3],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            ],
        }
        if action_id:
            frame["action_id"] = action_id
        frames.append(frame)
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "controlled-tabletop-cup-action-001",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "push_left_intervention",
            "fps": 2.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/controlled-real/cup-action-001/capture.json",
                "outputs/controlled-real/cup-action-001/rgb/",
                "outputs/controlled-real/cup-action-001/gaussians/",
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
                "end_timestamp": 1.0,
                "actor": "fixture-human",
                "vector": [-0.1, 0.0, 0.0],
            }
        ],
        "frames": frames,
    }
