from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.evaluation.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
    ObjectStateControlledPredictionThresholds,
    evaluate_objectstate_controlled_prediction_candidates,
    read_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_eval_summary,
)
from objgauss.datasets.objectstate_controlled_real_manifest import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
)
from objgauss.evaluation.objectstate_controlled_real_rows import (
    evaluate_controlled_real_manifest_reality_gate,
)
from objgauss.evaluation.objectstate_reality_gate import ObjectStateRealityGateThresholds


def test_controlled_prediction_eval_outputs_pass_row_for_state_prediction():
    summary = evaluate_objectstate_controlled_prediction_candidates(
        _capture_manifest(),
        _prediction_candidates(),
    )
    manifest = summary["controlled_real_manifest"]

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA
    assert summary["status"] == "objectstate_controlled_prediction_eval_pass"
    assert summary["metrics"]["prediction_count"] == 2
    assert summary["metrics"]["state_ade"] == pytest.approx(0.005)
    assert summary["metrics"]["history_ade"] == pytest.approx(0.04)
    assert summary["metrics"]["prediction_gap_vs_history_model"] == pytest.approx(-0.035)
    assert summary["pass_gates"]["state_ade_at_or_below_threshold"] is True
    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert manifest["evidence_rows"][1]["evidence_kind"] == "prediction"
    assert manifest["evidence_rows"][1]["status"] == "pass"
    assert manifest["evidence_rows"][1]["metrics"]["state_ade"] == pytest.approx(0.005)

    gate = evaluate_controlled_real_manifest_reality_gate(
        manifest,
        thresholds=ObjectStateRealityGateThresholds(
            require_identity_pass_row=False,
            require_intervention_pass_row=False,
        ),
    )
    assert gate.as_dict()["status"] == "objectstate_reality_gate_pass"
    assert validate_objectstate_controlled_prediction_eval_summary(summary) is summary


def test_controlled_prediction_eval_fails_bad_state_prediction():
    candidates = _prediction_candidates()
    for item in candidates["predictions"]:
        item["predicted_position"][0] += 0.2

    summary = evaluate_objectstate_controlled_prediction_candidates(
        _capture_manifest(),
        candidates,
    )
    prediction_row = summary["controlled_real_manifest"]["evidence_rows"][1]

    assert summary["status"] == "objectstate_controlled_prediction_eval_fail"
    assert summary["metrics"]["state_ade"] > 0.05
    assert summary["metrics"]["prediction_gap_vs_history_model"] > 0.02
    assert summary["pass_gates"]["state_ade_at_or_below_threshold"] is False
    assert summary["pass_gates"]["prediction_gap_at_or_below_threshold"] is False
    assert prediction_row["status"] == "fail"
    assert "state_ade_at_or_below_threshold" in prediction_row["failure_reason"]


def test_controlled_prediction_eval_rejects_unknown_target_pose():
    candidates = _prediction_candidates()
    candidates["predictions"][0]["target_frame_id"] = "missing-frame"

    with pytest.raises(ValueError, match="unknown target frame/object pose"):
        evaluate_objectstate_controlled_prediction_candidates(
            _capture_manifest(),
            candidates,
        )


def test_controlled_prediction_eval_rejects_duplicate_prediction_tuple():
    candidates = _prediction_candidates()
    candidates["predictions"].append(dict(candidates["predictions"][0]))

    with pytest.raises(ValueError, match="duplicate source/target/object"):
        evaluate_objectstate_controlled_prediction_candidates(
            _capture_manifest(),
            candidates,
        )


def test_controlled_prediction_candidates_read_json_file(tmp_path):
    candidates_path = tmp_path / "prediction-candidates.json"
    candidates_path.write_text(json.dumps(_prediction_candidates()), encoding="utf-8")

    candidates = read_objectstate_controlled_prediction_candidates(candidates_path)

    assert candidates["schema"] == OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA
    assert candidates["sample_id"] == "controlled-tabletop-cup-box-prediction-001"
    assert validate_objectstate_controlled_prediction_candidates(candidates) == candidates


def test_object_state_eval_controlled_prediction_cli_writes_summary_and_manifest(
    tmp_path,
    capsys,
):
    capture_path = tmp_path / "capture.json"
    candidates_path = tmp_path / "prediction-candidates.json"
    summary_path = tmp_path / "prediction-summary.json"
    controlled_real_path = tmp_path / "controlled-real.json"
    capture_path.write_text(json.dumps(_capture_manifest()), encoding="utf-8")
    candidates_path.write_text(json.dumps(_prediction_candidates()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "eval-controlled-prediction",
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

    assert f"schema={OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA}" in stdout
    assert "prediction_eval_status=objectstate_controlled_prediction_eval_pass" in stdout
    assert "state_ade=0.005000" in stdout
    assert "history_ade=0.040000" in stdout
    assert "prediction_gap_vs_history_model=-0.035000" in stdout
    assert summary["status"] == "objectstate_controlled_prediction_eval_pass"
    assert controlled_real["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert controlled_real["evidence_rows"][1]["status"] == "pass"


def _prediction_candidates():
    return {
        "schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
        "sample_id": "controlled-tabletop-cup-box-prediction-001",
        "candidate": {
            "candidate_id": "candidate-objectstate-predictor-v0",
            "source": "fixture state-vs-history future pose predictions",
            "artifact_refs": [
                "outputs/controlled-real/cup-box-prediction-001/predictions.json"
            ],
        },
        "predictions": [
            {
                "source_frame_id": "frame-000000",
                "target_frame_id": "frame-000002",
                "object_id": "cup-001",
                "predicted_position": [0.125, 0.2, 0.3],
                "history_baseline_position": [0.08, 0.2, 0.3],
                "confidence": 0.95,
            },
            {
                "source_frame_id": "frame-000000",
                "target_frame_id": "frame-000002",
                "object_id": "box-001",
                "predicted_position": [0.425, 0.2, 0.3],
                "history_baseline_position": [0.38, 0.2, 0.3],
                "confidence": 0.95,
            },
        ],
    }


def _capture_manifest():
    frames = []
    for frame_index, timestamp in enumerate((0.0, 0.5, 1.0)):
        frame_objects = []
        for object_id, x in (("cup-001", 0.1), ("box-001", 0.4)):
            frame_objects.append(
                {
                    "object_id": object_id,
                    "visible": True,
                    "occlusion_fraction": 0.0,
                    "pose": {
                        "position": [x + 0.01 * frame_index, 0.2, 0.3],
                        "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                    },
                }
            )
        frames.append(
            {
                "frame_id": f"frame-{frame_index:06d}",
                "timestamp": timestamp,
                "observation": {
                    "rgb": f"rgb/{frame_index:06d}.png",
                    "gaussian": f"gaussians/{frame_index:06d}.ply",
                },
                "objects": frame_objects,
            }
        )
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "controlled-tabletop-cup-box-prediction-001",
            "source_kind": "controlled_real",
            "object_category": "cup_box",
            "scenario": "future_pose_prediction",
            "fps": 2.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/controlled-real/cup-box-prediction-001/capture.json",
                "outputs/controlled-real/cup-box-prediction-001/rgb/",
                "outputs/controlled-real/cup-box-prediction-001/gaussians/",
            ],
            "license": "local controlled capture; not public release",
        },
        "objects": [
            {"object_id": "cup-001", "category": "cup", "instance_label": "blue cup"},
            {"object_id": "box-001", "category": "box", "instance_label": "red box"},
        ],
        "actions": [],
        "frames": frames,
    }
