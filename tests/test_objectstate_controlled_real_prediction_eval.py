from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_real_evidence_bundle import (
    objectstate_controlled_real_evidence_bundle_adapter_summary,
)
from objgauss.evaluation.objectstate_controlled_real_identity_eval import (
    OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA,
    objectstate_controlled_real_identity_eval,
)
from objgauss.evaluation.objectstate_controlled_real_prediction_eval import (
    OBJECTSTATE_CONTROLLED_REAL_PREDICTION_EVAL_SCHEMA,
    objectstate_controlled_real_prediction_eval,
    validate_objectstate_controlled_real_prediction_eval,
)
from objgauss.evaluation.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
)


def test_controlled_real_prediction_eval_passes_after_identity_eval():
    bundle = _real_bundle()
    identity = _identity_eval(bundle)

    summary = objectstate_controlled_real_prediction_eval(
        bundle,
        identity_eval=identity,
        prediction_candidates=_prediction_candidates(),
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REAL_PREDICTION_EVAL_SCHEMA
    assert summary["status"] == "objectstate_controlled_real_prediction_eval_pass"
    assert summary["row_counts"]["evaluated_prediction_rows"] == 1
    assert summary["row_counts"]["prediction_pass_rows"] == 1
    assert summary["metrics"]["state_ade"] == pytest.approx(0.0)
    assert summary["metrics"]["hold_last_ade"] == pytest.approx(0.1)
    assert summary["metrics"]["kalman_ade"] == pytest.approx(0.0)
    assert summary["metrics"]["identity_consistency_rate"] == 1.0
    assert summary["pass_gates"]["constant_velocity_baseline_present_if_history_allows"]
    assert summary["evaluated_real_bundle"]["gate_accounting_rows"][1][
        "accounting_status"
    ] == "pass"
    assert validate_objectstate_controlled_real_prediction_eval(summary) == summary


def test_controlled_real_prediction_eval_blocks_without_identity_eval():
    bundle = _real_bundle()

    summary = objectstate_controlled_real_prediction_eval(
        bundle,
        prediction_candidates=_prediction_candidates(),
    )

    assert summary["status"] == "objectstate_controlled_real_prediction_eval_blocked"
    assert summary["row_counts"]["evaluated_prediction_rows"] == 0
    assert summary["row_counts"]["prediction_rows_blocked"] == 1
    assert summary["prediction_accounting_rows"][0]["reason"] == "identity_not_stable"
    assert summary["evaluated_real_bundle"]["gate_accounting_rows"][1][
        "accounting_status"
    ] == "evidence_incomplete"


def test_controlled_real_prediction_eval_keeps_missing_transition_incomplete():
    bundle = _real_bundle()
    identity = _identity_eval(bundle)
    for row in bundle["gate_accounting_rows"]:
        if row["evidence_kind"] == "prediction":
            row.pop("transition_id", None)
            row.pop("object_id", None)

    summary = objectstate_controlled_real_prediction_eval(
        bundle,
        identity_eval=identity,
        prediction_candidates=_prediction_candidates(),
    )

    assert summary["status"] == "objectstate_controlled_real_prediction_eval_incomplete"
    assert summary["row_counts"]["prediction_rows_evidence_incomplete"] == 1
    assert summary["prediction_accounting_rows"][0]["reason"] == "missing_before_pose"
    assert summary["evaluated_real_bundle"]["gate_accounting_rows"][1][
        "accounting_status"
    ] == "evidence_incomplete"


def test_controlled_real_prediction_eval_blocks_identity_inconsistent_transition():
    bundle = _real_bundle()
    identity = _identity_eval(bundle)
    for row in bundle["identity_link_rows"]:
        if row["frame_id"] == "frame-000002":
            row["physical_identity_id"] = "cup-other"

    summary = objectstate_controlled_real_prediction_eval(
        bundle,
        identity_eval=identity,
        prediction_candidates=_prediction_candidates(),
    )

    assert summary["status"] == "objectstate_controlled_real_prediction_eval_blocked"
    assert summary["prediction_accounting_rows"][0]["reason"] == "identity_not_stable"
    assert summary["row_counts"]["prediction_rows_blocked"] == 1
    assert summary["metrics"]["identity_consistency_rate"] == 0.0


def test_controlled_real_prediction_eval_fails_when_underperforming_hold_last():
    bundle = _real_bundle()
    identity = _identity_eval(bundle)
    candidates = _prediction_candidates()
    candidates["predictions"][0]["predicted_position"] = [0.4, 0.0, 0.0]

    summary = objectstate_controlled_real_prediction_eval(
        bundle,
        identity_eval=identity,
        prediction_candidates=candidates,
    )

    assert summary["status"] == "objectstate_controlled_real_prediction_eval_fail"
    assert summary["row_counts"]["prediction_fail_rows"] == 1
    assert summary["metrics"]["state_ade"] == pytest.approx(0.2)
    assert summary["metrics"]["hold_last_ade"] == pytest.approx(0.1)
    assert "prediction_model_underperforms_hold_last" in summary["issues"]
    assert summary["evaluated_real_bundle"]["gate_accounting_rows"][1][
        "accounting_status"
    ] == "fail"


def test_controlled_real_prediction_eval_cli_writes_artifacts(tmp_path, capsys):
    bundle = _real_bundle()
    identity = _identity_eval(bundle)
    candidates = _prediction_candidates()
    bundle_path = tmp_path / "real-bundle.json"
    identity_path = tmp_path / "identity-summary.json"
    candidates_path = tmp_path / "prediction-candidates.json"
    output_dir = tmp_path / "prediction-eval"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    identity_path.write_text(json.dumps(identity), encoding="utf-8")
    candidates_path.write_text(json.dumps(candidates), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "eval-controlled-real-prediction",
                str(bundle_path),
                "--identity-eval",
                str(identity_path),
                "--prediction-candidates",
                str(candidates_path),
                "--output-dir",
                str(output_dir),
                "--require-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(
        (output_dir / "controlled-real-prediction-summary.json").read_text(
            encoding="utf-8"
        )
    )
    artifact_manifest = json.loads(
        (output_dir / "controlled-real-prediction-artifact-manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert f"schema={OBJECTSTATE_CONTROLLED_REAL_PREDICTION_EVAL_SCHEMA}" in stdout
    assert "evaluated_prediction_rows=1" in stdout
    assert "state_ade=0.000000" in stdout
    assert summary["status"] == "objectstate_controlled_real_prediction_eval_pass"
    assert (output_dir / "controlled-real-prediction-accounting.csv").exists()
    assert (output_dir / "controlled-real-prediction-errors.csv").exists()
    assert (output_dir / "controlled-real-prediction-baselines.json").exists()
    assert (output_dir / "evaluated-real-bundle.json").exists()
    assert artifact_manifest["artifacts"]["summary"].endswith(
        "controlled-real-prediction-summary.json"
    )


def _real_bundle():
    bundle = objectstate_controlled_real_evidence_bundle_adapter_summary(
        _capture_manifest(),
        source_summary_ref="outputs/captures/cup/capture-manifest.json",
    )["bundle"]
    second_transition = next(
        row
        for row in bundle["state_transition_rows"]
        if row["source_frame_id"] == "frame-000001"
        and row["target_frame_id"] == "frame-000002"
    )
    for row in bundle["gate_accounting_rows"]:
        if row["evidence_kind"] == "prediction":
            row["transition_id"] = second_transition["transition_id"]
            row["object_id"] = second_transition["object_id"]
    return bundle


def _identity_eval(bundle):
    return objectstate_controlled_real_identity_eval(
        bundle,
        teacher_evidence=_teacher_evidence(bundle),
        min_identity_retrieval_at_1=0.75,
    )


def _teacher_evidence(bundle):
    return {
        "schema": OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA,
        "sample_id": bundle["sample"]["sample_id"],
        "teacher_evidence_source": "manual_fixture",
        "evidence_policy": "semantic",
        "allowed_for_evaluation": True,
        "provenance": {
            "producer": "unit-test",
            "feature_space": "fixture-slot",
            "input_refs": ["fixture://capture"],
            "generation_method": "manual non-identity fixture",
        },
        "assignments": [
            {
                "object_pose_row_id": pose["row_id"],
                "slot_id": "slot-cup",
                "embedding": [1.0, 0.0],
                "confidence": 1.0,
            }
            for pose in bundle["object_pose_rows"]
        ],
    }


def _prediction_candidates():
    return {
        "schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
        "sample_id": "controlled-tabletop-cup-real-prediction-001",
        "candidate": {
            "candidate_id": "candidate-objectstate-predictor-v0",
            "source": "fixture real state-vs-baseline future pose predictions",
            "artifact_refs": [
                "outputs/controlled-real/cup-real-prediction-001/predictions.json"
            ],
        },
        "predictions": [
            {
                "source_frame_id": "frame-000001",
                "target_frame_id": "frame-000002",
                "object_id": "cup-001",
                "predicted_position": [0.2, 0.0, 0.0],
                "history_baseline_position": [0.1, 0.0, 0.0],
                "confidence": 0.95,
            }
        ],
    }


def _capture_manifest():
    return {
        "schema": "objgauss-objectstate-controlled-capture-manifest-v1",
        "sample": {
            "sample_id": "controlled-tabletop-cup-real-prediction-001",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "push_left",
            "fps": 2.0,
            "capture_device": "cam-001",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": ["outputs/captures/cup/capture-manifest.json"],
            "license": "local-research",
        },
        "objects": [{"object_id": "cup-001", "category": "cup"}],
        "actions": [],
        "frames": [
            _frame("frame-000000", 0.0, [0.0, 0.0, 0.0]),
            _frame("frame-000001", 0.5, [0.1, 0.0, 0.0]),
            _frame("frame-000002", 1.0, [0.2, 0.0, 0.0]),
        ],
    }


def _frame(frame_id, timestamp, position):
    return {
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
