from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.evaluation.objectstate_controlled_identity_eval import (
    OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
    ObjectStateControlledIdentityThresholds,
    evaluate_objectstate_controlled_identity_predictions,
    read_objectstate_controlled_identity_predictions,
    validate_objectstate_controlled_identity_eval_summary,
    validate_objectstate_controlled_identity_predictions,
)
from objgauss.datasets.objectstate_controlled_real_manifest import (
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
)
from objgauss.evaluation.objectstate_controlled_real_rows import (
    evaluate_controlled_real_manifest_reality_gate,
)
from objgauss.evaluation.objectstate_reality_gate import ObjectStateRealityGateThresholds


def test_controlled_identity_eval_outputs_pass_row_for_stable_tracks():
    summary = evaluate_objectstate_controlled_identity_predictions(
        _capture_manifest(),
        _predictions(("cup-track", "box-track"), ("cup-track", "box-track"), ("cup-track", "box-track")),
    )
    manifest = summary["controlled_real_manifest"]

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA
    assert summary["status"] == "objectstate_controlled_identity_eval_pass"
    assert summary["metrics"]["idf1"] == 1.0
    assert summary["metrics"]["track_retrieval_recall_at_1"] == 1.0
    assert summary["metrics"]["long_term_drift_rate"] == 0.0
    assert summary["metrics"]["fragmentation_rate"] == 0.0
    assert summary["metrics"]["swap_rate"] == 0.0
    assert summary["metrics"]["identity_collapse"] is False
    assert summary["metrics"]["reconstruction_noise_robustness"] == 1.0
    assert summary["metrics"]["reconstruction_noise_variant_count"] == 2
    assert manifest["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert manifest["evidence_rows"][0]["evidence_kind"] == "identity"
    assert manifest["evidence_rows"][0]["status"] == "pass"
    assert manifest["evidence_rows"][0]["metrics"]["idf1"] == 1.0
    assert (
        manifest["evidence_rows"][0]["metrics"]["track_retrieval_recall_at_1"]
        == 1.0
    )
    gate = evaluate_controlled_real_manifest_reality_gate(
        manifest,
        thresholds=ObjectStateRealityGateThresholds(
            require_prediction_pass_row=False,
            require_intervention_pass_row=False,
        ),
    )
    assert gate.as_dict()["status"] == "objectstate_reality_gate_pass"
    assert validate_objectstate_controlled_identity_eval_summary(summary) is summary


def test_controlled_identity_eval_fails_fragmented_identity():
    summary = evaluate_objectstate_controlled_identity_predictions(
        _capture_manifest(),
        _predictions(("cup-a", "box-track"), ("cup-b", "box-track"), ("cup-b", "box-track")),
    )
    identity_row = summary["controlled_real_manifest"]["evidence_rows"][0]

    assert summary["status"] == "objectstate_controlled_identity_eval_fail"
    assert summary["metrics"]["idf1"] < 0.95
    assert summary["metrics"]["fragmentation_rate"] > 0.0
    assert summary["metrics"]["long_term_drift_rate"] > 0.0
    assert summary["pass_gates"]["fragmentation_at_or_below_threshold"] is False
    assert summary["pass_gates"]["long_term_drift_at_or_below_threshold"] is False
    assert identity_row["status"] == "fail"
    assert "fragmentation_at_or_below_threshold" in identity_row["failure_reason"]


def test_controlled_identity_eval_detects_swaps():
    summary = evaluate_objectstate_controlled_identity_predictions(
        _capture_manifest(),
        _predictions(("cup-track", "box-track"), ("box-track", "cup-track"), ("cup-track", "box-track")),
    )

    assert summary["status"] == "objectstate_controlled_identity_eval_fail"
    assert summary["metrics"]["swap_rate"] == 1.0
    assert summary["pass_gates"]["swap_at_or_below_threshold"] is False


def test_controlled_identity_eval_detects_identity_collapse():
    summary = evaluate_objectstate_controlled_identity_predictions(
        _capture_manifest(),
        _predictions(("shared-track", "shared-track"), ("cup-track", "box-track"), ("cup-track", "box-track")),
        thresholds=ObjectStateControlledIdentityThresholds(
            min_idf1=0.0,
            max_fragmentation_rate=1.0,
            max_swap_rate=1.0,
        ),
    )

    assert summary["status"] == "objectstate_controlled_identity_eval_fail"
    assert summary["metrics"]["identity_collapse"] is True
    assert summary["pass_gates"]["identity_collapse_absent"] is False


def test_controlled_identity_eval_requires_reconstruction_noise_evidence():
    predictions = _predictions()
    predictions["candidate"].pop("identity_evidence")

    summary = evaluate_objectstate_controlled_identity_predictions(
        _capture_manifest(),
        predictions,
    )

    assert summary["status"] == "objectstate_controlled_identity_eval_fail"
    assert summary["metrics"]["reconstruction_noise_robustness"] is None
    assert summary["metrics"]["reconstruction_noise_variant_count"] == 0
    assert summary["pass_gates"]["reconstruction_noise_evidence_present"] is False
    assert (
        summary["pass_gates"][
            "reconstruction_noise_robustness_at_or_above_threshold"
        ]
        is False
    )


def test_controlled_identity_eval_rejects_unknown_prediction_pair():
    predictions = _predictions(("cup-track", "box-track"), ("cup-track", "box-track"), ("cup-track", "box-track"))
    predictions["predictions"][0]["frame_id"] = "missing-frame"

    with pytest.raises(ValueError, match="unknown frame"):
        evaluate_objectstate_controlled_identity_predictions(_capture_manifest(), predictions)


def test_controlled_identity_eval_rejects_duplicate_prediction_pair():
    predictions = _predictions(("cup-track", "box-track"), ("cup-track", "box-track"), ("cup-track", "box-track"))
    predictions["predictions"][1] = dict(predictions["predictions"][0])

    with pytest.raises(ValueError, match="duplicate raw track observation"):
        evaluate_objectstate_controlled_identity_predictions(_capture_manifest(), predictions)


def test_controlled_identity_eval_cannot_pass_unbounded_far_observations():
    predictions = _predictions()
    predictions["candidate"].pop("max_association_distance")
    for row in predictions["predictions"]:
        row["predicted_position"] = [1000.0, 1000.0, 1000.0]

    summary = evaluate_objectstate_controlled_identity_predictions(
        _capture_manifest(),
        predictions,
    )

    assert summary["status"] == "objectstate_controlled_identity_eval_fail"
    assert summary["metrics"]["idf1"] == 1.0
    assert summary["metrics"]["association_max_distance"] > 1000.0
    assert summary["metrics"]["unmatched_prediction_count"] == 0
    assert (
        summary["pass_gates"][
            "finite_association_distance_threshold_configured"
        ]
        is False
    )


def test_controlled_identity_eval_rejects_non_finite_association_distance():
    predictions = _predictions()
    predictions["candidate"]["max_association_distance"] = float("inf")

    with pytest.raises(ValueError, match="max_association_distance must be finite"):
        evaluate_objectstate_controlled_identity_predictions(
            _capture_manifest(),
            predictions,
        )


def test_controlled_identity_eval_legacy_preassociated_rows_cannot_pass():
    summary = evaluate_objectstate_controlled_identity_predictions(
        _capture_manifest(),
        _legacy_predictions(),
        thresholds=ObjectStateControlledIdentityThresholds(
            min_idf1=0.0,
            min_track_retrieval_recall_at_1=0.0,
            max_fragmentation_rate=1.0,
            max_long_term_drift_rate=1.0,
            max_swap_rate=1.0,
            min_reconstruction_noise_robustness=0.0,
            min_reconstruction_noise_variants=1,
            require_no_identity_collapse=False,
        ),
    )

    assert summary["status"] == "objectstate_controlled_identity_eval_fail"
    assert (
        summary["metrics"]["prediction_association_mode"]
        == "legacy_gt_object_preassociated"
    )
    assert summary["metrics"]["raw_prediction_observations"] is False
    assert summary["pass_gates"]["raw_prediction_observations_only"] is False
    gate = evaluate_controlled_real_manifest_reality_gate(
        summary["controlled_real_manifest"],
        thresholds=ObjectStateRealityGateThresholds(
            require_prediction_pass_row=False,
            require_intervention_pass_row=False,
        ),
    )
    identity_row = gate.rows[0]
    assert identity_row.status == "fail"
    assert identity_row.metrics["raw_prediction_observations"] is False
    assert "raw_prediction_observations_required" in identity_row.failure_reason
    assert gate.as_dict()["status"] == "objectstate_reality_gate_fail"


def test_controlled_identity_predictions_read_json_file(tmp_path):
    predictions_path = tmp_path / "identity-predictions.json"
    predictions_path.write_text(json.dumps(_predictions()), encoding="utf-8")

    predictions = read_objectstate_controlled_identity_predictions(predictions_path)

    assert predictions["schema"] == OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA
    assert predictions["sample_id"] == "controlled-tabletop-cup-box-identity-001"
    assert validate_objectstate_controlled_identity_predictions(predictions) == predictions


def test_object_state_eval_controlled_identity_cli_writes_summary_and_manifest(
    tmp_path,
    capsys,
):
    capture_path = tmp_path / "capture.json"
    predictions_path = tmp_path / "identity-predictions.json"
    summary_path = tmp_path / "identity-summary.json"
    controlled_real_path = tmp_path / "controlled-real.json"
    capture_path.write_text(json.dumps(_capture_manifest()), encoding="utf-8")
    predictions_path.write_text(json.dumps(_predictions()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "eval-controlled-identity",
                str(capture_path),
                str(predictions_path),
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

    assert f"schema={OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA}" in stdout
    assert "identity_eval_status=objectstate_controlled_identity_eval_pass" in stdout
    assert "idf1=1.000000" in stdout
    assert "track_retrieval_recall_at_1=1.000000" in stdout
    assert "long_term_drift_rate=0.000000" in stdout
    assert "reconstruction_noise_robustness=1.000000" in stdout
    assert summary["status"] == "objectstate_controlled_identity_eval_pass"
    assert controlled_real["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert controlled_real["evidence_rows"][0]["status"] == "pass"


def _predictions(
    frame0: tuple[str, str] = ("cup-track", "box-track"),
    frame1: tuple[str, str] = ("cup-track", "box-track"),
    frame2: tuple[str, str] = ("cup-track", "box-track"),
):
    identities_by_frame = (frame0, frame1, frame2)
    rows = []
    for frame_index, identities in enumerate(identities_by_frame):
        for x, identity in zip((0.1, 0.4), identities):
            rows.append(
                {
                    "frame_id": f"frame-{frame_index:06d}",
                    "predicted_identity": identity,
                    "predicted_position": [
                        x + 0.01 * frame_index,
                        0.2,
                        0.3,
                    ],
                    "confidence": 0.95,
                }
            )
    return {
        "schema": OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
        "sample_id": "controlled-tabletop-cup-box-identity-001",
        "association_mode": "raw_track_observations",
        "candidate": {
            "candidate_id": "candidate-objectstate-identity-v0",
            "source": "fixture candidate identity tracks",
            "artifact_refs": [
                "outputs/controlled-real/cup-box-identity-001/objectstates.json"
            ],
            "identity_evidence": {
                "reconstruction_noise_robustness": 1.0,
                "reconstruction_noise_variant_count": 2,
                "source": "fixture repeated Gaussian reconstruction noise variants",
            },
            "max_association_distance": 0.05,
        },
        "predictions": rows,
    }


def _legacy_predictions():
    predictions = _predictions()
    predictions.pop("association_mode")
    for row_index, row in enumerate(predictions["predictions"]):
        row.pop("predicted_position")
        row["object_id"] = ("cup-001", "box-001")[row_index % 2]
    return predictions


def _capture_manifest():
    frames = []
    for frame_index, timestamp in enumerate((0.0, 0.033333, 0.066667)):
        frame_objects = []
        for object_id, x in (("cup-001", 0.1), ("box-001", 0.4)):
            frame_objects.append(
                {
                    "object_id": object_id,
                    "visible": not (object_id == "cup-001" and frame_index == 1),
                    "occlusion_fraction": 0.8 if object_id == "cup-001" and frame_index == 1 else 0.0,
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
            "sample_id": "controlled-tabletop-cup-box-identity-001",
            "source_kind": "controlled_real",
            "object_category": "cup_box",
            "scenario": "cross_view_occlusion_reappearance",
            "fps": 30.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/controlled-real/cup-box-identity-001/capture.json",
                "outputs/controlled-real/cup-box-identity-001/rgb/",
                "outputs/controlled-real/cup-box-identity-001/gaussians/",
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
