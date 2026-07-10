from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_real_evidence_bundle import (
    objectstate_controlled_real_evidence_bundle_adapter_summary,
)
from objgauss.evaluation.objectstate_controlled_real_identity_eval import (
    OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA,
    OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA,
    objectstate_controlled_real_identity_eval,
    validate_objectstate_controlled_real_identity_eval,
)


def test_controlled_real_identity_eval_keeps_gt_keyed_teacher_evidence_diagnostic():
    bundle = _real_bundle(two_objects=True)
    teacher = _teacher_evidence(bundle, collapse=False)

    summary = objectstate_controlled_real_identity_eval(
        bundle,
        teacher_evidence=teacher,
        min_identity_retrieval_at_1=0.75,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA
    assert summary["status"] == "objectstate_controlled_real_identity_eval_fail"
    assert summary["row_counts"]["evaluated_identity_rows"] == 1
    assert summary["row_counts"]["identity_pass_rows"] == 0
    assert summary["row_counts"]["identity_fail_rows"] == 1
    assert summary["row_counts"]["identity_rows_blocked"] == 0
    assert summary["metrics"]["identity_retrieval_at_1"] == 1.0
    assert summary["teacher_evidence_coverage"] == 1.0
    assert summary["pass_gates"]["raw_prediction_observations_only"] is False
    assert (
        summary["identity_accounting_rows"][0]["pass_gates"][
            "raw_prediction_observations_only"
        ]
        is False
    )
    assert "GT object_pose_row_id" in summary["identity_accounting_rows"][0][
        "reason"
    ]
    assert summary["evaluated_real_bundle"]["gate_accounting_rows"][0][
        "accounting_status"
    ] == "fail"
    assert validate_objectstate_controlled_real_identity_eval(summary) == summary


def test_controlled_real_identity_eval_blocks_without_teacher_evidence():
    bundle = _real_bundle(two_objects=True)

    summary = objectstate_controlled_real_identity_eval(bundle)

    assert summary["status"] == "objectstate_controlled_real_identity_eval_blocked"
    assert summary["row_counts"]["evaluated_identity_rows"] == 0
    assert summary["row_counts"]["identity_rows_blocked"] == 1
    assert summary["identity_accounting_rows"][0]["reason"] == "missing_teacher_evidence"
    assert summary["evaluated_real_bundle"]["gate_accounting_rows"][0][
        "accounting_status"
    ] == "evidence_incomplete"


def test_controlled_real_identity_eval_keeps_missing_identity_link_incomplete():
    bundle = _real_bundle(two_objects=True)
    bundle["identity_link_rows"] = []

    summary = objectstate_controlled_real_identity_eval(
        bundle,
        teacher_evidence=_teacher_evidence(bundle, collapse=False),
    )

    assert summary["status"] == "objectstate_controlled_real_identity_eval_incomplete"
    assert summary["row_counts"]["identity_rows_evidence_incomplete"] == 1
    assert summary["identity_accounting_rows"][0]["reason"] == "missing_identity_link"


def test_controlled_real_identity_eval_fails_bad_teacher_metrics():
    bundle = _real_bundle(two_objects=True)
    teacher = _teacher_evidence(bundle, collapse=True)

    summary = objectstate_controlled_real_identity_eval(
        bundle,
        teacher_evidence=teacher,
        min_identity_retrieval_at_1=0.75,
    )

    assert summary["status"] == "objectstate_controlled_real_identity_eval_fail"
    assert summary["row_counts"]["identity_fail_rows"] == 1
    assert summary["metrics"]["identity_retrieval_at_1"] == 0.0
    assert summary["evaluated_real_bundle"]["gate_accounting_rows"][0][
        "accounting_status"
    ] == "fail"


def test_controlled_real_identity_eval_rejects_oracle_leakage():
    bundle = _real_bundle(two_objects=True)
    teacher = _teacher_evidence(bundle, collapse=False)
    teacher["provenance"]["physical_identity"] = "leak"

    with pytest.raises(ValueError, match="forbidden GT leakage"):
        objectstate_controlled_real_identity_eval(bundle, teacher_evidence=teacher)


def test_controlled_real_identity_eval_validator_rejects_forged_legacy_pass():
    bundle = _real_bundle(two_objects=True)
    summary = objectstate_controlled_real_identity_eval(
        bundle,
        teacher_evidence=_teacher_evidence(bundle, collapse=False),
    )
    summary["status"] = "objectstate_controlled_real_identity_eval_pass"
    summary["row_counts"]["identity_pass_rows"] = 1
    summary["row_counts"]["identity_fail_rows"] = 0
    summary["identity_accounting_rows"][0]["eval_status"] = "pass"

    with pytest.raises(ValueError, match="cannot produce pass"):
        validate_objectstate_controlled_real_identity_eval(summary)


def test_controlled_real_identity_eval_cli_writes_artifacts(tmp_path, capsys):
    bundle = _real_bundle(two_objects=True)
    teacher = _teacher_evidence(bundle, collapse=False)
    bundle_path = tmp_path / "real-bundle.json"
    teacher_path = tmp_path / "teacher-evidence.json"
    output_dir = tmp_path / "identity-eval"
    bundle_path.write_text(json.dumps(bundle), encoding="utf-8")
    teacher_path.write_text(json.dumps(teacher), encoding="utf-8")

    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "object-state",
                "eval-controlled-real-identity",
                str(bundle_path),
                "--teacher-evidence",
                str(teacher_path),
                "--output-dir",
                str(output_dir),
                "--require-pass",
            ]
        )
    assert exc_info.value.code == 2

    stdout = capsys.readouterr().out
    summary = json.loads(
        (output_dir / "controlled-real-identity-summary.json").read_text(
            encoding="utf-8"
        )
    )
    artifact_manifest = json.loads(
        (output_dir / "controlled-real-identity-artifact-manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert f"schema={OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA}" in stdout
    assert "evaluated_identity_rows=1" in stdout
    assert summary["status"] == "objectstate_controlled_real_identity_eval_fail"
    assert (output_dir / "controlled-real-identity-accounting.csv").exists()
    assert (output_dir / "controlled-real-identity-matching.json").exists()
    assert (output_dir / "controlled-real-identity-pairwise-distances.csv").exists()
    assert artifact_manifest["artifacts"]["summary"].endswith(
        "controlled-real-identity-summary.json"
    )


def _real_bundle(*, two_objects: bool):
    bundle = objectstate_controlled_real_evidence_bundle_adapter_summary(
        _capture_manifest(two_objects=two_objects),
        source_summary_ref="outputs/captures/cup/capture-manifest.json",
    )["bundle"]
    if two_objects:
        for row in bundle["gate_accounting_rows"]:
            if row["evidence_kind"] == "identity":
                row.pop("object_id", None)
    return bundle


def _teacher_evidence(bundle, *, collapse: bool):
    assignments = []
    for pose in bundle["object_pose_rows"]:
        object_id = pose["object_id"]
        slot_id = "slot-collapse" if collapse else f"slot-{object_id}"
        assignments.append(
            {
                "object_pose_row_id": pose["row_id"],
                "slot_id": slot_id,
                "embedding": [1.0, 0.0] if slot_id.endswith("cup-001") else [0.0, 1.0],
                "confidence": 1.0,
            }
        )
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
        "assignments": assignments,
    }


def _capture_manifest(*, two_objects: bool):
    objects = [{"object_id": "cup-001", "category": "cup"}]
    if two_objects:
        objects.append({"object_id": "box-001", "category": "box"})
    frames = [
        _frame("f000", 0.0, {"cup-001": [0.0, 0.0, 0.0], "box-001": [1.0, 0.0, 0.0]}),
        _frame("f001", 0.1, {"cup-001": [-0.1, 0.0, 0.0], "box-001": [1.1, 0.0, 0.0]}),
    ]
    if not two_objects:
        for frame in frames:
            frame["objects"] = [item for item in frame["objects"] if item["object_id"] == "cup-001"]
    return {
        "schema": "objgauss-objectstate-controlled-capture-manifest-v1",
        "sample": {
            "sample_id": "controlled-tabletop-cup-001",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "push_left",
            "fps": 30.0,
            "capture_device": "cam-001",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": ["outputs/captures/cup/capture-manifest.json"],
            "license": "local-research",
        },
        "objects": objects,
        "actions": [
            {
                "action_id": "push-left-001",
                "action_type": "push_left",
                "object_id": "cup-001",
                "start_timestamp": 0.02,
                "end_timestamp": 0.08,
                "actor": "hand-001",
                "vector": [-0.1, 0.0, 0.0],
            }
        ],
        "frames": frames,
    }


def _frame(frame_id, timestamp, positions):
    return {
        "frame_id": frame_id,
        "timestamp": timestamp,
        "action_id": "push-left-001",
        "observation": {
            "rgb": f"rgb/{frame_id}.png",
            "gaussian": f"gaussians/{frame_id}.ply",
        },
        "objects": [
            {
                "object_id": object_id,
                "visible": True,
                "occlusion_fraction": 0.0,
                "pose": {
                    "position": position,
                    "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                },
            }
            for object_id, position in positions.items()
        ],
    }
