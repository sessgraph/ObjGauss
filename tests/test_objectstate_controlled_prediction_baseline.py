from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.core.objectstate_controlled_prediction_baseline import (
    OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA,
    validate_objectstate_controlled_prediction_baseline_summary,
    write_objectstate_controlled_prediction_baseline_candidates,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    evaluate_objectstate_controlled_prediction_candidates,
    validate_objectstate_controlled_prediction_candidates,
)
from objgauss.core.objectstate_controlled_reality_candidate_template import (
    write_objectstate_controlled_reality_candidate_templates_from_manifest,
)


def test_controlled_prediction_baseline_generates_eval_ready_candidates(tmp_path):
    capture_path, template_path, output_dir = _write_prediction_template(tmp_path)

    summary = write_objectstate_controlled_prediction_baseline_candidates(
        capture_path,
        template_path,
        output_dir=output_dir,
        policy="constant_velocity",
        candidate_id="fixture-constant-velocity",
        candidate_source="unit-test controlled prediction baseline",
        artifact_ref="outputs/controlled-real/baseline/objectstates.json",
        confidence=0.8,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA
    assert validate_objectstate_controlled_prediction_baseline_summary(summary) == summary
    assert summary["sample_id"] == "controlled-baseline-cup-box-001"
    assert summary["row_counts"]["prediction_candidates"] == 4
    assert summary["row_counts"]["constant_velocity_rows"] == 2
    assert summary["row_counts"]["hold_rows"] == 2
    assert summary["policy"]["uses_target_pose_values"] is False
    assert summary["claim_policy"]["does_not_read_target_pose_values"] is True
    assert (output_dir / "prediction-candidates.baseline-filled.template.json").is_file()
    assert (output_dir / "prediction-candidates.json").is_file()
    assert (output_dir / "prediction-finalize-summary.json").is_file()

    candidates = _read_json(output_dir / "prediction-candidates.json")
    validated_candidates = validate_objectstate_controlled_prediction_candidates(
        candidates
    )
    assert validated_candidates["schema"] == candidates["schema"]
    assert len(validated_candidates["predictions"]) == len(candidates["predictions"])
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
        _capture_manifest(),
        candidates,
    )
    assert prediction_eval["status"] == "objectstate_controlled_prediction_eval_pass"
    assert prediction_eval["metrics"]["prediction_count"] == 4
    assert prediction_eval["metrics"]["state_ade"] == pytest.approx(0.005)
    assert prediction_eval["metrics"]["history_ade"] == pytest.approx(0.01)
    assert prediction_eval["metrics"]["prediction_gap_vs_history_model"] == pytest.approx(
        -0.005
    )


def test_controlled_prediction_baseline_does_not_use_target_pose_values(tmp_path):
    capture_path, template_path, _output_dir = _write_prediction_template(tmp_path)
    mutated_manifest = _capture_manifest()
    mutated_manifest["frames"][2]["objects"][0]["pose"]["position"] = [9.0, 9.0, 9.0]
    mutated_manifest["frames"][2]["objects"][1]["pose"]["position"] = [-9.0, -9.0, -9.0]
    mutated_capture_path = tmp_path / "capture-mutated-target.json"
    _write_json(mutated_capture_path, mutated_manifest)

    original_dir = tmp_path / "original-baseline"
    mutated_dir = tmp_path / "mutated-baseline"
    original = write_objectstate_controlled_prediction_baseline_candidates(
        capture_path,
        template_path,
        output_dir=original_dir,
        candidate_id="fixture-no-leakage",
        artifact_ref="outputs/controlled-real/baseline/objectstates.json",
    )
    mutated = write_objectstate_controlled_prediction_baseline_candidates(
        mutated_capture_path,
        template_path,
        output_dir=mutated_dir,
        candidate_id="fixture-no-leakage",
        artifact_ref="outputs/controlled-real/baseline/objectstates.json",
    )

    assert original["claim_policy"]["uses_target_timestamp_only"] is True
    assert mutated["claim_policy"]["uses_target_timestamp_only"] is True
    original_rows = _prediction_rows_by_key(original_dir / "prediction-candidates.json")
    mutated_rows = _prediction_rows_by_key(mutated_dir / "prediction-candidates.json")
    assert original_rows.keys() == mutated_rows.keys()
    for key in original_rows:
        assert mutated_rows[key]["predicted_position"] == pytest.approx(
            original_rows[key]["predicted_position"]
        )
        assert mutated_rows[key]["history_baseline_position"] == pytest.approx(
            original_rows[key]["history_baseline_position"]
        )


def test_object_state_generate_controlled_prediction_baseline_candidates_cli(
    tmp_path,
    capsys,
):
    capture_path, template_path, output_dir = _write_prediction_template(tmp_path)
    summary_path = tmp_path / "prediction-baseline-summary.json"

    assert (
        main(
            [
                "object-state",
                "generate-controlled-prediction-baseline-candidates",
                str(capture_path),
                str(template_path),
                "--output-dir",
                str(output_dir),
                "--candidate-id",
                "cli-constant-velocity",
                "--candidate-source",
                "cli unit-test baseline",
                "--artifact-ref",
                "outputs/controlled-real/baseline/objectstates.json",
                "--confidence",
                "0.7",
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = _read_json(summary_path)

    assert f"schema={OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA}" in stdout
    assert "sample_id=controlled-baseline-cup-box-001" in stdout
    assert "policy=constant_velocity" in stdout
    assert "prediction_candidate_count=4" in stdout
    assert "constant_velocity_rows=2" in stdout
    assert "hold_rows=2" in stdout
    assert "filled_prediction_template=" in stdout
    assert "prediction_candidates=" in stdout
    assert "prediction_finalize_summary=" in stdout
    assert "eval_prediction_command=" in stdout
    assert "audit_prediction_evidence_package_command=" in stdout
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA
    assert summary["candidate"]["candidate_id"] == "cli-constant-velocity"
    assert (output_dir / "prediction-candidates.json").is_file()


def _write_prediction_template(tmp_path):
    capture_path = tmp_path / "capture-manifest.json"
    output_dir = tmp_path / "reality-candidates"
    _write_json(capture_path, _capture_manifest())
    write_objectstate_controlled_reality_candidate_templates_from_manifest(
        capture_path,
        output_dir=output_dir,
        candidate_id="template-draft",
        candidate_source="unit-test draft template",
        artifact_ref="outputs/controlled-real/baseline/objectstates.json",
    )
    return capture_path, output_dir / "prediction-candidates.template.json", output_dir


def _prediction_rows_by_key(path):
    payload = _read_json(path)
    return {
        (row["source_frame_id"], row["target_frame_id"], row["object_id"]): row
        for row in payload["predictions"]
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
            "sample_id": "controlled-baseline-cup-box-001",
            "source_kind": "controlled_real",
            "object_category": "cup_box",
            "scenario": "future_pose_prediction_baseline",
            "fps": 2.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": [
                "outputs/controlled-real/baseline/capture.json",
                "outputs/controlled-real/baseline/rgb/",
                "outputs/controlled-real/baseline/gaussians/",
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


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
