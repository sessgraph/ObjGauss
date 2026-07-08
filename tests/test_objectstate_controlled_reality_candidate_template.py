from __future__ import annotations

import csv
import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_controlled_intervention_eval import (
    validate_objectstate_controlled_intervention_candidates,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    validate_objectstate_controlled_prediction_candidates,
)
from objgauss.core.objectstate_controlled_reality_candidate_template import (
    OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
    finalize_objectstate_controlled_reality_candidate_templates,
    validate_objectstate_controlled_intervention_candidates_template,
    validate_objectstate_controlled_prediction_candidates_template,
    validate_objectstate_controlled_reality_candidate_finalize_summary,
    validate_objectstate_controlled_reality_candidate_template_summary,
    write_objectstate_controlled_reality_candidate_templates,
)


def test_controlled_reality_candidate_templates_are_draft_only(tmp_path):
    bundle_root = tmp_path / "bundle"
    output_dir = tmp_path / "candidate-templates"
    _write_capture_bundle(bundle_root)

    summary = write_objectstate_controlled_reality_candidate_templates(
        bundle_root,
        output_dir=output_dir,
        candidate_id="candidate-draft",
        candidate_source="test fixture",
        artifact_ref="outputs/controlled-real/test/objectstates.json",
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA
    assert summary["sample"]["sample_id"] == "controlled-tabletop-cup-001"
    assert summary["row_counts"]["prediction_drafts"] == 2
    assert summary["row_counts"]["intervention_drafts"] == 1
    assert summary["readiness"]["capture_prediction_stage_ready"] is True
    assert summary["readiness"]["capture_intervention_stage_ready"] is True
    assert summary["issues"] == []
    assert validate_objectstate_controlled_reality_candidate_template_summary(summary) == summary

    prediction_template = json.loads(
        (output_dir / "prediction-candidates.template.json").read_text(
            encoding="utf-8"
        )
    )
    intervention_template = json.loads(
        (output_dir / "intervention-candidates.template.json").read_text(
            encoding="utf-8"
        )
    )
    readme = (output_dir / "README.md").read_text(encoding="utf-8")

    assert prediction_template["schema"] == (
        OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA
    )
    assert intervention_template["schema"] == (
        OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA
    )
    assert prediction_template["template_status"] == "draft_not_valid_for_eval"
    assert intervention_template["template_status"] == "draft_not_valid_for_eval"
    assert prediction_template["target_eval_schema"] != prediction_template["schema"]
    assert intervention_template["target_eval_schema"] != intervention_template["schema"]
    assert prediction_template["predictions"][0]["predicted_position"].startswith("TODO")
    assert intervention_template["interventions"][0][
        "action_conditioned_position"
    ].startswith("TODO")
    assert "target_position" not in json.dumps(prediction_template)
    assert "target_position" not in json.dumps(intervention_template)
    assert "finalize_candidates" in summary["next_commands"]
    assert "finalize-controlled-reality-candidates" in summary["next_commands"][
        "finalize_candidates"
    ]
    assert "prediction-candidates.template.json" in summary["next_commands"][
        "finalize_candidates"
    ]
    assert "audit-controlled-reality-bundle-readiness" in readme
    assert "finalize-controlled-reality-candidates" in readme
    assert "controlled-reality-bundle-handoff" in readme
    assert "not valid evaluator" in readme

    validate_objectstate_controlled_prediction_candidates_template(
        prediction_template
    )
    validate_objectstate_controlled_intervention_candidates_template(
        intervention_template
    )
    with pytest.raises(ValueError, match="unsupported controlled prediction"):
        validate_objectstate_controlled_prediction_candidates(prediction_template)
    with pytest.raises(ValueError, match="unsupported controlled intervention"):
        validate_objectstate_controlled_intervention_candidates(intervention_template)


def test_object_state_init_controlled_reality_candidates_cli(tmp_path, capsys):
    bundle_root = tmp_path / "bundle"
    output_dir = tmp_path / "candidate-templates"
    summary_path = tmp_path / "candidate-template-summary.json"
    _write_capture_bundle(bundle_root)

    assert (
        main(
            [
                "object-state",
                "init-controlled-reality-candidates",
                str(bundle_root),
                "--output-dir",
                str(output_dir),
                "--candidate-id",
                "candidate-cli",
                "--candidate-source",
                "cli fixture",
                "--artifact-ref",
                "outputs/controlled-real/test/objectstates.json",
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA}" in stdout
    assert "sample_id=controlled-tabletop-cup-001" in stdout
    assert "prediction_drafts=2" in stdout
    assert "intervention_drafts=1" in stdout
    assert "finalize_candidates_command=" in stdout
    assert "audit_full_readiness_command=" in stdout
    assert "full_handoff_command=" in stdout
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA
    assert (output_dir / "prediction-candidates.template.json").is_file()
    assert (output_dir / "intervention-candidates.template.json").is_file()


def test_finalize_controlled_reality_candidate_templates_outputs_eval_json(tmp_path):
    bundle_root = tmp_path / "bundle"
    template_dir = tmp_path / "candidate-templates"
    output_dir = tmp_path / "finalized-candidates"
    _write_capture_bundle(bundle_root)
    write_objectstate_controlled_reality_candidate_templates(
        bundle_root,
        output_dir=template_dir,
        candidate_id="candidate-filled",
        candidate_source="unit-test predictor",
        artifact_ref="outputs/controlled-real/test/objectstates.json",
    )
    prediction_template_path = template_dir / "prediction-candidates.template.json"
    intervention_template_path = template_dir / "intervention-candidates.template.json"
    _fill_prediction_template(prediction_template_path)
    _fill_intervention_template(intervention_template_path)

    summary = finalize_objectstate_controlled_reality_candidate_templates(
        prediction_template_path,
        intervention_template_path,
        output_dir=output_dir,
        bundle_root=bundle_root,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA
    assert summary["sample_id"] == "controlled-tabletop-cup-001"
    assert summary["row_counts"]["prediction_candidates"] == 2
    assert summary["row_counts"]["intervention_candidates"] == 1
    assert validate_objectstate_controlled_reality_candidate_finalize_summary(summary) == summary
    prediction_candidates = json.loads(
        (output_dir / "prediction-candidates.json").read_text(encoding="utf-8")
    )
    intervention_candidates = json.loads(
        (output_dir / "intervention-candidates.json").read_text(encoding="utf-8")
    )
    assert validate_objectstate_controlled_prediction_candidates(
        prediction_candidates
    )["schema"] == "objgauss-objectstate-controlled-prediction-candidates-v1"
    assert validate_objectstate_controlled_intervention_candidates(
        intervention_candidates
    )["schema"] == "objgauss-objectstate-controlled-intervention-candidates-v1"
    assert "authoring_reference" not in json.dumps(prediction_candidates)
    assert "authoring_reference" not in json.dumps(intervention_candidates)
    assert prediction_candidates["predictions"][0]["predicted_position"] == [
        0.1,
        0.0,
        0.2,
    ]
    assert intervention_candidates["interventions"][0][
        "action_conditioned_position"
    ] == [0.2, 0.0, 0.2]
    assert str(bundle_root) in summary["next_commands"]["audit_full_readiness"]


def test_finalize_controlled_reality_candidates_rejects_todo_and_gt_leakage(tmp_path):
    bundle_root = tmp_path / "bundle"
    template_dir = tmp_path / "candidate-templates"
    output_dir = tmp_path / "finalized-candidates"
    _write_capture_bundle(bundle_root)
    write_objectstate_controlled_reality_candidate_templates(
        bundle_root,
        output_dir=template_dir,
        candidate_id="candidate-filled",
        candidate_source="unit-test predictor",
        artifact_ref="outputs/controlled-real/test/objectstates.json",
    )
    prediction_template_path = template_dir / "prediction-candidates.template.json"
    intervention_template_path = template_dir / "intervention-candidates.template.json"
    _fill_intervention_template(intervention_template_path)

    with pytest.raises(ValueError, match="predicted_position must be replaced"):
        finalize_objectstate_controlled_reality_candidate_templates(
            prediction_template_path,
            intervention_template_path,
            output_dir=output_dir,
        )

    _fill_prediction_template(prediction_template_path)
    prediction_template = _read_json(prediction_template_path)
    prediction_template["predictions"][0]["target_position"] = [0.1, 0.0, 0.2]
    _write_json(prediction_template_path, prediction_template)
    with pytest.raises(ValueError, match="forbidden GT leakage fields"):
        finalize_objectstate_controlled_reality_candidate_templates(
            prediction_template_path,
            intervention_template_path,
            output_dir=output_dir,
        )


def test_object_state_finalize_controlled_reality_candidates_cli(tmp_path, capsys):
    bundle_root = tmp_path / "bundle"
    template_dir = tmp_path / "candidate-templates"
    output_dir = tmp_path / "finalized-candidates"
    summary_path = tmp_path / "finalize-summary.json"
    _write_capture_bundle(bundle_root)
    write_objectstate_controlled_reality_candidate_templates(
        bundle_root,
        output_dir=template_dir,
        candidate_id="candidate-cli-filled",
        candidate_source="cli predictor",
        artifact_ref="outputs/controlled-real/test/objectstates.json",
    )
    prediction_template_path = template_dir / "prediction-candidates.template.json"
    intervention_template_path = template_dir / "intervention-candidates.template.json"
    _fill_prediction_template(prediction_template_path)
    _fill_intervention_template(intervention_template_path)

    assert (
        main(
            [
                "object-state",
                "finalize-controlled-reality-candidates",
                str(prediction_template_path),
                str(intervention_template_path),
                "--output-dir",
                str(output_dir),
                "--bundle-root",
                str(bundle_root),
                "--summary-output",
                str(summary_path),
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA}" in stdout
    assert "prediction_candidate_count=2" in stdout
    assert "intervention_candidate_count=1" in stdout
    assert "audit_full_readiness_command=" in stdout
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA
    assert (output_dir / "prediction-candidates.json").is_file()
    assert (output_dir / "intervention-candidates.json").is_file()


def test_controlled_reality_candidate_templates_refuse_overwrite(tmp_path):
    bundle_root = tmp_path / "bundle"
    output_dir = tmp_path / "candidate-templates"
    _write_capture_bundle(bundle_root)

    write_objectstate_controlled_reality_candidate_templates(
        bundle_root,
        output_dir=output_dir,
    )
    with pytest.raises(FileExistsError, match="refuses to overwrite"):
        write_objectstate_controlled_reality_candidate_templates(
            bundle_root,
            output_dir=output_dir,
        )

    forced = write_objectstate_controlled_reality_candidate_templates(
        bundle_root,
        output_dir=output_dir,
        force=True,
    )
    assert forced["row_counts"]["prediction_drafts"] == 2


def _write_capture_bundle(root):
    root.mkdir(parents=True)
    (root / "sample.json").write_text(
        json.dumps(
            {
                "sample_id": "controlled-tabletop-cup-001",
                "source_kind": "controlled_real",
                "object_category": "cup",
                "scenario": "prediction_intervention_template_fixture",
                "fps": 30.0,
                "capture_device": "fixture-camera",
                "observation_modalities": ["rgb", "gaussian"],
                "artifact_refs": ["capture-manifest.json", "rgb/", "gaussians/"],
                "license": "local controlled capture",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        root / "objects.csv",
        [
            "object_id",
            "category",
            "instance_label",
            "dimension_x_m",
            "dimension_y_m",
            "dimension_z_m",
        ],
        [
            {
                "object_id": "cup-001",
                "category": "cup",
                "instance_label": "blue cup",
            }
        ],
    )
    _write_csv(
        root / "frames.csv",
        [
            "frame_id",
            "timestamp",
            "rgb",
            "gaussian",
            "action_id",
            "view_id",
            "lighting_id",
            "camera_x",
            "camera_y",
            "camera_z",
            "camera_qx",
            "camera_qy",
            "camera_qz",
            "camera_qw",
        ],
        [
            {
                "frame_id": "frame-000000",
                "timestamp": "0.0",
                "rgb": "rgb/frame-000000.png",
                "gaussian": "gaussians/frame-000000.ply",
                "view_id": "front",
                "lighting_id": "bright",
                "camera_x": "0.0",
                "camera_y": "0.0",
                "camera_z": "1.0",
                "camera_qx": "0.0",
                "camera_qy": "0.0",
                "camera_qz": "0.0",
                "camera_qw": "1.0",
            },
            {
                "frame_id": "frame-000001",
                "timestamp": "1.0",
                "rgb": "rgb/frame-000001.png",
                "gaussian": "gaussians/frame-000001.ply",
                "action_id": "push-right-001",
                "view_id": "front",
                "lighting_id": "bright",
                "camera_x": "0.0",
                "camera_y": "0.0",
                "camera_z": "1.0",
                "camera_qx": "0.0",
                "camera_qy": "0.0",
                "camera_qz": "0.0",
                "camera_qw": "1.0",
            },
            {
                "frame_id": "frame-000002",
                "timestamp": "2.0",
                "rgb": "rgb/frame-000002.png",
                "gaussian": "gaussians/frame-000002.ply",
                "action_id": "push-right-001",
                "view_id": "side",
                "lighting_id": "dim",
                "camera_x": "0.2",
                "camera_y": "0.0",
                "camera_z": "1.0",
                "camera_qx": "0.0",
                "camera_qy": "0.0",
                "camera_qz": "0.0",
                "camera_qw": "1.0",
            },
        ],
    )
    _write_csv(
        root / "annotations.csv",
        [
            "frame_id",
            "object_id",
            "visible",
            "occlusion_fraction",
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
        ],
        [
            _annotation("frame-000000", 0.0, 0.0, 0.2),
            _annotation("frame-000001", 0.1, 0.0, 0.2),
            _annotation("frame-000002", 0.2, 0.0, 0.2),
        ],
    )
    _write_csv(
        root / "actions.csv",
        [
            "action_id",
            "action_type",
            "object_id",
            "start_timestamp",
            "end_timestamp",
            "actor",
            "target_object_id",
            "vector_x",
            "vector_y",
            "vector_z",
        ],
        [
            {
                "action_id": "push-right-001",
                "action_type": "push_right",
                "object_id": "cup-001",
                "start_timestamp": "1.0",
                "end_timestamp": "2.0",
                "actor": "human",
                "vector_x": "1.0",
                "vector_y": "0.0",
                "vector_z": "0.0",
            }
        ],
    )


def _fill_prediction_template(path):
    payload = _read_json(path)
    rows = payload["predictions"]
    rows[0]["predicted_position"] = [0.1, 0.0, 0.2]
    rows[0]["history_baseline_position"] = [0.0, 0.0, 0.2]
    rows[0]["confidence"] = 0.9
    rows[1]["predicted_position"] = [0.2, 0.0, 0.2]
    rows[1]["history_baseline_position"] = [0.1, 0.0, 0.2]
    rows[1]["confidence"] = 0.8
    _write_json(path, payload)


def _fill_intervention_template(path):
    payload = _read_json(path)
    row = payload["interventions"][0]
    row["action_conditioned_position"] = [0.2, 0.0, 0.2]
    row["no_action_baseline_position"] = [0.1, 0.0, 0.2]
    row["confidence"] = 0.9
    _write_json(path, payload)


def _annotation(frame_id, x, y, z):
    return {
        "frame_id": frame_id,
        "object_id": "cup-001",
        "visible": "true",
        "occlusion_fraction": "0.0",
        "x": str(x),
        "y": str(y),
        "z": str(z),
        "qx": "0.0",
        "qy": "0.0",
        "qz": "0.0",
        "qw": "1.0",
    }


def _write_csv(path, header, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
