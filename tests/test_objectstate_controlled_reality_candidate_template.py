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
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
    validate_objectstate_controlled_intervention_candidates_template,
    validate_objectstate_controlled_prediction_candidates_template,
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
    assert "audit-controlled-reality-bundle-readiness" in readme
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
    assert "audit_full_readiness_command=" in stdout
    assert "full_handoff_command=" in stdout
    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA
    assert (output_dir / "prediction-candidates.template.json").is_file()
    assert (output_dir / "intervention-candidates.template.json").is_file()


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
