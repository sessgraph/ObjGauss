from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.objectstate_controlled_reality_bundle_handoff import (
    objectstate_controlled_reality_bundle_handoff,
)
from objgauss.core.objectstate_controlled_reality_bundle_readiness import (
    objectstate_controlled_reality_bundle_readiness,
)
from objgauss.core.objectstate_controlled_reality_candidate_template import (
    finalize_objectstate_controlled_reality_candidate_templates,
    write_objectstate_controlled_reality_candidate_templates,
)
from objgauss.core.objectstate_controlled_reality_evidence_package import (
    OBJECTSTATE_CONTROLLED_REALITY_EVIDENCE_PACKAGE_SCHEMA,
    objectstate_controlled_reality_evidence_package,
    validate_objectstate_controlled_reality_evidence_package_summary,
)
from objgauss.core.trainable_artifact import TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA

PNG_BYTES = b"\x89PNG\r\n\x1a\n"
PLY_BYTES = (
    b"ply\n"
    b"format ascii 1.0\n"
    b"element vertex 1\n"
    b"property float x\n"
    b"property float y\n"
    b"property float z\n"
    b"end_header\n"
    b"0 0 0\n"
)


def test_controlled_reality_evidence_package_is_reviewable(tmp_path):
    _write_reviewable_package(tmp_path)

    summary = objectstate_controlled_reality_evidence_package(tmp_path)

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REALITY_EVIDENCE_PACKAGE_SCHEMA
    assert (
        summary["status"]
        == "objectstate_controlled_reality_evidence_package_reviewable"
    )
    assert summary["sample_id"] == "controlled-tabletop-cup-box-reality-001"
    assert all(summary["reviewability_gates"].values())
    assert summary["row_accounting"]["pass_row_count"] == 3
    assert summary["row_accounting"]["blocked_row_count"] == 0
    assert summary["row_accounting"][
        "identity_prediction_intervention_rows_present"
    ] is True
    assert summary["output_consistency"]["matches"] is True
    assert summary["issues"] == []
    assert (
        validate_objectstate_controlled_reality_evidence_package_summary(summary)
        == summary
    )


def test_controlled_reality_evidence_package_reports_missing_outputs(tmp_path):
    paths = _write_reviewable_package(tmp_path)
    (paths["handoff_dir"] / "prediction-eval-summary.json").unlink()

    summary = objectstate_controlled_reality_evidence_package(tmp_path)

    assert (
        summary["status"]
        == "objectstate_controlled_reality_evidence_package_incomplete"
    )
    assert summary["reviewability_gates"]["required_files_present"] is False
    assert (
        summary["reviewability_gates"]["standalone_outputs_match_handoff_summary"]
        is False
    )
    assert any("prediction_eval_summary" in issue for issue in summary["issues"])


def test_object_state_audit_controlled_reality_evidence_package_cli(
    tmp_path,
    capsys,
):
    _write_reviewable_package(tmp_path)
    summary_path = tmp_path / "evidence-package-summary.json"

    assert (
        main(
            [
                "object-state",
                "audit-controlled-reality-evidence-package",
                str(tmp_path),
                "--summary-output",
                str(summary_path),
                "--require-reviewable",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_REALITY_EVIDENCE_PACKAGE_SCHEMA}" in stdout
    assert "reviewable=true" in stdout
    assert "pass_rows=3" in stdout
    assert "blocked_rows=0" in stdout
    assert "issue_count=0" in stdout
    assert (
        summary["status"]
        == "objectstate_controlled_reality_evidence_package_reviewable"
    )


def _write_reviewable_package(root):
    _write_bundle(root, include_frame_files=True)
    artifact = _trainable_artifact()
    artifact_path = root / "objectstates.json"
    artifact_path.write_text(json.dumps(artifact), encoding="utf-8")

    candidate_dir = root / "reality-candidates"
    template_summary = write_objectstate_controlled_reality_candidate_templates(
        root,
        output_dir=candidate_dir,
        candidate_id="candidate-objectstate-phase1-v0",
        candidate_source="fixture objectstate predictor",
        artifact_ref=str(artifact_path),
    )
    _write_json(candidate_dir / "template-summary.json", template_summary)
    _fill_prediction_template(candidate_dir / "prediction-candidates.template.json")
    _fill_intervention_template(
        candidate_dir / "intervention-candidates.template.json"
    )
    finalize_summary = finalize_objectstate_controlled_reality_candidate_templates(
        candidate_dir / "prediction-candidates.template.json",
        candidate_dir / "intervention-candidates.template.json",
        output_dir=candidate_dir,
        bundle_root=root,
    )
    _write_json(candidate_dir / "finalize-summary.json", finalize_summary)

    prediction_candidates = _read_json(candidate_dir / "prediction-candidates.json")
    intervention_candidates = _read_json(candidate_dir / "intervention-candidates.json")
    readiness = objectstate_controlled_reality_bundle_readiness(
        root,
        artifact_path,
        candidate_dir / "prediction-candidates.json",
        candidate_dir / "intervention-candidates.json",
        max_centroid_distance=0.05,
        hash_files=True,
    )
    _write_json(candidate_dir / "full-readiness-summary.json", readiness)

    handoff = objectstate_controlled_reality_bundle_handoff(
        root,
        artifact,
        prediction_candidates,
        intervention_candidates,
        candidate_id="candidate-objectstate-phase1-v0",
        max_centroid_distance=0.05,
        candidate_artifact_path=artifact_path,
        hash_files=True,
        hash_candidate_artifact=True,
    )
    handoff_dir = root / "reality-handoff"
    handoff_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        handoff_dir / "reality-bundle-handoff-summary.json",
        handoff,
    )
    _write_json(handoff_dir / "prediction-eval-summary.json", handoff["prediction_eval"])
    _write_json(
        handoff_dir / "intervention-eval-summary.json",
        handoff["intervention_eval"],
    )
    _write_json(
        handoff_dir / "controlled-real.json",
        handoff["controlled_real_manifest"],
    )
    _write_json(
        handoff_dir / "controlled-real-summary.json",
        handoff["controlled_real_summary"],
    )
    (handoff_dir / "blocked-rows.md").write_text(
        handoff["controlled_real_summary"]["blocked_rows_markdown"],
        encoding="utf-8",
    )
    return {"candidate_dir": candidate_dir, "handoff_dir": handoff_dir}


def _fill_prediction_template(path):
    payload = _read_json(path)
    target_positions = {
        ("frame-000001", "cup-001"): [0.11, 0.2, 0.3],
        ("frame-000002", "cup-001"): [0.12, 0.2, 0.3],
        ("frame-000001", "box-001"): [0.41, 0.2, 0.3],
        ("frame-000002", "box-001"): [0.42, 0.2, 0.3],
    }
    source_positions = {
        ("frame-000000", "cup-001"): [0.10, 0.2, 0.3],
        ("frame-000001", "cup-001"): [0.11, 0.2, 0.3],
        ("frame-000000", "box-001"): [0.40, 0.2, 0.3],
        ("frame-000001", "box-001"): [0.41, 0.2, 0.3],
    }
    for row in payload["predictions"]:
        object_id = row["object_id"]
        row["predicted_position"] = target_positions[
            (row["target_frame_id"], object_id)
        ]
        row["history_baseline_position"] = source_positions[
            (row["source_frame_id"], object_id)
        ]
        row["confidence"] = 0.95
    _write_json(path, payload)


def _fill_intervention_template(path):
    payload = _read_json(path)
    for row in payload["interventions"]:
        row["action_conditioned_position"] = [0.12, 0.2, 0.3]
        row["no_action_baseline_position"] = [0.11, 0.2, 0.3]
        row["confidence"] = 0.95
    _write_json(path, payload)


def _write_bundle(root, *, include_frame_files: bool) -> None:
    (root / "sample.json").write_text(
        json.dumps(
            {
                "sample_id": "controlled-tabletop-cup-box-reality-001",
                "source_kind": "controlled_real",
                "object_category": "cup_box",
                "scenario": "cross_view_occlusion_reappearance_push_right",
                "fps": 30.0,
                "capture_device": "fixture-camera",
                "observation_modalities": ["rgb", "gaussian"],
                "artifact_refs": [
                    "capture-manifest.json",
                    "rgb/",
                    "gaussians/",
                ],
                "license": "local controlled capture; not public release",
            }
        ),
        encoding="utf-8",
    )
    (root / "objects.csv").write_text(
        "\n".join(
            (
                "object_id,category,instance_label,dimension_x_m,dimension_y_m,dimension_z_m",
                "cup-001,cup,blue cup,0.08,0.08,0.10",
                "box-001,box,red box,0.12,0.12,0.08",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "frames.csv").write_text(
        "\n".join(
            (
                "frame_id,timestamp,rgb,gaussian,action_id,view_id,lighting_id,camera_x,camera_y,camera_z,camera_qx,camera_qy,camera_qz,camera_qw",
                "frame-000000,0.000000,rgb/000000.png,gaussians/000000.ply,,front,bright,0.00,0.0,0.0,0,0,0,1",
                "frame-000001,0.033333,rgb/000001.png,gaussians/000001.ply,push-right-001,front,dim,0.02,0.0,0.0,0,0,0,1",
                "frame-000002,0.066667,rgb/000002.png,gaussians/000002.ply,,right,dim,0.04,0.0,0.0,0,0,0,1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "annotations.csv").write_text(
        "\n".join(
            (
                "frame_id,object_id,visible,occlusion_fraction,x,y,z,qx,qy,qz,qw",
                "frame-000000,cup-001,true,0.0,0.10,0.20,0.30,0,0,0,1",
                "frame-000000,box-001,true,0.0,0.40,0.20,0.30,0,0,0,1",
                "frame-000001,cup-001,false,0.8,0.11,0.20,0.30,0,0,0,1",
                "frame-000001,box-001,true,0.0,0.41,0.20,0.30,0,0,0,1",
                "frame-000002,cup-001,true,0.0,0.12,0.20,0.30,0,0,0,1",
                "frame-000002,box-001,true,0.0,0.42,0.20,0.30,0,0,0,1",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "actions.csv").write_text(
        "\n".join(
            (
                "action_id,action_type,object_id,start_timestamp,end_timestamp,actor,target_object_id,vector_x,vector_y,vector_z",
                "push-right-001,push_right,cup-001,0.033333,0.066667,scripted-hand,,0.01,0.0,0.0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    if include_frame_files:
        (root / "rgb").mkdir(parents=True, exist_ok=True)
        (root / "gaussians").mkdir(parents=True, exist_ok=True)
        for index in range(3):
            (root / "rgb" / f"{index:06d}.png").write_bytes(PNG_BYTES)
            (root / "gaussians" / f"{index:06d}.ply").write_bytes(PLY_BYTES)


def _trainable_artifact():
    object_states = []
    assignments = []
    for frame_index in range(3):
        cup_x = 0.1 + 0.01 * frame_index
        box_x = 0.4 + 0.01 * frame_index
        object_states.append(
            {
                "frame_index": frame_index,
                "states": [
                    _state(0, [cup_x, 0.2, 0.3]),
                    _state(1, [box_x, 0.2, 0.3]),
                ],
                "derived_object_ids": [0, 1],
            }
        )
        assignments.append(
            {
                "frame_index": frame_index,
                "shape": [2, 2],
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
            }
        )
    return {
        "schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "kind": "trainable_kernel_mvp_model",
        "label": "fixture-trainable-objectstates",
        "source": {
            "input": "outputs/controlled-real/cup-box-reality-001/objectstates.json",
            "sample": None,
        },
        "training": {
            "schema": "objgauss-v1-trainable-kernel-mvp-v1",
            "frame_count": len(object_states),
        },
        "renderer_api": {},
        "learned_parameters": {"decoder_colors": []},
        "assignments": assignments,
        "object_states": object_states,
        "identity_evidence": {
            "reconstruction_noise_robustness": 1.0,
            "reconstruction_noise_variant_count": 2,
            "source": "fixture repeated Gaussian reconstruction noise variants",
        },
        "artifact_policy": {
            "git_policy": "do_not_commit_training_outputs_by_default",
        },
    }


def _state(state_id: int, centroid: list[float]):
    return {
        "id": state_id,
        "slot_mass": 1.0,
        "confidence": 0.92,
        "mass_fraction": 0.5,
        "assignment_entropy": 0.0,
        "normalized_assignment_entropy": 0.0,
        "centroid": centroid,
        "bbox": [
            [centroid[0] - 0.01, centroid[1] - 0.01, centroid[2] - 0.01],
            [centroid[0] + 0.01, centroid[1] + 0.01, centroid[2] + 0.01],
        ],
        "feature": [centroid[0], centroid[1], centroid[2]],
        "status": "active",
        "diagnostics": [],
    }


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
