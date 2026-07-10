from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.evaluation.objectstate_controlled_intervention_eval import (
    OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
)
from objgauss.evaluation.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
)
from objgauss.pipelines.objectstate_controlled_reality_bundle_readiness import (
    OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_READINESS_SCHEMA,
    objectstate_controlled_reality_bundle_readiness,
    validate_objectstate_controlled_reality_bundle_readiness_summary,
)
from objgauss.pipelines.trainable_artifact import TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA

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


def test_controlled_reality_bundle_readiness_reports_ready_inputs(tmp_path):
    _write_bundle(tmp_path, include_frame_files=True)
    artifact_path = tmp_path / "objectstates.json"
    predictions_path = tmp_path / "prediction-candidates.json"
    interventions_path = tmp_path / "intervention-candidates.json"
    artifact_path.write_text(json.dumps(_trainable_artifact()), encoding="utf-8")
    predictions_path.write_text(json.dumps(_prediction_candidates()), encoding="utf-8")
    interventions_path.write_text(
        json.dumps(_intervention_candidates()),
        encoding="utf-8",
    )

    summary = objectstate_controlled_reality_bundle_readiness(
        tmp_path,
        artifact_path,
        predictions_path,
        interventions_path,
        max_centroid_distance=0.05,
        hash_files=True,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_READINESS_SCHEMA
    assert summary["status"] == "objectstate_controlled_reality_bundle_readiness_ready"
    assert summary["readiness"]["full_reality_handoff_ready"] is True
    assert summary["readiness"]["intervention_action_gt_ready"] is True
    assert summary["trainable_artifact"]["identity_prediction_count"] == 6
    assert summary["prediction_candidates"]["record_count"] == 1
    assert summary["intervention_candidates"]["record_count"] == 1
    assert summary["hard_blockers"] == []
    assert validate_objectstate_controlled_reality_bundle_readiness_summary(
        summary
    ) == summary


def test_controlled_reality_bundle_readiness_blocks_sample_mismatch(tmp_path):
    _write_bundle(tmp_path, include_frame_files=True)
    artifact_path = tmp_path / "objectstates.json"
    predictions_path = tmp_path / "prediction-candidates.json"
    interventions_path = tmp_path / "intervention-candidates.json"
    predictions = _prediction_candidates()
    predictions["sample_id"] = "wrong-sample"
    artifact_path.write_text(json.dumps(_trainable_artifact()), encoding="utf-8")
    predictions_path.write_text(json.dumps(predictions), encoding="utf-8")
    interventions_path.write_text(
        json.dumps(_intervention_candidates()),
        encoding="utf-8",
    )

    summary = objectstate_controlled_reality_bundle_readiness(
        tmp_path,
        artifact_path,
        predictions_path,
        interventions_path,
        max_centroid_distance=0.05,
    )

    assert summary["status"] == "objectstate_controlled_reality_bundle_readiness_blocked"
    assert summary["readiness"]["prediction_candidates_schema_ready"] is True
    assert summary["readiness"]["prediction_candidates_binding_ready"] is False
    assert "prediction_candidates_binding_ready" in summary["hard_blockers"]
    assert any(
        "candidate sample_id does not match" in item
        for item in summary["prediction_candidates"]["binding_issues"]
    )


def test_controlled_reality_bundle_readiness_blocks_weak_intervention_action_gt(
    tmp_path,
):
    _write_bundle(tmp_path, include_frame_files=True)
    rows = (tmp_path / "actions.csv").read_text(encoding="utf-8").splitlines()
    rows[1] = (
        "push-right-001,push_right,cup-001,0.033333,0.066667,"
        "scripted-hand,,0.0,0.0,0.0"
    )
    (tmp_path / "actions.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    artifact_path = tmp_path / "objectstates.json"
    predictions_path = tmp_path / "prediction-candidates.json"
    interventions_path = tmp_path / "intervention-candidates.json"
    artifact_path.write_text(json.dumps(_trainable_artifact()), encoding="utf-8")
    predictions_path.write_text(json.dumps(_prediction_candidates()), encoding="utf-8")
    interventions_path.write_text(
        json.dumps(_intervention_candidates()),
        encoding="utf-8",
    )

    summary = objectstate_controlled_reality_bundle_readiness(
        tmp_path,
        artifact_path,
        predictions_path,
        interventions_path,
        max_centroid_distance=0.05,
    )

    assert summary["status"] == "objectstate_controlled_reality_bundle_readiness_blocked"
    assert summary["readiness"]["intervention_action_gt_ready"] is False
    assert summary["readiness"]["full_reality_handoff_ready"] is False
    assert any("non-zero vector" in item for item in summary["hard_blockers"])


def test_object_state_audit_controlled_reality_bundle_readiness_cli_writes_summary(
    tmp_path,
    capsys,
):
    _write_bundle(tmp_path, include_frame_files=True)
    artifact_path = tmp_path / "objectstates.json"
    predictions_path = tmp_path / "prediction-candidates.json"
    interventions_path = tmp_path / "intervention-candidates.json"
    summary_path = tmp_path / "reality-readiness.json"
    artifact_path.write_text(json.dumps(_trainable_artifact()), encoding="utf-8")
    predictions_path.write_text(json.dumps(_prediction_candidates()), encoding="utf-8")
    interventions_path.write_text(
        json.dumps(_intervention_candidates()),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "object-state",
                "audit-controlled-reality-bundle-readiness",
                str(tmp_path),
                str(artifact_path),
                str(predictions_path),
                str(interventions_path),
                "--summary-output",
                str(summary_path),
                "--max-centroid-distance",
                "0.05",
                "--hash-files",
                "--require-ready",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_READINESS_SCHEMA}" in stdout
    assert "full_reality_handoff_ready=true" in stdout
    assert "intervention_action_gt_ready=true" in stdout
    assert "hard_blocker_count=0" in stdout
    assert summary["readiness"]["full_reality_handoff_ready"] is True


def _prediction_candidates():
    return {
        "schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
        "sample_id": "controlled-tabletop-cup-box-readiness-001",
        "candidate": {
            "candidate_id": "candidate-objectstate-prediction-v0",
            "source": "fixture future pose predictions",
            "artifact_refs": [
                "outputs/controlled-real/cup-box-readiness-001/predictions.json"
            ],
        },
        "predictions": [
            {
                "source_frame_id": "frame-000001",
                "target_frame_id": "frame-000002",
                "object_id": "cup-001",
                "predicted_position": [0.12, 0.2, 0.3],
                "history_baseline_position": [0.11, 0.2, 0.3],
            }
        ],
    }


def _intervention_candidates():
    return {
        "schema": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
        "sample_id": "controlled-tabletop-cup-box-readiness-001",
        "candidate": {
            "candidate_id": "candidate-objectstate-intervention-v0",
            "source": "fixture action-conditioned future pose predictions",
            "artifact_refs": [
                "outputs/controlled-real/cup-box-readiness-001/interventions.json"
            ],
        },
        "interventions": [
            {
                "source_frame_id": "frame-000001",
                "target_frame_id": "frame-000002",
                "object_id": "cup-001",
                "action_id": "push-right-001",
                "action_conditioned_position": [0.12, 0.2, 0.3],
                "no_action_baseline_position": [0.11, 0.2, 0.3],
            }
        ],
    }


def _write_bundle(root, *, include_frame_files: bool) -> None:
    (root / "sample.json").write_text(
        json.dumps(
            {
                "sample_id": "controlled-tabletop-cup-box-readiness-001",
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
                "cup-001,cup,blue cup,0.08,0.08,0.12",
                "box-001,box,red box,0.12,0.12,0.12",
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
            "input": "outputs/controlled-real/cup-box-readiness-001/objectstates.json",
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
