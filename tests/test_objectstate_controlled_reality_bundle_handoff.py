from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.core.objectstate_controlled_intervention_eval import (
    OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
)
from objgauss.core.objectstate_controlled_reality_bundle_handoff import (
    OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA,
    objectstate_controlled_reality_bundle_handoff,
    validate_objectstate_controlled_reality_bundle_handoff_summary,
)
from objgauss.core.objectstate_public_interaction_reality_rows import (
    OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA,
    objectstate_public_interaction_reality_rows_summary,
    validate_objectstate_public_interaction_reality_rows_summary,
)
from objgauss.core.objectstate_reality_row_ledger import (
    objectstate_reality_row_ledger,
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


def test_controlled_reality_bundle_handoff_merges_three_pass_rows(tmp_path):
    _write_bundle(tmp_path, include_frame_files=True)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)

    summary = objectstate_controlled_reality_bundle_handoff(
        tmp_path,
        artifact,
        _prediction_candidates(),
        _intervention_candidates(),
        candidate_id="stable-objectstate-slots",
        max_centroid_distance=0.05,
        candidate_artifact_path=artifact_path,
        hash_files=True,
        hash_candidate_artifact=True,
    )

    assert summary["schema"] == OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA
    assert summary["status"] == "objectstate_controlled_reality_bundle_handoff_pass"
    assert summary["handoff_gates"] == {
        "capture_bundle_acceptance_pass": True,
        "identity_handoff_pass": True,
        "prediction_eval_pass": True,
        "intervention_eval_pass": True,
        "full_reality_gate_pass": True,
    }
    assert summary["identity_eval"]["status"] == "objectstate_controlled_identity_eval_pass"
    assert (
        summary["prediction_eval"]["status"]
        == "objectstate_controlled_prediction_eval_pass"
    )
    assert (
        summary["intervention_eval"]["status"]
        == "objectstate_controlled_intervention_eval_pass"
    )
    assert summary["controlled_real_summary"]["gate"]["status"] == (
        "objectstate_reality_gate_pass"
    )
    assert summary["controlled_real_summary"]["pass_row_count"] == 3
    assert summary["controlled_real_summary"]["blocked_row_count"] == 0
    assert [
        row["status"] for row in summary["controlled_real_manifest"]["evidence_rows"]
    ] == ["pass", "pass", "pass"]
    assert validate_objectstate_controlled_reality_bundle_handoff_summary(
        summary
    ) == summary


def test_controlled_reality_bundle_handoff_fails_wrong_direction_intervention(
    tmp_path,
):
    _write_bundle(tmp_path, include_frame_files=True)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)
    interventions = _intervention_candidates()
    interventions["interventions"][0]["action_conditioned_position"] = [0.10, 0.2, 0.3]

    summary = objectstate_controlled_reality_bundle_handoff(
        tmp_path,
        artifact,
        _prediction_candidates(),
        interventions,
        candidate_id="stable-objectstate-slots",
        max_centroid_distance=0.05,
        candidate_artifact_path=artifact_path,
    )

    assert summary["status"] == "objectstate_controlled_reality_bundle_handoff_fail"
    assert summary["handoff_gates"]["intervention_eval_pass"] is False
    assert summary["handoff_gates"]["full_reality_gate_pass"] is False
    assert (
        summary["intervention_eval"]["pass_gates"][
            "wrong_direction_rate_at_or_below_threshold"
        ]
        is False
    )
    assert summary["controlled_real_manifest"]["evidence_rows"][2]["status"] == "fail"
    assert "controlled intervention eval did not pass" in summary["issues"]


def test_controlled_reality_bundle_handoff_blocks_weak_intervention_action_gt(
    tmp_path,
):
    _write_bundle(tmp_path, include_frame_files=True)
    rows = (tmp_path / "actions.csv").read_text(encoding="utf-8").splitlines()
    rows[1] = (
        "push-right-001,push_right,cup-001,0.033333,0.066667,"
        "scripted-hand,,0.0,0.0,0.0"
    )
    (tmp_path / "actions.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)

    with pytest.raises(
        ValueError,
        match="requires intervention action GT readiness.*non-zero vector",
    ):
        objectstate_controlled_reality_bundle_handoff(
            tmp_path,
            artifact,
            _prediction_candidates(),
            _intervention_candidates(),
            candidate_id="stable-objectstate-slots",
            max_centroid_distance=0.05,
            candidate_artifact_path=artifact_path,
        )


def test_object_state_controlled_reality_bundle_handoff_cli_writes_artifacts(
    tmp_path,
    capsys,
):
    _write_bundle(tmp_path, include_frame_files=True)
    artifact_path = tmp_path / "objectstates.json"
    predictions_path = tmp_path / "prediction-candidates.json"
    interventions_path = tmp_path / "intervention-candidates.json"
    output_dir = tmp_path / "reality-handoff"
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
                "controlled-reality-bundle-handoff",
                str(tmp_path),
                str(artifact_path),
                str(predictions_path),
                str(interventions_path),
                "--output-dir",
                str(output_dir),
                "--candidate-id",
                "cli-reality-objectstate-slots",
                "--max-centroid-distance",
                "0.05",
                "--hash-files",
                "--hash-candidate-artifact",
                "--require-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(
        (output_dir / "reality-bundle-handoff-summary.json").read_text(
            encoding="utf-8"
        )
    )
    controlled_real = json.loads(
        (output_dir / "controlled-real.json").read_text(encoding="utf-8")
    )
    controlled_real_summary = json.loads(
        (output_dir / "controlled-real-summary.json").read_text(encoding="utf-8")
    )
    prediction_eval = json.loads(
        (output_dir / "prediction-eval-summary.json").read_text(encoding="utf-8")
    )
    intervention_eval = json.loads(
        (output_dir / "intervention-eval-summary.json").read_text(encoding="utf-8")
    )

    assert f"schema={OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA}" in stdout
    assert (
        "reality_bundle_handoff_status="
        "objectstate_controlled_reality_bundle_handoff_pass"
        in stdout
    )
    assert "full_reality_gate_status=objectstate_reality_gate_pass" in stdout
    assert "identity_eval_status=objectstate_controlled_identity_eval_pass" in stdout
    assert "prediction_eval_status=objectstate_controlled_prediction_eval_pass" in stdout
    assert (
        "intervention_eval_status=objectstate_controlled_intervention_eval_pass"
        in stdout
    )
    assert "pass_rows=3" in stdout
    assert "blocked_rows=0" in stdout
    assert "action_conditioned_ade=0.000000" in stdout
    assert summary["status"] == "objectstate_controlled_reality_bundle_handoff_pass"
    assert controlled_real_summary["gate"]["status"] == "objectstate_reality_gate_pass"
    assert [row["status"] for row in controlled_real["evidence_rows"]] == [
        "pass",
        "pass",
        "pass",
    ]
    assert prediction_eval["status"] == "objectstate_controlled_prediction_eval_pass"
    assert (
        intervention_eval["status"]
        == "objectstate_controlled_intervention_eval_pass"
    )
    assert "No blocked ObjectState reality rows." in (
        output_dir / "blocked-rows.md"
    ).read_text(encoding="utf-8")


def test_public_interaction_reality_rows_convert_handoff_to_public_replay(tmp_path):
    _write_bundle(tmp_path, include_frame_files=True)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)
    handoff = objectstate_controlled_reality_bundle_handoff(
        tmp_path,
        artifact,
        _prediction_candidates(),
        _intervention_candidates(),
        candidate_id="stable-public-interaction-slots",
        max_centroid_distance=0.05,
        candidate_artifact_path=artifact_path,
    )

    summary = objectstate_public_interaction_reality_rows_summary(
        handoff,
        source_summary_ref="outputs/captures/hot3d-clip/reality-bundle-handoff-summary.json",
    )

    assert summary["schema"] == OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA
    assert summary["source_kind"] == "public_replay"
    assert summary["gate"]["status"] == "objectstate_reality_gate_pass"
    assert summary["pass_row_count"] == 3
    assert {row["source_kind"] for row in summary["rows"]} == {"public_replay"}
    intervention = next(
        row for row in summary["rows"] if row["evidence_kind"] == "intervention"
    )
    assert intervention["metrics"]["action_challenge_present"] is True
    assert (
        summary["claim_policy"]["converts_public_interaction_rows_to_public_replay"]
        is True
    )
    assert validate_objectstate_public_interaction_reality_rows_summary(
        summary
    ) == summary

    summary_path = tmp_path / "public-interaction-reality-rows.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    ledger = objectstate_reality_row_ledger((summary_path,))
    assert ledger["schema"] == "objgauss-objectstate-reality-row-ledger-v1"
    assert ledger["row_count"] == 3
    assert ledger["gate"]["status"] == "objectstate_reality_gate_pass"
    assert ledger["state_variable_evidence_matrix"][-1]["challenge_status"] == (
        "objectstate_state_variable_challenge_present"
    )


def test_object_state_audit_public_interaction_reality_rows_cli(
    tmp_path,
    capsys,
):
    _write_bundle(tmp_path, include_frame_files=True)
    artifact = _trainable_artifact()
    artifact_path = _write_candidate_artifact_file(tmp_path, artifact)
    handoff = objectstate_controlled_reality_bundle_handoff(
        tmp_path,
        artifact,
        _prediction_candidates(),
        _intervention_candidates(),
        candidate_id="stable-public-interaction-slots",
        max_centroid_distance=0.05,
        candidate_artifact_path=artifact_path,
    )
    handoff_path = tmp_path / "reality-bundle-handoff-summary.json"
    summary_path = tmp_path / "public-interaction-reality-rows.json"
    blocked_path = tmp_path / "public-interaction-blocked-rows.md"
    handoff_path.write_text(json.dumps(handoff), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "audit-public-interaction-reality-rows",
                str(handoff_path),
                "--summary-output",
                str(summary_path),
                "--blocked-rows-output",
                str(blocked_path),
                "--require-pass",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert f"schema={OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA}" in stdout
    assert "source_kind=public_replay" in stdout
    assert "gate_status=objectstate_reality_gate_pass" in stdout
    assert "row=intervention:pass:public_replay" in stdout
    assert summary["pass_row_count"] == 3
    assert "No blocked ObjectState reality rows." in blocked_path.read_text(
        encoding="utf-8"
    )


def _prediction_candidates():
    return {
        "schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
        "sample_id": "controlled-tabletop-cup-box-reality-001",
        "candidate": {
            "candidate_id": "candidate-objectstate-prediction-v0",
            "source": "fixture future pose predictions",
            "artifact_refs": [
                "outputs/controlled-real/cup-box-reality-001/predictions.json"
            ],
        },
        "predictions": [
            {
                "source_frame_id": "frame-000001",
                "target_frame_id": "frame-000002",
                "object_id": "cup-001",
                "predicted_position": [0.12, 0.2, 0.3],
                "history_baseline_position": [0.11, 0.2, 0.3],
                "confidence": 0.95,
            }
        ],
    }


def _intervention_candidates():
    return {
        "schema": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
        "sample_id": "controlled-tabletop-cup-box-reality-001",
        "candidate": {
            "candidate_id": "candidate-objectstate-intervention-v0",
            "source": "fixture action-conditioned future pose predictions",
            "artifact_refs": [
                "outputs/controlled-real/cup-box-reality-001/interventions.json"
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
                "confidence": 0.95,
            }
        ],
    }


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
                "object_id,category,instance_label",
                "cup-001,cup,blue cup",
                "box-001,box,red box",
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


def _write_candidate_artifact_file(root, artifact):
    path = root / "objectstates.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


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
