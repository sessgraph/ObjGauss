from __future__ import annotations

import json

import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_controlled_capture import (
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
)
from objgauss.evaluation.objectstate_controlled_identity_eval import (
    OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
    evaluate_objectstate_controlled_identity_predictions,
    validate_objectstate_controlled_identity_predictions,
)
from objgauss.pipelines.objectstate_identity_prediction_adapter import (
    objectstate_identity_predictions_from_trainable_artifact,
    read_trainable_kernel_identity_source,
)
from objgauss.pipelines.trainable_artifact import TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA
from objgauss.pipelines.trainable_artifact import write_trainable_kernel_model_artifact
from objgauss.pipelines.trainable_kernel import (
    make_trainable_kernel_mvp_fixture,
    train_kernel_mvp,
)


def test_trainable_artifact_adapter_exports_controlled_identity_predictions():
    predictions = objectstate_identity_predictions_from_trainable_artifact(
        _capture_manifest(),
        _trainable_artifact(),
        candidate_id="stable-objectstate-slots",
        artifact_refs=("outputs/controlled-real/cup-box/objectstates.json",),
        max_centroid_distance=0.05,
    )

    assert predictions["schema"] == OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA
    assert predictions["sample_id"] == "controlled-tabletop-cup-box-identity-001"
    assert predictions["association_mode"] == "raw_track_observations"
    assert predictions["candidate"]["candidate_id"] == "stable-objectstate-slots"
    assert predictions["candidate"]["identity_evidence"] == {
        "reconstruction_noise_robustness": 1.0,
        "reconstruction_noise_variant_count": 2,
        "source": "fixture repeated Gaussian reconstruction noise variants",
    }
    assert len(predictions["predictions"]) == 6
    assert predictions["predictions"][0]["predicted_identity"] == "object-state-101"
    assert predictions["predictions"][1]["predicted_identity"] == "object-state-202"
    assert predictions["predictions"][0]["predicted_position"] == pytest.approx(
        (0.1, 0.2, 0.3)
    )
    assert all("object_id" not in row for row in predictions["predictions"])
    assert validate_objectstate_controlled_identity_predictions(predictions) == predictions

    summary = evaluate_objectstate_controlled_identity_predictions(
        _capture_manifest(),
        predictions,
    )

    assert summary["status"] == "objectstate_controlled_identity_eval_pass"
    assert summary["metrics"]["idf1"] == 1.0
    assert summary["metrics"]["track_retrieval_recall_at_1"] == 1.0
    assert summary["metrics"]["long_term_drift_rate"] == 0.0
    assert summary["metrics"]["fragmentation_rate"] == 0.0
    assert summary["metrics"]["swap_rate"] == 0.0
    assert summary["metrics"]["reconstruction_noise_robustness"] == 1.0


def test_trainable_artifact_adapter_surfaces_fragmented_slots():
    artifact = _trainable_artifact(
        slot_ids_by_frame=((0, 1), (1, 0), (1, 0)),
        persistent_ids_follow_slots=True,
    )

    predictions = objectstate_identity_predictions_from_trainable_artifact(
        _capture_manifest(),
        artifact,
        candidate_id="fragmented-objectstate-slots",
    )
    summary = evaluate_objectstate_controlled_identity_predictions(
        _capture_manifest(),
        predictions,
    )

    assert summary["status"] == "objectstate_controlled_identity_eval_fail"
    assert summary["metrics"]["fragmentation_rate"] > 0.0
    assert summary["metrics"]["swap_rate"] > 0.0


def test_trainable_serializer_to_identity_adapter_preserves_persistent_ids(tmp_path):
    result = train_kernel_mvp(
        make_trainable_kernel_mvp_fixture(),
        slots=2,
        iterations=2,
        learning_rate=0.2,
        seed=4,
    )
    artifact_path = tmp_path / "objectstates.json"
    written = write_trainable_kernel_model_artifact(
        artifact_path,
        result,
        input_path="fixture://serializer-roundtrip",
    )
    artifact = read_trainable_kernel_identity_source(artifact_path)

    first_states = written["object_states"][0]["states"]
    assert all(state["id"] == state["persistent_id"] for state in first_states)
    assert all(state["id"] != state["slot"] for state in first_states)
    assert all(0.0 <= state["confidence"] <= 1.0 for state in first_states)

    capture = _capture_manifest_for_artifact(artifact)
    predictions = objectstate_identity_predictions_from_trainable_artifact(
        capture,
        artifact,
    )
    identities_by_frame: dict[str, set[str]] = {}
    for row in predictions["predictions"]:
        identities_by_frame.setdefault(row["frame_id"], set()).add(
            row["predicted_identity"]
        )
        assert "object_id" not in row
    assert all(len(values) == 2 for values in identities_by_frame.values())
    assert all(
        identity.startswith("object-state-")
        for values in identities_by_frame.values()
        for identity in values
    )


def test_trainable_artifact_adapter_rejects_frame_count_mismatch():
    artifact = _trainable_artifact()
    artifact["object_states"] = artifact["object_states"][:2]
    artifact["assignments"] = artifact["assignments"][:2]
    artifact["training"]["frame_count"] = 2

    with pytest.raises(ValueError, match="frame count must match capture frames"):
        objectstate_identity_predictions_from_trainable_artifact(
            _capture_manifest(),
            artifact,
        )


def test_trainable_artifact_adapter_does_not_read_capture_pose_or_object_order():
    capture = _capture_manifest()
    reference = objectstate_identity_predictions_from_trainable_artifact(
        capture,
        _trainable_artifact(),
    )
    for frame_index, frame in enumerate(capture["frames"]):
        frame["objects"].reverse()
        for object_index, item in enumerate(frame["objects"]):
            item["pose"]["position"] = [
                100.0 + frame_index,
                -100.0 - object_index,
                50.0,
            ]

    changed_ground_truth = objectstate_identity_predictions_from_trainable_artifact(
        capture,
        _trainable_artifact(),
    )

    assert changed_ground_truth == reference


def test_trainable_artifact_adapter_can_filter_far_centroids():
    predictions = objectstate_identity_predictions_from_trainable_artifact(
        _capture_manifest(),
        _trainable_artifact(x_offset=10.0),
        max_centroid_distance=0.01,
    )
    summary = evaluate_objectstate_controlled_identity_predictions(
        _capture_manifest(),
        predictions,
    )

    assert len(predictions["predictions"]) == 6
    assert predictions["candidate"]["max_association_distance"] == 0.01
    assert summary["status"] == "objectstate_controlled_identity_eval_fail"
    assert summary["metrics"]["missing_prediction_count"] == 6
    assert summary["metrics"]["unmatched_prediction_count"] == 6


def test_trainable_artifact_adapter_rejects_non_finite_centroid_distance():
    with pytest.raises(ValueError, match="max_centroid_distance must be finite"):
        objectstate_identity_predictions_from_trainable_artifact(
            _capture_manifest(),
            _trainable_artifact(),
            max_centroid_distance=float("inf"),
        )


def test_trainable_artifact_adapter_requires_explicit_confidence():
    artifact = _trainable_artifact()
    artifact["object_states"][0]["states"][0].pop("confidence")

    with pytest.raises(ValueError, match="requires confidence"):
        objectstate_identity_predictions_from_trainable_artifact(
            _capture_manifest(),
            artifact,
        )


def test_trainable_artifact_adapter_preserves_legacy_confidence():
    artifact = _trainable_artifact()
    for frame in artifact["object_states"]:
        for state in frame["states"]:
            state.pop("persistent_id")
            state.pop("slot")
            state.pop("object_id")
    artifact["object_states"][0]["states"][0]["confidence"] = 0.37

    predictions = objectstate_identity_predictions_from_trainable_artifact(
        _capture_manifest(),
        artifact,
    )

    assert predictions["predictions"][0]["confidence"] == pytest.approx(0.37)


def test_trainable_artifact_identity_source_reads_json(tmp_path):
    artifact_path = tmp_path / "objectstates.json"
    artifact_path.write_text(json.dumps(_trainable_artifact()), encoding="utf-8")

    artifact = read_trainable_kernel_identity_source(artifact_path)

    assert artifact["schema"] == TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA
    assert artifact["label"] == "fixture-trainable-objectstates"


def test_object_state_export_identity_predictions_cli_writes_eval_input(
    tmp_path,
    capsys,
):
    capture_path = tmp_path / "capture.json"
    artifact_path = tmp_path / "objectstates.json"
    predictions_path = tmp_path / "identity-predictions.json"
    capture_path.write_text(json.dumps(_capture_manifest()), encoding="utf-8")
    artifact_path.write_text(json.dumps(_trainable_artifact()), encoding="utf-8")

    assert (
        main(
            [
                "object-state",
                "export-identity-predictions",
                str(capture_path),
                str(artifact_path),
                "--output",
                str(predictions_path),
                "--candidate-id",
                "cli-objectstate-slots",
                "--max-centroid-distance",
                "0.05",
            ]
        )
        == 0
    )

    stdout = capsys.readouterr().out
    predictions = json.loads(predictions_path.read_text(encoding="utf-8"))
    summary = evaluate_objectstate_controlled_identity_predictions(
        _capture_manifest(),
        predictions,
    )

    assert f"schema={OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA}" in stdout
    assert "candidate_id=cli-objectstate-slots" in stdout
    assert "prediction_count=6" in stdout
    assert predictions["candidate"]["artifact_refs"] == [str(artifact_path)]
    assert summary["status"] == "objectstate_controlled_identity_eval_pass"


def _capture_manifest():
    frames = []
    for frame_index, timestamp in enumerate((0.0, 0.033333, 0.066667)):
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


def _trainable_artifact(
    *,
    slot_ids_by_frame: tuple[tuple[int, int], ...] = ((0, 1), (0, 1), (0, 1)),
    x_offset: float = 0.0,
    include_identity_evidence: bool = True,
    persistent_ids_follow_slots: bool = False,
):
    object_states = []
    assignments = []
    for frame_index, slot_ids in enumerate(slot_ids_by_frame):
        cup_slot, box_slot = slot_ids
        cup_x = 0.1 + 0.01 * frame_index + x_offset
        box_x = 0.4 + 0.01 * frame_index + x_offset
        object_states.append(
            {
                "frame_index": frame_index,
                "states": [
                    _state(
                        cup_slot,
                        [cup_x, 0.2, 0.3],
                        persistent_id=cup_slot if persistent_ids_follow_slots else 101,
                    ),
                    _state(
                        box_slot,
                        [box_x, 0.2, 0.3],
                        persistent_id=box_slot if persistent_ids_follow_slots else 202,
                    ),
                ],
                "derived_object_ids": [cup_slot, box_slot],
            }
        )
        assignments.append(
            {
                "frame_index": frame_index,
                "shape": [2, 2],
                "matrix": [[1.0, 0.0], [0.0, 1.0]],
            }
        )
    artifact = {
        "schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "kind": "trainable_kernel_mvp_model",
        "label": "fixture-trainable-objectstates",
        "source": {
            "input": "outputs/controlled-real/cup-box-identity-001/objectstates.json",
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
        "artifact_policy": {
            "git_policy": "do_not_commit_training_outputs_by_default",
        },
    }
    if include_identity_evidence:
        artifact["identity_evidence"] = {
            "reconstruction_noise_robustness": 1.0,
            "reconstruction_noise_variant_count": 2,
            "source": "fixture repeated Gaussian reconstruction noise variants",
        }
    return artifact


def _state(slot: int, centroid: list[float], *, persistent_id: int):
    return {
        "id": persistent_id,
        "persistent_id": persistent_id,
        "slot": slot,
        "object_id": slot,
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


def _capture_manifest_for_artifact(artifact: dict):
    persistent_ids = [
        int(state["id"])
        for state in artifact["object_states"][0]["states"]
    ]
    frames = []
    for frame_index, artifact_frame in enumerate(artifact["object_states"]):
        states_by_id = {
            int(state["id"]): state
            for state in artifact_frame["states"]
        }
        frames.append(
            {
                "frame_id": f"frame-{frame_index:06d}",
                "timestamp": frame_index / 30.0,
                "observation": {
                    "rgb": f"rgb/{frame_index:06d}.png",
                    "gaussian": f"gaussians/{frame_index:06d}.ply",
                },
                "objects": [
                    {
                        "object_id": f"physical-{persistent_id}",
                        "visible": True,
                        "occlusion_fraction": 0.0,
                        "pose": {
                            "position": states_by_id[persistent_id]["centroid"],
                            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                    for persistent_id in persistent_ids
                ],
            }
        )
    return {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "serializer-identity-roundtrip",
            "source_kind": "controlled_real",
            "object_category": "fixture",
            "scenario": "serializer_roundtrip",
            "fps": 30.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": ["fixture://serializer-roundtrip"],
            "license": "test fixture",
        },
        "objects": [
            {
                "object_id": f"physical-{persistent_id}",
                "category": "fixture",
                "instance_label": f"fixture {persistent_id}",
            }
            for persistent_id in persistent_ids
        ],
        "actions": [],
        "frames": frames,
    }
