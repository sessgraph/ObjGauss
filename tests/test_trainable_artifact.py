from __future__ import annotations

import json

import pytest

from objgauss.pipelines.trainable_artifact import (
    TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
    trainable_kernel_model_artifact,
    validate_trainable_kernel_model_artifact,
    write_trainable_kernel_model_artifact,
)
from objgauss.pipelines.trainable_kernel import (
    bind_image_targets_to_frames,
    make_trainable_kernel_mvp_fixture,
    train_kernel_mvp,
)
from objgauss.pipelines.training_renderer import evaluate_training_renderer_loss


def test_trainable_kernel_model_artifact_captures_model_state(tmp_path):
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    result = train_kernel_mvp(
        frames,
        slots=2,
        iterations=8,
        learning_rate=0.35,
        image_render_weight=0.5,
        seed=12,
    )
    renderer_api = evaluate_training_renderer_loss(
        frames,
        result.assignments,
        result.decoder_colors,
    ).as_dict()

    artifact = trainable_kernel_model_artifact(
        result,
        input_path="fixture://trainable-kernel-mvp",
        renderer_api=renderer_api,
    )

    assert artifact["schema"] == TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA
    assert artifact["kind"] == "trainable_kernel_mvp_model"
    assert artifact["source"]["input"] == "fixture://trainable-kernel-mvp"
    assert artifact["renderer_api"]["status"] == "ready"
    assert artifact["training"]["weights"]["image_render"] == 0.5
    assert len(artifact["assignments"]) == 2
    assert artifact["assignments"][0]["shape"] == [6, 2]
    assert len(artifact["object_states"][0]["states"]) == 2
    first_state = artifact["object_states"][0]["states"][0]
    assert "bbox" in first_state
    assert first_state["id"] == first_state["persistent_id"]
    assert first_state["slot"] == first_state["object_id"]
    assert first_state["id"] != first_state["object_id"]
    training_state = artifact["training"]["object_states"][0][0]
    assert training_state["id"] == training_state["persistent_id"]
    assert training_state["slot"] == training_state["object_id"]
    assert 0.0 <= first_state["confidence"] <= 1.0
    assert artifact["learned_parameters"]["decoder_colors"]
    assert validate_trainable_kernel_model_artifact(artifact) is True

    legacy_artifact = json.loads(json.dumps(artifact))
    for frame in legacy_artifact["object_states"]:
        for state in frame["states"]:
            state.pop("persistent_id")
            state.pop("slot")
            state.pop("object_id")
    for frame in legacy_artifact["training"]["object_states"]:
        for state in frame:
            state.pop("persistent_id")
            state.pop("slot")
            state.pop("object_id")
    assert validate_trainable_kernel_model_artifact(legacy_artifact) is True

    output = tmp_path / "trainable-model.json"
    written = write_trainable_kernel_model_artifact(
        output,
        result,
        input_path="fixture://trainable-kernel-mvp",
        renderer_api=renderer_api,
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert written["schema"] == TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA
    assert loaded["schema"] == TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA
    assert loaded["artifact_policy"]["git_policy"] == "do_not_commit_training_outputs_by_default"


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("persistent_id_mismatch", "id must match persistent_id"),
        ("object_id_mismatch", "slot must match object_id"),
        ("partial_canonical_addresses", "requires persistent_id, slot, and object_id"),
        ("negative_id", "id must be non-negative"),
        ("negative_slot", "slot must be non-negative"),
        ("duplicate_id", "id must be unique within frame 0"),
        ("duplicate_slot", "slot must be unique within frame 0"),
        ("missing_confidence", "requires confidence"),
        ("non_finite_confidence", "confidence must be finite"),
        ("out_of_range_confidence", "confidence must be in \\[0, 1\\]"),
    ),
)
def test_trainable_artifact_validator_enforces_canonical_objectstate_abi(
    case,
    message,
):
    artifact = _canonical_abi_artifact()
    first = artifact["object_states"][0]["states"][0]
    second = artifact["object_states"][0]["states"][1]
    if case == "persistent_id_mismatch":
        first["persistent_id"] = 999
    elif case == "object_id_mismatch":
        first["object_id"] = 999
    elif case == "partial_canonical_addresses":
        first.pop("object_id")
    elif case == "negative_id":
        first["id"] = first["persistent_id"] = -1
    elif case == "negative_slot":
        first["slot"] = first["object_id"] = -1
    elif case == "duplicate_id":
        second["id"] = second["persistent_id"] = first["id"]
    elif case == "duplicate_slot":
        second["slot"] = second["object_id"] = first["slot"]
    elif case == "missing_confidence":
        first.pop("confidence")
    elif case == "non_finite_confidence":
        first["confidence"] = float("nan")
    elif case == "out_of_range_confidence":
        first["confidence"] = 1.01

    with pytest.raises((TypeError, ValueError), match=message):
        validate_trainable_kernel_model_artifact(artifact)


def test_trainable_artifact_validator_explicitly_separates_legacy_layout():
    artifact = _canonical_abi_artifact()
    for state in artifact["object_states"][0]["states"]:
        state.pop("persistent_id")
        state.pop("slot")
        state.pop("object_id")

    with pytest.raises(ValueError, match="cannot mix canonical and legacy layouts"):
        validate_trainable_kernel_model_artifact(artifact)

    artifact["training"].pop("object_states")
    assert validate_trainable_kernel_model_artifact(artifact) is True

    artifact["object_states"][0]["states"][0].pop("confidence")
    with pytest.raises(ValueError, match="requires confidence"):
        validate_trainable_kernel_model_artifact(artifact)


def _canonical_abi_artifact():
    states = [
        {
            "id": 101,
            "persistent_id": 101,
            "slot": 0,
            "object_id": 0,
            "confidence": 0.75,
        },
        {
            "id": 202,
            "persistent_id": 202,
            "slot": 1,
            "object_id": 1,
            "confidence": 0.8,
        },
    ]
    return {
        "schema": TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
        "kind": "trainable_kernel_mvp_model",
        "source": {},
        "training": {
            "schema": "objgauss-v1-trainable-kernel-mvp-v1",
            "frame_count": 1,
            "object_states": [json.loads(json.dumps(states))],
        },
        "learned_parameters": {},
        "assignments": [{}],
        "object_states": [
            {
                "frame_index": 0,
                "states": states,
                "derived_object_ids": [0, 1],
            }
        ],
        "artifact_policy": {},
    }
