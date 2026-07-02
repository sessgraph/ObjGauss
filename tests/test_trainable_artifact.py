from __future__ import annotations

import json

from objgauss.core.trainable_artifact import (
    TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
    trainable_kernel_model_artifact,
    validate_trainable_kernel_model_artifact,
    write_trainable_kernel_model_artifact,
)
from objgauss.core.trainable_kernel import (
    bind_image_targets_to_frames,
    make_trainable_kernel_mvp_fixture,
    train_kernel_mvp,
)
from objgauss.core.training_renderer import evaluate_training_renderer_loss


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
    assert "bbox" in artifact["object_states"][0]["states"][0]
    assert artifact["learned_parameters"]["decoder_colors"]
    assert validate_trainable_kernel_model_artifact(artifact) is True

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
