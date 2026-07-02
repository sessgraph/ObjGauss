from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.trainable_kernel import (
    bind_image_targets_to_frames,
    make_trainable_kernel_mvp_fixture,
    train_kernel_mvp,
)
from objgauss.core.training_renderer import (
    CPU_IMAGE_SPLAT_GRADIENT_PATH,
    CPU_IMAGE_SPLAT_RENDERER,
    TRAINING_RENDERER_API_SCHEMA,
    evaluate_training_renderer_loss,
    validate_training_renderer_summary,
)


def test_training_renderer_reconstructs_bound_image_target_with_known_assignments():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    assignment = np.zeros((6, 2), dtype=np.float32)
    assignment[:3, 0] = 1.0
    assignment[3:, 1] = 1.0
    decoder_colors = np.vstack([frames[0].target_rgb[0], frames[0].target_rgb[3]]).astype(np.float32)

    result = evaluate_training_renderer_loss(
        frames[:1],
        [assignment],
        decoder_colors,
    )
    payload = result.as_dict()

    assert result.schema == TRAINING_RENDERER_API_SCHEMA
    assert payload["status"] == "ready"
    assert payload["renderer_name"] == CPU_IMAGE_SPLAT_RENDERER
    assert payload["gradient_path"] == CPU_IMAGE_SPLAT_GRADIENT_PATH
    assert result.image_render_loss == pytest.approx(0.0, abs=1e-12)
    assert payload["gradients"]["decoder_colors_shape"] == [2, 3]
    assert payload["gradients"]["assignment_shapes"] == [[6, 2]]
    assert payload["gradients"]["decoder_colors_l2"] == pytest.approx(0.0, abs=1e-7)
    assert validate_training_renderer_summary(payload) is True


def test_training_renderer_evaluates_trainable_result_with_gradients():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    train_result = train_kernel_mvp(
        frames,
        slots=2,
        iterations=8,
        learning_rate=0.35,
        seed=9,
    )

    render_result = evaluate_training_renderer_loss(
        frames,
        train_result.assignments,
        train_result.decoder_colors,
    )
    payload = render_result.as_dict()

    assert render_result.image_render_loss >= 0
    assert len(render_result.rendered_images) == 2
    assert render_result.rendered_images[0].shape == (8, 8, 3)
    assert render_result.gradient_decoder_colors.shape == (2, 3)
    assert render_result.gradient_assignments[0].shape == (6, 2)
    assert payload["frame_losses"][0]["supervised_pixels"] > 0
    assert payload["differentiable_fields"] == ["decoder_colors", "assignments"]
    assert payload["frozen_fields"] == ["positions", "camera", "visibility_mask", "point_radius"]


def test_training_renderer_requires_image_targets():
    frames = make_trainable_kernel_mvp_fixture()
    assignment = np.full((6, 2), 0.5, dtype=np.float32)
    decoder_colors = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="image_target is required"):
        evaluate_training_renderer_loss(frames[:1], [assignment], decoder_colors)


def test_training_renderer_requires_normalized_assignments():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    assignment = np.ones((6, 2), dtype=np.float32)
    decoder_colors = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="rows must sum to 1"):
        evaluate_training_renderer_loss(frames[:1], [assignment], decoder_colors)
