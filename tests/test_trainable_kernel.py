from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.trainable_kernel import (
    TRAINABLE_KERNEL_MVP_SCHEMA,
    TrainableKernelFrame,
    make_trainable_kernel_mvp_fixture,
    train_kernel_mvp,
)


def test_trainable_kernel_mvp_reduces_end_to_end_loss():
    result = train_kernel_mvp(
        make_trainable_kernel_mvp_fixture(),
        slots=2,
        iterations=35,
        learning_rate=0.5,
        seed=3,
    )

    assert result.schema == TRAINABLE_KERNEL_MVP_SCHEMA
    assert result.frame_count == 2
    assert result.slots == 2
    assert result.final_loss.total_loss < result.initial_loss.total_loss
    assert result.final_loss.render_loss < result.initial_loss.render_loss
    assert len(result.assignments) == 2
    assert result.assignments[0].shape == (6, 2)
    assert len(result.object_state_projections) == 2
    assert result.object_state_projections[0].states[0].status == "active"
    assert result.object_state_projections[0].states[1].status == "active"
    assert result.rendered_rgb[0].shape == (6, 3)
    assert result.decoder_colors.shape == (2, 3)

    summary = result.as_dict()
    assert summary["loss_decreased"] is True
    assert summary["render_loss_decreased"] is True
    assert summary["object_states"][0][0]["slot_mass"] > 0


def test_trainable_kernel_validates_frame_shapes():
    frame = TrainableKernelFrame(
        positions=np.zeros((2, 3), dtype=np.float32),
        features=np.zeros((3, 2), dtype=np.float32),
        target_rgb=np.zeros((2, 3), dtype=np.float32),
    )

    with pytest.raises(ValueError, match="features rows must match positions"):
        train_kernel_mvp([frame], slots=2)

    with pytest.raises(ValueError, match="slots must be >= 1"):
        train_kernel_mvp(make_trainable_kernel_mvp_fixture(), slots=0)
