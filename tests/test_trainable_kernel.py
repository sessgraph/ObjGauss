from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.trainable_kernel import (
    TRAINABLE_KERNEL_MVP_SCHEMA,
    TrainableKernelFrame,
    make_trainable_kernel_mvp_fixture,
    train_kernel_mvp,
    train_kernel_mvp_from_cloud,
    trainable_kernel_sample_from_cloud,
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


def test_trainable_kernel_sample_from_object_cloud_uses_object_id_targets():
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=4,
        temporal_offset=0.02,
        seed=2,
    )

    assert sample.slots == 2
    assert sample.source_count == 6
    assert sample.sampled_count == 4
    assert sample.target_source == "object_id_one_hot_targets"
    assert sample.object_id_mapping == {5: 0, 9: 1}
    assert len(sample.frames) == 2
    assert sample.frames[0].target_assignment.shape == (4, 2)
    np.testing.assert_allclose(sample.frames[0].target_assignment.sum(axis=1), np.ones(4))
    assert not np.allclose(sample.frames[0].positions, sample.frames[1].positions)


def test_trainable_kernel_from_cloud_reduces_sample_loss():
    result, sample = train_kernel_mvp_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=6,
        iterations=45,
        learning_rate=0.5,
        seed=4,
    )

    assert sample.target_source == "object_id_one_hot_targets"
    assert result.slots == 2
    assert result.final_loss.total_loss < result.initial_loss.total_loss
    assert result.final_loss.render_loss < result.initial_loss.render_loss
    assert result.object_state_projections[0].states[0].status == "active"


def test_trainable_kernel_sample_requires_slots_without_object_ids():
    cloud = _object_cloud(include_object_ids=False)

    with pytest.raises(ValueError, match="slots is required"):
        trainable_kernel_sample_from_cloud(cloud)

    sample = trainable_kernel_sample_from_cloud(cloud, slots=2, frame_count=1)

    assert sample.target_source == "feature_quantile_pseudo_targets"
    assert sample.object_id_mapping == {}
    assert sample.frames[0].target_assignment is None


def _object_cloud(*, include_object_ids: bool = True) -> GaussianCloud:
    fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("opacity", "f4"),
    ]
    if include_object_ids:
        fields.append(("object_id", "i4"))
    vertices = np.zeros(6, dtype=np.dtype(fields))
    vertices["x"] = np.array([-1.0, -0.8, -0.7, 0.8, 1.0, 1.2], dtype=np.float32)
    vertices["y"] = np.array([0.0, 0.1, -0.1, 0.0, 0.12, -0.08], dtype=np.float32)
    vertices["z"] = 0.0
    vertices["red"] = np.array([240, 230, 220, 20, 30, 40], dtype=np.uint8)
    vertices["green"] = np.array([30, 35, 40, 220, 230, 225], dtype=np.uint8)
    vertices["blue"] = np.array([25, 30, 35, 220, 230, 235], dtype=np.uint8)
    vertices["opacity"] = 1.0
    if include_object_ids:
        vertices["object_id"] = np.array([5, 5, 5, 9, 9, 9], dtype=np.int32)
    return GaussianCloud(vertices=vertices, source_format="ascii")
