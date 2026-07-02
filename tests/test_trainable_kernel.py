from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.trainable_kernel import (
    TRAINABLE_KERNEL_MVP_SCHEMA,
    TRAINABLE_IMAGE_TARGET_CONTRACT_SCHEMA,
    TrainableKernelFrame,
    bind_image_targets_to_frames,
    image_target_contract_summary,
    make_trainable_kernel_mvp_fixture,
    train_kernel_mvp,
    train_kernel_mvp_from_cloud,
    trainable_kernel_sample_from_cloud,
    validate_image_target_contract_summary,
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
    assert result.image_renderer == "point"
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


def test_trainable_image_targets_bind_camera_and_visibility_contract():
    frames = bind_image_targets_to_frames(
        make_trainable_kernel_mvp_fixture(),
        width=12,
        height=10,
        point_radius=1,
    )

    assert all(frame.image_target is not None for frame in frames)
    target = frames[0].image_target
    assert target is not None
    assert target.image.shape == (10, 12, 3)
    assert target.visibility_mask.shape == (10, 12)
    assert np.mean(target.visibility_mask) > 0
    assert target.camera.width == 12
    assert target.camera.height == 10
    assert target.camera.intrinsics.shape == (3, 3)
    assert target.camera.camera_to_world.shape == (4, 4)

    summary = image_target_contract_summary(tuple(frame.image_target for frame in frames))
    assert summary["schema"] == TRAINABLE_IMAGE_TARGET_CONTRACT_SCHEMA
    assert summary["status"] == "image_targets_bound"
    assert summary["targets_bound"] == 2
    assert summary["visibility_policies"] == ["covered_pixels"]
    assert validate_image_target_contract_summary(summary) is True


def test_trainable_kernel_summary_marks_bound_image_targets():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    result = train_kernel_mvp(
        frames,
        slots=2,
        iterations=12,
        learning_rate=0.4,
        seed=5,
    )
    summary = result.as_dict()

    assert summary["render_target_mode"] == "image_space_targets_bound"
    assert summary["image_target_contract"]["status"] == "image_targets_bound"
    assert summary["image_target_contract"]["targets"][0]["shape"] == [8, 8, 3]
    assert result.final_loss.total_loss < result.initial_loss.total_loss


def test_trainable_kernel_can_optimize_image_render_loss():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    result = train_kernel_mvp(
        frames,
        slots=2,
        iterations=18,
        learning_rate=0.35,
        render_weight=0.0,
        image_render_weight=1.0,
        object_weight=0.0,
        temporal_weight=0.0,
        seed=10,
    )
    summary = result.as_dict()

    assert result.initial_loss.render_loss > 0
    assert result.initial_loss.image_render_loss > 0
    assert result.final_loss.image_render_loss < result.initial_loss.image_render_loss
    assert result.final_loss.total_loss < result.initial_loss.total_loss
    assert summary["weights"]["image_render"] == 1.0
    assert summary["image_renderer"] == "point"
    assert summary["image_render_loss_decreased"] is True


def test_trainable_kernel_can_route_image_loss_to_gsplat_adapter(monkeypatch):
    calls = []

    def fake_gsplat_renderer(frames, assignments, decoder_colors):
        calls.append(
            {
                "frames": len(frames),
                "assignment_shape": assignments[0].shape,
                "decoder_shape": decoder_colors.shape,
            }
        )
        return SimpleNamespace(image_render_loss=0.123)

    monkeypatch.setattr(
        "objgauss.core.gsplat_training_renderer.evaluate_gsplat_training_renderer_loss",
        fake_gsplat_renderer,
    )
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    result = train_kernel_mvp(
        frames,
        slots=2,
        iterations=1,
        learning_rate=0.25,
        render_weight=0.0,
        image_render_weight=1.0,
        object_weight=0.0,
        temporal_weight=0.0,
        image_renderer="gsplat",
        seed=12,
    )

    assert result.image_renderer == "gsplat"
    assert result.as_dict()["image_renderer"] == "gsplat"
    assert result.final_loss.image_render_loss == pytest.approx(0.123)
    assert calls
    assert calls[0]["frames"] == 2
    assert calls[0]["assignment_shape"] == (6, 2)
    assert calls[0]["decoder_shape"] == (2, 3)


def test_trainable_kernel_rejects_unknown_image_renderer():
    with pytest.raises(ValueError, match="image_renderer must be one of"):
        train_kernel_mvp(
            make_trainable_kernel_mvp_fixture(),
            slots=2,
            iterations=2,
            image_renderer="spark",
        )


def test_trainable_kernel_image_render_weight_requires_targets():
    with pytest.raises(ValueError, match="requires every frame to bind image_target"):
        train_kernel_mvp(
            make_trainable_kernel_mvp_fixture(),
            slots=2,
            iterations=2,
            image_render_weight=1.0,
        )


def test_trainable_kernel_sample_requires_slots_without_object_ids():
    cloud = _object_cloud(include_object_ids=False)

    with pytest.raises(ValueError, match="slots is required"):
        trainable_kernel_sample_from_cloud(cloud)

    sample = trainable_kernel_sample_from_cloud(cloud, slots=2, frame_count=1)

    assert sample.target_source == "feature_quantile_pseudo_targets"
    assert sample.object_id_mapping == {}
    assert sample.frames[0].target_assignment is None


def test_trainable_kernel_sample_can_bind_image_targets():
    result, sample = train_kernel_mvp_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=4,
        bind_image_targets=True,
        image_width=10,
        image_height=9,
        iterations=12,
        learning_rate=0.4,
        seed=7,
    )

    assert sample.frames[0].image_target is not None
    assert sample.as_dict()["image_target_contract"]["status"] == "image_targets_bound"
    assert result.as_dict()["image_target_contract"]["targets_bound"] == 2


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
