from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.gaussian import GaussianCloud
from objgauss.pipelines.objectstate_model_v0 import (
    OBJECTSTATE_MODEL_V0_STATE_SCHEMA,
    OBJECTSTATE_MODEL_V0_TRAINING_SCHEMA,
    objectstate_model_v0_loss_and_gradient,
    objectstate_model_v0_state_from_dict,
    train_objectstate_model_v0,
    validate_objectstate_model_v0_training_summary,
)


def test_objectstate_model_v0_trains_encoder_head_on_heldout_frames():
    cloud = _frame_cloud()

    result = train_objectstate_model_v0(
        cloud,
        hidden_dim=12,
        heldout_stride=3,
        iterations=320,
        learning_rate=0.12,
        compactness_weight=0.01,
        semantic_weight=0.01,
        weight_decay=0.0,
        seed=0,
    )
    summary = result.as_dict()

    assert summary["schema"] == OBJECTSTATE_MODEL_V0_TRAINING_SCHEMA
    assert summary["status"] == "objectstate_model_v0_training_pass"
    assert summary["model_family"] == "gaussian-object-encoder-assignment-head-v0"
    assert summary["state_schema"] == OBJECTSTATE_MODEL_V0_STATE_SCHEMA
    assert summary["loss"]["total_decreased"] is True
    assert summary["loss"]["supervised_decreased"] is True
    assert summary["heldout_after_metrics"]["ari"] > summary["heldout_before_metrics"]["ari"]
    assert summary["heldout_after_metrics"]["ari"] == pytest.approx(1.0)
    assert summary["heldout_after_metrics"]["purity"] == pytest.approx(1.0)
    assert summary["split"]["frame_overlap_count"] == 0
    assert summary["split"]["row_overlap_count"] == 0
    assert set(summary["split"]["train_frame_ids"]).isdisjoint(
        summary["split"]["heldout_frame_ids"]
    )
    assert summary["claim_policy"]["same_scene_heldout_frames_only"] is True
    assert validate_objectstate_model_v0_training_summary(summary) == summary

    checkpoint = result.final_state.as_dict(include_arrays=True)
    restored = objectstate_model_v0_state_from_dict(checkpoint)
    expected = result.final_state.predict(cloud)
    actual = restored.predict(cloud)
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    np.testing.assert_array_equal(np.argmax(actual, axis=1), np.argmax(expected, axis=1))


def test_objectstate_model_v0_same_seed_is_exactly_reproducible():
    cloud = _frame_cloud()
    options = {
        "hidden_dim": 10,
        "heldout_stride": 3,
        "iterations": 90,
        "learning_rate": 0.1,
        "compactness_weight": 0.01,
        "semantic_weight": 0.01,
        "weight_decay": 1e-4,
        "seed": 17,
    }

    first = train_objectstate_model_v0(cloud, **options)
    second = train_objectstate_model_v0(cloud, **options)

    assert first.final_state.as_dict(include_arrays=True) == second.final_state.as_dict(
        include_arrays=True
    )
    assert first.as_dict() == second.as_dict()
    np.testing.assert_array_equal(first.initial_assignment, second.initial_assignment)
    np.testing.assert_array_equal(first.final_assignment, second.final_assignment)


def test_objectstate_model_v0_combined_logit_gradient_matches_finite_difference():
    logits = np.asarray(
        [[0.2, -0.1], [-0.4, 0.5], [0.1, 0.3], [0.7, -0.2]],
        dtype=np.float32,
    )
    labels = np.asarray([0, 1, 1, 0], dtype=np.int64)
    spatial = np.asarray(
        [[-1.0, 0.0], [1.0, 0.1], [0.8, -0.1], [-0.9, 0.2]],
        dtype=np.float32,
    )
    semantic = np.asarray(
        [[1.0, 0.0], [0.0, 1.0], [0.1, 0.9], [0.9, 0.1]],
        dtype=np.float32,
    )
    _, analytic = objectstate_model_v0_loss_and_gradient(
        logits,
        labels,
        spatial,
        semantic,
        assignment_weight=1.0,
        compactness_weight=0.07,
        semantic_weight=0.11,
    )
    epsilon = 1e-3
    numeric = np.zeros_like(logits)
    for row in range(logits.shape[0]):
        for column in range(logits.shape[1]):
            positive = logits.copy()
            negative = logits.copy()
            positive[row, column] += epsilon
            negative[row, column] -= epsilon
            positive_loss, _ = objectstate_model_v0_loss_and_gradient(
                positive,
                labels,
                spatial,
                semantic,
                compactness_weight=0.07,
                semantic_weight=0.11,
            )
            negative_loss, _ = objectstate_model_v0_loss_and_gradient(
                negative,
                labels,
                spatial,
                semantic,
                compactness_weight=0.07,
                semantic_weight=0.11,
            )
            numeric[row, column] = (
                positive_loss.total_loss - negative_loss.total_loss
            ) / (2.0 * epsilon)
    np.testing.assert_allclose(analytic, numeric, atol=2e-4, rtol=2e-3)


def test_objectstate_model_v0_requires_non_overlapping_frame_source():
    cloud = _frame_cloud(frame_count=1)

    with pytest.raises(ValueError, match="at least two source frames"):
        train_objectstate_model_v0(cloud, iterations=1)


def test_objectstate_model_v0_keeps_missing_heldout_object_coverage_reviewable():
    cloud = _frame_cloud(frame_count=4)
    heldout_frame = cloud.vertices["source_frame"] == 0
    cloud.vertices["object_id"][heldout_frame] = 4

    result = train_objectstate_model_v0(
        cloud,
        heldout_stride=4,
        iterations=80,
        learning_rate=0.12,
        weight_decay=0.0,
        seed=0,
    )
    summary = result.as_dict()

    assert summary["split"]["train_object_ids"] == [0, 1]
    assert summary["split"]["heldout_object_ids"] == [0]
    assert summary["status"] == "objectstate_model_v0_training_reviewable"


def _frame_cloud(*, frame_count: int = 6) -> GaussianCloud:
    points_per_object = 4
    row_count = frame_count * 2 * points_per_object
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "f4"),
            ("green", "f4"),
            ("blue", "f4"),
            ("opacity", "f4"),
            ("object_id", "i4"),
            ("source_frame", "i4"),
        ]
    )
    vertices = np.zeros(row_count, dtype=dtype)
    cursor = 0
    for frame in range(frame_count):
        for object_id, center, color in (
            (4, -1.0, (1.0, 0.05, 0.0)),
            (9, 1.0, (0.0, 0.05, 1.0)),
        ):
            for point in range(points_per_object):
                vertices["x"][cursor] = center + 0.03 * point + 0.01 * frame
                vertices["y"][cursor] = 0.02 * point
                vertices["z"][cursor] = 0.01 * frame
                vertices["red"][cursor] = color[0]
                vertices["green"][cursor] = color[1]
                vertices["blue"][cursor] = color[2]
                vertices["opacity"][cursor] = 1.0
                vertices["object_id"][cursor] = object_id
                vertices["source_frame"][cursor] = frame
                cursor += 1
    return GaussianCloud(vertices)
