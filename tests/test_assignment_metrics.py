from __future__ import annotations

import numpy as np

from objgauss.core.assignment_metrics import (
    assignment_clustering_metrics,
    instance_segmentation_metrics,
)


def test_assignment_clustering_metrics_are_label_permutation_invariant():
    target = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    predicted = np.asarray([2, 2, 0, 0, 1, 1], dtype=np.int64)

    metrics = assignment_clustering_metrics(predicted, target)

    assert metrics["ari"] == 1.0
    assert metrics["mean_best_iou"] == 1.0
    assert metrics["purity"] == 1.0


def test_instance_metrics_use_exact_permutation_invariant_matching():
    target = np.asarray([0, 0, 1, 1, 2, 2], dtype=np.int64)
    predicted = np.asarray([7, 7, 4, 4, 9, 9], dtype=np.int64)

    metrics = instance_segmentation_metrics(predicted, target)

    assert metrics["hungarian_mean_iou"] == 1.0
    assert metrics["object_recall_iou_0_5"] == 1.0
    assert metrics["object_recall_iou_0_75"] == 1.0
    assert metrics["object_count_error"] == 0
    assert metrics["merge_rate"] == 0.0
    assert metrics["split_rate"] == 0.0
    assert {(row["target_id"], row["predicted_id"]) for row in metrics["matching"]} == {
        (0, 7),
        (1, 4),
        (2, 9),
    }


def test_instance_metrics_expose_merges_splits_and_missing_objects():
    target = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
    predicted = np.asarray([5, 5, 6, 6, 5, 5, 5, 5], dtype=np.int64)

    metrics = instance_segmentation_metrics(predicted, target)

    assert metrics["object_count_error"] == 0
    assert metrics["merge_count"] == 1
    assert metrics["merge_rate"] == 0.5
    assert metrics["split_count"] == 1
    assert metrics["split_rate"] == 0.5
    assert metrics["hungarian_mean_iou"] < 1.0
