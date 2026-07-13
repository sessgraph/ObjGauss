from __future__ import annotations

from typing import Any

import numpy as np

__all__ = (
    "assignment_clustering_metrics",
    "cluster_purity",
    "instance_segmentation_metrics",
    "mean_best_iou",
)


def assignment_clustering_metrics(
    predicted_labels: np.ndarray,
    target_labels: np.ndarray,
) -> dict[str, Any]:
    predicted, target = _label_pair(predicted_labels, target_labels)
    return {
        "target_slots": int(np.unique(target).shape[0]),
        "mean_best_iou": mean_best_iou(predicted, target),
        "ari": _adjusted_rand_index(predicted, target),
        "purity": cluster_purity(predicted, target),
    }


def mean_best_iou(predicted_labels: np.ndarray, target_labels: np.ndarray) -> float:
    predicted, target = _label_pair(predicted_labels, target_labels)
    target_slots = np.unique(target)
    predicted_slots = np.unique(predicted)
    if target_slots.size == 0 or predicted_slots.size == 0:
        return 0.0
    scores: list[float] = []
    for target_slot in target_slots:
        target_mask = target == target_slot
        best = 0.0
        for predicted_slot in predicted_slots:
            predicted_mask = predicted == predicted_slot
            union = int(np.logical_or(target_mask, predicted_mask).sum())
            if union:
                best = max(
                    best,
                    float(np.logical_and(target_mask, predicted_mask).sum() / union),
                )
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def cluster_purity(predicted_labels: np.ndarray, target_labels: np.ndarray) -> float:
    predicted, target = _label_pair(predicted_labels, target_labels)
    if predicted.size == 0:
        return 0.0
    correct = 0
    for predicted_slot in np.unique(predicted):
        values, counts = np.unique(target[predicted == predicted_slot], return_counts=True)
        if values.size:
            correct += int(np.max(counts))
    return float(correct / predicted.shape[0])


def instance_segmentation_metrics(
    predicted_labels: np.ndarray,
    target_labels: np.ndarray,
    *,
    material_fraction: float = 0.1,
) -> dict[str, Any]:
    """Return permutation-invariant instance metrics with exact IoU matching."""

    predicted, target = _label_pair(predicted_labels, target_labels)
    if not 0.0 < material_fraction <= 0.5:
        raise ValueError("material_fraction must be in (0, 0.5]")
    predicted_ids = np.unique(predicted)
    target_ids = np.unique(target)
    iou = _iou_matrix(predicted, target, predicted_ids, target_ids)
    target_to_prediction = _maximum_iou_matching(iou)
    matched_ious = np.zeros(target_ids.shape[0], dtype=np.float64)
    matching: list[dict[str, int | float]] = []
    for target_index, predicted_index in enumerate(target_to_prediction):
        if predicted_index < 0 or predicted_index >= predicted_ids.shape[0]:
            continue
        score = float(iou[predicted_index, target_index])
        matched_ious[target_index] = score
        matching.append(
            {
                "target_id": int(target_ids[target_index]),
                "predicted_id": int(predicted_ids[predicted_index]),
                "iou": score,
            }
        )

    merge_count = 0
    for predicted_id in predicted_ids:
        mask = predicted == predicted_id
        overlaps = [
            int(np.logical_and(mask, target == target_id).sum()) / int(mask.sum())
            for target_id in target_ids
        ]
        if sum(value >= material_fraction for value in overlaps) > 1:
            merge_count += 1

    split_count = 0
    for target_id in target_ids:
        mask = target == target_id
        overlaps = [
            int(np.logical_and(mask, predicted == predicted_id).sum()) / int(mask.sum())
            for predicted_id in predicted_ids
        ]
        if sum(value >= material_fraction for value in overlaps) > 1:
            split_count += 1

    return {
        **assignment_clustering_metrics(predicted, target),
        "predicted_object_count": int(predicted_ids.shape[0]),
        "target_object_count": int(target_ids.shape[0]),
        "object_count_error": abs(int(predicted_ids.shape[0] - target_ids.shape[0])),
        "hungarian_mean_iou": float(np.mean(matched_ious)) if matched_ious.size else 0.0,
        "object_recall_iou_0_5": float(np.mean(matched_ious >= 0.5)) if matched_ious.size else 0.0,
        "object_recall_iou_0_75": float(np.mean(matched_ious >= 0.75)) if matched_ious.size else 0.0,
        "merge_count": merge_count,
        "merge_rate": float(merge_count / predicted_ids.shape[0]) if predicted_ids.size else 0.0,
        "split_count": split_count,
        "split_rate": float(split_count / target_ids.shape[0]) if target_ids.size else 0.0,
        "material_fraction": float(material_fraction),
        "matching": matching,
    }


def _label_pair(
    predicted_labels: np.ndarray,
    target_labels: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted_labels)
    target = np.asarray(target_labels)
    if predicted.ndim != 1 or target.ndim != 1:
        raise ValueError("assignment metric labels must be 1D")
    if predicted.shape != target.shape:
        raise ValueError("assignment metric labels must have the same shape")
    if not np.all(np.isfinite(predicted)) or not np.all(np.isfinite(target)):
        raise ValueError("assignment metric labels must be finite")
    if not np.allclose(predicted, np.rint(predicted)) or not np.allclose(target, np.rint(target)):
        raise ValueError("assignment metric labels must be integers")
    return (
        np.rint(predicted).astype(np.int64, copy=False),
        np.rint(target).astype(np.int64, copy=False),
    )


def _iou_matrix(
    predicted: np.ndarray,
    target: np.ndarray,
    predicted_ids: np.ndarray,
    target_ids: np.ndarray,
) -> np.ndarray:
    scores = np.zeros((predicted_ids.shape[0], target_ids.shape[0]), dtype=np.float64)
    for predicted_index, predicted_id in enumerate(predicted_ids):
        predicted_mask = predicted == predicted_id
        for target_index, target_id in enumerate(target_ids):
            target_mask = target == target_id
            union = int(np.logical_or(predicted_mask, target_mask).sum())
            if union:
                scores[predicted_index, target_index] = float(
                    np.logical_and(predicted_mask, target_mask).sum() / union
                )
    return scores


def _maximum_iou_matching(iou: np.ndarray) -> np.ndarray:
    """Map each target column to one prediction row using Hungarian matching."""

    scores = np.asarray(iou, dtype=np.float64)
    if scores.ndim != 2:
        raise ValueError("IoU matrix must be 2D")
    predicted_count, target_count = scores.shape
    if target_count == 0:
        return np.empty(0, dtype=np.int64)
    width = max(predicted_count, target_count)
    cost = np.ones((target_count, width), dtype=np.float64)
    if predicted_count:
        cost[:, :predicted_count] = 1.0 - scores.T

    # Rectangular Hungarian algorithm for rows <= columns.  Dummy prediction
    # columns carry IoU zero and therefore represent unmatched GT instances.
    u = np.zeros(target_count + 1, dtype=np.float64)
    v = np.zeros(width + 1, dtype=np.float64)
    p = np.zeros(width + 1, dtype=np.int64)
    way = np.zeros(width + 1, dtype=np.int64)
    for row in range(1, target_count + 1):
        p[0] = row
        column0 = 0
        minimum = np.full(width + 1, np.inf, dtype=np.float64)
        used = np.zeros(width + 1, dtype=bool)
        while True:
            used[column0] = True
            row0 = int(p[column0])
            delta = np.inf
            column1 = 0
            for column in range(1, width + 1):
                if used[column]:
                    continue
                current = cost[row0 - 1, column - 1] - u[row0] - v[column]
                if current < minimum[column]:
                    minimum[column] = current
                    way[column] = column0
                if minimum[column] < delta:
                    delta = minimum[column]
                    column1 = column
            for column in range(width + 1):
                if used[column]:
                    u[p[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column0 = column1
            if p[column0] == 0:
                break
        while True:
            column1 = int(way[column0])
            p[column0] = p[column1]
            column0 = column1
            if column0 == 0:
                break

    target_to_prediction = np.full(target_count, -1, dtype=np.int64)
    for column in range(1, width + 1):
        if p[column] and column <= predicted_count:
            target_to_prediction[p[column] - 1] = column - 1
    return target_to_prediction


def _adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    if labels_a.size < 2:
        return 1.0
    _, rows = np.unique(labels_a, return_inverse=True)
    _, cols = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((int(rows.max()) + 1, int(cols.max()) + 1), dtype=np.int64)
    np.add.at(contingency, (rows, cols), 1)
    sum_comb = float(_comb2(contingency).sum())
    row_comb = float(_comb2(contingency.sum(axis=1)).sum())
    col_comb = float(_comb2(contingency.sum(axis=0)).sum())
    total_comb = float(_comb2(np.asarray([labels_a.size], dtype=np.int64))[0])
    if total_comb <= 0.0:
        return 1.0
    expected = row_comb * col_comb / total_comb
    maximum = 0.5 * (row_comb + col_comb)
    denominator = maximum - expected
    if abs(denominator) <= 1e-12:
        return 1.0 if abs(sum_comb - maximum) <= 1e-12 else 0.0
    return float((sum_comb - expected) / denominator)


def _comb2(values: np.ndarray) -> np.ndarray:
    resolved = values.astype(np.float64, copy=False)
    return resolved * (resolved - 1.0) / 2.0
