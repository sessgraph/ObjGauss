from __future__ import annotations

from typing import Any

import numpy as np

from objgauss.clustering import summarize_labels
from objgauss.object_field import ObjectField

_EPS = 1e-8


def object_emergence_metrics(
    field: ObjectField,
    *,
    positions_xyz: np.ndarray | None = None,
    reference: ObjectField | None = None,
) -> dict[str, Any]:
    """Compute conservative object-emergence observability metrics.

    This intentionally reports a partial Object Emergence Score. Current ObjGauss
    can measure assignment, spatial compactness, and reference stability; true
    occlusion/render-delta and gradient-coherence probes are left explicit.
    """

    probabilities = field.probabilities()
    labels = field.labels()
    assignment = _assignment_metrics(probabilities, labels)
    spatial = _spatial_metrics(labels, positions_xyz, field.slots) if positions_xyz is not None else None
    stability = _stability_metrics(reference, field) if reference is not None else None
    score = _object_emergence_score(
        assignment_confidence=assignment["assignment_confidence"],
        spatial_compactness_score=spatial["compactness_score"] if spatial else None,
        stability_score=stability["stability_score"] if stability else None,
    )
    return {
        "gaussians": field.gaussian_count,
        "slots": field.slots,
        "assignment": assignment,
        "spatial": spatial,
        "stability": stability,
        "object_emergence_score": score,
    }


def adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    labels_a = np.asarray(labels_a, dtype=np.int64)
    labels_b = np.asarray(labels_b, dtype=np.int64)
    if labels_a.ndim != 1 or labels_b.ndim != 1:
        raise ValueError("labels must be 1D arrays")
    if labels_a.shape != labels_b.shape:
        raise ValueError("label arrays must have the same shape")
    if labels_a.size < 2:
        return 1.0

    _, rows = np.unique(labels_a, return_inverse=True)
    _, cols = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((int(rows.max()) + 1, int(cols.max()) + 1), dtype=np.int64)
    np.add.at(contingency, (rows, cols), 1)

    sum_comb = float(_comb2(contingency).sum())
    row_comb = float(_comb2(contingency.sum(axis=1)).sum())
    col_comb = float(_comb2(contingency.sum(axis=0)).sum())
    total_comb = float(_comb2(np.array([labels_a.size], dtype=np.int64))[0])
    if total_comb <= 0:
        return 1.0

    expected = row_comb * col_comb / total_comb
    max_index = 0.5 * (row_comb + col_comb)
    denominator = max_index - expected
    if abs(denominator) <= _EPS:
        return 1.0 if np.array_equal(labels_a, labels_b) else 0.0
    return float((sum_comb - expected) / denominator)


def _assignment_metrics(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    slots = probabilities.shape[1]
    entropy_per_gaussian = -np.sum(
        probabilities * np.log(np.clip(probabilities, _EPS, 1.0)),
        axis=1,
    )
    normalized_entropy = (
        np.zeros_like(entropy_per_gaussian)
        if slots == 1
        else entropy_per_gaussian / np.log(slots)
    )
    slot_mass = probabilities.mean(axis=0)
    mass_entropy = _entropy(slot_mass)
    hard_counts = np.bincount(labels, minlength=slots).astype(np.int64)
    return {
        "mean_entropy": float(np.mean(entropy_per_gaussian)),
        "mean_normalized_entropy": float(np.mean(normalized_entropy)),
        "assignment_confidence": float(1.0 - np.mean(normalized_entropy)),
        "low_entropy_fraction": _safe_fraction(np.count_nonzero(normalized_entropy <= 0.2), labels.size),
        "high_entropy_fraction": _safe_fraction(np.count_nonzero(normalized_entropy >= 0.8), labels.size),
        "effective_slots": float(np.exp(mass_entropy)),
        "slot_mass": [float(value) for value in slot_mass],
        "hard_slot_counts": [int(value) for value in hard_counts],
        "active_slots": len(summarize_labels(labels)),
    }


def _spatial_metrics(labels: np.ndarray, positions_xyz: np.ndarray, slots: int) -> dict[str, Any]:
    positions_xyz = np.asarray(positions_xyz, dtype=np.float32)
    if positions_xyz.ndim != 2 or positions_xyz.shape[1] != 3:
        raise ValueError("positions must be an Nx3 array")
    if positions_xyz.shape[0] != labels.shape[0]:
        raise ValueError("positions and labels must have the same row count")

    extent = positions_xyz.max(axis=0) - positions_xyz.min(axis=0)
    scene_radius_sq = float(np.sum(extent * extent))
    if scene_radius_sq <= _EPS:
        scene_radius_sq = 1.0

    per_slot: list[dict[str, Any]] = []
    weighted_normalized_radius = 0.0
    for slot in range(slots):
        selected = labels == slot
        count = int(np.count_nonzero(selected))
        if count == 0:
            per_slot.append(
                {
                    "slot": slot,
                    "gaussians": 0,
                    "mean_squared_radius": None,
                    "normalized_mean_squared_radius": None,
                }
            )
            continue
        points = positions_xyz[selected]
        centroid = points.mean(axis=0)
        mean_squared_radius = float(np.mean(np.sum((points - centroid) ** 2, axis=1)))
        normalized = float(mean_squared_radius / scene_radius_sq)
        weighted_normalized_radius += normalized * count
        per_slot.append(
            {
                "slot": slot,
                "gaussians": count,
                "mean_squared_radius": mean_squared_radius,
                "normalized_mean_squared_radius": normalized,
            }
        )

    overall = float(weighted_normalized_radius / max(labels.size, 1))
    return {
        "overall_normalized_compactness": overall,
        "compactness_score": float(1.0 - min(1.0, overall)),
        "per_slot": per_slot,
    }


def _stability_metrics(reference: ObjectField, current: ObjectField) -> dict[str, Any]:
    if reference.gaussian_count != current.gaussian_count:
        raise ValueError("reference and current fields must have the same Gaussian count")
    reference_labels = reference.labels()
    current_labels = current.labels()
    ari = adjusted_rand_index(reference_labels, current_labels)
    matching, matched = _greedy_slot_matching(reference_labels, current_labels, reference.slots, current.slots)
    return {
        "adjusted_rand_index": ari,
        "stability_score": float(max(0.0, min(1.0, ari))),
        "matched_label_agreement": _safe_fraction(matched, reference.gaussian_count),
        "slot_matching": matching,
    }


def _greedy_slot_matching(
    reference_labels: np.ndarray,
    current_labels: np.ndarray,
    reference_slots: int,
    current_slots: int,
) -> tuple[list[dict[str, int]], int]:
    confusion = np.zeros((reference_slots, current_slots), dtype=np.int64)
    np.add.at(confusion, (reference_labels, current_labels), 1)
    candidates = [
        (int(confusion[row, col]), row, col)
        for row in range(reference_slots)
        for col in range(current_slots)
        if confusion[row, col] > 0
    ]
    candidates.sort(reverse=True)
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    matching: list[dict[str, int]] = []
    matched = 0
    for overlap, row, col in candidates:
        if row in used_rows or col in used_cols:
            continue
        used_rows.add(row)
        used_cols.add(col)
        matched += overlap
        matching.append(
            {
                "reference_slot": row,
                "current_slot": col,
                "overlap_gaussians": overlap,
            }
        )
    matching.sort(key=lambda item: item["reference_slot"])
    return matching, matched


def _object_emergence_score(
    *,
    assignment_confidence: float,
    spatial_compactness_score: float | None,
    stability_score: float | None,
) -> dict[str, Any]:
    weights = {
        "assignment_confidence": 0.25,
        "stability": 0.35,
        "spatial_compactness": 0.20,
        "occlusion_effect": 0.20,
    }
    components: dict[str, float | None] = {
        "assignment_confidence": float(max(0.0, min(1.0, assignment_confidence))),
        "stability": stability_score,
        "spatial_compactness": spatial_compactness_score,
        "occlusion_effect": None,
    }
    missing = [name for name, value in components.items() if value is None]
    available = {name: value for name, value in components.items() if value is not None}
    weight_sum = sum(weights[name] for name in available)
    score = None
    if weight_sum > 0:
        score = float(sum(weights[name] * float(value) for name, value in available.items()) / weight_sum)
    return {
        "score": score,
        "complete": len(missing) == 0,
        "components": components,
        "weights": weights,
        "missing_components": missing,
        "unsupported_components": ["gradient_coherence"],
    }


def _entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=np.float32)
    return float(-np.sum(values * np.log(np.clip(values, _EPS, 1.0))))


def _comb2(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values * (values - 1.0) / 2.0


def _safe_fraction(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0
