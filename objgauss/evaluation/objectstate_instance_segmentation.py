from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from objgauss.core.assignment_metrics import instance_segmentation_metrics
from objgauss.core.clustering import cluster_features
from objgauss.core.features import colors, positions
from objgauss.datasets.objectstate_multi_object_synthetic import (
    MultiObjectSyntheticDataset,
)

OBJECTSTATE_MULTI_OBJECT_BENCHMARK_SCHEMA = (
    "objgauss-objectstate-multi-object-benchmark-v1"
)

_CANDIDATES = (
    "xyz_kmeans",
    "rgb_kmeans",
    "connected_components_3d",
    "objectstate_model_v0",
)
_AGGREGATE_METRICS = (
    "ari",
    "hungarian_mean_iou",
    "object_recall_iou_0_5",
    "object_recall_iou_0_75",
    "object_count_error",
    "merge_rate",
    "split_rate",
)

__all__ = (
    "OBJECTSTATE_MULTI_OBJECT_BENCHMARK_SCHEMA",
    "connected_component_labels_3d",
    "evaluate_multi_object_instance_benchmark",
    "normalized_features",
)


def evaluate_multi_object_instance_benchmark(
    dataset: MultiObjectSyntheticDataset,
    model_assignment: np.ndarray,
    *,
    connected_component_radius: float = 0.18,
    seed: int = 0,
) -> dict[str, Any]:
    cloud = dataset.cloud
    assignment = np.asarray(model_assignment, dtype=np.float32)
    if assignment.ndim != 2 or assignment.shape[0] != cloud.count:
        raise ValueError("model_assignment must have one row per Gaussian")
    if not np.all(np.isfinite(assignment)):
        raise ValueError("model_assignment must be finite")
    if connected_component_radius <= 0.0:
        raise ValueError("connected_component_radius must be > 0")

    leakage_gate = _leakage_gate(dataset)
    scene_rows: list[dict[str, Any]] = []
    all_positions = positions(cloud)
    all_colors = colors(cloud)
    scene_values = cloud.vertices["scene_id"].astype(np.int64, copy=False)
    target_values = cloud.vertices["gt_instance_id"].astype(np.int64, copy=False)
    model_labels = np.argmax(assignment, axis=1).astype(np.int64, copy=False)
    scene_by_id = {int(scene["scene_id"]): scene for scene in dataset.scenes}

    for scene_id in dataset.heldout_scene_ids:
        mask = scene_values == scene_id
        xyz = all_positions[mask]
        rgb = all_colors[mask]
        target = target_values[mask]
        target_count = int(np.unique(target).shape[0])
        candidates = {
            "xyz_kmeans": cluster_features(
                normalized_features(xyz),
                clusters=target_count,
                seed=seed + scene_id,
            ).labels,
            "rgb_kmeans": cluster_features(
                normalized_features(rgb),
                clusters=target_count,
                seed=seed + scene_id,
            ).labels,
            "connected_components_3d": connected_component_labels_3d(
                xyz,
                radius=connected_component_radius,
            ),
            "objectstate_model_v0": model_labels[mask],
        }
        evaluations = {
            name: instance_segmentation_metrics(labels, target)
            for name, labels in candidates.items()
        }
        scene_rows.append(
            {
                "scene_id": int(scene_id),
                "gaussian_count": int(mask.sum()),
                "target_object_count": target_count,
                "stress": {
                    "contact": bool(scene_by_id[scene_id]["contact"]),
                    "partial_observation": bool(
                        scene_by_id[scene_id]["partial_observation"]
                    ),
                    "same_color_instance_ids": [0, 1],
                },
                "candidates": evaluations,
            }
        )

    aggregate = {
        candidate: _aggregate_candidate(scene_rows, candidate)
        for candidate in _CANDIDATES
    }
    model_score = aggregate["objectstate_model_v0"]["hungarian_mean_iou"]
    baseline_scores = {
        name: aggregate[name]["hungarian_mean_iou"]
        for name in _CANDIDATES
        if name != "objectstate_model_v0"
    }
    best_baseline = max(baseline_scores, key=baseline_scores.get)
    best_baseline_score = baseline_scores[best_baseline]
    delta = float(model_score - best_baseline_score)
    if not leakage_gate["passed"]:
        verdict = "invalid_leakage_gate"
    elif delta > 1e-6:
        verdict = "model_better_than_recorded_baselines"
    elif abs(delta) <= 1e-6:
        verdict = "model_tied_best_recorded_baseline"
    else:
        verdict = "model_not_better_than_recorded_baselines"

    return {
        "schema": OBJECTSTATE_MULTI_OBJECT_BENCHMARK_SCHEMA,
        "kind": "multi_object_instance_segmentation_benchmark",
        "status": "reviewable" if leakage_gate["passed"] else "invalid",
        "dataset_schema": dataset.schema,
        "target": {
            "field": "gt_instance_id",
            "source": "procedural_instance_authorship",
            "derived_from_rgb": False,
        },
        "split": {
            "field": "scene_id",
            "policy": "deterministic_complete_scene_holdout",
            "train_scene_ids": list(dataset.train_scene_ids),
            "heldout_scene_ids": list(dataset.heldout_scene_ids),
            "scene_overlap_count": len(
                set(dataset.train_scene_ids) & set(dataset.heldout_scene_ids)
            ),
        },
        "candidate_contract": {
            "names": list(_CANDIDATES),
            "kmeans_uses_target_object_count": True,
            "connected_component_radius": float(connected_component_radius),
            "model_feature_fields": [
                "x",
                "y",
                "z",
                "red",
                "green",
                "blue",
                "opacity",
            ],
        },
        "metric_contract": {
            "matching": "maximum_iou_bipartite_hungarian",
            "unmatched_target_iou": 0.0,
            "material_fraction": 0.1,
            "metrics": list(_AGGREGATE_METRICS),
        },
        "leakage_gate": leakage_gate,
        "scenes": scene_rows,
        "aggregate": aggregate,
        "comparison": {
            "best_baseline": best_baseline,
            "best_baseline_hungarian_mean_iou": float(best_baseline_score),
            "model_hungarian_mean_iou": float(model_score),
            "model_delta": delta,
            "verdict": verdict,
        },
        "claim_policy": {
            "synthetic_instance_segmentation_only": True,
            "failed_result_is_valid_evidence": True,
            "does_not_claim_real_identity": True,
            "does_not_claim_prediction_or_intervention": True,
        },
    }


def _leakage_gate(dataset: MultiObjectSyntheticDataset) -> dict[str, Any]:
    manifest = dataset.as_dict()
    source = manifest["source"]
    stress = manifest["stress_contract"]
    heldout = set(dataset.heldout_scene_ids)
    scene_by_id = {int(scene["scene_id"]): scene for scene in dataset.scenes}
    checks = {
        "complete_scene_split": not bool(
            set(dataset.train_scene_ids) & set(dataset.heldout_scene_ids)
        ),
        "independent_instance_target": (
            source["type"] == "procedural_instance_authorship"
            and source["target_derived_from_rgb"] is False
        ),
        "target_excluded_from_model_features": (
            "gt_instance_id" not in source["model_feature_fields"]
        ),
        "same_color_pair_present": stress["same_color_instance_ids"] == [0, 1],
        "heldout_contact_present": any(
            scene_by_id[scene_id]["contact"] for scene_id in heldout
        ),
        "heldout_partial_observation_present": any(
            scene_by_id[scene_id]["partial_observation"] for scene_id in heldout
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": [
            {"name": name, "passed": bool(passed)}
            for name, passed in checks.items()
        ],
    }


def _aggregate_candidate(
    scene_rows: list[dict[str, Any]],
    candidate: str,
) -> dict[str, float | int]:
    metrics = [row["candidates"][candidate] for row in scene_rows]
    return {
        "scene_count": len(metrics),
        **{
            name: float(np.mean([float(item[name]) for item in metrics]))
            for name in _AGGREGATE_METRICS
        },
    }


def normalized_features(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    mean = array.mean(axis=0, keepdims=True)
    std = array.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return ((array - mean) / std).astype(np.float32, copy=False)


def connected_component_labels_3d(
    xyz: np.ndarray,
    *,
    radius: float,
) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float32)
    count = points.shape[0]
    parent = np.arange(count, dtype=np.int64)

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    radius_squared = float(radius * radius)
    for left in range(count):
        distances = np.sum((points[left + 1 :] - points[left]) ** 2, axis=1)
        for offset in np.flatnonzero(distances <= radius_squared):
            union(left, left + 1 + int(offset))

    roots = np.asarray([find(index) for index in range(count)], dtype=np.int64)
    unique_roots, labels = np.unique(roots, return_inverse=True)
    counts = np.bincount(labels)
    minimum_component = max(4, int(np.ceil(count * 0.02)))
    large = np.flatnonzero(counts >= minimum_component)
    if large.size == 0:
        large = np.asarray([int(np.argmax(counts))], dtype=np.int64)
    centroids = np.asarray(
        [points[labels == component].mean(axis=0) for component in large],
        dtype=np.float32,
    )
    remapped = np.empty(count, dtype=np.int64)
    large_lookup = {int(component): index for index, component in enumerate(large)}
    for component in range(unique_roots.shape[0]):
        mask = labels == component
        if component in large_lookup:
            remapped[mask] = large_lookup[component]
        else:
            center = points[mask].mean(axis=0)
            nearest = int(np.argmin(np.sum((centroids - center) ** 2, axis=1)))
            remapped[mask] = nearest
    return remapped
