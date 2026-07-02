from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ClusteringResult:
    labels: np.ndarray
    centers: np.ndarray
    inertia: float
    backend: str


def cluster_features(
    features: np.ndarray,
    *,
    clusters: int,
    seed: int = 0,
    max_iter: int = 100,
) -> ClusteringResult:
    """Cluster Gaussian features with sklearn when available, else NumPy."""

    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2:
        raise ValueError("features must be a 2D array")
    if features.shape[0] == 0:
        raise ValueError("cannot cluster an empty Gaussian cloud")
    if clusters < 1:
        raise ValueError("clusters must be >= 1")
    if clusters > features.shape[0]:
        raise ValueError("clusters cannot exceed the Gaussian count")

    sklearn_result = _cluster_with_sklearn(features, clusters, seed, max_iter)
    if sklearn_result is not None:
        return sklearn_result
    return _cluster_with_numpy(features, clusters, seed, max_iter)


def summarize_labels(labels: np.ndarray) -> list[tuple[int, int]]:
    labels = np.asarray(labels)
    unique, counts = np.unique(labels, return_counts=True)
    return [(int(label), int(count)) for label, count in zip(unique, counts)]


def _cluster_with_sklearn(
    features: np.ndarray,
    clusters: int,
    seed: int,
    max_iter: int,
) -> ClusteringResult | None:
    try:
        from sklearn.cluster import KMeans
    except Exception:
        return None

    model = KMeans(
        n_clusters=clusters,
        random_state=seed,
        n_init=10,
        max_iter=max_iter,
    )
    labels = model.fit_predict(features).astype(np.int32, copy=False)
    return ClusteringResult(
        labels=labels,
        centers=model.cluster_centers_.astype(np.float32, copy=False),
        inertia=float(model.inertia_),
        backend="sklearn",
    )


def _cluster_with_numpy(
    features: np.ndarray,
    clusters: int,
    seed: int,
    max_iter: int,
) -> ClusteringResult:
    rng = np.random.default_rng(seed)
    centers = _init_kmeans_plus_plus(features, clusters, rng)
    labels = np.zeros(features.shape[0], dtype=np.int32)

    for _ in range(max_iter):
        distances = _squared_distances(features, centers)
        next_labels = np.argmin(distances, axis=1).astype(np.int32, copy=False)
        if np.array_equal(labels, next_labels):
            break
        labels = next_labels

        for index in range(clusters):
            members = features[labels == index]
            if members.size:
                centers[index] = members.mean(axis=0)
            else:
                farthest = int(np.argmax(np.min(distances, axis=1)))
                centers[index] = features[farthest]
                labels[farthest] = index

    final_distances = _squared_distances(features, centers)
    labels = np.argmin(final_distances, axis=1).astype(np.int32, copy=False)
    inertia = float(final_distances[np.arange(features.shape[0]), labels].sum())
    return ClusteringResult(labels=labels, centers=centers, inertia=inertia, backend="numpy")


def _init_kmeans_plus_plus(
    features: np.ndarray,
    clusters: int,
    rng: np.random.Generator,
) -> np.ndarray:
    centers = np.empty((clusters, features.shape[1]), dtype=np.float32)
    first = int(rng.integers(0, features.shape[0]))
    centers[0] = features[first]

    closest = _squared_distances(features, centers[:1]).reshape(-1)
    for index in range(1, clusters):
        total = float(closest.sum())
        if total <= 1e-12:
            choice = int(rng.integers(0, features.shape[0]))
        else:
            choice = int(rng.choice(features.shape[0], p=closest / total))
        centers[index] = features[choice]
        closest = np.minimum(closest, _squared_distances(features, centers[index : index + 1]).reshape(-1))
    return centers


def _squared_distances(values: np.ndarray, centers: np.ndarray) -> np.ndarray:
    diff = values[:, None, :] - centers[None, :, :]
    return np.sum(diff * diff, axis=2)
