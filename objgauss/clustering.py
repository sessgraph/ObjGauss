"""Compatibility wrapper for core baseline clustering."""

from objgauss.core.clustering import ClusteringResult, cluster_features, summarize_labels

__all__ = [
    "ClusteringResult",
    "cluster_features",
    "summarize_labels",
]
