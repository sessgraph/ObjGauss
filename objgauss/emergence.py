"""Compatibility wrapper for object-emergence evaluation."""

from objgauss.evaluation.emergence import (
    adjusted_rand_index,
    mask_proxy_occlusion_delta,
    object_emergence_curve,
    object_emergence_metrics,
    write_emergence_curve_csv,
)

__all__ = [
    "adjusted_rand_index",
    "mask_proxy_occlusion_delta",
    "object_emergence_curve",
    "object_emergence_metrics",
    "write_emergence_curve_csv",
]
