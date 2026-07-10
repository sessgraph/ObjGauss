"""Evaluation and promotion-policy entry points."""

from objgauss.evaluation.baseline_comparison import (
    compare_baseline_candidates,
    render_comparison_markdown,
    write_comparison_markdown,
)
from objgauss.evaluation.emergence import (
    adjusted_rand_index,
    mask_proxy_occlusion_delta,
    object_emergence_curve,
    object_emergence_metrics,
    write_emergence_curve_csv,
)

__all__ = [
    "adjusted_rand_index",
    "compare_baseline_candidates",
    "mask_proxy_occlusion_delta",
    "object_emergence_curve",
    "object_emergence_metrics",
    "render_comparison_markdown",
    "write_comparison_markdown",
    "write_emergence_curve_csv",
]
