"""Compatibility wrapper for comparison and promotion policy evaluation."""

from objgauss.evaluation.baseline_comparison import (
    DEVELOPMENT_STAGE_NOTICE,
    compare_baseline_candidates,
    render_comparison_markdown,
    write_comparison_markdown,
)

__all__ = [
    "DEVELOPMENT_STAGE_NOTICE",
    "compare_baseline_candidates",
    "render_comparison_markdown",
    "write_comparison_markdown",
]
