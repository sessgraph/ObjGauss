"""Compatibility wrapper for core comparison and promotion policy algorithms."""

from objgauss.core.baseline_comparison import (
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
