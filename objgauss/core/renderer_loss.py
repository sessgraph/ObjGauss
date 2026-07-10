"""Deprecated compatibility import for renderer-loss boundary reporting."""

from objgauss.pipelines.renderer_loss import (
    RENDERER_LOSS_BOUNDARY_SCHEMA,
    RendererLossBoundaryReport,
    renderer_loss_boundary_report,
    validate_renderer_loss_boundary_summary,
)

__all__ = (
    "RENDERER_LOSS_BOUNDARY_SCHEMA",
    "RendererLossBoundaryReport",
    "renderer_loss_boundary_report",
    "validate_renderer_loss_boundary_summary",
)
