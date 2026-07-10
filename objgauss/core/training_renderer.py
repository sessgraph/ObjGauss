"""Deprecated compatibility import for the CPU training renderer."""

from objgauss.pipelines.training_renderer import (
    CPU_IMAGE_SPLAT_GRADIENT_PATH,
    CPU_IMAGE_SPLAT_RENDERER,
    TRAINING_RENDERER_API_SCHEMA,
    TrainingRendererFrameLoss,
    TrainingRendererLossResult,
    evaluate_training_renderer_loss,
    validate_training_renderer_summary,
)

__all__ = (
    "TRAINING_RENDERER_API_SCHEMA",
    "CPU_IMAGE_SPLAT_RENDERER",
    "CPU_IMAGE_SPLAT_GRADIENT_PATH",
    "TrainingRendererFrameLoss",
    "TrainingRendererLossResult",
    "evaluate_training_renderer_loss",
    "validate_training_renderer_summary",
)
