"""Deprecated compatibility import for the gsplat training renderer."""

from objgauss.pipelines.gsplat_training_renderer import (
    GSPLAT_AVAILABILITY_SCHEMA,
    GSPLAT_GRADIENT_PATH,
    GSPLAT_RENDERER,
    GSPLAT_SYNTHETIC_GAUSSIAN_POLICY,
    GSPLAT_TRAINING_INPUT_SCHEMA,
    GsplatRendererAvailability,
    GsplatTrainingInput,
    build_gsplat_training_input,
    build_gsplat_training_input_from_object_state,
    evaluate_gsplat_training_renderer_loss,
    gsplat_renderer_availability,
)

__all__ = (
    "GSPLAT_RENDERER",
    "GSPLAT_GRADIENT_PATH",
    "GSPLAT_AVAILABILITY_SCHEMA",
    "GSPLAT_TRAINING_INPUT_SCHEMA",
    "GSPLAT_SYNTHETIC_GAUSSIAN_POLICY",
    "GsplatRendererAvailability",
    "GsplatTrainingInput",
    "gsplat_renderer_availability",
    "build_gsplat_training_input",
    "build_gsplat_training_input_from_object_state",
    "evaluate_gsplat_training_renderer_loss",
)
