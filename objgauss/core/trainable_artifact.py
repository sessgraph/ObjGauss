"""Deprecated compatibility import for trainable model artifacts."""

from objgauss.pipelines.trainable_artifact import (
    TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA,
    trainable_kernel_model_artifact,
    validate_trainable_kernel_model_artifact,
    write_trainable_kernel_model_artifact,
)

__all__ = (
    "TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA",
    "trainable_kernel_model_artifact",
    "write_trainable_kernel_model_artifact",
    "validate_trainable_kernel_model_artifact",
)
