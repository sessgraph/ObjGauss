"""Deprecated compatibility import for the identity prediction adapter."""

from objgauss.pipelines.objectstate_identity_prediction_adapter import (
    objectstate_identity_predictions_from_trainable_artifact,
    read_trainable_kernel_identity_source,
)

__all__ = (
    "objectstate_identity_predictions_from_trainable_artifact",
    "read_trainable_kernel_identity_source",
)
