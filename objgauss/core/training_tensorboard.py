"""Deprecated compatibility import for TensorBoard export orchestration."""

from objgauss.pipelines.training_tensorboard import (
    TENSORBOARD_SCALAR_EXPORT_SCHEMA,
    ScalarWriter,
    write_solver_decoder_tensorboard_events,
)

__all__ = (
    "TENSORBOARD_SCALAR_EXPORT_SCHEMA",
    "ScalarWriter",
    "write_solver_decoder_tensorboard_events",
)
