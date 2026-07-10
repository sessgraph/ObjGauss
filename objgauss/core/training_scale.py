"""Deprecated compatibility import for solver-decoder scale planning."""

from objgauss.pipelines.training_scale import (
    TRAINING_SCALE_PLAN_SCHEMA,
    solver_decoder_training_scale_plan,
    validate_solver_decoder_training_scale_plan,
)

__all__ = (
    "TRAINING_SCALE_PLAN_SCHEMA",
    "solver_decoder_training_scale_plan",
    "validate_solver_decoder_training_scale_plan",
)
