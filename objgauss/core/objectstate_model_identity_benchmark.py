"""Deprecated compatibility import for the model identity benchmark."""

from objgauss.evaluation.objectstate_model_identity_benchmark import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS,
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA,
    ObjectStateModelIdentityBenchmarkScenario,
    ObjectStateModelIdentityBenchmarkThresholds,
    objectstate_model_identity_benchmark_summary,
    validate_objectstate_model_identity_benchmark_summary,
)

__all__ = (
    "OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA",
    "OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS",
    "ObjectStateModelIdentityBenchmarkThresholds",
    "ObjectStateModelIdentityBenchmarkScenario",
    "objectstate_model_identity_benchmark_summary",
    "validate_objectstate_model_identity_benchmark_summary",
)
