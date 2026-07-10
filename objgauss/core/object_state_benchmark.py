"""Deprecated compatibility import for ObjectState stability benchmarking."""

from objgauss.evaluation.object_state_benchmark import (
    DEFAULT_OBJECT_STATE_BENCHMARK_THRESHOLDS,
    OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA,
    object_state_stability_benchmark,
    validate_object_state_stability_benchmark,
    write_object_state_stability_benchmark,
)

__all__ = (
    "OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA",
    "DEFAULT_OBJECT_STATE_BENCHMARK_THRESHOLDS",
    "object_state_stability_benchmark",
    "write_object_state_stability_benchmark",
    "validate_object_state_stability_benchmark",
)
