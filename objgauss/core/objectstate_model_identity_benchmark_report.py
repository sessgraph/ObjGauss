"""Deprecated compatibility import for the identity benchmark report."""

from objgauss.evaluation.objectstate_model_identity_benchmark_report import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES,
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_SCHEMA,
    objectstate_model_identity_benchmark_report_difficulty_by_scenario,
    objectstate_model_identity_benchmark_report_scenarios,
    validate_objectstate_model_identity_benchmark_report_summary,
    write_objectstate_model_identity_benchmark_report,
)

__all__ = (
    "OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_SCHEMA",
    "OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES",
    "write_objectstate_model_identity_benchmark_report",
    "objectstate_model_identity_benchmark_report_scenarios",
    "objectstate_model_identity_benchmark_report_difficulty_by_scenario",
    "validate_objectstate_model_identity_benchmark_report_summary",
)
