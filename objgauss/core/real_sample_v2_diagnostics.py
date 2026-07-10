"""Deprecated compatibility import for real-sample v2 diagnostics."""

from objgauss.pipelines.real_sample_v2_diagnostics import (
    REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA,
    RealSampleV2DiagnosticsReport,
    evaluate_real_sample_v2_diagnostics,
    real_sample_v2_diagnostics_from_cloud,
    validate_real_sample_v2_diagnostics_summary,
)

__all__ = (
    "REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA",
    "RealSampleV2DiagnosticsReport",
    "real_sample_v2_diagnostics_from_cloud",
    "evaluate_real_sample_v2_diagnostics",
    "validate_real_sample_v2_diagnostics_summary",
)
