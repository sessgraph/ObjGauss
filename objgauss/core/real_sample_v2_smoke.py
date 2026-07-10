"""Deprecated compatibility import for the real-sample v2 smoke pipeline."""

from objgauss.pipelines.real_sample_v2_smoke import (
    REAL_SAMPLE_V2_SMOKE_SCHEMA,
    RealSampleV2SmokeReport,
    evaluate_real_sample_v2_smoke,
    real_sample_v2_smoke_from_cloud,
    validate_real_sample_v2_smoke_summary,
)

__all__ = (
    "REAL_SAMPLE_V2_SMOKE_SCHEMA",
    "RealSampleV2SmokeReport",
    "real_sample_v2_smoke_from_cloud",
    "evaluate_real_sample_v2_smoke",
    "validate_real_sample_v2_smoke_summary",
)
