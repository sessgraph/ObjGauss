"""Deprecated compatibility import for bounded-normalization cross-sample runs."""

from objgauss.pipelines.real_sample_v2_bounded_normalization_cross_sample import (
    REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA,
    RealSampleV2BoundedNormalizationCrossSampleInput,
    RealSampleV2BoundedNormalizationCrossSampleReport,
    real_sample_v2_bounded_normalization_cross_sample_from_clouds,
    validate_real_sample_v2_bounded_normalization_cross_sample_summary,
)

__all__ = (
    "REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA",
    "RealSampleV2BoundedNormalizationCrossSampleInput",
    "RealSampleV2BoundedNormalizationCrossSampleReport",
    "real_sample_v2_bounded_normalization_cross_sample_from_clouds",
    "validate_real_sample_v2_bounded_normalization_cross_sample_summary",
)
