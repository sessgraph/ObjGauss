"""Deprecated compatibility import for real-sample v2 segmentation quality."""

from objgauss.pipelines.real_sample_v2_segmentation_quality import (
    REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA,
    RealSampleV2SegmentationQualityReport,
    real_sample_v2_segmentation_quality_from_cloud,
    real_sample_v2_segmentation_quality_from_projected_cloud,
    real_sample_v2_segmentation_quality_from_purity_report,
    validate_real_sample_v2_segmentation_quality_summary,
)

__all__ = (
    "REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA",
    "RealSampleV2SegmentationQualityReport",
    "real_sample_v2_segmentation_quality_from_cloud",
    "real_sample_v2_segmentation_quality_from_purity_report",
    "real_sample_v2_segmentation_quality_from_projected_cloud",
    "validate_real_sample_v2_segmentation_quality_summary",
)
