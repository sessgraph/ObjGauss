"""Deprecated compatibility import for real-sample v2 viewer preview."""

from objgauss.pipelines.real_sample_v2_viewer_preview import (
    REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT,
    REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT,
    REAL_SAMPLE_V2_VIEWER_PREVIEW_SCHEMA,
    RealSampleV2ViewerPreviewReport,
    real_sample_v2_viewer_preview_from_cloud,
    real_sample_v2_viewer_preview_from_handoff,
    validate_real_sample_v2_viewer_preview_summary,
)

__all__ = (
    "REAL_SAMPLE_V2_VIEWER_PREVIEW_SCHEMA",
    "REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT",
    "REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT",
    "RealSampleV2ViewerPreviewReport",
    "real_sample_v2_viewer_preview_from_cloud",
    "real_sample_v2_viewer_preview_from_handoff",
    "validate_real_sample_v2_viewer_preview_summary",
)
