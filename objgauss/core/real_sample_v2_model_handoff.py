"""Deprecated compatibility import for the real-sample v2 model handoff."""

from objgauss.pipelines.real_sample_v2_model_handoff import (
    REAL_SAMPLE_V2_EFFECT_PREVIEW_SCHEMA,
    REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA,
    RealSampleV2ModelHandoffReport,
    evaluate_real_sample_v2_model_handoff,
    real_sample_v2_model_handoff_from_cloud,
    render_real_sample_v2_model_handoff_html,
    validate_real_sample_v2_effect_preview,
    validate_real_sample_v2_model_handoff_summary,
)

__all__ = (
    "REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA",
    "REAL_SAMPLE_V2_EFFECT_PREVIEW_SCHEMA",
    "RealSampleV2ModelHandoffReport",
    "real_sample_v2_model_handoff_from_cloud",
    "evaluate_real_sample_v2_model_handoff",
    "validate_real_sample_v2_model_handoff_summary",
    "validate_real_sample_v2_effect_preview",
    "render_real_sample_v2_model_handoff_html",
)
