"""Deprecated compatibility import for CLIP mask-scoring orchestration."""

from objgauss.pipelines.clip_scoring import (
    CLIP_LABEL_PRESETS,
    DEFAULT_BACKGROUND_LABELS,
    DEFAULT_PROMPT_TEMPLATES,
    ClipMaskScorer,
    ClipScoringResult,
    HashClipMaskScorer,
    TransformersClipMaskScorer,
    read_clip_labels,
    score_mask_manifest_with_clip,
)

__all__ = (
    "CLIP_LABEL_PRESETS",
    "DEFAULT_BACKGROUND_LABELS",
    "DEFAULT_PROMPT_TEMPLATES",
    "ClipMaskScorer",
    "ClipScoringResult",
    "HashClipMaskScorer",
    "TransformersClipMaskScorer",
    "read_clip_labels",
    "score_mask_manifest_with_clip",
)
