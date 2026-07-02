"""Semantic slot and optional CLIP scoring entry points."""

from objgauss.core.clip_scoring import (
    ClipMaskScorer,
    ClipScoringResult,
    HashClipMaskScorer,
    TransformersClipMaskScorer,
    read_clip_labels,
    score_mask_manifest_with_clip,
)
from objgauss.core.semantic_slots import SlotAlignmentResult, align_mask_manifest_slots

__all__ = [
    "ClipMaskScorer",
    "ClipScoringResult",
    "HashClipMaskScorer",
    "SlotAlignmentResult",
    "TransformersClipMaskScorer",
    "align_mask_manifest_slots",
    "read_clip_labels",
    "score_mask_manifest_with_clip",
]
