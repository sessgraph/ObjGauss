"""Deprecated compatibility import for semantic-slot alignment orchestration."""

from objgauss.pipelines.semantic_slots import (
    DEFAULT_SLOT_BACKGROUND_LABELS,
    SlotAlignmentResult,
    align_mask_manifest_slots,
)

__all__ = (
    "DEFAULT_SLOT_BACKGROUND_LABELS",
    "SlotAlignmentResult",
    "align_mask_manifest_slots",
)
