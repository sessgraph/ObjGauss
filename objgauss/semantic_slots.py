"""Compatibility wrapper for core semantic slot alignment algorithms."""

from objgauss.core.semantic_slots import (
    DEFAULT_SLOT_BACKGROUND_LABELS,
    SlotAlignmentResult,
    align_mask_manifest_slots,
)

__all__ = [
    "DEFAULT_SLOT_BACKGROUND_LABELS",
    "SlotAlignmentResult",
    "align_mask_manifest_slots",
]
