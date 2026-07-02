"""Compatibility wrapper for object-aware Gaussian LOD metadata."""

from objgauss.core.lod import (
    DEFAULT_LOD_RATIOS,
    DEFAULT_LOD_SELECTION,
    LOD_SCHEMA,
    annotate_lod_byte_ranges,
    attach_object_aware_lod_metadata,
    lod_counts_for_count,
    normalize_lod_ratios,
)

__all__ = [
    "DEFAULT_LOD_RATIOS",
    "DEFAULT_LOD_SELECTION",
    "LOD_SCHEMA",
    "annotate_lod_byte_ranges",
    "attach_object_aware_lod_metadata",
    "lod_counts_for_count",
    "normalize_lod_ratios",
]
