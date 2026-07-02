"""Compatibility wrapper for object-aware Gaussian chunk indexing."""

from objgauss.core.chunk_index import (
    CHUNK_INDEX_SCHEMA,
    DEFAULT_SORT_KEY,
    ChunkIndexResult,
    ChunkIndexValidationResult,
    build_chunk_index,
    morton_codes_for_points,
    read_chunk_index,
    validate_chunk_index,
    write_chunk_index,
)
from objgauss.core.lod import DEFAULT_LOD_RATIOS, LOD_SCHEMA

__all__ = [
    "CHUNK_INDEX_SCHEMA",
    "DEFAULT_LOD_RATIOS",
    "DEFAULT_SORT_KEY",
    "LOD_SCHEMA",
    "ChunkIndexResult",
    "ChunkIndexValidationResult",
    "build_chunk_index",
    "morton_codes_for_points",
    "read_chunk_index",
    "validate_chunk_index",
    "write_chunk_index",
]
