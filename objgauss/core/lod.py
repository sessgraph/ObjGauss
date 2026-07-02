from __future__ import annotations

import math
from typing import Any

LOD_SCHEMA = "objgauss-object-aware-lod-v1"
DEFAULT_LOD_RATIOS = (1.0, 0.5, 0.2, 0.05)
DEFAULT_LOD_SELECTION = "object-aware-morton-prefix"


def attach_object_aware_lod_metadata(
    index: dict[str, Any],
    *,
    ratios: tuple[float, ...] | list[float] = DEFAULT_LOD_RATIOS,
    selection: str = DEFAULT_LOD_SELECTION,
) -> dict[str, Any]:
    normalized = normalize_lod_ratios(ratios)
    chunks = [dict(chunk) for chunk in index.get("chunks", [])]
    objects = [dict(entry) for entry in index.get("objects", [])]
    chunks_by_id = {int(chunk["chunk_id"]): chunk for chunk in chunks}
    chunk_lod_levels: dict[int, list[dict[str, Any]]] = {
        int(chunk["chunk_id"]): [
            _chunk_level_summary(
                level=level_index,
                ratio=ratio,
                chunk=chunk,
                record_count=0,
            )
            for level_index, ratio in enumerate(normalized)
        ]
        for chunk in chunks
    }
    global_counts = [0 for _ in normalized]
    object_level_counts: list[list[int]] = []

    for object_entry in objects:
        object_count = int(object_entry["gaussian_count"])
        counts = lod_counts_for_count(object_count, normalized)
        object_level_counts.append(counts)
        object_chunk_ids = [int(value) for value in object_entry.get("chunk_ids", [])]
        object_start = _object_sorted_start(object_chunk_ids, chunks_by_id)
        levels = []
        for level_index, (ratio, target_count) in enumerate(zip(normalized, counts)):
            global_counts[level_index] += target_count
            chunk_ranges = _assign_level_to_object_chunks(
                target_count=target_count,
                level_index=level_index,
                ratio=ratio,
                chunk_ids=object_chunk_ids,
                chunks_by_id=chunks_by_id,
                chunk_lod_levels=chunk_lod_levels,
            )
            levels.append(
                {
                    "level": level_index,
                    "ratio": ratio,
                    "gaussian_count": target_count,
                    "sorted_index_range": [object_start, object_start + target_count],
                    "chunk_ranges": chunk_ranges,
                }
            )
        object_entry["lod"] = {
            "schema": LOD_SCHEMA,
            "selection": selection,
            "levels": levels,
        }

    for chunk in chunks:
        chunk_id = int(chunk["chunk_id"])
        chunk["lod"] = {
            "schema": LOD_SCHEMA,
            "selection": selection,
            "levels": chunk_lod_levels[chunk_id],
        }

    updated = dict(index)
    updated["objects"] = objects
    updated["chunks"] = chunks
    updated["lod"] = {
        "schema": LOD_SCHEMA,
        "selection": selection,
        "sort_key": index.get("sort_key"),
        "chunk_range_mode": "prefix-records",
        "object_guarantee": {
            "mode": "at-least-one-per-object-per-positive-level",
            "min_per_object": 1,
        },
        "levels": [
            {
                "level": level_index,
                "ratio": ratio,
                "gaussian_count": int(global_counts[level_index]),
                "object_count": len(objects),
                "min_object_gaussians": int(min(counts[level_index] for counts in object_level_counts))
                if object_level_counts
                else 0,
                "max_object_gaussians": int(max(counts[level_index] for counts in object_level_counts))
                if object_level_counts
                else 0,
            }
            for level_index, ratio in enumerate(normalized)
        ],
    }
    return updated


def annotate_lod_byte_ranges(index: dict[str, Any], *, record_byte_size: int) -> dict[str, Any]:
    if record_byte_size <= 0:
        raise ValueError("record_byte_size must be positive")
    updated = dict(index)
    chunks = [dict(chunk) for chunk in index.get("chunks", [])]
    for chunk in chunks:
        byte_offset = chunk.get("byte_offset")
        lod = chunk.get("lod")
        if not isinstance(byte_offset, int) or not isinstance(lod, dict):
            continue
        levels = []
        for level in lod.get("levels", []):
            level = dict(level)
            record_count = int(level.get("record_count", level.get("gaussian_count", 0)))
            level["byte_offset"] = byte_offset
            level["byte_length"] = int(record_count * record_byte_size)
            levels.append(level)
        chunk["lod"] = {**lod, "levels": levels}
    updated["chunks"] = chunks
    return updated


def normalize_lod_ratios(ratios: tuple[float, ...] | list[float]) -> tuple[float, ...]:
    values = []
    for raw_ratio in ratios:
        ratio = float(raw_ratio)
        if not math.isfinite(ratio) or ratio <= 0.0:
            continue
        values.append(min(ratio, 1.0))
    if not values:
        values = list(DEFAULT_LOD_RATIOS)
    values.append(1.0)
    normalized = []
    for ratio in sorted(values, reverse=True):
        if not any(abs(ratio - existing) < 1e-9 for existing in normalized):
            normalized.append(ratio)
    return tuple(normalized)


def lod_counts_for_count(count: int, ratios: tuple[float, ...] | list[float]) -> list[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    counts = []
    previous = int(count)
    for level_index, ratio in enumerate(normalize_lod_ratios(ratios)):
        if level_index == 0:
            level_count = int(count)
        else:
            level_count = max(1, int(math.ceil(count * ratio)))
            level_count = min(previous, level_count)
        counts.append(level_count)
        previous = level_count
    return counts


def _assign_level_to_object_chunks(
    *,
    target_count: int,
    level_index: int,
    ratio: float,
    chunk_ids: list[int],
    chunks_by_id: dict[int, dict[str, Any]],
    chunk_lod_levels: dict[int, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    remaining = target_count
    chunk_ranges = []
    for chunk_id in chunk_ids:
        chunk = chunks_by_id[chunk_id]
        chunk_count = int(chunk["gaussian_count"])
        record_count = min(max(remaining, 0), chunk_count)
        chunk_lod_levels[chunk_id][level_index] = _chunk_level_summary(
            level=level_index,
            ratio=ratio,
            chunk=chunk,
            record_count=record_count,
        )
        if record_count > 0:
            start = int(chunk["sorted_index_range"][0])
            chunk_ranges.append(
                {
                    "chunk_id": chunk_id,
                    "record_count": record_count,
                    "local_record_range": [0, record_count],
                    "sorted_index_range": [start, start + record_count],
                }
            )
        remaining -= record_count
    return chunk_ranges


def _chunk_level_summary(
    *,
    level: int,
    ratio: float,
    chunk: dict[str, Any],
    record_count: int,
) -> dict[str, Any]:
    start = int(chunk["sorted_index_range"][0])
    return {
        "level": int(level),
        "ratio": float(ratio),
        "gaussian_count": int(record_count),
        "record_count": int(record_count),
        "local_record_range": [0, int(record_count)],
        "sorted_index_range": [start, start + int(record_count)],
    }


def _object_sorted_start(chunk_ids: list[int], chunks_by_id: dict[int, dict[str, Any]]) -> int:
    if not chunk_ids:
        return 0
    return int(chunks_by_id[chunk_ids[0]]["sorted_index_range"][0])
