from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.lod import (
    DEFAULT_LOD_RATIOS,
    LOD_SCHEMA,
    attach_object_aware_lod_metadata,
    normalize_lod_ratios,
)

CHUNK_INDEX_SCHEMA = "objgauss-chunk-index-v1"
DEFAULT_SORT_KEY = "object_id+morton_xyz"


@dataclass(frozen=True)
class ChunkIndexResult:
    index: dict[str, Any]
    sorted_indices: np.ndarray


@dataclass(frozen=True)
class ChunkIndexValidationResult:
    passed: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    chunk_count: int
    gaussian_count: int
    object_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": "objgauss-chunk-index-validation-v1",
            "passed": self.passed,
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "chunk_count": self.chunk_count,
            "gaussian_count": self.gaussian_count,
            "object_count": self.object_count,
        }


def build_chunk_index(
    cloud: GaussianCloud,
    *,
    chunk_size_target: int = 8192,
    sort_key: str = DEFAULT_SORT_KEY,
    include_source_indices: bool = False,
    lod_levels: tuple[float, ...] | list[float] = DEFAULT_LOD_RATIOS,
) -> ChunkIndexResult:
    if sort_key != DEFAULT_SORT_KEY:
        raise ValueError(f"unsupported sort_key: {sort_key!r}")
    if chunk_size_target <= 0:
        raise ValueError("chunk_size_target must be positive")
    cloud.require_fields(("x", "y", "z", "object_id"))
    vertices = cloud.vertices
    count = cloud.count
    object_ids = vertices["object_id"].astype(np.int64, copy=False)
    unique_objects = np.unique(object_ids)
    morton_codes = morton_codes_for_points(vertices["x"], vertices["y"], vertices["z"])
    sorted_indices = np.lexsort((morton_codes, object_ids)).astype(np.int64, copy=False)
    sorted_objects = object_ids[sorted_indices]
    chunks: list[dict[str, Any]] = []
    object_summaries: list[dict[str, Any]] = []
    chunk_id = 0
    for object_id in unique_objects.tolist():
        object_positions = np.flatnonzero(sorted_objects == object_id)
        if object_positions.size == 0:
            continue
        object_sorted_indices = sorted_indices[object_positions]
        object_chunk_ids: list[int] = []
        for local_start in range(0, object_sorted_indices.size, chunk_size_target):
            local_end = min(local_start + chunk_size_target, object_sorted_indices.size)
            chunk_indices = object_sorted_indices[local_start:local_end]
            chunk = _chunk_summary(
                vertices,
                chunk_indices,
                chunk_id=chunk_id,
                object_id=int(object_id),
                sorted_start=int(object_positions[local_start]),
                include_source_indices=include_source_indices,
            )
            chunks.append(chunk)
            object_chunk_ids.append(chunk_id)
            chunk_id += 1
        object_summaries.append(
            {
                "object_id": int(object_id),
                "gaussian_count": int(object_sorted_indices.size),
                "chunk_ids": object_chunk_ids,
            }
        )
    index = {
        "schema": CHUNK_INDEX_SCHEMA,
        "sort_key": sort_key,
        "chunk_size_target": int(chunk_size_target),
        "gaussian_count": int(count),
        "object_count": int(unique_objects.size),
        "object_id_coverage": {
            "field": "object_id",
            "mode": "complete",
            "has_object_ids": True,
            "object_count": int(unique_objects.size),
        },
        "bounds": _bounds(vertices, np.arange(count, dtype=np.int64)),
        "objects": object_summaries,
        "chunks": chunks,
    }
    index = attach_object_aware_lod_metadata(index, ratios=lod_levels)
    validation = validate_chunk_index(index)
    if not validation.passed:
        raise ValueError("; ".join(validation.errors))
    return ChunkIndexResult(index=index, sorted_indices=sorted_indices)


def write_chunk_index(path: str | Path, index: dict[str, Any]) -> None:
    validation = validate_chunk_index(index)
    if not validation.passed:
        raise ValueError("; ".join(validation.errors))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_chunk_index(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    validation = validate_chunk_index(payload)
    if not validation.passed:
        raise ValueError("; ".join(validation.errors))
    return payload


def validate_chunk_index(payload: dict[str, Any]) -> ChunkIndexValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    if payload.get("schema") != CHUNK_INDEX_SCHEMA:
        errors.append(f"schema must be {CHUNK_INDEX_SCHEMA!r}")
    if payload.get("sort_key") != DEFAULT_SORT_KEY:
        errors.append(f"sort_key must be {DEFAULT_SORT_KEY!r}")
    gaussian_count = payload.get("gaussian_count")
    object_count = payload.get("object_count")
    if not _positive_int(gaussian_count):
        errors.append("gaussian_count must be a positive integer")
        gaussian_count = 0
    if not _positive_int(object_count):
        errors.append("object_count must be a positive integer")
        object_count = 0
    if not _positive_int(payload.get("chunk_size_target")):
        errors.append("chunk_size_target must be a positive integer")
    coverage = payload.get("object_id_coverage")
    if not isinstance(coverage, dict) or coverage.get("has_object_ids") is not True:
        errors.append("object_id_coverage.has_object_ids must be true")
    elif coverage.get("object_count") != object_count:
        errors.append("object_id_coverage.object_count must match object_count")
    chunks = payload.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        errors.append("chunks must be a non-empty list")
        chunks = []
    seen_chunk_ids: set[int] = set()
    total_chunk_gaussians = 0
    previous_object_id: int | None = None
    for index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            errors.append(f"chunks[{index}] must be an object")
            continue
        chunk_id = chunk.get("chunk_id")
        if not isinstance(chunk_id, int) or chunk_id < 0:
            errors.append(f"chunks[{index}].chunk_id must be a non-negative integer")
        elif chunk_id in seen_chunk_ids:
            errors.append(f"chunks[{index}].chunk_id is duplicated")
        else:
            seen_chunk_ids.add(chunk_id)
        object_id = chunk.get("object_id")
        if not isinstance(object_id, int):
            errors.append(f"chunks[{index}].object_id must be an integer")
        elif previous_object_id is not None and object_id < previous_object_id:
            errors.append("chunks must be ordered by non-decreasing object_id")
        if isinstance(object_id, int):
            previous_object_id = object_id
        chunk_count = chunk.get("gaussian_count")
        if not _positive_int(chunk_count):
            errors.append(f"chunks[{index}].gaussian_count must be a positive integer")
        else:
            total_chunk_gaussians += chunk_count
        sorted_range = chunk.get("sorted_index_range")
        if not _valid_range(sorted_range):
            errors.append(f"chunks[{index}].sorted_index_range must be [start, end]")
        if not _valid_bounds(chunk.get("aabb_min")) or not _valid_bounds(chunk.get("aabb_max")):
            errors.append(f"chunks[{index}] must include numeric aabb_min and aabb_max")
    if gaussian_count and total_chunk_gaussians != gaussian_count:
        errors.append("sum of chunk gaussian_count must match gaussian_count")
    objects = payload.get("objects")
    if not isinstance(objects, list) or len(objects) != object_count:
        errors.append("objects length must match object_count")
    elif chunks:
        object_ids = {chunk.get("object_id") for chunk in chunks if isinstance(chunk, dict)}
        summary_ids = {entry.get("object_id") for entry in objects if isinstance(entry, dict)}
        if object_ids != summary_ids:
            errors.append("objects summary must cover the same object ids as chunks")
    lod = payload.get("lod")
    if not isinstance(lod, dict) or not isinstance(lod.get("levels"), list) or not lod.get("levels"):
        warnings.append("lod.levels is missing; index can be used for chunking but not LOD planning")
    else:
        _validate_lod_metadata(
            lod,
            objects=objects if isinstance(objects, list) else [],
            chunks=chunks,
            gaussian_count=int(gaussian_count or 0),
            object_count=int(object_count or 0),
            errors=errors,
        )
    return ChunkIndexValidationResult(
        passed=not errors,
        errors=tuple(errors),
        warnings=tuple(warnings),
        chunk_count=len(chunks),
        gaussian_count=int(gaussian_count or 0),
        object_count=int(object_count or 0),
    )


def morton_codes_for_points(x: np.ndarray, y: np.ndarray, z: np.ndarray, *, bits: int = 10) -> np.ndarray:
    max_value = (1 << bits) - 1
    quantized = []
    for values in (x, y, z):
        numeric = np.asarray(values, dtype=np.float64)
        finite = np.isfinite(numeric)
        if not finite.any():
            quantized.append(np.zeros(numeric.shape, dtype=np.uint32))
            continue
        min_value = float(np.min(numeric[finite]))
        max_coord = float(np.max(numeric[finite]))
        span = max(max_coord - min_value, 1e-12)
        normalized = np.clip((numeric - min_value) / span, 0.0, 1.0)
        quantized.append(np.rint(normalized * max_value).astype(np.uint32))
    qx, qy, qz = quantized
    codes = np.zeros(qx.shape, dtype=np.uint64)
    for bit in range(bits):
        codes |= ((qx >> bit) & 1).astype(np.uint64) << (3 * bit)
        codes |= ((qy >> bit) & 1).astype(np.uint64) << (3 * bit + 1)
        codes |= ((qz >> bit) & 1).astype(np.uint64) << (3 * bit + 2)
    return codes


def _chunk_summary(
    vertices: np.ndarray,
    indices: np.ndarray,
    *,
    chunk_id: int,
    object_id: int,
    sorted_start: int,
    include_source_indices: bool,
) -> dict[str, Any]:
    bounds = _bounds(vertices, indices)
    center = [
        (bounds["aabb_min"][axis] + bounds["aabb_max"][axis]) * 0.5
        for axis in range(3)
    ]
    radius = max(
        _distance(center, [bounds["aabb_min"][0], bounds["aabb_min"][1], bounds["aabb_min"][2]]),
        _distance(center, [bounds["aabb_max"][0], bounds["aabb_max"][1], bounds["aabb_max"][2]]),
    )
    chunk = {
        "chunk_id": int(chunk_id),
        "object_id": int(object_id),
        "object_ids": [int(object_id)],
        "gaussian_count": int(indices.size),
        "sorted_index_range": [int(sorted_start), int(sorted_start + indices.size)],
        "aabb_min": bounds["aabb_min"],
        "aabb_max": bounds["aabb_max"],
        "center": center,
        "radius": float(radius),
    }
    if include_source_indices:
        chunk["source_indices"] = [int(value) for value in indices.tolist()]
    return chunk


def _bounds(vertices: np.ndarray, indices: np.ndarray) -> dict[str, list[float]]:
    if indices.size == 0:
        return {"aabb_min": [0.0, 0.0, 0.0], "aabb_max": [0.0, 0.0, 0.0]}
    return {
        "aabb_min": [
            float(np.min(vertices[field][indices]))
            for field in ("x", "y", "z")
        ],
        "aabb_max": [
            float(np.max(vertices[field][indices]))
            for field in ("x", "y", "z")
        ],
    }


def _validate_lod_metadata(
    lod: dict[str, Any],
    *,
    objects: list[Any],
    chunks: list[Any],
    gaussian_count: int,
    object_count: int,
    errors: list[str],
) -> None:
    if lod.get("schema") != LOD_SCHEMA:
        errors.append(f"lod.schema must be {LOD_SCHEMA!r}")
    ratios = normalize_lod_ratios([level.get("ratio", 1.0) for level in lod.get("levels", []) if isinstance(level, dict)])
    expected_level_count = len(ratios)
    previous_count: int | None = None
    for level_index, level in enumerate(lod.get("levels", [])):
        if not isinstance(level, dict):
            errors.append(f"lod.levels[{level_index}] must be an object")
            continue
        if level.get("level") != level_index:
            errors.append(f"lod.levels[{level_index}].level must equal {level_index}")
        count = level.get("gaussian_count")
        if not _positive_int(count):
            errors.append(f"lod.levels[{level_index}].gaussian_count must be positive")
            continue
        if level_index == 0 and gaussian_count and count != gaussian_count:
            errors.append("lod.levels[0].gaussian_count must match gaussian_count")
        if level.get("object_count") is not None and level.get("object_count") != object_count:
            errors.append(f"lod.levels[{level_index}].object_count must match object_count")
        if previous_count is not None and count > previous_count:
            errors.append("lod.levels gaussian_count must be non-increasing")
        previous_count = count

    for object_index, entry in enumerate(objects):
        if not isinstance(entry, dict):
            continue
        object_lod = entry.get("lod")
        if not isinstance(object_lod, dict):
            errors.append(f"objects[{object_index}].lod is required")
            continue
        _validate_lod_level_counts(
            object_lod.get("levels"),
            path=f"objects[{object_index}].lod.levels",
            expected_level_count=expected_level_count,
            max_count=entry.get("gaussian_count"),
            require_positive=True,
            errors=errors,
        )

    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            continue
        chunk_lod = chunk.get("lod")
        if not isinstance(chunk_lod, dict):
            errors.append(f"chunks[{chunk_index}].lod is required")
            continue
        _validate_lod_level_counts(
            chunk_lod.get("levels"),
            path=f"chunks[{chunk_index}].lod.levels",
            expected_level_count=expected_level_count,
            max_count=chunk.get("gaussian_count"),
            require_positive=False,
            errors=errors,
        )


def _validate_lod_level_counts(
    levels: object,
    *,
    path: str,
    expected_level_count: int,
    max_count: object,
    require_positive: bool,
    errors: list[str],
) -> None:
    if not isinstance(levels, list) or len(levels) != expected_level_count:
        errors.append(f"{path} length must match lod.levels")
        return
    previous_count: int | None = None
    for level_index, level in enumerate(levels):
        if not isinstance(level, dict):
            errors.append(f"{path}[{level_index}] must be an object")
            continue
        count = level.get("gaussian_count")
        valid_count = _positive_int(count) if require_positive else _non_negative_int(count)
        if not valid_count:
            errors.append(
                f"{path}[{level_index}].gaussian_count must be "
                f"{'positive' if require_positive else 'non-negative'}"
            )
            continue
        if isinstance(max_count, int) and count > max_count:
            errors.append(f"{path}[{level_index}].gaussian_count exceeds parent count")
        if previous_count is not None and count > previous_count:
            errors.append(f"{path} gaussian_count must be non-increasing")
        previous_count = count


def _distance(left: list[float], right: list[float]) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)))


def _positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _non_negative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_range(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 2
        and isinstance(value[0], int)
        and isinstance(value[1], int)
        and 0 <= value[0] < value[1]
    )


def _valid_bounds(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == 3
        and all(isinstance(entry, (int, float)) and np.isfinite(entry) for entry in value)
    )
