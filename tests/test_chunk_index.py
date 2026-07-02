from __future__ import annotations

import pytest

import numpy as np

from objgauss.chunk_index import validate_chunk_index as historical_validate_chunk_index
from objgauss.core import GaussianCloud, build_chunk_index
from objgauss.core.chunk_index import (
    CHUNK_INDEX_SCHEMA,
    DEFAULT_SORT_KEY,
    validate_chunk_index,
    write_chunk_index,
    read_chunk_index,
)
from objgauss.core.lod import DEFAULT_LOD_RATIOS, LOD_SCHEMA


def test_build_chunk_index_groups_by_object_then_morton_order():
    result = build_chunk_index(_object_cloud(), chunk_size_target=2, include_source_indices=True)
    index = result.index

    assert index["schema"] == CHUNK_INDEX_SCHEMA
    assert index["sort_key"] == DEFAULT_SORT_KEY
    assert index["gaussian_count"] == 6
    assert index["object_count"] == 2
    assert index["object_id_coverage"] == {
        "field": "object_id",
        "mode": "complete",
        "has_object_ids": True,
        "object_count": 2,
    }
    assert result.sorted_indices.tolist() == [1, 3, 5, 0, 2, 4]

    chunks = index["chunks"]
    assert [chunk["object_id"] for chunk in chunks] == [0, 0, 1, 1]
    assert [chunk["gaussian_count"] for chunk in chunks] == [2, 1, 2, 1]
    assert [chunk["source_indices"] for chunk in chunks] == [[1, 3], [5], [0, 2], [4]]
    assert [chunk["sorted_index_range"] for chunk in chunks] == [[0, 2], [2, 3], [3, 5], [5, 6]]
    assert chunks[0]["aabb_min"] == [0.0, 10.0, 0.0]
    assert chunks[0]["aabb_max"] == [1.0, 10.0, 0.0]
    assert chunks[2]["aabb_min"] == [0.0, -10.0, 0.0]
    assert chunks[2]["aabb_max"] == [1.0, -10.0, 0.0]
    assert [
        {
            "object_id": entry["object_id"],
            "gaussian_count": entry["gaussian_count"],
            "chunk_ids": entry["chunk_ids"],
        }
        for entry in index["objects"]
    ] == [
        {"object_id": 0, "gaussian_count": 3, "chunk_ids": [0, 1]},
        {"object_id": 1, "gaussian_count": 3, "chunk_ids": [2, 3]},
    ]

    validation = validate_chunk_index(index)
    assert validation.passed
    assert validation.chunk_count == 4
    assert validation.gaussian_count == 6
    assert validation.object_count == 2


def test_build_chunk_index_emits_deterministic_object_aware_lod_metadata():
    result = build_chunk_index(_lod_cloud(), chunk_size_target=10)
    index = result.index

    assert index["lod"]["schema"] == LOD_SCHEMA
    assert [level["ratio"] for level in index["lod"]["levels"]] == list(DEFAULT_LOD_RATIOS)
    assert [level["gaussian_count"] for level in index["lod"]["levels"]] == [40, 20, 8, 2]
    assert index["lod"]["object_guarantee"] == {
        "mode": "at-least-one-per-object-per-positive-level",
        "min_per_object": 1,
    }

    for entry in index["objects"]:
        counts = [level["gaussian_count"] for level in entry["lod"]["levels"]]
        assert counts == [20, 10, 4, 1]
        assert all(count >= 1 for count in counts)
        assert counts == sorted(counts, reverse=True)

    chunk_counts_by_level = [
        [level["gaussian_count"] for level in chunk["lod"]["levels"]]
        for chunk in index["chunks"]
    ]
    assert chunk_counts_by_level == [
        [10, 10, 4, 1],
        [10, 0, 0, 0],
        [10, 10, 4, 1],
        [10, 0, 0, 0],
    ]
    assert index["objects"][0]["lod"]["levels"][2]["chunk_ranges"] == [
        {
            "chunk_id": 0,
            "record_count": 4,
            "local_record_range": [0, 4],
            "sorted_index_range": [0, 4],
        }
    ]

    validation = validate_chunk_index(index)
    assert validation.passed


def test_chunk_index_roundtrip_json(tmp_path):
    result = build_chunk_index(_object_cloud(), chunk_size_target=3)
    path = tmp_path / "scene.index.json"

    write_chunk_index(path, result.index)
    loaded = read_chunk_index(path)

    assert loaded == result.index


def test_chunk_index_requires_object_id():
    vertices = np.zeros(2, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    cloud = GaussianCloud(vertices=vertices)

    with pytest.raises(ValueError, match="object_id"):
        build_chunk_index(cloud)


def test_validate_chunk_index_rejects_bad_schema_and_counts():
    result = build_chunk_index(_object_cloud(), chunk_size_target=2)
    payload = {
        **result.index,
        "schema": "bad-schema",
        "gaussian_count": result.index["gaussian_count"] + 1,
    }

    validation = validate_chunk_index(payload)

    assert not validation.passed
    assert "schema must be 'objgauss-chunk-index-v1'" in validation.errors
    assert "sum of chunk gaussian_count must match gaussian_count" in validation.errors


def test_historical_chunk_index_wrapper_uses_core_implementation():
    assert historical_validate_chunk_index is validate_chunk_index


def _object_cloud() -> GaussianCloud:
    vertices = np.zeros(
        6,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("object_id", "i4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["object_id"] = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
    vertices["x"] = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    vertices["y"] = np.array([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=np.float32)
    vertices["z"] = np.zeros(6, dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="fixture")


def _lod_cloud() -> GaussianCloud:
    vertices = np.zeros(
        40,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("object_id", "i4"),
        ],
    )
    vertices["object_id"] = np.repeat(np.array([0, 1], dtype=np.int32), 20)
    vertices["x"] = np.tile(np.arange(20, dtype=np.float32), 2)
    vertices["y"] = np.repeat(np.array([0.0, 10.0], dtype=np.float32), 20)
    vertices["z"] = np.zeros(40, dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="fixture")
