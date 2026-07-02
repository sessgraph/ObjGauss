from __future__ import annotations

import numpy as np

from objgauss.core.chunk_index import build_chunk_index
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.lod import LOD_SCHEMA
from objgauss.core.ogc_payload import write_ogc_payload
from objgauss.core.quantization import (
    QUANTIZED_PAYLOAD_SCHEMA,
    QUANTIZED_RECORD_DTYPE,
    QUANTIZED_RECORD_FORMAT,
    QUANTIZATION_ESTIMATE_SCHEMA,
    QUANTIZATION_SCHEMA,
    attach_quantization_metadata,
    estimate_quantized_payload_size,
    read_quantized_ogc_payload,
    write_quantized_ogc_payload,
)
from objgauss.quantization import (
    attach_quantization_metadata as historical_attach_quantization,
    write_quantized_ogc_payload as historical_write_quantized_ogc_payload,
)


def test_attach_quantization_metadata_estimates_smaller_chunk_local_payload():
    result = build_chunk_index(_object_cloud(), chunk_size_target=10)

    index = attach_quantization_metadata(result.index, raw_record_byte_size=23)

    quantization = index["quantization"]
    estimate = quantization["estimate"]
    assert quantization["schema"] == QUANTIZATION_SCHEMA
    assert quantization["status"] == "metadata_estimate_only"
    assert estimate["schema"] == QUANTIZATION_ESTIMATE_SCHEMA
    assert estimate["raw_payload_byte_size"] == 40 * 23
    assert estimate["quantized_record_byte_size"] == 10
    assert estimate["quantized_payload_byte_size"] == 40 * 10
    assert estimate["quantized_payload_byte_size"] < estimate["raw_payload_byte_size"]
    assert estimate["index_metadata_included"] is False

    assert index["object_id_coverage"]["has_object_ids"] is True
    assert index["lod"]["schema"] == LOD_SCHEMA
    assert [chunk["object_id"] for chunk in index["chunks"]] == [0, 0, 1, 1]
    assert [chunk["quantization"]["estimated_byte_offset"] for chunk in index["chunks"]] == [0, 100, 200, 300]
    assert [chunk["quantization"]["estimated_byte_length"] for chunk in index["chunks"]] == [100, 100, 100, 100]
    assert [level["estimated_quantized_byte_length"] for level in index["chunks"][0]["lod"]["levels"]] == [
        100,
        100,
        40,
        10,
    ]


def test_estimate_quantized_payload_size_can_use_payload_record_metadata():
    result = build_chunk_index(_object_cloud(), chunk_size_target=10)
    index = {
        **result.index,
        "payload": {"record_byte_size": 23, "byte_size": 920},
    }

    estimate = estimate_quantized_payload_size(index)

    assert estimate["raw_record_byte_size"] == 23
    assert estimate["raw_payload_byte_size"] == 920
    assert estimate["quantized_payload_byte_size"] == 400


def test_write_quantized_ogc_payload_is_smaller_and_roundtrips_with_bounded_error(tmp_path):
    cloud = _object_cloud()
    raw = write_ogc_payload(tmp_path / "scene.raw.ogc", cloud, chunk_size_target=10)
    quantized = write_quantized_ogc_payload(
        tmp_path / "scene.quantized.ogc",
        cloud,
        index_path=tmp_path / "scene.quantized.index.json",
        chunk_size_target=10,
    )

    assert quantized.byte_size == cloud.count * QUANTIZED_RECORD_DTYPE.itemsize
    assert quantized.byte_size < raw.byte_size
    assert quantized.index["payload"]["schema"] == QUANTIZED_PAYLOAD_SCHEMA
    assert quantized.index["payload"]["record_format"] == QUANTIZED_RECORD_FORMAT
    assert quantized.index["payload"]["record_byte_size"] == QUANTIZED_RECORD_DTYPE.itemsize
    assert quantized.index["quantization"]["status"] == "actual_payload_prototype"
    assert quantized.index["compression"]["layout"] == "object-aware-chunked-local-quantized"
    assert quantized.index["compression"]["quantization"]["status"] == "actual_payload_prototype"
    assert quantized.index["quantization"]["estimate"]["raw_payload_byte_size"] == raw.byte_size
    assert quantized.index["quantization"]["estimate"]["quantized_payload_byte_size"] == quantized.byte_size

    chunks = quantized.index["chunks"]
    assert [chunk["object_id"] for chunk in chunks] == [0, 0, 1, 1]
    assert [chunk["record_format"] for chunk in chunks] == [QUANTIZED_RECORD_FORMAT] * 4
    assert [chunk["byte_length"] for chunk in chunks] == [10 * QUANTIZED_RECORD_DTYPE.itemsize] * 4
    assert chunks[0]["lod"]["schema"] == LOD_SCHEMA
    assert chunks[0]["lod"]["levels"][2]["byte_length"] == 4 * QUANTIZED_RECORD_DTYPE.itemsize
    assert chunks[0]["lod"]["levels"][2]["estimated_quantized_byte_length"] == 4 * QUANTIZED_RECORD_DTYPE.itemsize

    decoded = read_quantized_ogc_payload(quantized.payload_path, tmp_path / "scene.quantized.index.json")
    sorted_vertices = cloud.vertices[quantized.sorted_indices]

    assert decoded["object_id"].tolist() == sorted_vertices["object_id"].tolist()
    assert decoded["red"].tolist() == sorted_vertices["red"].tolist()
    assert decoded["green"].tolist() == sorted_vertices["green"].tolist()
    assert decoded["blue"].tolist() == sorted_vertices["blue"].tolist()
    np.testing.assert_allclose(decoded["x"], sorted_vertices["x"], atol=1e-3)
    np.testing.assert_allclose(decoded["y"], sorted_vertices["y"], atol=1e-6)
    np.testing.assert_allclose(decoded["z"], sorted_vertices["z"], atol=1e-6)
    np.testing.assert_allclose(decoded["opacity"], sorted_vertices["opacity"], atol=1.0 / 255.0)


def test_historical_quantization_wrapper_uses_core_implementation():
    assert historical_attach_quantization is attach_quantization_metadata
    assert historical_write_quantized_ogc_payload is write_quantized_ogc_payload


def _object_cloud() -> GaussianCloud:
    vertices = np.zeros(
        40,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("object_id", "i4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("opacity", "f4"),
        ],
    )
    vertices["object_id"] = np.repeat(np.array([0, 1], dtype=np.int32), 20)
    vertices["x"] = np.tile(np.arange(20, dtype=np.float32), 2)
    vertices["y"] = np.repeat(np.array([0.0, 10.0], dtype=np.float32), 20)
    vertices["z"] = np.zeros(40, dtype=np.float32)
    vertices["red"] = np.arange(40, dtype=np.uint8)
    vertices["green"] = np.arange(40, dtype=np.uint8)
    vertices["blue"] = np.arange(40, dtype=np.uint8)
    vertices["opacity"] = np.linspace(0.1, 0.9, 40, dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="fixture")
