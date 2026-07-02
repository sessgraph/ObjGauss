from __future__ import annotations

import hashlib

import numpy as np
import pytest

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.ogc_payload import (
    OGC_PAYLOAD_SCHEMA,
    OGC_RECORD_DTYPE,
    OGC_RECORD_FORMAT,
    read_ogc_payload,
    write_ogc_payload,
)
from objgauss.ogc_payload import write_ogc_payload as historical_write_ogc_payload


def test_write_ogc_payload_records_chunk_byte_ranges_and_preserves_object_ids(tmp_path):
    cloud = _object_cloud()
    payload_path = tmp_path / "scene.ogc"
    index_path = tmp_path / "scene.index.json"

    result = write_ogc_payload(payload_path, cloud, index_path=index_path, chunk_size_target=2)

    assert payload_path.stat().st_size == 6 * OGC_RECORD_DTYPE.itemsize
    assert result.byte_size == payload_path.stat().st_size
    assert result.sha256 == hashlib.sha256(payload_path.read_bytes()).hexdigest()
    assert result.index["payload"]["schema"] == OGC_PAYLOAD_SCHEMA
    assert result.index["payload"]["record_format"] == OGC_RECORD_FORMAT
    assert result.index["payload"]["record_byte_size"] == OGC_RECORD_DTYPE.itemsize
    assert result.index["payload"]["sha256"] == result.sha256
    assert result.index["quantization"]["schema"] == "objgauss-local-quantization-v1"
    assert result.index["quantization"]["estimate"]["quantized_payload_byte_size"] < result.byte_size
    assert result.index["compression"]["quantization"]["estimated_payload_byte_size"] == 6 * 10
    assert index_path.exists()

    chunks = result.index["chunks"]
    assert [chunk["byte_offset"] for chunk in chunks] == [
        0,
        2 * OGC_RECORD_DTYPE.itemsize,
        3 * OGC_RECORD_DTYPE.itemsize,
        5 * OGC_RECORD_DTYPE.itemsize,
    ]
    assert [chunk["byte_length"] for chunk in chunks] == [
        2 * OGC_RECORD_DTYPE.itemsize,
        1 * OGC_RECORD_DTYPE.itemsize,
        2 * OGC_RECORD_DTYPE.itemsize,
        1 * OGC_RECORD_DTYPE.itemsize,
    ]
    assert [chunk["record_count"] for chunk in chunks] == [2, 1, 2, 1]
    assert [level["byte_length"] for level in chunks[0]["lod"]["levels"]] == [
        2 * OGC_RECORD_DTYPE.itemsize,
        2 * OGC_RECORD_DTYPE.itemsize,
        1 * OGC_RECORD_DTYPE.itemsize,
        1 * OGC_RECORD_DTYPE.itemsize,
    ]
    assert [level["byte_offset"] for level in chunks[0]["lod"]["levels"]] == [0, 0, 0, 0]
    assert [level["byte_length"] for level in chunks[1]["lod"]["levels"]] == [
        1 * OGC_RECORD_DTYPE.itemsize,
        0,
        0,
        0,
    ]

    records = read_ogc_payload(payload_path, index_path)

    assert records.dtype == OGC_RECORD_DTYPE
    assert records["object_id"].tolist() == [0, 0, 0, 1, 1, 1]
    assert records["x"].tolist() == pytest.approx([0.0, 1.0, 2.0, 0.0, 1.0, 2.0])
    assert records["red"].tolist() == [20, 40, 60, 10, 30, 50]
    assert records["opacity"].tolist() == pytest.approx([0.2, 0.4, 0.6, 0.1, 0.3, 0.5])


def test_write_ogc_payload_requires_object_ids(tmp_path):
    vertices = np.zeros(2, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
    cloud = GaussianCloud(vertices=vertices)

    with pytest.raises(ValueError, match="object_id"):
        write_ogc_payload(tmp_path / "bad.ogc", cloud)


def test_historical_ogc_payload_wrapper_uses_core_implementation():
    assert historical_write_ogc_payload is write_ogc_payload


def _object_cloud() -> GaussianCloud:
    vertices = np.zeros(
        6,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("opacity", "f4"),
            ("object_id", "i4"),
        ],
    )
    vertices["object_id"] = np.array([1, 0, 1, 0, 1, 0], dtype=np.int32)
    vertices["x"] = np.array([0.0, 0.0, 1.0, 1.0, 2.0, 2.0], dtype=np.float32)
    vertices["y"] = np.array([-10.0, 10.0, -10.0, 10.0, -10.0, 10.0], dtype=np.float32)
    vertices["z"] = np.zeros(6, dtype=np.float32)
    vertices["red"] = np.array([10, 20, 30, 40, 50, 60], dtype=np.uint8)
    vertices["green"] = np.array([11, 21, 31, 41, 51, 61], dtype=np.uint8)
    vertices["blue"] = np.array([12, 22, 32, 42, 52, 62], dtype=np.uint8)
    vertices["opacity"] = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="fixture")
