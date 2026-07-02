from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.core.chunk_index import (
    ChunkIndexResult,
    build_chunk_index,
    read_chunk_index,
    write_chunk_index,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.lod import annotate_lod_byte_ranges
from objgauss.core.quantization import attach_quantization_metadata

OGC_PAYLOAD_SCHEMA = "objgauss-ogc-payload-v0"
OGC_RECORD_FORMAT = "objgauss-ogc-record-v0"
OGC_RECORD_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("opacity", "<f4"),
        ("object_id", "<i4"),
    ]
)


@dataclass(frozen=True)
class OgcPayloadWriteResult:
    payload_path: str
    index: dict[str, Any]
    sorted_indices: np.ndarray
    byte_size: int
    sha256: str


def write_ogc_payload(
    payload_path: str | Path,
    cloud: GaussianCloud,
    *,
    index_path: str | Path | None = None,
    chunk_size_target: int = 8192,
) -> OgcPayloadWriteResult:
    chunk_result = build_chunk_index(cloud, chunk_size_target=chunk_size_target)
    payload_path = Path(payload_path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    index = _index_for_payload(chunk_result, payload_path=payload_path, cloud=cloud)
    with payload_path.open("wb") as file:
        for chunk in index["chunks"]:
            start, end = chunk["sorted_index_range"]
            source_indices = chunk_result.sorted_indices[start:end]
            records = records_from_cloud(cloud, source_indices)
            byte_offset = int(file.tell())
            file.write(records.tobytes(order="C"))
            byte_length = int(file.tell() - byte_offset)
            chunk["byte_offset"] = byte_offset
            chunk["byte_length"] = byte_length
            chunk["record_format"] = OGC_RECORD_FORMAT
            chunk["record_count"] = int(records.shape[0])
    byte_size = payload_path.stat().st_size
    digest = _sha256(payload_path)
    index["payload"]["byte_size"] = byte_size
    index["payload"]["sha256"] = digest
    index = annotate_lod_byte_ranges(index, record_byte_size=OGC_RECORD_DTYPE.itemsize)
    index = attach_quantization_metadata(index, raw_record_byte_size=OGC_RECORD_DTYPE.itemsize)
    if index_path is not None:
        write_chunk_index(index_path, index)
    return OgcPayloadWriteResult(
        payload_path=str(payload_path),
        index=index,
        sorted_indices=chunk_result.sorted_indices,
        byte_size=byte_size,
        sha256=digest,
    )


def read_ogc_payload(payload_path: str | Path, index: dict[str, Any] | str | Path) -> np.ndarray:
    payload_path = Path(payload_path)
    payload_index = read_chunk_index(index) if isinstance(index, (str, Path)) else index
    records = np.empty(int(payload_index["gaussian_count"]), dtype=OGC_RECORD_DTYPE)
    cursor = 0
    with payload_path.open("rb") as file:
        for chunk in payload_index["chunks"]:
            record_count = int(chunk["record_count"])
            byte_length = int(chunk["byte_length"])
            expected_length = record_count * OGC_RECORD_DTYPE.itemsize
            if byte_length != expected_length:
                raise ValueError(
                    f"chunk {chunk.get('chunk_id')} byte_length {byte_length} does not match "
                    f"{record_count} records"
                )
            file.seek(int(chunk["byte_offset"]))
            data = file.read(byte_length)
            if len(data) != byte_length:
                raise ValueError(f"chunk {chunk.get('chunk_id')} payload is truncated")
            records[cursor : cursor + record_count] = np.frombuffer(data, dtype=OGC_RECORD_DTYPE, count=record_count)
            cursor += record_count
    if cursor != records.shape[0]:
        raise ValueError("payload record count does not match chunk index gaussian_count")
    return records


def records_from_cloud(cloud: GaussianCloud, indices: np.ndarray) -> np.ndarray:
    cloud.require_fields(("x", "y", "z", "object_id"))
    vertices = cloud.vertices
    records = np.zeros(indices.shape[0], dtype=OGC_RECORD_DTYPE)
    for field in ("x", "y", "z"):
        records[field] = vertices[field][indices].astype(np.float32, copy=False)
    records["object_id"] = vertices["object_id"][indices].astype(np.int32, copy=False)
    if all(field in cloud.fields for field in ("red", "green", "blue")):
        for field in ("red", "green", "blue"):
            records[field] = vertices[field][indices].astype(np.uint8, copy=False)
    elif all(field in cloud.fields for field in ("f_dc_0", "f_dc_1", "f_dc_2")):
        for source, target in (("f_dc_0", "red"), ("f_dc_1", "green"), ("f_dc_2", "blue")):
            records[target] = _float_color_to_u8(vertices[source][indices])
    if "opacity" in cloud.fields:
        records["opacity"] = vertices["opacity"][indices].astype(np.float32, copy=False)
    else:
        records["opacity"] = np.ones(indices.shape[0], dtype=np.float32)
    return records


def _index_for_payload(
    chunk_result: ChunkIndexResult,
    *,
    payload_path: Path,
    cloud: GaussianCloud,
) -> dict[str, Any]:
    index = {
        **chunk_result.index,
        "payload": {
            "schema": OGC_PAYLOAD_SCHEMA,
            "path": str(payload_path),
            "format": ".ogc",
            "record_format": OGC_RECORD_FORMAT,
            "record_byte_size": OGC_RECORD_DTYPE.itemsize,
            "field_schema": [
                {"name": name, "dtype": str(OGC_RECORD_DTYPE.fields[name][0])}
                for name in OGC_RECORD_DTYPE.names or ()
            ],
            "byte_size": 0,
            "sha256": "",
        },
        "compression": {
            "codec": "objgauss-ogc-prototype",
            "version": "0.1",
            "layout": "object-aware-chunked-uncompressed",
        },
        "source_fields": list(cloud.fields),
    }
    return index


def _float_color_to_u8(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float32)
    if numeric.size == 0:
        return np.zeros(0, dtype=np.uint8)
    if float(np.nanmin(numeric)) < 0.0 or float(np.nanmax(numeric)) > 1.0:
        numeric = np.clip((numeric + 1.0) * 127.5, 0.0, 255.0)
    else:
        numeric = np.clip(numeric * 255.0, 0.0, 255.0)
    return np.rint(numeric).astype(np.uint8)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
