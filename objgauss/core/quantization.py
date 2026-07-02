from __future__ import annotations

import hashlib
from copy import deepcopy
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

QUANTIZATION_SCHEMA = "objgauss-local-quantization-v1"
QUANTIZATION_ESTIMATE_SCHEMA = "objgauss-quantization-estimate-v1"
DEFAULT_QUANTIZATION_POLICY_ID = "chunk-aabb-uint16-rgb8-opacity8-v0"
QUANTIZED_PAYLOAD_SCHEMA = "objgauss-ogc-quantized-payload-v0"
QUANTIZED_RECORD_FORMAT = "objgauss-ogc-quantized-record-v0"
QUANTIZED_RECORD_DTYPE = np.dtype(
    [
        ("x", "<u2"),
        ("y", "<u2"),
        ("z", "<u2"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("opacity", "u1"),
    ]
)
DEQUANTIZED_RECORD_DTYPE = np.dtype(
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
RAW_REFERENCE_RECORD_BYTE_SIZE = 23


@dataclass(frozen=True)
class QuantizedPayloadWriteResult:
    payload_path: str
    index: dict[str, Any]
    sorted_indices: np.ndarray
    byte_size: int
    sha256: str


def attach_quantization_metadata(
    index: dict[str, Any],
    *,
    raw_record_byte_size: int | None = None,
) -> dict[str, Any]:
    updated = deepcopy(index)
    policy = default_quantization_policy(index)
    estimate = estimate_quantized_payload_size(
        index,
        raw_record_byte_size=raw_record_byte_size,
        quantized_record_byte_size=int(policy["estimated_record_byte_size"]),
    )
    updated["quantization"] = {
        "schema": QUANTIZATION_SCHEMA,
        "status": "metadata_estimate_only",
        "policy_id": DEFAULT_QUANTIZATION_POLICY_ID,
        "scope": "chunk",
        "layout": "object-aware-chunk-local",
        "fields": policy["fields"],
        "preserves": policy["preserves"],
        "not_implemented": [
            "actual_quantized_payload_writer",
            "vector_quantization",
            "adaptive_sh_bandwidth",
            "entropy_coding",
        ],
        "estimate": estimate,
    }
    updated["chunks"] = _annotate_chunk_quantization(updated.get("chunks", []), policy=policy)
    compression = dict(updated.get("compression", {}))
    compression["quantization"] = {
        "schema": QUANTIZATION_SCHEMA,
        "status": "metadata_estimate_only",
        "policy_id": DEFAULT_QUANTIZATION_POLICY_ID,
        "estimated_record_byte_size": int(policy["estimated_record_byte_size"]),
        "estimated_payload_byte_size": int(estimate["quantized_payload_byte_size"]),
        "estimated_compression_ratio": estimate["estimated_compression_ratio"],
    }
    updated["compression"] = compression
    return updated


def write_quantized_ogc_payload(
    payload_path: str | Path,
    cloud: GaussianCloud,
    *,
    index_path: str | Path | None = None,
    chunk_size_target: int = 8192,
) -> QuantizedPayloadWriteResult:
    chunk_result = build_chunk_index(cloud, chunk_size_target=chunk_size_target)
    payload_path = Path(payload_path)
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    index = _index_for_quantized_payload(chunk_result, payload_path=payload_path, cloud=cloud)
    with payload_path.open("wb") as file:
        for chunk in index["chunks"]:
            start, end = chunk["sorted_index_range"]
            source_indices = chunk_result.sorted_indices[start:end]
            records = quantized_records_from_cloud(cloud, source_indices, chunk=chunk)
            byte_offset = int(file.tell())
            file.write(records.tobytes(order="C"))
            byte_length = int(file.tell() - byte_offset)
            chunk["byte_offset"] = byte_offset
            chunk["byte_length"] = byte_length
            chunk["record_format"] = QUANTIZED_RECORD_FORMAT
            chunk["record_count"] = int(records.shape[0])
    byte_size = payload_path.stat().st_size
    digest = _sha256(payload_path)
    index["payload"]["byte_size"] = byte_size
    index["payload"]["sha256"] = digest
    index = annotate_lod_byte_ranges(index, record_byte_size=QUANTIZED_RECORD_DTYPE.itemsize)
    index = attach_quantization_metadata(index, raw_record_byte_size=RAW_REFERENCE_RECORD_BYTE_SIZE)
    index["quantization"]["status"] = "actual_payload_prototype"
    index["quantization"]["not_implemented"] = [
        "vector_quantization",
        "adaptive_sh_bandwidth",
        "entropy_coding",
    ]
    index["compression"]["layout"] = "object-aware-chunked-local-quantized"
    index["compression"]["quantization"]["status"] = "actual_payload_prototype"
    if index_path is not None:
        write_chunk_index(index_path, index)
    return QuantizedPayloadWriteResult(
        payload_path=str(payload_path),
        index=index,
        sorted_indices=chunk_result.sorted_indices,
        byte_size=byte_size,
        sha256=digest,
    )


def read_quantized_ogc_payload(payload_path: str | Path, index: dict[str, Any] | str | Path) -> np.ndarray:
    payload_path = Path(payload_path)
    payload_index = read_chunk_index(index) if isinstance(index, (str, Path)) else index
    records = np.empty(int(payload_index["gaussian_count"]), dtype=DEQUANTIZED_RECORD_DTYPE)
    cursor = 0
    with payload_path.open("rb") as file:
        for chunk in payload_index["chunks"]:
            record_count = int(chunk["record_count"])
            byte_length = int(chunk["byte_length"])
            expected_length = record_count * QUANTIZED_RECORD_DTYPE.itemsize
            if byte_length != expected_length:
                raise ValueError(
                    f"chunk {chunk.get('chunk_id')} byte_length {byte_length} does not match "
                    f"{record_count} quantized records"
                )
            file.seek(int(chunk["byte_offset"]))
            data = file.read(byte_length)
            if len(data) != byte_length:
                raise ValueError(f"chunk {chunk.get('chunk_id')} payload is truncated")
            quantized = np.frombuffer(data, dtype=QUANTIZED_RECORD_DTYPE, count=record_count)
            records[cursor : cursor + record_count] = dequantize_records(quantized, chunk=chunk)
            cursor += record_count
    if cursor != records.shape[0]:
        raise ValueError("payload record count does not match chunk index gaussian_count")
    return records


def quantized_records_from_cloud(
    cloud: GaussianCloud,
    indices: np.ndarray,
    *,
    chunk: dict[str, Any],
) -> np.ndarray:
    cloud.require_fields(("x", "y", "z", "object_id"))
    vertices = cloud.vertices
    records = np.zeros(indices.shape[0], dtype=QUANTIZED_RECORD_DTYPE)
    bounds_min = np.asarray(chunk["aabb_min"], dtype=np.float32)
    bounds_max = np.asarray(chunk["aabb_max"], dtype=np.float32)
    span = np.maximum(bounds_max - bounds_min, 1e-12)
    for axis, field in enumerate(("x", "y", "z")):
        values = vertices[field][indices].astype(np.float32, copy=False)
        normalized = np.clip((values - bounds_min[axis]) / span[axis], 0.0, 1.0)
        records[field] = np.rint(normalized * 65535.0).astype(np.uint16)
    if all(field in cloud.fields for field in ("red", "green", "blue")):
        for field in ("red", "green", "blue"):
            records[field] = vertices[field][indices].astype(np.uint8, copy=False)
    elif all(field in cloud.fields for field in ("f_dc_0", "f_dc_1", "f_dc_2")):
        for source, target in (("f_dc_0", "red"), ("f_dc_1", "green"), ("f_dc_2", "blue")):
            records[target] = _float_color_to_u8(vertices[source][indices])
    if "opacity" in cloud.fields:
        records["opacity"] = _opacity_to_u8(vertices["opacity"][indices])
    else:
        records["opacity"] = np.full(indices.shape[0], 255, dtype=np.uint8)
    return records


def dequantize_records(records: np.ndarray, *, chunk: dict[str, Any]) -> np.ndarray:
    bounds_min = np.asarray(chunk["aabb_min"], dtype=np.float32)
    bounds_max = np.asarray(chunk["aabb_max"], dtype=np.float32)
    span = bounds_max - bounds_min
    output = np.zeros(records.shape[0], dtype=DEQUANTIZED_RECORD_DTYPE)
    for axis, field in enumerate(("x", "y", "z")):
        output[field] = bounds_min[axis] + (records[field].astype(np.float32) / 65535.0) * span[axis]
    for field in ("red", "green", "blue"):
        output[field] = records[field]
    output["opacity"] = records["opacity"].astype(np.float32) / 255.0
    output["object_id"] = int(chunk["object_id"])
    return output


def estimate_quantized_payload_size(
    index: dict[str, Any],
    *,
    raw_record_byte_size: int | None = None,
    quantized_record_byte_size: int = 10,
) -> dict[str, Any]:
    if quantized_record_byte_size <= 0:
        raise ValueError("quantized_record_byte_size must be positive")
    gaussian_count = _positive_int(index.get("gaussian_count"), "gaussian_count")
    resolved_raw_record_byte_size = raw_record_byte_size or _payload_record_byte_size(index)
    raw_payload_bytes = (
        gaussian_count * resolved_raw_record_byte_size
        if raw_record_byte_size is not None
        else _payload_byte_size(index) or gaussian_count * resolved_raw_record_byte_size
    )
    quantized_payload_bytes = gaussian_count * quantized_record_byte_size
    ratio = raw_payload_bytes / quantized_payload_bytes if quantized_payload_bytes else 0.0
    chunks = index.get("chunks") if isinstance(index.get("chunks"), list) else []
    return {
        "schema": QUANTIZATION_ESTIMATE_SCHEMA,
        "policy_id": DEFAULT_QUANTIZATION_POLICY_ID,
        "gaussian_count": gaussian_count,
        "chunk_count": len(chunks),
        "raw_record_byte_size": resolved_raw_record_byte_size,
        "raw_payload_byte_size": int(raw_payload_bytes),
        "quantized_record_byte_size": int(quantized_record_byte_size),
        "quantized_payload_byte_size": int(quantized_payload_bytes),
        "estimated_compression_ratio": round(ratio, 6),
        "index_metadata_included": False,
    }


def _index_for_quantized_payload(
    chunk_result: ChunkIndexResult,
    *,
    payload_path: Path,
    cloud: GaussianCloud,
) -> dict[str, Any]:
    return {
        **chunk_result.index,
        "payload": {
            "schema": QUANTIZED_PAYLOAD_SCHEMA,
            "path": str(payload_path),
            "format": ".ogc",
            "record_format": QUANTIZED_RECORD_FORMAT,
            "record_byte_size": QUANTIZED_RECORD_DTYPE.itemsize,
            "raw_reference_record_byte_size": RAW_REFERENCE_RECORD_BYTE_SIZE,
            "field_schema": [
                {"name": name, "dtype": str(QUANTIZED_RECORD_DTYPE.fields[name][0])}
                for name in QUANTIZED_RECORD_DTYPE.names or ()
            ],
            "byte_size": 0,
            "sha256": "",
        },
        "compression": {
            "codec": "objgauss-ogc-prototype",
            "version": "0.1",
            "layout": "object-aware-chunked-local-quantized",
        },
        "source_fields": list(cloud.fields),
    }


def default_quantization_policy(index: dict[str, Any] | None = None) -> dict[str, Any]:
    source_fields = list(index.get("source_fields", [])) if isinstance(index, dict) else []
    color_source = ["red", "green", "blue"]
    if source_fields and not all(field in source_fields for field in color_source):
        if all(field in source_fields for field in ("f_dc_0", "f_dc_1", "f_dc_2")):
            color_source = ["f_dc_0", "f_dc_1", "f_dc_2"]
    fields = [
        {
            "name": "xyz",
            "source_fields": ["x", "y", "z"],
            "codec": "chunk-aabb-uint16x3",
            "bytes_per_gaussian": 6,
            "chunk_metadata": ["aabb_min", "aabb_max"],
            "decode": "aabb_min + uint16 / 65535 * (aabb_max - aabb_min)",
        },
        {
            "name": "rgb",
            "source_fields": color_source,
            "codec": "rgb-uint8x3",
            "bytes_per_gaussian": 3,
        },
        {
            "name": "opacity",
            "source_fields": ["opacity"],
            "codec": "opacity-uint8",
            "bytes_per_gaussian": 1,
        },
        {
            "name": "object_id",
            "source_fields": ["object_id"],
            "codec": "chunk-object-id",
            "bytes_per_gaussian": 0,
            "stored_in": "chunk.object_id",
        },
        {
            "name": "sh_rest",
            "source_fields": ["f_rest_*"],
            "codec": "planned-adaptive-sh-vq",
            "bytes_per_gaussian": 0,
            "status": "planned",
        },
    ]
    estimated_record_byte_size = sum(int(field["bytes_per_gaussian"]) for field in fields)
    return {
        "schema": QUANTIZATION_SCHEMA,
        "policy_id": DEFAULT_QUANTIZATION_POLICY_ID,
        "scope": "chunk",
        "estimated_record_byte_size": estimated_record_byte_size,
        "fields": fields,
        "preserves": {
            "object_id": "chunk.object_id",
            "chunk_boundaries": True,
            "lod_metadata": True,
            "sort_key": "object_id+morton_xyz",
        },
    }


def _annotate_chunk_quantization(chunks: object, *, policy: dict[str, Any]) -> list[dict[str, Any]]:
    if not isinstance(chunks, list):
        return []
    record_byte_size = int(policy["estimated_record_byte_size"])
    cursor = 0
    annotated = []
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        entry = dict(chunk)
        count = _positive_int(entry.get("gaussian_count"), "chunk.gaussian_count")
        byte_length = count * record_byte_size
        entry["quantization"] = {
            "schema": QUANTIZATION_SCHEMA,
            "policy_id": DEFAULT_QUANTIZATION_POLICY_ID,
            "estimated_record_byte_size": record_byte_size,
            "estimated_byte_offset": cursor,
            "estimated_byte_length": byte_length,
            "object_id_storage": "chunk.object_id",
            "aabb_source": "chunk.aabb_min/aabb_max",
        }
        if isinstance(entry.get("lod"), dict):
            entry["lod"] = _annotate_chunk_lod_quantization(
                entry["lod"],
                estimated_byte_offset=cursor,
                record_byte_size=record_byte_size,
            )
        annotated.append(entry)
        cursor += byte_length
    return annotated


def _annotate_chunk_lod_quantization(
    lod: dict[str, Any],
    *,
    estimated_byte_offset: int,
    record_byte_size: int,
) -> dict[str, Any]:
    levels = []
    for level in lod.get("levels", []):
        level = dict(level)
        record_count = int(level.get("record_count", level.get("gaussian_count", 0)))
        level["estimated_quantized_byte_offset"] = estimated_byte_offset
        level["estimated_quantized_byte_length"] = record_count * record_byte_size
        levels.append(level)
    return {**lod, "levels": levels}


def _payload_record_byte_size(index: dict[str, Any]) -> int:
    payload = index.get("payload")
    if not isinstance(payload, dict):
        raise ValueError("payload.record_byte_size is required when raw_record_byte_size is omitted")
    return _positive_int(payload.get("record_byte_size"), "payload.record_byte_size")


def _payload_byte_size(index: dict[str, Any]) -> int | None:
    payload = index.get("payload")
    if not isinstance(payload, dict):
        return None
    value = payload.get("byte_size")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


def _positive_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _opacity_to_u8(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float32)
    if numeric.size == 0:
        return np.zeros(0, dtype=np.uint8)
    if float(np.nanmin(numeric)) < 0.0 or float(np.nanmax(numeric)) > 1.0:
        numeric = 1.0 / (1.0 + np.exp(-numeric))
    numeric = np.clip(numeric, 0.0, 1.0)
    return np.rint(numeric * 255.0).astype(np.uint8)


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
