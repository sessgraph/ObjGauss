from __future__ import annotations

import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RenderedNerfDataset:
    dataset_path: Path
    transforms_train_path: Path
    frames: int
    image_size: int
    triangles: int


@dataclass(frozen=True)
class GltfMesh:
    positions: np.ndarray
    normals: np.ndarray
    indices: np.ndarray


_COMPONENT_DTYPES = {
    5121: np.uint8,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
_TYPE_COMPONENTS = {
    "SCALAR": 1,
    "VEC2": 2,
    "VEC3": 3,
    "VEC4": 4,
}


def render_gltf_nerf_dataset(
    gltf_path: str | Path,
    output_dir: str | Path,
    *,
    frames: int = 16,
    image_size: int = 256,
    radius: float = 2.0,
    elevation_degrees: float = 22.0,
    fov_degrees: float = 42.0,
) -> RenderedNerfDataset:
    if frames < 2:
        raise ValueError("frames must be >= 2")
    if image_size < 32:
        raise ValueError("image_size must be >= 32")

    gltf_path = Path(gltf_path)
    output_dir = Path(output_dir)
    train_dir = output_dir / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    mesh = load_gltf_mesh(gltf_path)
    target = _mesh_center(mesh.positions)
    model_radius = max(float(np.linalg.norm(mesh.positions - target, axis=1).max()), 1e-3)
    camera_radius = radius * model_radius
    elevation = math.radians(elevation_degrees)
    camera_angle_x = math.radians(fov_degrees)

    frame_entries: list[dict[str, Any]] = []
    for index in range(frames):
        angle = 2.0 * math.pi * float(index) / float(frames)
        eye = target + camera_radius * np.array(
            [
                math.cos(elevation) * math.sin(angle),
                math.sin(elevation),
                math.cos(elevation) * math.cos(angle),
            ],
            dtype=np.float32,
        )
        c2w = _look_at_c2w(eye, target)
        rgba = _render_mesh(mesh, c2w, image_size=image_size, camera_angle_x=camera_angle_x)
        frame_name = f"r_{index}"
        _write_rgba_png(train_dir / f"{frame_name}.png", rgba)
        frame_entries.append(
            {
                "file_path": f"./train/{frame_name}",
                "rotation": 2.0 * math.pi / float(frames),
                "transform_matrix": c2w.tolist(),
            }
        )

    payload = {
        "camera_angle_x": camera_angle_x,
        "source_type": "gltf-orbit-render",
        "source_gltf": str(gltf_path),
        "frames": frame_entries,
    }
    transforms_path = output_dir / "transforms_train.json"
    for split in ("train", "val", "test"):
        (output_dir / f"transforms_{split}.json").write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    return RenderedNerfDataset(
        dataset_path=output_dir,
        transforms_train_path=transforms_path,
        frames=frames,
        image_size=image_size,
        triangles=int(mesh.indices.size // 3),
    )


def load_gltf_mesh(gltf_path: str | Path) -> GltfMesh:
    gltf_path = Path(gltf_path)
    payload = json.loads(gltf_path.read_text(encoding="utf-8"))
    meshes = payload.get("meshes") or []
    if not meshes:
        raise ValueError(f"{gltf_path} contains no meshes")
    position_chunks: list[np.ndarray] = []
    normal_chunks: list[np.ndarray] = []
    index_chunks: list[np.ndarray] = []
    vertex_offset = 0
    for mesh in meshes:
        for primitive in mesh.get("primitives") or []:
            attributes = primitive.get("attributes") or {}
            if "POSITION" not in attributes:
                continue
            if "indices" not in primitive:
                continue
            positions = _read_accessor(payload, gltf_path.parent, int(attributes["POSITION"])).astype(
                np.float32,
                copy=False,
            )
            indices = _read_accessor(payload, gltf_path.parent, int(primitive["indices"])).astype(
                np.int32,
                copy=False,
            )
            normals = (
                _read_accessor(payload, gltf_path.parent, int(attributes["NORMAL"])).astype(np.float32, copy=False)
                if "NORMAL" in attributes
                else _fallback_vertex_normals(positions, indices)
            )
            if positions.ndim != 2 or positions.shape[1] != 3:
                raise ValueError("POSITION accessor must be VEC3")
            if normals.shape != positions.shape:
                raise ValueError("NORMAL accessor must match POSITION shape")
            if indices.size % 3 != 0:
                raise ValueError("triangle index count must be divisible by 3")
            position_chunks.append(positions)
            normal_chunks.append(normals)
            index_chunks.append(indices.reshape(-1) + vertex_offset)
            vertex_offset += int(positions.shape[0])
    if not position_chunks:
        raise ValueError(f"{gltf_path} contains no indexed POSITION primitives")
    return GltfMesh(
        positions=np.concatenate(position_chunks, axis=0),
        normals=_normalize_rows(np.concatenate(normal_chunks, axis=0)),
        indices=np.concatenate(index_chunks, axis=0),
    )


def _read_accessor(payload: dict[str, Any], root: Path, accessor_index: int) -> np.ndarray:
    accessors = payload.get("accessors") or []
    buffer_views = payload.get("bufferViews") or []
    buffers = payload.get("buffers") or []
    accessor = accessors[accessor_index]
    view = buffer_views[int(accessor["bufferView"])]
    buffer = buffers[int(view.get("buffer", 0))]
    dtype = _COMPONENT_DTYPES[int(accessor["componentType"])]
    component_count = _TYPE_COMPONENTS[str(accessor["type"])]
    count = int(accessor["count"])
    byte_offset = int(view.get("byteOffset", 0)) + int(accessor.get("byteOffset", 0))
    byte_stride = view.get("byteStride")
    data = (root / str(buffer["uri"])).read_bytes()
    dtype = np.dtype(dtype).newbyteorder("<")
    if byte_stride:
        rows = np.empty((count, component_count), dtype=dtype)
        for index in range(count):
            start = byte_offset + index * int(byte_stride)
            rows[index] = np.frombuffer(data, dtype=dtype, count=component_count, offset=start)
    else:
        rows = np.frombuffer(data, dtype=dtype, count=count * component_count, offset=byte_offset)
        rows = rows.reshape(count, component_count)
    return rows[:, 0] if component_count == 1 else rows


def _render_mesh(
    mesh: GltfMesh,
    c2w: np.ndarray,
    *,
    image_size: int,
    camera_angle_x: float,
) -> np.ndarray:
    height = width = image_size
    color = np.zeros((height, width, 4), dtype=np.uint8)
    depth = np.full((height, width), np.inf, dtype=np.float32)
    focal = 0.5 * width / math.tan(0.5 * camera_angle_x)
    rotation = c2w[:3, :3]
    eye = c2w[:3, 3]
    camera = (mesh.positions - eye) @ rotation
    z = camera[:, 2]
    view_depth = -z
    valid = view_depth > 1e-4
    projected = np.zeros((mesh.positions.shape[0], 2), dtype=np.float32)
    projected[valid, 0] = focal * (camera[valid, 0] / view_depth[valid]) + width * 0.5
    projected[valid, 1] = -focal * (camera[valid, 1] / view_depth[valid]) + height * 0.5

    light = _normalize(np.array([0.35, 0.75, 0.55], dtype=np.float32))
    base_color = np.array([176.0, 142.0, 91.0], dtype=np.float32)
    accent = np.array([82.0, 98.0, 112.0], dtype=np.float32)
    triangles = mesh.indices.reshape(-1, 3)
    for triangle in triangles:
        if not np.all(valid[triangle]):
            continue
        pts = projected[triangle]
        min_x = max(0, int(math.floor(float(pts[:, 0].min()))))
        max_x = min(width - 1, int(math.ceil(float(pts[:, 0].max()))))
        min_y = max(0, int(math.floor(float(pts[:, 1].min()))))
        max_y = min(height - 1, int(math.ceil(float(pts[:, 1].max()))))
        if min_x > max_x or min_y > max_y:
            continue
        area = _edge(pts[0], pts[1], pts[2])
        if abs(area) < 1e-6:
            continue
        for y in range(min_y, max_y + 1):
            for x in range(min_x, max_x + 1):
                p = np.array([x + 0.5, y + 0.5], dtype=np.float32)
                w0 = _edge(pts[1], pts[2], p) / area
                w1 = _edge(pts[2], pts[0], p) / area
                w2 = 1.0 - w0 - w1
                if w0 < -1e-5 or w1 < -1e-5 or w2 < -1e-5:
                    continue
                current_depth = (
                    w0 * view_depth[triangle[0]]
                    + w1 * view_depth[triangle[1]]
                    + w2 * view_depth[triangle[2]]
                )
                if current_depth >= depth[y, x]:
                    continue
                normal = _normalize(
                    w0 * mesh.normals[triangle[0]]
                    + w1 * mesh.normals[triangle[1]]
                    + w2 * mesh.normals[triangle[2]]
                )
                shade = 0.36 + 0.64 * abs(float(np.dot(normal, light)))
                mix = 0.18 + 0.18 * math.sin(0.07 * x + 0.11 * y)
                rgb = np.clip((1.0 - mix) * base_color + mix * accent, 0, 255) * shade
                color[y, x, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
                color[y, x, 3] = 255
                depth[y, x] = current_depth
    return color


def _fallback_vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(positions, dtype=np.float32)
    for i0, i1, i2 in indices.astype(np.int32).reshape(-1, 3):
        face = np.cross(positions[i1] - positions[i0], positions[i2] - positions[i0])
        normals[i0] += face
        normals[i1] += face
        normals[i2] += face
    return _normalize_rows(normals)


def _look_at_c2w(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    forward = _normalize(target - eye)
    world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    right = _normalize(np.cross(forward, world_up))
    up = _normalize(np.cross(right, forward))
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 0] = right
    matrix[:3, 1] = up
    matrix[:3, 2] = -forward
    matrix[:3, 3] = eye
    return matrix


def _mesh_center(positions: np.ndarray) -> np.ndarray:
    bounds_min = positions.min(axis=0)
    bounds_max = positions.max(axis=0)
    return ((bounds_min + bounds_max) * 0.5).astype(np.float32)


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-8)


def _normalize(value: np.ndarray) -> np.ndarray:
    return value / max(float(np.linalg.norm(value)), 1e-8)


def _edge(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((c[0] - a[0]) * (b[1] - a[1]) - (c[1] - a[1]) * (b[0] - a[0]))


def _write_rgba_png(path: Path, rgba: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width, channels = rgba.shape
    if channels != 4:
        raise ValueError("PNG writer expects RGBA input")
    raw = bytearray()
    for row in rgba:
        raw.append(0)
        raw.extend(row.tobytes())
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
    png += _png_chunk(b"IDAT", zlib.compress(bytes(raw)))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind)
    checksum = zlib.crc32(data, checksum)
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum & 0xFFFFFFFF)
