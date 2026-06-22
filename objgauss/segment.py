from __future__ import annotations

import numpy as np

from objgauss.features import SH_C0
from objgauss.gaussians import GaussianCloud
from objgauss.ply import append_or_replace_property


_PALETTE = np.array(
    [
        (230, 57, 70),
        (42, 157, 143),
        (69, 123, 157),
        (244, 162, 97),
        (131, 56, 236),
        (255, 190, 11),
        (29, 53, 87),
        (138, 201, 38),
        (255, 89, 94),
        (25, 130, 196),
        (106, 76, 147),
        (255, 202, 58),
    ],
    dtype=np.uint8,
)


def assign_object_ids(cloud: GaussianCloud, labels: np.ndarray) -> GaussianCloud:
    labels = np.asarray(labels, dtype=np.int32)
    vertices = append_or_replace_property(cloud.vertices, "object_id", labels, np.int32)
    return cloud.with_vertices(vertices)


def apply_object_colors(
    cloud: GaussianCloud,
    *,
    object_id_field: str = "object_id",
    rewrite_sh: bool = False,
) -> GaussianCloud:
    if object_id_field not in cloud.fields:
        raise ValueError(f"PLY vertex data has no {object_id_field!r} property")

    labels = cloud.vertices[object_id_field].astype(np.int64, copy=False)
    colors = _colors_for_labels(labels)
    vertices = cloud.vertices
    vertices = append_or_replace_property(vertices, "red", colors[:, 0], np.uint8)
    vertices = append_or_replace_property(vertices, "green", colors[:, 1], np.uint8)
    vertices = append_or_replace_property(vertices, "blue", colors[:, 2], np.uint8)

    if rewrite_sh:
        vertices = _rewrite_dc_channels(vertices, colors)

    return cloud.with_vertices(vertices)


def filter_objects(
    cloud: GaussianCloud,
    object_ids: set[int],
    *,
    mode: str,
    object_id_field: str = "object_id",
) -> GaussianCloud:
    if object_id_field not in cloud.fields:
        raise ValueError(f"PLY vertex data has no {object_id_field!r} property")
    if mode not in {"keep", "remove"}:
        raise ValueError("mode must be 'keep' or 'remove'")

    labels = cloud.vertices[object_id_field].astype(np.int64, copy=False)
    selected = np.isin(labels, list(object_ids))
    mask = selected if mode == "keep" else ~selected
    return cloud.with_vertices(cloud.vertices[mask].copy())


def parse_object_ids(raw: str) -> set[int]:
    ids = {int(part.strip()) for part in raw.split(",") if part.strip()}
    if not ids:
        raise ValueError("at least one object id is required")
    return ids


def _colors_for_labels(labels: np.ndarray) -> np.ndarray:
    if labels.size == 0:
        return np.empty((0, 3), dtype=np.uint8)

    colors = np.empty((labels.shape[0], 3), dtype=np.uint8)
    for label in np.unique(labels):
        if 0 <= label < len(_PALETTE):
            color = _PALETTE[int(label)]
        else:
            color = _hash_color(int(label))
        colors[labels == label] = color
    return colors


def _hash_color(label: int) -> np.ndarray:
    value = (label * 2654435761) & 0xFFFFFFFF
    red = 64 + (value & 0x7F)
    green = 64 + ((value >> 8) & 0x7F)
    blue = 64 + ((value >> 16) & 0x7F)
    return np.array((red, green, blue), dtype=np.uint8)


def _rewrite_dc_channels(vertices: np.ndarray, colors: np.ndarray) -> np.ndarray:
    required = ("f_dc_0", "f_dc_1", "f_dc_2")
    if not all(name in (vertices.dtype.names or ()) for name in required):
        raise ValueError("cannot rewrite SH colors because f_dc_0..2 are missing")

    rgb = colors.astype(np.float32) / 255.0
    sh = (rgb - 0.5) / SH_C0
    output = vertices.copy()
    for index, name in enumerate(required):
        output[name] = sh[:, index].astype(output.dtype.fields[name][0], copy=False)
    return output
