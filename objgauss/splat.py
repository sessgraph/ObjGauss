from __future__ import annotations

from pathlib import Path

import numpy as np

from objgauss.gaussians import GaussianCloud

_SPLAT_ROW_BYTES = 32


def read_splat(path: str | Path) -> GaussianCloud:
    """Read antimatter15/cakewalk `.splat` files as a basic Gaussian cloud."""

    path = Path(path)
    payload = path.read_bytes()
    if len(payload) % _SPLAT_ROW_BYTES != 0:
        raise ValueError(
            f"{path} size is not divisible by {_SPLAT_ROW_BYTES}; unsupported .splat layout"
        )

    count = len(payload) // _SPLAT_ROW_BYTES
    raw = np.frombuffer(payload, dtype=np.uint8).reshape(count, _SPLAT_ROW_BYTES)
    floats = raw[:, :24].copy().view("<f4").reshape(count, 6)

    vertices = np.empty(
        count,
        dtype=np.dtype(
            [
                ("x", "<f4"),
                ("y", "<f4"),
                ("z", "<f4"),
                ("scale_0", "<f4"),
                ("scale_1", "<f4"),
                ("scale_2", "<f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("opacity", "<f4"),
                ("rot_0", "u1"),
                ("rot_1", "u1"),
                ("rot_2", "u1"),
                ("rot_3", "u1"),
            ]
        ),
    )
    vertices["x"] = floats[:, 0]
    vertices["y"] = floats[:, 1]
    vertices["z"] = floats[:, 2]
    vertices["scale_0"] = floats[:, 3]
    vertices["scale_1"] = floats[:, 4]
    vertices["scale_2"] = floats[:, 5]
    vertices["red"] = raw[:, 24]
    vertices["green"] = raw[:, 25]
    vertices["blue"] = raw[:, 26]
    vertices["opacity"] = raw[:, 27].astype(np.float32) / 255.0
    vertices["rot_0"] = raw[:, 28]
    vertices["rot_1"] = raw[:, 29]
    vertices["rot_2"] = raw[:, 30]
    vertices["rot_3"] = raw[:, 31]

    return GaussianCloud(
        vertices=vertices,
        comments=("converted from antimatter15/cakewalk .splat format",),
        source_format="binary_little_endian",
    )


def write_splat(path: str | Path, cloud: GaussianCloud) -> None:
    """Write a basic antimatter15/cakewalk `.splat` file from a Gaussian cloud."""

    cloud.require_fields(("x", "y", "z"))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = cloud.count
    rows = np.zeros((count, _SPLAT_ROW_BYTES), dtype=np.uint8)
    floats = np.zeros((count, 6), dtype="<f4")
    floats[:, 0] = cloud.vertices["x"].astype(np.float32, copy=False)
    floats[:, 1] = cloud.vertices["y"].astype(np.float32, copy=False)
    floats[:, 2] = cloud.vertices["z"].astype(np.float32, copy=False)
    for axis, field in enumerate(("scale_0", "scale_1", "scale_2"), start=3):
        if field in cloud.fields:
            floats[:, axis] = cloud.vertices[field].astype(np.float32, copy=False)
        else:
            floats[:, axis] = 0.02
    rows[:, :24] = floats.view(np.uint8).reshape(count, 24)

    rows[:, 24] = _channel(cloud, "red", default=210)
    rows[:, 25] = _channel(cloud, "green", default=210)
    rows[:, 26] = _channel(cloud, "blue", default=210)
    rows[:, 27] = _opacity(cloud)
    rows[:, 28] = _channel(cloud, "rot_0", default=0)
    rows[:, 29] = _channel(cloud, "rot_1", default=0)
    rows[:, 30] = _channel(cloud, "rot_2", default=0)
    rows[:, 31] = _channel(cloud, "rot_3", default=255)
    path.write_bytes(rows.tobytes())


def _channel(cloud: GaussianCloud, field: str, *, default: int) -> np.ndarray:
    if field not in cloud.fields:
        return np.full(cloud.count, default, dtype=np.uint8)
    values = cloud.vertices[field]
    if values.dtype.kind == "f" and float(np.nanmax(values)) <= 1.0:
        values = values * 255.0
    return np.clip(values, 0, 255).astype(np.uint8)


def _opacity(cloud: GaussianCloud) -> np.ndarray:
    if "opacity" not in cloud.fields:
        return np.full(cloud.count, 220, dtype=np.uint8)
    values = cloud.vertices["opacity"].astype(np.float32, copy=False)
    if values.size and float(np.nanmin(values)) >= 0.0 and float(np.nanmax(values)) <= 1.0:
        return np.clip(values * 255.0, 0, 255).astype(np.uint8)
    activated = 1.0 / (1.0 + np.exp(-np.clip(values, -80.0, 80.0)))
    return np.clip(activated * 255.0, 0, 255).astype(np.uint8)
