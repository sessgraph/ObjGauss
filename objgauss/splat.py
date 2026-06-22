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
