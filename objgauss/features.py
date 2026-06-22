from __future__ import annotations

import numpy as np

from objgauss.gaussians import GaussianCloud

SH_C0 = 0.28209479177387814


def extract_features(
    cloud: GaussianCloud,
    *,
    spatial_weight: float = 1.0,
    color_weight: float = 0.5,
    opacity_weight: float = 0.2,
    normalize: bool = True,
) -> np.ndarray:
    """Build `[xyz, rgb, opacity]` features for object clustering."""

    xyz = positions(cloud)
    rgb = colors(cloud)
    alpha = opacity(cloud)[:, None]

    blocks = [
        (xyz, spatial_weight),
        (rgb, color_weight),
        (alpha, opacity_weight),
    ]
    features = np.concatenate(
        [_normalize(block) * weight if normalize else block * weight for block, weight in blocks],
        axis=1,
    )
    return features.astype(np.float32, copy=False)


def positions(cloud: GaussianCloud) -> np.ndarray:
    cloud.require_fields(("x", "y", "z"))
    return _stack_fields(cloud.vertices, ("x", "y", "z")).astype(np.float32, copy=False)


def colors(cloud: GaussianCloud) -> np.ndarray:
    fields = cloud.fields
    if all(name in fields for name in ("red", "green", "blue")):
        rgb = _stack_fields(cloud.vertices, ("red", "green", "blue")).astype(np.float32)
        if rgb.size and np.nanmax(rgb) > 1.0:
            rgb /= 255.0
        return np.clip(rgb, 0.0, 1.0)

    if all(name in fields for name in ("f_dc_0", "f_dc_1", "f_dc_2")):
        dc = _stack_fields(cloud.vertices, ("f_dc_0", "f_dc_1", "f_dc_2")).astype(np.float32)
        return np.clip(dc * SH_C0 + 0.5, 0.0, 1.0)

    return np.zeros((cloud.count, 3), dtype=np.float32)


def opacity(cloud: GaussianCloud) -> np.ndarray:
    if "opacity" not in cloud.fields:
        return np.ones(cloud.count, dtype=np.float32)

    values = cloud.vertices["opacity"].astype(np.float32, copy=False)
    if values.size and np.nanmin(values) >= 0.0 and np.nanmax(values) <= 1.0:
        return values
    return _sigmoid(values)


def _stack_fields(vertices: np.ndarray, names: tuple[str, ...]) -> np.ndarray:
    return np.column_stack([vertices[name] for name in names])


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.shape[0] == 0:
        return values
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (values - mean) / std


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-values))
