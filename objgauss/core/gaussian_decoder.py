from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from objgauss.core.object_state import ObjectStateProjection, validate_assignment_matrix

OBJECT_STATE_GAUSSIAN_DECODE_SCHEMA = "objgauss-object-state-gaussian-decode-v1"
OBJECT_STATE_GAUSSIAN_POLICY = "object-state-synthetic-isotropic-gaussian-v1"


@dataclass(frozen=True)
class ObjectStateGaussianDecode:
    schema: str
    gaussian_policy: str
    color_policy: str
    geometry_policy: str
    opacity_policy: str
    object_count: int
    gaussian_count: int
    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray
    object_ids: np.ndarray
    object_colors: np.ndarray
    object_opacity_logits: np.ndarray | None
    object_opacity_scales: np.ndarray | None

    def as_dict(self) -> dict[str, Any]:
        object_opacity_enabled = self.object_opacity_logits is not None
        shapes: dict[str, list[int] | None] = {
            "means": list(self.means.shape),
            "quats": list(self.quats.shape),
            "scales": list(self.scales.shape),
            "opacities": list(self.opacities.shape),
            "colors": list(self.colors.shape),
            "object_ids": list(self.object_ids.shape),
            "object_colors": list(self.object_colors.shape),
            "object_opacity_logits": None
            if self.object_opacity_logits is None
            else list(self.object_opacity_logits.shape),
            "object_opacity_scales": None
            if self.object_opacity_scales is None
            else list(self.object_opacity_scales.shape),
        }
        return {
            "schema": self.schema,
            "gaussian_policy": self.gaussian_policy,
            "color_policy": self.color_policy,
            "geometry_policy": self.geometry_policy,
            "opacity_policy": self.opacity_policy,
            "object_count": int(self.object_count),
            "gaussian_count": int(self.gaussian_count),
            "shapes": shapes,
            "object_id_source": "ObjectStateProjection.derived_object_ids",
            "differentiable_fields": [
                "decoder.object_colors",
                "assignment",
                *(
                    ["decoder.object_opacity_logits"]
                    if object_opacity_enabled
                    else []
                ),
            ],
            "frozen_fields": [
                "means",
                "quats",
                "scales",
                *(
                    ["source_opacities"]
                    if object_opacity_enabled
                    else ["opacities"]
                ),
            ],
            "object_opacity_scale_policy": (
                "sigmoid-clamp-object-opacity-v1" if object_opacity_enabled else None
            ),
        }


def decode_gaussian_from_object_state(
    positions: np.ndarray,
    projection: ObjectStateProjection,
    object_colors: np.ndarray,
    *,
    object_opacity_logits: np.ndarray | None = None,
    default_scale: float = 0.01,
    default_opacity: float = 1.0,
    min_object_opacity_scale: float = 0.05,
    max_object_opacity_scale: float = 1.0,
) -> ObjectStateGaussianDecode:
    """Decode ObjectState slots into renderer-native Gaussian parameter arrays.

    v1 keeps geometry, orientation, and cameras frozen. The default path keeps
    opacity constant; callers can explicitly pass object-level opacity logits
    to derive per-Gaussian opacity without introducing per-Gaussian trainable
    opacity parameters.
    """

    xyz = _array2d(positions, "positions", columns=3)
    assignment = validate_assignment_matrix(projection.assignment, evidence_count=xyz.shape[0])
    if len(projection.states) != assignment.shape[1]:
        raise ValueError("projection states must match assignment slot count")
    if projection.derived_object_ids.shape[0] != xyz.shape[0]:
        raise ValueError("projection derived_object_ids must match positions")
    colors = _array2d(object_colors, "object_colors", columns=3)
    if colors.shape[0] != assignment.shape[1]:
        raise ValueError("object_colors rows must match projection slots")
    if default_scale <= 0.0:
        raise ValueError("default_scale must be > 0")
    if not 0.0 <= default_opacity <= 1.0:
        raise ValueError("default_opacity must be in [0, 1]")
    if not 0.0 <= min_object_opacity_scale <= max_object_opacity_scale <= 1.0:
        raise ValueError("object opacity scale bounds must satisfy 0 <= min <= max <= 1")

    gaussian_colors = np.clip(assignment @ colors, 0.0, 1.0).astype(np.float32, copy=False)
    quats = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    quats[:, 0] = 1.0
    scales = np.full((xyz.shape[0], 3), float(default_scale), dtype=np.float32)
    opacity_policy = "constant-opacity-v1"
    opacity_logits = None
    opacity_scales = None
    if object_opacity_logits is None:
        opacities = np.full((xyz.shape[0],), float(default_opacity), dtype=np.float32)
    else:
        opacity_logits = _array1d(object_opacity_logits, "object_opacity_logits")
        if opacity_logits.shape[0] != assignment.shape[1]:
            raise ValueError("object_opacity_logits length must match projection slots")
        opacity_scales = object_opacity_scales_from_logits(
            opacity_logits,
            min_scale=min_object_opacity_scale,
            max_scale=max_object_opacity_scale,
        )
        opacities = np.clip(
            float(default_opacity) * (assignment @ opacity_scales),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
        opacity_policy = "object-opacity-soft-assignment-v1"

    return ObjectStateGaussianDecode(
        schema=OBJECT_STATE_GAUSSIAN_DECODE_SCHEMA,
        gaussian_policy=OBJECT_STATE_GAUSSIAN_POLICY,
        color_policy="object-color-soft-assignment-v1",
        geometry_policy="freeze-source-means-synthetic-isotropic-scale-v1",
        opacity_policy=opacity_policy,
        object_count=int(assignment.shape[1]),
        gaussian_count=int(xyz.shape[0]),
        means=xyz.astype(np.float32, copy=False),
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=gaussian_colors,
        object_ids=projection.derived_object_ids.astype(np.int32, copy=False),
        object_colors=np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False),
        object_opacity_logits=(
            None if opacity_logits is None else opacity_logits.astype(np.float32, copy=True)
        ),
        object_opacity_scales=(
            None if opacity_scales is None else opacity_scales.astype(np.float32, copy=True)
        ),
    )


def object_opacity_scales_from_logits(
    object_opacity_logits: np.ndarray,
    *,
    min_scale: float = 0.05,
    max_scale: float = 1.0,
) -> np.ndarray:
    logits = _array1d(object_opacity_logits, "object_opacity_logits")
    if not 0.0 <= min_scale <= max_scale <= 1.0:
        raise ValueError("object opacity scale bounds must satisfy 0 <= min <= max <= 1")
    clipped_logits = np.clip(logits, -60.0, 60.0)
    sigmoid = 1.0 / (1.0 + np.exp(-clipped_logits))
    return np.clip(sigmoid, min_scale, max_scale).astype(np.float32, copy=False)


def _array1d(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a 1D array")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one value")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def _array2d(value: np.ndarray, label: str, *, columns: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{label} must have {columns} columns")
    if array.shape[1] == 0:
        raise ValueError(f"{label} must have at least one column")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array
