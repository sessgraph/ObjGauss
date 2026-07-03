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

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "gaussian_policy": self.gaussian_policy,
            "color_policy": self.color_policy,
            "geometry_policy": self.geometry_policy,
            "opacity_policy": self.opacity_policy,
            "object_count": int(self.object_count),
            "gaussian_count": int(self.gaussian_count),
            "shapes": {
                "means": list(self.means.shape),
                "quats": list(self.quats.shape),
                "scales": list(self.scales.shape),
                "opacities": list(self.opacities.shape),
                "colors": list(self.colors.shape),
                "object_ids": list(self.object_ids.shape),
                "object_colors": list(self.object_colors.shape),
            },
            "object_id_source": "ObjectStateProjection.derived_object_ids",
            "differentiable_fields": ["object_colors", "assignment"],
            "frozen_fields": ["means", "quats", "scales", "opacities"],
        }


def decode_gaussian_from_object_state(
    positions: np.ndarray,
    projection: ObjectStateProjection,
    object_colors: np.ndarray,
    *,
    default_scale: float = 0.01,
    default_opacity: float = 1.0,
) -> ObjectStateGaussianDecode:
    """Decode ObjectState slots into renderer-native Gaussian parameter arrays.

    v1 keeps geometry, orientation, opacity, and cameras frozen. The trainable
    path is deliberately narrow: object-level colors and soft assignments
    produce per-Gaussian colors for the differentiable renderer.
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

    gaussian_colors = np.clip(assignment @ colors, 0.0, 1.0).astype(np.float32, copy=False)
    quats = np.zeros((xyz.shape[0], 4), dtype=np.float32)
    quats[:, 0] = 1.0
    scales = np.full((xyz.shape[0], 3), float(default_scale), dtype=np.float32)
    opacities = np.full((xyz.shape[0],), float(default_opacity), dtype=np.float32)

    return ObjectStateGaussianDecode(
        schema=OBJECT_STATE_GAUSSIAN_DECODE_SCHEMA,
        gaussian_policy=OBJECT_STATE_GAUSSIAN_POLICY,
        color_policy="object-color-soft-assignment-v1",
        geometry_policy="freeze-source-means-synthetic-isotropic-scale-v1",
        opacity_policy="constant-opacity-v1",
        object_count=int(assignment.shape[1]),
        gaussian_count=int(xyz.shape[0]),
        means=xyz.astype(np.float32, copy=False),
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=gaussian_colors,
        object_ids=projection.derived_object_ids.astype(np.int32, copy=False),
        object_colors=np.clip(colors, 0.0, 1.0).astype(np.float32, copy=False),
    )


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
