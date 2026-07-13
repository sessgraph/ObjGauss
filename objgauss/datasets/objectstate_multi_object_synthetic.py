from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from objgauss.core.gaussian import GaussianCloud

OBJECTSTATE_MULTI_OBJECT_DATASET_SCHEMA = (
    "objgauss-objectstate-multi-object-synthetic-v1"
)

_INSTANCE_SPECS = (
    {"instance_id": 0, "name": "red_cube_a", "shape_id": 0, "shape": "cube", "rgb": (196, 55, 55)},
    {"instance_id": 1, "name": "red_cube_b", "shape_id": 0, "shape": "cube", "rgb": (196, 55, 55)},
    {"instance_id": 2, "name": "blue_cup", "shape_id": 1, "shape": "cup", "rgb": (60, 125, 220)},
    {"instance_id": 3, "name": "gray_tool", "shape_id": 2, "shape": "tool", "rgb": (105, 110, 120)},
)

__all__ = (
    "OBJECTSTATE_MULTI_OBJECT_DATASET_SCHEMA",
    "MultiObjectSyntheticDataset",
    "build_multi_object_synthetic_dataset",
)


@dataclass(frozen=True)
class MultiObjectSyntheticDataset:
    cloud: GaussianCloud
    scenes: tuple[dict[str, Any], ...]
    train_scene_ids: tuple[int, ...]
    heldout_scene_ids: tuple[int, ...]
    seed: int
    points_per_instance: int
    schema: str = OBJECTSTATE_MULTI_OBJECT_DATASET_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": "procedural_multi_object_instance_dataset",
            "source": {
                "type": "procedural_instance_authorship",
                "target_field": "gt_instance_id",
                "target_derived_from_rgb": False,
                "model_feature_fields": [
                    "x",
                    "y",
                    "z",
                    "red",
                    "green",
                    "blue",
                    "opacity",
                ],
                "dataset_only_fields": ["scene_id", "gt_instance_id", "shape_id"],
            },
            "counts": {
                "scenes": len(self.scenes),
                "gaussians": self.cloud.count,
                "instances_per_scene": len(_INSTANCE_SPECS),
            },
            "split": {
                "field": "scene_id",
                "policy": "deterministic_complete_scene_holdout",
                "train_scene_ids": list(self.train_scene_ids),
                "heldout_scene_ids": list(self.heldout_scene_ids),
                "scene_overlap_count": len(
                    set(self.train_scene_ids) & set(self.heldout_scene_ids)
                ),
            },
            "stress_contract": {
                "same_color_instance_ids": [0, 1],
                "same_color_rgb": list(_INSTANCE_SPECS[0]["rgb"]),
                "contact_scene_ids": [
                    scene["scene_id"] for scene in self.scenes if scene["contact"]
                ],
                "partial_observation_scene_ids": [
                    scene["scene_id"]
                    for scene in self.scenes
                    if scene["partial_observation"]
                ],
                "layout_varies_by_scene": True,
                "similar_shape_instance_ids": [0, 1],
            },
            "instances": [dict(spec) for spec in _INSTANCE_SPECS],
            "scenes": [dict(scene) for scene in self.scenes],
            "seed": self.seed,
            "points_per_instance": self.points_per_instance,
            "claim_policy": {
                "synthetic_instance_evidence_only": True,
                "does_not_claim_real_identity": True,
                "does_not_claim_prediction_or_intervention": True,
            },
        }


def build_multi_object_synthetic_dataset(
    *,
    scene_count: int = 12,
    points_per_instance: int = 128,
    heldout_stride: int = 3,
    split_seed: int = 0,
    seed: int = 20260713,
) -> MultiObjectSyntheticDataset:
    """Build deterministic multi-object Gaussian scenes with independent GT.

    The two cubes intentionally share shape and RGB.  `gt_instance_id` comes
    from procedural object authorship and is never computed from row features.
    """

    if scene_count < 6:
        raise ValueError("multi-object benchmark requires at least six scenes")
    if points_per_instance < 64:
        raise ValueError("points_per_instance must be >= 64")
    if heldout_stride < 2:
        raise ValueError("heldout_stride must be >= 2")

    scene_ids = tuple(range(scene_count))
    offset = int(split_seed) % heldout_stride
    heldout_scene_ids = tuple(
        scene_id
        for index, scene_id in enumerate(scene_ids)
        if index % heldout_stride == offset
    )
    train_scene_ids = tuple(
        scene_id for scene_id in scene_ids if scene_id not in heldout_scene_ids
    )
    if not train_scene_ids or not heldout_scene_ids:
        raise ValueError("multi-object benchmark requires train and held-out scenes")

    rows: list[np.ndarray] = []
    scenes: list[dict[str, Any]] = []
    for scene_id in scene_ids:
        rng = np.random.default_rng(seed + scene_id * 7919)
        contact = scene_id % 2 == 0
        partial_observation = scene_id % 3 == 0
        centers = _scene_centers(rng, contact=contact)
        scene_rows: list[np.ndarray] = []
        observed_counts: dict[int, int] = {}
        for spec in _INSTANCE_SPECS:
            instance_id = int(spec["instance_id"])
            local = _sample_shape(
                str(spec["shape"]),
                points_per_instance,
                rng,
            )
            yaw = float(rng.uniform(-np.pi, np.pi))
            points = _rotate_z(local, yaw) + centers[instance_id]
            if partial_observation and instance_id == 1:
                # A deterministic missing-surface stressor that represents a
                # partially observed instance.  GT remains instance-authored.
                view_axis = np.asarray([np.cos(yaw), np.sin(yaw), 0.0])
                depth = (points - centers[instance_id]) @ view_axis
                keep = depth <= np.quantile(depth, 0.65)
                points = points[keep]
            observed_counts[instance_id] = int(points.shape[0])
            scene_rows.append(
                _object_vertices(
                    points,
                    scene_id=scene_id,
                    instance_id=instance_id,
                    shape_id=int(spec["shape_id"]),
                    rgb=tuple(int(item) for item in spec["rgb"]),
                    rng=rng,
                )
            )
        vertices = np.concatenate(scene_rows)
        rows.append(vertices)
        scenes.append(
            {
                "scene_id": scene_id,
                "split": "heldout" if scene_id in heldout_scene_ids else "train",
                "gaussian_count": int(vertices.shape[0]),
                "instance_ids": [int(spec["instance_id"]) for spec in _INSTANCE_SPECS],
                "contact": contact,
                "contact_instance_ids": [0, 1] if contact else [],
                "partial_observation": partial_observation,
                "partial_observation_instance_ids": [1] if partial_observation else [],
                "observed_point_counts": {
                    str(instance_id): observed_counts[instance_id]
                    for instance_id in sorted(observed_counts)
                },
            }
        )

    cloud = GaussianCloud(
        vertices=np.concatenate(rows),
        comments=(
            "ObjGauss procedural multi-object instance benchmark",
            "gt_instance_id is dataset-only supervision, not an RGB-derived label",
        ),
        source_format="procedural-instance-v1",
    )
    return MultiObjectSyntheticDataset(
        cloud=cloud,
        scenes=tuple(scenes),
        train_scene_ids=train_scene_ids,
        heldout_scene_ids=heldout_scene_ids,
        seed=seed,
        points_per_instance=points_per_instance,
    )


def _scene_centers(rng: np.random.Generator, *, contact: bool) -> np.ndarray:
    angle = float(rng.uniform(-np.pi, np.pi))
    radius = float(rng.uniform(0.85, 1.15))
    anchors = np.asarray(
        [
            [-radius, -0.55, 0.0],
            [radius, 0.45, 0.0],
            [-0.25, 0.9, 0.0],
            [0.35, -0.95, 0.0],
        ],
        dtype=np.float32,
    )
    permutation = rng.permutation(anchors.shape[0])
    centers = anchors[permutation].copy()
    rotation = np.asarray(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    centers = centers @ rotation.T
    centers[:, :2] += rng.normal(0.0, 0.08, size=(centers.shape[0], 2))
    if contact:
        direction = np.asarray([np.cos(angle), np.sin(angle), 0.0], dtype=np.float32)
        centers[1] = centers[0] + direction * 0.56
    return centers


def _sample_shape(
    shape: str,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if shape == "cube":
        points = rng.uniform(-0.28, 0.28, size=(count, 3)).astype(np.float32)
        points[:, 2] += 0.30
        return points
    if shape == "cup":
        angles = rng.uniform(0.0, 2.0 * np.pi, size=count)
        heights = rng.uniform(0.0, 0.72, size=count)
        radius = rng.normal(0.27, 0.018, size=count)
        points = np.column_stack(
            [radius * np.cos(angles), radius * np.sin(angles), heights]
        ).astype(np.float32)
        rim = rng.random(count) < 0.18
        points[rim, 2] = rng.normal(0.72, 0.012, size=int(rim.sum()))
        return points
    if shape == "tool":
        handle_count = int(round(count * 0.72))
        handle = rng.uniform(
            [-0.48, -0.09, 0.08],
            [0.32, 0.09, 0.22],
            size=(handle_count, 3),
        )
        head = rng.uniform(
            [0.22, -0.34, 0.04],
            [0.46, 0.34, 0.30],
            size=(count - handle_count, 3),
        )
        return np.concatenate([handle, head]).astype(np.float32)
    raise ValueError(f"unsupported procedural shape: {shape}")


def _rotate_z(points: np.ndarray, yaw: float) -> np.ndarray:
    rotation = np.asarray(
        [
            [np.cos(yaw), -np.sin(yaw), 0.0],
            [np.sin(yaw), np.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return np.asarray(points, dtype=np.float32) @ rotation.T


def _object_vertices(
    points: np.ndarray,
    *,
    scene_id: int,
    instance_id: int,
    shape_id: int,
    rgb: tuple[int, int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("opacity", "f4"),
            ("scale_0", "f4"),
            ("scale_1", "f4"),
            ("scale_2", "f4"),
            ("rot_0", "f4"),
            ("rot_1", "f4"),
            ("rot_2", "f4"),
            ("rot_3", "f4"),
            ("scene_id", "i4"),
            ("gt_instance_id", "i4"),
            ("shape_id", "i4"),
        ]
    )
    vertices = np.zeros(points.shape[0], dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = points[:, axis]
    color_noise = rng.normal(0.0, 2.0, size=(points.shape[0], 3))
    resolved_rgb = np.clip(np.asarray(rgb)[None, :] + color_noise, 0, 255).astype(np.uint8)
    for channel, name in enumerate(("red", "green", "blue")):
        vertices[name] = resolved_rgb[:, channel]
    vertices["opacity"] = rng.uniform(0.82, 0.98, size=points.shape[0]).astype(np.float32)
    for name in ("scale_0", "scale_1", "scale_2"):
        # Canonical GaussianCloud scale fields are direct positive scales.
        # Keeping them positive also makes the same rows valid for `.splat`
        # browser serialization; log-scale values belong at model adapters.
        vertices[name] = np.float32(0.035)
    vertices["rot_0"] = np.float32(1.0)
    vertices["scene_id"] = scene_id
    vertices["gt_instance_id"] = instance_id
    vertices["shape_id"] = shape_id
    return vertices
