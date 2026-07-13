from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io import append_or_replace_property
from objgauss.datasets.objectstate_multi_object_synthetic import (
    _INSTANCE_SPECS,
    _object_vertices,
    _rotate_z,
    _sample_shape,
)

OBJECTSTATE_MODEL_DIAGNOSTIC_DATASET_SCHEMA = (
    "objgauss-objectstate-model-diagnostic-synthetic-v1"
)
OBJECTSTATE_MODEL_DIAGNOSTIC_CASES = (
    "near_same_color_cubes",
    "cube_cup_contact",
    "cube_behind_cup",
    "cross_view",
)

_CASE_IDS = {
    "train_layout": 0,
    "near_same_color_cubes": 1,
    "cube_cup_contact": 2,
    "cube_behind_cup": 3,
    "cross_view": 4,
}
_HELDOUT_SCENES = (0, 3, 6, 9, 12)

__all__ = (
    "OBJECTSTATE_MODEL_DIAGNOSTIC_CASES",
    "OBJECTSTATE_MODEL_DIAGNOSTIC_DATASET_SCHEMA",
    "ObjectStateModelDiagnosticDataset",
    "build_objectstate_model_diagnostic_dataset",
    "diagnostic_semantic_proxy",
)


@dataclass(frozen=True)
class ObjectStateModelDiagnosticDataset:
    cloud: GaussianCloud
    observations: tuple[dict[str, Any], ...]
    train_scene_ids: tuple[int, ...]
    heldout_scene_ids: tuple[int, ...]
    train_layout_ids: tuple[int, ...]
    heldout_layout_ids: tuple[int, ...]
    cross_view_pairs: tuple[dict[str, Any], ...]
    seed: int
    points_per_instance: int
    schema: str = OBJECTSTATE_MODEL_DIAGNOSTIC_DATASET_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": "objectstate_model_diagnostic_synthetic_dataset",
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
                    "semantic_shape_proxy",
                ],
                "dataset_only_fields": [
                    "scene_id",
                    "layout_id",
                    "view_id",
                    "case_id",
                    "gt_instance_id",
                    "shape_id",
                ],
            },
            "counts": {
                "observations": len(self.observations),
                "gaussians": self.cloud.count,
                "instances_per_observation": len(_INSTANCE_SPECS),
            },
            "split": {
                "field": "scene_id",
                "policy": "deterministic_complete_scene_holdout",
                "train_scene_ids": list(self.train_scene_ids),
                "heldout_scene_ids": list(self.heldout_scene_ids),
                "scene_overlap_count": len(
                    set(self.train_scene_ids) & set(self.heldout_scene_ids)
                ),
                "train_layout_ids": list(self.train_layout_ids),
                "heldout_layout_ids": list(self.heldout_layout_ids),
                "layout_overlap_count": len(
                    set(self.train_layout_ids) & set(self.heldout_layout_ids)
                ),
            },
            "hard_cases": list(OBJECTSTATE_MODEL_DIAGNOSTIC_CASES),
            "observations": [dict(item) for item in self.observations],
            "cross_view_pairs": [dict(item) for item in self.cross_view_pairs],
            "semantic_proxy": {
                "name": "procedural_shape_class_one_hot",
                "classes": ["cube", "cup", "tool"],
                "feature_dim": 3,
                "uses_gt_instance_id": False,
                "same_vector_for_instance_ids": [0, 1],
                "role": "oracle_style_class_semantic_upper_bound",
                "deployable_teacher": False,
            },
            "seed": self.seed,
            "points_per_instance": self.points_per_instance,
            "claim_policy": {
                "synthetic_diagnostic_only": True,
                "semantic_proxy_is_not_learned_teacher": True,
                "does_not_claim_object_discovery": True,
                "does_not_claim_real_identity": True,
                "does_not_claim_prediction_or_intervention": True,
            },
        }


def build_objectstate_model_diagnostic_dataset(
    *,
    points_per_instance: int = 128,
    seed: int = 20260713,
) -> ObjectStateModelDiagnosticDataset:
    """Build eight train observations and five named held-out hard cases."""

    if points_per_instance < 64:
        raise ValueError("diagnostic points_per_instance must be >= 64")

    rows: list[np.ndarray] = []
    observations: list[dict[str, Any]] = []
    train_scene_ids = tuple(scene for scene in range(13) if scene not in _HELDOUT_SCENES)
    heldout_scene_ids = _HELDOUT_SCENES

    for scene_id in range(13):
        spec = _observation_spec(scene_id, seed=seed)
        layout_rng = np.random.default_rng(seed + int(spec["layout_id"]) * 7919)
        camera_yaw = float(spec["camera_yaw"])
        scene_rows: list[np.ndarray] = []
        observed_counts: dict[int, int] = {}
        for instance in _INSTANCE_SPECS:
            instance_id = int(instance["instance_id"])
            local = _sample_shape(
                str(instance["shape"]),
                points_per_instance,
                layout_rng,
            )
            object_yaw = float(layout_rng.uniform(-np.pi, np.pi))
            world = _rotate_z(local, object_yaw) + np.asarray(
                spec["centers"][instance_id],
                dtype=np.float32,
            )
            camera = _camera_coordinates(world, yaw=camera_yaw)
            camera = _visibility_filter(
                camera,
                case=str(spec["case"]),
                instance_id=instance_id,
                view_id=int(spec["view_id"]),
            )
            observed_counts[instance_id] = int(camera.shape[0])
            vertices = _object_vertices(
                camera,
                scene_id=scene_id,
                instance_id=instance_id,
                shape_id=int(instance["shape_id"]),
                rgb=tuple(int(value) for value in instance["rgb"]),
                rng=layout_rng,
            )
            for name, value in (
                ("layout_id", int(spec["layout_id"])),
                ("view_id", int(spec["view_id"])),
                ("case_id", _CASE_IDS[str(spec["case"])]),
            ):
                vertices = append_or_replace_property(
                    vertices,
                    name,
                    np.full(vertices.shape[0], value, dtype=np.int32),
                    "i4",
                )
            scene_rows.append(vertices)

        vertices = np.concatenate(scene_rows)
        shuffle_rng = np.random.default_rng(seed + scene_id * 104729 + 17)
        vertices = vertices[shuffle_rng.permutation(vertices.shape[0])]
        rows.append(vertices)
        observations.append(
            {
                "scene_id": scene_id,
                "layout_id": int(spec["layout_id"]),
                "view_id": int(spec["view_id"]),
                "split": "heldout" if scene_id in heldout_scene_ids else "train",
                "case": str(spec["case"]),
                "camera_yaw_radians": camera_yaw,
                "gaussian_count": int(vertices.shape[0]),
                "instance_ids": [int(item["instance_id"]) for item in _INSTANCE_SPECS],
                "observed_point_counts": {
                    str(key): observed_counts[key] for key in sorted(observed_counts)
                },
                "contact_pairs": [list(pair) for pair in spec["contact_pairs"]],
                "occluded_instance_ids": list(spec["occluded_instance_ids"]),
            }
        )

    train_layout_ids = tuple(
        sorted({int(row["layout_id"]) for row in observations if row["split"] == "train"})
    )
    heldout_layout_ids = tuple(
        sorted({int(row["layout_id"]) for row in observations if row["split"] == "heldout"})
    )
    cloud = GaussianCloud(
        vertices=np.concatenate(rows),
        comments=(
            "ObjGauss ObjectState Model Diagnostic 001 synthetic hard cases",
            "gt_instance_id and shape_id are dataset-only; semantic proxy is class-level",
        ),
        source_format="procedural-model-diagnostic-v1",
    )
    return ObjectStateModelDiagnosticDataset(
        cloud=cloud,
        observations=tuple(observations),
        train_scene_ids=train_scene_ids,
        heldout_scene_ids=heldout_scene_ids,
        train_layout_ids=train_layout_ids,
        heldout_layout_ids=heldout_layout_ids,
        cross_view_pairs=(
            {
                "pair_id": "layout-103-cross-view",
                "layout_id": 103,
                "anchor_scene_id": 9,
                "target_scene_id": 12,
            },
        ),
        seed=seed,
        points_per_instance=points_per_instance,
    )


def diagnostic_semantic_proxy(cloud: GaussianCloud) -> np.ndarray:
    """Return class semantics shared by both cube instances, never instance ids."""

    if "shape_id" not in cloud.fields:
        raise ValueError("diagnostic semantic proxy requires shape_id")
    shape = np.asarray(cloud.vertices["shape_id"], dtype=np.int64)
    if np.any(shape < 0) or np.any(shape > 2):
        raise ValueError("diagnostic shape_id must be in [0,2]")
    features = np.zeros((cloud.count, 3), dtype=np.float32)
    features[np.arange(cloud.count), shape] = 1.0
    return features


def _observation_spec(scene_id: int, *, seed: int) -> dict[str, Any]:
    if scene_id == 0:
        return _spec(
            "near_same_color_cubes",
            layout_id=100,
            centers=((-0.28, 0.0, 0.0), (0.28, 0.0, 0.0), (-1.1, 0.9, 0.0), (1.1, -0.9, 0.0)),
            contact_pairs=((0, 1),),
        )
    if scene_id == 3:
        return _spec(
            "cube_cup_contact",
            layout_id=101,
            centers=((0.0, 0.0, 0.0), (-1.1, 0.9, 0.0), (0.53, 0.0, 0.0), (1.1, -0.9, 0.0)),
            contact_pairs=((0, 2),),
        )
    if scene_id == 6:
        return _spec(
            "cube_behind_cup",
            layout_id=102,
            centers=((-1.0, -0.7, 0.0), (0.0, 0.58, 0.0), (0.0, 0.0, 0.0), (1.1, -0.8, 0.0)),
            occluded_instance_ids=(1,),
        )
    if scene_id in {9, 12}:
        return _spec(
            "cross_view",
            layout_id=103,
            view_id=0 if scene_id == 9 else 1,
            camera_yaw=0.0 if scene_id == 9 else 2.2,
            centers=((-0.7, -0.2, 0.0), (0.7, 0.2, 0.0), (-0.25, 0.95, 0.0), (0.35, -0.95, 0.0)),
        )

    rng = np.random.default_rng(seed + scene_id * 4099)
    centers = np.asarray(
        [(-1.0, -0.6, 0.0), (1.0, 0.5, 0.0), (-0.3, 1.0, 0.0), (0.4, -1.0, 0.0)],
        dtype=np.float32,
    )
    centers = centers[rng.permutation(4)]
    centers[:, :2] += rng.normal(0.0, 0.08, size=(4, 2))
    return _spec(
        "train_layout",
        layout_id=scene_id,
        camera_yaw=float(rng.uniform(-0.8, 0.8)),
        centers=tuple(tuple(float(value) for value in row) for row in centers),
    )


def _spec(
    case: str,
    *,
    layout_id: int,
    centers: tuple[tuple[float, float, float], ...],
    view_id: int = 0,
    camera_yaw: float = 0.0,
    contact_pairs: tuple[tuple[int, int], ...] = (),
    occluded_instance_ids: tuple[int, ...] = (),
) -> dict[str, Any]:
    return {
        "case": case,
        "layout_id": layout_id,
        "view_id": view_id,
        "camera_yaw": camera_yaw,
        "centers": centers,
        "contact_pairs": contact_pairs,
        "occluded_instance_ids": occluded_instance_ids,
    }


def _camera_coordinates(points: np.ndarray, *, yaw: float) -> np.ndarray:
    return _rotate_z(np.asarray(points, dtype=np.float32), -float(yaw))


def _visibility_filter(
    points: np.ndarray,
    *,
    case: str,
    instance_id: int,
    view_id: int,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float32)
    if case == "cube_behind_cup" and instance_id == 1:
        # Depth-ordered proxy: cup and cube overlap in camera x/z; retain one
        # side of the farther cube to model a 65% missing-surface observation.
        return values[values[:, 0] <= np.quantile(values[:, 0], 0.35)]
    if case == "cross_view":
        quantile = np.quantile(values[:, 0], 0.85 if view_id == 0 else 0.15)
        return values[values[:, 0] <= quantile] if view_id == 0 else values[values[:, 0] >= quantile]
    return values
