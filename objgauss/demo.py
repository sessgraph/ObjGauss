from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile

import numpy as np

from objgauss.clustering import summarize_labels
from objgauss.features import positions
from objgauss.gaussians import GaussianCloud
from objgauss.mask_voting import (
    project_points,
    train_object_field_from_votes,
    training_summary,
    vote_masks_to_gaussians,
)
from objgauss.object_field import ObjectField, save_object_field
from objgauss.ply import read_ply, write_ply
from objgauss.segment import apply_object_colors, assign_object_ids


@dataclass(frozen=True)
class V1ClosureDemoResult:
    manifest_path: Path
    mask_manifest_path: Path
    initial_field_path: Path
    trained_field_path: Path
    output_ply_path: Path
    public_ply_path: Path | None
    gaussian_count: int
    object_count: int
    supervised_gaussians: int
    initial_loss: float
    final_loss: float


def build_v1_closure_demo(
    *,
    input_ply: str | Path = "public/samples/plush_objects.ply",
    splat_path: str | Path = "public/samples/plush.splat",
    output_dir: str | Path = "outputs/demos/v1-closure",
    public_dir: str | Path | None = "public/samples",
    image_size: int = 512,
    iterations: int = 160,
    learning_rate: float = 1.0,
) -> V1ClosureDemoResult:
    input_ply = Path(input_ply)
    splat_path = Path(splat_path)
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    cloud = read_ply(input_ply)
    if "object_id" not in cloud.fields:
        raise ValueError("v1 closure demo input PLY must include object_id")
    labels = _compact_labels(cloud.vertices["object_id"].astype(np.int32, copy=False))
    object_count = int(labels.max()) + 1
    field = ObjectField(np.zeros((cloud.count, object_count), dtype=np.float32))

    mask_manifest_path = output_dir / "mask-manifest.json"
    mask_summary = _write_projection_mask_manifest(
        cloud=cloud,
        labels=labels,
        path=mask_manifest_path,
        masks_dir=masks_dir,
        image_size=image_size,
    )
    votes = vote_masks_to_gaussians(cloud, mask_manifest_path, slots=object_count)
    training = train_object_field_from_votes(
        field,
        votes,
        iterations=iterations,
        learning_rate=learning_rate,
    )

    initial_field_path = output_dir / "object_field_initial.npz"
    trained_field_path = output_dir / "object_field_trained.npz"
    output_ply_path = output_dir / "plush_v1_objects.ply"
    save_object_field(initial_field_path, field)
    save_object_field(trained_field_path, training.field)

    labeled = assign_object_ids(cloud, training.field.labels())
    colored = apply_object_colors(labeled)
    write_ply(output_ply_path, colored, fmt="binary_little_endian")

    public_ply_path = None
    if public_dir is not None:
        public_ply_path = Path(public_dir) / "plush_v1_objects.ply"
        public_ply_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(output_ply_path, public_ply_path)

    manifest_path = output_dir / "v1-closure-manifest.json"
    manifest = {
        "demo": "ObjGauss v1 closure",
        "input_ply": str(input_ply),
        "splat_path": str(splat_path),
        "real_splat_exists": splat_path.exists(),
        "gaussian_count": cloud.count,
        "object_count": object_count,
        "mask_manifest": str(mask_manifest_path),
        "initial_field": str(initial_field_path),
        "trained_field": str(trained_field_path),
        "output_ply": str(output_ply_path),
        "public_ply": str(public_ply_path) if public_ply_path else None,
        "source_object_counts": summarize_labels(labels),
        "trained_object_counts": summarize_labels(training.field.labels()),
        "mask_summary": mask_summary,
        "training": training_summary(training),
        "acceptance": {
            "real_3dgs_scene_can_render": splat_path.exists(),
            "object_field_saved": trained_field_path.exists(),
            "mask_votes_supervise_gaussians": training.supervised_gaussians > 0,
            "projection_loss_decreased": training.final_loss < training.initial_loss,
            "viewer_ply_available": output_ply_path.exists(),
            "public_viewer_ply_available": public_ply_path.exists() if public_ply_path else False,
        },
        "viewer_steps": [
            "打开前端页面",
            "在素材库加载 ObjGauss v1 闭环样例",
            "真实 Splat 模式查看原始 3DGS 外观",
            "切换对象聚类色或执行隔离/删除后进入点云编辑视图",
            "点击对象列表里的对象，执行只看所选或预览删除",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    return V1ClosureDemoResult(
        manifest_path=manifest_path,
        mask_manifest_path=mask_manifest_path,
        initial_field_path=initial_field_path,
        trained_field_path=trained_field_path,
        output_ply_path=output_ply_path,
        public_ply_path=public_ply_path,
        gaussian_count=cloud.count,
        object_count=object_count,
        supervised_gaussians=training.supervised_gaussians,
        initial_loss=training.initial_loss,
        final_loss=training.final_loss,
    )


def _write_projection_mask_manifest(
    *,
    cloud: GaussianCloud,
    labels: np.ndarray,
    path: Path,
    masks_dir: Path,
    image_size: int,
) -> dict[str, int]:
    if image_size <= 0:
        raise ValueError("image_size must be positive")
    xyz = positions(cloud)
    view_directions = (
        np.array([1.0, 0.55, 1.0], dtype=np.float32),
        np.array([-1.0, 0.55, 1.0], dtype=np.float32),
        np.array([0.15, 0.8, -1.0], dtype=np.float32),
    )
    frames = []
    visible_count = 0
    mask_count = 0
    for frame_index, direction in enumerate(view_directions):
        c2w = _demo_camera_transform(xyz, direction=direction)
        projection = project_points(
            xyz,
            transform_matrix=c2w,
            width=image_size,
            height=image_size,
            camera_angle_x=np.pi / 3.0,
        )
        masks = []
        for label in sorted(int(value) for value in np.unique(labels)):
            selected = projection.visible & (labels == label)
            if not np.any(selected):
                continue
            mask = np.zeros((image_size, image_size), dtype=bool)
            x = np.clip(np.floor(projection.u[selected]).astype(np.int64), 0, image_size - 1)
            y = np.clip(np.floor(projection.v[selected]).astype(np.int64), 0, image_size - 1)
            mask[y, x] = True
            mask_path = masks_dir / f"frame_{frame_index:03d}_slot_{label}.npy"
            np.save(mask_path, mask)
            masks.append(
                {
                    "slot": label,
                    "label": f"object-{label}",
                    "mask_path": str(mask_path.relative_to(path.parent)),
                }
            )
            visible_count += int(np.count_nonzero(selected))
            mask_count += 1
        frames.append(
            {
                "name": f"synthetic-acceptance-view-{frame_index}",
                "transform_matrix": c2w.tolist(),
                "masks": masks,
            }
        )

    payload = {
        "width": image_size,
        "height": image_size,
        "camera_angle_x": float(np.pi / 3.0),
        "frames": frames,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return {
        "frames": len(frames),
        "masks": mask_count,
        "visible_gaussians": visible_count,
    }


def _demo_camera_transform(xyz: np.ndarray, *, direction: np.ndarray) -> np.ndarray:
    center = xyz.mean(axis=0)
    span = np.ptp(xyz, axis=0)
    distance = max(float(np.linalg.norm(span)) * 1.8, 2.0)
    eye = center + _normalize(direction) * distance
    return _look_at(eye, center)


def _look_at(eye: np.ndarray, target: np.ndarray) -> np.ndarray:
    up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    backward = _normalize(eye - target)
    right = _normalize(np.cross(up, backward))
    true_up = np.cross(backward, right)
    matrix = np.eye(4, dtype=np.float32)
    matrix[:3, 0] = right
    matrix[:3, 1] = true_up
    matrix[:3, 2] = backward
    matrix[:3, 3] = eye
    return matrix


def _normalize(value: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(value))
    if norm < 1e-8:
        raise ValueError("cannot normalize a zero vector")
    return value / norm


def _compact_labels(labels: np.ndarray) -> np.ndarray:
    unique = sorted(int(value) for value in np.unique(labels))
    mapping = {value: index for index, value in enumerate(unique)}
    return np.array([mapping[int(value)] for value in labels], dtype=np.int32)
