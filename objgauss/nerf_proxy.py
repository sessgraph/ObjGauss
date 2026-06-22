from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from typing import Any

import numpy as np

from objgauss.clustering import summarize_labels
from objgauss.gaussians import GaussianCloud
from objgauss.mask_voting import (
    train_object_field_from_votes,
    training_summary,
    vote_masks_to_gaussians,
)
from objgauss.masks import read_png_rgba
from objgauss.object_field import ObjectField, save_object_field
from objgauss.ply import write_ply
from objgauss.segment import apply_object_colors, assign_object_ids
from objgauss.splat import write_splat


LEGO_COLOR_SLOTS = (
    {"slot": 0, "label": "yellow"},
    {"slot": 1, "label": "red"},
    {"slot": 2, "label": "dark"},
    {"slot": 3, "label": "other"},
)


@dataclass(frozen=True)
class LegoAlphaClosureResult:
    manifest_path: Path
    mask_manifest_path: Path
    raw_ply_path: Path
    splat_path: Path
    trained_field_path: Path
    output_ply_path: Path
    public_ply_path: Path | None
    public_splat_path: Path | None
    gaussian_count: int
    object_count: int
    supervised_gaussians: int
    initial_loss: float
    final_loss: float


def build_lego_alpha_closure_demo(
    *,
    dataset: str | Path = "outputs/assets/training/nerf-synthetic-lego",
    output_dir: str | Path = "outputs/demos/lego-alpha-closure",
    public_dir: str | Path | None = "public/samples",
    split: str = "train",
    max_frames: int = 12,
    sample_stride: int = 8,
    depth: float = 4.0,
    alpha_threshold: int = 16,
    iterations: int = 160,
    learning_rate: float = 1.0,
) -> LegoAlphaClosureResult:
    if max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if sample_stride < 1:
        raise ValueError("sample_stride must be >= 1")
    if depth <= 0:
        raise ValueError("depth must be > 0")

    dataset = Path(dataset)
    output_dir = Path(output_dir)
    masks_dir = output_dir / "masks"
    output_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    transforms_path = dataset / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise ValueError(f"missing NeRF transforms file: {transforms_path}")
    transforms = json.loads(transforms_path.read_text(encoding="utf-8"))
    camera_angle_x = float(transforms["camera_angle_x"])
    frames = transforms.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{transforms_path} must contain frames")

    cloud, mask_summary = _build_proxy_cloud_and_masks(
        dataset=dataset,
        frames=frames[:max_frames],
        split=split,
        camera_angle_x=camera_angle_x,
        masks_dir=masks_dir,
        mask_manifest_path=output_dir / "mask-manifest.json",
        sample_stride=sample_stride,
        depth=depth,
        alpha_threshold=alpha_threshold,
    )

    raw_ply_path = output_dir / "lego_proxy_raw.ply"
    splat_path = output_dir / "lego_proxy.splat"
    initial_field_path = output_dir / "object_field_initial.npz"
    trained_field_path = output_dir / "object_field_trained.npz"
    output_ply_path = output_dir / "lego_v1_objects.ply"
    write_ply(raw_ply_path, cloud, fmt="binary_little_endian")
    write_splat(splat_path, cloud)

    field = ObjectField(
        np.zeros((cloud.count, len(LEGO_COLOR_SLOTS)), dtype=np.float32)
    )
    save_object_field(initial_field_path, field)
    votes = vote_masks_to_gaussians(
        cloud,
        output_dir / "mask-manifest.json",
        slots=field.slots,
    )
    training = train_object_field_from_votes(
        field,
        votes,
        iterations=iterations,
        learning_rate=learning_rate,
    )
    save_object_field(trained_field_path, training.field)

    labeled = assign_object_ids(cloud, training.field.labels())
    colored = apply_object_colors(labeled)
    write_ply(output_ply_path, colored, fmt="binary_little_endian")

    public_ply_path = None
    public_splat_path = None
    if public_dir is not None:
        public_dir = Path(public_dir)
        public_dir.mkdir(parents=True, exist_ok=True)
        public_ply_path = public_dir / "lego_alpha_v1_objects.ply"
        public_splat_path = public_dir / "lego_alpha_proxy.splat"
        copyfile(output_ply_path, public_ply_path)
        copyfile(splat_path, public_splat_path)

    manifest_path = output_dir / "lego-alpha-closure-manifest.json"
    public_assets = bool(public_ply_path and public_splat_path and public_ply_path.exists() and public_splat_path.exists())
    manifest = {
        "demo": "ObjGauss Lego alpha closure proxy",
        "dataset": str(dataset),
        "split": split,
        "gaussian_source": "nerf_rgba_pose_proxy",
        "semantic_source": "nerf_rgba_color_masks",
        "note": (
            "This is a lightweight Gaussian proxy from posed RGBA images, "
            "not full 3DGS optimization."
        ),
        "gaussian_count": cloud.count,
        "object_count": len(LEGO_COLOR_SLOTS),
        "slots": list(LEGO_COLOR_SLOTS),
        "raw_ply": str(raw_ply_path),
        "splat_path": str(splat_path),
        "mask_manifest": str(output_dir / "mask-manifest.json"),
        "initial_field": str(initial_field_path),
        "trained_field": str(trained_field_path),
        "output_ply": str(output_ply_path),
        "public_ply": str(public_ply_path) if public_ply_path else None,
        "public_splat": str(public_splat_path) if public_splat_path else None,
        "sample": {
            "frames": mask_summary["frames"],
            "sample_stride": sample_stride,
            "depth": depth,
            "alpha_threshold": alpha_threshold,
        },
        "mask_summary": mask_summary,
        "trained_object_counts": summarize_labels(training.field.labels()),
        "training": training_summary(training),
        "acceptance": {
            "multiview_images_used": mask_summary["frames"] > 0,
            "gaussian_proxy_saved": raw_ply_path.exists() and splat_path.exists(),
            "real_mask_manifest_saved": (output_dir / "mask-manifest.json").exists(),
            "mask_votes_supervise_gaussians": training.supervised_gaussians > 0,
            "projection_loss_decreased": training.final_loss < training.initial_loss,
            "viewer_ply_available": output_ply_path.exists(),
            "public_assets_available": public_assets,
        },
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    return LegoAlphaClosureResult(
        manifest_path=manifest_path,
        mask_manifest_path=output_dir / "mask-manifest.json",
        raw_ply_path=raw_ply_path,
        splat_path=splat_path,
        trained_field_path=trained_field_path,
        output_ply_path=output_ply_path,
        public_ply_path=public_ply_path,
        public_splat_path=public_splat_path,
        gaussian_count=cloud.count,
        object_count=len(LEGO_COLOR_SLOTS),
        supervised_gaussians=training.supervised_gaussians,
        initial_loss=training.initial_loss,
        final_loss=training.final_loss,
    )


def _build_proxy_cloud_and_masks(
    *,
    dataset: Path,
    frames: list[Any],
    split: str,
    camera_angle_x: float,
    masks_dir: Path,
    mask_manifest_path: Path,
    sample_stride: int,
    depth: float,
    alpha_threshold: int,
) -> tuple[GaussianCloud, dict[str, Any]]:
    rows: list[np.ndarray] = []
    manifest_frames: list[dict[str, Any]] = []
    slot_pixel_counts = np.zeros(len(LEGO_COLOR_SLOTS), dtype=np.int64)
    sampled_slot_counts = np.zeros(len(LEGO_COLOR_SLOTS), dtype=np.int64)
    width = height = 0

    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("NeRF frame entries must be objects")
        image_path = _resolve_nerf_image(dataset, frame.get("file_path"))
        transform = np.asarray(frame.get("transform_matrix"), dtype=np.float32)
        if transform.shape != (4, 4):
            raise ValueError("NeRF frame transform_matrix must be 4x4")
        rgba = read_png_rgba(image_path)
        if width == 0:
            height, width = rgba.shape[:2]
        elif rgba.shape[:2] != (height, width):
            raise ValueError(f"{image_path} shape does not match previous frames")

        labels = _classify_lego_rgba(rgba)
        foreground = rgba[:, :, 3] >= alpha_threshold
        masks: list[dict[str, Any]] = []
        for slot in range(len(LEGO_COLOR_SLOTS)):
            mask = foreground & (labels == slot)
            count = int(np.count_nonzero(mask))
            slot_pixel_counts[slot] += count
            if count == 0:
                continue
            mask_path = masks_dir / f"{split}_{frame_index:04d}_slot_{slot}.npy"
            np.save(mask_path, mask)
            masks.append(
                {
                    "slot": slot,
                    "label": LEGO_COLOR_SLOTS[slot]["label"],
                    "mask_path": str(mask_path.relative_to(mask_manifest_path.parent)),
                }
            )

        sampled = _sample_mask(foreground, sample_stride=sample_stride)
        if np.any(sampled):
            rows.append(
                _points_from_pixels(
                    rgba=rgba,
                    labels=labels,
                    sampled=sampled,
                    transform=transform,
                    width=width,
                    height=height,
                    camera_angle_x=camera_angle_x,
                    depth=depth,
                    frame_index=frame_index,
                )
            )
            sampled_labels = labels[sampled]
            for slot in range(len(LEGO_COLOR_SLOTS)):
                sampled_slot_counts[slot] += int(
                    np.count_nonzero(sampled_labels == slot)
                )

        manifest_frames.append(
            {
                "name": f"{split}-{frame_index:04d}",
                "image_path": str(image_path.relative_to(dataset)),
                "transform_matrix": transform.tolist(),
                "masks": masks,
            }
        )

    if not rows:
        raise ValueError("no foreground pixels were sampled from NeRF RGBA images")
    vertices = np.concatenate(rows)
    manifest = {
        "width": width,
        "height": height,
        "camera_angle_x": camera_angle_x,
        "source": str(dataset),
        "source_type": "nerf-rgba-color-masks",
        "slots": list(LEGO_COLOR_SLOTS),
        "frames": manifest_frames,
    }
    mask_manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return (
        GaussianCloud(
            vertices=vertices,
            comments=("generated from NeRF Synthetic Lego RGBA views",),
            source_format="binary_little_endian",
        ),
        {
            "frames": len(manifest_frames),
            "masks": sum(len(frame["masks"]) for frame in manifest_frames),
            "width": width,
            "height": height,
            "slot_pixel_counts": _slot_counts(slot_pixel_counts),
            "sampled_slot_counts": _slot_counts(sampled_slot_counts),
            "sampled_gaussians": int(vertices.shape[0]),
        },
    )


def _points_from_pixels(
    *,
    rgba: np.ndarray,
    labels: np.ndarray,
    sampled: np.ndarray,
    transform: np.ndarray,
    width: int,
    height: int,
    camera_angle_x: float,
    depth: float,
    frame_index: int,
) -> np.ndarray:
    y, x = np.nonzero(sampled)
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)
    camera = np.ones((x.shape[0], 4), dtype=np.float32)
    camera[:, 0] = ((x.astype(np.float32) + 0.5) - width * 0.5) / focal * depth
    camera[:, 1] = -((y.astype(np.float32) + 0.5) - height * 0.5) / focal * depth
    camera[:, 2] = -depth
    world = camera @ transform.T

    vertices = np.empty(
        x.shape[0],
        dtype=np.dtype(
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("scale_0", "f4"),
                ("scale_1", "f4"),
                ("scale_2", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
                ("opacity", "f4"),
                ("source_frame", "i4"),
                ("slot_hint", "i4"),
            ]
        ),
    )
    vertices["x"] = world[:, 0]
    vertices["y"] = world[:, 1]
    vertices["z"] = world[:, 2]
    vertices["scale_0"] = 0.035
    vertices["scale_1"] = 0.035
    vertices["scale_2"] = 0.035
    vertices["red"] = rgba[y, x, 0]
    vertices["green"] = rgba[y, x, 1]
    vertices["blue"] = rgba[y, x, 2]
    vertices["opacity"] = np.clip(rgba[y, x, 3].astype(np.float32) / 255.0, 0.35, 1.0)
    vertices["source_frame"] = frame_index
    vertices["slot_hint"] = labels[y, x].astype(np.int32, copy=False)
    return vertices


def _classify_lego_rgba(rgba: np.ndarray) -> np.ndarray:
    red = rgba[:, :, 0]
    green = rgba[:, :, 1]
    blue = rgba[:, :, 2]
    labels = np.full(red.shape, 3, dtype=np.int32)
    labels[(red > 120) & (green > 100) & (blue < 120)] = 0
    labels[(red > 120) & (green < 110) & (blue < 120)] = 1
    labels[np.maximum.reduce((red, green, blue)) < 85] = 2
    return labels


def _sample_mask(mask: np.ndarray, *, sample_stride: int) -> np.ndarray:
    sampled = np.zeros(mask.shape, dtype=bool)
    sampled[::sample_stride, ::sample_stride] = True
    return sampled & mask


def _slot_counts(counts: np.ndarray) -> list[dict[str, int | str]]:
    return [
        {"slot": int(slot["slot"]), "label": str(slot["label"]), "count": int(counts[index])}
        for index, slot in enumerate(LEGO_COLOR_SLOTS)
    ]


def _resolve_nerf_image(dataset: Path, file_path: object) -> Path:
    if not isinstance(file_path, str):
        raise ValueError("NeRF frame is missing file_path")
    raw = file_path[2:] if file_path.startswith("./") else file_path
    candidate = dataset / raw
    if candidate.suffix:
        return candidate
    return candidate.with_suffix(".png")
