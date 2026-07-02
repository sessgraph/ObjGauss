from __future__ import annotations

import json
import os
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class MaskManifestResult:
    manifest_path: Path
    frames: int
    masks: int
    width: int
    height: int
    foreground_pixels: int


@dataclass(frozen=True)
class AlphaFgBgMaskManifestResult:
    manifest_path: Path
    frames: int
    masks: int
    width: int
    height: int
    foreground_pixels: int
    background_pixels: int
    ignore_pixels: int


@dataclass(frozen=True)
class ColorMaskManifestResult:
    manifest_path: Path
    frames: int
    masks: int
    width: int
    height: int
    foreground_pixels: int
    slot_pixel_counts: tuple[dict[str, int | str], ...]


@dataclass(frozen=True)
class SamMaskManifestResult:
    manifest_path: Path
    frames: int
    masks: int
    width: int
    height: int
    mask_pixels: int
    slots: int


@dataclass(frozen=True)
class MaskManifestSplitResult:
    train_manifest_path: Path
    heldout_manifest_path: Path
    source_frames: int
    train_frames: int
    heldout_frames: int
    train_masks: int
    heldout_masks: int


@dataclass(frozen=True)
class MaskManifestValidationResult:
    manifest_path: Path
    passed: bool
    frames: int
    masks: int
    slots: tuple[int, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    frame_stats: tuple[dict[str, Any], ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "manifest": str(self.manifest_path),
            "passed": self.passed,
            "frames": self.frames,
            "masks": self.masks,
            "slots": list(self.slots),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
            "frame_stats": list(self.frame_stats),
        }


LEGO_COLOR_SLOTS = (
    {"slot": 0, "label": "yellow"},
    {"slot": 1, "label": "red"},
    {"slot": 2, "label": "dark"},
    {"slot": 3, "label": "other"},
)


def split_mask_manifest(
    source: str | Path,
    *,
    train_output: str | Path,
    heldout_output: str | Path,
    heldout_every: int = 4,
    heldout_offset: int | None = None,
) -> MaskManifestSplitResult:
    if heldout_every < 2:
        raise ValueError("heldout_every must be >= 2")
    if heldout_offset is None:
        heldout_offset = heldout_every - 1
    if heldout_offset < 0 or heldout_offset >= heldout_every:
        raise ValueError("heldout_offset must be in [0, heldout_every)")

    source = Path(source)
    train_output = Path(train_output)
    heldout_output = Path(heldout_output)
    payload = json.loads(source.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or len(frames) < 2:
        raise ValueError("mask manifest split requires at least two frames")

    train_frames: list[dict[str, Any]] = []
    heldout_frames: list[dict[str, Any]] = []
    for index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            raise ValueError("each mask frame must be an object")
        if index % heldout_every == heldout_offset:
            heldout_frames.append(
                _rewrite_manifest_frame_paths(frame, source.parent, heldout_output.parent)
            )
        else:
            train_frames.append(
                _rewrite_manifest_frame_paths(frame, source.parent, train_output.parent)
            )

    if not train_frames:
        raise ValueError("mask manifest split produced no train frames")
    if not heldout_frames:
        raise ValueError("mask manifest split produced no held-out frames")

    train_manifest = _subset_mask_manifest(
        payload,
        source=source,
        split_kind="train",
        frames=train_frames,
        heldout_every=heldout_every,
        heldout_offset=heldout_offset,
    )
    heldout_manifest = _subset_mask_manifest(
        payload,
        source=source,
        split_kind="heldout",
        frames=heldout_frames,
        heldout_every=heldout_every,
        heldout_offset=heldout_offset,
    )
    _write_manifest_json(train_output, train_manifest)
    _write_manifest_json(heldout_output, heldout_manifest)
    return MaskManifestSplitResult(
        train_manifest_path=train_output,
        heldout_manifest_path=heldout_output,
        source_frames=len(frames),
        train_frames=len(train_frames),
        heldout_frames=len(heldout_frames),
        train_masks=_count_manifest_masks(train_frames),
        heldout_masks=_count_manifest_masks(heldout_frames),
    )


def build_nerf_alpha_mask_manifest(
    dataset: str | Path,
    *,
    output: str | Path,
    split: str = "train",
    max_frames: int | None = None,
    slot: int = 0,
    label: str = "foreground",
    threshold: int = 1,
) -> MaskManifestResult:
    if slot < 0:
        raise ValueError("slot must be >= 0")
    if threshold < 0 or threshold > 255:
        raise ValueError("threshold must be in [0, 255]")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    dataset = Path(dataset)
    output = Path(output)
    transforms_path = dataset / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise ValueError(f"missing NeRF transforms file: {transforms_path}")
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    camera_angle_x = payload.get("camera_angle_x")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{transforms_path} must contain a non-empty frames list")
    if camera_angle_x is None:
        raise ValueError(f"{transforms_path} is missing camera_angle_x")

    output.parent.mkdir(parents=True, exist_ok=True)
    masks_dir = output.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    manifest_frames: list[dict[str, Any]] = []
    width = 0
    height = 0
    foreground_pixels = 0

    for frame_index, frame in enumerate(frames[:max_frames]):
        if not isinstance(frame, dict):
            raise ValueError("NeRF frame entries must be objects")
        image_path = resolve_nerf_image(dataset, frame.get("file_path"))
        alpha = read_png_alpha(image_path)
        if width == 0:
            height, width = alpha.shape
        elif alpha.shape != (height, width):
            raise ValueError(f"{image_path} shape {alpha.shape} does not match {height}x{width}")
        mask = alpha >= threshold
        foreground_pixels += int(np.count_nonzero(mask))
        mask_path = masks_dir / f"{split}_{frame_index:04d}_slot_{slot}.npy"
        np.save(mask_path, mask)
        manifest_frames.append(
            {
                "name": f"{split}-{frame_index:04d}",
                "image_path": str(image_path.relative_to(dataset)),
                "transform_matrix": frame.get("transform_matrix"),
                "masks": [
                    {
                        "slot": slot,
                        "label": label,
                        "mask_path": str(mask_path.relative_to(output.parent)),
                    }
                ],
            }
        )

    manifest = {
        "width": width,
        "height": height,
        "camera_angle_x": float(camera_angle_x),
        "source": str(dataset),
        "source_type": "nerf-alpha",
        "split": split,
        "frames": manifest_frames,
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return MaskManifestResult(
        manifest_path=output,
        frames=len(manifest_frames),
        masks=len(manifest_frames),
        width=width,
        height=height,
        foreground_pixels=foreground_pixels,
    )


def build_nerf_alpha_fgbg_mask_manifest(
    dataset: str | Path,
    *,
    output: str | Path,
    split: str = "train",
    max_frames: int | None = None,
    foreground_threshold: int = 200,
    background_threshold: int = 20,
    foreground_slot: int = 1,
    background_slot: int = 0,
    foreground_confidence: float = 1.0,
    background_confidence: float = 0.05,
) -> AlphaFgBgMaskManifestResult:
    if not 0 <= background_slot:
        raise ValueError("background_slot must be >= 0")
    if not 0 <= foreground_slot:
        raise ValueError("foreground_slot must be >= 0")
    if background_slot == foreground_slot:
        raise ValueError("background_slot and foreground_slot must be different")
    if not 0 <= background_threshold < foreground_threshold <= 255:
        raise ValueError("thresholds must satisfy 0 <= background < foreground <= 255")
    if foreground_confidence <= 0:
        raise ValueError("foreground_confidence must be > 0")
    if background_confidence <= 0:
        raise ValueError("background_confidence must be > 0")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    dataset = Path(dataset)
    output = Path(output)
    transforms_path = dataset / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise ValueError(f"missing NeRF transforms file: {transforms_path}")
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    camera_angle_x = payload.get("camera_angle_x")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{transforms_path} must contain a non-empty frames list")
    if camera_angle_x is None:
        raise ValueError(f"{transforms_path} is missing camera_angle_x")

    output.parent.mkdir(parents=True, exist_ok=True)
    masks_dir = output.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    manifest_frames: list[dict[str, Any]] = []
    width = 0
    height = 0
    foreground_pixels = 0
    background_pixels = 0
    ignore_pixels = 0
    mask_count = 0

    for frame_index, frame in enumerate(frames[:max_frames]):
        if not isinstance(frame, dict):
            raise ValueError("NeRF frame entries must be objects")
        image_path = resolve_nerf_image(dataset, frame.get("file_path"))
        alpha = read_png_alpha(image_path)
        if width == 0:
            height, width = alpha.shape
        elif alpha.shape != (height, width):
            raise ValueError(f"{image_path} shape {alpha.shape} does not match {height}x{width}")

        foreground = alpha > foreground_threshold
        background = alpha < background_threshold
        ignore = ~(foreground | background)
        foreground_count = int(np.count_nonzero(foreground))
        background_count = int(np.count_nonzero(background))
        ignore_count = int(np.count_nonzero(ignore))
        foreground_pixels += foreground_count
        background_pixels += background_count
        ignore_pixels += ignore_count

        background_path = masks_dir / f"{split}_{frame_index:04d}_slot_{background_slot:02d}.npy"
        foreground_path = masks_dir / f"{split}_{frame_index:04d}_slot_{foreground_slot:02d}.npy"
        ignore_path = masks_dir / f"{split}_{frame_index:04d}_ignore.npy"
        np.save(background_path, background)
        np.save(foreground_path, foreground)
        np.save(ignore_path, ignore)
        mask_count += 2

        manifest_frames.append(
            {
                "name": f"{split}-{frame_index:04d}",
                "frame_index": frame_index,
                "image_path": str(image_path.relative_to(dataset)),
                "transform_frame_path": str(frame.get("file_path")),
                "transform_matrix": frame.get("transform_matrix"),
                "width": width,
                "height": height,
                "ignore_mask_path": str(ignore_path.relative_to(output.parent)),
                "ignore_pixels": ignore_count,
                "masks": [
                    {
                        "slot": background_slot,
                        "slot_id": background_slot,
                        "label": "background",
                        "name": "background",
                        "type": "background",
                        "mask_path": str(background_path.relative_to(output.parent)),
                        "source": "alpha",
                        "polarity": "background",
                        "confidence": float(background_confidence),
                        "area": background_count,
                    },
                    {
                        "slot": foreground_slot,
                        "slot_id": foreground_slot,
                        "label": "foreground",
                        "name": "foreground",
                        "type": "foreground",
                        "mask_path": str(foreground_path.relative_to(output.parent)),
                        "source": "alpha",
                        "polarity": "foreground",
                        "confidence": float(foreground_confidence),
                        "area": foreground_count,
                    },
                ],
            }
        )

    slots = sorted(
        [
            {
                "slot": int(background_slot),
                "slot_id": int(background_slot),
                "name": "background",
                "label": "background",
                "type": "background",
            },
            {
                "slot": int(foreground_slot),
                "slot_id": int(foreground_slot),
                "name": "foreground",
                "label": "foreground",
                "type": "foreground",
            },
        ],
        key=lambda item: int(item["slot"]),
    )
    manifest = {
        "width": width,
        "height": height,
        "image_width": width,
        "image_height": height,
        "camera_angle_x": float(camera_angle_x),
        "source": str(dataset),
        "source_type": "nerf-alpha-fgbg",
        "split": split,
        "slot_count": len(slots),
        "slots": slots,
        "alpha_thresholds": {
            "foreground_gt": int(foreground_threshold),
            "background_lt": int(background_threshold),
            "ignore": f"{background_threshold} <= alpha <= {foreground_threshold}",
            "foreground_confidence": float(foreground_confidence),
            "background_confidence": float(background_confidence),
        },
        "frames": manifest_frames,
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return AlphaFgBgMaskManifestResult(
        manifest_path=output,
        frames=len(manifest_frames),
        masks=mask_count,
        width=width,
        height=height,
        foreground_pixels=foreground_pixels,
        background_pixels=background_pixels,
        ignore_pixels=ignore_pixels,
    )


def build_nerf_rgba_color_mask_manifest(
    dataset: str | Path,
    *,
    output: str | Path,
    split: str = "train",
    max_frames: int | None = None,
    alpha_threshold: int = 16,
) -> ColorMaskManifestResult:
    if alpha_threshold < 0 or alpha_threshold > 255:
        raise ValueError("alpha_threshold must be in [0, 255]")
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")

    dataset = Path(dataset)
    output = Path(output)
    transforms_path = dataset / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise ValueError(f"missing NeRF transforms file: {transforms_path}")
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    camera_angle_x = payload.get("camera_angle_x")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{transforms_path} must contain a non-empty frames list")
    if camera_angle_x is None:
        raise ValueError(f"{transforms_path} is missing camera_angle_x")

    output.parent.mkdir(parents=True, exist_ok=True)
    masks_dir = output.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    manifest_frames: list[dict[str, Any]] = []
    slot_pixel_counts = np.zeros(len(LEGO_COLOR_SLOTS), dtype=np.int64)
    width = 0
    height = 0
    foreground_pixels = 0
    mask_count = 0

    for frame_index, frame in enumerate(frames[:max_frames]):
        if not isinstance(frame, dict):
            raise ValueError("NeRF frame entries must be objects")
        image_path = resolve_nerf_image(dataset, frame.get("file_path"))
        rgba = read_image_rgba(image_path)
        if width == 0:
            height, width = rgba.shape[:2]
        elif rgba.shape[:2] != (height, width):
            raise ValueError(f"{image_path} shape does not match {height}x{width}")

        labels = classify_lego_rgba(rgba)
        foreground = rgba[:, :, 3] >= alpha_threshold
        foreground_pixels += int(np.count_nonzero(foreground))
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
                    "mask_path": str(mask_path.relative_to(output.parent)),
                }
            )
            mask_count += 1

        manifest_frames.append(
            {
                "name": f"{split}-{frame_index:04d}",
                "image_path": str(image_path.relative_to(dataset)),
                "transform_matrix": frame.get("transform_matrix"),
                "masks": masks,
            }
        )

    manifest = {
        "width": width,
        "height": height,
        "camera_angle_x": float(camera_angle_x),
        "source": str(dataset),
        "source_type": "nerf-rgba-color-masks",
        "split": split,
        "alpha_threshold": alpha_threshold,
        "slots": list(LEGO_COLOR_SLOTS),
        "frames": manifest_frames,
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return ColorMaskManifestResult(
        manifest_path=output,
        frames=len(manifest_frames),
        masks=mask_count,
        width=width,
        height=height,
        foreground_pixels=foreground_pixels,
        slot_pixel_counts=tuple(slot_count_summary(slot_pixel_counts)),
    )


def build_nerf_sam_mask_manifest(
    dataset: str | Path,
    *,
    output: str | Path,
    checkpoint: str | Path | None = None,
    model_type: str = "vit_b",
    device: str = "cpu",
    split: str = "train",
    max_frames: int | None = None,
    max_masks_per_frame: int = 8,
    min_area: int = 1,
    max_area_fraction: float = 1.0,
    max_image_size: int | None = None,
    points_per_side: int = 32,
    pred_iou_thresh: float = 0.88,
    stability_score_thresh: float = 0.95,
    generator: Any | Callable[[np.ndarray], list[dict[str, Any]]] | None = None,
) -> SamMaskManifestResult:
    if max_frames is not None and max_frames < 1:
        raise ValueError("max_frames must be >= 1")
    if max_masks_per_frame < 1:
        raise ValueError("max_masks_per_frame must be >= 1")
    if min_area < 1:
        raise ValueError("min_area must be >= 1")
    if not 0.0 < max_area_fraction <= 1.0:
        raise ValueError("max_area_fraction must be in (0, 1]")
    if max_image_size is not None and max_image_size < 8:
        raise ValueError("max_image_size must be >= 8")

    dataset = Path(dataset)
    output = Path(output)
    transforms_path = dataset / f"transforms_{split}.json"
    if not transforms_path.exists():
        raise ValueError(f"missing NeRF transforms file: {transforms_path}")
    payload = json.loads(transforms_path.read_text(encoding="utf-8"))
    camera_angle_x = payload.get("camera_angle_x")
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError(f"{transforms_path} must contain a non-empty frames list")
    if camera_angle_x is None:
        raise ValueError(f"{transforms_path} is missing camera_angle_x")

    generator = generator or _create_sam_generator(
        checkpoint=checkpoint,
        model_type=model_type,
        device=device,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    masks_dir = output.parent / "masks"
    masks_dir.mkdir(parents=True, exist_ok=True)
    manifest_frames: list[dict[str, Any]] = []
    width = 0
    height = 0
    mask_count = 0
    mask_pixels = 0

    for frame_index, frame in enumerate(frames[:max_frames]):
        if not isinstance(frame, dict):
            raise ValueError("NeRF frame entries must be objects")
        image_path = resolve_nerf_image(dataset, frame.get("file_path"))
        rgba = read_image_rgba(image_path)
        if max_image_size is not None:
            rgba = resize_rgba_max_size(rgba, max_image_size)
        if width == 0:
            height, width = rgba.shape[:2]
        elif rgba.shape[:2] != (height, width):
            raise ValueError(f"{image_path} shape does not match {height}x{width}")

        generated_masks = _generate_sam_masks(generator, rgba[:, :, :3])
        selected_masks = _select_sam_masks(
            generated_masks,
            height=height,
            width=width,
            min_area=min_area,
            max_area_fraction=max_area_fraction,
            max_masks=max_masks_per_frame,
        )
        masks: list[dict[str, Any]] = []
        for slot, sam_mask in enumerate(selected_masks):
            segmentation = np.asarray(sam_mask["segmentation"], dtype=bool)
            area = int(np.count_nonzero(segmentation))
            mask_pixels += area
            mask_path = masks_dir / f"{split}_{frame_index:04d}_sam_slot_{slot}.npy"
            np.save(mask_path, segmentation)
            masks.append(
                {
                    "slot": slot,
                    "label": f"sam-area-rank-{slot}",
                    "mask_path": str(mask_path.relative_to(output.parent)),
                    "confidence": _sam_confidence(sam_mask),
                    "area": area,
                    "bbox": _sam_bbox(sam_mask),
                }
            )
            mask_count += 1
        if masks:
            manifest_frames.append(
                {
                    "name": f"{split}-{frame_index:04d}",
                    "image_path": str(image_path.relative_to(dataset)),
                    "transform_matrix": frame.get("transform_matrix"),
                    "masks": masks,
                }
            )

    if mask_count == 0:
        raise ValueError("SAM did not produce any masks after filtering")

    manifest = {
        "width": width,
        "height": height,
        "camera_angle_x": float(camera_angle_x),
        "source": str(dataset),
        "source_type": "sam-automatic-mask-generator",
        "split": split,
        "slots": [
            {"slot": slot, "label": f"sam-area-rank-{slot}"}
            for slot in range(max_masks_per_frame)
        ],
        "sam": {
            "model_type": model_type,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
            "device": device,
            "points_per_side": points_per_side,
            "pred_iou_thresh": pred_iou_thresh,
            "stability_score_thresh": stability_score_thresh,
            "min_area": min_area,
            "max_area_fraction": max_area_fraction,
            "max_image_size": max_image_size,
            "max_masks_per_frame": max_masks_per_frame,
        },
        "frames": manifest_frames,
    }
    output.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return SamMaskManifestResult(
        manifest_path=output,
        frames=len(manifest_frames),
        masks=mask_count,
        width=width,
        height=height,
        mask_pixels=mask_pixels,
        slots=max_masks_per_frame,
    )


def validate_mask_manifest(
    manifest_path: str | Path,
    *,
    dataset: str | Path | None = None,
    max_overlap_fraction: float = 0.0,
    max_mask_area_fraction: float = 0.98,
    allow_empty: bool = False,
) -> MaskManifestValidationResult:
    if max_overlap_fraction < 0:
        raise ValueError("max_overlap_fraction must be >= 0")
    if not 0.0 < max_mask_area_fraction <= 1.0:
        raise ValueError("max_mask_area_fraction must be in (0, 1]")

    manifest_path = Path(manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = manifest_path.parent
    source_root = _mask_manifest_source_root(payload, dataset)
    width = _optional_int(payload.get("width") or payload.get("image_width"))
    height = _optional_int(payload.get("height") or payload.get("image_height"))
    frames = payload.get("frames")
    errors: list[str] = []
    warnings: list[str] = []
    frame_stats: list[dict[str, Any]] = []
    observed_slots: set[int] = set()
    total_masks = 0

    if not isinstance(frames, list) or not frames:
        errors.append("mask manifest must contain a non-empty frames list")
        frames = []

    for frame_index, frame in enumerate(frames):
        if not isinstance(frame, dict):
            errors.append(f"frame {frame_index} is not an object")
            continue
        frame_width = _optional_int(frame.get("width")) or width
        frame_height = _optional_int(frame.get("height")) or height
        image_path = frame.get("image_path")
        if source_root is not None and isinstance(image_path, str) and image_path:
            image_file = source_root / image_path
            if not image_file.exists():
                errors.append(f"frame {frame_index} image is missing: {image_file}")
            else:
                try:
                    image = read_image_rgba(image_file)
                    image_height, image_width = image.shape[:2]
                    if frame_width is not None and frame_height is not None:
                        if (image_height, image_width) != (frame_height, frame_width):
                            errors.append(
                                f"frame {frame_index} image shape {image_height}x{image_width} "
                                f"does not match {frame_height}x{frame_width}"
                            )
                except Exception as exc:
                    errors.append(f"frame {frame_index} image could not be read: {exc}")
        elif source_root is not None:
            warnings.append(f"frame {frame_index} does not declare image_path")

        masks = frame.get("masks")
        if not isinstance(masks, list) or not masks:
            errors.append(f"frame {frame_index} has no masks")
            continue
        mask_arrays: list[np.ndarray] = []
        mask_stats: list[dict[str, Any]] = []
        for mask_index, mask in enumerate(masks):
            total_masks += 1
            if not isinstance(mask, dict):
                errors.append(f"frame {frame_index} mask {mask_index} is not an object")
                continue
            slot = mask.get("slot", mask.get("slot_id"))
            try:
                slot_id = int(slot)
                observed_slots.add(slot_id)
            except Exception:
                errors.append(f"frame {frame_index} mask {mask_index} has invalid slot")
                slot_id = -1
            mask_path = mask.get("mask_path")
            if not isinstance(mask_path, str) or not mask_path:
                errors.append(f"frame {frame_index} mask {mask_index} is missing mask_path")
                continue
            path = root / mask_path
            if not path.exists():
                errors.append(f"frame {frame_index} mask {mask_index} is missing: {path}")
                continue
            try:
                mask_array = _load_boolean_mask(path)
            except Exception as exc:
                errors.append(f"frame {frame_index} mask {mask_index} invalid array: {exc}")
                continue
            if frame_width is not None and frame_height is not None:
                if mask_array.shape != (frame_height, frame_width):
                    errors.append(
                        f"frame {frame_index} mask {mask_index} shape {mask_array.shape} "
                        f"does not match {frame_height}x{frame_width}"
                    )
            area = int(np.count_nonzero(mask_array))
            total_pixels = int(mask_array.size)
            if area == 0 and not allow_empty:
                errors.append(f"frame {frame_index} slot {slot_id} mask is empty")
            if area > total_pixels * max_mask_area_fraction:
                errors.append(
                    f"frame {frame_index} slot {slot_id} mask covers {area / total_pixels:.6f}, "
                    f"max allowed is {max_mask_area_fraction:.6f}"
                )
            mask_arrays.append(mask_array)
            mask_stats.append(
                {
                    "slot": int(slot_id),
                    "pixels": area,
                    "fraction": 0.0 if total_pixels == 0 else float(area / total_pixels),
                }
            )

        overlap_pixels = 0
        overlap_fraction = 0.0
        if mask_arrays:
            stacked = np.stack(mask_arrays, axis=0)
            overlap_pixels = int(np.count_nonzero(stacked.sum(axis=0) > 1))
            overlap_fraction = float(overlap_pixels / stacked.shape[1] / stacked.shape[2])
            if overlap_fraction > max_overlap_fraction:
                errors.append(
                    f"frame {frame_index} overlap_fraction={overlap_fraction:.6f} "
                    f"exceeds {max_overlap_fraction:.6f}"
                )

        ignore_pixels = None
        ignore_mask_path = frame.get("ignore_mask_path")
        if isinstance(ignore_mask_path, str) and ignore_mask_path:
            try:
                ignore = _load_boolean_mask(root / ignore_mask_path)
                ignore_pixels = int(np.count_nonzero(ignore))
                if frame_width is not None and frame_height is not None:
                    if ignore.shape != (frame_height, frame_width):
                        errors.append(
                            f"frame {frame_index} ignore mask shape {ignore.shape} "
                            f"does not match {frame_height}x{frame_width}"
                        )
            except Exception as exc:
                errors.append(f"frame {frame_index} ignore mask invalid array: {exc}")

        frame_stats.append(
            {
                "frame_index": int(frame.get("frame_index", frame_index)),
                "masks": mask_stats,
                "overlap_pixels": overlap_pixels,
                "overlap_fraction": overlap_fraction,
                "ignore_pixels": ignore_pixels,
            }
        )

    observed_slots.update(_slots_from_manifest_slots(payload.get("slots")))
    if observed_slots:
        expected = set(range(max(observed_slots) + 1))
        if observed_slots != expected:
            errors.append(
                f"slot ids must be contiguous from 0; observed={sorted(observed_slots)}"
            )

    return MaskManifestValidationResult(
        manifest_path=manifest_path,
        passed=not errors,
        frames=len(frames),
        masks=total_masks,
        slots=tuple(sorted(observed_slots)),
        errors=tuple(errors),
        warnings=tuple(warnings),
        frame_stats=tuple(frame_stats),
    )


def read_png_alpha(path: str | Path) -> np.ndarray:
    return read_png_rgba(path)[:, :, 3]


def read_image_rgba(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".png":
        return read_png_rgba(path)
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError(
            f"{path} is not a PNG file; reading JPEG or other image formats "
            "requires optional dependency 'Pillow'"
        ) from exc
    with Image.open(path) as image:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)


def resize_rgba_max_size(rgba: np.ndarray, max_size: int) -> np.ndarray:
    if max(rgba.shape[:2]) <= max_size:
        return rgba
    try:
        from PIL import Image
    except ImportError as exc:
        raise ValueError("resizing SAM input images requires optional dependency 'Pillow'") from exc
    height, width = rgba.shape[:2]
    scale = float(max_size) / float(max(height, width))
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    image = Image.fromarray(rgba, mode="RGBA")
    resized = image.resize(target, Image.Resampling.BILINEAR)
    return np.asarray(resized, dtype=np.uint8)


def read_png_rgba(path: str | Path) -> np.ndarray:
    path = Path(path)
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG file: {path}")

    offset = 8
    width = height = bit_depth = color_type = None
    compressed = bytearray()
    while offset < len(data):
        length = struct.unpack(">I", data[offset : offset + 4])[0]
        chunk_type = data[offset + 4 : offset + 8]
        chunk_data = data[offset + 8 : offset + 8 + length]
        offset += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, compression, filter_method, interlace = struct.unpack(
                ">IIBBBBB",
                chunk_data,
            )
            if bit_depth != 8 or color_type != 6:
                raise ValueError(f"{path} must be an 8-bit RGBA PNG")
            if compression != 0 or filter_method != 0 or interlace != 0:
                raise ValueError(f"{path} uses unsupported PNG encoding")
        elif chunk_type == b"IDAT":
            compressed.extend(chunk_data)
        elif chunk_type == b"IEND":
            break

    if width is None or height is None:
        raise ValueError(f"{path} has no PNG IHDR")
    raw = zlib.decompress(bytes(compressed))
    return _decode_png_scanlines(raw, width=width, height=height, channels=4)


def _decode_png_scanlines(raw: bytes, *, width: int, height: int, channels: int) -> np.ndarray:
    stride = width * channels
    output = np.zeros((height, stride), dtype=np.uint8)
    offset = 0
    for row in range(height):
        filter_type = raw[offset]
        offset += 1
        scanline = np.frombuffer(raw[offset : offset + stride], dtype=np.uint8).astype(np.int16)
        offset += stride
        left = np.zeros(stride, dtype=np.int16)
        left[channels:] = scanline[:-channels]
        up = output[row - 1].astype(np.int16) if row > 0 else np.zeros(stride, dtype=np.int16)
        up_left = np.zeros(stride, dtype=np.int16)
        if row > 0:
            up_left[channels:] = output[row - 1, :-channels].astype(np.int16)
        if filter_type == 0:
            restored = scanline
        elif filter_type == 1:
            restored = scanline + left
        elif filter_type == 2:
            restored = scanline + up
        elif filter_type == 3:
            restored = scanline + ((left + up) // 2)
        elif filter_type == 4:
            restored = scanline + _paeth(left, up, up_left)
        else:
            raise ValueError(f"unsupported PNG filter type: {filter_type}")
        output[row] = np.mod(restored, 256).astype(np.uint8)
    return output.reshape(height, width, channels)


def _paeth(left: np.ndarray, up: np.ndarray, up_left: np.ndarray) -> np.ndarray:
    estimate = left + up - up_left
    left_distance = np.abs(estimate - left)
    up_distance = np.abs(estimate - up)
    up_left_distance = np.abs(estimate - up_left)
    return np.where(
        (left_distance <= up_distance) & (left_distance <= up_left_distance),
        left,
        np.where(up_distance <= up_left_distance, up, up_left),
    )


def classify_lego_rgba(rgba: np.ndarray) -> np.ndarray:
    if rgba.ndim != 3 or rgba.shape[2] != 4:
        raise ValueError("rgba must be an HxWx4 array")
    red = rgba[:, :, 0]
    green = rgba[:, :, 1]
    blue = rgba[:, :, 2]
    labels = np.full(red.shape, 3, dtype=np.int32)
    labels[(red > 120) & (green > 100) & (blue < 120)] = 0
    labels[(red > 120) & (green < 110) & (blue < 120)] = 1
    labels[np.maximum.reduce((red, green, blue)) < 85] = 2
    return labels


def slot_count_summary(counts: np.ndarray) -> list[dict[str, int | str]]:
    return [
        {"slot": int(slot["slot"]), "label": str(slot["label"]), "count": int(counts[index])}
        for index, slot in enumerate(LEGO_COLOR_SLOTS)
    ]


def _subset_mask_manifest(
    payload: dict[str, Any],
    *,
    source: Path,
    split_kind: str,
    frames: list[dict[str, Any]],
    heldout_every: int,
    heldout_offset: int,
) -> dict[str, Any]:
    output = {
        key: value
        for key, value in payload.items()
        if key not in {"frames", "split_manifest"}
    }
    output["frames"] = frames
    output["split_manifest"] = {
        "source": str(source),
        "kind": split_kind,
        "heldout_every": int(heldout_every),
        "heldout_offset": int(heldout_offset),
        "source_frames": len(payload.get("frames") or []),
        "frames": len(frames),
    }
    return output


def _rewrite_manifest_frame_paths(
    frame: dict[str, Any],
    source_root: Path,
    target_root: Path,
) -> dict[str, Any]:
    rewritten = json.loads(json.dumps(frame))
    masks = rewritten.get("masks")
    if not isinstance(masks, list):
        return rewritten
    for mask in masks:
        if not isinstance(mask, dict):
            continue
        mask_path = mask.get("mask_path")
        if not isinstance(mask_path, str) or not mask_path:
            continue
        path = Path(mask_path)
        if path.is_absolute():
            continue
        absolute = (source_root / path).resolve()
        mask["mask_path"] = os.path.relpath(absolute, target_root.resolve())
    return rewritten


def _count_manifest_masks(frames: list[dict[str, Any]]) -> int:
    return sum(len(frame.get("masks") or []) for frame in frames)


def _write_manifest_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _mask_manifest_source_root(
    payload: dict[str, Any],
    dataset: str | Path | None,
) -> Path | None:
    if dataset is not None:
        return Path(dataset)
    source = payload.get("source")
    if isinstance(source, str) and source:
        return Path(source)
    return None


def _load_boolean_mask(path: Path) -> np.ndarray:
    array = np.load(path)
    if array.ndim != 2:
        raise ValueError(f"{path} must be a 2D mask array")
    if array.dtype == np.bool_:
        return array.astype(bool, copy=False)
    unique = np.unique(array)
    if not np.all(np.isin(unique, [0, 1])):
        raise ValueError(f"{path} must be bool or contain only 0/1 values")
    return array.astype(bool, copy=False)


def _slots_from_manifest_slots(value: object) -> set[int]:
    slots: set[int] = set()
    if not isinstance(value, list):
        return slots
    for slot in value:
        if not isinstance(slot, dict):
            continue
        raw = slot.get("slot", slot.get("slot_id"))
        if raw is None:
            continue
        slots.add(int(raw))
    return slots


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _create_sam_generator(
    *,
    checkpoint: str | Path | None,
    model_type: str,
    device: str,
    points_per_side: int,
    pred_iou_thresh: float,
    stability_score_thresh: float,
):
    if checkpoint is None:
        raise ValueError("SAM checkpoint is required")
    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        raise ValueError(f"missing SAM checkpoint: {checkpoint}")
    try:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
    except ImportError as exc:
        raise ValueError(
            "SAM mask generation requires optional dependency 'segment-anything' "
            "and a local checkpoint; install it outside ObjGauss and pass --checkpoint"
        ) from exc

    if model_type not in sam_model_registry:
        available = ", ".join(sorted(sam_model_registry))
        raise ValueError(f"unknown SAM model_type {model_type!r}; available: {available}")
    model = sam_model_registry[model_type](checkpoint=str(checkpoint))
    model.to(device=device)
    return SamAutomaticMaskGenerator(
        model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
    )


def _generate_sam_masks(generator: Any, image_rgb: np.ndarray) -> list[dict[str, Any]]:
    if hasattr(generator, "generate"):
        masks = generator.generate(image_rgb)
    else:
        masks = generator(image_rgb)
    if not isinstance(masks, list):
        raise ValueError("SAM generator must return a list of mask dictionaries")
    return masks


def _select_sam_masks(
    masks: list[dict[str, Any]],
    *,
    height: int,
    width: int,
    min_area: int,
    max_area_fraction: float,
    max_masks: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    max_area = int(np.floor(height * width * max_area_fraction))
    for mask in masks:
        if not isinstance(mask, dict) or "segmentation" not in mask:
            continue
        segmentation = np.asarray(mask["segmentation"], dtype=bool)
        if segmentation.shape != (height, width):
            raise ValueError(
                f"SAM mask shape {segmentation.shape} does not match {height}x{width}"
            )
        area = int(mask.get("area", np.count_nonzero(segmentation)))
        if min_area <= area <= max_area:
            selected.append(mask)
    selected.sort(
        key=lambda mask: int(mask.get("area", np.count_nonzero(mask["segmentation"]))),
        reverse=True,
    )
    return selected[:max_masks]


def _sam_confidence(mask: dict[str, Any]) -> float:
    if "predicted_iou" in mask:
        return float(mask["predicted_iou"])
    if "stability_score" in mask:
        return float(mask["stability_score"])
    return 1.0


def _sam_bbox(mask: dict[str, Any]) -> list[float] | None:
    bbox = mask.get("bbox")
    if not isinstance(bbox, list | tuple) or len(bbox) != 4:
        return None
    return [float(value) for value in bbox]


def resolve_nerf_image(dataset: Path, file_path: object) -> Path:
    if not isinstance(file_path, str):
        raise ValueError("NeRF frame is missing file_path")
    raw = file_path[2:] if file_path.startswith("./") else file_path
    candidate = dataset / raw
    if candidate.suffix:
        return candidate
    return candidate.with_suffix(".png")
