from __future__ import annotations

import json
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


LEGO_COLOR_SLOTS = (
    {"slot": 0, "label": "yellow"},
    {"slot": 1, "label": "red"},
    {"slot": 2, "label": "dark"},
    {"slot": 3, "label": "other"},
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
