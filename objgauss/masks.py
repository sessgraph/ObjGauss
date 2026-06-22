from __future__ import annotations

import json
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
        rgba = read_png_rgba(image_path)
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


def read_png_alpha(path: str | Path) -> np.ndarray:
    return read_png_rgba(path)[:, :, 3]


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


def resolve_nerf_image(dataset: Path, file_path: object) -> Path:
    if not isinstance(file_path, str):
        raise ValueError("NeRF frame is missing file_path")
    raw = file_path[2:] if file_path.startswith("./") else file_path
    candidate = dataset / raw
    if candidate.suffix:
        return candidate
    return candidate.with_suffix(".png")
