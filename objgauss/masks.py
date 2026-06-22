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
        image_path = _resolve_nerf_image(dataset, frame.get("file_path"))
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


def _resolve_nerf_image(dataset: Path, file_path: object) -> Path:
    if not isinstance(file_path, str):
        raise ValueError("NeRF frame is missing file_path")
    raw = file_path[2:] if file_path.startswith("./") else file_path
    candidate = dataset / raw
    if candidate.suffix:
        return candidate
    return candidate.with_suffix(".png")
