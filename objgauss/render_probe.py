from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from objgauss.features import colors, opacity, positions
from objgauss.gaussians import GaussianCloud
from objgauss.object_field import ObjectField

_EPS = 1e-8


@dataclass(frozen=True)
class RenderProbeFrame:
    transform_matrix: np.ndarray
    width: int
    height: int
    camera_angle_x: float


def load_render_probe_frames(
    manifest_path: str | Path,
    *,
    max_frames: int | None = None,
    max_size: int = 128,
) -> list[RenderProbeFrame]:
    if max_size < 8:
        raise ValueError("render max_size must be >= 8")
    path = Path(manifest_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    frames = payload.get("frames")
    if not isinstance(frames, list) or not frames:
        raise ValueError("mask manifest must contain a non-empty frames list")

    default_width = _optional_int(payload.get("width"))
    default_height = _optional_int(payload.get("height"))
    default_angle_x = _optional_float(payload.get("camera_angle_x"))
    output: list[RenderProbeFrame] = []
    for frame in frames[:max_frames]:
        if not isinstance(frame, dict):
            raise ValueError("each mask frame must be an object")
        width = _required_int(frame.get("width", default_width), "frame width")
        height = _required_int(frame.get("height", default_height), "frame height")
        scale = min(1.0, float(max_size) / float(max(width, height)))
        render_width = max(1, int(round(width * scale)))
        render_height = max(1, int(round(height * scale)))
        output.append(
            RenderProbeFrame(
                transform_matrix=_required_matrix(frame.get("transform_matrix")),
                width=render_width,
                height=render_height,
                camera_angle_x=_required_float(
                    frame.get("camera_angle_x", default_angle_x),
                    "camera_angle_x",
                ),
            )
        )
    return output


def render_occlusion_delta(
    cloud: GaussianCloud,
    field: ObjectField,
    frames: Sequence[RenderProbeFrame],
) -> dict[str, Any]:
    """Measure visual effect of removing each hard object slot.

    This is a deterministic CPU render probe: Gaussian centers are projected into
    mask-manifest cameras, the nearest visible center owns each pixel, and
    full-vs-removed RGBA images are compared. It is not a covariance-aware 3DGS
    training renderer, but it measures a real image-space removal delta instead
    of reusing the mask-vote loss.
    """

    if cloud.count != field.gaussian_count:
        raise ValueError(f"field has {field.gaussian_count} gaussians for cloud with {cloud.count}")
    if not frames:
        raise ValueError("render occlusion requires at least one frame")

    xyz = positions(cloud)
    rgb = colors(cloud)
    alpha = np.clip(opacity(cloud), 0.0, 1.0).astype(np.float32, copy=False)
    labels = field.labels()

    frame_summaries: list[dict[str, Any]] = []
    per_slot_accumulators = [
        {
            "slot": slot,
            "delta_l1": [],
            "relative_delta_l1": [],
            "affected_fraction": [],
            "visible_gaussians": [],
        }
        for slot in range(field.slots)
    ]

    for index, frame in enumerate(frames):
        projection = _project_for_render(xyz, frame)
        visible_indices = np.flatnonzero(projection["visible"])
        base = _render_frontmost(
            projection["pixel"][visible_indices],
            projection["depth"][visible_indices],
            rgb[visible_indices],
            alpha[visible_indices],
            width=frame.width,
            height=frame.height,
        )
        base_energy = float(np.mean(np.abs(base)))
        frame_per_slot: list[dict[str, Any]] = []
        for slot in range(field.slots):
            keep = labels[visible_indices] != slot
            removed = _render_frontmost(
                projection["pixel"][visible_indices][keep],
                projection["depth"][visible_indices][keep],
                rgb[visible_indices][keep],
                alpha[visible_indices][keep],
                width=frame.width,
                height=frame.height,
            )
            diff = np.abs(base - removed)
            delta_l1 = float(np.mean(diff))
            relative = float(delta_l1 / max(base_energy, _EPS))
            affected = np.any(diff > 1e-5, axis=2)
            affected_fraction = float(np.count_nonzero(affected) / max(affected.size, 1))
            visible_slot = int(np.count_nonzero(labels[visible_indices] == slot))
            item = {
                "slot": slot,
                "delta_l1": delta_l1,
                "relative_delta_l1": relative,
                "affected_fraction": affected_fraction,
                "visible_gaussians": visible_slot,
            }
            frame_per_slot.append(item)
            per_slot_accumulators[slot]["delta_l1"].append(delta_l1)
            per_slot_accumulators[slot]["relative_delta_l1"].append(relative)
            per_slot_accumulators[slot]["affected_fraction"].append(affected_fraction)
            per_slot_accumulators[slot]["visible_gaussians"].append(visible_slot)

        frame_summaries.append(
            {
                "frame": index,
                "width": frame.width,
                "height": frame.height,
                "visible_gaussians": int(visible_indices.size),
                "base_coverage_fraction": _coverage_fraction(base),
                "base_rgba_mean_abs": base_energy,
                "per_slot": frame_per_slot,
            }
        )

    per_slot = [
        {
            "slot": int(accumulator["slot"]),
            "mean_delta_l1": _mean(accumulator["delta_l1"]),
            "mean_relative_delta_l1": _mean(accumulator["relative_delta_l1"]),
            "mean_affected_fraction": _mean(accumulator["affected_fraction"]),
            "mean_visible_gaussians": _mean(accumulator["visible_gaussians"]),
        }
        for accumulator in per_slot_accumulators
    ]
    relative_values = [float(item["mean_relative_delta_l1"]) for item in per_slot]
    delta_values = [float(item["mean_delta_l1"]) for item in per_slot]
    affected_values = [float(item["mean_affected_fraction"]) for item in per_slot]
    return {
        "kind": "point_splat_render_l1",
        "frames": len(frames),
        "mean_delta_l1": _mean(delta_values),
        "max_delta_l1": float(np.max(delta_values)) if delta_values else 0.0,
        "mean_relative_delta_l1": _mean(relative_values),
        "max_relative_delta_l1": float(np.max(relative_values)) if relative_values else 0.0,
        "mean_affected_fraction": _mean(affected_values),
        "occlusion_effect_score": float(min(1.0, max(0.0, _mean(relative_values)))),
        "per_slot": per_slot,
        "frames_summary": frame_summaries,
    }


def _render_frontmost(
    pixel: np.ndarray,
    depth: np.ndarray,
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.float32)
    if pixel.size == 0:
        return image

    order = np.lexsort((depth, pixel))
    sorted_pixel = pixel[order]
    first = np.concatenate(([True], sorted_pixel[1:] != sorted_pixel[:-1]))
    chosen = order[first]
    flat = pixel[chosen]
    y = flat // width
    x = flat - y * width
    chosen_alpha = alpha[chosen]
    image[y, x, :3] = rgb[chosen] * chosen_alpha[:, None]
    image[y, x, 3] = chosen_alpha
    return image


def _project_for_render(xyz: np.ndarray, frame: RenderProbeFrame) -> dict[str, np.ndarray]:
    c2w = np.asarray(frame.transform_matrix, dtype=np.float32)
    homogeneous = np.concatenate([xyz, np.ones((xyz.shape[0], 1), dtype=np.float32)], axis=1)
    camera = homogeneous @ np.linalg.inv(c2w).T
    forward = -camera[:, 2]
    focal = 0.5 * frame.width / np.tan(0.5 * frame.camera_angle_x)
    u = focal * (camera[:, 0] / np.maximum(forward, 1e-4)) + frame.width * 0.5
    v = frame.height * 0.5 - focal * (camera[:, 1] / np.maximum(forward, 1e-4))
    visible = (
        (forward > 1e-4)
        & (u >= 0)
        & (u < frame.width)
        & (v >= 0)
        & (v < frame.height)
        & np.isfinite(u)
        & np.isfinite(v)
    )
    x = np.clip(np.floor(u).astype(np.int64), 0, frame.width - 1)
    y = np.clip(np.floor(v).astype(np.int64), 0, frame.height - 1)
    return {
        "visible": visible,
        "pixel": (y * frame.width + x).astype(np.int64, copy=False),
        "depth": forward.astype(np.float32, copy=False),
    }


def _coverage_fraction(image: np.ndarray) -> float:
    alpha = image[:, :, 3]
    return float(np.count_nonzero(alpha > 1e-5) / max(alpha.size, 1))


def _mean(values: Sequence[float | int]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _required_matrix(value: object) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transform_matrix must be a finite 4x4 matrix")
    return matrix


def _required_int(value: object, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} is required")
    return int(value)


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _required_float(value: object, name: str) -> float:
    if value is None:
        raise ValueError(f"{name} is required")
    return float(value)


def _optional_float(value: object) -> float | None:
    return None if value is None else float(value)
