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
_RENDER_KIND = "scale_aware_cpu_splat_l1"


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
    mask-manifest cameras, scale/opacity-aware splat footprints are rasterized,
    and full-vs-removed RGBA images are compared. It is still not a
    covariance-aware 3DGS training renderer, but it measures image-space removal
    delta with Gaussian footprint support instead of reusing the mask-vote loss.
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
            "target_delta_l1": [],
            "non_target_delta_l1": [],
            "locality_score": [],
            "visible_gaussians": [],
        }
        for slot in range(field.slots)
    ]

    for index, frame in enumerate(frames):
        projection = _project_for_render(xyz, frame)
        sigma_px = _gaussian_sigma_pixels(cloud, projection["depth"], frame)
        visible_indices = np.flatnonzero(projection["visible"])
        base = _render_scale_aware_splats(
            projection["u"][visible_indices],
            projection["v"][visible_indices],
            projection["depth"][visible_indices],
            sigma_px[visible_indices],
            rgb[visible_indices],
            alpha[visible_indices],
            width=frame.width,
            height=frame.height,
        )
        base_energy = float(np.mean(np.abs(base)))
        frame_per_slot: list[dict[str, Any]] = []
        for slot in range(field.slots):
            keep = labels[visible_indices] != slot
            removed = _render_scale_aware_splats(
                projection["u"][visible_indices][keep],
                projection["v"][visible_indices][keep],
                projection["depth"][visible_indices][keep],
                sigma_px[visible_indices][keep],
                rgb[visible_indices][keep],
                alpha[visible_indices][keep],
                width=frame.width,
                height=frame.height,
            )
            slot_only = _render_scale_aware_splats(
                projection["u"][visible_indices][~keep],
                projection["v"][visible_indices][~keep],
                projection["depth"][visible_indices][~keep],
                sigma_px[visible_indices][~keep],
                rgb[visible_indices][~keep],
                alpha[visible_indices][~keep],
                width=frame.width,
                height=frame.height,
            )
            diff = np.abs(base - removed)
            delta_l1 = float(np.mean(diff))
            relative = float(delta_l1 / max(base_energy, _EPS))
            affected = np.any(diff > 1e-5, axis=2)
            affected_fraction = float(np.count_nonzero(affected) / max(affected.size, 1))
            target = slot_only[:, :, 3] > 1e-5
            target_delta = _masked_mean_abs(diff, target)
            non_target_delta = _masked_mean_abs(diff, ~target)
            target_energy = float(np.sum(diff[target])) if np.any(target) else 0.0
            total_energy = float(np.sum(diff))
            locality = float(target_energy / max(total_energy, _EPS)) if total_energy > 0 else 0.0
            visible_slot = int(np.count_nonzero(labels[visible_indices] == slot))
            item = {
                "slot": slot,
                "delta_l1": delta_l1,
                "relative_delta_l1": relative,
                "affected_fraction": affected_fraction,
                "target_delta_l1": target_delta,
                "non_target_delta_l1": non_target_delta,
                "locality_score": locality,
                "visible_gaussians": visible_slot,
            }
            frame_per_slot.append(item)
            per_slot_accumulators[slot]["delta_l1"].append(delta_l1)
            per_slot_accumulators[slot]["relative_delta_l1"].append(relative)
            per_slot_accumulators[slot]["affected_fraction"].append(affected_fraction)
            per_slot_accumulators[slot]["target_delta_l1"].append(target_delta)
            per_slot_accumulators[slot]["non_target_delta_l1"].append(non_target_delta)
            per_slot_accumulators[slot]["locality_score"].append(locality)
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
            "mean_target_delta_l1": _mean(accumulator["target_delta_l1"]),
            "mean_non_target_delta_l1": _mean(accumulator["non_target_delta_l1"]),
            "mean_locality_score": _mean(accumulator["locality_score"]),
            "mean_visible_gaussians": _mean(accumulator["visible_gaussians"]),
        }
        for accumulator in per_slot_accumulators
    ]
    relative_values = [float(item["mean_relative_delta_l1"]) for item in per_slot]
    delta_values = [float(item["mean_delta_l1"]) for item in per_slot]
    affected_values = [float(item["mean_affected_fraction"]) for item in per_slot]
    target_values = [float(item["mean_target_delta_l1"]) for item in per_slot]
    non_target_values = [float(item["mean_non_target_delta_l1"]) for item in per_slot]
    locality_values = [float(item["mean_locality_score"]) for item in per_slot]
    return {
        "kind": _RENDER_KIND,
        "frames": len(frames),
        "mean_delta_l1": _mean(delta_values),
        "max_delta_l1": float(np.max(delta_values)) if delta_values else 0.0,
        "mean_relative_delta_l1": _mean(relative_values),
        "max_relative_delta_l1": float(np.max(relative_values)) if relative_values else 0.0,
        "mean_affected_fraction": _mean(affected_values),
        "mean_target_delta_l1": _mean(target_values),
        "mean_non_target_delta_l1": _mean(non_target_values),
        "mean_locality_score": _mean(locality_values),
        "occlusion_effect_score": float(min(1.0, max(0.0, _mean(relative_values)))),
        "per_slot": per_slot,
        "frames_summary": frame_summaries,
    }


def _render_scale_aware_splats(
    u: np.ndarray,
    v: np.ndarray,
    depth: np.ndarray,
    sigma_px: np.ndarray,
    rgb: np.ndarray,
    alpha: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    image = np.zeros((height, width, 4), dtype=np.float32)
    if u.size == 0:
        return image

    sigma = np.clip(np.asarray(sigma_px, dtype=np.float32), 0.75, 4.0)
    base_x = np.floor(u).astype(np.int64)
    base_y = np.floor(v).astype(np.int64)
    max_radius = int(np.ceil(float(np.max(sigma)) * 2.0))
    pixels: list[np.ndarray] = []
    depths: list[np.ndarray] = []
    colors: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for dy in range(-max_radius, max_radius + 1):
        for dx in range(-max_radius, max_radius + 1):
            x = base_x + dx
            y = base_y + dy
            in_frame = (x >= 0) & (x < width) & (y >= 0) & (y < height)
            if not np.any(in_frame):
                continue
            distance_sq = (x.astype(np.float32) + 0.5 - u) ** 2 + (
                y.astype(np.float32) + 0.5 - v
            ) ** 2
            within = in_frame & (distance_sq <= (sigma * 2.0) ** 2)
            if not np.any(within):
                continue
            gaussian_weight = np.exp(-0.5 * distance_sq[within] / np.maximum(sigma[within] ** 2, _EPS))
            contribution_alpha = alpha[within] * gaussian_weight.astype(np.float32, copy=False)
            keep = contribution_alpha > 1e-4
            if not np.any(keep):
                continue
            pixels.append((y[within][keep] * width + x[within][keep]).astype(np.int64, copy=False))
            depths.append(depth[within][keep])
            colors.append(rgb[within][keep])
            weights.append(contribution_alpha[keep])

    if not pixels:
        return image

    pixel = np.concatenate(pixels)
    contribution_depth = np.concatenate(depths)
    contribution_rgb = np.concatenate(colors)
    contribution_alpha = np.concatenate(weights)

    order = np.lexsort((contribution_depth, pixel))
    sorted_pixel = pixel[order]
    first = np.concatenate(([True], sorted_pixel[1:] != sorted_pixel[:-1]))
    chosen = order[first]
    flat = pixel[chosen]
    y = flat // width
    x = flat - y * width
    chosen_alpha = contribution_alpha[chosen]
    image[y, x, :3] = contribution_rgb[chosen] * chosen_alpha[:, None]
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
        "u": u.astype(np.float32, copy=False),
        "v": v.astype(np.float32, copy=False),
        "depth": forward.astype(np.float32, copy=False),
    }


def _gaussian_sigma_pixels(
    cloud: GaussianCloud,
    forward_depth: np.ndarray,
    frame: RenderProbeFrame,
) -> np.ndarray:
    world_radius = _gaussian_world_radius(cloud)
    focal = 0.5 * frame.width / np.tan(0.5 * frame.camera_angle_x)
    sigma = focal * world_radius / np.maximum(forward_depth, 1e-4)
    return np.clip(sigma.astype(np.float32, copy=False), 0.75, 4.0)


def _gaussian_world_radius(cloud: GaussianCloud) -> np.ndarray:
    scale_fields = ("scale_0", "scale_1", "scale_2")
    if not all(name in cloud.fields for name in scale_fields):
        return np.full(cloud.count, 0.02, dtype=np.float32)

    values = np.column_stack([cloud.vertices[name] for name in scale_fields]).astype(np.float32)
    finite = np.isfinite(values)
    if not np.all(finite):
        values = np.where(finite, values, 0.02)

    if values.size and float(np.nanmin(values)) < 0.0:
        values = np.exp(np.clip(values, -20.0, 5.0))
    values = np.abs(values)
    radius = np.max(values, axis=1)
    radius = np.where(radius > 1e-6, radius, 0.02)
    return radius.astype(np.float32, copy=False)


def _coverage_fraction(image: np.ndarray) -> float:
    alpha = image[:, :, 3]
    return float(np.count_nonzero(alpha > 1e-5) / max(alpha.size, 1))


def _masked_mean_abs(diff: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return 0.0
    return float(np.mean(diff[mask]))


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
