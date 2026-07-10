from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.gaussian_decoder import (
    object_opacity_scales_from_logits,
    object_scale_multipliers_from_log_offsets,
)
from objgauss.pipelines.trainable_kernel import (
    TrainableKernelFrame,
    TrainableKernelImageTarget,
    validate_trainable_image_target,
)

TRAINING_RENDERER_API_SCHEMA = "objgauss-training-renderer-api-v1"
CPU_IMAGE_SPLAT_RENDERER = "cpu-image-point-splat-differentiable-v1"
CPU_IMAGE_SPLAT_GRADIENT_PATH = "analytic-color-assignment-gradient-v1"

__all__ = (
    "TRAINING_RENDERER_API_SCHEMA",
    "CPU_IMAGE_SPLAT_RENDERER",
    "CPU_IMAGE_SPLAT_GRADIENT_PATH",
    "TrainingRendererFrameLoss",
    "TrainingRendererLossResult",
    "evaluate_training_renderer_loss",
    "validate_training_renderer_summary",
)


@dataclass(frozen=True)
class TrainingRendererFrameLoss:
    frame_index: int
    image_render_loss: float
    supervised_pixels: int
    visibility_policy: str
    rendered_shape: tuple[int, int, int]
    point_count: int
    max_abs_error: float
    mean_abs_error: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "image_render_loss": float(self.image_render_loss),
            "supervised_pixels": int(self.supervised_pixels),
            "visibility_policy": self.visibility_policy,
            "rendered_shape": [int(value) for value in self.rendered_shape],
            "point_count": int(self.point_count),
            "max_abs_error": float(self.max_abs_error),
            "mean_abs_error": float(self.mean_abs_error),
        }


@dataclass(frozen=True)
class TrainingRendererLossResult:
    schema: str
    renderer_name: str
    gradient_path: str
    frame_count: int
    image_render_loss: float
    frame_losses: tuple[TrainingRendererFrameLoss, ...]
    rendered_images: tuple[np.ndarray, ...]
    gradient_decoder_colors: np.ndarray
    gradient_decoder_opacity_logits: np.ndarray
    gradient_decoder_scale_log_offsets: np.ndarray
    gradient_assignments: tuple[np.ndarray, ...]
    differentiable_fields: tuple[str, ...]
    frozen_fields: tuple[str, ...]
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        assignment_shapes = [
            [int(value) for value in gradient.shape]
            for gradient in self.gradient_assignments
        ]
        assignment_l2 = [
            float(np.linalg.norm(gradient))
            for gradient in self.gradient_assignments
        ]
        return {
            "schema": self.schema,
            "status": "ready" if not self.blockers else "blocked",
            "renderer_name": self.renderer_name,
            "gradient_path": self.gradient_path,
            "frame_count": int(self.frame_count),
            "image_render_loss": float(self.image_render_loss),
            "frame_losses": [loss.as_dict() for loss in self.frame_losses],
            "differentiable_fields": list(self.differentiable_fields),
            "frozen_fields": list(self.frozen_fields),
            "gradients": {
                "decoder_colors_shape": [int(value) for value in self.gradient_decoder_colors.shape],
                "decoder_colors_l2": float(np.linalg.norm(self.gradient_decoder_colors)),
                "decoder_opacity_logits_shape": [
                    int(value)
                    for value in self.gradient_decoder_opacity_logits.shape
                ],
                "decoder_opacity_logits_l2": float(
                    np.linalg.norm(self.gradient_decoder_opacity_logits)
                ),
                "decoder_scale_log_offsets_shape": [
                    int(value)
                    for value in self.gradient_decoder_scale_log_offsets.shape
                ],
                "decoder_scale_log_offsets_l2": float(
                    np.linalg.norm(self.gradient_decoder_scale_log_offsets)
                ),
                "assignment_shapes": assignment_shapes,
                "assignment_l2": assignment_l2,
            },
            "blockers": list(self.blockers),
        }


def evaluate_training_renderer_loss(
    frames: Sequence[TrainableKernelFrame],
    assignments: Sequence[np.ndarray],
    decoder_colors: np.ndarray,
    *,
    decoder_opacity_logits: np.ndarray | None = None,
    decoder_scale_log_offsets: np.ndarray | None = None,
    default_opacity: float = 1.0,
    renderer_name: str = CPU_IMAGE_SPLAT_RENDERER,
    gradient_path: str = CPU_IMAGE_SPLAT_GRADIENT_PATH,
) -> TrainingRendererLossResult:
    """Evaluate a small image-space renderer loss with an explicit gradient path.

    This is a training loss producer, not the browser viewer renderer. It keeps
    camera projection and visibility fixed, then exposes analytic gradients for
    decoder colors and soft assignments so later renderers can preserve the same
    contract.
    """

    if not frames:
        raise ValueError("at least one frame is required")
    if len(frames) != len(assignments):
        raise ValueError("assignments must have one matrix per frame")
    colors = _array2d(decoder_colors, "decoder_colors", columns=3)
    if colors.shape[0] == 0:
        raise ValueError("decoder_colors must contain at least one slot")
    if not 0.0 <= default_opacity <= 1.0:
        raise ValueError("default_opacity must be in [0, 1]")
    opacity_logits = _optional_array1d(decoder_opacity_logits, "decoder_opacity_logits")
    if opacity_logits is not None and opacity_logits.shape[0] != colors.shape[0]:
        raise ValueError("decoder_opacity_logits length must match decoder_colors rows")
    scale_log_offsets = _optional_array1d(decoder_scale_log_offsets, "decoder_scale_log_offsets")
    if scale_log_offsets is not None and scale_log_offsets.shape[0] != colors.shape[0]:
        raise ValueError("decoder_scale_log_offsets length must match decoder_colors rows")

    rendered_images: list[np.ndarray] = []
    frame_losses: list[TrainingRendererFrameLoss] = []
    assignment_gradients: list[np.ndarray] = []
    decoder_gradient = np.zeros_like(colors, dtype=np.float32)
    opacity_gradient = (
        np.zeros((0,), dtype=np.float32)
        if opacity_logits is None
        else np.zeros_like(opacity_logits, dtype=np.float32)
    )
    scale_gradient = (
        np.zeros((0,), dtype=np.float32)
        if scale_log_offsets is None
        else np.zeros_like(scale_log_offsets, dtype=np.float32)
    )
    total_loss = 0.0

    for frame_index, (frame, assignment) in enumerate(zip(frames, assignments, strict=True)):
        frame_result = _evaluate_frame(
            frame,
            assignment,
            colors,
            opacity_logits=opacity_logits,
            scale_log_offsets=scale_log_offsets,
            default_opacity=default_opacity,
            frame_index=frame_index,
        )
        rendered_images.append(frame_result["rendered_image"])
        frame_losses.append(frame_result["frame_loss"])
        assignment_gradients.append(frame_result["gradient_assignment"])
        decoder_gradient += frame_result["gradient_decoder_colors"] / len(frames)
        if opacity_logits is not None:
            opacity_gradient += frame_result["gradient_decoder_opacity_logits"] / len(frames)
        if scale_log_offsets is not None:
            scale_gradient += frame_result["gradient_decoder_scale_log_offsets"] / len(frames)
        total_loss += frame_result["frame_loss"].image_render_loss / len(frames)

    differentiable_fields = ["decoder_colors", "assignments"]
    frozen_fields = ["positions", "camera", "visibility_mask", "point_radius"]
    if opacity_logits is not None:
        differentiable_fields.append("decoder_opacity_logits")
        frozen_fields.append("source_opacities")
    if scale_log_offsets is not None:
        differentiable_fields.append("decoder_scale_log_offsets")
        frozen_fields.append("point_scale_proxy_base")

    return TrainingRendererLossResult(
        schema=TRAINING_RENDERER_API_SCHEMA,
        renderer_name=renderer_name,
        gradient_path=gradient_path,
        frame_count=len(frames),
        image_render_loss=float(total_loss),
        frame_losses=tuple(frame_losses),
        rendered_images=tuple(rendered_images),
        gradient_decoder_colors=decoder_gradient,
        gradient_decoder_opacity_logits=opacity_gradient,
        gradient_decoder_scale_log_offsets=scale_gradient,
        gradient_assignments=tuple(gradient / len(frames) for gradient in assignment_gradients),
        differentiable_fields=tuple(differentiable_fields),
        frozen_fields=tuple(frozen_fields),
        blockers=(),
    )


def validate_training_renderer_summary(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        raise TypeError("training renderer summary must be a dict")
    if payload.get("schema") != TRAINING_RENDERER_API_SCHEMA:
        raise ValueError(f"unsupported training renderer schema: {payload.get('schema')}")
    required = (
        "status",
        "renderer_name",
        "gradient_path",
        "frame_count",
        "image_render_loss",
        "frame_losses",
        "differentiable_fields",
        "frozen_fields",
        "gradients",
        "blockers",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"training renderer summary missing keys: {', '.join(missing)}")
    if payload.get("status") != "ready":
        raise ValueError("training renderer summary is not ready")
    return True


def _evaluate_frame(
    frame: TrainableKernelFrame,
    assignment: np.ndarray,
    decoder_colors: np.ndarray,
    *,
    opacity_logits: np.ndarray | None,
    scale_log_offsets: np.ndarray | None,
    default_opacity: float,
    frame_index: int,
) -> dict[str, Any]:
    target = frame.image_target
    if target is None:
        raise ValueError(f"frames[{frame_index}].image_target is required")
    validate_trainable_image_target(target)
    positions = _array2d(frame.positions, f"frames[{frame_index}].positions", columns=3)
    assignment = _array2d(assignment, f"assignments[{frame_index}]", columns=decoder_colors.shape[0])
    if assignment.shape[0] != positions.shape[0]:
        raise ValueError(f"assignments[{frame_index}] rows must match frame positions")
    if np.any(assignment < 0.0):
        raise ValueError(f"assignments[{frame_index}] must be non-negative")
    row_sums = assignment.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5, rtol=0.0):
        raise ValueError(f"assignments[{frame_index}] rows must sum to 1")

    point_color = np.clip(assignment @ decoder_colors, 0.0, 1.0)
    opacity_scales = None
    scale_multipliers = None
    point_opacity = np.ones((assignment.shape[0],), dtype=np.float32)
    point_scale = np.ones((assignment.shape[0],), dtype=np.float32)
    if opacity_logits is not None:
        opacity_scales = object_opacity_scales_from_logits(opacity_logits)
        point_opacity = np.clip(
            float(default_opacity) * (assignment @ opacity_scales),
            0.0,
            1.0,
        ).astype(np.float32, copy=False)
    if scale_log_offsets is not None:
        scale_multipliers = object_scale_multipliers_from_log_offsets(scale_log_offsets)
        point_scale = (assignment @ scale_multipliers).astype(np.float32, copy=False)
    contribution_scale = (point_opacity * point_scale).astype(np.float32, copy=False)
    point_rgb = point_color * contribution_scale[:, None]
    rendered_image, counts, point_pixels = _render_point_splat_image(
        positions,
        point_rgb,
        target,
    )
    mask = np.asarray(target.visibility_mask, dtype=bool)
    diff = rendered_image - np.asarray(target.image, dtype=np.float32)
    supervised = diff[mask]
    if supervised.size == 0:
        raise ValueError(f"frames[{frame_index}].image_target has no supervised pixels")
    image_render_loss = float(np.mean(supervised ** 2))
    abs_error = np.abs(supervised)

    gradient_image = np.zeros_like(rendered_image, dtype=np.float32)
    gradient_image[mask] = (2.0 / supervised.size) * diff[mask]
    gradient_point_rgb = _point_rgb_gradient(point_pixels, counts, gradient_image)
    gradient_point_color = gradient_point_rgb * contribution_scale[:, None]
    gradient_decoder_colors = assignment.T @ gradient_point_color
    gradient_assignment = gradient_point_color @ decoder_colors.T
    if opacity_logits is None:
        gradient_opacity_logits = np.zeros((0,), dtype=np.float32)
    else:
        assert opacity_scales is not None
        gradient_point_opacity = np.sum(
            gradient_point_rgb * point_color * point_scale[:, None],
            axis=1,
        )
        gradient_opacity_scales = float(default_opacity) * (assignment.T @ gradient_point_opacity)
        gradient_opacity_logits = gradient_opacity_scales * _object_opacity_logit_derivative(opacity_logits)
        gradient_assignment = gradient_assignment + (
            (float(default_opacity) * gradient_point_opacity)[:, None]
            * opacity_scales[None, :]
        )
    if scale_log_offsets is None:
        gradient_scale_log_offsets = np.zeros((0,), dtype=np.float32)
    else:
        assert scale_multipliers is not None
        gradient_point_scale = np.sum(
            gradient_point_rgb * point_color * point_opacity[:, None],
            axis=1,
        )
        gradient_scale_multipliers = assignment.T @ gradient_point_scale
        gradient_scale_log_offsets = (
            gradient_scale_multipliers
            * _object_scale_log_offset_derivative(scale_log_offsets)
        )
        gradient_assignment = gradient_assignment + (
            gradient_point_scale[:, None] * scale_multipliers[None, :]
        )

    return {
        "rendered_image": rendered_image,
        "gradient_decoder_colors": gradient_decoder_colors.astype(np.float32, copy=False),
        "gradient_decoder_opacity_logits": gradient_opacity_logits.astype(np.float32, copy=False),
        "gradient_decoder_scale_log_offsets": gradient_scale_log_offsets.astype(np.float32, copy=False),
        "gradient_assignment": gradient_assignment.astype(np.float32, copy=False),
        "frame_loss": TrainingRendererFrameLoss(
            frame_index=frame_index,
            image_render_loss=image_render_loss,
            supervised_pixels=int(np.count_nonzero(mask)),
            visibility_policy=target.visibility_policy,
            rendered_shape=tuple(int(value) for value in rendered_image.shape),
            point_count=int(positions.shape[0]),
            max_abs_error=float(np.max(abs_error)),
            mean_abs_error=float(np.mean(abs_error)),
        ),
    }


def _render_point_splat_image(
    positions: np.ndarray,
    point_rgb: np.ndarray,
    target: TrainableKernelImageTarget,
) -> tuple[np.ndarray, np.ndarray, tuple[tuple[tuple[int, int], ...], ...]]:
    height, width = int(target.image.shape[0]), int(target.image.shape[1])
    pixels = _project_pixels(positions, target)
    image = np.zeros((height, width, 3), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    point_pixels: list[tuple[tuple[int, int], ...]] = []
    radius = int(target.point_radius)
    for pixel, color in zip(pixels, point_rgb, strict=True):
        px, py = int(pixel[0]), int(pixel[1])
        covered: list[tuple[int, int]] = []
        for y in range(max(0, py - radius), min(height, py + radius + 1)):
            for x in range(max(0, px - radius), min(width, px + radius + 1)):
                image[y, x] += color
                counts[y, x] += 1.0
                covered.append((y, x))
        point_pixels.append(tuple(covered))
    covered_mask = counts > 0
    image[covered_mask] = image[covered_mask] / counts[covered_mask, None]
    return image, counts, tuple(point_pixels)


def _point_rgb_gradient(
    point_pixels: Sequence[Sequence[tuple[int, int]]],
    counts: np.ndarray,
    gradient_image: np.ndarray,
) -> np.ndarray:
    gradient = np.zeros((len(point_pixels), 3), dtype=np.float32)
    for point_index, pixels in enumerate(point_pixels):
        for y, x in pixels:
            count = float(counts[y, x])
            if count > 0:
                gradient[point_index] += gradient_image[y, x] / count
    return gradient


def _object_opacity_logit_derivative(logits: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(logits, dtype=np.float32), -60.0, 60.0)
    sigmoid = 1.0 / (1.0 + np.exp(-clipped))
    derivative = sigmoid * (1.0 - sigmoid)
    derivative[sigmoid < 0.05] = 0.0
    derivative[sigmoid > 1.0] = 0.0
    return derivative.astype(np.float32, copy=False)


def _object_scale_log_offset_derivative(log_offsets: np.ndarray) -> np.ndarray:
    values = np.asarray(log_offsets, dtype=np.float32)
    min_log = np.log(0.75)
    max_log = np.log(1.25)
    multipliers = object_scale_multipliers_from_log_offsets(values)
    active = (values >= min_log) & (values <= max_log)
    return np.where(active, multipliers, 0.0).astype(np.float32, copy=False)


def _optional_array1d(value: np.ndarray | None, label: str) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a 1D array")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one value")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def _project_pixels(positions: np.ndarray, target: TrainableKernelImageTarget) -> np.ndarray:
    intrinsics = np.asarray(target.camera.intrinsics, dtype=np.float32)
    xy1 = np.column_stack(
        [
            positions[:, 0],
            positions[:, 1],
            np.ones(positions.shape[0], dtype=np.float32),
        ]
    )
    projected = xy1 @ intrinsics.T
    width, height = int(target.camera.width), int(target.camera.height)
    px = np.clip(np.rint(projected[:, 0]), 0, width - 1).astype(np.int32)
    py = np.clip(np.rint(projected[:, 1]), 0, height - 1).astype(np.int32)
    return np.column_stack([px, py])


def _array2d(value: np.ndarray, label: str, *, columns: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{label} must have {columns} columns")
    if array.shape[1] == 0:
        raise ValueError(f"{label} must have at least one column")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array
