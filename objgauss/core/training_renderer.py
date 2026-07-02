from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.trainable_kernel import (
    TrainableKernelFrame,
    TrainableKernelImageTarget,
    validate_trainable_image_target,
)

TRAINING_RENDERER_API_SCHEMA = "objgauss-training-renderer-api-v1"
CPU_IMAGE_SPLAT_RENDERER = "cpu-image-point-splat-differentiable-v1"
CPU_IMAGE_SPLAT_GRADIENT_PATH = "analytic-color-assignment-gradient-v1"


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

    rendered_images: list[np.ndarray] = []
    frame_losses: list[TrainingRendererFrameLoss] = []
    assignment_gradients: list[np.ndarray] = []
    decoder_gradient = np.zeros_like(colors, dtype=np.float32)
    total_loss = 0.0

    for frame_index, (frame, assignment) in enumerate(zip(frames, assignments, strict=True)):
        frame_result = _evaluate_frame(
            frame,
            assignment,
            colors,
            frame_index=frame_index,
        )
        rendered_images.append(frame_result["rendered_image"])
        frame_losses.append(frame_result["frame_loss"])
        assignment_gradients.append(frame_result["gradient_assignment"])
        decoder_gradient += frame_result["gradient_decoder_colors"] / len(frames)
        total_loss += frame_result["frame_loss"].image_render_loss / len(frames)

    return TrainingRendererLossResult(
        schema=TRAINING_RENDERER_API_SCHEMA,
        renderer_name=renderer_name,
        gradient_path=gradient_path,
        frame_count=len(frames),
        image_render_loss=float(total_loss),
        frame_losses=tuple(frame_losses),
        rendered_images=tuple(rendered_images),
        gradient_decoder_colors=decoder_gradient,
        gradient_assignments=tuple(gradient / len(frames) for gradient in assignment_gradients),
        differentiable_fields=("decoder_colors", "assignments"),
        frozen_fields=("positions", "camera", "visibility_mask", "point_radius"),
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

    point_rgb = np.clip(assignment @ decoder_colors, 0.0, 1.0)
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
    gradient_decoder_colors = assignment.T @ gradient_point_rgb
    gradient_assignment = gradient_point_rgb @ decoder_colors.T

    return {
        "rendered_image": rendered_image,
        "gradient_decoder_colors": gradient_decoder_colors.astype(np.float32, copy=False),
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
