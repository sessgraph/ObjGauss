from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.gaussian_decoder import OBJECT_STATE_GAUSSIAN_DECODE_SCHEMA
from objgauss.core.object_state import (
    ObjectStateProjection,
    project_object_states,
    validate_assignment_matrix,
)
from objgauss.core.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_GSPLAT,
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
    TrainableKernelFrame,
    TrainableKernelImageTarget,
    image_target_contract_summary,
)
from objgauss.core.training_renderer import TrainingRendererLossResult

OBJECT_STATE_GAUSSIAN_DECODER_STATE_SCHEMA = "objgauss-object-state-gaussian-decoder-state-v1"
OBJECT_STATE_GAUSSIAN_DECODER_TRAINING_SCHEMA = "objgauss-object-state-gaussian-decoder-training-v1"
_EPS = 1e-8


@dataclass(frozen=True)
class ObjectStateGaussianDecoderState:
    object_colors: np.ndarray
    step: int = 0
    source: str = "initialized"
    schema: str = OBJECT_STATE_GAUSSIAN_DECODER_STATE_SCHEMA

    @property
    def slots(self) -> int:
        return int(self.object_colors.shape[0])

    def as_dict(self) -> dict[str, Any]:
        colors = validate_object_state_gaussian_decoder_state(self).object_colors
        return {
            "schema": self.schema,
            "source": self.source,
            "step": int(self.step),
            "slots": int(colors.shape[0]),
            "trained_fields": ["object_colors"],
            "object_colors": np.round(colors, 6).tolist(),
        }


@dataclass(frozen=True)
class ObjectStateGaussianDecoderLoss:
    iteration: int
    total_loss: float
    render_loss: float
    image_render_loss: float
    object_loss: float
    temporal_loss: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "total_loss": float(self.total_loss),
            "render_loss": float(self.render_loss),
            "image_render_loss": float(self.image_render_loss),
            "object_loss": float(self.object_loss),
            "temporal_loss": float(self.temporal_loss),
        }


@dataclass(frozen=True)
class ObjectStateGaussianDecoderTrainingResult:
    schema: str
    decoder_schema: str
    slots: int
    frame_count: int
    iterations: int
    learning_rate: float
    render_weight: float
    image_render_weight: float
    object_weight: float
    temporal_weight: float
    image_renderer: str
    gaussian_scale: float
    gaussian_opacity: float
    initial_state: ObjectStateGaussianDecoderState
    final_state: ObjectStateGaussianDecoderState
    initial_loss: ObjectStateGaussianDecoderLoss
    final_loss: ObjectStateGaussianDecoderLoss
    history: tuple[ObjectStateGaussianDecoderLoss, ...]
    assignments: tuple[np.ndarray, ...]
    object_state_projections: tuple[ObjectStateProjection, ...]
    rendered_rgb: tuple[np.ndarray, ...]
    final_renderer_api: TrainingRendererLossResult | None
    image_targets: tuple[TrainableKernelImageTarget | None, ...]
    trained_fields: tuple[str, ...]
    frozen_fields: tuple[str, ...]
    gpu_used: bool
    vram_reserve_gb: int

    def as_dict(self) -> dict[str, Any]:
        renderer_api = self.final_renderer_api.as_dict() if self.final_renderer_api is not None else None
        return {
            "schema": self.schema,
            "kind": "object_state_gaussian_decoder_training",
            "decoder_schema": self.decoder_schema,
            "slots": int(self.slots),
            "frame_count": int(self.frame_count),
            "iterations": int(self.iterations),
            "learning_rate": float(self.learning_rate),
            "weights": {
                "render": float(self.render_weight),
                "image_render": float(self.image_render_weight),
                "object": float(self.object_weight),
                "temporal": float(self.temporal_weight),
            },
            "image_renderer": self.image_renderer,
            "gaussian_policy": {
                "default_scale": float(self.gaussian_scale),
                "default_opacity": float(self.gaussian_opacity),
            },
            "initial_loss": self.initial_loss.as_dict(),
            "final_loss": self.final_loss.as_dict(),
            "loss_decreased": bool(self.final_loss.total_loss < self.initial_loss.total_loss),
            "render_loss_decreased": bool(self.final_loss.render_loss < self.initial_loss.render_loss),
            "image_render_loss_decreased": bool(
                self.final_loss.image_render_loss < self.initial_loss.image_render_loss
            ),
            "initial_decoder_state": self.initial_state.as_dict(),
            "final_decoder_state": self.final_state.as_dict(),
            "history": [loss.as_dict() for loss in self.history],
            "trained_fields": list(self.trained_fields),
            "frozen_fields": list(self.frozen_fields),
            "renderer_api": renderer_api,
            "image_target_contract": image_target_contract_summary(self.image_targets),
            "gpu_policy": {
                "uses_gpu": bool(self.gpu_used),
                "vram_reserve_gb": int(self.vram_reserve_gb),
            },
            "assignments": [
                {
                    "frame_index": index,
                    "shape": [int(value) for value in assignment.shape],
                    "slot_mass": np.round(assignment.sum(axis=0), 6).tolist(),
                }
                for index, assignment in enumerate(self.assignments)
            ],
            "object_states": [
                [
                    {
                        "id": int(state.id),
                        "slot_mass": float(state.slot_mass),
                        "confidence": float(state.confidence),
                        "centroid": np.round(state.centroid, 6).tolist(),
                        "status": state.status,
                    }
                    for state in projection.states
                ]
                for projection in self.object_state_projections
            ],
        }


@dataclass(frozen=True)
class _DecoderTrainingEval:
    loss: ObjectStateGaussianDecoderLoss
    gradient: np.ndarray
    renderer_api: TrainingRendererLossResult | None
    projections: tuple[ObjectStateProjection, ...]
    rendered_rgb: tuple[np.ndarray, ...]


def initialize_object_state_gaussian_decoder(
    *,
    slots: int,
    seed: int = 0,
    low: float = 0.15,
    high: float = 0.85,
) -> ObjectStateGaussianDecoderState:
    if slots < 1:
        raise ValueError("slots must be >= 1")
    if not 0.0 <= low <= high <= 1.0:
        raise ValueError("decoder color init range must be within [0, 1]")
    rng = np.random.default_rng(seed)
    return ObjectStateGaussianDecoderState(
        object_colors=rng.uniform(low, high, size=(slots, 3)).astype(np.float32),
        step=0,
        source="random_object_color_init",
    )


def train_object_state_gaussian_decoder(
    frames: Sequence[TrainableKernelFrame],
    assignments: Sequence[np.ndarray],
    *,
    initial_state: ObjectStateGaussianDecoderState | None = None,
    iterations: int = 8,
    learning_rate: float = 0.5,
    render_weight: float = 0.0,
    image_render_weight: float = 1.0,
    object_weight: float = 0.0,
    temporal_weight: float = 0.0,
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    gaussian_scale: float = 0.5,
    gaussian_opacity: float = 1.0,
    seed: int = 0,
    record_every: int | None = None,
    vram_reserve_gb: int = 1,
) -> ObjectStateGaussianDecoderTrainingResult:
    """Train only ObjectState Gaussian decoder colors against renderer loss."""

    checked_frames, checked_assignments = _validate_frames_and_assignments(frames, assignments)
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")
    if gaussian_scale <= 0:
        raise ValueError("gaussian_scale must be > 0")
    if not 0.0 <= gaussian_opacity <= 1.0:
        raise ValueError("gaussian_opacity must be in [0, 1]")
    if vram_reserve_gb < 0:
        raise ValueError("vram_reserve_gb must be >= 0")
    for name, weight in {
        "render_weight": render_weight,
        "image_render_weight": image_render_weight,
        "object_weight": object_weight,
        "temporal_weight": temporal_weight,
    }.items():
        if weight < 0:
            raise ValueError(f"{name} must be >= 0")
    if image_render_weight > 0 and not all(frame.image_target is not None for frame in checked_frames):
        raise ValueError("image_render_weight requires every frame to bind image_target")
    if object_weight > 0 and not all(frame.target_assignment is not None for frame in checked_frames):
        raise ValueError("object_weight requires every frame to bind target_assignment")

    slots = int(checked_assignments[0].shape[1])
    state = initial_state or initialize_object_state_gaussian_decoder(slots=slots, seed=seed)
    state = validate_object_state_gaussian_decoder_state(state)
    if state.slots != slots:
        raise ValueError("decoder state slots must match assignment slot count")

    record_stride = iterations if record_every is None else max(1, int(record_every))
    history: list[ObjectStateGaussianDecoderLoss] = []
    colors = state.object_colors.astype(np.float32, copy=True)

    initial_eval = _evaluate_decoder_training(
        checked_frames,
        checked_assignments,
        colors,
        iteration=0,
        render_weight=render_weight,
        image_render_weight=image_render_weight,
        object_weight=object_weight,
        temporal_weight=temporal_weight,
        image_renderer=image_renderer,
        gaussian_scale=gaussian_scale,
        gaussian_opacity=gaussian_opacity,
    )
    history.append(initial_eval.loss)

    for iteration in range(1, iterations + 1):
        current_eval = _evaluate_decoder_training(
            checked_frames,
            checked_assignments,
            colors,
            iteration=iteration - 1,
            render_weight=render_weight,
            image_render_weight=image_render_weight,
            object_weight=object_weight,
            temporal_weight=temporal_weight,
            image_renderer=image_renderer,
            gaussian_scale=gaussian_scale,
            gaussian_opacity=gaussian_opacity,
        )
        colors = np.clip(colors - learning_rate * current_eval.gradient, 0.0, 1.0)
        if iteration == iterations or iteration % record_stride == 0:
            history.append(
                _evaluate_decoder_training(
                    checked_frames,
                    checked_assignments,
                    colors,
                    iteration=iteration,
                    render_weight=render_weight,
                    image_render_weight=image_render_weight,
                    object_weight=object_weight,
                    temporal_weight=temporal_weight,
                    image_renderer=image_renderer,
                    gaussian_scale=gaussian_scale,
                    gaussian_opacity=gaussian_opacity,
                ).loss
            )

    final_eval = _evaluate_decoder_training(
        checked_frames,
        checked_assignments,
        colors,
        iteration=iterations,
        render_weight=render_weight,
        image_render_weight=image_render_weight,
        object_weight=object_weight,
        temporal_weight=temporal_weight,
        image_renderer=image_renderer,
        gaussian_scale=gaussian_scale,
        gaussian_opacity=gaussian_opacity,
    )
    if history[-1].iteration != iterations:
        history.append(final_eval.loss)
    else:
        history[-1] = final_eval.loss

    return ObjectStateGaussianDecoderTrainingResult(
        schema=OBJECT_STATE_GAUSSIAN_DECODER_TRAINING_SCHEMA,
        decoder_schema=OBJECT_STATE_GAUSSIAN_DECODE_SCHEMA,
        slots=slots,
        frame_count=len(checked_frames),
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        render_weight=float(render_weight),
        image_render_weight=float(image_render_weight),
        object_weight=float(object_weight),
        temporal_weight=float(temporal_weight),
        image_renderer=image_renderer,
        gaussian_scale=float(gaussian_scale),
        gaussian_opacity=float(gaussian_opacity),
        initial_state=state,
        final_state=ObjectStateGaussianDecoderState(
            object_colors=colors.astype(np.float32, copy=True),
            step=int(iterations),
            source="trained_renderer_gradient_object_colors",
        ),
        initial_loss=initial_eval.loss,
        final_loss=final_eval.loss,
        history=tuple(history),
        assignments=checked_assignments,
        object_state_projections=final_eval.projections,
        rendered_rgb=final_eval.rendered_rgb,
        final_renderer_api=final_eval.renderer_api,
        image_targets=tuple(frame.image_target for frame in checked_frames),
        trained_fields=("object_colors",),
        frozen_fields=("assignments", "means", "quats", "scales", "opacities", "cameras"),
        gpu_used=image_renderer == TRAINING_IMAGE_RENDERER_GSPLAT,
        vram_reserve_gb=int(vram_reserve_gb),
    )


def validate_object_state_gaussian_decoder_state(
    state: ObjectStateGaussianDecoderState,
) -> ObjectStateGaussianDecoderState:
    if state.schema != OBJECT_STATE_GAUSSIAN_DECODER_STATE_SCHEMA:
        raise ValueError(f"unsupported decoder state schema: {state.schema}")
    colors = _array2d(state.object_colors, "object_colors", columns=3)
    if colors.shape[0] < 1:
        raise ValueError("object_colors must contain at least one slot")
    if np.any(colors < 0.0) or np.any(colors > 1.0):
        raise ValueError("object_colors must be in [0, 1]")
    if state.step < 0:
        raise ValueError("step must be >= 0")
    return state


def _validate_frames_and_assignments(
    frames: Sequence[TrainableKernelFrame],
    assignments: Sequence[np.ndarray],
) -> tuple[tuple[TrainableKernelFrame, ...], tuple[np.ndarray, ...]]:
    checked_frames = tuple(frames)
    checked_assignments = tuple(np.asarray(assignment, dtype=np.float32) for assignment in assignments)
    if not checked_frames:
        raise ValueError("at least one frame is required")
    if len(checked_frames) != len(checked_assignments):
        raise ValueError("assignments must have one matrix per frame")
    slots: int | None = None
    for index, (frame, assignment) in enumerate(zip(checked_frames, checked_assignments, strict=True)):
        positions = _array2d(frame.positions, f"frames[{index}].positions", columns=3)
        features = _array2d(frame.features, f"frames[{index}].features")
        target_rgb = _array2d(frame.target_rgb, f"frames[{index}].target_rgb", columns=3)
        if features.shape[0] != positions.shape[0]:
            raise ValueError(f"frames[{index}].features rows must match positions")
        if target_rgb.shape[0] != positions.shape[0]:
            raise ValueError(f"frames[{index}].target_rgb rows must match positions")
        checked = validate_assignment_matrix(assignment, evidence_count=positions.shape[0])
        checked_assignments = (
            *checked_assignments[:index],
            checked,
            *checked_assignments[index + 1 :],
        )
        slots = checked.shape[1] if slots is None else slots
        if checked.shape[1] != slots:
            raise ValueError("all assignments must have the same slot count")
    return checked_frames, checked_assignments


def _evaluate_decoder_training(
    frames: tuple[TrainableKernelFrame, ...],
    assignments: tuple[np.ndarray, ...],
    colors: np.ndarray,
    *,
    iteration: int,
    render_weight: float,
    image_render_weight: float,
    object_weight: float,
    temporal_weight: float,
    image_renderer: str,
    gaussian_scale: float,
    gaussian_opacity: float,
) -> _DecoderTrainingEval:
    colors = _array2d(colors, "object_colors", columns=3)
    render_loss, render_gradient, rendered_rgb = _point_render_loss_and_gradient(frames, assignments, colors)
    image_render_loss = 0.0
    image_gradient = np.zeros_like(colors, dtype=np.float32)
    renderer_api = None
    if image_render_weight > 0:
        renderer_api = _evaluate_image_renderer(
            frames,
            assignments,
            colors,
            image_renderer=image_renderer,
            gaussian_scale=gaussian_scale,
            gaussian_opacity=gaussian_opacity,
        )
        image_render_loss = float(renderer_api.image_render_loss)
        image_gradient = renderer_api.gradient_decoder_colors.astype(np.float32, copy=False)
    projections = tuple(
        project_object_states(_frame_cloud(frame), assignment, evidence_features=frame.features)
        for frame, assignment in zip(frames, assignments, strict=True)
    )
    object_loss = _object_assignment_loss(frames, assignments)
    temporal_loss = _temporal_centroid_loss(projections)
    total_loss = (
        render_weight * render_loss
        + image_render_weight * image_render_loss
        + object_weight * object_loss
        + temporal_weight * temporal_loss
    )
    gradient = (
        render_weight * render_gradient
        + image_render_weight * image_gradient
    ).astype(np.float32, copy=False)
    return _DecoderTrainingEval(
        loss=ObjectStateGaussianDecoderLoss(
            iteration=int(iteration),
            total_loss=float(total_loss),
            render_loss=float(render_loss),
            image_render_loss=float(image_render_loss),
            object_loss=float(object_loss),
            temporal_loss=float(temporal_loss),
        ),
        gradient=gradient,
        renderer_api=renderer_api,
        projections=projections,
        rendered_rgb=rendered_rgb,
    )


def _evaluate_image_renderer(
    frames: tuple[TrainableKernelFrame, ...],
    assignments: tuple[np.ndarray, ...],
    colors: np.ndarray,
    *,
    image_renderer: str,
    gaussian_scale: float,
    gaussian_opacity: float,
) -> TrainingRendererLossResult:
    if image_renderer == TRAINING_IMAGE_RENDERER_POINT:
        from objgauss.core.training_renderer import evaluate_training_renderer_loss

        return evaluate_training_renderer_loss(frames, assignments, colors)
    elif image_renderer == TRAINING_IMAGE_RENDERER_GSPLAT:
        from objgauss.core.gsplat_training_renderer import (
            evaluate_gsplat_training_renderer_loss as evaluate_training_renderer_loss,
        )
        return evaluate_training_renderer_loss(
            frames,
            assignments,
            colors,
            default_scale=gaussian_scale,
            default_opacity=gaussian_opacity,
        )
    else:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")


def _point_render_loss_and_gradient(
    frames: tuple[TrainableKernelFrame, ...],
    assignments: tuple[np.ndarray, ...],
    colors: np.ndarray,
) -> tuple[float, np.ndarray, tuple[np.ndarray, ...]]:
    losses: list[float] = []
    gradient = np.zeros_like(colors, dtype=np.float32)
    rendered: list[np.ndarray] = []
    for frame, assignment in zip(frames, assignments, strict=True):
        predicted = np.clip(assignment @ colors, 0.0, 1.0).astype(np.float32, copy=False)
        target = _array2d(frame.target_rgb, "frame.target_rgb", columns=3)
        diff = predicted - target
        losses.append(float(np.mean(diff ** 2)))
        gradient += (assignment.T @ ((2.0 / max(float(diff.size), _EPS)) * diff)) / len(frames)
        rendered.append(predicted)
    return float(np.mean(losses)), gradient.astype(np.float32, copy=False), tuple(rendered)


def _object_assignment_loss(
    frames: tuple[TrainableKernelFrame, ...],
    assignments: tuple[np.ndarray, ...],
) -> float:
    losses: list[float] = []
    for frame, assignment in zip(frames, assignments, strict=True):
        if frame.target_assignment is None:
            continue
        target = validate_assignment_matrix(frame.target_assignment, evidence_count=assignment.shape[0])
        if target.shape[1] != assignment.shape[1]:
            raise ValueError("target_assignment slots must match assignment slots")
        losses.append(float(-np.mean(np.sum(target * np.log(np.clip(assignment, _EPS, 1.0)), axis=1))))
    return float(np.mean(losses)) if losses else 0.0


def _temporal_centroid_loss(projections: tuple[ObjectStateProjection, ...]) -> float:
    if len(projections) < 2:
        return 0.0
    losses: list[float] = []
    previous = projections[0]
    for current in projections[1:]:
        count = min(len(previous.states), len(current.states))
        if count == 0:
            previous = current
            continue
        previous_centroids = np.vstack([state.centroid for state in previous.states[:count]])
        current_centroids = np.vstack([state.centroid for state in current.states[:count]])
        losses.append(float(np.mean((current_centroids - previous_centroids) ** 2)))
        previous = current
    return float(np.mean(losses)) if losses else 0.0


def _frame_cloud(frame: TrainableKernelFrame) -> GaussianCloud:
    xyz = _array2d(frame.positions, "frame.positions", columns=3)
    vertices = np.zeros(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ],
    )
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    return GaussianCloud(vertices=vertices, source_format="trainable_frame")


def _array2d(value: np.ndarray, label: str, *, columns: int | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{label} must have {columns} columns")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array
