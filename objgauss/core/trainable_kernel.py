from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.features import colors, extract_features, positions
from objgauss.core.object_state import ObjectStateProjection, project_object_states

TRAINABLE_KERNEL_MVP_SCHEMA = "objgauss-v1-trainable-kernel-mvp-v1"
TRAINABLE_IMAGE_TARGET_CONTRACT_SCHEMA = "objgauss-train-image-target-contract-v1"
TRAINABLE_IMAGE_TARGET_SCHEMA = "objgauss-train-image-target-v1"
_EPS = 1e-8


@dataclass(frozen=True)
class TrainableKernelCamera:
    width: int
    height: int
    intrinsics: np.ndarray
    camera_to_world: np.ndarray
    projection_model: str = "orthographic_xy_debug_v1"
    convention: str = "pixel_centers_y_down_camera_to_world"

    def as_dict(self) -> dict[str, Any]:
        return {
            "width": int(self.width),
            "height": int(self.height),
            "projection_model": self.projection_model,
            "convention": self.convention,
            "intrinsics": np.round(self.intrinsics, 6).tolist(),
            "camera_to_world": np.round(self.camera_to_world, 6).tolist(),
        }


@dataclass(frozen=True)
class TrainableKernelImageTarget:
    image: np.ndarray
    camera: TrainableKernelCamera
    visibility_mask: np.ndarray
    point_radius: int = 1
    visibility_policy: str = "covered_pixels"
    color_space: str = "linear_rgb"
    source: str = "synthetic_point_splat_debug"

    def as_dict(self) -> dict[str, Any]:
        image = _image_array(self.image, "image_target.image")
        mask = _visibility_mask_array(
            self.visibility_mask,
            "image_target.visibility_mask",
            height=image.shape[0],
            width=image.shape[1],
        )
        return {
            "schema": TRAINABLE_IMAGE_TARGET_SCHEMA,
            "kind": "image_space_rgb",
            "shape": [int(image.shape[0]), int(image.shape[1]), int(image.shape[2])],
            "dtype": "float32",
            "color_space": self.color_space,
            "visibility_policy": self.visibility_policy,
            "point_radius": int(self.point_radius),
            "visibility_coverage": float(np.mean(mask)),
            "image_sha256": _array_sha256(image),
            "mean_rgb": np.round(np.mean(image, axis=(0, 1)), 6).tolist(),
            "camera": self.camera.as_dict(),
            "source": self.source,
        }


@dataclass(frozen=True)
class TrainableKernelFrame:
    positions: np.ndarray
    features: np.ndarray
    target_rgb: np.ndarray
    target_assignment: np.ndarray | None = None
    image_target: TrainableKernelImageTarget | None = None


@dataclass(frozen=True)
class TrainableKernelLoss:
    iteration: int
    total_loss: float
    render_loss: float
    object_loss: float
    temporal_loss: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "total_loss": float(self.total_loss),
            "render_loss": float(self.render_loss),
            "object_loss": float(self.object_loss),
            "temporal_loss": float(self.temporal_loss),
        }


@dataclass(frozen=True)
class TrainableKernelResult:
    schema: str
    slots: int
    frame_count: int
    iterations: int
    learning_rate: float
    render_weight: float
    object_weight: float
    temporal_weight: float
    initial_loss: TrainableKernelLoss
    final_loss: TrainableKernelLoss
    history: tuple[TrainableKernelLoss, ...]
    assignments: tuple[np.ndarray, ...]
    object_state_projections: tuple[ObjectStateProjection, ...]
    rendered_rgb: tuple[np.ndarray, ...]
    decoder_colors: np.ndarray
    image_targets: tuple[TrainableKernelImageTarget | None, ...]

    def as_dict(self) -> dict[str, Any]:
        image_contract = image_target_contract_summary(self.image_targets)
        return {
            "schema": self.schema,
            "slots": int(self.slots),
            "frame_count": int(self.frame_count),
            "iterations": int(self.iterations),
            "learning_rate": float(self.learning_rate),
            "weights": {
                "render": float(self.render_weight),
                "object": float(self.object_weight),
                "temporal": float(self.temporal_weight),
            },
            "initial_loss": self.initial_loss.as_dict(),
            "final_loss": self.final_loss.as_dict(),
            "loss_decreased": bool(self.final_loss.total_loss < self.initial_loss.total_loss),
            "render_loss_decreased": bool(self.final_loss.render_loss < self.initial_loss.render_loss),
            "render_target_mode": (
                "image_space_targets_bound"
                if image_contract["status"] == "image_targets_bound"
                else "point_rgb_rows"
            ),
            "image_target_contract": image_contract,
            "decoder_colors": np.round(self.decoder_colors, 6).tolist(),
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
class TrainableKernelSample:
    frames: tuple[TrainableKernelFrame, ...]
    slots: int
    source_count: int
    sampled_count: int
    target_source: str
    object_id_mapping: dict[int, int]

    def as_dict(self) -> dict[str, Any]:
        return {
            "slots": int(self.slots),
            "frame_count": len(self.frames),
            "source_count": int(self.source_count),
            "sampled_count": int(self.sampled_count),
            "target_source": self.target_source,
            "image_target_contract": image_target_contract_summary(
                tuple(frame.image_target for frame in self.frames)
            ),
            "object_id_mapping": {
                str(object_id): int(slot)
                for object_id, slot in self.object_id_mapping.items()
            },
        }


@dataclass(frozen=True)
class _ValidatedFrame:
    positions: np.ndarray
    features: np.ndarray
    target_rgb: np.ndarray
    object_targets: np.ndarray
    image_target: TrainableKernelImageTarget | None
    cloud: GaussianCloud


@dataclass(frozen=True)
class _ForwardPass:
    loss: TrainableKernelLoss
    assignments: tuple[np.ndarray, ...]
    projections: tuple[ObjectStateProjection, ...]
    rendered_rgb: tuple[np.ndarray, ...]
    decoder_colors: np.ndarray


def train_kernel_mvp(
    frames: Sequence[TrainableKernelFrame],
    *,
    slots: int,
    iterations: int = 40,
    learning_rate: float = 0.35,
    render_weight: float = 1.0,
    object_weight: float = 1.0,
    temporal_weight: float = 0.02,
    finite_difference_epsilon: float = 1e-3,
    seed: int = 0,
    record_every: int | None = None,
) -> TrainableKernelResult:
    """Train the smallest ObjGauss v1 kernel loop without external ML deps.

    This is an explicit MVP: point observations are treated as the render target,
    and optimization uses finite-difference gradients so the whole
    `perceive -> A -> ObjectState -> Gaussian decode -> render -> loss` path is
    exercised without pulling in torch or a full splat rasterizer.
    """

    if slots < 1:
        raise ValueError("slots must be >= 1")
    validated = _validate_frames(frames, slots=slots)
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if finite_difference_epsilon <= 0:
        raise ValueError("finite_difference_epsilon must be > 0")
    for name, weight in {
        "render_weight": render_weight,
        "object_weight": object_weight,
        "temporal_weight": temporal_weight,
    }.items():
        if weight < 0:
            raise ValueError(f"{name} must be >= 0")

    rng = np.random.default_rng(seed)
    logits = [
        rng.normal(0.0, 0.025, size=(frame.positions.shape[0], slots)).astype(np.float32)
        for frame in validated
    ]
    colors = rng.uniform(0.15, 0.85, size=(slots, 3)).astype(np.float32)
    params = _pack_parameters(logits, colors)
    color_offset = _logit_parameter_size(validated, slots)
    record_stride = iterations if record_every is None else max(1, int(record_every))
    history: list[TrainableKernelLoss] = []

    initial = _forward(
        params,
        validated,
        slots,
        iteration=0,
        render_weight=render_weight,
        object_weight=object_weight,
        temporal_weight=temporal_weight,
    )
    history.append(initial.loss)

    for iteration in range(1, iterations + 1):
        gradient = _finite_difference_gradient(
            params,
            validated,
            slots,
            epsilon=finite_difference_epsilon,
            render_weight=render_weight,
            object_weight=object_weight,
            temporal_weight=temporal_weight,
        )
        params = params - learning_rate * gradient
        params[color_offset:] = np.clip(params[color_offset:], 0.0, 1.0)
        if iteration == iterations or iteration % record_stride == 0:
            history.append(
                _forward(
                    params,
                    validated,
                    slots,
                    iteration=iteration,
                    render_weight=render_weight,
                    object_weight=object_weight,
                    temporal_weight=temporal_weight,
                ).loss
            )

    final = _forward(
        params,
        validated,
        slots,
        iteration=iterations,
        render_weight=render_weight,
        object_weight=object_weight,
        temporal_weight=temporal_weight,
    )
    if history[-1].iteration != iterations:
        history.append(final.loss)
    else:
        history[-1] = final.loss

    return TrainableKernelResult(
        schema=TRAINABLE_KERNEL_MVP_SCHEMA,
        slots=int(slots),
        frame_count=len(validated),
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        render_weight=float(render_weight),
        object_weight=float(object_weight),
        temporal_weight=float(temporal_weight),
        initial_loss=initial.loss,
        final_loss=final.loss,
        history=tuple(history),
        assignments=final.assignments,
        object_state_projections=final.projections,
        rendered_rgb=final.rendered_rgb,
        decoder_colors=final.decoder_colors,
        image_targets=tuple(frame.image_target for frame in validated),
    )


def train_kernel_mvp_from_cloud(
    cloud: GaussianCloud,
    *,
    slots: int | None = None,
    frame_count: int = 2,
    max_points: int | None = 24,
    object_id_field: str = "object_id",
    temporal_offset: float = 0.01,
    bind_image_targets: bool = False,
    image_width: int = 16,
    image_height: int = 16,
    point_radius: int = 1,
    visibility_policy: str = "covered_pixels",
    seed: int = 0,
    iterations: int = 40,
    learning_rate: float = 0.35,
    render_weight: float = 1.0,
    object_weight: float = 1.0,
    temporal_weight: float = 0.02,
    finite_difference_epsilon: float = 1e-3,
    record_every: int | None = None,
) -> tuple[TrainableKernelResult, TrainableKernelSample]:
    sample = trainable_kernel_sample_from_cloud(
        cloud,
        slots=slots,
        frame_count=frame_count,
        max_points=max_points,
        object_id_field=object_id_field,
        temporal_offset=temporal_offset,
        bind_image_targets=bind_image_targets,
        image_width=image_width,
        image_height=image_height,
        point_radius=point_radius,
        visibility_policy=visibility_policy,
        seed=seed,
    )
    result = train_kernel_mvp(
        sample.frames,
        slots=sample.slots,
        iterations=iterations,
        learning_rate=learning_rate,
        render_weight=render_weight,
        object_weight=object_weight,
        temporal_weight=temporal_weight,
        finite_difference_epsilon=finite_difference_epsilon,
        seed=seed,
        record_every=record_every,
    )
    return result, sample


def trainable_kernel_sample_from_cloud(
    cloud: GaussianCloud,
    *,
    slots: int | None = None,
    frame_count: int = 2,
    max_points: int | None = 24,
    object_id_field: str = "object_id",
    temporal_offset: float = 0.01,
    bind_image_targets: bool = False,
    image_width: int = 16,
    image_height: int = 16,
    point_radius: int = 1,
    visibility_policy: str = "covered_pixels",
    seed: int = 0,
) -> TrainableKernelSample:
    if cloud.count == 0:
        raise ValueError("cloud must contain at least one Gaussian")
    if frame_count < 1:
        raise ValueError("frame_count must be >= 1")
    if max_points is not None and max_points < 1:
        raise ValueError("max_points must be >= 1")
    if temporal_offset < 0:
        raise ValueError("temporal_offset must be >= 0")
    if image_width < 1:
        raise ValueError("image_width must be >= 1")
    if image_height < 1:
        raise ValueError("image_height must be >= 1")
    if point_radius < 0:
        raise ValueError("point_radius must be >= 0")

    object_ids = _object_ids_or_none(cloud, object_id_field)
    resolved_slots, object_mapping, target_source = _resolve_sample_slots(
        object_ids,
        slots=slots,
        object_id_field=object_id_field,
    )
    selected = _sample_indices(
        cloud.count,
        object_ids=object_ids,
        max_points=max_points,
        seed=seed,
    )
    xyz = positions(cloud)[selected]
    feature_matrix = extract_features(cloud)[selected]
    target_rgb = colors(cloud)[selected]
    target_assignment = (
        _one_hot_object_targets(object_ids[selected], object_mapping, resolved_slots)
        if object_ids is not None and object_mapping
        else None
    )
    frames = tuple(
        TrainableKernelFrame(
            positions=_temporal_positions(
                xyz,
                frame_index=frame_index,
                frame_count=frame_count,
                temporal_offset=temporal_offset,
                target_assignment=target_assignment,
            ),
            features=feature_matrix,
            target_rgb=target_rgb,
            target_assignment=target_assignment,
        )
        for frame_index in range(frame_count)
    )
    if bind_image_targets:
        frames = bind_image_targets_to_frames(
            frames,
            width=image_width,
            height=image_height,
            point_radius=point_radius,
            visibility_policy=visibility_policy,
            source="sampled_gaussian_point_splat_debug",
        )
    return TrainableKernelSample(
        frames=frames,
        slots=resolved_slots,
        source_count=cloud.count,
        sampled_count=len(selected),
        target_source=target_source,
        object_id_mapping=object_mapping,
    )


def make_trainable_kernel_mvp_fixture() -> tuple[TrainableKernelFrame, ...]:
    positions_0 = np.array(
        [
            [-1.00, 0.00, 0.00],
            [-0.86, 0.10, 0.00],
            [-1.12, -0.08, 0.00],
            [0.92, 0.02, 0.00],
            [1.08, 0.12, 0.00],
            [1.16, -0.08, 0.00],
        ],
        dtype=np.float32,
    )
    positions_1 = positions_0 + np.array(
        [
            [0.05, 0.02, 0.00],
            [0.05, 0.01, 0.00],
            [0.06, 0.02, 0.00],
            [-0.04, 0.01, 0.00],
            [-0.04, 0.00, 0.00],
            [-0.03, 0.01, 0.00],
        ],
        dtype=np.float32,
    )
    rgb_0 = np.array([0.92, 0.22, 0.16], dtype=np.float32)
    rgb_1 = np.array([0.10, 0.70, 0.86], dtype=np.float32)
    target = np.vstack([np.tile(rgb_0, (3, 1)), np.tile(rgb_1, (3, 1))]).astype(np.float32)
    features_0 = np.column_stack([positions_0[:, :2], target]).astype(np.float32)
    features_1 = np.column_stack([positions_1[:, :2], target]).astype(np.float32)
    return (
        TrainableKernelFrame(positions=positions_0, features=features_0, target_rgb=target),
        TrainableKernelFrame(positions=positions_1, features=features_1, target_rgb=target),
    )


def bind_image_targets_to_frames(
    frames: Sequence[TrainableKernelFrame],
    *,
    width: int = 16,
    height: int = 16,
    point_radius: int = 1,
    visibility_policy: str = "covered_pixels",
    source: str = "synthetic_point_splat_debug",
) -> tuple[TrainableKernelFrame, ...]:
    if width < 1:
        raise ValueError("width must be >= 1")
    if height < 1:
        raise ValueError("height must be >= 1")
    if point_radius < 0:
        raise ValueError("point_radius must be >= 0")
    return tuple(
        TrainableKernelFrame(
            positions=frame.positions,
            features=frame.features,
            target_rgb=frame.target_rgb,
            target_assignment=frame.target_assignment,
            image_target=make_trainable_image_target(
                frame,
                width=width,
                height=height,
                point_radius=point_radius,
                visibility_policy=visibility_policy,
                source=source,
            ),
        )
        for frame in frames
    )


def make_trainable_image_target(
    frame: TrainableKernelFrame,
    *,
    width: int = 16,
    height: int = 16,
    point_radius: int = 1,
    visibility_policy: str = "covered_pixels",
    source: str = "synthetic_point_splat_debug",
) -> TrainableKernelImageTarget:
    if width < 1:
        raise ValueError("width must be >= 1")
    if height < 1:
        raise ValueError("height must be >= 1")
    if point_radius < 0:
        raise ValueError("point_radius must be >= 0")
    positions_2d = _array2d(frame.positions, "frame.positions", columns=3)
    target_rgb = _array2d(frame.target_rgb, "frame.target_rgb", columns=3)
    if positions_2d.shape[0] != target_rgb.shape[0]:
        raise ValueError("frame.target_rgb rows must match positions")

    pixels, intrinsics = _orthographic_pixels(positions_2d, width=width, height=height)
    image = np.zeros((height, width, 3), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    for pixel, color in zip(pixels, np.clip(target_rgb, 0.0, 1.0), strict=True):
        px, py = int(pixel[0]), int(pixel[1])
        for y in range(max(0, py - point_radius), min(height, py + point_radius + 1)):
            for x in range(max(0, px - point_radius), min(width, px + point_radius + 1)):
                image[y, x] += color
                counts[y, x] += 1.0
    covered = counts > 0
    image[covered] = image[covered] / counts[covered, None]
    if visibility_policy == "covered_pixels":
        visibility_mask = covered
    elif visibility_policy == "all_pixels":
        visibility_mask = np.ones((height, width), dtype=bool)
    else:
        raise ValueError("visibility_policy must be 'covered_pixels' or 'all_pixels'")
    camera = TrainableKernelCamera(
        width=width,
        height=height,
        intrinsics=intrinsics,
        camera_to_world=np.eye(4, dtype=np.float32),
    )
    target = TrainableKernelImageTarget(
        image=image,
        camera=camera,
        visibility_mask=visibility_mask,
        point_radius=point_radius,
        visibility_policy=visibility_policy,
        source=source,
    )
    validate_trainable_image_target(target)
    return target


def image_target_contract_summary(
    image_targets: Sequence[TrainableKernelImageTarget | None],
) -> dict[str, Any]:
    targets = [
        target.as_dict() if target is not None else None
        for target in image_targets
    ]
    bound = [target for target in targets if target is not None]
    blockers: list[str] = []
    if not targets:
        blockers.append("no_frames")
    elif len(bound) != len(targets):
        blockers.append("missing_image_target_for_some_frames")
    status = "image_targets_bound" if targets and not blockers else "point_targets_only"
    visibility_policies = sorted(
        {str(target["visibility_policy"]) for target in bound}
    )
    return {
        "schema": TRAINABLE_IMAGE_TARGET_CONTRACT_SCHEMA,
        "status": status,
        "frame_count": len(targets),
        "target_kind": "image_space_rgb",
        "targets_bound": len(bound),
        "targets": targets,
        "camera_contract": {
            "required": ["width", "height", "intrinsics", "camera_to_world", "projection_model"],
            "projection_model": "orthographic_xy_debug_v1",
            "extrinsics_convention": "camera_to_world_4x4",
        },
        "visibility_policies": visibility_policies,
        "loss_telemetry_contract": {
            "required_terms": ["image_render_loss", "visibility_policy", "renderer_name"],
            "still_required_for_training": ["renderer_gradient_path"],
        },
        "blockers": blockers,
    }


def validate_trainable_image_target(target: TrainableKernelImageTarget) -> bool:
    image = _image_array(target.image, "image_target.image")
    mask = _visibility_mask_array(
        target.visibility_mask,
        "image_target.visibility_mask",
        height=image.shape[0],
        width=image.shape[1],
    )
    camera = target.camera
    if camera.width != image.shape[1] or camera.height != image.shape[0]:
        raise ValueError("image_target camera width/height must match image shape")
    intrinsics = np.asarray(camera.intrinsics, dtype=np.float32)
    camera_to_world = np.asarray(camera.camera_to_world, dtype=np.float32)
    if intrinsics.shape != (3, 3):
        raise ValueError("image_target camera intrinsics must have shape 3x3")
    if camera_to_world.shape != (4, 4):
        raise ValueError("image_target camera_to_world must have shape 4x4")
    if not np.isfinite(intrinsics).all() or not np.isfinite(camera_to_world).all():
        raise ValueError("image_target camera matrices must be finite")
    if target.visibility_policy not in {"covered_pixels", "all_pixels"}:
        raise ValueError("image_target visibility_policy is unsupported")
    if target.point_radius < 0:
        raise ValueError("image_target point_radius must be >= 0")
    if not mask.any():
        raise ValueError("image_target visibility_mask must supervise at least one pixel")
    return True


def validate_image_target_contract_summary(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        raise TypeError("image target contract summary must be a dict")
    if payload.get("schema") != TRAINABLE_IMAGE_TARGET_CONTRACT_SCHEMA:
        raise ValueError(f"unsupported image target contract schema: {payload.get('schema')}")
    required = (
        "status",
        "frame_count",
        "target_kind",
        "targets_bound",
        "camera_contract",
        "visibility_policies",
        "loss_telemetry_contract",
        "blockers",
    )
    missing = [key for key in required if key not in payload]
    if missing:
        raise ValueError(f"image target contract missing keys: {', '.join(missing)}")
    return True


def _object_ids_or_none(cloud: GaussianCloud, field: str) -> np.ndarray | None:
    if field not in cloud.fields:
        return None
    values = np.asarray(cloud.vertices[field], dtype=np.int32)
    if values.shape[0] != cloud.count:
        raise ValueError(f"{field} has {values.shape[0]} values for {cloud.count} gaussians")
    return values


def _resolve_sample_slots(
    object_ids: np.ndarray | None,
    *,
    slots: int | None,
    object_id_field: str,
) -> tuple[int, dict[int, int], str]:
    if slots is not None and slots < 1:
        raise ValueError("slots must be >= 1")
    if object_ids is None:
        if slots is None:
            raise ValueError("slots is required when the cloud has no object_id field")
        return int(slots), {}, "feature_quantile_pseudo_targets"

    unique_ids = tuple(int(value) for value in np.unique(object_ids))
    if not unique_ids:
        raise ValueError(f"{object_id_field} field did not contain any object ids")
    resolved_slots = int(slots) if slots is not None else len(unique_ids)
    if len(unique_ids) > resolved_slots:
        raise ValueError(
            f"{object_id_field} has {len(unique_ids)} unique ids but slots={resolved_slots}"
        )
    mapping = {object_id: index for index, object_id in enumerate(unique_ids)}
    return resolved_slots, mapping, f"{object_id_field}_one_hot_targets"


def _sample_indices(
    count: int,
    *,
    object_ids: np.ndarray | None,
    max_points: int | None,
    seed: int,
) -> np.ndarray:
    if max_points is None or count <= max_points:
        return np.arange(count, dtype=np.int64)
    if object_ids is None:
        return _even_sample(np.arange(count, dtype=np.int64), max_points)

    selected: list[int] = []
    unique_ids = np.unique(object_ids)
    rng = np.random.default_rng(seed)
    quotas = _balanced_object_quotas(object_ids, max_points=max_points)
    for object_id in unique_ids:
        object_indices = np.flatnonzero(object_ids == object_id).astype(np.int64, copy=False)
        quota = min(int(quotas[int(object_id)]), object_indices.shape[0])
        if quota <= 0:
            continue
        if object_indices.shape[0] <= quota:
            chosen = object_indices
        else:
            shuffled = object_indices.copy()
            rng.shuffle(shuffled)
            chosen = np.sort(shuffled[:quota])
        selected.extend(int(index) for index in chosen)
    if len(selected) < max_points:
        missing = max_points - len(selected)
        remaining = np.setdiff1d(np.arange(count, dtype=np.int64), np.asarray(selected, dtype=np.int64))
        selected.extend(int(index) for index in _even_sample(remaining, missing))
    return np.asarray(sorted(selected[:max_points]), dtype=np.int64)


def _balanced_object_quotas(object_ids: np.ndarray, *, max_points: int) -> dict[int, int]:
    unique_ids, counts = np.unique(object_ids, return_counts=True)
    quotas: dict[int, int] = {}
    remaining = max_points
    total = int(np.sum(counts))
    for object_id, count in zip(unique_ids, counts, strict=True):
        proportional = int(np.floor((int(count) / max(total, 1)) * max_points))
        quota = max(1, proportional)
        quotas[int(object_id)] = min(quota, int(count))
        remaining -= quotas[int(object_id)]
    cursor = 0
    while remaining > 0 and unique_ids.size:
        object_id = int(unique_ids[cursor % unique_ids.size])
        if quotas[object_id] < int(counts[cursor % counts.size]):
            quotas[object_id] += 1
            remaining -= 1
        cursor += 1
        if cursor > unique_ids.size * max_points:
            break
    return quotas


def _even_sample(indices: np.ndarray, count: int) -> np.ndarray:
    if count <= 0 or indices.size == 0:
        return np.empty(0, dtype=np.int64)
    if indices.size <= count:
        return indices.astype(np.int64, copy=False)
    positions = np.linspace(0, indices.size - 1, count, dtype=np.int64)
    return indices[positions].astype(np.int64, copy=False)


def _one_hot_object_targets(
    object_ids: np.ndarray,
    mapping: dict[int, int],
    slots: int,
) -> np.ndarray:
    targets = np.zeros((object_ids.shape[0], slots), dtype=np.float32)
    for row, object_id in enumerate(object_ids):
        targets[row, mapping[int(object_id)]] = 1.0
    return targets


def _temporal_positions(
    xyz: np.ndarray,
    *,
    frame_index: int,
    frame_count: int,
    temporal_offset: float,
    target_assignment: np.ndarray | None,
) -> np.ndarray:
    if temporal_offset <= 0 or frame_count <= 1:
        return xyz.copy()
    centered_index = frame_index - (frame_count - 1) / 2.0
    if target_assignment is None:
        direction = np.ones((xyz.shape[0], 1), dtype=np.float32)
    else:
        slot = np.argmax(target_assignment, axis=1).astype(np.float32, copy=False)
        direction = (slot[:, None] * 2.0 - 1.0).astype(np.float32, copy=False)
    offset = np.column_stack(
        [
            direction[:, 0] * temporal_offset * centered_index,
            np.zeros(xyz.shape[0], dtype=np.float32),
            np.zeros(xyz.shape[0], dtype=np.float32),
        ]
    )
    return (xyz + offset).astype(np.float32, copy=False)


def _validate_frames(frames: Sequence[TrainableKernelFrame], *, slots: int) -> tuple[_ValidatedFrame, ...]:
    if not frames:
        raise ValueError("at least one frame is required")
    validated: list[_ValidatedFrame] = []
    feature_dim: int | None = None
    for index, frame in enumerate(frames):
        positions = _array2d(frame.positions, f"frames[{index}].positions", columns=3)
        features = _array2d(frame.features, f"frames[{index}].features")
        target_rgb = _array2d(frame.target_rgb, f"frames[{index}].target_rgb", columns=3)
        if positions.shape[0] == 0:
            raise ValueError(f"frames[{index}] must contain at least one evidence record")
        if features.shape[0] != positions.shape[0]:
            raise ValueError(f"frames[{index}].features rows must match positions")
        if target_rgb.shape[0] != positions.shape[0]:
            raise ValueError(f"frames[{index}].target_rgb rows must match positions")
        if feature_dim is None:
            feature_dim = int(features.shape[1])
        elif features.shape[1] != feature_dim:
            raise ValueError("all frames must use the same feature dimension")
        object_targets = _assignment_targets(
            frame,
            features=features,
            index=index,
            evidence_count=positions.shape[0],
            slots=slots,
        )
        validated.append(
            _ValidatedFrame(
                positions=positions,
                features=features,
                target_rgb=np.clip(target_rgb, 0.0, 1.0),
                object_targets=object_targets,
                image_target=_validated_image_target(frame.image_target, index=index),
                cloud=_cloud_from_positions(positions),
            )
        )
    return tuple(validated)


def _validated_image_target(
    target: TrainableKernelImageTarget | None,
    *,
    index: int,
) -> TrainableKernelImageTarget | None:
    if target is None:
        return None
    try:
        validate_trainable_image_target(target)
    except ValueError as exc:
        raise ValueError(f"frames[{index}].image_target invalid: {exc}") from exc
    return target


def _assignment_targets(
    frame: TrainableKernelFrame,
    *,
    features: np.ndarray,
    index: int,
    evidence_count: int,
    slots: int,
) -> np.ndarray:
    if frame.target_assignment is not None:
        targets = _array2d(frame.target_assignment, f"frames[{index}].target_assignment", columns=slots)
        if targets.shape[0] != evidence_count:
            raise ValueError(f"frames[{index}].target_assignment rows must match positions")
        row_sums = targets.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=1e-5, rtol=0.0):
            raise ValueError(f"frames[{index}].target_assignment rows must sum to 1")
        if np.any(targets < 0.0):
            raise ValueError(f"frames[{index}].target_assignment must be non-negative")
        return targets
    return _feature_quantile_targets(features, slots=slots)


def _feature_quantile_targets(features: np.ndarray, *, slots: int) -> np.ndarray:
    evidence_count = int(features.shape[0])
    targets = np.zeros((evidence_count, slots), dtype=np.float32)
    order = np.argsort(features[:, 0], kind="stable")
    for rank, row in enumerate(order):
        slot = min(slots - 1, int(rank * slots / max(evidence_count, 1)))
        targets[int(row), slot] = 1.0
    return targets


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


def _image_array(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"{label} must have shape HxWx3")
    if array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError(f"{label} must have positive width and height")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{label} values must be in [0, 1]")
    return array


def _visibility_mask_array(
    value: np.ndarray,
    label: str,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    array = np.asarray(value, dtype=bool)
    if array.shape != (height, width):
        raise ValueError(f"{label} must have shape {height}x{width}")
    return array


def _orthographic_pixels(
    positions_xyz: np.ndarray,
    *,
    width: int,
    height: int,
) -> tuple[np.ndarray, np.ndarray]:
    xy = np.asarray(positions_xyz[:, :2], dtype=np.float32)
    min_xy = np.min(xy, axis=0)
    max_xy = np.max(xy, axis=0)
    span = np.maximum(max_xy - min_xy, np.array([_EPS, _EPS], dtype=np.float32))
    scale_x = 0.0 if width == 1 else (width - 1) / float(span[0])
    scale_y = 0.0 if height == 1 else (height - 1) / float(span[1])
    px = np.zeros(xy.shape[0], dtype=np.float32) if width == 1 else (xy[:, 0] - min_xy[0]) * scale_x
    py = np.zeros(xy.shape[0], dtype=np.float32) if height == 1 else (max_xy[1] - xy[:, 1]) * scale_y
    pixels = np.column_stack(
        [
            np.clip(np.rint(px), 0, width - 1).astype(np.int32),
            np.clip(np.rint(py), 0, height - 1).astype(np.int32),
        ]
    )
    intrinsics = np.array(
        [
            [scale_x, 0.0, -float(min_xy[0]) * scale_x],
            [0.0, -scale_y, float(max_xy[1]) * scale_y],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return pixels, intrinsics


def _array_sha256(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array.astype(np.float32, copy=False))
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _cloud_from_positions(positions: np.ndarray) -> GaussianCloud:
    vertices = np.zeros(
        positions.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ],
    )
    vertices["x"] = positions[:, 0]
    vertices["y"] = positions[:, 1]
    vertices["z"] = positions[:, 2]
    return GaussianCloud(vertices=vertices, source_format="trainable-kernel-mvp")


def _pack_parameters(logits: Sequence[np.ndarray], decoder_colors: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [*(np.asarray(logit, dtype=np.float32).ravel() for logit in logits), decoder_colors.ravel()]
    ).astype(np.float32, copy=False)


def _unpack_parameters(
    params: np.ndarray,
    frames: Sequence[_ValidatedFrame],
    slots: int,
) -> tuple[tuple[np.ndarray, ...], np.ndarray]:
    cursor = 0
    logits: list[np.ndarray] = []
    for frame in frames:
        size = frame.positions.shape[0] * slots
        logits.append(params[cursor : cursor + size].reshape(frame.positions.shape[0], slots))
        cursor += size
    colors = params[cursor : cursor + slots * 3].reshape(slots, 3)
    return tuple(logits), colors


def _logit_parameter_size(frames: Sequence[_ValidatedFrame], slots: int) -> int:
    return int(sum(frame.positions.shape[0] * slots for frame in frames))


def _forward(
    params: np.ndarray,
    frames: Sequence[_ValidatedFrame],
    slots: int,
    *,
    iteration: int,
    render_weight: float,
    object_weight: float,
    temporal_weight: float,
) -> _ForwardPass:
    logits, colors = _unpack_parameters(params, frames, slots)
    assignments = tuple(_softmax(frame_logits, axis=1) for frame_logits in logits)
    rendered = tuple(assignment @ colors for assignment in assignments)
    projections = tuple(
        project_object_states(frame.cloud, assignment, evidence_features=frame.features)
        for frame, assignment in zip(frames, assignments, strict=True)
    )
    render_loss = _render_loss(rendered, frames)
    object_loss = _object_assignment_loss(assignments, frames)
    temporal_loss = _temporal_centroid_loss(projections)
    total_loss = (
        render_weight * render_loss
        + object_weight * object_loss
        + temporal_weight * temporal_loss
    )
    loss = TrainableKernelLoss(
        iteration=int(iteration),
        total_loss=float(total_loss),
        render_loss=float(render_loss),
        object_loss=float(object_loss),
        temporal_loss=float(temporal_loss),
    )
    return _ForwardPass(
        loss=loss,
        assignments=assignments,
        projections=projections,
        rendered_rgb=rendered,
        decoder_colors=colors.copy(),
    )


def _finite_difference_gradient(
    params: np.ndarray,
    frames: Sequence[_ValidatedFrame],
    slots: int,
    *,
    epsilon: float,
    render_weight: float,
    object_weight: float,
    temporal_weight: float,
) -> np.ndarray:
    gradient = np.zeros_like(params, dtype=np.float32)
    for index in range(params.size):
        plus = params.copy()
        minus = params.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_loss = _forward(
            plus,
            frames,
            slots,
            iteration=-1,
            render_weight=render_weight,
            object_weight=object_weight,
            temporal_weight=temporal_weight,
        ).loss.total_loss
        minus_loss = _forward(
            minus,
            frames,
            slots,
            iteration=-1,
            render_weight=render_weight,
            object_weight=object_weight,
            temporal_weight=temporal_weight,
        ).loss.total_loss
        gradient[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
    return gradient


def _render_loss(rendered: Sequence[np.ndarray], frames: Sequence[_ValidatedFrame]) -> float:
    losses = [
        float(np.mean((predicted - frame.target_rgb) ** 2))
        for predicted, frame in zip(rendered, frames, strict=True)
    ]
    return float(np.mean(losses))


def _object_assignment_loss(
    assignments: Sequence[np.ndarray],
    frames: Sequence[_ValidatedFrame],
) -> float:
    losses = []
    for assignment, frame in zip(assignments, frames, strict=True):
        cross_entropy = -np.sum(
            frame.object_targets * np.log(np.clip(assignment, _EPS, 1.0)),
            axis=1,
        )
        entropy = -np.sum(assignment * np.log(np.clip(assignment, _EPS, 1.0)), axis=1)
        normalized_entropy = 0.0 if assignment.shape[1] <= 1 else entropy / np.log(assignment.shape[1])
        mass = assignment.sum(axis=0)
        mass_fraction = mass / max(float(mass.sum()), _EPS)
        target_fraction = np.full(assignment.shape[1], 1.0 / assignment.shape[1], dtype=np.float32)
        balance = float(np.mean((mass_fraction - target_fraction) ** 2))
        losses.append(float(np.mean(cross_entropy)) + 0.05 * float(np.mean(normalized_entropy)) + balance)
    return float(np.mean(losses))


def _temporal_centroid_loss(projections: Sequence[ObjectStateProjection]) -> float:
    if len(projections) < 2:
        return 0.0
    total = 0.0
    count = 0
    for previous, current in zip(projections[:-1], projections[1:], strict=True):
        for previous_state, current_state in zip(previous.states, current.states, strict=True):
            if not np.isfinite(previous_state.centroid).all() or not np.isfinite(current_state.centroid).all():
                continue
            total += float(np.mean((current_state.centroid - previous_state.centroid) ** 2))
            count += 1
    return 0.0 if count == 0 else total / count


def _softmax(values: np.ndarray, axis: int) -> np.ndarray:
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return (exp / np.sum(exp, axis=axis, keepdims=True)).astype(np.float32, copy=False)
