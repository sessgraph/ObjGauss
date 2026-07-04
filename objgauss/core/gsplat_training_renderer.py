from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable, Sequence

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.gaussian_decoder import (
    ObjectStateGaussianDecode,
    decode_gaussian_from_object_state,
)
from objgauss.core.object_state import ObjectStateProjection, project_object_states
from objgauss.core.trainable_kernel import (
    TrainableKernelFrame,
    validate_trainable_image_target,
)
from objgauss.core.training_renderer import (
    TRAINING_RENDERER_API_SCHEMA,
    TrainingRendererFrameLoss,
    TrainingRendererLossResult,
)

GSPLAT_RENDERER = "gsplat-rasterization-v1"
GSPLAT_GRADIENT_PATH = "torch-autograd-gsplat-rasterization-v1"
GSPLAT_AVAILABILITY_SCHEMA = "objgauss-gsplat-training-renderer-availability-v1"
GSPLAT_TRAINING_INPUT_SCHEMA = "objgauss-gsplat-training-input-v1"
GSPLAT_SYNTHETIC_GAUSSIAN_POLICY = "object-state-synthetic-isotropic-gaussian-v1"


@dataclass(frozen=True)
class GsplatRendererAvailability:
    schema: str
    renderer_name: str
    gradient_path: str
    available: bool
    require_cuda: bool
    torch_version: str | None
    gsplat_version: str | None
    cuda_available: bool | None
    blockers: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "renderer_name": self.renderer_name,
            "gradient_path": self.gradient_path,
            "available": bool(self.available),
            "require_cuda": bool(self.require_cuda),
            "torch_version": self.torch_version,
            "gsplat_version": self.gsplat_version,
            "cuda_available": self.cuda_available,
            "blockers": list(self.blockers),
        }


@dataclass(frozen=True)
class GsplatTrainingInput:
    schema: str
    renderer_name: str
    gradient_path: str
    frame_index: int
    means: np.ndarray
    quats: np.ndarray
    scales: np.ndarray
    opacities: np.ndarray
    colors: np.ndarray
    viewmats: np.ndarray
    intrinsics: np.ndarray
    width: int
    height: int
    target_image: np.ndarray
    visibility_mask: np.ndarray
    gaussian_policy: str
    decoder_schema: str
    object_state_slots: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "renderer_name": self.renderer_name,
            "gradient_path": self.gradient_path,
            "frame_index": int(self.frame_index),
            "gaussian_count": int(self.means.shape[0]),
            "camera_count": int(self.viewmats.shape[0]),
            "width": int(self.width),
            "height": int(self.height),
            "gaussian_policy": self.gaussian_policy,
            "decoder_schema": self.decoder_schema,
            "object_state_slots": int(self.object_state_slots),
            "shapes": {
                "means": list(self.means.shape),
                "quats": list(self.quats.shape),
                "scales": list(self.scales.shape),
                "opacities": list(self.opacities.shape),
                "colors": list(self.colors.shape),
                "viewmats": list(self.viewmats.shape),
                "Ks": list(self.intrinsics.shape),
                "target_image": list(self.target_image.shape),
                "visibility_mask": list(self.visibility_mask.shape),
            },
            "visibility_coverage": float(np.mean(self.visibility_mask)),
        }


def gsplat_renderer_availability(
    *,
    require_cuda: bool = True,
    _importer: Callable[[str], Any] = import_module,
) -> GsplatRendererAvailability:
    blockers: list[str] = []
    torch_module: Any | None = None
    gsplat_module: Any | None = None
    torch_version: str | None = None
    gsplat_version: str | None = None
    cuda_available: bool | None = None

    try:
        torch_module = _importer("torch")
        torch_version = str(getattr(torch_module, "__version__", "unknown"))
    except ImportError:
        blockers.append("optional_dependency_missing:torch")

    if torch_module is not None:
        cuda = getattr(torch_module, "cuda", None)
        cuda_is_available = getattr(cuda, "is_available", None)
        cuda_available = bool(cuda_is_available()) if callable(cuda_is_available) else False
        if require_cuda and not cuda_available:
            blockers.append("cuda_not_available")

    try:
        gsplat_module = _importer("gsplat")
        gsplat_version = str(getattr(gsplat_module, "__version__", "unknown"))
    except ImportError:
        blockers.append("optional_dependency_missing:gsplat")

    if gsplat_module is not None:
        try:
            _resolve_gsplat_rasterization(_importer)
        except (ImportError, AttributeError):
            blockers.append("gsplat_rasterization_api_missing")

    return GsplatRendererAvailability(
        schema=GSPLAT_AVAILABILITY_SCHEMA,
        renderer_name=GSPLAT_RENDERER,
        gradient_path=GSPLAT_GRADIENT_PATH,
        available=not blockers,
        require_cuda=bool(require_cuda),
        torch_version=torch_version,
        gsplat_version=gsplat_version,
        cuda_available=cuda_available,
        blockers=tuple(blockers),
    )


def build_gsplat_training_input(
    frame: TrainableKernelFrame,
    assignment: np.ndarray,
    decoder_colors: np.ndarray,
    *,
    decoder_opacity_logits: np.ndarray | None = None,
    frame_index: int = 0,
    default_scale: float = 0.01,
    default_opacity: float = 1.0,
) -> GsplatTrainingInput:
    if frame.image_target is None:
        raise ValueError("gsplat renderer input requires frame.image_target")
    validate_trainable_image_target(frame.image_target)
    if default_scale <= 0:
        raise ValueError("default_scale must be > 0")
    if not 0.0 <= default_opacity <= 1.0:
        raise ValueError("default_opacity must be in [0, 1]")

    colors = _array2d(decoder_colors, "decoder_colors", columns=3)
    assignment = _array2d(assignment, "assignment", columns=colors.shape[0])
    positions = _array2d(frame.positions, "frame.positions", columns=3)
    if assignment.shape[0] != positions.shape[0]:
        raise ValueError("assignment rows must match frame positions")
    if np.any(assignment < 0.0):
        raise ValueError("assignment must be non-negative")
    if not np.allclose(assignment.sum(axis=1), 1.0, atol=1e-5, rtol=0.0):
        raise ValueError("assignment rows must sum to 1")

    projection = _object_state_projection_from_frame(frame, assignment)
    return build_gsplat_training_input_from_object_state(
        frame,
        projection,
        colors,
        decoder_opacity_logits=decoder_opacity_logits,
        frame_index=frame_index,
        default_scale=default_scale,
        default_opacity=default_opacity,
    )


def build_gsplat_training_input_from_object_state(
    frame: TrainableKernelFrame,
    projection: ObjectStateProjection,
    decoder_colors: np.ndarray,
    *,
    decoder_opacity_logits: np.ndarray | None = None,
    frame_index: int = 0,
    default_scale: float = 0.01,
    default_opacity: float = 1.0,
) -> GsplatTrainingInput:
    if frame.image_target is None:
        raise ValueError("gsplat renderer input requires frame.image_target")
    validate_trainable_image_target(frame.image_target)
    decoded = decode_gaussian_from_object_state(
        frame.positions,
        projection,
        decoder_colors,
        object_opacity_logits=decoder_opacity_logits,
        default_scale=default_scale,
        default_opacity=default_opacity,
    )
    return _gsplat_input_from_decode(
        frame,
        decoded,
        frame_index=frame_index,
    )


def _gsplat_input_from_decode(
    frame: TrainableKernelFrame,
    decoded: ObjectStateGaussianDecode,
    *,
    frame_index: int,
) -> GsplatTrainingInput:
    if frame.image_target is None:
        raise ValueError("gsplat renderer input requires frame.image_target")

    target = frame.image_target
    image = np.asarray(target.image, dtype=np.float32)
    mask = np.asarray(target.visibility_mask, dtype=bool)
    camera_to_world = np.asarray(target.camera.camera_to_world, dtype=np.float32)
    viewmat = np.linalg.inv(camera_to_world).astype(np.float32, copy=False)
    intrinsics = np.asarray(target.camera.intrinsics, dtype=np.float32)

    return GsplatTrainingInput(
        schema=GSPLAT_TRAINING_INPUT_SCHEMA,
        renderer_name=GSPLAT_RENDERER,
        gradient_path=GSPLAT_GRADIENT_PATH,
        frame_index=int(frame_index),
        means=decoded.means,
        quats=decoded.quats,
        scales=decoded.scales,
        opacities=decoded.opacities,
        colors=decoded.colors,
        viewmats=viewmat[None, :, :],
        intrinsics=intrinsics[None, :, :],
        width=int(target.camera.width),
        height=int(target.camera.height),
        target_image=image,
        visibility_mask=mask,
        gaussian_policy=decoded.gaussian_policy,
        decoder_schema=decoded.schema,
        object_state_slots=decoded.object_count,
    )


def evaluate_gsplat_training_renderer_loss(
    frames: Sequence[TrainableKernelFrame],
    assignments: Sequence[np.ndarray],
    decoder_colors: np.ndarray,
    *,
    decoder_opacity_logits: np.ndarray | None = None,
    require_cuda: bool = True,
    device: str | None = None,
    default_scale: float = 0.01,
    default_opacity: float = 1.0,
    _importer: Callable[[str], Any] = import_module,
) -> TrainingRendererLossResult:
    if not frames:
        raise ValueError("at least one frame is required")
    if len(frames) != len(assignments):
        raise ValueError("assignments must have one matrix per frame")
    colors = _array2d(decoder_colors, "decoder_colors", columns=3)
    opacity_logits = _optional_array1d(decoder_opacity_logits, "decoder_opacity_logits")
    if opacity_logits is not None and opacity_logits.shape[0] != colors.shape[0]:
        raise ValueError("decoder_opacity_logits length must match decoder_colors rows")
    inputs = tuple(
        build_gsplat_training_input(
            frame,
            assignment,
            colors,
            decoder_opacity_logits=opacity_logits,
            frame_index=frame_index,
            default_scale=default_scale,
            default_opacity=default_opacity,
        )
        for frame_index, (frame, assignment) in enumerate(zip(frames, assignments, strict=True))
    )
    availability = gsplat_renderer_availability(
        require_cuda=require_cuda,
        _importer=_importer,
    )
    if not availability.available:
        raise RuntimeError(
            "gsplat training renderer unavailable: "
            + ", ".join(availability.blockers)
        )

    torch = _importer("torch")
    rasterization = _resolve_gsplat_rasterization(_importer)
    resolved_device = device or ("cuda" if require_cuda else "cpu")
    color_params = torch.tensor(colors, dtype=torch.float32, device=resolved_device, requires_grad=True)
    opacity_params = (
        None
        if opacity_logits is None
        else torch.tensor(
            opacity_logits,
            dtype=torch.float32,
            device=resolved_device,
            requires_grad=True,
        )
    )
    assignment_tensors = [
        torch.tensor(
            np.asarray(assignment, dtype=np.float32),
            dtype=torch.float32,
            device=resolved_device,
            requires_grad=True,
        )
        for assignment in assignments
    ]

    rendered_images: list[np.ndarray] = []
    frame_losses: list[TrainingRendererFrameLoss] = []
    losses = []

    for input_record, assignment_tensor in zip(inputs, assignment_tensors, strict=True):
        means = _torch_tensor(torch, input_record.means, resolved_device)
        quats = _torch_tensor(torch, input_record.quats, resolved_device)
        scales = _torch_tensor(torch, input_record.scales, resolved_device)
        if opacity_params is None:
            opacities = _torch_tensor(torch, input_record.opacities, resolved_device)
        else:
            opacity_scales = _torch_object_opacity_scales(torch, opacity_params)
            opacities = torch.clamp(
                float(default_opacity) * (assignment_tensor @ opacity_scales),
                0.0,
                1.0,
            )
        viewmats = _torch_tensor(torch, input_record.viewmats, resolved_device)
        intrinsics = _torch_tensor(torch, input_record.intrinsics, resolved_device)
        target_image = _torch_tensor(torch, input_record.target_image, resolved_device)
        visibility_mask = torch.tensor(input_record.visibility_mask, dtype=torch.bool, device=resolved_device)

        point_colors = torch.clamp(assignment_tensor @ color_params, 0.0, 1.0)
        render_output = _call_gsplat_rasterization(
            rasterization,
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=point_colors,
            viewmats=viewmats,
            intrinsics=intrinsics,
            width=input_record.width,
            height=input_record.height,
        )
        rendered = _rendered_rgb_tensor(render_output)
        diff = rendered - target_image
        supervised = diff[visibility_mask]
        if supervised.numel() == 0:
            raise ValueError(f"frames[{input_record.frame_index}].image_target has no supervised pixels")
        frame_loss = torch.mean(supervised ** 2)
        losses.append(frame_loss)

        rendered_np = rendered.detach().cpu().numpy().astype(np.float32, copy=False)
        rendered_images.append(rendered_np)
        abs_error = np.abs((rendered_np - input_record.target_image)[input_record.visibility_mask])
        frame_losses.append(
            TrainingRendererFrameLoss(
                frame_index=input_record.frame_index,
                image_render_loss=float(frame_loss.detach().cpu().item()),
                supervised_pixels=int(np.count_nonzero(input_record.visibility_mask)),
                visibility_policy=str(frames[input_record.frame_index].image_target.visibility_policy),
                rendered_shape=tuple(int(value) for value in rendered_np.shape),
                point_count=int(input_record.means.shape[0]),
                max_abs_error=float(np.max(abs_error)),
                mean_abs_error=float(np.mean(abs_error)),
            )
        )

    total_loss = sum(losses) / len(losses)
    total_loss.backward()
    decoder_gradient = color_params.grad.detach().cpu().numpy().astype(np.float32, copy=False)
    opacity_gradient = (
        np.zeros((0,), dtype=np.float32)
        if opacity_params is None
        else opacity_params.grad.detach().cpu().numpy().astype(np.float32, copy=False)
    )
    assignment_gradients = tuple(
        tensor.grad.detach().cpu().numpy().astype(np.float32, copy=False)
        for tensor in assignment_tensors
    )
    differentiable_fields = ["decoder_colors", "assignments"]
    frozen_fields = ["means", "quats", "scales", "opacities", "camera"]
    if opacity_params is not None:
        differentiable_fields.append("decoder_opacity_logits")
        frozen_fields = ["means", "quats", "scales", "source_opacities", "camera"]

    return TrainingRendererLossResult(
        schema=TRAINING_RENDERER_API_SCHEMA,
        renderer_name=GSPLAT_RENDERER,
        gradient_path=GSPLAT_GRADIENT_PATH,
        frame_count=len(inputs),
        image_render_loss=float(total_loss.detach().cpu().item()),
        frame_losses=tuple(frame_losses),
        rendered_images=tuple(rendered_images),
        gradient_decoder_colors=decoder_gradient,
        gradient_decoder_opacity_logits=opacity_gradient,
        gradient_assignments=assignment_gradients,
        differentiable_fields=tuple(differentiable_fields),
        frozen_fields=tuple(frozen_fields),
        blockers=(),
    )


def _resolve_gsplat_rasterization(
    importer: Callable[[str], Any],
) -> Callable[..., Any]:
    rendering = importer("gsplat.rendering")
    rasterization = getattr(rendering, "rasterization", None)
    if not callable(rasterization):
        raise AttributeError("gsplat.rendering.rasterization is missing")
    return rasterization


def _call_gsplat_rasterization(
    rasterization: Callable[..., Any],
    *,
    means: Any,
    quats: Any,
    scales: Any,
    opacities: Any,
    colors: Any,
    viewmats: Any,
    intrinsics: Any,
    width: int,
    height: int,
) -> Any:
    kwargs = {
        "means": means,
        "quats": quats,
        "scales": scales,
        "opacities": opacities,
        "colors": colors,
        "viewmats": viewmats,
        "Ks": intrinsics,
        "width": int(width),
        "height": int(height),
    }
    try:
        return rasterization(**kwargs, render_mode="RGB")
    except TypeError:
        return rasterization(**kwargs)


def _rendered_rgb_tensor(render_output: Any) -> Any:
    rendered = render_output[0] if isinstance(render_output, (tuple, list)) else render_output
    if getattr(rendered, "ndim", None) == 4:
        rendered = rendered[0]
    if getattr(rendered, "shape", ())[-1] > 3:
        rendered = rendered[..., :3]
    return rendered


def _torch_tensor(torch: Any, value: np.ndarray, device: str) -> Any:
    return torch.tensor(np.asarray(value, dtype=np.float32), dtype=torch.float32, device=device)


def _torch_object_opacity_scales(torch: Any, logits: Any) -> Any:
    clipped = torch.clamp(logits, -60.0, 60.0)
    return torch.clamp(torch.sigmoid(clipped), 0.05, 1.0)


def _object_state_projection_from_frame(
    frame: TrainableKernelFrame,
    assignment: np.ndarray,
) -> ObjectStateProjection:
    positions = _array2d(frame.positions, "frame.positions", columns=3)
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
    features = np.asarray(frame.features, dtype=np.float32)
    return project_object_states(
        GaussianCloud(vertices=vertices, source_format="trainable_frame"),
        assignment,
        evidence_features=features,
    )


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
