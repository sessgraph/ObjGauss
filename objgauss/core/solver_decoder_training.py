from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from objgauss.core.gaussian_decoder import (
    OBJECT_STATE_GAUSSIAN_DECODE_SCHEMA,
    object_opacity_scales_from_logits,
)
from objgauss.core.gaussian_decoder_training import (
    ObjectStateGaussianDecoderState,
    _evaluate_image_renderer,
    _frame_cloud,
    _temporal_centroid_loss,
    initialize_object_state_gaussian_decoder,
    object_state_gaussian_decoder_state_from_dict,
    validate_object_state_gaussian_decoder_state,
)
from objgauss.core.object_emergence_solver import (
    ObjectEmergenceAssignmentPrediction,
    ObjectEmergenceEvidence,
    ObjectEmergenceSolverState,
    initialize_object_emergence_solver,
    object_emergence_solver_state_from_dict,
    predict_object_emergence_assignment,
    validate_object_emergence_evidence,
    validate_object_emergence_solver_state,
)
from objgauss.core.object_state import ObjectStateProjection, project_object_states, validate_assignment_matrix
from objgauss.core.trainable_kernel import (
    TRAINING_IMAGE_RENDERER_GSPLAT,
    TRAINING_IMAGE_RENDERER_POINT,
    TRAINING_IMAGE_RENDERERS,
    TrainableKernelFrame,
    TrainableKernelImageTarget,
    image_target_contract_summary,
)
from objgauss.core.training_renderer import TrainingRendererLossResult

SOLVER_DECODER_JOINT_TRAINING_SCHEMA = "objgauss-solver-decoder-joint-training-v1"
SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA = "objgauss-solver-decoder-joint-checkpoint-v1"
_EPS = 1e-8
_DEFAULT_OPACITY_INIT_LOGIT = 6.0


@dataclass(frozen=True)
class SolverDecoderJointLoss:
    iteration: int
    total_loss: float
    image_render_loss: float
    object_loss: float
    entropy_loss: float
    balance_loss: float
    temporal_loss: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "total_loss": float(self.total_loss),
            "image_render_loss": float(self.image_render_loss),
            "object_loss": float(self.object_loss),
            "entropy_loss": float(self.entropy_loss),
            "balance_loss": float(self.balance_loss),
            "temporal_loss": float(self.temporal_loss),
        }


@dataclass(frozen=True)
class SolverDecoderJointTrainingResult:
    schema: str
    decoder_schema: str
    slots: int
    frame_count: int
    iterations: int
    solver_learning_rate: float
    decoder_learning_rate: float
    decoder_opacity_learning_rate: float
    train_decoder_opacity: bool
    image_render_weight: float
    object_weight: float
    entropy_weight: float
    balance_weight: float
    temporal_weight: float
    image_renderer: str
    gaussian_scale: float
    gaussian_opacity: float
    initial_solver_state: ObjectEmergenceSolverState
    final_solver_state: ObjectEmergenceSolverState
    initial_decoder_state: ObjectStateGaussianDecoderState
    final_decoder_state: ObjectStateGaussianDecoderState
    initial_loss: SolverDecoderJointLoss
    final_loss: SolverDecoderJointLoss
    history: tuple[SolverDecoderJointLoss, ...]
    predictions: tuple[ObjectEmergenceAssignmentPrediction, ...]
    object_state_projections: tuple[ObjectStateProjection, ...]
    final_renderer_api: TrainingRendererLossResult | None
    image_targets: tuple[TrainableKernelImageTarget | None, ...]
    trained_fields: tuple[str, ...]
    frozen_fields: tuple[str, ...]
    gpu_used: bool
    vram_reserve_gb: int

    def as_dict(self, *, include_weights: bool = False, include_assignments: bool = False) -> dict[str, Any]:
        renderer_api = self.final_renderer_api.as_dict() if self.final_renderer_api is not None else None
        return {
            "schema": self.schema,
            "kind": "solver_decoder_joint_training",
            "decoder_schema": self.decoder_schema,
            "slots": int(self.slots),
            "frame_count": int(self.frame_count),
            "iterations": int(self.iterations),
            "learning_rates": {
                "solver": float(self.solver_learning_rate),
                "decoder": float(self.decoder_learning_rate),
                "decoder_opacity": float(self.decoder_opacity_learning_rate),
            },
            "train_decoder_opacity": bool(self.train_decoder_opacity),
            "weights": {
                "image_render": float(self.image_render_weight),
                "object": float(self.object_weight),
                "entropy": float(self.entropy_weight),
                "balance": float(self.balance_weight),
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
            "image_render_loss_decreased": bool(
                self.final_loss.image_render_loss < self.initial_loss.image_render_loss
            ),
            "object_loss_decreased": bool(self.final_loss.object_loss < self.initial_loss.object_loss),
            "initial_solver_state": self.initial_solver_state.as_dict(include_weights=include_weights),
            "final_solver_state": self.final_solver_state.as_dict(include_weights=include_weights),
            "initial_decoder_state": self.initial_decoder_state.as_dict(),
            "final_decoder_state": self.final_decoder_state.as_dict(),
            "decoder_opacity": _decoder_opacity_summary(
                self.final_decoder_state.object_opacity_logits
            ),
            "history": [loss.as_dict() for loss in self.history],
            "predictions": [
                prediction.as_dict(include_assignment=include_assignments)
                for prediction in self.predictions
            ],
            "trained_fields": list(self.trained_fields),
            "frozen_fields": list(self.frozen_fields),
            "renderer_api": renderer_api,
            "image_target_contract": image_target_contract_summary(self.image_targets),
            "gpu_policy": {
                "uses_gpu": bool(self.gpu_used),
                "vram_reserve_gb": int(self.vram_reserve_gb),
            },
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
class _JointEval:
    loss: SolverDecoderJointLoss
    predictions: tuple[ObjectEmergenceAssignmentPrediction, ...]
    projections: tuple[ObjectStateProjection, ...]
    renderer_api: TrainingRendererLossResult | None
    solver_gradient: tuple[np.ndarray, np.ndarray, np.ndarray]
    decoder_gradient: np.ndarray
    decoder_opacity_gradient: np.ndarray


def train_solver_decoder_joint(
    frames: Sequence[TrainableKernelFrame],
    *,
    slots: int | None = None,
    initial_solver_state: ObjectEmergenceSolverState | None = None,
    initial_decoder_state: ObjectStateGaussianDecoderState | None = None,
    iterations: int = 4,
    solver_learning_rate: float = 0.05,
    decoder_learning_rate: float = 0.5,
    train_decoder_opacity: bool = False,
    decoder_opacity_learning_rate: float = 0.02,
    decoder_opacity_init_logit: float = _DEFAULT_OPACITY_INIT_LOGIT,
    image_render_weight: float = 1.0,
    object_weight: float = 0.1,
    entropy_weight: float = 0.0,
    balance_weight: float = 0.0,
    temporal_weight: float = 0.0,
    image_renderer: str = TRAINING_IMAGE_RENDERER_POINT,
    gaussian_scale: float = 0.5,
    gaussian_opacity: float = 1.0,
    solver_temperature: float | None = None,
    seed: int = 0,
    record_every: int | None = None,
    vram_reserve_gb: int = 1,
) -> SolverDecoderJointTrainingResult:
    checked_frames = _validate_frames(frames)
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if solver_learning_rate <= 0:
        raise ValueError("solver_learning_rate must be > 0")
    if decoder_learning_rate <= 0:
        raise ValueError("decoder_learning_rate must be > 0")
    if train_decoder_opacity and decoder_opacity_learning_rate <= 0:
        raise ValueError("decoder_opacity_learning_rate must be > 0 when train_decoder_opacity is enabled")
    if not np.isfinite(decoder_opacity_init_logit):
        raise ValueError("decoder_opacity_init_logit must be finite")
    if image_renderer not in TRAINING_IMAGE_RENDERERS:
        raise ValueError(f"image_renderer must be one of: {', '.join(TRAINING_IMAGE_RENDERERS)}")
    if gaussian_scale <= 0:
        raise ValueError("gaussian_scale must be > 0")
    if not 0.0 <= gaussian_opacity <= 1.0:
        raise ValueError("gaussian_opacity must be in [0, 1]")
    if solver_temperature is not None and solver_temperature <= 0:
        raise ValueError("solver_temperature must be > 0")
    if temporal_weight != 0.0:
        raise ValueError("temporal_weight is tracked but not optimized in joint MVP")
    for name, weight in {
        "image_render_weight": image_render_weight,
        "object_weight": object_weight,
        "entropy_weight": entropy_weight,
        "balance_weight": balance_weight,
        "temporal_weight": temporal_weight,
    }.items():
        if weight < 0:
            raise ValueError(f"{name} must be >= 0")
    if image_render_weight > 0 and not all(frame.image_target is not None for frame in checked_frames):
        raise ValueError("image_render_weight requires every frame to bind image_target")
    if object_weight > 0 and not all(frame.target_assignment is not None for frame in checked_frames):
        raise ValueError("object_weight requires every frame to bind target_assignment")
    if vram_reserve_gb < 0:
        raise ValueError("vram_reserve_gb must be >= 0")

    evidence_frames = _evidence_from_frames(checked_frames)
    solver_state = _initial_solver_state(
        evidence_frames,
        slots=slots,
        initial_solver_state=initial_solver_state,
        solver_temperature=solver_temperature,
        seed=seed,
    )
    initial_solver = solver_state
    decoder_state = initial_decoder_state or initialize_object_state_gaussian_decoder(
        slots=solver_state.config.slots,
        seed=seed,
    )
    decoder_state = validate_object_state_gaussian_decoder_state(decoder_state)
    if decoder_state.slots != solver_state.config.slots:
        raise ValueError("decoder state slots must match solver slots")
    if train_decoder_opacity and decoder_state.object_opacity_logits is None:
        decoder_state = replace(
            decoder_state,
            object_opacity_logits=np.full(
                solver_state.config.slots,
                float(decoder_opacity_init_logit),
                dtype=np.float32,
            ),
            source="initialized_object_opacity_logits",
        )
        decoder_state = validate_object_state_gaussian_decoder_state(decoder_state)

    initial_solver_step = int(solver_state.step)
    initial_decoder_step = int(decoder_state.step)
    history: list[SolverDecoderJointLoss] = []
    colors = decoder_state.object_colors.astype(np.float32, copy=True)
    opacity_logits = (
        _copy_optional_array(decoder_state.object_opacity_logits)
        if train_decoder_opacity
        else None
    )
    record_stride = iterations if record_every is None else max(1, int(record_every))
    initial_eval = _evaluate_joint(
        checked_frames,
        evidence_frames,
        solver_state,
        colors,
        decoder_opacity_logits=opacity_logits,
        iteration=0,
        image_render_weight=image_render_weight,
        object_weight=object_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        temporal_weight=temporal_weight,
        image_renderer=image_renderer,
        gaussian_scale=gaussian_scale,
        gaussian_opacity=gaussian_opacity,
    )
    history.append(initial_eval.loss)

    for iteration in range(1, iterations + 1):
        current = _evaluate_joint(
            checked_frames,
            evidence_frames,
            solver_state,
            colors,
            decoder_opacity_logits=opacity_logits,
            iteration=iteration - 1,
            image_render_weight=image_render_weight,
            object_weight=object_weight,
            entropy_weight=entropy_weight,
            balance_weight=balance_weight,
            temporal_weight=temporal_weight,
            image_renderer=image_renderer,
            gaussian_scale=gaussian_scale,
            gaussian_opacity=gaussian_opacity,
        )
        feature_grad, position_grad, bias_grad = current.solver_gradient
        solver_state = _apply_solver_gradient(
            solver_state,
            feature_grad,
            position_grad,
            bias_grad,
            learning_rate=solver_learning_rate,
            step=initial_solver_step + iteration,
        )
        colors = np.clip(colors - decoder_learning_rate * current.decoder_gradient, 0.0, 1.0)
        if train_decoder_opacity:
            if opacity_logits is None:
                raise ValueError("train_decoder_opacity requires initialized opacity logits")
            opacity_logits = np.clip(
                opacity_logits - decoder_opacity_learning_rate * current.decoder_opacity_gradient,
                -25.0,
                25.0,
            ).astype(np.float32, copy=False)
        if iteration == iterations or iteration % record_stride == 0:
            history.append(
                _evaluate_joint(
                    checked_frames,
                    evidence_frames,
                    solver_state,
                    colors,
                    decoder_opacity_logits=opacity_logits,
                    iteration=iteration,
                    image_render_weight=image_render_weight,
                    object_weight=object_weight,
                    entropy_weight=entropy_weight,
                    balance_weight=balance_weight,
                    temporal_weight=temporal_weight,
                    image_renderer=image_renderer,
                    gaussian_scale=gaussian_scale,
                    gaussian_opacity=gaussian_opacity,
                ).loss
            )

    final_eval = _evaluate_joint(
        checked_frames,
        evidence_frames,
        solver_state,
        colors,
        decoder_opacity_logits=opacity_logits,
        iteration=iterations,
        image_render_weight=image_render_weight,
        object_weight=object_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        temporal_weight=temporal_weight,
        image_renderer=image_renderer,
        gaussian_scale=gaussian_scale,
        gaussian_opacity=gaussian_opacity,
    )
    if history[-1].iteration != iterations:
        history.append(final_eval.loss)
    else:
        history[-1] = final_eval.loss

    return SolverDecoderJointTrainingResult(
        schema=SOLVER_DECODER_JOINT_TRAINING_SCHEMA,
        decoder_schema=OBJECT_STATE_GAUSSIAN_DECODE_SCHEMA,
        slots=solver_state.config.slots,
        frame_count=len(checked_frames),
        iterations=int(iterations),
        solver_learning_rate=float(solver_learning_rate),
        decoder_learning_rate=float(decoder_learning_rate),
        decoder_opacity_learning_rate=float(decoder_opacity_learning_rate),
        train_decoder_opacity=bool(train_decoder_opacity),
        image_render_weight=float(image_render_weight),
        object_weight=float(object_weight),
        entropy_weight=float(entropy_weight),
        balance_weight=float(balance_weight),
        temporal_weight=float(temporal_weight),
        image_renderer=image_renderer,
        gaussian_scale=float(gaussian_scale),
        gaussian_opacity=float(gaussian_opacity),
        initial_solver_state=initial_solver,
        final_solver_state=solver_state,
        initial_decoder_state=decoder_state,
        final_decoder_state=ObjectStateGaussianDecoderState(
            object_colors=colors.astype(np.float32, copy=True),
            object_opacity_logits=(
                _copy_optional_array(opacity_logits)
                if train_decoder_opacity
                else _copy_optional_array(decoder_state.object_opacity_logits)
            ),
            object_scale_log_offsets=_copy_optional_array(decoder_state.object_scale_log_offsets),
            step=initial_decoder_step + int(iterations),
            source=(
                "joint_trained_renderer_gradient_object_colors_and_opacity"
                if train_decoder_opacity
                else "joint_trained_renderer_gradient_object_colors"
            ),
        ),
        initial_loss=initial_eval.loss,
        final_loss=final_eval.loss,
        history=tuple(history),
        predictions=final_eval.predictions,
        object_state_projections=final_eval.projections,
        final_renderer_api=final_eval.renderer_api,
        image_targets=tuple(frame.image_target for frame in checked_frames),
        trained_fields=(
            "solver.feature_weights",
            "solver.position_weights",
            "solver.bias",
            "decoder.object_colors",
            *(
                ("decoder.object_opacity_logits",)
                if train_decoder_opacity
                else ()
            ),
        ),
        frozen_fields=(
            "means",
            "quats",
            "scales",
            "source_opacities" if train_decoder_opacity else "opacities",
            "cameras",
            "dynamic_k",
        ),
        gpu_used=image_renderer == TRAINING_IMAGE_RENDERER_GSPLAT,
        vram_reserve_gb=int(vram_reserve_gb),
    )


def solver_decoder_joint_checkpoint(
    result: SolverDecoderJointTrainingResult,
    *,
    input_path: str | None = None,
    source_gaussians: int | None = None,
    sampled_gaussians: int | None = None,
    target_source: str | None = None,
    assignment_source: str | None = None,
    object_id_mapping: dict[int, int] | dict[str, int] | None = None,
    solver_checkpoint: str | None = None,
    resume_checkpoint: str | None = None,
    vram_reserve_gb: int = 1,
) -> dict[str, Any]:
    if result.schema != SOLVER_DECODER_JOINT_TRAINING_SCHEMA:
        raise ValueError(f"unsupported joint training schema: {result.schema}")
    if vram_reserve_gb < 0:
        raise ValueError("vram_reserve_gb must be >= 0")
    renderer_api = result.final_renderer_api.as_dict() if result.final_renderer_api is not None else None
    payload = {
        "schema": SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA,
        "kind": "solver_decoder_joint_checkpoint",
        "training_schema": result.schema,
        "decoder_schema": result.decoder_schema,
        "source": {
            "input": input_path,
            "source_gaussians": None if source_gaussians is None else int(source_gaussians),
            "sampled_gaussians": None if sampled_gaussians is None else int(sampled_gaussians),
            "target_source": target_source,
            "assignment_source": assignment_source,
            "object_id_mapping": _string_key_mapping(object_id_mapping),
            "solver_checkpoint": solver_checkpoint,
            "resume_checkpoint": resume_checkpoint,
        },
        "training": {
            "iterations": int(result.iterations),
            "learning_rates": {
                "solver": float(result.solver_learning_rate),
                "decoder": float(result.decoder_learning_rate),
                "decoder_opacity": float(result.decoder_opacity_learning_rate),
            },
            "train_decoder_opacity": bool(result.train_decoder_opacity),
            "weights": {
                "image_render": float(result.image_render_weight),
                "object": float(result.object_weight),
                "entropy": float(result.entropy_weight),
                "balance": float(result.balance_weight),
                "temporal": float(result.temporal_weight),
            },
            "image_renderer": result.image_renderer,
            "gaussian_policy": {
                "default_scale": float(result.gaussian_scale),
                "default_opacity": float(result.gaussian_opacity),
            },
            "initial_loss": result.initial_loss.as_dict(),
            "final_loss": result.final_loss.as_dict(),
            "loss_decreased": bool(result.final_loss.total_loss < result.initial_loss.total_loss),
            "image_render_loss_decreased": bool(
                result.final_loss.image_render_loss < result.initial_loss.image_render_loss
            ),
            "object_loss_decreased": bool(result.final_loss.object_loss < result.initial_loss.object_loss),
        },
        "solver_state": result.final_solver_state.as_dict(include_weights=True),
        "decoder_state": result.final_decoder_state.as_dict(),
        "trained_fields": list(result.trained_fields),
        "frozen_fields": list(result.frozen_fields),
        "renderer_api": renderer_api,
        "image_target_contract": image_target_contract_summary(result.image_targets),
        "gpu_policy": {
            "uses_gpu": bool(result.gpu_used),
            "vram_reserve_gb": int(vram_reserve_gb),
        },
        "export_policy": {
            "repository_write": "do_not_commit_training_checkpoints",
            "intended_locations": ["/tmp", "ignored outputs/"],
            "large_artifacts": "keep_out_of_git",
        },
    }
    return validate_solver_decoder_joint_checkpoint(payload)


def validate_solver_decoder_joint_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("joint checkpoint payload must be a dict")
    if payload.get("schema") != SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported joint checkpoint schema: {payload.get('schema')}")
    if payload.get("kind") != "solver_decoder_joint_checkpoint":
        raise ValueError("joint checkpoint kind must be solver_decoder_joint_checkpoint")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("joint checkpoint missing training")
    for key in ("initial_loss", "final_loss"):
        if not isinstance(training.get(key), dict):
            raise ValueError(f"joint checkpoint missing training.{key}")
        for loss_key in (
            "total_loss",
            "image_render_loss",
            "object_loss",
            "entropy_loss",
            "balance_loss",
            "temporal_loss",
        ):
            if loss_key not in training[key]:
                raise ValueError(f"joint checkpoint training.{key} missing {loss_key}")
            float(training[key][loss_key])
    solver_state, decoder_state = solver_decoder_joint_states_from_dict(payload)
    if solver_state.config.slots != decoder_state.slots:
        raise ValueError("joint checkpoint solver and decoder slots must match")
    gpu_policy = payload.get("gpu_policy")
    if not isinstance(gpu_policy, dict):
        raise ValueError("joint checkpoint missing gpu_policy")
    if "vram_reserve_gb" not in gpu_policy:
        raise ValueError("joint checkpoint missing gpu_policy.vram_reserve_gb")
    if int(gpu_policy["vram_reserve_gb"]) < 0:
        raise ValueError("gpu_policy.vram_reserve_gb must be >= 0")
    if not isinstance(payload.get("trained_fields"), list):
        raise ValueError("joint checkpoint missing trained_fields")
    if not isinstance(payload.get("frozen_fields"), list):
        raise ValueError("joint checkpoint missing frozen_fields")
    return payload


def solver_decoder_joint_states_from_dict(
    payload: dict[str, Any],
) -> tuple[ObjectEmergenceSolverState, ObjectStateGaussianDecoderState]:
    if not isinstance(payload, dict):
        raise TypeError("joint checkpoint payload must be a dict")
    if payload.get("schema") != SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported joint checkpoint schema: {payload.get('schema')}")
    solver_payload = payload.get("solver_state")
    decoder_payload = payload.get("decoder_state")
    if not isinstance(solver_payload, dict):
        raise ValueError("joint checkpoint missing solver_state")
    if not isinstance(decoder_payload, dict):
        raise ValueError("joint checkpoint missing decoder_state")
    solver_state = object_emergence_solver_state_from_dict(solver_payload)
    decoder_state = object_state_gaussian_decoder_state_from_dict(decoder_payload)
    return solver_state, decoder_state


def _evaluate_joint(
    frames: tuple[TrainableKernelFrame, ...],
    evidence_frames: tuple[ObjectEmergenceEvidence, ...],
    solver_state: ObjectEmergenceSolverState,
    colors: np.ndarray,
    *,
    decoder_opacity_logits: np.ndarray | None,
    iteration: int,
    image_render_weight: float,
    object_weight: float,
    entropy_weight: float,
    balance_weight: float,
    temporal_weight: float,
    image_renderer: str,
    gaussian_scale: float,
    gaussian_opacity: float,
) -> _JointEval:
    predictions = tuple(
        predict_object_emergence_assignment(evidence, solver_state)
        for evidence in evidence_frames
    )
    assignments = tuple(prediction.assignment for prediction in predictions)
    renderer_api = None
    image_loss = 0.0
    assignment_gradients = tuple(np.zeros_like(assignment, dtype=np.float32) for assignment in assignments)
    decoder_gradient = np.zeros_like(colors, dtype=np.float32)
    decoder_opacity_gradient = (
        np.zeros((0,), dtype=np.float32)
        if decoder_opacity_logits is None
        else np.zeros_like(decoder_opacity_logits, dtype=np.float32)
    )
    if image_render_weight > 0:
        renderer_api = _evaluate_image_renderer(
            frames,
            assignments,
            colors,
            decoder_opacity_logits=decoder_opacity_logits,
            image_renderer=image_renderer,
            gaussian_scale=gaussian_scale,
            gaussian_opacity=gaussian_opacity,
        )
        image_loss = float(renderer_api.image_render_loss)
        assignment_gradients = tuple(
            image_render_weight * gradient
            for gradient in renderer_api.gradient_assignments
        )
        decoder_gradient = image_render_weight * renderer_api.gradient_decoder_colors
        if decoder_opacity_logits is not None:
            decoder_opacity_gradient = (
                image_render_weight * renderer_api.gradient_decoder_opacity_logits
            )

    object_loss, object_assignment_gradients = _object_loss_and_gradient(frames, assignments)
    entropy_loss, entropy_gradients = _entropy_loss_and_gradient(assignments)
    balance_loss, balance_gradients = _balance_loss_and_gradient(assignments)
    projections = tuple(
        project_object_states(_frame_cloud(frame), assignment, evidence_features=frame.features)
        for frame, assignment in zip(frames, assignments, strict=True)
    )
    temporal_loss = _temporal_centroid_loss(projections)
    total = (
        image_render_weight * image_loss
        + object_weight * object_loss
        + entropy_weight * entropy_loss
        + balance_weight * balance_loss
        + temporal_weight * temporal_loss
    )
    combined_assignment_gradients = tuple(
        renderer_gradient
        + object_weight * object_gradient
        + entropy_weight * entropy_gradient
        + balance_weight * balance_gradient
        for renderer_gradient, object_gradient, entropy_gradient, balance_gradient in zip(
            assignment_gradients,
            object_assignment_gradients,
            entropy_gradients,
            balance_gradients,
            strict=True,
        )
    )
    solver_gradient = _solver_gradient_from_assignments(
        evidence_frames,
        predictions,
        solver_state,
        combined_assignment_gradients,
    )
    return _JointEval(
        loss=SolverDecoderJointLoss(
            iteration=int(iteration),
            total_loss=float(total),
            image_render_loss=float(image_loss),
            object_loss=float(object_loss),
            entropy_loss=float(entropy_loss),
            balance_loss=float(balance_loss),
            temporal_loss=float(temporal_loss),
        ),
        predictions=predictions,
        projections=projections,
        renderer_api=renderer_api,
        solver_gradient=solver_gradient,
        decoder_gradient=decoder_gradient.astype(np.float32, copy=False),
        decoder_opacity_gradient=decoder_opacity_gradient.astype(np.float32, copy=False),
    )


def _solver_gradient_from_assignments(
    evidence_frames: tuple[ObjectEmergenceEvidence, ...],
    predictions: tuple[ObjectEmergenceAssignmentPrediction, ...],
    solver_state: ObjectEmergenceSolverState,
    assignment_gradients: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_grad = np.zeros_like(solver_state.feature_weights, dtype=np.float32)
    position_grad = np.zeros_like(solver_state.position_weights, dtype=np.float32)
    bias_grad = np.zeros_like(solver_state.bias, dtype=np.float32)
    temperature = max(float(solver_state.config.temperature), _EPS)
    for evidence, prediction, assignment_gradient in zip(
        evidence_frames,
        predictions,
        assignment_gradients,
        strict=True,
    ):
        positions, features, _target = validate_object_emergence_evidence(
            evidence,
            slots=solver_state.config.slots,
        )
        assignment = prediction.assignment
        inner = np.sum(assignment_gradient * assignment, axis=1, keepdims=True)
        grad_logits = (assignment * (assignment_gradient - inner)) / temperature
        feature_grad += solver_state.config.feature_weight * (features.T @ grad_logits)
        position_grad += solver_state.config.position_weight * (positions.T @ grad_logits)
        bias_grad += np.sum(grad_logits, axis=0)
    return (
        feature_grad.astype(np.float32, copy=False),
        position_grad.astype(np.float32, copy=False),
        bias_grad.astype(np.float32, copy=False),
    )


def _apply_solver_gradient(
    solver_state: ObjectEmergenceSolverState,
    feature_grad: np.ndarray,
    position_grad: np.ndarray,
    bias_grad: np.ndarray,
    *,
    learning_rate: float,
    step: int,
) -> ObjectEmergenceSolverState:
    return validate_object_emergence_solver_state(
        replace(
            solver_state,
            feature_weights=np.clip(
                solver_state.feature_weights - learning_rate * feature_grad,
                -25.0,
                25.0,
            ).astype(np.float32, copy=False),
            position_weights=np.clip(
                solver_state.position_weights - learning_rate * position_grad,
                -25.0,
                25.0,
            ).astype(np.float32, copy=False),
            bias=np.clip(
                solver_state.bias - learning_rate * bias_grad,
                -25.0,
                25.0,
            ).astype(np.float32, copy=False),
            step=int(step),
            source="joint_trained_renderer_gradient_assignment",
        )
    )


def _object_loss_and_gradient(
    frames: tuple[TrainableKernelFrame, ...],
    assignments: tuple[np.ndarray, ...],
) -> tuple[float, tuple[np.ndarray, ...]]:
    losses: list[float] = []
    gradients: list[np.ndarray] = []
    frame_count = max(len(frames), 1)
    for frame, assignment in zip(frames, assignments, strict=True):
        if frame.target_assignment is None:
            gradients.append(np.zeros_like(assignment, dtype=np.float32))
            continue
        target = validate_assignment_matrix(frame.target_assignment, evidence_count=assignment.shape[0])
        if target.shape[1] != assignment.shape[1]:
            raise ValueError("target_assignment slots must match solver slots")
        clipped = np.clip(assignment, _EPS, 1.0)
        losses.append(float(-np.mean(np.sum(target * np.log(clipped), axis=1))))
        gradients.append((-(target / clipped) / max(float(assignment.shape[0]), _EPS) / frame_count).astype(np.float32))
    return float(np.mean(losses)) if losses else 0.0, tuple(gradients)


def _entropy_loss_and_gradient(assignments: tuple[np.ndarray, ...]) -> tuple[float, tuple[np.ndarray, ...]]:
    losses: list[float] = []
    gradients: list[np.ndarray] = []
    frame_count = max(len(assignments), 1)
    for assignment in assignments:
        if assignment.shape[1] <= 1:
            gradients.append(np.zeros_like(assignment, dtype=np.float32))
            continue
        clipped = np.clip(assignment, _EPS, 1.0)
        normalizer = np.log(float(assignment.shape[1]))
        entropy = -np.sum(assignment * np.log(clipped), axis=1) / normalizer
        losses.append(float(np.mean(entropy)))
        gradients.append((-(np.log(clipped) + 1.0) / normalizer / max(float(assignment.shape[0]), _EPS) / frame_count).astype(np.float32))
    return float(np.mean(losses)) if losses else 0.0, tuple(gradients)


def _balance_loss_and_gradient(assignments: tuple[np.ndarray, ...]) -> tuple[float, tuple[np.ndarray, ...]]:
    losses: list[float] = []
    gradients: list[np.ndarray] = []
    frame_count = max(len(assignments), 1)
    for assignment in assignments:
        evidence_count = max(float(assignment.shape[0]), _EPS)
        slots = assignment.shape[1]
        mass_fraction = np.sum(assignment, axis=0) / evidence_count
        target = np.full(slots, 1.0 / float(slots), dtype=np.float32)
        delta = mass_fraction - target
        losses.append(float(np.mean(delta ** 2)))
        per_slot = (2.0 / float(slots)) * delta / evidence_count / frame_count
        gradients.append(np.tile(per_slot[None, :], (assignment.shape[0], 1)).astype(np.float32))
    return float(np.mean(losses)) if losses else 0.0, tuple(gradients)


def _validate_frames(frames: Sequence[TrainableKernelFrame]) -> tuple[TrainableKernelFrame, ...]:
    checked = tuple(frames)
    if not checked:
        raise ValueError("at least one frame is required")
    for index, frame in enumerate(checked):
        evidence = ObjectEmergenceEvidence(
            positions=frame.positions,
            features=frame.features,
            target_assignment=frame.target_assignment,
            frame_index=index,
        )
        validate_object_emergence_evidence(evidence)
        target_rgb = np.asarray(frame.target_rgb, dtype=np.float32)
        if target_rgb.ndim != 2 or target_rgb.shape != (evidence.evidence_count, 3):
            raise ValueError(f"frames[{index}].target_rgb must have shape N x 3")
    return checked


def _copy_optional_array(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float32).astype(np.float32, copy=True)


def _decoder_opacity_summary(object_opacity_logits: np.ndarray | None) -> dict[str, Any]:
    if object_opacity_logits is None:
        return {
            "enabled": False,
            "scale_min": None,
            "scale_mean": None,
            "scale_max": None,
        }
    scales = object_opacity_scales_from_logits(np.asarray(object_opacity_logits, dtype=np.float32))
    return {
        "enabled": True,
        "scale_min": float(np.min(scales)),
        "scale_mean": float(np.mean(scales)),
        "scale_max": float(np.max(scales)),
    }


def _evidence_from_frames(frames: tuple[TrainableKernelFrame, ...]) -> tuple[ObjectEmergenceEvidence, ...]:
    return tuple(
        ObjectEmergenceEvidence(
            positions=frame.positions,
            features=frame.features,
            target_assignment=frame.target_assignment,
            frame_index=index,
            source="trainable_kernel_frame",
        )
        for index, frame in enumerate(frames)
    )


def _initial_solver_state(
    evidence_frames: tuple[ObjectEmergenceEvidence, ...],
    *,
    slots: int | None,
    initial_solver_state: ObjectEmergenceSolverState | None,
    solver_temperature: float | None,
    seed: int,
) -> ObjectEmergenceSolverState:
    if initial_solver_state is not None:
        state = _solver_state_with_temperature(
            validate_object_emergence_solver_state(initial_solver_state),
            solver_temperature=solver_temperature,
        )
        for evidence in evidence_frames:
            validate_object_emergence_evidence(evidence, slots=state.config.slots)
            if evidence.feature_dim != state.config.feature_dim:
                raise ValueError("evidence feature_dim must match solver state")
        return state
    resolved_slots = slots
    target_slots = {
        int(evidence.target_assignment.shape[1])
        for evidence in evidence_frames
        if evidence.target_assignment is not None
    }
    if resolved_slots is None and len(target_slots) == 1:
        resolved_slots = target_slots.pop()
    if resolved_slots is None:
        raise ValueError("slots is required when frames have no target_assignment")
    return initialize_object_emergence_solver(
        slots=int(resolved_slots),
        feature_dim=evidence_frames[0].feature_dim,
        temperature=1.0 if solver_temperature is None else float(solver_temperature),
        seed=seed,
    )


def _solver_state_with_temperature(
    state: ObjectEmergenceSolverState,
    *,
    solver_temperature: float | None,
) -> ObjectEmergenceSolverState:
    if solver_temperature is None:
        return state
    if solver_temperature <= 0:
        raise ValueError("solver_temperature must be > 0")
    temperature = float(solver_temperature)
    if abs(float(state.config.temperature) - temperature) <= _EPS:
        return state
    config = replace(state.config, temperature=temperature)
    return validate_object_emergence_solver_state(
        replace(
            state,
            config=config,
            source=f"{state.source}|temperature_override",
        )
    )


def _string_key_mapping(mapping: dict[int, int] | dict[str, int] | None) -> dict[str, int] | None:
    if mapping is None:
        return None
    return {str(key): int(value) for key, value in mapping.items()}
