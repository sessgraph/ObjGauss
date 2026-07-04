from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Sequence

import numpy as np

from objgauss.core.assignment_evidence import (
    AssignmentEvidenceBatch,
    validate_assignment_evidence_batch,
)
from objgauss.core.assignment_losses import assignment_loss_v2_breakdown
from objgauss.core.features import extract_features, positions
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_state import (
    ObjectStateProjection,
    project_object_states,
    validate_assignment_matrix,
)

OBJECT_EMERGENCE_MODEL_SCHEMA = "objgauss-object-emergence-model-v1"
OBJECT_EMERGENCE_SOLVER_STATE_SCHEMA = "objgauss-object-emergence-solver-state-v1"
OBJECT_EMERGENCE_ASSIGNMENT_SCHEMA = "objgauss-object-emergence-assignment-v1"
OBJECT_EMERGENCE_TRAINING_SCHEMA = "objgauss-object-emergence-solver-training-v1"
OBJECT_EMERGENCE_SOLVER_CHECKPOINT_SCHEMA = "objgauss-object-emergence-solver-checkpoint-v1"
ASSIGNMENT_MVP_TRAINING_SCHEMA = "objgauss-assignment-mvp-training-v1"
_EPS = 1e-8


@dataclass(frozen=True)
class ObjectEmergenceEvidence:
    positions: np.ndarray
    features: np.ndarray
    target_assignment: np.ndarray | None = None
    frame_index: int = 0
    source: str = "unknown"

    @property
    def evidence_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def feature_dim(self) -> int:
        return int(self.features.shape[1])

    def as_dict(self) -> dict[str, Any]:
        positions_array, features_array, target = validate_object_emergence_evidence(self)
        return {
            "schema": OBJECT_EMERGENCE_MODEL_SCHEMA,
            "kind": "object_emergence_evidence",
            "source": self.source,
            "frame_index": int(self.frame_index),
            "evidence_count": int(positions_array.shape[0]),
            "position_dim": int(positions_array.shape[1]),
            "feature_dim": int(features_array.shape[1]),
            "has_target_assignment": target is not None,
            "target_slots": None if target is None else int(target.shape[1]),
        }


@dataclass(frozen=True)
class ObjectEmergenceSolverConfig:
    slots: int
    feature_dim: int
    position_dim: int = 3
    temperature: float = 1.0
    feature_weight: float = 1.0
    position_weight: float = 1.0
    model_family: str = "linear-softmax-assignment"

    def as_dict(self) -> dict[str, Any]:
        _validate_solver_config(self)
        return {
            "schema": OBJECT_EMERGENCE_MODEL_SCHEMA,
            "slots": int(self.slots),
            "feature_dim": int(self.feature_dim),
            "position_dim": int(self.position_dim),
            "temperature": float(self.temperature),
            "feature_weight": float(self.feature_weight),
            "position_weight": float(self.position_weight),
            "model_family": self.model_family,
        }


@dataclass(frozen=True)
class ObjectEmergenceSolverState:
    config: ObjectEmergenceSolverConfig
    feature_weights: np.ndarray
    position_weights: np.ndarray
    bias: np.ndarray
    step: int = 0
    source: str = "initialized"
    schema: str = OBJECT_EMERGENCE_SOLVER_STATE_SCHEMA

    def as_dict(self, *, include_weights: bool = False) -> dict[str, Any]:
        state = validate_object_emergence_solver_state(self)
        payload: dict[str, Any] = {
            "schema": state.schema,
            "source": state.source,
            "step": int(state.step),
            "config": state.config.as_dict(),
            "weight_shapes": {
                "feature_weights": list(state.feature_weights.shape),
                "position_weights": list(state.position_weights.shape),
                "bias": list(state.bias.shape),
            },
        }
        if include_weights:
            payload["weights"] = {
                "feature_weights": np.round(state.feature_weights, 6).tolist(),
                "position_weights": np.round(state.position_weights, 6).tolist(),
                "bias": np.round(state.bias, 6).tolist(),
            }
        return payload


@dataclass(frozen=True)
class ObjectEmergenceAssignmentPrediction:
    assignment: np.ndarray
    logits: np.ndarray
    top_slots: np.ndarray
    confidence: np.ndarray
    slot_mass: np.ndarray
    mean_normalized_entropy: float
    diagnostics: tuple[str, ...]
    frame_index: int
    solver_step: int
    source: str
    schema: str = OBJECT_EMERGENCE_ASSIGNMENT_SCHEMA

    @property
    def evidence_count(self) -> int:
        return int(self.assignment.shape[0])

    @property
    def slots(self) -> int:
        return int(self.assignment.shape[1])

    def as_dict(self, *, include_assignment: bool = False) -> dict[str, Any]:
        validate_assignment_matrix(self.assignment)
        payload: dict[str, Any] = {
            "schema": self.schema,
            "source": self.source,
            "frame_index": int(self.frame_index),
            "solver_step": int(self.solver_step),
            "evidence_count": self.evidence_count,
            "slots": self.slots,
            "mean_normalized_entropy": float(self.mean_normalized_entropy),
            "slot_mass": np.round(self.slot_mass, 6).tolist(),
            "top_slots": self.top_slots.astype(int).tolist(),
            "confidence": np.round(self.confidence, 6).tolist(),
            "diagnostics": list(self.diagnostics),
        }
        if include_assignment:
            payload["assignment"] = np.round(self.assignment, 6).tolist()
            payload["logits"] = np.round(self.logits, 6).tolist()
        return payload


@dataclass(frozen=True)
class ObjectEmergenceSolverLoss:
    iteration: int
    total_loss: float
    assignment_loss: float
    entropy_loss: float
    balance_loss: float
    temporal_loss: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "total_loss": float(self.total_loss),
            "assignment_loss": float(self.assignment_loss),
            "entropy_loss": float(self.entropy_loss),
            "balance_loss": float(self.balance_loss),
            "temporal_loss": float(self.temporal_loss),
        }


@dataclass(frozen=True)
class ObjectEmergenceSolverTrainingResult:
    schema: str
    initial_state: ObjectEmergenceSolverState
    final_state: ObjectEmergenceSolverState
    initial_loss: ObjectEmergenceSolverLoss
    final_loss: ObjectEmergenceSolverLoss
    history: tuple[ObjectEmergenceSolverLoss, ...]
    predictions: tuple[ObjectEmergenceAssignmentPrediction, ...]
    assignment_weight: float
    entropy_weight: float
    balance_weight: float
    temporal_weight: float
    learning_rate: float
    iterations: int
    finite_difference_epsilon: float

    def as_dict(self, *, include_weights: bool = False, include_assignments: bool = False) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": "object_emergence_solver_training",
            "iterations": int(self.iterations),
            "learning_rate": float(self.learning_rate),
            "finite_difference_epsilon": float(self.finite_difference_epsilon),
            "weights": {
                "assignment": float(self.assignment_weight),
                "entropy": float(self.entropy_weight),
                "balance": float(self.balance_weight),
                "temporal": float(self.temporal_weight),
            },
            "initial_loss": self.initial_loss.as_dict(),
            "final_loss": self.final_loss.as_dict(),
            "loss_decreased": bool(self.final_loss.total_loss < self.initial_loss.total_loss),
            "assignment_loss_decreased": bool(
                self.final_loss.assignment_loss < self.initial_loss.assignment_loss
            ),
            "balance_loss_decreased": bool(
                self.final_loss.balance_loss < self.initial_loss.balance_loss
            ),
            "initial_solver_state": self.initial_state.as_dict(include_weights=include_weights),
            "final_solver_state": self.final_state.as_dict(include_weights=include_weights),
            "history": [loss.as_dict() for loss in self.history],
            "predictions": [
                prediction.as_dict(include_assignment=include_assignments)
                for prediction in self.predictions
            ],
        }


@dataclass(frozen=True)
class _SolverTrainingEval:
    state: ObjectEmergenceSolverState
    loss: ObjectEmergenceSolverLoss
    predictions: tuple[ObjectEmergenceAssignmentPrediction, ...]


def object_emergence_solver_checkpoint(
    result: ObjectEmergenceSolverTrainingResult,
    *,
    input_path: str | None = None,
    source_gaussians: int | None = None,
    sampled_gaussians: int | None = None,
    target_source: str | None = None,
    object_id_mapping: dict[int, int] | dict[str, int] | None = None,
    vram_reserve_gb: int = 1,
) -> dict[str, Any]:
    if result.schema != OBJECT_EMERGENCE_TRAINING_SCHEMA:
        raise ValueError(f"unsupported solver training schema: {result.schema}")
    if vram_reserve_gb < 0:
        raise ValueError("vram_reserve_gb must be >= 0")
    payload = {
        "schema": OBJECT_EMERGENCE_SOLVER_CHECKPOINT_SCHEMA,
        "kind": "object_emergence_solver_checkpoint",
        "training_schema": result.schema,
        "source": {
            "input": input_path,
            "source_gaussians": None if source_gaussians is None else int(source_gaussians),
            "sampled_gaussians": None if sampled_gaussians is None else int(sampled_gaussians),
            "target_source": target_source,
            "object_id_mapping": _string_key_mapping(object_id_mapping),
        },
        "training": {
            "iterations": int(result.iterations),
            "learning_rate": float(result.learning_rate),
            "finite_difference_epsilon": float(result.finite_difference_epsilon),
            "weights": {
                "assignment": float(result.assignment_weight),
                "entropy": float(result.entropy_weight),
                "balance": float(result.balance_weight),
                "temporal": float(result.temporal_weight),
            },
            "initial_loss": result.initial_loss.as_dict(),
            "final_loss": result.final_loss.as_dict(),
            "loss_decreased": bool(result.final_loss.total_loss < result.initial_loss.total_loss),
            "assignment_loss_decreased": bool(
                result.final_loss.assignment_loss < result.initial_loss.assignment_loss
            ),
        },
        "solver_state": result.final_state.as_dict(include_weights=True),
        "gpu_policy": {
            "uses_gpu": False,
            "full_renderer_training": "suspended_until_torch_gsplat_cuda_available",
            "vram_reserve_gb": int(vram_reserve_gb),
        },
        "export_policy": {
            "repository_write": "do_not_commit_training_checkpoints",
            "intended_locations": ["/tmp", "ignored outputs/"],
            "large_artifacts": "keep_out_of_git",
        },
    }
    return validate_object_emergence_solver_checkpoint(payload)


def assignment_mvp_training_summary(
    result: ObjectEmergenceSolverTrainingResult,
    evidence_batches: Sequence[AssignmentEvidenceBatch],
    *,
    include_solver_summary: bool = False,
) -> dict[str, Any]:
    if result.schema != OBJECT_EMERGENCE_TRAINING_SCHEMA:
        raise ValueError(f"unsupported solver training schema: {result.schema}")
    batches = tuple(validate_assignment_evidence_batch(batch) for batch in evidence_batches)
    if not batches:
        raise ValueError("evidence_batches must contain at least one batch")
    solver_frames = tuple(_object_emergence_evidence_from_assignment_batch(batch) for batch in batches)
    for frame in solver_frames:
        validate_object_emergence_evidence(frame, slots=result.final_state.config.slots)
    initial_predictions = tuple(
        predict_object_emergence_assignment(frame, result.initial_state)
        for frame in solver_frames
    )
    final_predictions = tuple(
        predict_object_emergence_assignment(frame, result.final_state)
        for frame in solver_frames
    )
    target_assignments = tuple(batch.target_assignment for batch in batches)
    initial_loss = assignment_loss_v2_breakdown(
        tuple(prediction.assignment for prediction in initial_predictions),
        target_assignments=target_assignments,
        entropy_weight=result.entropy_weight,
        balance_weight=result.balance_weight,
        supervised_weight=result.assignment_weight,
    )
    final_loss = assignment_loss_v2_breakdown(
        tuple(prediction.assignment for prediction in final_predictions),
        target_assignments=target_assignments,
        entropy_weight=result.entropy_weight,
        balance_weight=result.balance_weight,
        supervised_weight=result.assignment_weight,
    )
    payload: dict[str, Any] = {
        "schema": ASSIGNMENT_MVP_TRAINING_SCHEMA,
        "kind": "fixed_k_assignment_mvp",
        "solver_training_schema": result.schema,
        "evidence_schema": batches[0].schema,
        "frame_count": len(batches),
        "slots": int(result.final_state.config.slots),
        "iterations": int(result.iterations),
        "fixed_k": True,
        "renderer_loss": "not_used",
        "dynamic_k": "disabled",
        "initial_loss_v2": initial_loss.as_dict(),
        "final_loss_v2": final_loss.as_dict(),
        "loss_decreased": bool(final_loss.total_loss < initial_loss.total_loss),
        "supervised_loss_decreased": bool(
            final_loss.supervised_loss < initial_loss.supervised_loss
        ),
        "evidence": [batch.as_dict() for batch in batches],
    }
    if include_solver_summary:
        payload["solver_training"] = result.as_dict()
    return payload


def object_emergence_solver_state_from_dict(payload: dict[str, Any]) -> ObjectEmergenceSolverState:
    if not isinstance(payload, dict):
        raise TypeError("solver state payload must be a dict")
    if payload.get("schema") == OBJECT_EMERGENCE_SOLVER_CHECKPOINT_SCHEMA:
        state_payload = payload.get("solver_state")
        if not isinstance(state_payload, dict):
            raise ValueError("solver checkpoint missing solver_state")
        return object_emergence_solver_state_from_dict(state_payload)
    if payload.get("schema") != OBJECT_EMERGENCE_SOLVER_STATE_SCHEMA:
        raise ValueError(f"unsupported solver state schema: {payload.get('schema')}")
    config_payload = payload.get("config")
    weights_payload = payload.get("weights")
    if not isinstance(config_payload, dict):
        raise ValueError("solver state missing config")
    if not isinstance(weights_payload, dict):
        raise ValueError("solver state missing weights")
    config = ObjectEmergenceSolverConfig(
        slots=int(config_payload.get("slots")),
        feature_dim=int(config_payload.get("feature_dim")),
        position_dim=int(config_payload.get("position_dim", 3)),
        temperature=float(config_payload.get("temperature", 1.0)),
        feature_weight=float(config_payload.get("feature_weight", 1.0)),
        position_weight=float(config_payload.get("position_weight", 1.0)),
        model_family=str(config_payload.get("model_family", "linear-softmax-assignment")),
    )
    state = ObjectEmergenceSolverState(
        config=config,
        feature_weights=np.asarray(weights_payload.get("feature_weights"), dtype=np.float32),
        position_weights=np.asarray(weights_payload.get("position_weights"), dtype=np.float32),
        bias=np.asarray(weights_payload.get("bias"), dtype=np.float32),
        step=int(payload.get("step", 0)),
        source=str(payload.get("source", "checkpoint")),
        schema=str(payload.get("schema")),
    )
    return validate_object_emergence_solver_state(state)


def validate_object_emergence_solver_checkpoint(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("solver checkpoint payload must be a dict")
    if payload.get("schema") != OBJECT_EMERGENCE_SOLVER_CHECKPOINT_SCHEMA:
        raise ValueError(f"unsupported solver checkpoint schema: {payload.get('schema')}")
    if payload.get("kind") != "object_emergence_solver_checkpoint":
        raise ValueError("solver checkpoint kind must be object_emergence_solver_checkpoint")
    training = payload.get("training")
    if not isinstance(training, dict):
        raise ValueError("solver checkpoint missing training")
    for key in ("initial_loss", "final_loss"):
        if not isinstance(training.get(key), dict):
            raise ValueError(f"solver checkpoint missing training.{key}")
        for loss_key in ("total_loss", "assignment_loss", "entropy_loss", "balance_loss", "temporal_loss"):
            if loss_key not in training[key]:
                raise ValueError(f"solver checkpoint training.{key} missing {loss_key}")
            float(training[key][loss_key])
    state = object_emergence_solver_state_from_dict(payload)
    gpu_policy = payload.get("gpu_policy")
    if not isinstance(gpu_policy, dict):
        raise ValueError("solver checkpoint missing gpu_policy")
    if "vram_reserve_gb" not in gpu_policy:
        raise ValueError("solver checkpoint missing gpu_policy.vram_reserve_gb")
    if int(gpu_policy["vram_reserve_gb"]) < 0:
        raise ValueError("gpu_policy.vram_reserve_gb must be >= 0")
    if state.config.slots < 1:
        raise ValueError("solver checkpoint must contain at least one slot")
    return payload


def evidence_from_gaussian_cloud(
    cloud: GaussianCloud,
    *,
    frame_index: int = 0,
    source: str | None = None,
    target_assignment: np.ndarray | None = None,
) -> ObjectEmergenceEvidence:
    return ObjectEmergenceEvidence(
        positions=positions(cloud),
        features=extract_features(cloud),
        target_assignment=target_assignment,
        frame_index=frame_index,
        source=source or "gaussian_cloud",
    )


def _object_emergence_evidence_from_assignment_batch(
    batch: AssignmentEvidenceBatch,
) -> ObjectEmergenceEvidence:
    checked = validate_assignment_evidence_batch(batch)
    return ObjectEmergenceEvidence(
        positions=checked.positions,
        features=checked.features,
        target_assignment=checked.target_assignment,
        frame_index=checked.frame_index,
        source=f"assignment_evidence_batch:{checked.source}",
    )


def object_id_targets_from_cloud(
    cloud: GaussianCloud,
    *,
    object_id_field: str = "object_id",
    slots: int | None = None,
) -> tuple[np.ndarray, dict[int, int]]:
    if object_id_field not in cloud.fields:
        raise ValueError(f"cloud is missing object id field: {object_id_field}")
    object_ids = np.asarray(cloud.vertices[object_id_field], dtype=np.int64)
    unique_ids = tuple(int(value) for value in np.unique(object_ids))
    if slots is None:
        resolved_slots = len(unique_ids)
    else:
        resolved_slots = int(slots)
        if resolved_slots < len(unique_ids):
            raise ValueError(
                f"slots={resolved_slots} cannot represent {len(unique_ids)} unique object ids"
            )
    if resolved_slots < 1:
        raise ValueError("slots must be >= 1")
    mapping = {object_id: index for index, object_id in enumerate(unique_ids)}
    targets = np.zeros((object_ids.shape[0], resolved_slots), dtype=np.float32)
    for row, object_id in enumerate(object_ids):
        targets[row, mapping[int(object_id)]] = 1.0
    return targets, mapping


def initialize_object_emergence_solver(
    *,
    slots: int,
    feature_dim: int,
    position_dim: int = 3,
    temperature: float = 1.0,
    seed: int = 0,
    scale: float = 0.025,
) -> ObjectEmergenceSolverState:
    if scale < 0:
        raise ValueError("scale must be >= 0")
    config = ObjectEmergenceSolverConfig(
        slots=slots,
        feature_dim=feature_dim,
        position_dim=position_dim,
        temperature=temperature,
    )
    _validate_solver_config(config)
    rng = np.random.default_rng(seed)
    feature_weights = rng.normal(0.0, scale, size=(feature_dim, slots)).astype(np.float32)
    position_weights = rng.normal(0.0, scale, size=(position_dim, slots)).astype(np.float32)
    bias = np.zeros(slots, dtype=np.float32)
    return ObjectEmergenceSolverState(
        config=config,
        feature_weights=feature_weights,
        position_weights=position_weights,
        bias=bias,
        source="random_linear_softmax_init",
    )


def train_object_emergence_solver(
    evidence_frames: Sequence[ObjectEmergenceEvidence],
    *,
    slots: int | None = None,
    initial_state: ObjectEmergenceSolverState | None = None,
    iterations: int = 40,
    learning_rate: float = 0.35,
    assignment_weight: float = 1.0,
    entropy_weight: float = 0.01,
    balance_weight: float = 0.05,
    temporal_weight: float = 0.02,
    finite_difference_epsilon: float = 1e-3,
    seed: int = 0,
    record_every: int | None = None,
) -> ObjectEmergenceSolverTrainingResult:
    frames = _validate_training_frames(evidence_frames)
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    if finite_difference_epsilon <= 0:
        raise ValueError("finite_difference_epsilon must be > 0")
    for name, weight in {
        "assignment_weight": assignment_weight,
        "entropy_weight": entropy_weight,
        "balance_weight": balance_weight,
        "temporal_weight": temporal_weight,
    }.items():
        if weight < 0:
            raise ValueError(f"{name} must be >= 0")
    if assignment_weight > 0 and not all(frame.target_assignment is not None for frame in frames):
        raise ValueError("assignment_weight requires every evidence frame to bind target_assignment")

    state = initial_state
    if state is None:
        resolved_slots = _resolve_training_slots(frames, slots=slots)
        state = initialize_object_emergence_solver(
            slots=resolved_slots,
            feature_dim=frames[0].feature_dim,
            seed=seed,
        )
    state = validate_object_emergence_solver_state(state)
    for frame in frames:
        validate_object_emergence_evidence(frame, slots=state.config.slots)
        if frame.feature_dim != state.config.feature_dim:
            raise ValueError("all evidence feature dimensions must match solver state")
    params = _pack_solver_state(state)
    record_stride = iterations if record_every is None else max(1, int(record_every))
    history: list[ObjectEmergenceSolverLoss] = []

    initial_eval = _evaluate_solver_training(
        frames,
        state,
        iteration=0,
        assignment_weight=assignment_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        temporal_weight=temporal_weight,
    )
    history.append(initial_eval.loss)
    for iteration in range(1, iterations + 1):
        gradient = _finite_difference_solver_gradient(
            params,
            state,
            frames,
            iteration=iteration - 1,
            epsilon=finite_difference_epsilon,
            assignment_weight=assignment_weight,
            entropy_weight=entropy_weight,
            balance_weight=balance_weight,
            temporal_weight=temporal_weight,
        )
        params = np.clip(params - learning_rate * gradient, -25.0, 25.0)
        state = _unpack_solver_state(
            params,
            template=state,
            step=iteration,
            source="trained_numpy_finite_difference",
        )
        if iteration == iterations or iteration % record_stride == 0:
            history.append(
                _evaluate_solver_training(
                    frames,
                    state,
                    iteration=iteration,
                    assignment_weight=assignment_weight,
                    entropy_weight=entropy_weight,
                    balance_weight=balance_weight,
                    temporal_weight=temporal_weight,
                ).loss
            )

    final_eval = _evaluate_solver_training(
        frames,
        state,
        iteration=iterations,
        assignment_weight=assignment_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        temporal_weight=temporal_weight,
    )
    if history[-1].iteration != iterations:
        history.append(final_eval.loss)
    else:
        history[-1] = final_eval.loss
    return ObjectEmergenceSolverTrainingResult(
        schema=OBJECT_EMERGENCE_TRAINING_SCHEMA,
        initial_state=initial_eval.state,
        final_state=state,
        initial_loss=initial_eval.loss,
        final_loss=final_eval.loss,
        history=tuple(history),
        predictions=final_eval.predictions,
        assignment_weight=float(assignment_weight),
        entropy_weight=float(entropy_weight),
        balance_weight=float(balance_weight),
        temporal_weight=float(temporal_weight),
        learning_rate=float(learning_rate),
        iterations=int(iterations),
        finite_difference_epsilon=float(finite_difference_epsilon),
    )


def predict_object_emergence_assignment(
    evidence: ObjectEmergenceEvidence,
    state: ObjectEmergenceSolverState,
) -> ObjectEmergenceAssignmentPrediction:
    positions_array, features_array, _target = validate_object_emergence_evidence(
        evidence,
        slots=state.config.slots,
    )
    state = validate_object_emergence_solver_state(state)
    if features_array.shape[1] != state.config.feature_dim:
        raise ValueError(
            f"evidence feature_dim {features_array.shape[1]} does not match solver feature_dim "
            f"{state.config.feature_dim}"
        )
    if positions_array.shape[1] != state.config.position_dim:
        raise ValueError(
            f"evidence position_dim {positions_array.shape[1]} does not match solver position_dim "
            f"{state.config.position_dim}"
        )

    logits = (
        state.config.feature_weight * (features_array @ state.feature_weights)
        + state.config.position_weight * (positions_array @ state.position_weights)
        + state.bias[None, :]
    ).astype(np.float32, copy=False)
    assignment = _softmax(logits, temperature=state.config.temperature)
    validate_assignment_matrix(assignment, evidence_count=positions_array.shape[0])
    entropy = _normalized_row_entropy(assignment)
    slot_mass = assignment.sum(axis=0).astype(np.float32, copy=False)
    top_slots = np.argmax(assignment, axis=1).astype(np.int32, copy=False)
    confidence = np.max(assignment, axis=1).astype(np.float32, copy=False)
    diagnostics = _prediction_diagnostics(assignment, entropy, slot_mass)
    return ObjectEmergenceAssignmentPrediction(
        assignment=assignment,
        logits=logits,
        top_slots=top_slots,
        confidence=confidence,
        slot_mass=slot_mass,
        mean_normalized_entropy=float(np.mean(entropy)) if entropy.size else 0.0,
        diagnostics=diagnostics,
        frame_index=int(evidence.frame_index),
        solver_step=int(state.step),
        source=state.source,
    )


def project_object_emergence_prediction(
    cloud: GaussianCloud,
    prediction: ObjectEmergenceAssignmentPrediction,
    *,
    evidence_features: np.ndarray | None = None,
) -> ObjectStateProjection:
    validate_assignment_matrix(prediction.assignment, evidence_count=cloud.count)
    return project_object_states(
        cloud,
        prediction.assignment,
        evidence_features=evidence_features,
    )


def validate_object_emergence_evidence(
    evidence: ObjectEmergenceEvidence,
    *,
    slots: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    positions_array = _float_matrix("positions", evidence.positions)
    features_array = _float_matrix("features", evidence.features)
    if positions_array.shape[1] != 3:
        raise ValueError("positions must have shape N x 3")
    if features_array.shape[0] != positions_array.shape[0]:
        raise ValueError("features rows must match positions")
    target = None
    if evidence.target_assignment is not None:
        target = validate_assignment_matrix(
            evidence.target_assignment,
            evidence_count=positions_array.shape[0],
        )
        if slots is not None and target.shape[1] != int(slots):
            raise ValueError(f"target_assignment has {target.shape[1]} slots for solver K={int(slots)}")
    return positions_array, features_array, target


def validate_object_emergence_solver_state(
    state: ObjectEmergenceSolverState,
) -> ObjectEmergenceSolverState:
    _validate_solver_config(state.config)
    feature_weights = _float_matrix("feature_weights", state.feature_weights)
    position_weights = _float_matrix("position_weights", state.position_weights)
    bias = _float_vector("bias", state.bias)
    if feature_weights.shape != (state.config.feature_dim, state.config.slots):
        raise ValueError("feature_weights shape must be feature_dim x slots")
    if position_weights.shape != (state.config.position_dim, state.config.slots):
        raise ValueError("position_weights shape must be position_dim x slots")
    if bias.shape != (state.config.slots,):
        raise ValueError("bias shape must match slots")
    if state.step < 0:
        raise ValueError("step must be >= 0")
    return state


def _validate_training_frames(
    evidence_frames: Sequence[ObjectEmergenceEvidence],
) -> tuple[ObjectEmergenceEvidence, ...]:
    frames = tuple(evidence_frames)
    if not frames:
        raise ValueError("evidence_frames must contain at least one frame")
    first_positions, first_features, _first_target = validate_object_emergence_evidence(frames[0])
    for frame in frames[1:]:
        positions_array, features_array, _target = validate_object_emergence_evidence(frame)
        if positions_array.shape[0] != first_positions.shape[0]:
            raise ValueError("all evidence frames must have the same evidence_count")
        if features_array.shape[1] != first_features.shape[1]:
            raise ValueError("all evidence frames must have the same feature_dim")
    return frames


def _resolve_training_slots(
    frames: tuple[ObjectEmergenceEvidence, ...],
    *,
    slots: int | None,
) -> int:
    target_slots = {
        int(frame.target_assignment.shape[1])
        for frame in frames
        if frame.target_assignment is not None
    }
    if slots is not None:
        resolved = int(slots)
        if resolved < 1:
            raise ValueError("slots must be >= 1")
        if target_slots and target_slots != {resolved}:
            raise ValueError("slots must match target_assignment slot count")
        return resolved
    if len(target_slots) == 1:
        return target_slots.pop()
    if len(target_slots) > 1:
        raise ValueError("all target_assignment slot counts must match")
    raise ValueError("slots is required when evidence has no target_assignment")


def _evaluate_solver_training(
    frames: tuple[ObjectEmergenceEvidence, ...],
    state: ObjectEmergenceSolverState,
    *,
    iteration: int,
    assignment_weight: float,
    entropy_weight: float,
    balance_weight: float,
    temporal_weight: float,
) -> _SolverTrainingEval:
    predictions = tuple(predict_object_emergence_assignment(frame, state) for frame in frames)
    assignment_loss = _assignment_cross_entropy_loss(frames, predictions)
    entropy_loss = float(np.mean([prediction.mean_normalized_entropy for prediction in predictions]))
    balance_loss = _balance_loss(predictions)
    temporal_loss = _temporal_centroid_loss(frames, predictions)
    total = (
        assignment_weight * assignment_loss
        + entropy_weight * entropy_loss
        + balance_weight * balance_loss
        + temporal_weight * temporal_loss
    )
    return _SolverTrainingEval(
        state=state,
        predictions=predictions,
        loss=ObjectEmergenceSolverLoss(
            iteration=int(iteration),
            total_loss=float(total),
            assignment_loss=float(assignment_loss),
            entropy_loss=float(entropy_loss),
            balance_loss=float(balance_loss),
            temporal_loss=float(temporal_loss),
        ),
    )


def _assignment_cross_entropy_loss(
    frames: tuple[ObjectEmergenceEvidence, ...],
    predictions: tuple[ObjectEmergenceAssignmentPrediction, ...],
) -> float:
    losses: list[float] = []
    for frame, prediction in zip(frames, predictions):
        if frame.target_assignment is None:
            continue
        target = validate_assignment_matrix(
            frame.target_assignment,
            evidence_count=prediction.evidence_count,
        )
        losses.append(float(-np.mean(np.sum(target * np.log(np.clip(prediction.assignment, _EPS, 1.0)), axis=1))))
    return float(np.mean(losses)) if losses else 0.0


def _balance_loss(predictions: tuple[ObjectEmergenceAssignmentPrediction, ...]) -> float:
    losses: list[float] = []
    for prediction in predictions:
        if prediction.evidence_count == 0:
            continue
        mass_fraction = prediction.slot_mass / max(float(prediction.evidence_count), _EPS)
        target = np.full(prediction.slots, 1.0 / float(prediction.slots), dtype=np.float32)
        losses.append(float(np.mean((mass_fraction - target) ** 2)))
    return float(np.mean(losses)) if losses else 0.0


def _temporal_centroid_loss(
    frames: tuple[ObjectEmergenceEvidence, ...],
    predictions: tuple[ObjectEmergenceAssignmentPrediction, ...],
) -> float:
    if len(frames) < 2:
        return 0.0
    losses: list[float] = []
    previous = _prediction_centroids(frames[0], predictions[0])
    for frame, prediction in zip(frames[1:], predictions[1:]):
        current = _prediction_centroids(frame, prediction)
        losses.append(float(np.mean((current - previous) ** 2)))
        previous = current
    return float(np.mean(losses)) if losses else 0.0


def _prediction_centroids(
    frame: ObjectEmergenceEvidence,
    prediction: ObjectEmergenceAssignmentPrediction,
) -> np.ndarray:
    positions_array, _features, _target = validate_object_emergence_evidence(
        frame,
        slots=prediction.slots,
    )
    mass = np.maximum(prediction.slot_mass.astype(np.float32), _EPS)
    return ((prediction.assignment.T @ positions_array) / mass[:, None]).astype(np.float32, copy=False)


def _finite_difference_solver_gradient(
    params: np.ndarray,
    template: ObjectEmergenceSolverState,
    frames: tuple[ObjectEmergenceEvidence, ...],
    *,
    iteration: int,
    epsilon: float,
    assignment_weight: float,
    entropy_weight: float,
    balance_weight: float,
    temporal_weight: float,
) -> np.ndarray:
    gradient = np.zeros_like(params, dtype=np.float32)
    for index in range(params.shape[0]):
        plus = params.copy()
        minus = params.copy()
        plus[index] += epsilon
        minus[index] -= epsilon
        plus_loss = _evaluate_solver_training(
            frames,
            _unpack_solver_state(
                plus,
                template=template,
                step=iteration,
                source=template.source,
            ),
            iteration=iteration,
            assignment_weight=assignment_weight,
            entropy_weight=entropy_weight,
            balance_weight=balance_weight,
            temporal_weight=temporal_weight,
        ).loss.total_loss
        minus_loss = _evaluate_solver_training(
            frames,
            _unpack_solver_state(
                minus,
                template=template,
                step=iteration,
                source=template.source,
            ),
            iteration=iteration,
            assignment_weight=assignment_weight,
            entropy_weight=entropy_weight,
            balance_weight=balance_weight,
            temporal_weight=temporal_weight,
        ).loss.total_loss
        gradient[index] = (plus_loss - minus_loss) / (2.0 * epsilon)
    return gradient


def _string_key_mapping(mapping: dict[int, int] | dict[str, int] | None) -> dict[str, int]:
    if mapping is None:
        return {}
    return {str(key): int(value) for key, value in mapping.items()}


def _pack_solver_state(state: ObjectEmergenceSolverState) -> np.ndarray:
    state = validate_object_emergence_solver_state(state)
    return np.concatenate(
        [
            state.feature_weights.reshape(-1),
            state.position_weights.reshape(-1),
            state.bias.reshape(-1),
        ]
    ).astype(np.float32, copy=False)


def _unpack_solver_state(
    params: np.ndarray,
    *,
    template: ObjectEmergenceSolverState,
    step: int,
    source: str,
) -> ObjectEmergenceSolverState:
    params = np.asarray(params, dtype=np.float32)
    feature_size = template.config.feature_dim * template.config.slots
    position_size = template.config.position_dim * template.config.slots
    expected_size = feature_size + position_size + template.config.slots
    if params.ndim != 1 or params.shape[0] != expected_size:
        raise ValueError("packed solver params have unexpected shape")
    feature_end = feature_size
    position_end = feature_end + position_size
    return replace(
        template,
        feature_weights=params[:feature_end].reshape(
            template.config.feature_dim,
            template.config.slots,
        ).astype(np.float32, copy=True),
        position_weights=params[feature_end:position_end].reshape(
            template.config.position_dim,
            template.config.slots,
        ).astype(np.float32, copy=True),
        bias=params[position_end:].astype(np.float32, copy=True),
        step=int(step),
        source=source,
    )


def _validate_solver_config(config: ObjectEmergenceSolverConfig) -> None:
    if config.slots < 1:
        raise ValueError("slots must be >= 1")
    if config.feature_dim < 1:
        raise ValueError("feature_dim must be >= 1")
    if config.position_dim < 1:
        raise ValueError("position_dim must be >= 1")
    if config.temperature <= 0:
        raise ValueError("temperature must be > 0")
    if config.feature_weight < 0:
        raise ValueError("feature_weight must be >= 0")
    if config.position_weight < 0:
        raise ValueError("position_weight must be >= 0")


def _float_matrix(name: str, values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError(f"{name} must be a 2D matrix")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _float_vector(name: str, values: np.ndarray) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    if vector.ndim != 1:
        raise ValueError(f"{name} must be a 1D vector")
    if not np.isfinite(vector).all():
        raise ValueError(f"{name} must contain only finite values")
    return vector


def _softmax(logits: np.ndarray, *, temperature: float) -> np.ndarray:
    scaled = logits / max(float(temperature), _EPS)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32, copy=False)


def _normalized_row_entropy(assignment: np.ndarray) -> np.ndarray:
    if assignment.shape[0] == 0:
        return np.zeros(0, dtype=np.float32)
    if assignment.shape[1] <= 1:
        return np.zeros(assignment.shape[0], dtype=np.float32)
    entropy = -np.sum(assignment * np.log(np.clip(assignment, _EPS, 1.0)), axis=1)
    return (entropy / np.log(float(assignment.shape[1]))).astype(np.float32, copy=False)


def _prediction_diagnostics(
    assignment: np.ndarray,
    normalized_entropy: np.ndarray,
    slot_mass: np.ndarray,
) -> tuple[str, ...]:
    diagnostics: list[str] = []
    if normalized_entropy.size and float(np.mean(normalized_entropy)) >= 0.8:
        diagnostics.append("mixed_assignment")
    total_mass = float(np.sum(slot_mass))
    if total_mass > _EPS:
        mass_fraction = slot_mass / total_mass
        if float(np.max(mass_fraction)) >= 0.9:
            diagnostics.append("slot_collapse_risk")
    inactive = tuple(int(index) for index, mass in enumerate(slot_mass) if float(mass) <= _EPS)
    if inactive:
        diagnostics.append("inactive_slots:" + ",".join(str(index) for index in inactive))
    if not diagnostics:
        diagnostics.append("ok")
    return tuple(diagnostics)
