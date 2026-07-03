from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

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

