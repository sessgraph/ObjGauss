from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.assignment_evidence import (
    AssignmentEvidenceBatch,
    validate_assignment_evidence_batch,
)
from objgauss.core.assignment_losses import (
    AssignmentLossV2Result,
    assignment_loss_v2_breakdown,
)
from objgauss.core.object_state import validate_assignment_matrix

ASSIGNMENT_SOLVER_V2_STATE_SCHEMA = "objgauss-assignment-solver-state-v2"
ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA = "objgauss-assignment-prediction-v2"
ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA = "objgauss-assignment-solver-v2-training-v1"
ASSIGNMENT_SOLVER_V2_COST_TERMS = ("feature", "position", "slot_bias")
_EPS = 1e-8


@dataclass(frozen=True)
class AssignmentSolverV2Config:
    slots: int
    feature_dim: int
    position_dim: int = 3
    temperature: float = 1.0
    feature_weight: float = 1.0
    position_weight: float = 1.0
    solver_family: str = "cost-softmax-assignment-v2"
    cost_terms: tuple[str, ...] = ASSIGNMENT_SOLVER_V2_COST_TERMS
    balance_policy: str = "loss-only-v1"
    temporal_policy: str = "disabled"
    matching_policy: str = "disabled"

    def as_dict(self) -> dict[str, Any]:
        config = validate_assignment_solver_v2_config(self)
        return {
            "slots": int(config.slots),
            "feature_dim": int(config.feature_dim),
            "position_dim": int(config.position_dim),
            "temperature": float(config.temperature),
            "feature_weight": float(config.feature_weight),
            "position_weight": float(config.position_weight),
            "solver_family": config.solver_family,
            "cost_terms": list(config.cost_terms),
            "balance_policy": config.balance_policy,
            "temporal_policy": config.temporal_policy,
            "matching_policy": config.matching_policy,
        }


@dataclass(frozen=True)
class AssignmentSolverV2State:
    config: AssignmentSolverV2Config
    feature_centers: np.ndarray
    position_centers: np.ndarray
    slot_bias: np.ndarray
    step: int = 0
    source: str = "initialized"
    schema: str = ASSIGNMENT_SOLVER_V2_STATE_SCHEMA

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        state = validate_assignment_solver_v2_state(self)
        payload: dict[str, Any] = {
            "schema": state.schema,
            "source": state.source,
            "step": int(state.step),
            "config": state.config.as_dict(),
            "array_shapes": {
                "feature_centers": list(state.feature_centers.shape),
                "position_centers": list(state.position_centers.shape),
                "slot_bias": list(state.slot_bias.shape),
            },
            "trained_fields": [
                "feature_centers",
                "position_centers",
                "slot_bias",
            ],
            "disabled_fields": [
                "temporal_memory",
                "matching_policy",
                "dynamic_k",
                "renderer_loss",
            ],
        }
        if include_arrays:
            payload["arrays"] = {
                "feature_centers": np.round(state.feature_centers, 6).tolist(),
                "position_centers": np.round(state.position_centers, 6).tolist(),
                "slot_bias": np.round(state.slot_bias, 6).tolist(),
            }
        return payload


@dataclass(frozen=True)
class AssignmentSolverV2Prediction:
    assignment: np.ndarray
    cost: np.ndarray
    logits: np.ndarray
    slot_mass: np.ndarray
    confidence: np.ndarray
    mean_normalized_entropy: float
    effective_slots: float
    diagnostics: tuple[str, ...]
    frame_index: int
    solver_step: int
    source: str
    schema: str = ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA

    @property
    def evidence_count(self) -> int:
        return int(self.assignment.shape[0])

    @property
    def slots(self) -> int:
        return int(self.assignment.shape[1])

    def as_dict(
        self,
        *,
        include_assignment: bool = False,
        include_cost: bool = False,
    ) -> dict[str, Any]:
        validate_assignment_matrix(self.assignment)
        payload: dict[str, Any] = {
            "schema": self.schema,
            "source": self.source,
            "frame_index": int(self.frame_index),
            "solver_step": int(self.solver_step),
            "evidence_count": self.evidence_count,
            "slots": self.slots,
            "slot_mass": np.round(self.slot_mass, 6).tolist(),
            "confidence": np.round(self.confidence, 6).tolist(),
            "mean_normalized_entropy": float(self.mean_normalized_entropy),
            "effective_slots": float(self.effective_slots),
            "diagnostics": list(self.diagnostics),
        }
        if include_assignment:
            payload["assignment"] = np.round(self.assignment, 6).tolist()
        if include_cost:
            payload["cost"] = np.round(self.cost, 6).tolist()
            payload["logits"] = np.round(self.logits, 6).tolist()
        return payload


@dataclass(frozen=True)
class AssignmentSolverV2LossRecord:
    iteration: int
    total_loss: float
    cluster_loss: float
    entropy_loss: float
    balance_loss: float
    supervised_loss: float
    temporal_loss: float = 0.0
    matching_loss: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        return {
            "iteration": int(self.iteration),
            "total_loss": float(self.total_loss),
            "cluster_loss": float(self.cluster_loss),
            "entropy_loss": float(self.entropy_loss),
            "balance_loss": float(self.balance_loss),
            "supervised_loss": float(self.supervised_loss),
            "temporal_loss": float(self.temporal_loss),
            "matching_loss": float(self.matching_loss),
        }


@dataclass(frozen=True)
class AssignmentSolverV2TrainingResult:
    initial_state: AssignmentSolverV2State
    final_state: AssignmentSolverV2State
    initial_loss: AssignmentSolverV2LossRecord
    final_loss: AssignmentSolverV2LossRecord
    history: tuple[AssignmentSolverV2LossRecord, ...]
    predictions: tuple[AssignmentSolverV2Prediction, ...]
    iterations: int
    learning_rate: float
    cluster_weight: float
    entropy_weight: float
    balance_weight: float
    supervised_weight: float
    schema: str = ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA

    def as_dict(
        self,
        *,
        include_state_arrays: bool = False,
        include_assignments: bool = False,
        include_cost: bool = False,
    ) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "kind": "assignment_solver_v2_training",
            "solver_family": "cost-softmax-assignment-v2",
            "iterations": int(self.iterations),
            "learning_rate": float(self.learning_rate),
            "weights": {
                "cluster": float(self.cluster_weight),
                "entropy": float(self.entropy_weight),
                "balance": float(self.balance_weight),
                "supervised": float(self.supervised_weight),
                "temporal": 0.0,
                "matching": 0.0,
            },
            "fixed_k": True,
            "renderer_loss": "not_used",
            "dynamic_k": "disabled",
            "temporal_policy": "disabled",
            "matching_policy": "disabled",
            "initial_loss": self.initial_loss.as_dict(),
            "final_loss": self.final_loss.as_dict(),
            "loss_decreased": bool(self.final_loss.total_loss < self.initial_loss.total_loss),
            "supervised_loss_decreased": bool(
                self.final_loss.supervised_loss < self.initial_loss.supervised_loss
            ),
            "initial_state": self.initial_state.as_dict(include_arrays=include_state_arrays),
            "final_state": self.final_state.as_dict(include_arrays=include_state_arrays),
            "history": [record.as_dict() for record in self.history],
            "predictions": [
                prediction.as_dict(
                    include_assignment=include_assignments,
                    include_cost=include_cost,
                )
                for prediction in self.predictions
            ],
            "non_goals": {
                "uses_gpu": False,
                "uses_renderer_loss": False,
                "uses_temporal_matching_loss": False,
                "mutates_dynamic_k": False,
                "uses_slot_attention": False,
                "uses_sinkhorn_or_ot": False,
            },
        }


@dataclass(frozen=True)
class _TrainingEval:
    predictions: tuple[AssignmentSolverV2Prediction, ...]
    loss_result: AssignmentLossV2Result
    loss_record: AssignmentSolverV2LossRecord


@dataclass(frozen=True)
class _TrainingGradient:
    feature_centers: np.ndarray
    position_centers: np.ndarray
    slot_bias: np.ndarray


def initialize_assignment_solver_v2(
    *,
    slots: int,
    feature_dim: int,
    position_dim: int = 3,
    temperature: float = 1.0,
    seed: int = 0,
    scale: float = 0.05,
) -> AssignmentSolverV2State:
    if scale < 0:
        raise ValueError("scale must be >= 0")
    config = AssignmentSolverV2Config(
        slots=int(slots),
        feature_dim=int(feature_dim),
        position_dim=int(position_dim),
        temperature=float(temperature),
    )
    validate_assignment_solver_v2_config(config)
    rng = np.random.default_rng(seed)
    feature_centers = rng.normal(0.0, scale, size=(config.slots, config.feature_dim)).astype(np.float32)
    position_centers = rng.normal(0.0, scale, size=(config.slots, config.position_dim)).astype(np.float32)
    slot_bias = np.zeros(config.slots, dtype=np.float32)
    return validate_assignment_solver_v2_state(
        AssignmentSolverV2State(
            config=config,
            feature_centers=feature_centers,
            position_centers=position_centers,
            slot_bias=slot_bias,
            source="random_cost_softmax_assignment_v2_init",
        )
    )


def assignment_solver_v2_state_from_dict(payload: dict[str, Any]) -> AssignmentSolverV2State:
    if not isinstance(payload, dict):
        raise TypeError("assignment solver v2 state payload must be a dict")
    if payload.get("schema") != ASSIGNMENT_SOLVER_V2_STATE_SCHEMA:
        raise ValueError(f"unsupported assignment solver v2 state schema: {payload.get('schema')}")
    config_payload = payload.get("config")
    arrays_payload = payload.get("arrays")
    if not isinstance(config_payload, dict):
        raise ValueError("assignment solver v2 state missing config")
    if not isinstance(arrays_payload, dict):
        raise ValueError("assignment solver v2 state missing arrays")
    config = AssignmentSolverV2Config(
        slots=int(config_payload.get("slots")),
        feature_dim=int(config_payload.get("feature_dim")),
        position_dim=int(config_payload.get("position_dim", 3)),
        temperature=float(config_payload.get("temperature", 1.0)),
        feature_weight=float(config_payload.get("feature_weight", 1.0)),
        position_weight=float(config_payload.get("position_weight", 1.0)),
        solver_family=str(config_payload.get("solver_family", "cost-softmax-assignment-v2")),
        cost_terms=tuple(config_payload.get("cost_terms", ASSIGNMENT_SOLVER_V2_COST_TERMS)),
        balance_policy=str(config_payload.get("balance_policy", "loss-only-v1")),
        temporal_policy=str(config_payload.get("temporal_policy", "disabled")),
        matching_policy=str(config_payload.get("matching_policy", "disabled")),
    )
    state = AssignmentSolverV2State(
        config=config,
        feature_centers=np.asarray(arrays_payload.get("feature_centers"), dtype=np.float32),
        position_centers=np.asarray(arrays_payload.get("position_centers"), dtype=np.float32),
        slot_bias=np.asarray(arrays_payload.get("slot_bias"), dtype=np.float32),
        step=int(payload.get("step", 0)),
        source=str(payload.get("source", "state_dict")),
        schema=str(payload.get("schema")),
    )
    return validate_assignment_solver_v2_state(state)


def predict_assignment_solver_v2(
    evidence: AssignmentEvidenceBatch,
    state: AssignmentSolverV2State,
) -> AssignmentSolverV2Prediction:
    batch = validate_assignment_evidence_batch(evidence)
    state = validate_assignment_solver_v2_state(state)
    if batch.feature_dim != state.config.feature_dim:
        raise ValueError("evidence feature_dim must match assignment solver v2 state")
    if batch.positions.shape[1] != state.config.position_dim:
        raise ValueError("evidence position_dim must match assignment solver v2 state")
    cost = _assignment_cost(batch, state)
    logits = (-cost).astype(np.float32, copy=False)
    assignment = _softmax(logits, temperature=state.config.temperature)
    validate_assignment_matrix(assignment, evidence_count=batch.evidence_count)
    entropy = _normalized_row_entropy(assignment)
    slot_mass = assignment.sum(axis=0).astype(np.float32, copy=False)
    confidence = np.max(assignment, axis=1).astype(np.float32, copy=False)
    mass_fraction = slot_mass / max(float(batch.evidence_count), _EPS)
    effective_slots = float(np.exp(-np.sum(mass_fraction * np.log(np.clip(mass_fraction, _EPS, 1.0)))))
    diagnostics = _prediction_diagnostics(assignment, entropy, slot_mass)
    return AssignmentSolverV2Prediction(
        assignment=assignment,
        cost=cost,
        logits=logits,
        slot_mass=slot_mass,
        confidence=confidence,
        mean_normalized_entropy=float(np.mean(entropy)) if entropy.size else 0.0,
        effective_slots=effective_slots,
        diagnostics=diagnostics,
        frame_index=batch.frame_index,
        solver_step=state.step,
        source=state.source,
    )


def train_assignment_solver_v2(
    evidence_batches: Sequence[AssignmentEvidenceBatch],
    *,
    slots: int | None = None,
    initial_state: AssignmentSolverV2State | None = None,
    iterations: int = 60,
    learning_rate: float = 0.2,
    cluster_weight: float = 0.1,
    entropy_weight: float = 0.01,
    balance_weight: float = 0.05,
    supervised_weight: float = 1.0,
    seed: int = 0,
    record_every: int | None = None,
) -> AssignmentSolverV2TrainingResult:
    batches = _validate_training_batches(evidence_batches)
    if iterations < 1:
        raise ValueError("iterations must be >= 1")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be > 0")
    _validate_training_weights(
        cluster_weight=cluster_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        supervised_weight=supervised_weight,
    )
    if supervised_weight > 0 and not all(batch.target_assignment is not None for batch in batches):
        raise ValueError("supervised_weight requires every evidence batch to bind target_assignment")
    state = initial_state
    if state is None:
        resolved_slots = _resolve_training_slots(batches, slots=slots)
        state = initialize_assignment_solver_v2(
            slots=resolved_slots,
            feature_dim=batches[0].feature_dim,
            position_dim=batches[0].positions.shape[1],
            seed=seed,
        )
    state = validate_assignment_solver_v2_state(state)
    for batch in batches:
        if batch.feature_dim != state.config.feature_dim:
            raise ValueError("all evidence feature dimensions must match assignment solver v2 state")
        if batch.positions.shape[1] != state.config.position_dim:
            raise ValueError("all evidence position dimensions must match assignment solver v2 state")
        if batch.target_assignment is not None and batch.target_assignment.shape[1] != state.config.slots:
            raise ValueError("target_assignment slots must match assignment solver v2 state")
    record_stride = iterations if record_every is None else max(1, int(record_every))
    history: list[AssignmentSolverV2LossRecord] = []
    initial_eval = _evaluate_training_state(
        batches,
        state,
        iteration=0,
        cluster_weight=cluster_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        supervised_weight=supervised_weight,
    )
    history.append(initial_eval.loss_record)
    current_state = state
    for iteration in range(1, iterations + 1):
        gradient = _training_gradient(
            batches,
            current_state,
            cluster_weight=cluster_weight,
            entropy_weight=entropy_weight,
            balance_weight=balance_weight,
            supervised_weight=supervised_weight,
        )
        current_state = _apply_gradient(
            current_state,
            gradient,
            learning_rate=learning_rate,
            step=iteration,
        )
        if iteration == iterations or iteration % record_stride == 0:
            history.append(
                _evaluate_training_state(
                    batches,
                    current_state,
                    iteration=iteration,
                    cluster_weight=cluster_weight,
                    entropy_weight=entropy_weight,
                    balance_weight=balance_weight,
                    supervised_weight=supervised_weight,
                ).loss_record
            )
    final_eval = _evaluate_training_state(
        batches,
        current_state,
        iteration=iterations,
        cluster_weight=cluster_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        supervised_weight=supervised_weight,
    )
    if history[-1].iteration == iterations:
        history[-1] = final_eval.loss_record
    else:
        history.append(final_eval.loss_record)
    return AssignmentSolverV2TrainingResult(
        initial_state=state,
        final_state=current_state,
        initial_loss=initial_eval.loss_record,
        final_loss=final_eval.loss_record,
        history=tuple(history),
        predictions=final_eval.predictions,
        iterations=int(iterations),
        learning_rate=float(learning_rate),
        cluster_weight=float(cluster_weight),
        entropy_weight=float(entropy_weight),
        balance_weight=float(balance_weight),
        supervised_weight=float(supervised_weight),
    )


def validate_assignment_solver_v2_config(
    config: AssignmentSolverV2Config,
) -> AssignmentSolverV2Config:
    if not isinstance(config, AssignmentSolverV2Config):
        raise TypeError("config must be AssignmentSolverV2Config")
    if int(config.slots) < 1:
        raise ValueError("slots must be >= 1")
    if int(config.feature_dim) < 1:
        raise ValueError("feature_dim must be >= 1")
    if int(config.position_dim) != 3:
        raise ValueError("position_dim must be 3")
    if float(config.temperature) <= 0:
        raise ValueError("temperature must be > 0")
    if float(config.feature_weight) < 0 or float(config.position_weight) < 0:
        raise ValueError("feature_weight and position_weight must be >= 0")
    if config.solver_family != "cost-softmax-assignment-v2":
        raise ValueError("solver_family must be cost-softmax-assignment-v2")
    if tuple(config.cost_terms) != ASSIGNMENT_SOLVER_V2_COST_TERMS:
        raise ValueError("cost_terms must be feature, position, slot_bias")
    if config.balance_policy != "loss-only-v1":
        raise ValueError("balance_policy must be loss-only-v1")
    if config.temporal_policy != "disabled":
        raise ValueError("temporal_policy must be disabled")
    if config.matching_policy != "disabled":
        raise ValueError("matching_policy must be disabled")
    return config


def validate_assignment_solver_v2_state(
    state: AssignmentSolverV2State,
) -> AssignmentSolverV2State:
    if not isinstance(state, AssignmentSolverV2State):
        raise TypeError("state must be AssignmentSolverV2State")
    if state.schema != ASSIGNMENT_SOLVER_V2_STATE_SCHEMA:
        raise ValueError(f"unsupported assignment solver v2 state schema: {state.schema}")
    config = validate_assignment_solver_v2_config(state.config)
    feature_centers = _float_matrix("feature_centers", state.feature_centers)
    position_centers = _float_matrix("position_centers", state.position_centers)
    slot_bias = _float_vector("slot_bias", state.slot_bias)
    if feature_centers.shape != (config.slots, config.feature_dim):
        raise ValueError("feature_centers shape must be slots x feature_dim")
    if position_centers.shape != (config.slots, config.position_dim):
        raise ValueError("position_centers shape must be slots x position_dim")
    if slot_bias.shape != (config.slots,):
        raise ValueError("slot_bias shape must match slots")
    if int(state.step) < 0:
        raise ValueError("step must be >= 0")
    return AssignmentSolverV2State(
        config=config,
        feature_centers=feature_centers,
        position_centers=position_centers,
        slot_bias=slot_bias,
        step=int(state.step),
        source=str(state.source),
        schema=state.schema,
    )


def validate_assignment_solver_v2_training_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("assignment solver v2 training summary must be a dict")
    if payload.get("schema") != ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA:
        raise ValueError(f"unsupported assignment solver v2 training schema: {payload.get('schema')}")
    if payload.get("kind") != "assignment_solver_v2_training":
        raise ValueError("assignment solver v2 training kind must be assignment_solver_v2_training")
    for key in ("initial_loss", "final_loss", "history", "predictions", "weights", "non_goals"):
        if key not in payload:
            raise ValueError(f"assignment solver v2 training summary missing {key}")
    for loss_key in ("total_loss", "cluster_loss", "entropy_loss", "balance_loss", "supervised_loss"):
        float(payload["initial_loss"][loss_key])
        float(payload["final_loss"][loss_key])
    if not isinstance(payload["history"], list) or not payload["history"]:
        raise ValueError("assignment solver v2 training history must be non-empty")
    if payload.get("renderer_loss") != "not_used":
        raise ValueError("assignment solver v2 training must not use renderer loss")
    if payload.get("dynamic_k") != "disabled":
        raise ValueError("assignment solver v2 training must keep dynamic-K disabled")
    non_goals = payload["non_goals"]
    if (
        non_goals.get("uses_gpu")
        or non_goals.get("uses_renderer_loss")
        or non_goals.get("uses_temporal_matching_loss")
        or non_goals.get("mutates_dynamic_k")
        or non_goals.get("uses_slot_attention")
        or non_goals.get("uses_sinkhorn_or_ot")
    ):
        raise ValueError("assignment solver v2 training summary violates non-goals")
    return payload


def _validate_training_batches(
    evidence_batches: Sequence[AssignmentEvidenceBatch],
) -> tuple[AssignmentEvidenceBatch, ...]:
    batches = tuple(validate_assignment_evidence_batch(batch) for batch in evidence_batches)
    if not batches:
        raise ValueError("evidence_batches must contain at least one batch")
    feature_dim = batches[0].feature_dim
    position_dim = batches[0].positions.shape[1]
    for batch in batches[1:]:
        if batch.feature_dim != feature_dim:
            raise ValueError("all evidence batches must have the same feature_dim")
        if batch.positions.shape[1] != position_dim:
            raise ValueError("all evidence batches must have the same position_dim")
    return batches


def _resolve_training_slots(
    batches: tuple[AssignmentEvidenceBatch, ...],
    *,
    slots: int | None,
) -> int:
    target_slots = {
        int(batch.target_assignment.shape[1])
        for batch in batches
        if batch.target_assignment is not None
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


def _validate_training_weights(
    *,
    cluster_weight: float,
    entropy_weight: float,
    balance_weight: float,
    supervised_weight: float,
) -> None:
    for name, value in {
        "cluster_weight": cluster_weight,
        "entropy_weight": entropy_weight,
        "balance_weight": balance_weight,
        "supervised_weight": supervised_weight,
    }.items():
        if float(value) < 0:
            raise ValueError(f"{name} must be >= 0")


def _evaluate_training_state(
    batches: tuple[AssignmentEvidenceBatch, ...],
    state: AssignmentSolverV2State,
    *,
    iteration: int,
    cluster_weight: float,
    entropy_weight: float,
    balance_weight: float,
    supervised_weight: float,
) -> _TrainingEval:
    predictions = tuple(predict_assignment_solver_v2(batch, state) for batch in batches)
    loss_result = assignment_loss_v2_breakdown(
        tuple(prediction.assignment for prediction in predictions),
        cluster_costs=tuple(prediction.cost for prediction in predictions),
        target_assignments=tuple(batch.target_assignment for batch in batches),
        cluster_weight=cluster_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        supervised_weight=supervised_weight,
    )
    return _TrainingEval(
        predictions=predictions,
        loss_result=loss_result,
        loss_record=_loss_record(iteration, loss_result),
    )


def _training_gradient(
    batches: tuple[AssignmentEvidenceBatch, ...],
    state: AssignmentSolverV2State,
    *,
    cluster_weight: float,
    entropy_weight: float,
    balance_weight: float,
    supervised_weight: float,
) -> _TrainingGradient:
    eval_result = _evaluate_training_state(
        batches,
        state,
        iteration=state.step,
        cluster_weight=cluster_weight,
        entropy_weight=entropy_weight,
        balance_weight=balance_weight,
        supervised_weight=supervised_weight,
    )
    feature_grad = np.zeros_like(state.feature_centers, dtype=np.float32)
    position_grad = np.zeros_like(state.position_centers, dtype=np.float32)
    bias_grad = np.zeros_like(state.slot_bias, dtype=np.float32)
    frame_count = max(len(batches), 1)
    for batch, prediction, assignment_grad in zip(
        batches,
        eval_result.predictions,
        eval_result.loss_result.gradients,
        strict=True,
    ):
        cost_grad = _cost_gradient_from_assignment_gradient(
            assignment_grad,
            prediction.assignment,
            temperature=state.config.temperature,
        )
        if cluster_weight > 0:
            cost_grad = cost_grad + (
                cluster_weight
                * prediction.assignment
                / max(float(batch.evidence_count), _EPS)
                / frame_count
            )
        frame_feature_grad, frame_position_grad, frame_bias_grad = _cost_parameter_gradient(
            batch,
            state,
            cost_grad,
        )
        feature_grad += frame_feature_grad
        position_grad += frame_position_grad
        bias_grad += frame_bias_grad
    return _TrainingGradient(
        feature_centers=feature_grad.astype(np.float32, copy=False),
        position_centers=position_grad.astype(np.float32, copy=False),
        slot_bias=bias_grad.astype(np.float32, copy=False),
    )


def _apply_gradient(
    state: AssignmentSolverV2State,
    gradient: _TrainingGradient,
    *,
    learning_rate: float,
    step: int,
) -> AssignmentSolverV2State:
    return validate_assignment_solver_v2_state(
        AssignmentSolverV2State(
            config=state.config,
            feature_centers=np.clip(
                state.feature_centers - learning_rate * gradient.feature_centers,
                -10.0,
                10.0,
            ).astype(np.float32, copy=False),
            position_centers=np.clip(
                state.position_centers - learning_rate * gradient.position_centers,
                -10.0,
                10.0,
            ).astype(np.float32, copy=False),
            slot_bias=np.clip(
                state.slot_bias - learning_rate * gradient.slot_bias,
                -10.0,
                10.0,
            ).astype(np.float32, copy=False),
            step=int(step),
            source="trained_cost_softmax_assignment_v2",
        )
    )


def _assignment_cost(
    batch: AssignmentEvidenceBatch,
    state: AssignmentSolverV2State,
) -> np.ndarray:
    features = _row_l2_normalize(batch.features)
    feature_centers = _row_l2_normalize(state.feature_centers)
    positions = _row_l2_normalize(batch.positions)
    position_centers = _row_l2_normalize(state.position_centers)
    feature_delta = features[:, None, :] - feature_centers[None, :, :]
    position_delta = positions[:, None, :] - position_centers[None, :, :]
    cost = (
        state.config.feature_weight * np.sum(feature_delta * feature_delta, axis=2)
        + state.config.position_weight * np.sum(position_delta * position_delta, axis=2)
        + state.slot_bias[None, :]
    )
    return cost.astype(np.float32, copy=False)


def _cost_gradient_from_assignment_gradient(
    assignment_gradient: np.ndarray,
    assignment: np.ndarray,
    *,
    temperature: float,
) -> np.ndarray:
    row_dot = np.sum(assignment_gradient * assignment, axis=1, keepdims=True)
    scaled_logit_grad = assignment * (assignment_gradient - row_dot)
    logit_grad = scaled_logit_grad / float(temperature)
    return (-logit_grad).astype(np.float32, copy=False)


def _cost_parameter_gradient(
    batch: AssignmentEvidenceBatch,
    state: AssignmentSolverV2State,
    cost_gradient: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features = _row_l2_normalize(batch.features)
    feature_centers = _row_l2_normalize(state.feature_centers)
    positions = _row_l2_normalize(batch.positions)
    position_centers = _row_l2_normalize(state.position_centers)
    feature_grad = np.zeros_like(state.feature_centers, dtype=np.float32)
    position_grad = np.zeros_like(state.position_centers, dtype=np.float32)
    for slot in range(state.config.slots):
        weights = cost_gradient[:, slot][:, None]
        feature_center_grad_normalized = (
            2.0
            * state.config.feature_weight
            * np.sum(weights * (feature_centers[slot][None, :] - features), axis=0)
        )
        position_center_grad_normalized = (
            2.0
            * state.config.position_weight
            * np.sum(weights * (position_centers[slot][None, :] - positions), axis=0)
        )
        feature_grad[slot] = _row_normalize_backward(
            state.feature_centers[slot],
            feature_center_grad_normalized,
        )
        position_grad[slot] = _row_normalize_backward(
            state.position_centers[slot],
            position_center_grad_normalized,
        )
    bias_grad = np.sum(cost_gradient, axis=0).astype(np.float32, copy=False)
    return feature_grad, position_grad, bias_grad


def _loss_record(iteration: int, loss: AssignmentLossV2Result) -> AssignmentSolverV2LossRecord:
    return AssignmentSolverV2LossRecord(
        iteration=int(iteration),
        total_loss=float(loss.total_loss),
        cluster_loss=float(loss.cluster_loss),
        entropy_loss=float(loss.entropy_loss),
        balance_loss=float(loss.balance_loss),
        supervised_loss=float(loss.supervised_loss),
        temporal_loss=float(loss.temporal_loss),
        matching_loss=float(loss.matching_loss),
    )


def _prediction_diagnostics(
    assignment: np.ndarray,
    entropy: np.ndarray,
    slot_mass: np.ndarray,
) -> tuple[str, ...]:
    diagnostics: list[str] = []
    if float(np.mean(entropy)) > 0.6:
        diagnostics.append("high_assignment_entropy")
    if np.any(slot_mass < 1e-3):
        diagnostics.append("inactive_slot")
    if np.max(slot_mass) / max(float(np.sum(slot_mass)), _EPS) > 0.9:
        diagnostics.append("slot_collapse_risk")
    return tuple(diagnostics)


def _softmax(logits: np.ndarray, *, temperature: float) -> np.ndarray:
    scaled = logits / max(float(temperature), _EPS)
    scaled = scaled - np.max(scaled, axis=1, keepdims=True)
    exp = np.exp(scaled)
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32, copy=False)


def _normalized_row_entropy(assignment: np.ndarray) -> np.ndarray:
    if assignment.shape[1] <= 1:
        return np.zeros(assignment.shape[0], dtype=np.float32)
    clipped = np.clip(assignment, _EPS, 1.0)
    entropy = -np.sum(assignment * np.log(clipped), axis=1)
    return (entropy / np.log(float(assignment.shape[1]))).astype(np.float32, copy=False)


def _row_l2_normalize(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return (values / np.maximum(norms, _EPS)).astype(np.float32, copy=False)


def _row_normalize_backward(raw: np.ndarray, grad_normalized: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(raw))
    if norm <= _EPS:
        return grad_normalized.astype(np.float32, copy=False)
    normalized = raw / norm
    projected = grad_normalized - normalized * float(np.dot(normalized, grad_normalized))
    return (projected / norm).astype(np.float32, copy=False)


def _float_matrix(label: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{label} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    return array.astype(np.float32, copy=False)


def _float_vector(label: str, value: np.ndarray) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a 1D array")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    return array.astype(np.float32, copy=False)
