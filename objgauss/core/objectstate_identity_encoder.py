from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.objectstate_identity_gate import (
    ObjectStateIdentityRow,
    validate_objectstate_identity_row,
)

OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA = (
    "objgauss-objectstate-identity-encoder-training-v1"
)
OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA = "objgauss-objectstate-identity-encoder-state-v1"
_TRAINING_STATUS_PASS = "objectstate_identity_encoder_training_pass"
_TRAINING_STATUS_FAIL = "objectstate_identity_encoder_training_fail"
_FEATURE_SOURCES = ("appearance", "appearance_geometry", "candidate_objectstate")
_EPS = 1e-8


@dataclass(frozen=True)
class ObjectStateIdentityEncoderConfig:
    input_dim: int
    embedding_dim: int = 3
    margin: float = 1.0
    learning_rate: float = 0.2
    weight_decay: float = 0.001
    feature_source: str = "appearance"
    seed: int = 0
    schema: str = OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        config = validate_objectstate_identity_encoder_config(self)
        return {
            "schema": config.schema,
            "embedding_dim": int(config.embedding_dim),
            "input_dim": int(config.input_dim),
            "margin": float(config.margin),
            "learning_rate": float(config.learning_rate),
            "weight_decay": float(config.weight_decay),
            "feature_source": config.feature_source,
            "seed": int(config.seed),
        }


@dataclass(frozen=True)
class ObjectStateIdentityEncoderState:
    config: ObjectStateIdentityEncoderConfig
    weights: np.ndarray
    bias: np.ndarray
    step: int = 0
    source: str = "contrastive_identity_training"
    schema: str = OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA

    def encode(self, rows: Sequence[ObjectStateIdentityRow]) -> np.ndarray:
        checked = validate_objectstate_identity_encoder_state(self)
        features = objectstate_identity_encoder_features(
            rows,
            feature_source=checked.config.feature_source,
        )
        if features.shape[1] != checked.config.input_dim:
            raise ValueError("encoder input feature dimension does not match state config")
        return (features @ checked.weights + checked.bias).astype(np.float32, copy=False)

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        state = validate_objectstate_identity_encoder_state(self)
        payload = {
            "schema": state.schema,
            "kind": "objectstate_identity_encoder_state",
            "source": state.source,
            "step": int(state.step),
            "config": state.config.as_dict(),
            "shapes": {
                "weights": list(state.weights.shape),
                "bias": list(state.bias.shape),
            },
        }
        if include_arrays:
            payload["weights"] = np.round(state.weights, 8).tolist()
            payload["bias"] = np.round(state.bias, 8).tolist()
        return payload


@dataclass(frozen=True)
class ObjectStateIdentityContrastiveLoss:
    total_loss: float
    positive_loss: float
    negative_loss: float
    regularization_loss: float
    pair_count: int
    positive_pair_count: int
    negative_pair_count: int
    active_negative_pair_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "total_loss": float(self.total_loss),
            "positive_loss": float(self.positive_loss),
            "negative_loss": float(self.negative_loss),
            "regularization_loss": float(self.regularization_loss),
            "pair_count": int(self.pair_count),
            "positive_pair_count": int(self.positive_pair_count),
            "negative_pair_count": int(self.negative_pair_count),
            "active_negative_pair_count": int(self.active_negative_pair_count),
        }


@dataclass(frozen=True)
class ObjectStateIdentityEncoderTrainingResult:
    rows: tuple[ObjectStateIdentityRow, ...]
    initial_state: ObjectStateIdentityEncoderState
    final_state: ObjectStateIdentityEncoderState
    initial_loss: ObjectStateIdentityContrastiveLoss
    final_loss: ObjectStateIdentityContrastiveLoss
    initial_retrieval: dict[str, Any]
    final_retrieval: dict[str, Any]
    loss_history: tuple[float, ...]
    schema: str = OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA

    @property
    def passed(self) -> bool:
        return (
            self.final_loss.total_loss < self.initial_loss.total_loss
            and self.final_loss.negative_loss < self.initial_loss.negative_loss
            and self.final_retrieval["recall_at_1"] >= self.initial_retrieval["recall_at_1"]
        )

    def as_dict(self) -> dict[str, Any]:
        summary = {
            "schema": self.schema,
            "kind": "objectstate_identity_encoder_training",
            "status": _TRAINING_STATUS_PASS if self.passed else _TRAINING_STATUS_FAIL,
            "training_role": "contrastive_identity_encoder_smoke",
            "row_count": len(self.rows),
            "identity_count": len({row.identity_key for row in self.rows}),
            "loss": {
                "initial": self.initial_loss.as_dict(),
                "final": self.final_loss.as_dict(),
                "loss_decreased": bool(
                    self.final_loss.total_loss < self.initial_loss.total_loss
                ),
                "positive_loss_decreased": bool(
                    self.final_loss.positive_loss <= self.initial_loss.positive_loss
                ),
                "negative_loss_decreased": bool(
                    self.final_loss.negative_loss < self.initial_loss.negative_loss
                ),
                "history": [float(value) for value in self.loss_history],
            },
            "retrieval": {
                "initial": self.initial_retrieval,
                "final": self.final_retrieval,
                "recall_not_degraded": bool(
                    self.final_retrieval["recall_at_1"]
                    >= self.initial_retrieval["recall_at_1"]
                ),
            },
            "initial_state": self.initial_state.as_dict(),
            "final_state": self.final_state.as_dict(),
            "non_goals": {
                "uses_identity_graph": False,
                "uses_replay_buffer": False,
                "uses_diffusion": False,
                "uses_renderer_loss": False,
                "mutates_viewer_defaults": False,
            },
        }
        return validate_objectstate_identity_encoder_training_summary(summary)


def train_objectstate_identity_encoder(
    rows: Sequence[ObjectStateIdentityRow],
    *,
    config: ObjectStateIdentityEncoderConfig | None = None,
    iterations: int = 80,
) -> ObjectStateIdentityEncoderTrainingResult:
    checked_rows = _validate_training_rows(rows)
    features = objectstate_identity_encoder_features(
        checked_rows,
        feature_source="appearance" if config is None else config.feature_source,
    )
    resolved_config = config or ObjectStateIdentityEncoderConfig(
        input_dim=features.shape[1],
        embedding_dim=min(3, max(1, features.shape[1])),
    )
    resolved_config = validate_objectstate_identity_encoder_config(resolved_config)
    if resolved_config.input_dim != features.shape[1]:
        raise ValueError("config input_dim must match selected feature source")
    if int(iterations) < 1:
        raise ValueError("iterations must be >= 1")
    state = initialize_objectstate_identity_encoder_state(resolved_config)
    labels = _identity_labels(checked_rows)
    initial_embeddings = state.encode(checked_rows)
    initial_loss, _ = _contrastive_loss_and_gradient(
        initial_embeddings,
        labels,
        margin=resolved_config.margin,
        weights=state.weights,
        weight_decay=resolved_config.weight_decay,
    )
    initial_retrieval = _retrieval_summary(initial_embeddings, checked_rows)
    history = [float(initial_loss.total_loss)]
    current_state = state
    for _ in range(int(iterations)):
        embeddings = current_state.encode(checked_rows)
        loss, grad_embeddings = _contrastive_loss_and_gradient(
            embeddings,
            labels,
            margin=resolved_config.margin,
            weights=current_state.weights,
            weight_decay=resolved_config.weight_decay,
        )
        grad_weights = features.T @ grad_embeddings
        grad_weights += resolved_config.weight_decay * current_state.weights
        grad_bias = np.sum(grad_embeddings, axis=0)
        current_state = ObjectStateIdentityEncoderState(
            config=resolved_config,
            weights=(
                current_state.weights - resolved_config.learning_rate * grad_weights
            ).astype(np.float32, copy=False),
            bias=(
                current_state.bias - resolved_config.learning_rate * grad_bias
            ).astype(np.float32, copy=False),
            step=current_state.step + 1,
            source=current_state.source,
        )
        history.append(float(loss.total_loss))
    final_embeddings = current_state.encode(checked_rows)
    final_loss, _ = _contrastive_loss_and_gradient(
        final_embeddings,
        labels,
        margin=resolved_config.margin,
        weights=current_state.weights,
        weight_decay=resolved_config.weight_decay,
    )
    final_retrieval = _retrieval_summary(final_embeddings, checked_rows)
    return ObjectStateIdentityEncoderTrainingResult(
        rows=checked_rows,
        initial_state=state,
        final_state=validate_objectstate_identity_encoder_state(current_state),
        initial_loss=initial_loss,
        final_loss=final_loss,
        initial_retrieval=initial_retrieval,
        final_retrieval=final_retrieval,
        loss_history=tuple(history),
    )


def initialize_objectstate_identity_encoder_state(
    config: ObjectStateIdentityEncoderConfig,
) -> ObjectStateIdentityEncoderState:
    checked = validate_objectstate_identity_encoder_config(config)
    rng = np.random.default_rng(checked.seed)
    weights = rng.normal(
        loc=0.0,
        scale=0.05,
        size=(checked.input_dim, checked.embedding_dim),
    ).astype(np.float32)
    bias = np.zeros(checked.embedding_dim, dtype=np.float32)
    return validate_objectstate_identity_encoder_state(
        ObjectStateIdentityEncoderState(config=checked, weights=weights, bias=bias)
    )


def objectstate_identity_encoder_features(
    rows: Sequence[ObjectStateIdentityRow],
    *,
    feature_source: str = "appearance",
) -> np.ndarray:
    checked_rows = _validate_training_rows(rows)
    if feature_source not in _FEATURE_SOURCES:
        raise ValueError(f"unsupported feature_source: {feature_source}")
    values = []
    for row in checked_rows:
        if feature_source == "appearance":
            value = row.appearance_embedding
        elif feature_source == "appearance_geometry":
            value = np.concatenate([row.appearance_embedding, row.geometry_embedding])
        else:
            value = row.objectstate_embedding
        values.append(np.asarray(value, dtype=np.float32))
    dims = {int(value.shape[0]) for value in values}
    if len(dims) != 1:
        raise ValueError("identity encoder input features must share one dimension")
    return np.vstack(values).astype(np.float32, copy=False)


def validate_objectstate_identity_encoder_config(
    config: ObjectStateIdentityEncoderConfig,
) -> ObjectStateIdentityEncoderConfig:
    if not isinstance(config, ObjectStateIdentityEncoderConfig):
        raise TypeError("config must be ObjectStateIdentityEncoderConfig")
    if config.schema != OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA:
        raise ValueError(f"unsupported identity encoder config schema: {config.schema}")
    if int(config.input_dim) < 1:
        raise ValueError("input_dim must be >= 1")
    if int(config.embedding_dim) < 1:
        raise ValueError("embedding_dim must be >= 1")
    if float(config.margin) <= 0.0:
        raise ValueError("margin must be > 0")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be > 0")
    if float(config.weight_decay) < 0.0:
        raise ValueError("weight_decay must be >= 0")
    if config.feature_source not in _FEATURE_SOURCES:
        raise ValueError(f"unsupported feature_source: {config.feature_source}")
    return ObjectStateIdentityEncoderConfig(
        input_dim=int(config.input_dim),
        embedding_dim=int(config.embedding_dim),
        margin=float(config.margin),
        learning_rate=float(config.learning_rate),
        weight_decay=float(config.weight_decay),
        feature_source=config.feature_source,
        seed=int(config.seed),
        schema=config.schema,
    )


def validate_objectstate_identity_encoder_state(
    state: ObjectStateIdentityEncoderState,
) -> ObjectStateIdentityEncoderState:
    if not isinstance(state, ObjectStateIdentityEncoderState):
        raise TypeError("state must be ObjectStateIdentityEncoderState")
    if state.schema != OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA:
        raise ValueError(f"unsupported identity encoder state schema: {state.schema}")
    config = validate_objectstate_identity_encoder_config(state.config)
    weights = _array2d(state.weights, "weights")
    bias = _vector(state.bias, "bias")
    if weights.shape != (config.input_dim, config.embedding_dim):
        raise ValueError("weights shape must match config input_dim and embedding_dim")
    if bias.shape[0] != config.embedding_dim:
        raise ValueError("bias shape must match embedding_dim")
    return ObjectStateIdentityEncoderState(
        config=config,
        weights=weights,
        bias=bias,
        step=int(state.step),
        source=str(state.source),
        schema=state.schema,
    )


def validate_objectstate_identity_encoder_training_summary(
    payload: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("identity encoder training summary must be a dict")
    if payload.get("schema") != OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA:
        raise ValueError(f"unsupported identity encoder training schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_identity_encoder_training":
        raise ValueError("identity encoder training kind is unsupported")
    if payload.get("status") not in {_TRAINING_STATUS_PASS, _TRAINING_STATUS_FAIL}:
        raise ValueError("identity encoder training status is unsupported")
    for key in ("loss", "retrieval", "initial_state", "final_state", "non_goals"):
        if key not in payload:
            raise ValueError(f"identity encoder training summary missing {key}")
    loss = payload["loss"]
    if not isinstance(loss.get("loss_decreased"), bool):
        raise ValueError("identity encoder training loss_decreased must be bool")
    retrieval = payload["retrieval"]
    if not isinstance(retrieval.get("recall_not_degraded"), bool):
        raise ValueError("identity encoder training recall_not_degraded must be bool")
    non_goals = payload["non_goals"]
    if (
        non_goals.get("uses_identity_graph")
        or non_goals.get("uses_replay_buffer")
        or non_goals.get("uses_diffusion")
        or non_goals.get("uses_renderer_loss")
        or non_goals.get("mutates_viewer_defaults")
    ):
        raise ValueError("identity encoder training cannot use graph, replay, diffusion, renderer, or viewer policy")
    return payload


def _contrastive_loss_and_gradient(
    embeddings: np.ndarray,
    labels: np.ndarray,
    *,
    margin: float,
    weights: np.ndarray,
    weight_decay: float,
) -> tuple[ObjectStateIdentityContrastiveLoss, np.ndarray]:
    grad = np.zeros_like(embeddings, dtype=np.float32)
    positive_loss = 0.0
    negative_loss = 0.0
    positive_count = 0
    negative_count = 0
    active_negative_count = 0
    pair_count = 0
    for left in range(embeddings.shape[0]):
        for right in range(left + 1, embeddings.shape[0]):
            pair_count += 1
            delta = embeddings[left] - embeddings[right]
            distance = float(np.linalg.norm(delta) + _EPS)
            if int(labels[left]) == int(labels[right]):
                positive_count += 1
                positive_loss += 0.5 * distance * distance
                grad[left] += delta
                grad[right] -= delta
            else:
                negative_count += 1
                violation = float(margin) - distance
                if violation <= 0.0:
                    continue
                active_negative_count += 1
                negative_loss += 0.5 * violation * violation
                coefficient = -violation / distance
                grad[left] += coefficient * delta
                grad[right] -= coefficient * delta
    if pair_count == 0:
        raise ValueError("contrastive identity loss requires at least one pair")
    scale = 1.0 / float(pair_count)
    regularization = 0.5 * float(weight_decay) * float(np.sum(weights * weights))
    total_loss = (positive_loss + negative_loss) * scale + regularization
    grad = (grad * scale).astype(np.float32, copy=False)
    return (
        ObjectStateIdentityContrastiveLoss(
            total_loss=float(total_loss),
            positive_loss=float(positive_loss * scale),
            negative_loss=float(negative_loss * scale),
            regularization_loss=float(regularization),
            pair_count=int(pair_count),
            positive_pair_count=int(positive_count),
            negative_pair_count=int(negative_count),
            active_negative_pair_count=int(active_negative_count),
        ),
        grad,
    )


def _retrieval_summary(
    embeddings: np.ndarray,
    rows: Sequence[ObjectStateIdentityRow],
) -> dict[str, Any]:
    correct = 0
    evaluated = 0
    checked_rows = tuple(rows)
    for index, row in enumerate(checked_rows):
        positive_exists = any(
            other_index != index and other.identity_key == row.identity_key
            for other_index, other in enumerate(checked_rows)
        )
        if not positive_exists:
            continue
        distances = np.linalg.norm(embeddings - embeddings[index], axis=1)
        distances[index] = np.inf
        nearest = int(np.argmin(distances))
        evaluated += 1
        if checked_rows[nearest].identity_key == row.identity_key:
            correct += 1
    return {
        "evaluated_count": int(evaluated),
        "correct_count": int(correct),
        "recall_at_1": float(correct / evaluated) if evaluated else 0.0,
    }


def _identity_labels(rows: Sequence[ObjectStateIdentityRow]) -> np.ndarray:
    identity_to_label = {
        identity_key: index
        for index, identity_key in enumerate(sorted({row.identity_key for row in rows}))
    }
    return np.asarray([identity_to_label[row.identity_key] for row in rows], dtype=np.int64)


def _validate_training_rows(
    rows: Sequence[ObjectStateIdentityRow],
) -> tuple[ObjectStateIdentityRow, ...]:
    checked = tuple(validate_objectstate_identity_row(row) for row in rows)
    if len(checked) < 2:
        raise ValueError("identity encoder training requires at least two rows")
    identity_counts: dict[tuple[str, int], int] = {}
    for row in checked:
        identity_counts[row.identity_key] = identity_counts.get(row.identity_key, 0) + 1
    if len(identity_counts) < 2:
        raise ValueError("identity encoder training requires at least two identities")
    if not any(count > 1 for count in identity_counts.values()):
        raise ValueError("identity encoder training requires at least one positive pair")
    return checked


def _array2d(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{label} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    return array.astype(np.float32, copy=False)


def _vector(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a 1D array")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    return array.astype(np.float32, copy=False)
