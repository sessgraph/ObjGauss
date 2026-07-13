from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np

from objgauss.core.assignment_metrics import assignment_clustering_metrics
from objgauss.core.features import colors, opacity, positions
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_state import object_state_projection_summary, project_object_states

OBJECTSTATE_MODEL_V0_STATE_SCHEMA = "objgauss-objectstate-model-v0-state-v1"
OBJECTSTATE_MODEL_V0_TRAINING_SCHEMA = "objgauss-objectstate-model-v0-training-v1"
OBJECTSTATE_MODEL_V0_FEATURE_ORDER = ("x", "y", "z", "red", "green", "blue", "opacity")
_EPS = 1e-8

__all__ = (
    "OBJECTSTATE_MODEL_V0_FEATURE_ORDER",
    "OBJECTSTATE_MODEL_V0_STATE_SCHEMA",
    "OBJECTSTATE_MODEL_V0_TRAINING_SCHEMA",
    "ObjectStateModelV0Config",
    "ObjectStateModelV0Loss",
    "ObjectStateModelV0State",
    "ObjectStateModelV0TrainingResult",
    "objectstate_model_v0_features",
    "objectstate_model_v0_loss_and_gradient",
    "objectstate_model_v0_state_from_dict",
    "train_objectstate_model_v0",
    "validate_objectstate_model_v0_state",
    "validate_objectstate_model_v0_training_summary",
)


@dataclass(frozen=True)
class ObjectStateModelV0Config:
    slots: int
    input_dim: int = 7
    hidden_dim: int = 24
    learning_rate: float = 0.08
    assignment_weight: float = 1.0
    compactness_weight: float = 0.02
    semantic_weight: float = 0.02
    weight_decay: float = 1e-4
    gradient_clip_norm: float = 5.0
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        config = _validate_config(self)
        return {
            "slots": config.slots,
            "input_dim": config.input_dim,
            "hidden_dim": config.hidden_dim,
            "learning_rate": config.learning_rate,
            "loss_weights": {
                "assignment": config.assignment_weight,
                "compactness": config.compactness_weight,
                "semantic_consistency": config.semantic_weight,
                "weight_decay": config.weight_decay,
            },
            "gradient_clip_norm": config.gradient_clip_norm,
            "seed": config.seed,
        }


@dataclass(frozen=True)
class ObjectStateModelV0State:
    config: ObjectStateModelV0Config
    feature_mean: np.ndarray
    feature_std: np.ndarray
    encoder_weight: np.ndarray
    encoder_bias: np.ndarray
    assignment_weight: np.ndarray
    assignment_bias: np.ndarray
    step: int = 0
    source: str = "objectstate_model_v0_training"
    schema: str = OBJECTSTATE_MODEL_V0_STATE_SCHEMA
    feature_order: tuple[str, ...] = OBJECTSTATE_MODEL_V0_FEATURE_ORDER

    def predict(self, cloud: GaussianCloud) -> np.ndarray:
        if self.feature_order != OBJECTSTATE_MODEL_V0_FEATURE_ORDER:
            raise ValueError(
                "custom-feature ObjectState Model v0 states require predict_features"
            )
        features = objectstate_model_v0_features(cloud)
        return self.predict_features(features)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        state = validate_objectstate_model_v0_state(self)
        values = _matrix(features, "features", columns=state.config.input_dim)
        normalized = (values - state.feature_mean) / state.feature_std
        hidden = np.tanh(normalized @ state.encoder_weight + state.encoder_bias)
        logits = hidden @ state.assignment_weight + state.assignment_bias
        return _softmax(logits)

    def as_dict(self, *, include_arrays: bool = False) -> dict[str, Any]:
        state = validate_objectstate_model_v0_state(self)
        payload: dict[str, Any] = {
            "schema": state.schema,
            "kind": "objectstate_model_v0_state",
            "model_family": "gaussian-object-encoder-assignment-head-v0",
            "feature_order": list(state.feature_order),
            "config": state.config.as_dict(),
            "step": state.step,
            "source": state.source,
            "shapes": {
                "feature_mean": list(state.feature_mean.shape),
                "feature_std": list(state.feature_std.shape),
                "encoder_weight": list(state.encoder_weight.shape),
                "encoder_bias": list(state.encoder_bias.shape),
                "assignment_weight": list(state.assignment_weight.shape),
                "assignment_bias": list(state.assignment_bias.shape),
            },
        }
        if include_arrays:
            for name in (
                "feature_mean",
                "feature_std",
                "encoder_weight",
                "encoder_bias",
                "assignment_weight",
                "assignment_bias",
            ):
                # Preserve the trained float32 values exactly. Rounding checkpoint
                # arrays changes soft assignments enough to break browser/Python
                # inference agreement even when the hard labels happen to match.
                payload[name] = getattr(state, name).tolist()
        return payload


@dataclass(frozen=True)
class ObjectStateModelV0Loss:
    total_loss: float
    supervised_loss: float
    compactness_loss: float
    semantic_consistency_loss: float
    regularization_loss: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "total_loss": float(self.total_loss),
            "supervised_loss": float(self.supervised_loss),
            "compactness_loss": float(self.compactness_loss),
            "semantic_consistency_loss": float(self.semantic_consistency_loss),
            "regularization_loss": float(self.regularization_loss),
        }


@dataclass(frozen=True)
class ObjectStateModelV0TrainingResult:
    cloud: GaussianCloud
    source_object_ids: tuple[int, ...]
    train_indices: np.ndarray
    heldout_indices: np.ndarray
    frame_field: str
    split_seed: int
    train_frame_ids: tuple[int, ...]
    heldout_frame_ids: tuple[int, ...]
    target_labels: np.ndarray
    initial_state: ObjectStateModelV0State
    final_state: ObjectStateModelV0State
    initial_loss: ObjectStateModelV0Loss
    final_loss: ObjectStateModelV0Loss
    loss_history: tuple[dict[str, float | int], ...]
    initial_assignment: np.ndarray
    final_assignment: np.ndarray
    train_before_metrics: dict[str, Any]
    train_after_metrics: dict[str, Any]
    heldout_before_metrics: dict[str, Any]
    heldout_after_metrics: dict[str, Any]
    projection: dict[str, Any]
    schema: str = OBJECTSTATE_MODEL_V0_TRAINING_SCHEMA

    @property
    def passed(self) -> bool:
        heldout_assignment = _assignment_statistics(
            self.final_assignment[self.heldout_indices]
        )
        train_object_ids = set(int(item) for item in self.target_labels[self.train_indices])
        heldout_object_ids = set(int(item) for item in self.target_labels[self.heldout_indices])
        expected_object_ids = set(range(self.final_state.config.slots))
        return bool(
            self.final_loss.total_loss < self.initial_loss.total_loss
            and self.final_loss.supervised_loss < self.initial_loss.supervised_loss
            and self.heldout_after_metrics["ari"] >= self.heldout_before_metrics["ari"]
            and self.heldout_after_metrics["ari"] >= 0.5
            and self.heldout_after_metrics["purity"] >= 0.5
            and heldout_assignment["active_hard_slots"] >= 2
            and train_object_ids == expected_object_ids
            and heldout_object_ids == expected_object_ids
            and set(self.train_frame_ids).isdisjoint(self.heldout_frame_ids)
        )

    def as_dict(self) -> dict[str, Any]:
        complete_scene_holdout = self.frame_field == "scene_id"
        payload = {
            "schema": self.schema,
            "kind": "objectstate_model_v0_training",
            "status": (
                "objectstate_model_v0_training_pass"
                if self.passed
                else "objectstate_model_v0_training_reviewable"
            ),
            "model_family": "gaussian-object-encoder-assignment-head-v0",
            "model_contract": (
                f"FeatureMatrix[{','.join(self.final_state.feature_order)}] -> "
                "ObjectEncoder(tanh) -> "
                "AssignmentHead -> A[N,K] -> ObjectStateProjection"
            ),
            "feature_order": list(self.final_state.feature_order),
            "state_schema": OBJECTSTATE_MODEL_V0_STATE_SCHEMA,
            "config": self.final_state.config.as_dict(),
            "split": {
                "field": self.frame_field,
                "seed": self.split_seed,
                "policy": (
                    "deterministic_complete_scene_holdout"
                    if complete_scene_holdout
                    else "deterministic_whole_frame_holdout"
                ),
                "train_frame_ids": list(self.train_frame_ids),
                "heldout_frame_ids": list(self.heldout_frame_ids),
                "train_group_ids": list(self.train_frame_ids),
                "heldout_group_ids": list(self.heldout_frame_ids),
                "train_row_count": int(self.train_indices.shape[0]),
                "heldout_row_count": int(self.heldout_indices.shape[0]),
                "frame_overlap_count": len(set(self.train_frame_ids) & set(self.heldout_frame_ids)),
                "group_overlap_count": len(set(self.train_frame_ids) & set(self.heldout_frame_ids)),
                "row_overlap_count": int(np.intersect1d(self.train_indices, self.heldout_indices).size),
                "source_object_ids": list(self.source_object_ids),
                "train_object_ids": sorted(
                    int(item) for item in np.unique(self.target_labels[self.train_indices])
                ),
                "heldout_object_ids": sorted(
                    int(item) for item in np.unique(self.target_labels[self.heldout_indices])
                ),
            },
            "loss": {
                "initial": self.initial_loss.as_dict(),
                "final": self.final_loss.as_dict(),
                "total_decreased": self.final_loss.total_loss < self.initial_loss.total_loss,
                "supervised_decreased": (
                    self.final_loss.supervised_loss < self.initial_loss.supervised_loss
                ),
                "history": list(self.loss_history),
            },
            "train_before_metrics": self.train_before_metrics,
            "train_after_metrics": self.train_after_metrics,
            "heldout_before_metrics": self.heldout_before_metrics,
            "heldout_after_metrics": self.heldout_after_metrics,
            "assignment_diagnostics": {
                "train_before": _assignment_statistics(
                    self.initial_assignment[self.train_indices]
                ),
                "train_after": _assignment_statistics(
                    self.final_assignment[self.train_indices]
                ),
                "heldout_before": _assignment_statistics(
                    self.initial_assignment[self.heldout_indices]
                ),
                "heldout_after": _assignment_statistics(
                    self.final_assignment[self.heldout_indices]
                ),
            },
            "generalization_gap": {
                key: float(self.train_after_metrics[key] - self.heldout_after_metrics[key])
                for key in ("mean_best_iou", "ari", "purity")
            },
            "projection": self.projection,
            "initial_state": self.initial_state.as_dict(),
            "final_state": self.final_state.as_dict(),
            "claim_policy": {
                **(
                    {"complete_scene_holdout": True}
                    if complete_scene_holdout
                    else {"same_scene_heldout_frames_only": True}
                ),
                "heldout_labels_not_used_for_gradient_updates": True,
                "target_object_id_is_supervision_only": True,
                **(
                    {"does_not_claim_cross_dataset_generalization": True}
                    if complete_scene_holdout
                    else {"does_not_claim_cross_scene_generalization": True}
                ),
                "does_not_claim_identity_gate_pass": True,
                "does_not_claim_reality_gate_pass": True,
                "does_not_claim_world_model": True,
            },
        }
        return validate_objectstate_model_v0_training_summary(payload)


def train_objectstate_model_v0(
    cloud: GaussianCloud,
    *,
    object_id_field: str = "object_id",
    frame_field: str = "source_frame",
    config: ObjectStateModelV0Config | None = None,
    hidden_dim: int = 24,
    heldout_stride: int = 4,
    iterations: int = 240,
    learning_rate: float = 0.08,
    assignment_weight: float = 1.0,
    compactness_weight: float = 0.02,
    semantic_weight: float = 0.02,
    weight_decay: float = 1e-4,
    seed: int = 0,
    split_seed: int | None = None,
    record_every: int | None = None,
    feature_matrix: np.ndarray | None = None,
    feature_order: Sequence[str] | None = None,
) -> ObjectStateModelV0TrainingResult:
    if iterations < 1 or iterations > 2000:
        raise ValueError("iterations must be in [1, 2000]")
    if feature_matrix is None:
        features = objectstate_model_v0_features(cloud)
        resolved_feature_order = OBJECTSTATE_MODEL_V0_FEATURE_ORDER
        if feature_order is not None and tuple(feature_order) != resolved_feature_order:
            raise ValueError(
                "feature_order without feature_matrix must match canonical Model v0 features"
            )
    else:
        features = _matrix(feature_matrix, "feature_matrix", rows=cloud.count)
        resolved_feature_order = _feature_order(
            feature_order,
            columns=features.shape[1],
        )
    target_labels, source_object_ids = _target_labels(cloud, object_id_field)
    resolved_split_seed = int(seed if split_seed is None else split_seed)
    train_indices, heldout_indices, train_frames, heldout_frames = _frame_split(
        cloud,
        frame_field=frame_field,
        heldout_stride=heldout_stride,
        seed=resolved_split_seed,
    )
    slots = len(source_object_ids)
    resolved_config = config or ObjectStateModelV0Config(
        slots=slots,
        input_dim=features.shape[1],
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        assignment_weight=assignment_weight,
        compactness_weight=compactness_weight,
        semantic_weight=semantic_weight,
        weight_decay=weight_decay,
        seed=seed,
    )
    resolved_config = _validate_config(resolved_config)
    if resolved_config.slots != slots:
        raise ValueError("config slots must match supervised object count")
    train_features = features[train_indices]
    feature_mean = train_features.mean(axis=0).astype(np.float32)
    feature_std = train_features.std(axis=0).astype(np.float32)
    feature_std = np.where(feature_std < _EPS, 1.0, feature_std).astype(np.float32)
    state = _initialize_state(
        resolved_config,
        feature_mean,
        feature_std,
        feature_order=resolved_feature_order,
    )
    initial_state = state
    initial_assignment = state.predict_features(features)
    normalized = (features - feature_mean) / feature_std
    train_values = normalized[train_indices]
    train_targets = target_labels[train_indices]
    initial_loss, _ = _state_loss_and_gradient(
        state,
        train_values,
        train_targets,
    )
    record_every = record_every or max(1, iterations // 12)
    history: list[dict[str, float | int]] = [{"step": 0, **initial_loss.as_dict()}]
    for step in range(1, iterations + 1):
        state, loss = _training_step(state, train_values, train_targets)
        if step % record_every == 0 or step == iterations:
            history.append({"step": step, **loss.as_dict()})
    final_loss, _ = _state_loss_and_gradient(state, train_values, train_targets)
    final_assignment = state.predict_features(features)
    projection = project_object_states(
        cloud,
        final_assignment,
        evidence_features=normalized,
    )
    return ObjectStateModelV0TrainingResult(
        cloud=cloud,
        source_object_ids=source_object_ids,
        train_indices=train_indices,
        heldout_indices=heldout_indices,
        frame_field=str(frame_field),
        split_seed=resolved_split_seed,
        train_frame_ids=train_frames,
        heldout_frame_ids=heldout_frames,
        target_labels=target_labels,
        initial_state=initial_state,
        final_state=state,
        initial_loss=initial_loss,
        final_loss=final_loss,
        loss_history=tuple(history),
        initial_assignment=initial_assignment,
        final_assignment=final_assignment,
        train_before_metrics=_metrics(initial_assignment, target_labels, train_indices),
        train_after_metrics=_metrics(final_assignment, target_labels, train_indices),
        heldout_before_metrics=_metrics(initial_assignment, target_labels, heldout_indices),
        heldout_after_metrics=_metrics(final_assignment, target_labels, heldout_indices),
        projection=object_state_projection_summary(projection),
    )


def objectstate_model_v0_features(cloud: GaussianCloud) -> np.ndarray:
    return np.column_stack(
        [positions(cloud), colors(cloud), opacity(cloud)]
    ).astype(np.float32, copy=False)


def objectstate_model_v0_loss_and_gradient(
    logits: np.ndarray,
    target_labels: np.ndarray,
    spatial_features: np.ndarray,
    semantic_features: np.ndarray,
    *,
    assignment_weight: float = 1.0,
    compactness_weight: float = 0.02,
    semantic_weight: float = 0.02,
) -> tuple[ObjectStateModelV0Loss, np.ndarray]:
    values = _matrix(logits, "logits")
    labels = _labels(target_labels, rows=values.shape[0], slots=values.shape[1])
    spatial = _matrix(spatial_features, "spatial_features", rows=values.shape[0])
    semantic = _matrix(semantic_features, "semantic_features", rows=values.shape[0])
    probabilities = _softmax(values)
    row_ids = np.arange(values.shape[0])
    supervised_loss = float(-np.mean(np.log(np.clip(probabilities[row_ids, labels], _EPS, 1.0))))
    one_hot = np.zeros_like(probabilities)
    one_hot[row_ids, labels] = 1.0
    grad_logits = assignment_weight * (probabilities - one_hot) / values.shape[0]
    compactness_loss, compactness_grad = _slot_variance(probabilities, spatial)
    semantic_loss, semantic_grad = _slot_variance(probabilities, semantic)
    grad_probabilities = (
        compactness_weight * compactness_grad + semantic_weight * semantic_grad
    )
    grad_logits += probabilities * (
        grad_probabilities
        - np.sum(probabilities * grad_probabilities, axis=1, keepdims=True)
    )
    total = (
        assignment_weight * supervised_loss
        + compactness_weight * compactness_loss
        + semantic_weight * semantic_loss
    )
    return (
        ObjectStateModelV0Loss(
            total_loss=float(total),
            supervised_loss=supervised_loss,
            compactness_loss=compactness_loss,
            semantic_consistency_loss=semantic_loss,
        ),
        grad_logits.astype(np.float32, copy=False),
    )


def objectstate_model_v0_state_from_dict(payload: Mapping[str, Any]) -> ObjectStateModelV0State:
    if not isinstance(payload, Mapping):
        raise TypeError("ObjectState Model v0 checkpoint must be a mapping")
    if payload.get("schema") != OBJECTSTATE_MODEL_V0_STATE_SCHEMA:
        raise ValueError(f"unsupported ObjectState Model v0 schema: {payload.get('schema')}")
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise ValueError("ObjectState Model v0 checkpoint requires config")
    loss_weights = config_payload.get("loss_weights")
    if not isinstance(loss_weights, Mapping):
        raise ValueError("ObjectState Model v0 checkpoint requires loss_weights")
    config = ObjectStateModelV0Config(
        slots=int(config_payload["slots"]),
        input_dim=int(config_payload["input_dim"]),
        hidden_dim=int(config_payload["hidden_dim"]),
        learning_rate=float(config_payload["learning_rate"]),
        assignment_weight=float(loss_weights["assignment"]),
        compactness_weight=float(loss_weights["compactness"]),
        semantic_weight=float(loss_weights["semantic_consistency"]),
        weight_decay=float(loss_weights["weight_decay"]),
        gradient_clip_norm=float(config_payload["gradient_clip_norm"]),
        seed=int(config_payload["seed"]),
    )
    return validate_objectstate_model_v0_state(
        ObjectStateModelV0State(
            config=config,
            feature_order=_feature_order(
                payload.get("feature_order", OBJECTSTATE_MODEL_V0_FEATURE_ORDER),
                columns=config.input_dim,
            ),
            feature_mean=np.asarray(payload["feature_mean"], dtype=np.float32),
            feature_std=np.asarray(payload["feature_std"], dtype=np.float32),
            encoder_weight=np.asarray(payload["encoder_weight"], dtype=np.float32),
            encoder_bias=np.asarray(payload["encoder_bias"], dtype=np.float32),
            assignment_weight=np.asarray(payload["assignment_weight"], dtype=np.float32),
            assignment_bias=np.asarray(payload["assignment_bias"], dtype=np.float32),
            step=int(payload.get("step", 0)),
            source=str(payload.get("source", "objectstate_model_v0_checkpoint")),
        )
    )


def validate_objectstate_model_v0_state(state: ObjectStateModelV0State) -> ObjectStateModelV0State:
    if not isinstance(state, ObjectStateModelV0State):
        raise TypeError("state must be ObjectStateModelV0State")
    if state.schema != OBJECTSTATE_MODEL_V0_STATE_SCHEMA:
        raise ValueError(f"unsupported ObjectState Model v0 schema: {state.schema}")
    config = _validate_config(state.config)
    feature_order = _feature_order(state.feature_order, columns=config.input_dim)
    mean = _vector(state.feature_mean, "feature_mean", length=config.input_dim)
    std = _vector(state.feature_std, "feature_std", length=config.input_dim)
    if np.any(std <= 0.0):
        raise ValueError("feature_std must be positive")
    return replace(
        state,
        config=config,
        feature_order=feature_order,
        feature_mean=mean,
        feature_std=std,
        encoder_weight=_matrix(
            state.encoder_weight,
            "encoder_weight",
            rows=config.input_dim,
            columns=config.hidden_dim,
        ),
        encoder_bias=_vector(state.encoder_bias, "encoder_bias", length=config.hidden_dim),
        assignment_weight=_matrix(
            state.assignment_weight,
            "assignment_weight",
            rows=config.hidden_dim,
            columns=config.slots,
        ),
        assignment_bias=_vector(state.assignment_bias, "assignment_bias", length=config.slots),
        step=int(state.step),
        source=str(state.source),
    )


def validate_objectstate_model_v0_training_summary(payload: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("ObjectState Model v0 training summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_MODEL_V0_TRAINING_SCHEMA:
        raise ValueError(f"unsupported ObjectState Model v0 training schema: {payload.get('schema')}")
    if payload.get("status") not in {
        "objectstate_model_v0_training_pass",
        "objectstate_model_v0_training_reviewable",
    }:
        raise ValueError("ObjectState Model v0 training status is unsupported")
    split = payload.get("split")
    if not isinstance(split, Mapping):
        raise ValueError("ObjectState Model v0 summary requires split")
    if (
        split.get("frame_overlap_count") != 0
        or split.get("group_overlap_count", 0) != 0
        or split.get("row_overlap_count") != 0
    ):
        raise ValueError("ObjectState Model v0 train/held-out split must not overlap")
    for key in ("train_before_metrics", "train_after_metrics", "heldout_before_metrics", "heldout_after_metrics"):
        metrics = payload.get(key)
        if not isinstance(metrics, Mapping):
            raise ValueError(f"ObjectState Model v0 summary requires {key}")
        for metric in ("ari", "mean_best_iou", "purity"):
            _finite(metrics.get(metric), f"{key}.{metric}")
    diagnostics = payload.get("assignment_diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise ValueError("ObjectState Model v0 summary requires assignment diagnostics")
    for split_name in ("train_before", "train_after", "heldout_before", "heldout_after"):
        values = diagnostics.get(split_name)
        if not isinstance(values, Mapping):
            raise ValueError(f"ObjectState Model v0 summary requires {split_name} diagnostics")
        for metric in ("mean_confidence", "mean_normalized_entropy", "effective_slots"):
            _finite(values.get(metric), f"assignment_diagnostics.{split_name}.{metric}")
    claim = payload.get("claim_policy")
    if not isinstance(claim, Mapping) or not all(bool(value) for value in claim.values()):
        raise ValueError("ObjectState Model v0 summary must preserve claim policy")
    if split.get("field") == "scene_id":
        if split.get("policy") != "deterministic_complete_scene_holdout":
            raise ValueError("scene_id split must use complete-scene holdout")
        if claim.get("complete_scene_holdout") is not True:
            raise ValueError("scene_id split must preserve complete-scene claim")
    elif claim.get("same_scene_heldout_frames_only") is not True:
        raise ValueError("frame split must preserve same-scene claim boundary")
    return dict(payload)


def _training_step(
    state: ObjectStateModelV0State,
    normalized_features: np.ndarray,
    labels: np.ndarray,
) -> tuple[ObjectStateModelV0State, ObjectStateModelV0Loss]:
    hidden = np.tanh(normalized_features @ state.encoder_weight + state.encoder_bias)
    logits = hidden @ state.assignment_weight + state.assignment_bias
    loss, grad_logits = objectstate_model_v0_loss_and_gradient(
        logits,
        labels,
        normalized_features[:, :3],
        normalized_features[:, 3:6],
        assignment_weight=state.config.assignment_weight,
        compactness_weight=state.config.compactness_weight,
        semantic_weight=state.config.semantic_weight,
    )
    regularization = 0.5 * state.config.weight_decay * (
        float(np.sum(state.encoder_weight**2)) + float(np.sum(state.assignment_weight**2))
    )
    loss = replace(
        loss,
        total_loss=loss.total_loss + regularization,
        regularization_loss=regularization,
    )
    grad_assignment_weight = hidden.T @ grad_logits + state.config.weight_decay * state.assignment_weight
    grad_assignment_bias = np.sum(grad_logits, axis=0)
    grad_hidden = grad_logits @ state.assignment_weight.T
    grad_encoder_pre = grad_hidden * (1.0 - hidden * hidden)
    grad_encoder_weight = (
        normalized_features.T @ grad_encoder_pre
        + state.config.weight_decay * state.encoder_weight
    )
    grad_encoder_bias = np.sum(grad_encoder_pre, axis=0)
    gradients = (
        grad_encoder_weight,
        grad_encoder_bias,
        grad_assignment_weight,
        grad_assignment_bias,
    )
    norm = float(np.sqrt(sum(float(np.sum(gradient**2)) for gradient in gradients)))
    scale = min(1.0, state.config.gradient_clip_norm / max(norm, _EPS))
    learning_rate = state.config.learning_rate
    return (
        replace(
            state,
            encoder_weight=(state.encoder_weight - learning_rate * scale * grad_encoder_weight).astype(np.float32),
            encoder_bias=(state.encoder_bias - learning_rate * scale * grad_encoder_bias).astype(np.float32),
            assignment_weight=(
                state.assignment_weight - learning_rate * scale * grad_assignment_weight
            ).astype(np.float32),
            assignment_bias=(state.assignment_bias - learning_rate * scale * grad_assignment_bias).astype(np.float32),
            step=state.step + 1,
        ),
        loss,
    )


def _state_loss_and_gradient(
    state: ObjectStateModelV0State,
    normalized_features: np.ndarray,
    labels: np.ndarray,
) -> tuple[ObjectStateModelV0Loss, np.ndarray]:
    hidden = np.tanh(normalized_features @ state.encoder_weight + state.encoder_bias)
    logits = hidden @ state.assignment_weight + state.assignment_bias
    loss, gradient = objectstate_model_v0_loss_and_gradient(
        logits,
        labels,
        normalized_features[:, :3],
        normalized_features[:, 3:6],
        assignment_weight=state.config.assignment_weight,
        compactness_weight=state.config.compactness_weight,
        semantic_weight=state.config.semantic_weight,
    )
    regularization = 0.5 * state.config.weight_decay * (
        float(np.sum(state.encoder_weight**2)) + float(np.sum(state.assignment_weight**2))
    )
    return (
        replace(
            loss,
            total_loss=loss.total_loss + regularization,
            regularization_loss=regularization,
        ),
        gradient,
    )


def _initialize_state(
    config: ObjectStateModelV0Config,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    *,
    feature_order: tuple[str, ...],
) -> ObjectStateModelV0State:
    rng = np.random.default_rng(config.seed)
    encoder_scale = np.sqrt(2.0 / (config.input_dim + config.hidden_dim))
    head_scale = np.sqrt(2.0 / (config.hidden_dim + config.slots))
    return validate_objectstate_model_v0_state(
        ObjectStateModelV0State(
            config=config,
            feature_order=feature_order,
            feature_mean=feature_mean,
            feature_std=feature_std,
            encoder_weight=rng.normal(0.0, encoder_scale, (config.input_dim, config.hidden_dim)).astype(np.float32),
            encoder_bias=np.zeros(config.hidden_dim, dtype=np.float32),
            assignment_weight=rng.normal(0.0, head_scale, (config.hidden_dim, config.slots)).astype(np.float32),
            assignment_bias=np.zeros(config.slots, dtype=np.float32),
        )
    )


def _frame_split(
    cloud: GaussianCloud,
    *,
    frame_field: str,
    heldout_stride: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, tuple[int, ...], tuple[int, ...]]:
    if heldout_stride < 2:
        raise ValueError("heldout_stride must be >= 2")
    if frame_field not in cloud.fields:
        raise ValueError(f"Gaussian cloud is missing frame field {frame_field!r}")
    values = np.asarray(cloud.vertices[frame_field], dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.allclose(values, np.rint(values)):
        raise ValueError("source_frame values must be finite integers")
    frames = np.rint(values).astype(np.int64)
    unique = tuple(int(item) for item in sorted(np.unique(frames)))
    if len(unique) < 2:
        raise ValueError("ObjectState Model v0 requires at least two source frames")
    offset = int(seed) % heldout_stride
    heldout_frames = tuple(frame for index, frame in enumerate(unique) if index % heldout_stride == offset)
    train_frames = tuple(frame for frame in unique if frame not in heldout_frames)
    if not heldout_frames or not train_frames:
        raise ValueError("ObjectState Model v0 frame split requires train and held-out frames")
    heldout_mask = np.isin(frames, heldout_frames)
    return (
        np.flatnonzero(~heldout_mask).astype(np.int64),
        np.flatnonzero(heldout_mask).astype(np.int64),
        train_frames,
        heldout_frames,
    )


def _target_labels(cloud: GaussianCloud, field: str) -> tuple[np.ndarray, tuple[int, ...]]:
    if field not in cloud.fields:
        raise ValueError(f"Gaussian cloud is missing supervision field {field!r}")
    values = np.asarray(cloud.vertices[field], dtype=np.float64)
    if not np.all(np.isfinite(values)) or not np.allclose(values, np.rint(values)):
        raise ValueError("object supervision must contain finite integer ids")
    original = np.rint(values).astype(np.int64)
    if np.any(original < 0):
        raise ValueError("object supervision must contain non-negative ids")
    source_ids = tuple(int(item) for item in sorted(np.unique(original)))
    if len(source_ids) < 2:
        raise ValueError("ObjectState Model v0 requires at least two supervised objects")
    mapping = {source_id: index for index, source_id in enumerate(source_ids)}
    return np.asarray([mapping[int(item)] for item in original], dtype=np.int64), source_ids


def _metrics(assignment: np.ndarray, labels: np.ndarray, indices: np.ndarray) -> dict[str, Any]:
    return assignment_clustering_metrics(np.argmax(assignment[indices], axis=1), labels[indices])


def _assignment_statistics(assignment: np.ndarray) -> dict[str, float | int]:
    values = _matrix(assignment, "assignment")
    confidence = np.max(values, axis=1)
    entropy = -np.sum(values * np.log(np.clip(values, _EPS, 1.0)), axis=1)
    normalized_entropy = entropy / np.log(values.shape[1])
    slot_mass = np.sum(values, axis=0)
    normalized_mass = slot_mass / np.sum(slot_mass)
    effective_slots = float(1.0 / np.sum(normalized_mass * normalized_mass))
    return {
        "row_count": int(values.shape[0]),
        "mean_confidence": float(np.mean(confidence)),
        "mean_normalized_entropy": float(np.mean(normalized_entropy)),
        "effective_slots": effective_slots,
        "active_hard_slots": int(np.unique(np.argmax(values, axis=1)).shape[0]),
    }


def _slot_variance(probabilities: np.ndarray, values: np.ndarray) -> tuple[float, np.ndarray]:
    mass = np.sum(probabilities, axis=0)
    centroids = (probabilities.T @ values) / mass[:, None]
    distances = np.sum((values[:, None, :] - centroids[None, :, :]) ** 2, axis=2)
    return (
        float(np.sum(probabilities * distances) / probabilities.shape[0]),
        (distances / probabilities.shape[0]).astype(np.float32, copy=False),
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(np.clip(shifted, -80.0, 0.0))
    return (exp / np.sum(exp, axis=1, keepdims=True)).astype(np.float32, copy=False)


def _validate_config(config: ObjectStateModelV0Config) -> ObjectStateModelV0Config:
    if not isinstance(config, ObjectStateModelV0Config):
        raise TypeError("config must be ObjectStateModelV0Config")
    if config.slots < 2 or config.input_dim < 1 or config.hidden_dim < 1:
        raise ValueError("ObjectState Model v0 config has invalid dimensions")
    for name in ("learning_rate", "assignment_weight", "gradient_clip_norm"):
        if float(getattr(config, name)) <= 0.0:
            raise ValueError(f"{name} must be > 0")
    for name in ("compactness_weight", "semantic_weight", "weight_decay"):
        if float(getattr(config, name)) < 0.0:
            raise ValueError(f"{name} must be >= 0")
    return config


def _feature_order(
    value: Sequence[str] | None,
    *,
    columns: int,
) -> tuple[str, ...]:
    if value is None:
        raise ValueError("custom feature_matrix requires feature_order")
    resolved = tuple(str(item) for item in value)
    if len(resolved) != columns or any(not item for item in resolved):
        raise ValueError("feature_order must name every feature_matrix column")
    if len(set(resolved)) != len(resolved):
        raise ValueError("feature_order names must be unique")
    return resolved


def _matrix(
    value: np.ndarray,
    name: str,
    *,
    rows: int | None = None,
    columns: int | None = None,
) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite 2D array")
    if rows is not None and array.shape[0] != rows:
        raise ValueError(f"{name} row count is invalid")
    if columns is not None and array.shape[1] != columns:
        raise ValueError(f"{name} column count is invalid")
    return array


def _vector(value: np.ndarray, name: str, *, length: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (length,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return array


def _labels(value: np.ndarray, *, rows: int, slots: int) -> np.ndarray:
    labels = np.asarray(value)
    if labels.shape != (rows,) or not np.all(np.isfinite(labels)) or not np.allclose(labels, np.rint(labels)):
        raise ValueError("target_labels must contain one finite integer per row")
    labels = np.rint(labels).astype(np.int64)
    if np.any(labels < 0) or np.any(labels >= slots):
        raise ValueError("target_labels fall outside assignment slots")
    return labels


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return float(value)
