from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from objgauss.core.assignment_evidence import (
    ASSIGNMENT_EVIDENCE_BATCH_SCHEMA,
    AssignmentEvidenceBatch,
    validate_assignment_evidence_batch,
)
from objgauss.core.assignment_solver_v2 import (
    ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA,
    ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
    AssignmentSolverV2State,
    predict_assignment_solver_v2,
    validate_assignment_solver_v2_state,
)
from objgauss.core.features import extract_features, positions
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_state import (
    ObjectStateProjection,
    project_object_states,
    validate_assignment_matrix,
)

OBJECTSTATE_ASSIGNMENT_MVP_SCHEMA = "objgauss-objectstate-assignment-mvp-v1"


def objectstate_assignment_mvp_summary(
    cloud: GaussianCloud,
    solver_state: AssignmentSolverV2State,
    *,
    target_assignment: np.ndarray | None = None,
    evidence_features: np.ndarray | None = None,
    source: str = "gaussian_cloud",
    include_assignment: bool = False,
) -> dict[str, Any]:
    """Run the minimum Gaussian -> A[N,K] -> ObjectStateProjection path."""

    state = validate_assignment_solver_v2_state(solver_state)
    evidence = validate_assignment_evidence_batch(
        AssignmentEvidenceBatch(
            positions=positions(cloud),
            features=extract_features(cloud)
            if evidence_features is None
            else np.asarray(evidence_features, dtype=np.float32),
            frame_index=0,
            target_assignment=target_assignment,
            source=source,
        )
    )
    prediction = predict_assignment_solver_v2(evidence, state)
    projection = project_object_states(
        cloud,
        prediction.assignment,
        evidence_features=evidence.features,
    )
    payload: dict[str, Any] = {
        "schema": OBJECTSTATE_ASSIGNMENT_MVP_SCHEMA,
        "kind": "objectstate_assignment_mvp",
        "status": "objectstate_assignment_mvp_ready",
        "model_contract": "Gaussian / AssignmentEvidence -> A[N,K] -> ObjectStateProjection",
        "source": str(source),
        "input": {
            "gaussian_count": cloud.count,
            "gaussian_fields": list(cloud.fields),
            "evidence_schema": ASSIGNMENT_EVIDENCE_BATCH_SCHEMA,
            "feature_dim": evidence.feature_dim,
            "position_dim": int(evidence.positions.shape[1]),
            "has_target_assignment": evidence.target_assignment is not None,
        },
        "solver": {
            "schema": ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
            "family": state.config.solver_family,
            "slots": state.config.slots,
            "step": state.step,
            "source": state.source,
            "cost_terms": list(state.config.cost_terms),
            "temporal_policy": state.config.temporal_policy,
            "matching_policy": state.config.matching_policy,
        },
        "assignment": {
            "schema": ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA,
            "shape": list(prediction.assignment.shape),
            "row_normalized": True,
            "slot_mass": np.round(prediction.slot_mass, 6).tolist(),
            "mean_confidence": float(np.mean(prediction.confidence)),
            "min_confidence": float(np.min(prediction.confidence)),
            "mean_normalized_entropy": float(prediction.mean_normalized_entropy),
            "effective_slots": float(prediction.effective_slots),
            "diagnostics": list(prediction.diagnostics),
        },
        "projection": _projection_summary(projection),
        "target_metrics": _target_metrics(
            prediction.assignment,
            evidence.target_assignment,
        ),
        "claim_policy": {
            "assignment_matrix_is_single_source_of_truth": True,
            "hard_object_id_is_derived": True,
            "model_mvp_is_inference_handoff": True,
            "does_not_claim_identity_gate_pass": True,
            "does_not_claim_reality_gate_pass": True,
            "does_not_claim_world_model": True,
        },
        "non_goals": {
            "trains_model": False,
            "uses_renderer_loss": False,
            "uses_gpu": False,
            "uses_slot_attention": False,
            "uses_transformer": False,
            "uses_replay_buffer": False,
            "uses_diffusion": False,
            "uses_dynamics_model": False,
            "mutates_viewer_defaults": False,
        },
    }
    if include_assignment:
        payload["assignment"]["matrix"] = np.round(prediction.assignment, 6).tolist()
        payload["assignment"]["derived_object_ids"] = (
            projection.derived_object_ids.astype(int).tolist()
        )
    return validate_objectstate_assignment_mvp_summary(payload)


def validate_objectstate_assignment_mvp_summary(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TypeError("ObjectState assignment MVP summary must be a mapping")
    if payload.get("schema") != OBJECTSTATE_ASSIGNMENT_MVP_SCHEMA:
        raise ValueError(f"unsupported ObjectState assignment MVP schema: {payload.get('schema')}")
    if payload.get("kind") != "objectstate_assignment_mvp":
        raise ValueError("ObjectState assignment MVP summary kind is unsupported")
    if payload.get("status") != "objectstate_assignment_mvp_ready":
        raise ValueError("ObjectState assignment MVP summary status is unsupported")
    _require_mapping(payload, "input")
    solver = _require_mapping(payload, "solver")
    if solver.get("schema") != ASSIGNMENT_SOLVER_V2_STATE_SCHEMA:
        raise ValueError("ObjectState assignment MVP solver schema is unsupported")
    assignment = _require_mapping(payload, "assignment")
    if assignment.get("schema") != ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA:
        raise ValueError("ObjectState assignment MVP assignment schema is unsupported")
    shape = assignment.get("shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 1 for item in shape)
    ):
        raise ValueError("ObjectState assignment MVP assignment shape must be [N,K]")
    for key in (
        "mean_confidence",
        "min_confidence",
        "mean_normalized_entropy",
        "effective_slots",
    ):
        _finite_number(assignment.get(key), f"assignment.{key}")
    if assignment.get("row_normalized") is not True:
        raise ValueError("ObjectState assignment MVP requires row_normalized assignment")
    projection = _require_mapping(payload, "projection")
    if projection.get("object_state_count") != shape[1]:
        raise ValueError("ObjectState assignment MVP projection count must match slots")
    if projection.get("evidence_count") != shape[0]:
        raise ValueError("ObjectState assignment MVP projection evidence_count must match N")
    target_metrics = payload.get("target_metrics")
    if target_metrics is not None:
        if not isinstance(target_metrics, Mapping):
            raise ValueError("ObjectState assignment MVP target_metrics must be mapping or null")
        for key in ("mean_best_iou", "ari", "purity"):
            value = _finite_number(target_metrics.get(key), f"target_metrics.{key}")
            if value < -1.0 or value > 1.0:
                raise ValueError(f"target metric {key} must be in [-1,1]")
    claim_policy = _require_mapping(payload, "claim_policy")
    if (
        not claim_policy.get("assignment_matrix_is_single_source_of_truth")
        or not claim_policy.get("hard_object_id_is_derived")
        or not claim_policy.get("model_mvp_is_inference_handoff")
        or not claim_policy.get("does_not_claim_identity_gate_pass")
        or not claim_policy.get("does_not_claim_reality_gate_pass")
        or not claim_policy.get("does_not_claim_world_model")
    ):
        raise ValueError("ObjectState assignment MVP summary must preserve claim policy")
    non_goals = _require_mapping(payload, "non_goals")
    if any(bool(value) for value in non_goals.values()):
        raise ValueError("ObjectState assignment MVP summary cannot claim non-goals")
    if "matrix" in assignment:
        validate_assignment_matrix(
            np.asarray(assignment["matrix"], dtype=np.float32),
            evidence_count=shape[0],
        )
    return dict(payload)


def _projection_summary(projection: ObjectStateProjection) -> dict[str, Any]:
    states = []
    for state in projection.states:
        states.append(
            {
                "id": int(state.id),
                "status": state.status,
                "slot_mass": float(state.slot_mass),
                "mass_fraction": float(state.mass_fraction),
                "confidence": float(state.confidence),
                "assignment_entropy": float(state.assignment_entropy),
                "centroid": np.round(state.centroid, 6).tolist(),
                "bbox": np.round(state.bbox, 6).tolist(),
                "diagnostics": list(state.diagnostics),
            }
        )
    return {
        "type": "ObjectStateProjection",
        "evidence_count": projection.evidence_count,
        "object_state_count": len(projection.states),
        "active_state_count": sum(1 for state in projection.states if state.status == "active"),
        "derived_object_id_source": "argmax(A[N,K])",
        "states": states,
    }


def _target_metrics(
    assignment: np.ndarray,
    target_assignment: np.ndarray | None,
) -> dict[str, Any] | None:
    if target_assignment is None:
        return None
    target = validate_assignment_matrix(target_assignment, evidence_count=assignment.shape[0])
    predicted_labels = np.argmax(assignment, axis=1).astype(np.int64, copy=False)
    target_labels = np.argmax(target, axis=1).astype(np.int64, copy=False)
    return {
        "target_slots": int(target.shape[1]),
        "mean_best_iou": _mean_best_iou(predicted_labels, target_labels),
        "ari": _adjusted_rand_index(predicted_labels, target_labels),
        "purity": _purity(predicted_labels, target_labels),
    }


def _mean_best_iou(predicted: np.ndarray, target: np.ndarray) -> float:
    target_slots = sorted(int(item) for item in np.unique(target))
    predicted_slots = sorted(int(item) for item in np.unique(predicted))
    if not target_slots or not predicted_slots:
        return 0.0
    scores = []
    for target_slot in target_slots:
        target_mask = target == target_slot
        best = 0.0
        for predicted_slot in predicted_slots:
            predicted_mask = predicted == predicted_slot
            union = np.logical_or(target_mask, predicted_mask).sum()
            if union == 0:
                continue
            intersection = np.logical_and(target_mask, predicted_mask).sum()
            best = max(best, float(intersection / union))
        scores.append(best)
    return float(np.mean(scores)) if scores else 0.0


def _purity(predicted: np.ndarray, target: np.ndarray) -> float:
    total = predicted.shape[0]
    if total == 0:
        return 0.0
    correct = 0
    for predicted_slot in np.unique(predicted):
        target_for_slot = target[predicted == predicted_slot]
        if target_for_slot.size == 0:
            continue
        counts = np.bincount(target_for_slot.astype(np.int64))
        correct += int(np.max(counts))
    return float(correct / total)


def _adjusted_rand_index(labels_a: np.ndarray, labels_b: np.ndarray) -> float:
    if labels_a.shape[0] != labels_b.shape[0]:
        raise ValueError("ARI labels must have the same length")
    n = int(labels_a.shape[0])
    if n < 2:
        return 1.0
    a_values, a_inverse = np.unique(labels_a, return_inverse=True)
    b_values, b_inverse = np.unique(labels_b, return_inverse=True)
    contingency = np.zeros((a_values.shape[0], b_values.shape[0]), dtype=np.int64)
    for a_idx, b_idx in zip(a_inverse, b_inverse, strict=True):
        contingency[int(a_idx), int(b_idx)] += 1
    sum_comb = float(np.sum(_comb2(contingency)))
    row_comb = float(np.sum(_comb2(contingency.sum(axis=1))))
    col_comb = float(np.sum(_comb2(contingency.sum(axis=0))))
    total_comb = float(_comb2(np.asarray([n], dtype=np.int64))[0])
    if total_comb == 0.0:
        return 1.0
    expected = row_comb * col_comb / total_comb
    maximum = 0.5 * (row_comb + col_comb)
    denominator = maximum - expected
    if abs(denominator) < 1e-12:
        return 1.0 if abs(sum_comb - maximum) < 1e-12 else 0.0
    return float((sum_comb - expected) / denominator)


def _comb2(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float64, copy=False)
    return values * (values - 1.0) / 2.0


def _require_mapping(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"ObjectState assignment MVP summary requires {key}")
    return value


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number
