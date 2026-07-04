from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from objgauss.core.assignment_evidence import (
    AssignmentEvidenceBatch,
    validate_assignment_evidence_batch,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_emergence_solver import (
    OBJECT_EMERGENCE_SOLVER_CHECKPOINT_SCHEMA,
    OBJECT_EMERGENCE_SOLVER_STATE_SCHEMA,
    ObjectEmergenceEvidence,
    ObjectEmergenceSolverState,
    object_emergence_solver_state_from_dict,
    predict_object_emergence_assignment,
    validate_object_emergence_solver_checkpoint,
    validate_object_emergence_solver_state,
)
from objgauss.core.object_state import (
    ObjectStabilityReport,
    ObjectTemporalMatchReport,
    match_object_states,
    object_state_stability_report,
    project_object_states,
    validate_assignment_matrix,
)

ASSIGNMENT_STABILITY_EVAL_SCHEMA = "objgauss-assignment-stability-eval-v1"
_BORDERLINE_ENTROPY_MARGIN = 0.03


def evaluate_assignment_stability(
    evidence_batches: Sequence[AssignmentEvidenceBatch],
    solver: ObjectEmergenceSolverState | dict[str, Any],
    *,
    entropy_threshold: float = 0.6,
    purity_threshold: float = 0.8,
    collapse_mass_fraction: float = 0.9,
    assignment_confidence_floor: float = 0.4,
    id_stability_threshold: float = 0.7,
    temporal_drift_threshold: float | None = None,
    solver_temperature: float | None = None,
) -> dict[str, Any]:
    """Replay an assignment solver and report ObjectState stability gates."""

    batches = tuple(validate_assignment_evidence_batch(batch) for batch in evidence_batches)
    if not batches:
        raise ValueError("evidence_batches must contain at least one batch")
    _validate_eval_thresholds(
        entropy_threshold=entropy_threshold,
        purity_threshold=purity_threshold,
        collapse_mass_fraction=collapse_mass_fraction,
        assignment_confidence_floor=assignment_confidence_floor,
        id_stability_threshold=id_stability_threshold,
        temporal_drift_threshold=temporal_drift_threshold,
    )
    solver_state, solver_schema, checkpoint = _solver_state_from_input(solver)
    solver_state = _solver_state_with_temperature(
        solver_state,
        solver_temperature=solver_temperature,
    )

    frames = []
    projections = []
    for batch in batches:
        evidence = _object_emergence_evidence_from_assignment_batch(batch)
        prediction = predict_object_emergence_assignment(evidence, solver_state)
        projection = project_object_states(
            _batch_cloud(batch),
            prediction.assignment,
            evidence_features=batch.features,
        )
        projections.append(projection)
        purity_labels = _target_labels(batch)
        stability = object_state_stability_report(
            projection,
            purity_labels=purity_labels,
            collapse_mass_fraction=collapse_mass_fraction,
            assignment_confidence_floor=assignment_confidence_floor,
            purity_floor=purity_threshold,
        )
        frames.append(
            _frame_eval_summary(
                batch=batch,
                prediction=prediction.as_dict(include_assignment=False),
                stability=stability,
                slots=_slot_eval_summary(stability),
            )
        )

    temporal = _temporal_eval_summary(projections)
    aggregate = _aggregate_eval(frames, temporal=temporal)
    gates = _eval_gates(
        aggregate,
        temporal,
        entropy_threshold=entropy_threshold,
        purity_threshold=purity_threshold,
        id_stability_threshold=id_stability_threshold,
        temporal_drift_threshold=temporal_drift_threshold,
    )
    status = _eval_status(gates)
    summary = {
        "schema": ASSIGNMENT_STABILITY_EVAL_SCHEMA,
        "kind": "assignment_stability_eval",
        "status": status,
        "checkpoint_schema": solver_schema,
        "source": _source_summary(checkpoint),
        "solver": {
            "step": int(solver_state.step),
            "slots": int(solver_state.config.slots),
            "feature_dim": int(solver_state.config.feature_dim),
            "position_dim": int(solver_state.config.position_dim),
            "temperature": float(solver_state.config.temperature),
            "temperature_override": solver_temperature is not None,
            "source": solver_state.source,
        },
        "thresholds": {
            "entropy": float(entropy_threshold),
            "entropy_borderline_margin": float(_BORDERLINE_ENTROPY_MARGIN),
            "purity": float(purity_threshold),
            "collapse_mass_fraction": float(collapse_mass_fraction),
            "assignment_confidence_floor": float(assignment_confidence_floor),
            "id_stability": float(id_stability_threshold),
            "temporal_drift": None
            if temporal_drift_threshold is None
            else float(temporal_drift_threshold),
        },
        "aggregate": aggregate,
        "gates": gates,
        "frames": frames,
        "temporal": temporal,
        "training_losses": _training_loss_summary(checkpoint),
        "gpu_policy": _gpu_policy_summary(checkpoint),
        "export_policy": {
            "repository_write": "do_not_commit_eval_outputs",
            "intended_locations": ["/tmp", "ignored outputs/"],
            "large_artifacts": "keep_out_of_git",
        },
    }
    return validate_assignment_stability_eval(summary)


def validate_assignment_stability_eval(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("assignment stability eval payload must be a dict")
    if payload.get("schema") != ASSIGNMENT_STABILITY_EVAL_SCHEMA:
        raise ValueError(f"unsupported assignment stability eval schema: {payload.get('schema')}")
    if payload.get("kind") != "assignment_stability_eval":
        raise ValueError("assignment stability eval kind must be assignment_stability_eval")
    if payload.get("status") not in {
        "assignment_stability_eval_pass",
        "assignment_stability_eval_borderline",
        "assignment_stability_eval_fail",
    }:
        raise ValueError("assignment stability eval status is unsupported")
    for key in ("aggregate", "gates", "frames", "thresholds", "solver", "temporal"):
        if key not in payload:
            raise ValueError(f"assignment stability eval missing {key}")
    if not isinstance(payload["frames"], list) or not payload["frames"]:
        raise ValueError("assignment stability eval requires at least one frame")
    gates = payload["gates"]
    for key in (
        "entropy_pass",
        "entropy_borderline",
        "no_collapse_pass",
        "purity_pass",
        "id_stability_pass",
        "temporal_drift_pass",
    ):
        if key not in gates:
            raise ValueError(f"assignment stability eval gates missing {key}")
    aggregate = payload["aggregate"]
    for key in (
        "mean_normalized_entropy",
        "max_mean_normalized_entropy",
        "assignment_confidence",
        "effective_slots",
        "max_dominant_slot_mass_fraction",
        "id_stability",
    ):
        float(aggregate[key])
    return payload


def _solver_state_from_input(
    solver: ObjectEmergenceSolverState | dict[str, Any],
) -> tuple[ObjectEmergenceSolverState, str, dict[str, Any] | None]:
    if isinstance(solver, ObjectEmergenceSolverState):
        return validate_object_emergence_solver_state(solver), solver.schema, None
    if not isinstance(solver, dict):
        raise TypeError("solver must be an ObjectEmergenceSolverState or checkpoint dict")
    schema = str(solver.get("schema"))
    if schema == OBJECT_EMERGENCE_SOLVER_CHECKPOINT_SCHEMA:
        validate_object_emergence_solver_checkpoint(solver)
    if schema not in {OBJECT_EMERGENCE_SOLVER_CHECKPOINT_SCHEMA, OBJECT_EMERGENCE_SOLVER_STATE_SCHEMA}:
        raise ValueError(f"unsupported assignment solver schema: {schema}")
    return object_emergence_solver_state_from_dict(solver), schema, solver


def _solver_state_with_temperature(
    state: ObjectEmergenceSolverState,
    *,
    solver_temperature: float | None,
) -> ObjectEmergenceSolverState:
    if solver_temperature is None:
        return state
    if solver_temperature <= 0:
        raise ValueError("solver_temperature must be > 0")
    config = replace(state.config, temperature=float(solver_temperature))
    return validate_object_emergence_solver_state(
        replace(
            state,
            config=config,
            source=f"{state.source}|temperature_eval_override",
        )
    )


def _validate_eval_thresholds(
    *,
    entropy_threshold: float,
    purity_threshold: float,
    collapse_mass_fraction: float,
    assignment_confidence_floor: float,
    id_stability_threshold: float,
    temporal_drift_threshold: float | None,
) -> None:
    for name, value in {
        "entropy_threshold": entropy_threshold,
        "purity_threshold": purity_threshold,
        "collapse_mass_fraction": collapse_mass_fraction,
        "assignment_confidence_floor": assignment_confidence_floor,
        "id_stability_threshold": id_stability_threshold,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    if temporal_drift_threshold is not None and temporal_drift_threshold < 0.0:
        raise ValueError("temporal_drift_threshold must be >= 0")


def _object_emergence_evidence_from_assignment_batch(
    batch: AssignmentEvidenceBatch,
) -> ObjectEmergenceEvidence:
    return ObjectEmergenceEvidence(
        positions=batch.positions,
        features=batch.features,
        target_assignment=batch.target_assignment,
        frame_index=batch.frame_index,
        source=f"assignment_stability_eval:{batch.source}",
    )


def _frame_eval_summary(
    *,
    batch: AssignmentEvidenceBatch,
    prediction: dict[str, Any],
    stability: ObjectStabilityReport,
    slots: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "frame_index": int(batch.frame_index),
        "evidence": batch.as_dict(),
        "evidence_count": int(stability.evidence_count),
        "slots": int(stability.slots),
        "mean_normalized_entropy": float(stability.mean_normalized_entropy),
        "assignment_confidence": float(stability.assignment_confidence),
        "effective_slots": float(stability.effective_slots),
        "slot_mass": list(stability.slot_mass),
        "slot_mass_fraction": list(stability.slot_mass_fraction),
        "inactive_slots": list(stability.inactive_slots),
        "low_confidence_slots": list(stability.low_confidence_slots),
        "mixed_slots": list(stability.mixed_slots),
        "slot_collapse": bool(stability.slot_collapse),
        "dominant_slot": stability.dominant_slot,
        "dominant_slot_mass_fraction": float(stability.dominant_slot_mass_fraction),
        "object_purity": stability.object_purity,
        "per_slot_purity": list(stability.per_slot_purity),
        "prediction": prediction,
        "slot_summaries": slots,
        "diagnostics": list(stability.diagnostics),
    }


def _slot_eval_summary(stability: ObjectStabilityReport) -> list[dict[str, Any]]:
    summaries = []
    for slot, mass in enumerate(stability.slot_mass):
        purity = stability.per_slot_purity[slot] if slot < len(stability.per_slot_purity) else None
        summaries.append(
            {
                "id": int(slot),
                "mass": float(mass),
                "mass_fraction": float(stability.slot_mass_fraction[slot]),
                "purity": purity,
                "inactive": int(slot) in stability.inactive_slots,
                "low_confidence": int(slot) in stability.low_confidence_slots,
                "mixed": int(slot) in stability.mixed_slots,
            }
        )
    return summaries


def _aggregate_eval(
    frames: Sequence[dict[str, Any]],
    *,
    temporal: dict[str, Any],
) -> dict[str, Any]:
    entropies = np.asarray([frame["mean_normalized_entropy"] for frame in frames], dtype=np.float32)
    confidences = np.asarray([frame["assignment_confidence"] for frame in frames], dtype=np.float32)
    effective_slots = np.asarray([frame["effective_slots"] for frame in frames], dtype=np.float32)
    dominant = np.asarray([frame["dominant_slot_mass_fraction"] for frame in frames], dtype=np.float32)
    purities = [
        float(frame["object_purity"])
        for frame in frames
        if frame.get("object_purity") is not None
    ]
    return {
        "frame_count": len(frames),
        "evidence_count": int(sum(int(frame["evidence_count"]) for frame in frames)),
        "mean_normalized_entropy": float(np.mean(entropies)),
        "max_mean_normalized_entropy": float(np.max(entropies)),
        "assignment_confidence": float(np.mean(confidences)),
        "min_assignment_confidence": float(np.min(confidences)),
        "effective_slots": float(np.mean(effective_slots)),
        "min_effective_slots": float(np.min(effective_slots)),
        "slot_collapse": any(bool(frame["slot_collapse"]) for frame in frames),
        "max_dominant_slot_mass_fraction": float(np.max(dominant)),
        "object_purity": None if not purities else float(np.mean(purities)),
        "min_object_purity": None if not purities else float(np.min(purities)),
        "inactive_slot_count": int(sum(len(frame["inactive_slots"]) for frame in frames)),
        "low_confidence_slot_count": int(sum(len(frame["low_confidence_slots"]) for frame in frames)),
        "mixed_slot_count": int(sum(len(frame["mixed_slots"]) for frame in frames)),
        "temporal_mean_drift": temporal["mean_temporal_drift"],
        "temporal_max_drift": temporal["max_temporal_drift"],
        "id_stability": temporal["id_stability"],
        "min_id_stability": temporal["min_id_stability"],
    }


def _eval_gates(
    aggregate: dict[str, Any],
    temporal: dict[str, Any],
    *,
    entropy_threshold: float,
    purity_threshold: float,
    id_stability_threshold: float,
    temporal_drift_threshold: float | None,
) -> dict[str, Any]:
    max_entropy = float(aggregate["max_mean_normalized_entropy"])
    purity = aggregate.get("min_object_purity")
    entropy_borderline = max_entropy <= float(entropy_threshold) + _BORDERLINE_ENTROPY_MARGIN
    id_stability_pass = (
        None
        if int(temporal["pair_count"]) == 0
        else bool(float(aggregate["min_id_stability"]) >= float(id_stability_threshold))
    )
    temporal_drift_pass = (
        None
        if temporal_drift_threshold is None or int(temporal["pair_count"]) == 0
        else bool(float(aggregate["temporal_max_drift"]) <= float(temporal_drift_threshold))
    )
    return {
        "entropy_pass": bool(max_entropy <= float(entropy_threshold)),
        "entropy_borderline": bool(entropy_borderline),
        "no_collapse_pass": not bool(aggregate["slot_collapse"]),
        "purity_pass": None if purity is None else bool(float(purity) >= float(purity_threshold)),
        "id_stability_pass": id_stability_pass,
        "temporal_drift_pass": temporal_drift_pass,
    }


def _eval_status(gates: dict[str, Any]) -> str:
    purity_ok = gates["purity_pass"] is None or gates["purity_pass"] is True
    id_ok = gates["id_stability_pass"] is None or gates["id_stability_pass"] is True
    drift_ok = gates["temporal_drift_pass"] is None or gates["temporal_drift_pass"] is True
    if gates["entropy_pass"] and gates["no_collapse_pass"] and purity_ok and id_ok and drift_ok:
        return "assignment_stability_eval_pass"
    if gates["entropy_borderline"] and gates["no_collapse_pass"] and purity_ok and id_ok and drift_ok:
        return "assignment_stability_eval_borderline"
    return "assignment_stability_eval_fail"


def _temporal_eval_summary(projections: Sequence[Any]) -> dict[str, Any]:
    if len(projections) < 2:
        return {
            "pair_count": 0,
            "matched_pair_count": 0,
            "mean_temporal_drift": 0.0,
            "max_temporal_drift": 0.0,
            "id_stability": 1.0,
            "min_id_stability": 1.0,
            "pairs": [],
        }
    pair_summaries = []
    mean_drifts = []
    max_drifts = []
    id_stability = []
    matched = 0
    for index in range(1, len(projections)):
        report = match_object_states(projections[index - 1], projections[index], include_inactive=False)
        pair = _temporal_pair_summary(index - 1, index, report)
        pair_summaries.append(pair)
        mean_drifts.append(report.mean_temporal_drift)
        max_drifts.append(report.max_temporal_drift)
        id_stability.append(pair["id_stability"])
        matched += len(report.matches)
    return {
        "pair_count": len(pair_summaries),
        "matched_pair_count": int(matched),
        "mean_temporal_drift": float(np.mean(mean_drifts)) if mean_drifts else 0.0,
        "max_temporal_drift": float(np.max(max_drifts)) if max_drifts else 0.0,
        "id_stability": float(np.mean(id_stability)) if id_stability else 1.0,
        "min_id_stability": float(np.min(id_stability)) if id_stability else 1.0,
        "pairs": pair_summaries,
    }


def _temporal_pair_summary(
    previous_index: int,
    current_index: int,
    report: ObjectTemporalMatchReport,
) -> dict[str, Any]:
    denominator = max(int(report.previous_count), int(report.current_count))
    id_stability = 1.0 if denominator == 0 else float(len(report.matches) / denominator)
    return {
        "previous_frame_index": int(previous_index),
        "current_frame_index": int(current_index),
        "matched_pair_count": len(report.matches),
        "previous_count": int(report.previous_count),
        "current_count": int(report.current_count),
        "unmatched_previous": list(report.unmatched_previous),
        "unmatched_current": list(report.unmatched_current),
        "id_stability": id_stability,
        "mean_temporal_drift": float(report.mean_temporal_drift),
        "max_temporal_drift": float(report.max_temporal_drift),
        "diagnostics": list(report.diagnostics),
        "matches": [
            {
                "previous_id": int(match.previous_id),
                "current_id": int(match.current_id),
                "cost": float(match.cost),
                "temporal_drift": float(match.temporal_drift),
                "bbox_iou": float(match.bbox_iou),
                "feature_distance": float(match.feature_distance),
                "mass_delta": float(match.mass_delta),
            }
            for match in report.matches
        ],
    }


def _target_labels(batch: AssignmentEvidenceBatch) -> np.ndarray | None:
    if batch.target_assignment is None:
        return None
    assignment = validate_assignment_matrix(batch.target_assignment)
    return np.argmax(assignment, axis=1).astype(np.int32, copy=False)


def _batch_cloud(batch: AssignmentEvidenceBatch) -> GaussianCloud:
    positions = np.asarray(batch.positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("batch.positions must have shape N x 3")
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
    return GaussianCloud(vertices=vertices, source_format="assignment-stability-eval")


def _source_summary(checkpoint: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(checkpoint, dict):
        return {}
    source = checkpoint.get("source")
    if not isinstance(source, dict):
        return {}
    return {
        "input": source.get("input"),
        "source_gaussians": source.get("source_gaussians"),
        "sampled_gaussians": source.get("sampled_gaussians"),
        "target_source": source.get("target_source"),
        "object_id_mapping": source.get("object_id_mapping", {}),
    }


def _training_loss_summary(checkpoint: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(checkpoint, dict):
        return None
    training = checkpoint.get("training")
    if not isinstance(training, dict):
        return None
    return {
        "initial": training.get("initial_loss"),
        "final": training.get("final_loss"),
        "loss_decreased": training.get("loss_decreased"),
        "assignment_loss_decreased": training.get("assignment_loss_decreased"),
    }


def _gpu_policy_summary(checkpoint: dict[str, Any] | None) -> dict[str, Any]:
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("gpu_policy"), dict):
        policy = dict(checkpoint["gpu_policy"])
        policy.setdefault("uses_gpu", False)
        policy.setdefault("vram_reserve_gb", 1)
        return policy
    return {
        "uses_gpu": False,
        "full_renderer_training": "not_used_assignment_eval",
        "vram_reserve_gb": 1,
    }
