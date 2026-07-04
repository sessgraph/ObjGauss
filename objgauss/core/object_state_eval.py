from __future__ import annotations

from dataclasses import replace
from typing import Any, Sequence

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_emergence_solver import (
    ObjectEmergenceEvidence,
    ObjectEmergenceSolverState,
    predict_object_emergence_assignment,
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
from objgauss.core.solver_decoder_training import (
    SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA,
    solver_decoder_joint_states_from_dict,
    validate_solver_decoder_joint_checkpoint,
)
from objgauss.core.trainable_kernel import TrainableKernelFrame, TrainableKernelSample

OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA = "objgauss-objectstate-checkpoint-eval-v1"
_BORDERLINE_ENTROPY_MARGIN = 0.03


def evaluate_solver_decoder_object_states(
    sample: TrainableKernelSample,
    checkpoint: dict[str, Any],
    *,
    entropy_threshold: float = 0.6,
    purity_threshold: float = 0.8,
    collapse_mass_fraction: float = 0.9,
    assignment_confidence_floor: float = 0.4,
    solver_temperature: float | None = None,
) -> dict[str, Any]:
    """Evaluate ObjectState emergence from a solver-decoder joint checkpoint.

    The evaluator is intentionally read-only: it replays the checkpoint solver
    over the supplied sample, projects `A[N,K]` into ObjectState, and reports
    stability gates without running optimizer steps or renderer loss.
    """

    checkpoint = validate_solver_decoder_joint_checkpoint(checkpoint)
    _validate_eval_thresholds(
        entropy_threshold=entropy_threshold,
        purity_threshold=purity_threshold,
        collapse_mass_fraction=collapse_mass_fraction,
        assignment_confidence_floor=assignment_confidence_floor,
    )
    solver_state, decoder_state = solver_decoder_joint_states_from_dict(checkpoint)
    solver_state = _solver_state_with_temperature(
        solver_state,
        solver_temperature=solver_temperature,
    )
    if solver_state.config.slots != sample.slots:
        raise ValueError("checkpoint solver slots must match sample slots")

    frames = []
    projections = []
    for frame_index, frame in enumerate(sample.frames):
        evidence = ObjectEmergenceEvidence(
            positions=frame.positions,
            features=frame.features,
            target_assignment=frame.target_assignment,
            frame_index=frame_index,
            source="solver_decoder_joint_checkpoint_eval",
        )
        prediction = predict_object_emergence_assignment(evidence, solver_state)
        projection = project_object_states(
            _frame_cloud(frame),
            prediction.assignment,
            evidence_features=frame.features,
        )
        projections.append(projection)
        purity_labels = _target_labels(frame)
        stability = object_state_stability_report(
            projection,
            purity_labels=purity_labels,
            collapse_mass_fraction=collapse_mass_fraction,
            assignment_confidence_floor=assignment_confidence_floor,
            purity_floor=purity_threshold,
        )
        frames.append(
            _frame_eval_summary(
                frame_index=frame_index,
                prediction=prediction.as_dict(include_assignment=False),
                stability=stability,
                slots=_slot_eval_summary(stability),
            )
        )

    temporal = _temporal_eval_summary(projections)
    aggregate = _aggregate_eval(frames, temporal=temporal)
    gates = _eval_gates(
        aggregate,
        entropy_threshold=entropy_threshold,
        purity_threshold=purity_threshold,
    )
    status = _eval_status(gates)
    training = checkpoint["training"]
    summary = {
        "schema": OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA,
        "kind": "solver_decoder_objectstate_eval",
        "status": status,
        "checkpoint_schema": SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA,
        "source": {
            "input": checkpoint.get("source", {}).get("input"),
            "source_gaussians": checkpoint.get("source", {}).get("source_gaussians"),
            "sampled_gaussians": sample.sampled_count,
            "target_source": sample.target_source,
            "object_id_mapping": {
                str(object_id): int(slot)
                for object_id, slot in sample.object_id_mapping.items()
            },
        },
        "sample": sample.as_dict(),
        "solver": {
            "step": int(solver_state.step),
            "slots": int(solver_state.config.slots),
            "feature_dim": int(solver_state.config.feature_dim),
            "temperature": float(solver_state.config.temperature),
            "temperature_override": solver_temperature is not None,
            "source": solver_state.source,
        },
        "decoder": {
            "step": int(decoder_state.step),
            "slots": int(decoder_state.slots),
            "source": decoder_state.source,
        },
        "training_losses": {
            "initial": training["initial_loss"],
            "final": training["final_loss"],
            "loss_decreased": bool(training["loss_decreased"]),
            "image_render_loss_decreased": bool(training["image_render_loss_decreased"]),
            "object_loss_decreased": bool(training["object_loss_decreased"]),
        },
        "thresholds": {
            "entropy": float(entropy_threshold),
            "entropy_borderline_margin": float(_BORDERLINE_ENTROPY_MARGIN),
            "purity": float(purity_threshold),
            "collapse_mass_fraction": float(collapse_mass_fraction),
            "assignment_confidence_floor": float(assignment_confidence_floor),
        },
        "aggregate": aggregate,
        "gates": gates,
        "frames": frames,
        "temporal": temporal,
        "trained_fields": list(checkpoint["trained_fields"]),
        "frozen_fields": list(checkpoint["frozen_fields"]),
        "gpu_policy": checkpoint["gpu_policy"],
        "export_policy": {
            "repository_write": "do_not_commit_eval_outputs",
            "intended_locations": ["/tmp", "ignored outputs/"],
            "large_artifacts": "keep_out_of_git",
        },
    }
    return validate_objectstate_checkpoint_eval(summary)


def validate_objectstate_checkpoint_eval(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("objectstate checkpoint eval payload must be a dict")
    if payload.get("schema") != OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA:
        raise ValueError(f"unsupported objectstate checkpoint eval schema: {payload.get('schema')}")
    if payload.get("kind") != "solver_decoder_objectstate_eval":
        raise ValueError("objectstate checkpoint eval kind must be solver_decoder_objectstate_eval")
    if payload.get("checkpoint_schema") != SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA:
        raise ValueError("objectstate checkpoint eval must reference a solver-decoder joint checkpoint")
    if payload.get("status") not in {
        "objectstate_eval_pass",
        "objectstate_eval_borderline",
        "objectstate_eval_fail",
    }:
        raise ValueError("objectstate checkpoint eval status is unsupported")
    for key in ("aggregate", "gates", "frames", "thresholds", "solver", "decoder"):
        if key not in payload:
            raise ValueError(f"objectstate checkpoint eval missing {key}")
    if not isinstance(payload["frames"], list) or not payload["frames"]:
        raise ValueError("objectstate checkpoint eval requires at least one frame")
    gates = payload["gates"]
    for key in ("entropy_pass", "no_collapse_pass", "purity_pass"):
        if key not in gates:
            raise ValueError(f"objectstate checkpoint eval gates missing {key}")
    aggregate = payload["aggregate"]
    for key in (
        "mean_normalized_entropy",
        "max_mean_normalized_entropy",
        "assignment_confidence",
        "effective_slots",
        "max_dominant_slot_mass_fraction",
    ):
        float(aggregate[key])
    return payload


def _validate_eval_thresholds(
    *,
    entropy_threshold: float,
    purity_threshold: float,
    collapse_mass_fraction: float,
    assignment_confidence_floor: float,
) -> None:
    for name, value in {
        "entropy_threshold": entropy_threshold,
        "purity_threshold": purity_threshold,
        "collapse_mass_fraction": collapse_mass_fraction,
        "assignment_confidence_floor": assignment_confidence_floor,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")


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


def _frame_eval_summary(
    *,
    frame_index: int,
    prediction: dict[str, Any],
    stability: ObjectStabilityReport,
    slots: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "frame_index": int(frame_index),
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
    }


def _eval_gates(
    aggregate: dict[str, Any],
    *,
    entropy_threshold: float,
    purity_threshold: float,
) -> dict[str, Any]:
    max_entropy = float(aggregate["max_mean_normalized_entropy"])
    purity = aggregate.get("min_object_purity")
    entropy_borderline = max_entropy <= float(entropy_threshold) + _BORDERLINE_ENTROPY_MARGIN
    return {
        "entropy_pass": bool(max_entropy <= float(entropy_threshold)),
        "entropy_borderline": bool(entropy_borderline),
        "no_collapse_pass": not bool(aggregate["slot_collapse"]),
        "purity_pass": None if purity is None else bool(float(purity) >= float(purity_threshold)),
    }


def _eval_status(gates: dict[str, Any]) -> str:
    purity_ok = gates["purity_pass"] is None or gates["purity_pass"] is True
    if gates["entropy_pass"] and gates["no_collapse_pass"] and purity_ok:
        return "objectstate_eval_pass"
    if gates["entropy_borderline"] and gates["no_collapse_pass"] and purity_ok:
        return "objectstate_eval_borderline"
    return "objectstate_eval_fail"


def _temporal_eval_summary(projections: Sequence[Any]) -> dict[str, Any]:
    if len(projections) < 2:
        return {
            "pair_count": 0,
            "matched_pair_count": 0,
            "mean_temporal_drift": 0.0,
            "max_temporal_drift": 0.0,
            "pairs": [],
        }
    pair_summaries = []
    mean_drifts = []
    max_drifts = []
    matched = 0
    for index in range(1, len(projections)):
        report = match_object_states(projections[index - 1], projections[index], include_inactive=True)
        pair_summaries.append(_temporal_pair_summary(index - 1, index, report))
        mean_drifts.append(report.mean_temporal_drift)
        max_drifts.append(report.max_temporal_drift)
        matched += len(report.matches)
    return {
        "pair_count": len(pair_summaries),
        "matched_pair_count": int(matched),
        "mean_temporal_drift": float(np.mean(mean_drifts)) if mean_drifts else 0.0,
        "max_temporal_drift": float(np.max(max_drifts)) if max_drifts else 0.0,
        "pairs": pair_summaries,
    }


def _temporal_pair_summary(
    previous_index: int,
    current_index: int,
    report: ObjectTemporalMatchReport,
) -> dict[str, Any]:
    return {
        "previous_frame_index": int(previous_index),
        "current_frame_index": int(current_index),
        "matched_pair_count": len(report.matches),
        "unmatched_previous": list(report.unmatched_previous),
        "unmatched_current": list(report.unmatched_current),
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


def _target_labels(frame: TrainableKernelFrame) -> np.ndarray | None:
    if frame.target_assignment is None:
        return None
    assignment = validate_assignment_matrix(frame.target_assignment)
    return np.argmax(assignment, axis=1).astype(np.int32, copy=False)


def _frame_cloud(frame: TrainableKernelFrame) -> GaussianCloud:
    positions = np.asarray(frame.positions, dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("frame.positions must have shape N x 3")
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
    return GaussianCloud(vertices=vertices, source_format="objectstate-checkpoint-eval")
