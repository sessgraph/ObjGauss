from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from objgauss.pipelines.trainable_artifact import validate_trainable_kernel_model_artifact

TRAINABLE_QUALITY_REPORT_SCHEMA = "objgauss-object-state-quality-report-v1"

__all__ = (
    "TRAINABLE_QUALITY_REPORT_SCHEMA",
    "trainable_quality_report",
    "write_trainable_quality_report",
    "validate_trainable_quality_report",
)


def trainable_quality_report(
    artifact: dict[str, Any],
    *,
    report_id: str | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    validate_trainable_kernel_model_artifact(artifact)
    metrics = _trainable_quality_metrics(artifact)
    gates = _quality_gates(metrics)
    status = "pass" if all(gate["status"] == "pass" for gate in gates) else "warn"
    report = {
        "schema": TRAINABLE_QUALITY_REPORT_SCHEMA,
        "report_id": report_id or f"{artifact.get('label', 'trainable-kernel')}-quality",
        "status": status,
        "source": source
        or {
            "type": "trainable_kernel_model_artifact",
            "input": artifact.get("source", {}).get("input"),
        },
        "metrics": metrics,
        "gates": gates,
        "limitations": [
            "Deterministic quality report derived from trainable kernel Debug OS artifact fields.",
            "Metrics are debug evidence, not a production quality certificate.",
        ],
    }
    validate_trainable_quality_report(report)
    return report


def write_trainable_quality_report(
    path: str | Path,
    artifact: dict[str, Any],
    *,
    report_id: str | None = None,
    source: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = trainable_quality_report(artifact, report_id=report_id, source=source)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_trainable_quality_report(report: dict[str, Any]) -> bool:
    if not isinstance(report, dict):
        raise TypeError("trainable quality report must be a dict")
    if report.get("schema") != TRAINABLE_QUALITY_REPORT_SCHEMA:
        raise ValueError(f"unsupported trainable quality report schema: {report.get('schema')}")
    if report.get("status") not in {"pass", "warn"}:
        raise ValueError("trainable quality report status must be pass or warn")
    metrics = report.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("trainable quality report metrics must be a non-empty object")
    gates = report.get("gates")
    if not isinstance(gates, list) or not gates:
        raise ValueError("trainable quality report gates must be a non-empty list")
    for gate in gates:
        if not isinstance(gate, dict):
            raise ValueError("trainable quality report gates must be objects")
        if gate.get("status") not in {"pass", "warn"}:
            raise ValueError("trainable quality report gate status must be pass or warn")
    return True


def _trainable_quality_metrics(artifact: dict[str, Any]) -> dict[str, float]:
    object_state_frames = artifact.get("object_states") if isinstance(artifact.get("object_states"), list) else []
    assignment_frames = artifact.get("assignments") if isinstance(artifact.get("assignments"), list) else []
    slots = _positive_int(artifact.get("training", {}).get("slots")) or _assignment_slot_count(assignment_frames)
    metrics = {
        "assignment_entropy": _round6(_mean(_state_values(object_state_frames, "normalized_assignment_entropy"))),
        "slot_utilization": _round6(_slot_utilization(object_state_frames, slots)),
        "object_purity": _round6(_assignment_confidence(assignment_frames)),
        "temporal_drift": _round6(_temporal_drift(object_state_frames)),
        "assignment_jitter": _round6(_assignment_jitter(assignment_frames)),
        "bbox_stability": _round6(_bbox_stability(object_state_frames)),
        "spatial_compactness": _round6(_spatial_compactness(object_state_frames)),
    }
    return {key: value for key, value in metrics.items() if value is not None}


def _quality_gates(metrics: dict[str, float]) -> list[dict[str, Any]]:
    return [
        _gate("slot_utilization", metrics.get("slot_utilization"), 0.7, direction="gte"),
        _gate("assignment_entropy", metrics.get("assignment_entropy"), 0.5, direction="lte"),
        _gate("temporal_drift", metrics.get("temporal_drift"), 0.08, direction="lte"),
    ]


def _gate(name: str, value: float | None, threshold: float, *, direction: str) -> dict[str, Any]:
    passed = False
    if value is not None:
        passed = value >= threshold if direction == "gte" else value <= threshold
    return {
        "name": name,
        "status": "pass" if passed else "warn",
        "value": value,
        "threshold": threshold,
    }


def _state_values(frames: list[Any], key: str) -> list[float]:
    values: list[float] = []
    for frame in frames:
        states = frame.get("states") if isinstance(frame, dict) else None
        if not isinstance(states, list):
            continue
        for state in states:
            if isinstance(state, dict):
                value = _finite_float(state.get(key))
                if value is not None:
                    values.append(value)
    return values


def _slot_utilization(frames: list[Any], slots: int | None) -> float | None:
    if not slots:
        return None
    ratios: list[float] = []
    for frame in frames:
        states = frame.get("states") if isinstance(frame, dict) else None
        if not isinstance(states, list):
            continue
        active = sum(
            1
            for state in states
            if isinstance(state, dict)
            and _finite_float(state.get("slot_mass")) is not None
            and float(state.get("slot_mass")) > 0
            and not str(state.get("status", "")).startswith("inactive")
        )
        ratios.append(active / slots)
    return _mean(ratios)


def _assignment_confidence(frames: list[Any]) -> float | None:
    maxima: list[float] = []
    for matrix in _assignment_matrices(frames):
        if matrix.size:
            maxima.extend(np.max(matrix, axis=1).astype(float).tolist())
    return _mean(maxima)


def _temporal_drift(frames: list[Any]) -> float | None:
    deltas: list[float] = []
    for previous, current in zip(frames, frames[1:]):
        previous_states = _states_by_id(previous)
        current_states = _states_by_id(current)
        for state_id, previous_state in previous_states.items():
            current_state = current_states.get(state_id)
            if current_state is None:
                continue
            previous_centroid = _array(previous_state.get("centroid"), columns=3)
            current_centroid = _array(current_state.get("centroid"), columns=3)
            if previous_centroid is not None and current_centroid is not None:
                deltas.append(float(np.linalg.norm(current_centroid - previous_centroid)))
    return _mean(deltas)


def _assignment_jitter(frames: list[Any]) -> float | None:
    deltas: list[float] = []
    matrices = _assignment_matrices(frames)
    for previous, current in zip(matrices, matrices[1:]):
        if previous.shape == current.shape:
            deltas.append(float(np.mean(np.abs(current - previous))))
    return _mean(deltas)


def _bbox_stability(frames: list[Any]) -> float | None:
    deltas: list[float] = []
    for previous, current in zip(frames, frames[1:]):
        previous_states = _states_by_id(previous)
        current_states = _states_by_id(current)
        for state_id, previous_state in previous_states.items():
            current_state = current_states.get(state_id)
            if current_state is None:
                continue
            previous_bbox = _array(previous_state.get("bbox"), columns=6)
            current_bbox = _array(current_state.get("bbox"), columns=6)
            if previous_bbox is not None and current_bbox is not None:
                deltas.append(float(np.linalg.norm(current_bbox - previous_bbox)))
    mean_delta = _mean(deltas)
    return None if mean_delta is None else 1.0 / (1.0 + mean_delta)


def _spatial_compactness(frames: list[Any]) -> float | None:
    diagonals: list[float] = []
    for frame in frames:
        states = frame.get("states") if isinstance(frame, dict) else None
        if not isinstance(states, list):
            continue
        for state in states:
            if not isinstance(state, dict):
                continue
            bbox = _array(state.get("bbox"), columns=6)
            if bbox is None:
                continue
            extent = np.maximum(bbox[3:] - bbox[:3], 0.0)
            diagonals.append(float(np.linalg.norm(extent)))
    mean_diagonal = _mean(diagonals)
    return None if mean_diagonal is None else 1.0 / (1.0 + mean_diagonal)


def _states_by_id(frame: Any) -> dict[int, dict[str, Any]]:
    states = frame.get("states") if isinstance(frame, dict) else None
    if not isinstance(states, list):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for state in states:
        if isinstance(state, dict):
            state_id = _positive_int(state.get("id"))
            if state_id is not None:
                result[state_id] = state
    return result


def _assignment_matrices(frames: list[Any]) -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for frame in frames:
        if not isinstance(frame, dict):
            continue
        matrix = frame.get("matrix")
        if not isinstance(matrix, list):
            continue
        array = np.asarray(matrix, dtype=float)
        if array.ndim == 2:
            matrices.append(array)
    return matrices


def _assignment_slot_count(frames: list[Any]) -> int | None:
    matrices = _assignment_matrices(frames)
    if not matrices:
        return None
    return int(matrices[0].shape[1])


def _array(value: Any, *, columns: int) -> np.ndarray | None:
    if not isinstance(value, list) or len(value) != columns:
        return None
    array = np.asarray(value, dtype=float)
    return array if array.shape == (columns,) and np.all(np.isfinite(array)) else None


def _mean(values: list[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(finite)) if finite else None


def _round6(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


def _finite_float(value: Any) -> float | None:
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    return resolved if math.isfinite(resolved) else None


def _positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        return None
    return resolved if resolved > 0 else None
