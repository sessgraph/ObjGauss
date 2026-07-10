from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.datasets.v2_stability_foundation import (
    SyntheticStabilityScenarioFixture,
    validate_synthetic_stability_scenario_fixture,
)

V2_STABILITY_DIAGNOSTICS_SCHEMA = "objgauss-v2-stability-diagnostics-v1"
V2_STABILITY_FAILURE_MODES = (
    "slot_swap",
    "identity_fragmentation",
    "object_merge",
    "background_absorption",
    "temporal_drift",
)
_DIAGNOSTIC_STATUS_CLEAN = "stability_diagnostics_clean"
_DIAGNOSTIC_STATUS_FAILURES = "stability_diagnostics_failures_detected"

__all__ = (
    "V2_STABILITY_DIAGNOSTICS_SCHEMA",
    "V2_STABILITY_FAILURE_MODES",
    "IdentitySlotObservation",
    "FailureModeEvent",
    "FailureModeClassifier",
    "SyntheticStabilityDiagnosticsReport",
    "diagnose_synthetic_stability_fixture",
    "expected_slots_for_synthetic_fixture",
    "validate_synthetic_stability_diagnostics_summary",
)


@dataclass(frozen=True)
class IdentitySlotObservation:
    frame_index: int
    oracle_object_id: int
    lineage_id: str
    expected_slot: int
    predicted_slot: int
    evidence_count: int
    mean_confidence: float

    @property
    def matches_expected(self) -> bool:
        return int(self.expected_slot) == int(self.predicted_slot)

    def as_dict(self) -> dict[str, Any]:
        return {
            "frame_index": int(self.frame_index),
            "oracle_object_id": int(self.oracle_object_id),
            "lineage_id": self.lineage_id,
            "expected_slot": int(self.expected_slot),
            "predicted_slot": int(self.predicted_slot),
            "evidence_count": int(self.evidence_count),
            "mean_confidence": float(self.mean_confidence),
            "matches_expected": bool(self.matches_expected),
        }


@dataclass(frozen=True)
class FailureModeEvent:
    mode: str
    frame_index: int | None
    oracle_object_ids: tuple[int, ...]
    expected_slots: tuple[int, ...]
    predicted_slots: tuple[int, ...]
    reason: str

    def as_dict(self) -> dict[str, Any]:
        if self.mode not in V2_STABILITY_FAILURE_MODES:
            raise ValueError(f"unsupported failure mode: {self.mode}")
        return {
            "mode": self.mode,
            "frame_index": None if self.frame_index is None else int(self.frame_index),
            "oracle_object_ids": [int(value) for value in self.oracle_object_ids],
            "expected_slots": [int(value) for value in self.expected_slots],
            "predicted_slots": [int(value) for value in self.predicted_slots],
            "reason": self.reason,
        }


@dataclass(frozen=True)
class FailureModeClassifier:
    background_slot: int = -1
    confidence_floor: float = 0.5

    def classify(
        self,
        observations: Sequence[IdentitySlotObservation],
    ) -> tuple[FailureModeEvent, ...]:
        checked = _validate_identity_slot_observations(observations)
        events: list[FailureModeEvent] = []
        events.extend(self._background_absorption_events(checked))
        events.extend(self._object_merge_events(checked))
        events.extend(self._slot_swap_events(checked))
        events.extend(self._identity_fragmentation_events(checked))
        events.extend(self._temporal_drift_events(checked))
        return tuple(events)

    def as_dict(self) -> dict[str, Any]:
        if self.confidence_floor < 0.0 or self.confidence_floor > 1.0:
            raise ValueError("confidence_floor must be in [0, 1]")
        return {
            "background_slot": int(self.background_slot),
            "confidence_floor": float(self.confidence_floor),
            "failure_modes": list(V2_STABILITY_FAILURE_MODES),
            "role": "diagnostic_only_not_gate",
        }

    def _background_absorption_events(
        self,
        observations: tuple[IdentitySlotObservation, ...],
    ) -> list[FailureModeEvent]:
        events = []
        for obs in observations:
            if int(obs.predicted_slot) == int(self.background_slot):
                reason = "visible identity assigned to background slot"
            elif float(obs.mean_confidence) < float(self.confidence_floor):
                reason = "visible identity assignment confidence below floor"
            else:
                continue
            events.append(
                FailureModeEvent(
                    mode="background_absorption",
                    frame_index=obs.frame_index,
                    oracle_object_ids=(obs.oracle_object_id,),
                    expected_slots=(obs.expected_slot,),
                    predicted_slots=(obs.predicted_slot,),
                    reason=reason,
                )
            )
        return events

    def _object_merge_events(
        self,
        observations: tuple[IdentitySlotObservation, ...],
    ) -> list[FailureModeEvent]:
        events = []
        for frame_index, frame_observations in _observations_by_frame(observations).items():
            by_predicted: dict[int, list[IdentitySlotObservation]] = {}
            for obs in frame_observations:
                if int(obs.predicted_slot) == int(self.background_slot):
                    continue
                by_predicted.setdefault(int(obs.predicted_slot), []).append(obs)
            for predicted_slot, group in sorted(by_predicted.items()):
                object_ids = tuple(sorted({int(obs.oracle_object_id) for obs in group}))
                expected_slots = tuple(sorted({int(obs.expected_slot) for obs in group}))
                if len(object_ids) <= 1 or len(expected_slots) <= 1:
                    continue
                events.append(
                    FailureModeEvent(
                        mode="object_merge",
                        frame_index=frame_index,
                        oracle_object_ids=object_ids,
                        expected_slots=expected_slots,
                        predicted_slots=(predicted_slot,),
                        reason="multiple oracle identities mapped to one predicted slot",
                    )
                )
        return events

    def _slot_swap_events(
        self,
        observations: tuple[IdentitySlotObservation, ...],
    ) -> list[FailureModeEvent]:
        events = []
        seen: set[tuple[int, int, int]] = set()
        for frame_index, frame_observations in _observations_by_frame(observations).items():
            by_expected = {int(obs.expected_slot): obs for obs in frame_observations}
            for obs in frame_observations:
                if int(obs.predicted_slot) == int(self.background_slot):
                    continue
                other = by_expected.get(int(obs.predicted_slot))
                if other is None or int(other.oracle_object_id) == int(obs.oracle_object_id):
                    continue
                if int(other.predicted_slot) != int(obs.expected_slot):
                    continue
                key = tuple(sorted((int(obs.oracle_object_id), int(other.oracle_object_id))))
                seen_key = (int(frame_index), key[0], key[1])
                if seen_key in seen:
                    continue
                seen.add(seen_key)
                events.append(
                    FailureModeEvent(
                        mode="slot_swap",
                        frame_index=frame_index,
                        oracle_object_ids=key,
                        expected_slots=tuple(sorted((int(obs.expected_slot), int(other.expected_slot)))),
                        predicted_slots=tuple(sorted((int(obs.predicted_slot), int(other.predicted_slot)))),
                        reason="two oracle identities exchanged expected slots",
                    )
                )
        return events

    def _identity_fragmentation_events(
        self,
        observations: tuple[IdentitySlotObservation, ...],
    ) -> list[FailureModeEvent]:
        events = []
        for object_id, object_observations in _observations_by_object(observations).items():
            predicted_slots = tuple(
                sorted(
                    {
                        int(obs.predicted_slot)
                        for obs in object_observations
                        if int(obs.predicted_slot) != int(self.background_slot)
                    }
                )
            )
            if len(predicted_slots) <= 1:
                continue
            first = object_observations[0]
            events.append(
                FailureModeEvent(
                    mode="identity_fragmentation",
                    frame_index=None,
                    oracle_object_ids=(object_id,),
                    expected_slots=(first.expected_slot,),
                    predicted_slots=predicted_slots,
                    reason="one oracle identity mapped to multiple predicted slots over time",
                )
            )
        return events

    def _temporal_drift_events(
        self,
        observations: tuple[IdentitySlotObservation, ...],
    ) -> list[FailureModeEvent]:
        events = []
        for object_id, object_observations in _observations_by_object(observations).items():
            ordered = sorted(object_observations, key=lambda obs: int(obs.frame_index))
            for previous, current in zip(ordered, ordered[1:]):
                if int(previous.predicted_slot) == int(self.background_slot):
                    continue
                if int(current.predicted_slot) == int(self.background_slot):
                    continue
                if int(previous.predicted_slot) == int(current.predicted_slot):
                    continue
                events.append(
                    FailureModeEvent(
                        mode="temporal_drift",
                        frame_index=current.frame_index,
                        oracle_object_ids=(object_id,),
                        expected_slots=(current.expected_slot,),
                        predicted_slots=(previous.predicted_slot, current.predicted_slot),
                        reason="one oracle identity changed predicted slot across visible observations",
                    )
                )
        return events


@dataclass(frozen=True)
class SyntheticStabilityDiagnosticsReport:
    fixture: SyntheticStabilityScenarioFixture
    classifier: FailureModeClassifier
    identity_observations: tuple[IdentitySlotObservation, ...]
    slot_transition_matrix: np.ndarray
    slot_transition_labels: tuple[int, ...]
    identity_confusion_graph: dict[str, Any]
    failure_modes: tuple[FailureModeEvent, ...]
    schema: str = V2_STABILITY_DIAGNOSTICS_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        summary = {
            "schema": self.schema,
            "kind": "synthetic_stability_diagnostics",
            "status": _DIAGNOSTIC_STATUS_CLEAN
            if not self.failure_modes
            else _DIAGNOSTIC_STATUS_FAILURES,
            "scenario_id": self.fixture.scenario_id,
            "scenario_kind": self.fixture.scenario_kind,
            "diagnostic_role": "diagnostic_only_not_gate",
            "classifier": self.classifier.as_dict(),
            "observation_count": len(self.identity_observations),
            "identity_observations": [
                observation.as_dict() for observation in self.identity_observations
            ],
            "slot_transition_labels": [int(value) for value in self.slot_transition_labels],
            "slot_transition_matrix": self.slot_transition_matrix.astype(int).tolist(),
            "identity_confusion_graph": self.identity_confusion_graph,
            "failure_mode_counts": _failure_mode_counts(self.failure_modes),
            "failure_modes": [event.as_dict() for event in self.failure_modes],
            "non_goals": {
                "trains_solver": False,
                "uses_renderer_loss": False,
                "acts_as_gate": False,
            },
        }
        return validate_synthetic_stability_diagnostics_summary(summary)


def diagnose_synthetic_stability_fixture(
    fixture: SyntheticStabilityScenarioFixture,
    *,
    predicted_slots: Sequence[np.ndarray | Sequence[int]] | None = None,
    predicted_assignments: Sequence[np.ndarray] | None = None,
    classifier: FailureModeClassifier | None = None,
) -> SyntheticStabilityDiagnosticsReport:
    fixture = validate_synthetic_stability_scenario_fixture(fixture)
    classifier = classifier or FailureModeClassifier()
    classifier.as_dict()
    row_slots, row_confidences = _prediction_rows(
        fixture,
        predicted_slots=predicted_slots,
        predicted_assignments=predicted_assignments,
    )
    identity_observations = _identity_observations_from_rows(
        fixture,
        row_slots=row_slots,
        row_confidences=row_confidences,
    )
    labels, matrix = _slot_transition_matrix(
        identity_observations,
        slot_count=fixture.world.oracle.slots,
        background_slot=classifier.background_slot,
    )
    graph = _identity_confusion_graph(identity_observations)
    failure_modes = classifier.classify(identity_observations)
    return SyntheticStabilityDiagnosticsReport(
        fixture=fixture,
        classifier=classifier,
        identity_observations=identity_observations,
        slot_transition_matrix=matrix,
        slot_transition_labels=labels,
        identity_confusion_graph=graph,
        failure_modes=failure_modes,
    )


def expected_slots_for_synthetic_fixture(
    fixture: SyntheticStabilityScenarioFixture,
) -> tuple[np.ndarray, ...]:
    fixture = validate_synthetic_stability_scenario_fixture(fixture)
    return tuple(observation.expected_slots.astype(np.int64, copy=True) for observation in fixture.observations)


def validate_synthetic_stability_diagnostics_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("synthetic stability diagnostics summary must be a dict")
    if payload.get("schema") != V2_STABILITY_DIAGNOSTICS_SCHEMA:
        raise ValueError(f"unsupported diagnostics schema: {payload.get('schema')}")
    if payload.get("kind") != "synthetic_stability_diagnostics":
        raise ValueError("diagnostics kind must be synthetic_stability_diagnostics")
    if payload.get("status") not in {_DIAGNOSTIC_STATUS_CLEAN, _DIAGNOSTIC_STATUS_FAILURES}:
        raise ValueError("diagnostics status is unsupported")
    if payload.get("diagnostic_role") != "diagnostic_only_not_gate":
        raise ValueError("diagnostics must be marked diagnostic_only_not_gate")
    for key in (
        "classifier",
        "identity_observations",
        "slot_transition_labels",
        "slot_transition_matrix",
        "identity_confusion_graph",
        "failure_mode_counts",
        "failure_modes",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"diagnostics summary missing {key}")
    labels = payload["slot_transition_labels"]
    matrix = payload["slot_transition_matrix"]
    if not isinstance(labels, list) or not labels:
        raise ValueError("slot_transition_labels must be a non-empty list")
    if len(matrix) != len(labels):
        raise ValueError("slot_transition_matrix rows must match labels")
    for row in matrix:
        if len(row) != len(labels):
            raise ValueError("slot_transition_matrix must be square")
    non_goals = payload["non_goals"]
    if non_goals.get("trains_solver") or non_goals.get("uses_renderer_loss") or non_goals.get("acts_as_gate"):
        raise ValueError("diagnostics summary cannot train, use renderer loss, or act as gate")
    for event in payload["failure_modes"]:
        if event.get("mode") not in V2_STABILITY_FAILURE_MODES:
            raise ValueError(f"unsupported failure mode: {event.get('mode')}")
    return payload


def _prediction_rows(
    fixture: SyntheticStabilityScenarioFixture,
    *,
    predicted_slots: Sequence[np.ndarray | Sequence[int]] | None,
    predicted_assignments: Sequence[np.ndarray] | None,
) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    if predicted_slots is not None and predicted_assignments is not None:
        raise ValueError("provide predicted_slots or predicted_assignments, not both")
    if predicted_slots is None and predicted_assignments is None:
        raise ValueError(
            "synthetic stability diagnostics require explicit predicted_slots "
            "or predicted_assignments"
        )
    if predicted_slots is not None:
        if len(predicted_slots) != len(fixture.observations):
            raise ValueError("predicted_slots must cover every observation frame")
        slots_out = []
        confidences = []
        for frame_index, (observation, slots) in enumerate(zip(fixture.observations, predicted_slots)):
            slot_array = _int_vector(slots, f"predicted_slots[{frame_index}]")
            if slot_array.shape[0] != observation.evidence.evidence_count:
                raise ValueError("predicted slot rows must match observation evidence rows")
            slots_out.append(slot_array)
            confidences.append(np.ones(slot_array.shape[0], dtype=np.float32))
        return tuple(slots_out), tuple(confidences)
    if predicted_assignments is None:
        raise ValueError("predicted_assignments must be provided")
    if len(predicted_assignments) != len(fixture.observations):
        raise ValueError("predicted_assignments must cover every observation frame")
    slots_out = []
    confidences = []
    for frame_index, (observation, assignment) in enumerate(zip(fixture.observations, predicted_assignments)):
        assignment_array = _assignment_array(assignment, f"predicted_assignments[{frame_index}]")
        if assignment_array.shape[0] != observation.evidence.evidence_count:
            raise ValueError("predicted assignment rows must match observation evidence rows")
        if assignment_array.shape[1] != fixture.world.oracle.slots:
            raise ValueError("predicted assignment columns must match fixture slot count")
        slots_out.append(np.argmax(assignment_array, axis=1).astype(np.int64, copy=False))
        confidences.append(np.max(assignment_array, axis=1).astype(np.float32, copy=False))
    return tuple(slots_out), tuple(confidences)


def _identity_observations_from_rows(
    fixture: SyntheticStabilityScenarioFixture,
    *,
    row_slots: Sequence[np.ndarray],
    row_confidences: Sequence[np.ndarray],
) -> tuple[IdentitySlotObservation, ...]:
    observations: list[IdentitySlotObservation] = []
    for frame, slots, confidences in zip(fixture.observations, row_slots, row_confidences):
        object_ids = frame.oracle_object_ids.astype(np.int64, copy=False)
        expected_slots = frame.expected_slots.astype(np.int64, copy=False)
        for object_id in sorted(set(object_ids.tolist())):
            mask = object_ids == int(object_id)
            object_slots = slots[mask]
            object_confidences = confidences[mask]
            predicted_slot = _dominant_slot(object_slots)
            expected_slot_values = sorted(set(expected_slots[mask].astype(int).tolist()))
            if len(expected_slot_values) != 1:
                raise ValueError("expected slots must be stable per oracle object and frame")
            lineage_values = sorted({frame.lineage_ids[index] for index, keep in enumerate(mask) if bool(keep)})
            if len(lineage_values) != 1:
                raise ValueError("lineage ids must be stable per oracle object and frame")
            observations.append(
                IdentitySlotObservation(
                    frame_index=frame.frame_index,
                    oracle_object_id=int(object_id),
                    lineage_id=lineage_values[0],
                    expected_slot=int(expected_slot_values[0]),
                    predicted_slot=int(predicted_slot),
                    evidence_count=int(mask.sum()),
                    mean_confidence=float(np.mean(object_confidences)),
                )
            )
    return tuple(observations)


def _slot_transition_matrix(
    observations: Sequence[IdentitySlotObservation],
    *,
    slot_count: int,
    background_slot: int,
) -> tuple[tuple[int, ...], np.ndarray]:
    canonical_slots = list(range(int(slot_count)))
    predicted_slots = sorted(
        {
            int(obs.predicted_slot)
            for obs in observations
            if int(obs.predicted_slot) != int(background_slot)
        }
    )
    extra_slots = [slot for slot in predicted_slots if slot not in canonical_slots]
    labels = tuple(canonical_slots + extra_slots + [int(background_slot)])
    label_index = {label: index for index, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for object_observations in _observations_by_object(observations).values():
        ordered = sorted(object_observations, key=lambda obs: int(obs.frame_index))
        for previous, current in zip(ordered, ordered[1:]):
            previous_index = label_index.get(int(previous.predicted_slot), label_index[int(background_slot)])
            current_index = label_index.get(int(current.predicted_slot), label_index[int(background_slot)])
            matrix[previous_index, current_index] += 1
    return labels, matrix


def _identity_confusion_graph(
    observations: Sequence[IdentitySlotObservation],
) -> dict[str, Any]:
    nodes = []
    edges = []
    object_ids = sorted({int(obs.oracle_object_id) for obs in observations})
    slots = sorted({int(obs.expected_slot) for obs in observations} | {int(obs.predicted_slot) for obs in observations})
    for object_id in object_ids:
        nodes.append({"id": f"object:{object_id}", "kind": "oracle_identity", "oracle_object_id": object_id})
    for slot in slots:
        nodes.append({"id": f"slot:{slot}", "kind": "predicted_slot", "slot": slot})
    for obs in observations:
        edges.append(
            {
                "source": f"object:{int(obs.oracle_object_id)}",
                "target": f"slot:{int(obs.predicted_slot)}",
                "frame_index": int(obs.frame_index),
                "oracle_object_id": int(obs.oracle_object_id),
                "lineage_id": obs.lineage_id,
                "expected_slot": int(obs.expected_slot),
                "predicted_slot": int(obs.predicted_slot),
                "evidence_count": int(obs.evidence_count),
                "mean_confidence": float(obs.mean_confidence),
                "matches_expected": bool(obs.matches_expected),
            }
        )
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "nodes": nodes,
        "edges": edges,
    }


def _failure_mode_counts(events: Sequence[FailureModeEvent]) -> dict[str, int]:
    counts = {mode: 0 for mode in V2_STABILITY_FAILURE_MODES}
    for event in events:
        counts[event.mode] += 1
    return counts


def _observations_by_frame(
    observations: Sequence[IdentitySlotObservation],
) -> dict[int, tuple[IdentitySlotObservation, ...]]:
    grouped: dict[int, list[IdentitySlotObservation]] = {}
    for obs in observations:
        grouped.setdefault(int(obs.frame_index), []).append(obs)
    return {key: tuple(sorted(value, key=lambda obs: int(obs.oracle_object_id))) for key, value in sorted(grouped.items())}


def _observations_by_object(
    observations: Sequence[IdentitySlotObservation],
) -> dict[int, tuple[IdentitySlotObservation, ...]]:
    grouped: dict[int, list[IdentitySlotObservation]] = {}
    for obs in observations:
        grouped.setdefault(int(obs.oracle_object_id), []).append(obs)
    return {key: tuple(sorted(value, key=lambda obs: int(obs.frame_index))) for key, value in sorted(grouped.items())}


def _validate_identity_slot_observations(
    observations: Sequence[IdentitySlotObservation],
) -> tuple[IdentitySlotObservation, ...]:
    checked = []
    for obs in observations:
        if not isinstance(obs, IdentitySlotObservation):
            raise TypeError("observations must contain IdentitySlotObservation")
        if int(obs.evidence_count) < 1:
            raise ValueError("identity observation evidence_count must be >= 1")
        checked.append(obs)
    return tuple(checked)


def _dominant_slot(values: np.ndarray) -> int:
    slots, counts = np.unique(values.astype(np.int64, copy=False), return_counts=True)
    max_count = int(np.max(counts))
    candidates = slots[counts == max_count]
    return int(np.min(candidates))


def _int_vector(value: np.ndarray | Sequence[int], label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a 1D array")
    if array.shape[0] == 0:
        raise ValueError(f"{label} must contain at least one row")
    return array.astype(np.int64, copy=False)


def _assignment_array(value: np.ndarray, label: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError(f"{label} must be a 2D array")
    if array.shape[0] == 0 or array.shape[1] == 0:
        raise ValueError(f"{label} must be non-empty")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain finite values")
    if np.any(array < 0.0):
        raise ValueError(f"{label} must be non-negative")
    row_sums = np.sum(array, axis=1)
    if np.any(row_sums <= 0.0):
        raise ValueError(f"{label} rows must have positive mass")
    return (array / row_sums[:, None]).astype(np.float32, copy=False)
