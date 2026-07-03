from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.features import extract_features, positions
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_field import ObjectField

_EPS = 1e-8
OBJECT_STATE_DELIVERY_BINDING_SCHEMA = "objgauss-object-state-delivery-binding-v1"
DYNAMIC_K_UPDATE_PLAN_SCHEMA = "objgauss-dynamic-k-update-plan-v1"


@dataclass(frozen=True)
class ObjectState:
    """Object-level projection produced from a fixed-K assignment matrix."""

    id: int
    slot_mass: float
    confidence: float
    mass_fraction: float
    assignment_entropy: float
    normalized_assignment_entropy: float
    centroid: np.ndarray
    bbox: np.ndarray
    feature: np.ndarray
    status: str
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class ObjectStateProjection:
    states: tuple[ObjectState, ...]
    assignment: np.ndarray
    derived_object_ids: np.ndarray

    @property
    def slots(self) -> int:
        return int(self.assignment.shape[1])

    @property
    def evidence_count(self) -> int:
        return int(self.assignment.shape[0])


@dataclass(frozen=True)
class ObjectStabilityReport:
    evidence_count: int
    slots: int
    assignment_confidence: float
    mean_normalized_entropy: float
    effective_slots: float
    slot_mass: tuple[float, ...]
    slot_mass_fraction: tuple[float, ...]
    inactive_slots: tuple[int, ...]
    low_confidence_slots: tuple[int, ...]
    mixed_slots: tuple[int, ...]
    slot_collapse: bool
    dominant_slot: int | None
    dominant_slot_mass_fraction: float
    object_purity: float | None
    per_slot_purity: tuple[float | None, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class ObjectStateMatch:
    previous_id: int
    current_id: int
    cost: float
    temporal_drift: float
    bbox_iou: float
    feature_distance: float
    mass_delta: float
    previous_status: str
    current_status: str


@dataclass(frozen=True)
class ObjectTemporalMatchReport:
    previous_count: int
    current_count: int
    matches: tuple[ObjectStateMatch, ...]
    unmatched_previous: tuple[int, ...]
    unmatched_current: tuple[int, ...]
    ignored_previous: tuple[int, ...]
    ignored_current: tuple[int, ...]
    mean_temporal_drift: float
    max_temporal_drift: float
    cost_matrix: np.ndarray
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class DynamicKProposal:
    kind: str
    source_ids: tuple[int, ...]
    target_id: int | None
    score: float
    threshold: float
    reason: str
    evidence: dict[str, Any]
    action: str = "proposal_only"


@dataclass(frozen=True)
class DynamicKProposalReport:
    slot_count: int
    proposal_count: int
    proposals: tuple[DynamicKProposal, ...]
    diagnostics: tuple[str, ...]


@dataclass(frozen=True)
class DynamicKUpdateAction:
    kind: str
    source_ids: tuple[int, ...]
    target_id: int | None
    accepted: bool
    slot_delta: int
    apply_at: str
    reason: str
    evidence: dict[str, Any]


@dataclass(frozen=True)
class DynamicKUpdatePlan:
    schema: str
    current_slot_count: int
    next_slot_count: int
    actions: tuple[DynamicKUpdateAction, ...]
    accepted_count: int
    blocked_count: int
    apply_at: str
    diagnostics: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "current_slot_count": int(self.current_slot_count),
            "next_slot_count": int(self.next_slot_count),
            "apply_at": self.apply_at,
            "accepted_count": int(self.accepted_count),
            "blocked_count": int(self.blocked_count),
            "diagnostics": list(self.diagnostics),
            "actions": [
                {
                    "kind": action.kind,
                    "source_ids": list(action.source_ids),
                    "target_id": action.target_id,
                    "accepted": bool(action.accepted),
                    "slot_delta": int(action.slot_delta),
                    "apply_at": action.apply_at,
                    "reason": action.reason,
                    "evidence": action.evidence,
                }
                for action in self.actions
            ],
        }


def validate_assignment_matrix(
    assignment: np.ndarray,
    *,
    evidence_count: int | None = None,
    row_sum_tolerance: float = 1e-5,
) -> np.ndarray:
    matrix = np.asarray(assignment, dtype=np.float32)
    if matrix.ndim != 2:
        raise ValueError("assignment must be a 2D N x K matrix")
    if matrix.shape[1] == 0:
        raise ValueError("assignment matrix needs at least one object slot")
    if evidence_count is not None and matrix.shape[0] != int(evidence_count):
        raise ValueError(
            f"assignment has {matrix.shape[0]} rows for {int(evidence_count)} evidence records"
        )
    if not np.isfinite(matrix).all():
        raise ValueError("assignment matrix must contain only finite values")
    if np.any(matrix < 0.0):
        raise ValueError("assignment matrix must be non-negative")
    if matrix.shape[0] > 0:
        row_sums = matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0, atol=row_sum_tolerance, rtol=0.0):
            raise ValueError("assignment rows must sum to 1")
    return matrix


def project_object_states_from_field(
    cloud: GaussianCloud,
    field: ObjectField,
    *,
    evidence_features: np.ndarray | None = None,
    support_threshold: float = 1e-6,
    low_confidence_mass: float = 0.5,
    mixed_entropy_threshold: float = 0.8,
) -> ObjectStateProjection:
    if cloud.count != field.gaussian_count:
        raise ValueError(f"field has {field.gaussian_count} gaussians for cloud with {cloud.count}")
    return project_object_states(
        cloud,
        field.probabilities(),
        evidence_features=evidence_features,
        support_threshold=support_threshold,
        low_confidence_mass=low_confidence_mass,
        mixed_entropy_threshold=mixed_entropy_threshold,
    )


def project_object_states(
    cloud: GaussianCloud,
    assignment: np.ndarray,
    *,
    evidence_features: np.ndarray | None = None,
    support_threshold: float = 1e-6,
    low_confidence_mass: float = 0.5,
    mixed_entropy_threshold: float = 0.8,
) -> ObjectStateProjection:
    matrix = validate_assignment_matrix(assignment, evidence_count=cloud.count)
    xyz = positions(cloud)
    features = _evidence_features(cloud, evidence_features)
    if not 0.0 <= support_threshold <= 1.0:
        raise ValueError("support_threshold must be in [0, 1]")
    if low_confidence_mass < 0.0:
        raise ValueError("low_confidence_mass must be >= 0")
    if not 0.0 <= mixed_entropy_threshold <= 1.0:
        raise ValueError("mixed_entropy_threshold must be in [0, 1]")

    row_entropy = _row_entropy(matrix)
    normalized_row_entropy = _normalized_row_entropy(row_entropy, matrix.shape[1])
    derived_ids = np.argmax(matrix, axis=1).astype(np.int32, copy=False) if cloud.count else np.empty(0, np.int32)
    states = tuple(
        _project_slot(
            slot,
            matrix[:, slot],
            xyz,
            features,
            row_entropy,
            normalized_row_entropy,
            evidence_count=cloud.count,
            support_threshold=support_threshold,
            low_confidence_mass=low_confidence_mass,
            mixed_entropy_threshold=mixed_entropy_threshold,
        )
        for slot in range(matrix.shape[1])
    )
    return ObjectStateProjection(states=states, assignment=matrix, derived_object_ids=derived_ids)


def object_state_stability_report(
    projection: ObjectStateProjection,
    *,
    purity_labels: np.ndarray | None = None,
    inactive_mass_threshold: float = 1e-6,
    low_confidence_mass: float = 0.5,
    mixed_entropy_threshold: float = 0.8,
    collapse_mass_fraction: float = 0.9,
    assignment_confidence_floor: float = 0.5,
    purity_floor: float = 0.8,
) -> ObjectStabilityReport:
    assignment = validate_assignment_matrix(projection.assignment)
    if len(projection.states) != assignment.shape[1]:
        raise ValueError("projection states must match assignment slot count")
    if not 0.0 <= inactive_mass_threshold:
        raise ValueError("inactive_mass_threshold must be >= 0")
    if low_confidence_mass < 0.0:
        raise ValueError("low_confidence_mass must be >= 0")
    if not 0.0 <= mixed_entropy_threshold <= 1.0:
        raise ValueError("mixed_entropy_threshold must be in [0, 1]")
    if not 0.0 <= collapse_mass_fraction <= 1.0:
        raise ValueError("collapse_mass_fraction must be in [0, 1]")
    if not 0.0 <= assignment_confidence_floor <= 1.0:
        raise ValueError("assignment_confidence_floor must be in [0, 1]")
    if not 0.0 <= purity_floor <= 1.0:
        raise ValueError("purity_floor must be in [0, 1]")

    entropy = _normalized_row_entropy(_row_entropy(assignment), assignment.shape[1])
    mean_entropy = float(np.mean(entropy)) if entropy.size else 0.0
    assignment_confidence = 0.0 if assignment.shape[0] == 0 else float(1.0 - mean_entropy)
    slot_mass_values = assignment.sum(axis=0).astype(np.float32, copy=False)
    total_mass = float(slot_mass_values.sum())
    slot_mass_fraction = (
        np.zeros_like(slot_mass_values, dtype=np.float32)
        if total_mass <= _EPS
        else (slot_mass_values / total_mass).astype(np.float32, copy=False)
    )
    dominant_slot = int(np.argmax(slot_mass_values)) if slot_mass_values.size and total_mass > _EPS else None
    dominant_fraction = (
        0.0 if dominant_slot is None else float(slot_mass_fraction[dominant_slot])
    )
    inactive_slots = tuple(
        int(index)
        for index, mass in enumerate(slot_mass_values)
        if float(mass) <= inactive_mass_threshold
    )
    low_confidence_slots = tuple(
        int(index)
        for index, mass in enumerate(slot_mass_values)
        if inactive_mass_threshold < float(mass) < low_confidence_mass
    )
    mixed_slots = tuple(
        int(state.id)
        for state in projection.states
        if state.normalized_assignment_entropy >= mixed_entropy_threshold and state.slot_mass > inactive_mass_threshold
    )
    per_slot_purity, object_purity = _purity_metrics(assignment, purity_labels)
    collapse = bool(total_mass > _EPS and dominant_fraction >= collapse_mass_fraction)
    diagnostics = _stability_diagnostics(
        evidence_count=assignment.shape[0],
        assignment_confidence=assignment_confidence,
        assignment_confidence_floor=assignment_confidence_floor,
        inactive_slots=inactive_slots,
        low_confidence_slots=low_confidence_slots,
        mixed_slots=mixed_slots,
        slot_collapse=collapse,
        dominant_slot=dominant_slot,
        object_purity=object_purity,
        purity_floor=purity_floor,
    )
    return ObjectStabilityReport(
        evidence_count=int(assignment.shape[0]),
        slots=int(assignment.shape[1]),
        assignment_confidence=assignment_confidence,
        mean_normalized_entropy=mean_entropy,
        effective_slots=_effective_slots(slot_mass_fraction, total_mass),
        slot_mass=tuple(float(value) for value in slot_mass_values),
        slot_mass_fraction=tuple(float(value) for value in slot_mass_fraction),
        inactive_slots=inactive_slots,
        low_confidence_slots=low_confidence_slots,
        mixed_slots=mixed_slots,
        slot_collapse=collapse,
        dominant_slot=dominant_slot,
        dominant_slot_mass_fraction=dominant_fraction,
        object_purity=object_purity,
        per_slot_purity=per_slot_purity,
        diagnostics=diagnostics,
    )


def match_object_states(
    previous: ObjectStateProjection | Sequence[ObjectState],
    current: ObjectStateProjection | Sequence[ObjectState],
    *,
    centroid_weight: float = 1.0,
    bbox_weight: float = 0.5,
    feature_weight: float = 1.0,
    mass_weight: float = 0.1,
    max_cost: float | None = None,
    include_inactive: bool = False,
    drift_warning_threshold: float | None = None,
) -> ObjectTemporalMatchReport:
    previous_states = _candidate_states(_states_tuple(previous), include_inactive=include_inactive)
    current_states = _candidate_states(_states_tuple(current), include_inactive=include_inactive)
    ignored_previous = tuple(
        int(state.id)
        for state in _states_tuple(previous)
        if not include_inactive and state.status == "inactive"
    )
    ignored_current = tuple(
        int(state.id)
        for state in _states_tuple(current)
        if not include_inactive and state.status == "inactive"
    )
    _validate_non_negative_weight("centroid_weight", centroid_weight)
    _validate_non_negative_weight("bbox_weight", bbox_weight)
    _validate_non_negative_weight("feature_weight", feature_weight)
    _validate_non_negative_weight("mass_weight", mass_weight)
    if max_cost is not None and max_cost < 0.0:
        raise ValueError("max_cost must be >= 0")
    if drift_warning_threshold is not None and drift_warning_threshold < 0.0:
        raise ValueError("drift_warning_threshold must be >= 0")

    limit = float("inf") if max_cost is None else float(max_cost)
    costs = _state_cost_matrix(
        previous_states,
        current_states,
        centroid_weight=centroid_weight,
        bbox_weight=bbox_weight,
        feature_weight=feature_weight,
        mass_weight=mass_weight,
    )
    matches = _greedy_matches(
        previous_states,
        current_states,
        costs,
        max_cost=limit,
        centroid_weight=centroid_weight,
        bbox_weight=bbox_weight,
        feature_weight=feature_weight,
        mass_weight=mass_weight,
    )
    matched_previous = {match.previous_id for match in matches}
    matched_current = {match.current_id for match in matches}
    unmatched_previous = tuple(
        int(state.id) for state in previous_states if int(state.id) not in matched_previous
    )
    unmatched_current = tuple(
        int(state.id) for state in current_states if int(state.id) not in matched_current
    )
    drifts = [match.temporal_drift for match in matches]
    mean_drift = float(np.mean(drifts)) if drifts else 0.0
    max_drift = float(np.max(drifts)) if drifts else 0.0
    diagnostics = _temporal_match_diagnostics(
        previous_count=len(previous_states),
        current_count=len(current_states),
        matches=matches,
        unmatched_previous=unmatched_previous,
        unmatched_current=unmatched_current,
        ignored_previous=ignored_previous,
        ignored_current=ignored_current,
        max_temporal_drift=max_drift,
        drift_warning_threshold=drift_warning_threshold,
    )
    return ObjectTemporalMatchReport(
        previous_count=len(previous_states),
        current_count=len(current_states),
        matches=matches,
        unmatched_previous=unmatched_previous,
        unmatched_current=unmatched_current,
        ignored_previous=ignored_previous,
        ignored_current=ignored_current,
        mean_temporal_drift=mean_drift,
        max_temporal_drift=max_drift,
        cost_matrix=costs,
        diagnostics=diagnostics,
    )


def object_state_delivery_summary(
    projection: ObjectStateProjection,
    *,
    stability_report: ObjectStabilityReport | None = None,
    chunk_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    assignment = validate_assignment_matrix(projection.assignment)
    if len(projection.states) != assignment.shape[1]:
        raise ValueError("projection states must match assignment slot count")
    if projection.derived_object_ids.shape[0] != assignment.shape[0]:
        raise ValueError("derived_object_ids must match assignment evidence count")
    stability = stability_report or object_state_stability_report(projection)
    child_counts = _derived_child_counts(projection.derived_object_ids, assignment.shape[1])
    active_ids = tuple(
        int(state.id)
        for state in projection.states
        if state.status != "inactive" and state.slot_mass > _EPS
    )
    object_ids_with_children = tuple(
        int(index) for index, count in enumerate(child_counts) if int(count) > 0
    )
    summary = {
        "schema": OBJECT_STATE_DELIVERY_BINDING_SCHEMA,
        "slot_count": int(assignment.shape[1]),
        "evidence_count": int(assignment.shape[0]),
        "active_object_ids": list(active_ids),
        "derived_object_id_source": "argmax_assignment",
        "derived_object_id_field": "object_id",
        "object_id_coverage": {
            "field": "object_id",
            "mode": "derived_from_assignment_argmax",
            "has_object_ids": bool(assignment.shape[0] > 0),
            "object_count": len(object_ids_with_children),
        },
        "stability": {
            "assignment_confidence": stability.assignment_confidence,
            "effective_slots": stability.effective_slots,
            "slot_collapse": stability.slot_collapse,
            "inactive_slots": list(stability.inactive_slots),
            "mixed_slots": list(stability.mixed_slots),
            "diagnostics": list(stability.diagnostics),
        },
        "gaussian_children": [
            {
                "object_id": int(state.id),
                "gaussian_count": int(child_counts[state.id]) if state.id < len(child_counts) else 0,
                "assignment_mass": float(state.slot_mass),
                "mass_fraction": float(state.mass_fraction),
                "status": state.status,
            }
            for state in projection.states
        ],
        "states": [_state_delivery_record(state) for state in projection.states],
    }
    if chunk_index is not None:
        summary["chunk_binding"] = _chunk_binding_summary(chunk_index, object_ids_with_children)
    return summary


def bind_object_states_to_artifact(
    artifact: dict[str, Any],
    projection: ObjectStateProjection,
    *,
    stability_report: ObjectStabilityReport | None = None,
    chunk_index: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(artifact, dict):
        raise TypeError("artifact must be a dict")
    summary = object_state_delivery_summary(
        projection,
        stability_report=stability_report,
        chunk_index=chunk_index,
    )
    bound = dict(artifact)
    bound["object_state_summary"] = summary
    bound.setdefault("object_count", summary["object_id_coverage"]["object_count"])
    if bound.get("role") in {"object_edit", "compressed_chunked"}:
        bound.setdefault("object_id_coverage", summary["object_id_coverage"])
    return bound


def dynamic_k_proposal_report(
    projection: ObjectStateProjection,
    *,
    stability_report: ObjectStabilityReport | None = None,
    temporal_match: ObjectTemporalMatchReport | None = None,
    inactive_mass_threshold: float = 1e-6,
    mixed_entropy_threshold: float = 0.8,
    duplicate_feature_distance_threshold: float = 0.05,
    duplicate_centroid_distance_threshold: float = 0.1,
) -> DynamicKProposalReport:
    if inactive_mass_threshold < 0.0:
        raise ValueError("inactive_mass_threshold must be >= 0")
    if not 0.0 <= mixed_entropy_threshold <= 1.0:
        raise ValueError("mixed_entropy_threshold must be in [0, 1]")
    if duplicate_feature_distance_threshold < 0.0:
        raise ValueError("duplicate_feature_distance_threshold must be >= 0")
    if duplicate_centroid_distance_threshold < 0.0:
        raise ValueError("duplicate_centroid_distance_threshold must be >= 0")
    stability = stability_report or object_state_stability_report(
        projection,
        inactive_mass_threshold=inactive_mass_threshold,
        mixed_entropy_threshold=mixed_entropy_threshold,
    )
    proposals: list[DynamicKProposal] = []
    for slot in stability.inactive_slots:
        proposals.append(
            DynamicKProposal(
                kind="remove_inactive",
                source_ids=(int(slot),),
                target_id=None,
                score=float(stability.slot_mass[slot]),
                threshold=float(inactive_mass_threshold),
                reason="slot mass is below inactive threshold",
                evidence={"slot_mass": float(stability.slot_mass[slot])},
            )
        )
    for slot in stability.mixed_slots:
        state = projection.states[int(slot)]
        proposals.append(
            DynamicKProposal(
                kind="split_mixed",
                source_ids=(int(slot),),
                target_id=None,
                score=float(state.normalized_assignment_entropy),
                threshold=float(mixed_entropy_threshold),
                reason="slot assignment entropy is high",
                evidence={
                    "normalized_assignment_entropy": float(state.normalized_assignment_entropy),
                    "slot_mass": float(state.slot_mass),
                },
            )
        )
    proposals.extend(
        _duplicate_merge_proposals(
            projection.states,
            feature_threshold=duplicate_feature_distance_threshold,
            centroid_threshold=duplicate_centroid_distance_threshold,
        )
    )
    if temporal_match is not None:
        current_by_id = {int(state.id): state for state in projection.states}
        for slot in temporal_match.unmatched_current:
            state = current_by_id.get(int(slot))
            if state is None or state.status == "inactive":
                continue
            proposals.append(
                DynamicKProposal(
                    kind="birth_unmatched",
                    source_ids=(int(slot),),
                    target_id=None,
                    score=float(state.confidence),
                    threshold=0.0,
                    reason="current object state is unmatched by temporal matching",
                    evidence={
                        "slot_mass": float(state.slot_mass),
                        "status": state.status,
                    },
                )
            )
    diagnostics = _dynamic_k_diagnostics(proposals)
    return DynamicKProposalReport(
        slot_count=int(projection.slots),
        proposal_count=len(proposals),
        proposals=tuple(proposals),
        diagnostics=diagnostics,
    )


def dynamic_k_update_plan(
    projection: ObjectStateProjection,
    *,
    proposal_report: DynamicKProposalReport | None = None,
    min_slot_count: int = 1,
    max_slot_count: int | None = None,
    apply_at: str = "epoch_boundary",
) -> DynamicKUpdatePlan:
    if min_slot_count < 1:
        raise ValueError("min_slot_count must be >= 1")
    if max_slot_count is not None and max_slot_count < min_slot_count:
        raise ValueError("max_slot_count must be >= min_slot_count")
    if apply_at != "epoch_boundary":
        raise ValueError("dynamic-K updates may only apply at epoch_boundary")
    report = proposal_report or dynamic_k_proposal_report(projection)
    if report.slot_count != projection.slots:
        raise ValueError("proposal_report slot_count must match projection slots")

    current_slots = int(projection.slots)
    next_slots = current_slots
    consumed_sources: set[int] = set()
    actions: list[DynamicKUpdateAction] = []
    for proposal in report.proposals:
        action, next_slots = _dynamic_k_action_from_proposal(
            proposal,
            next_slot_count=next_slots,
            consumed_sources=consumed_sources,
            min_slot_count=min_slot_count,
            max_slot_count=max_slot_count,
            apply_at=apply_at,
        )
        actions.append(action)
        if action.accepted:
            consumed_sources.update(int(source) for source in proposal.source_ids)
            if action.target_id is not None:
                consumed_sources.discard(int(action.target_id))

    accepted = sum(1 for action in actions if action.accepted)
    blocked = len(actions) - accepted
    diagnostics = _dynamic_k_update_diagnostics(
        current_slot_count=current_slots,
        next_slot_count=next_slots,
        accepted_count=accepted,
        blocked_count=blocked,
        proposal_count=report.proposal_count,
    )
    return DynamicKUpdatePlan(
        schema=DYNAMIC_K_UPDATE_PLAN_SCHEMA,
        current_slot_count=current_slots,
        next_slot_count=int(next_slots),
        actions=tuple(actions),
        accepted_count=int(accepted),
        blocked_count=int(blocked),
        apply_at=apply_at,
        diagnostics=diagnostics,
    )


def _states_tuple(states: ObjectStateProjection | Sequence[ObjectState]) -> tuple[ObjectState, ...]:
    if isinstance(states, ObjectStateProjection):
        return tuple(states.states)
    return tuple(states)


def _candidate_states(
    states: tuple[ObjectState, ...],
    *,
    include_inactive: bool,
) -> tuple[ObjectState, ...]:
    if include_inactive:
        return states
    return tuple(state for state in states if state.status != "inactive")


def _validate_non_negative_weight(name: str, value: float) -> None:
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0")


def _state_cost_matrix(
    previous: tuple[ObjectState, ...],
    current: tuple[ObjectState, ...],
    *,
    centroid_weight: float,
    bbox_weight: float,
    feature_weight: float,
    mass_weight: float,
) -> np.ndarray:
    costs = np.full((len(previous), len(current)), np.inf, dtype=np.float32)
    for prev_index, prev in enumerate(previous):
        for curr_index, curr in enumerate(current):
            cost, _, _, _, _ = _state_pair_components(
                prev,
                curr,
                centroid_weight=centroid_weight,
                bbox_weight=bbox_weight,
                feature_weight=feature_weight,
                mass_weight=mass_weight,
            )
            costs[prev_index, curr_index] = cost
    return costs


def _greedy_matches(
    previous: tuple[ObjectState, ...],
    current: tuple[ObjectState, ...],
    costs: np.ndarray,
    *,
    max_cost: float,
    centroid_weight: float,
    bbox_weight: float,
    feature_weight: float,
    mass_weight: float,
) -> tuple[ObjectStateMatch, ...]:
    candidates: list[tuple[float, int, int]] = []
    for prev_index, prev in enumerate(previous):
        for curr_index, curr in enumerate(current):
            cost = float(costs[prev_index, curr_index])
            if np.isfinite(cost) and cost <= max_cost:
                candidates.append((cost, int(prev.id), int(curr.id)))
    candidates.sort()

    previous_by_id = {int(state.id): state for state in previous}
    current_by_id = {int(state.id): state for state in current}
    used_previous: set[int] = set()
    used_current: set[int] = set()
    matches: list[ObjectStateMatch] = []
    for _, prev_id, curr_id in candidates:
        if prev_id in used_previous or curr_id in used_current:
            continue
        prev = previous_by_id[prev_id]
        curr = current_by_id[curr_id]
        cost, drift, bbox_iou, feature_distance, mass_delta = _state_pair_components(
            prev,
            curr,
            centroid_weight=centroid_weight,
            bbox_weight=bbox_weight,
            feature_weight=feature_weight,
            mass_weight=mass_weight,
        )
        matches.append(
            ObjectStateMatch(
                previous_id=prev_id,
                current_id=curr_id,
                cost=cost,
                temporal_drift=drift,
                bbox_iou=bbox_iou,
                feature_distance=feature_distance,
                mass_delta=mass_delta,
                previous_status=prev.status,
                current_status=curr.status,
            )
        )
        used_previous.add(prev_id)
        used_current.add(curr_id)
    return tuple(matches)


def _state_pair_components(
    previous: ObjectState,
    current: ObjectState,
    *,
    centroid_weight: float = 1.0,
    bbox_weight: float = 0.5,
    feature_weight: float = 1.0,
    mass_weight: float = 0.1,
) -> tuple[float, float, float, float, float]:
    drift = _centroid_distance(previous.centroid, current.centroid)
    bbox_iou = _bbox_iou(previous.bbox, current.bbox)
    feature_distance = _feature_distance(previous.feature, current.feature)
    mass_delta = _mass_delta(previous.slot_mass, current.slot_mass)
    if not np.isfinite(drift):
        return float("inf"), drift, bbox_iou, feature_distance, mass_delta
    cost = (
        centroid_weight * drift
        + bbox_weight * (1.0 - bbox_iou)
        + feature_weight * feature_distance
        + mass_weight * mass_delta
    )
    return float(cost), drift, bbox_iou, feature_distance, mass_delta


def _centroid_distance(previous: np.ndarray, current: np.ndarray) -> float:
    prev = np.asarray(previous, dtype=np.float32)
    curr = np.asarray(current, dtype=np.float32)
    if prev.shape != curr.shape or prev.ndim != 1 or not np.isfinite(prev).all() or not np.isfinite(curr).all():
        return float("inf")
    return float(np.linalg.norm(prev - curr))


def _bbox_iou(previous: np.ndarray, current: np.ndarray) -> float:
    prev = np.asarray(previous, dtype=np.float32)
    curr = np.asarray(current, dtype=np.float32)
    if prev.shape != (6,) or curr.shape != (6,):
        return 0.0
    if not np.isfinite(prev).all() or not np.isfinite(curr).all():
        return 0.0
    prev_min, prev_max = prev[:3], prev[3:]
    curr_min, curr_max = curr[:3], curr[3:]
    prev_extent = np.maximum(prev_max - prev_min, 0.0)
    curr_extent = np.maximum(curr_max - curr_min, 0.0)
    prev_volume = float(np.prod(prev_extent))
    curr_volume = float(np.prod(curr_extent))
    if prev_volume <= _EPS or curr_volume <= _EPS:
        return 1.0 if np.allclose(prev_min, curr_min) and np.allclose(prev_max, curr_max) else 0.0
    inter_min = np.maximum(prev_min, curr_min)
    inter_max = np.minimum(prev_max, curr_max)
    inter_extent = np.maximum(inter_max - inter_min, 0.0)
    inter_volume = float(np.prod(inter_extent))
    union = prev_volume + curr_volume - inter_volume
    if union <= _EPS:
        return 0.0
    return float(inter_volume / union)


def _feature_distance(previous: np.ndarray, current: np.ndarray) -> float:
    prev = np.asarray(previous, dtype=np.float32)
    curr = np.asarray(current, dtype=np.float32)
    if prev.shape != curr.shape or prev.ndim != 1 or not np.isfinite(prev).all() or not np.isfinite(curr).all():
        return 1.0
    prev_norm = float(np.linalg.norm(prev))
    curr_norm = float(np.linalg.norm(curr))
    if prev_norm <= _EPS or curr_norm <= _EPS:
        return 1.0
    cosine = float(np.dot(prev, curr) / (prev_norm * curr_norm))
    return float(1.0 - np.clip(cosine, -1.0, 1.0))


def _mass_delta(previous: float, current: float) -> float:
    scale = max(abs(float(previous)), abs(float(current)), _EPS)
    return float(abs(float(previous) - float(current)) / scale)


def _temporal_match_diagnostics(
    *,
    previous_count: int,
    current_count: int,
    matches: tuple[ObjectStateMatch, ...],
    unmatched_previous: tuple[int, ...],
    unmatched_current: tuple[int, ...],
    ignored_previous: tuple[int, ...],
    ignored_current: tuple[int, ...],
    max_temporal_drift: float,
    drift_warning_threshold: float | None,
) -> tuple[str, ...]:
    flags: list[str] = []
    if previous_count == 0:
        flags.append("no_previous_states")
    if current_count == 0:
        flags.append("no_current_states")
    if not matches:
        flags.append("no_matches")
    if unmatched_previous:
        flags.append("unmatched_previous")
    if unmatched_current:
        flags.append("unmatched_current")
    if ignored_previous or ignored_current:
        flags.append("ignored_inactive_states")
    if drift_warning_threshold is not None and max_temporal_drift > drift_warning_threshold:
        flags.append("high_temporal_drift")
    return tuple(flags)


def _derived_child_counts(object_ids: np.ndarray, slots: int) -> np.ndarray:
    ids = np.asarray(object_ids, dtype=np.int64)
    if ids.ndim != 1:
        raise ValueError("derived_object_ids must be a 1D array")
    if ids.size and (np.any(ids < 0) or np.any(ids >= slots)):
        raise ValueError("derived_object_ids must be within the slot range")
    return np.bincount(ids, minlength=slots).astype(np.int64, copy=False)


def _state_delivery_record(state: ObjectState) -> dict[str, Any]:
    return {
        "object_id": int(state.id),
        "status": state.status,
        "slot_mass": float(state.slot_mass),
        "confidence": float(state.confidence),
        "mass_fraction": float(state.mass_fraction),
        "assignment_entropy": float(state.assignment_entropy),
        "normalized_assignment_entropy": float(state.normalized_assignment_entropy),
        "centroid": _finite_or_none_list(state.centroid),
        "bbox": _finite_or_none_list(state.bbox),
        "feature_dim": int(state.feature.shape[0]) if state.feature.ndim == 1 else 0,
        "diagnostics": list(state.diagnostics),
    }


def _finite_or_none_list(values: np.ndarray) -> list[float] | None:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim != 1 or not np.isfinite(array).all():
        return None
    return [float(value) for value in array]


def _chunk_binding_summary(
    chunk_index: dict[str, Any],
    object_ids: tuple[int, ...],
) -> dict[str, Any]:
    chunk_ids_by_object: dict[str, list[int]] = {}
    index_object_ids: set[int] = set()
    objects = chunk_index.get("objects") if isinstance(chunk_index, dict) else None
    if isinstance(objects, list):
        for entry in objects:
            if not isinstance(entry, dict) or "object_id" not in entry:
                continue
            object_id = int(entry["object_id"])
            index_object_ids.add(object_id)
            chunk_ids = entry.get("chunk_ids")
            if isinstance(chunk_ids, list):
                chunk_ids_by_object[str(object_id)] = [int(chunk_id) for chunk_id in chunk_ids]
    chunks = chunk_index.get("chunks") if isinstance(chunk_index, dict) else None
    if isinstance(chunks, list):
        for chunk in chunks:
            if not isinstance(chunk, dict) or "object_id" not in chunk or "chunk_id" not in chunk:
                continue
            object_id = int(chunk["object_id"])
            index_object_ids.add(object_id)
            chunk_ids_by_object.setdefault(str(object_id), []).append(int(chunk["chunk_id"]))
    expected_ids = set(object_ids)
    missing = sorted(expected_ids - index_object_ids)
    extra = sorted(index_object_ids - expected_ids)
    return {
        "chunk_index_schema": chunk_index.get("schema") if isinstance(chunk_index, dict) else None,
        "compatible": not missing,
        "object_ids": list(object_ids),
        "missing_object_ids": missing,
        "extra_chunk_object_ids": extra,
        "chunk_ids_by_object": {
            key: sorted(set(value)) for key, value in sorted(chunk_ids_by_object.items())
        },
    }


def _duplicate_merge_proposals(
    states: tuple[ObjectState, ...],
    *,
    feature_threshold: float,
    centroid_threshold: float,
) -> tuple[DynamicKProposal, ...]:
    active = tuple(state for state in states if state.status != "inactive" and state.slot_mass > _EPS)
    proposals: list[DynamicKProposal] = []
    for left_index, left in enumerate(active):
        for right in active[left_index + 1 :]:
            feature_distance = _feature_distance(left.feature, right.feature)
            centroid_distance = _centroid_distance(left.centroid, right.centroid)
            if feature_distance <= feature_threshold and centroid_distance <= centroid_threshold:
                score = max(feature_distance / max(feature_threshold, _EPS), centroid_distance / max(centroid_threshold, _EPS))
                target = int(left.id if left.slot_mass >= right.slot_mass else right.id)
                proposals.append(
                    DynamicKProposal(
                        kind="merge_duplicate",
                        source_ids=(int(left.id), int(right.id)),
                        target_id=target,
                        score=float(score),
                        threshold=1.0,
                        reason="object states are near-duplicate by feature and centroid",
                        evidence={
                            "feature_distance": float(feature_distance),
                            "centroid_distance": float(centroid_distance),
                            "feature_threshold": float(feature_threshold),
                            "centroid_threshold": float(centroid_threshold),
                        },
                    )
                )
    return tuple(proposals)


def _dynamic_k_action_from_proposal(
    proposal: DynamicKProposal,
    *,
    next_slot_count: int,
    consumed_sources: set[int],
    min_slot_count: int,
    max_slot_count: int | None,
    apply_at: str,
) -> tuple[DynamicKUpdateAction, int]:
    sources = tuple(int(source) for source in proposal.source_ids)
    if any(source in consumed_sources for source in sources):
        return (
            _blocked_dynamic_k_action(
                proposal,
                apply_at=apply_at,
                reason="proposal conflicts with an already accepted action",
            ),
            next_slot_count,
        )
    if proposal.kind == "remove_inactive":
        if next_slot_count - 1 < min_slot_count:
            return (
                _blocked_dynamic_k_action(
                    proposal,
                    apply_at=apply_at,
                    reason="min_slot_count would be violated",
                ),
                next_slot_count,
            )
        return (
            _accepted_dynamic_k_action(
                proposal,
                slot_delta=-1,
                apply_at=apply_at,
                reason="drop inactive slot at epoch boundary",
            ),
            next_slot_count - 1,
        )
    if proposal.kind == "merge_duplicate":
        if proposal.target_id is None or int(proposal.target_id) not in sources:
            return (
                _blocked_dynamic_k_action(
                    proposal,
                    apply_at=apply_at,
                    reason="merge_duplicate requires target_id within source_ids",
                ),
                next_slot_count,
            )
        if next_slot_count - 1 < min_slot_count:
            return (
                _blocked_dynamic_k_action(
                    proposal,
                    apply_at=apply_at,
                    reason="min_slot_count would be violated",
                ),
                next_slot_count,
            )
        return (
            _accepted_dynamic_k_action(
                proposal,
                slot_delta=-1,
                apply_at=apply_at,
                reason="merge duplicate slots at epoch boundary",
            ),
            next_slot_count - 1,
        )
    if proposal.kind == "split_mixed":
        if max_slot_count is not None and next_slot_count + 1 > max_slot_count:
            return (
                _blocked_dynamic_k_action(
                    proposal,
                    apply_at=apply_at,
                    reason="max_slot_count would be violated",
                ),
                next_slot_count,
            )
        return (
            _accepted_dynamic_k_action(
                proposal,
                slot_delta=1,
                apply_at=apply_at,
                reason="allocate one child slot for mixed slot at epoch boundary",
            ),
            next_slot_count + 1,
        )
    if proposal.kind == "birth_unmatched":
        return (
            _accepted_dynamic_k_action(
                proposal,
                slot_delta=0,
                apply_at=apply_at,
                reason="accept unmatched current slot as born object",
            ),
            next_slot_count,
        )
    return (
        _blocked_dynamic_k_action(
            proposal,
            apply_at=apply_at,
            reason=f"unsupported dynamic-K proposal kind: {proposal.kind}",
        ),
        next_slot_count,
    )


def _accepted_dynamic_k_action(
    proposal: DynamicKProposal,
    *,
    slot_delta: int,
    apply_at: str,
    reason: str,
) -> DynamicKUpdateAction:
    return DynamicKUpdateAction(
        kind=proposal.kind,
        source_ids=tuple(int(source) for source in proposal.source_ids),
        target_id=None if proposal.target_id is None else int(proposal.target_id),
        accepted=True,
        slot_delta=int(slot_delta),
        apply_at=apply_at,
        reason=reason,
        evidence={
            **proposal.evidence,
            "proposal_score": float(proposal.score),
            "proposal_threshold": float(proposal.threshold),
            "proposal_reason": proposal.reason,
        },
    )


def _blocked_dynamic_k_action(
    proposal: DynamicKProposal,
    *,
    apply_at: str,
    reason: str,
) -> DynamicKUpdateAction:
    return DynamicKUpdateAction(
        kind=proposal.kind,
        source_ids=tuple(int(source) for source in proposal.source_ids),
        target_id=None if proposal.target_id is None else int(proposal.target_id),
        accepted=False,
        slot_delta=0,
        apply_at=apply_at,
        reason=reason,
        evidence={
            **proposal.evidence,
            "proposal_score": float(proposal.score),
            "proposal_threshold": float(proposal.threshold),
            "proposal_reason": proposal.reason,
        },
    )


def _dynamic_k_diagnostics(proposals: list[DynamicKProposal]) -> tuple[str, ...]:
    if not proposals:
        return ("no_dynamic_k_proposals",)
    kinds = tuple(sorted({proposal.kind for proposal in proposals}))
    return tuple(f"has_{kind}" for kind in kinds)


def _dynamic_k_update_diagnostics(
    *,
    current_slot_count: int,
    next_slot_count: int,
    accepted_count: int,
    blocked_count: int,
    proposal_count: int,
) -> tuple[str, ...]:
    flags: list[str] = ["epoch_boundary_update"]
    if proposal_count == 0:
        flags.append("no_dynamic_k_proposals")
    if accepted_count > 0:
        flags.append("has_accepted_actions")
    if blocked_count > 0:
        flags.append("has_blocked_actions")
    if next_slot_count > current_slot_count:
        flags.append("slot_count_increase")
    elif next_slot_count < current_slot_count:
        flags.append("slot_count_decrease")
    else:
        flags.append("slot_count_stable")
    return tuple(flags)


def _project_slot(
    slot: int,
    weights: np.ndarray,
    xyz: np.ndarray,
    features: np.ndarray,
    row_entropy: np.ndarray,
    normalized_row_entropy: np.ndarray,
    *,
    evidence_count: int,
    support_threshold: float,
    low_confidence_mass: float,
    mixed_entropy_threshold: float,
) -> ObjectState:
    mass = float(np.sum(weights))
    active = mass > _EPS
    centroid = _weighted_mean(xyz, weights, mass)
    feature = _weighted_mean(features, weights, mass)
    bbox = _bbox(xyz, weights > support_threshold)
    entropy = _weighted_scalar(row_entropy, weights, mass)
    normalized_entropy = _weighted_scalar(normalized_row_entropy, weights, mass)
    diagnostics = _slot_diagnostics(
        mass,
        normalized_entropy,
        low_confidence_mass=low_confidence_mass,
        mixed_entropy_threshold=mixed_entropy_threshold,
    )
    return ObjectState(
        id=int(slot),
        slot_mass=mass,
        confidence=mass,
        mass_fraction=float(mass / evidence_count) if evidence_count else 0.0,
        assignment_entropy=entropy,
        normalized_assignment_entropy=normalized_entropy,
        centroid=centroid,
        bbox=bbox,
        feature=feature,
        status=_status(diagnostics),
        diagnostics=diagnostics if active else ("inactive_slot",),
    )


def _evidence_features(cloud: GaussianCloud, features: np.ndarray | None) -> np.ndarray:
    values = extract_features(cloud) if features is None else np.asarray(features, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != cloud.count:
        raise ValueError("evidence_features must be an N x D matrix matching the cloud")
    return values.astype(np.float32, copy=False)


def _weighted_mean(values: np.ndarray, weights: np.ndarray, mass: float) -> np.ndarray:
    if mass <= _EPS:
        return np.full(values.shape[1], np.nan, dtype=np.float32)
    return ((weights.astype(np.float32) @ values.astype(np.float32)) / mass).astype(np.float32, copy=False)


def _weighted_scalar(values: np.ndarray, weights: np.ndarray, mass: float) -> float:
    if mass <= _EPS:
        return 0.0
    return float(np.dot(weights.astype(np.float32), values.astype(np.float32)) / mass)


def _bbox(xyz: np.ndarray, selected: np.ndarray) -> np.ndarray:
    if not np.any(selected):
        return np.full(6, np.nan, dtype=np.float32)
    points = xyz[selected]
    return np.concatenate([points.min(axis=0), points.max(axis=0)]).astype(np.float32, copy=False)


def _row_entropy(assignment: np.ndarray) -> np.ndarray:
    return -np.sum(assignment * np.log(np.clip(assignment, _EPS, 1.0)), axis=1)


def _normalized_row_entropy(entropy: np.ndarray, slots: int) -> np.ndarray:
    if slots == 1:
        return np.zeros_like(entropy, dtype=np.float32)
    return (entropy / np.log(slots)).astype(np.float32, copy=False)


def _effective_slots(slot_mass_fraction: np.ndarray, total_mass: float) -> float:
    if total_mass <= _EPS:
        return 0.0
    probabilities = slot_mass_fraction[slot_mass_fraction > 0.0]
    entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, _EPS, 1.0))))
    return float(np.exp(entropy))


def _purity_metrics(
    assignment: np.ndarray,
    labels: np.ndarray | None,
) -> tuple[tuple[float | None, ...], float | None]:
    if labels is None:
        return tuple(None for _ in range(assignment.shape[1])), None
    target_labels = np.asarray(labels)
    if target_labels.ndim != 1 or target_labels.shape[0] != assignment.shape[0]:
        raise ValueError("purity_labels must be a 1D array matching the evidence count")
    if target_labels.size == 0:
        return tuple(None for _ in range(assignment.shape[1])), None
    try:
        target_labels = target_labels.astype(np.int64, copy=False)
    except Exception as error:
        raise ValueError("purity_labels must contain integer labels") from error
    if np.any(target_labels < 0):
        raise ValueError("purity_labels must be non-negative")

    slot_purity: list[float | None] = []
    weighted_sum = 0.0
    weighted_mass = 0.0
    for slot in range(assignment.shape[1]):
        weights = assignment[:, slot]
        mass = float(weights.sum())
        if mass <= _EPS:
            slot_purity.append(None)
            continue
        label_mass = np.bincount(target_labels, weights=weights)
        purity = float(label_mass.max() / mass) if label_mass.size else 0.0
        slot_purity.append(purity)
        weighted_sum += purity * mass
        weighted_mass += mass
    overall = None if weighted_mass <= _EPS else float(weighted_sum / weighted_mass)
    return tuple(slot_purity), overall


def _stability_diagnostics(
    *,
    evidence_count: int,
    assignment_confidence: float,
    assignment_confidence_floor: float,
    inactive_slots: tuple[int, ...],
    low_confidence_slots: tuple[int, ...],
    mixed_slots: tuple[int, ...],
    slot_collapse: bool,
    dominant_slot: int | None,
    object_purity: float | None,
    purity_floor: float,
) -> tuple[str, ...]:
    flags: list[str] = []
    if evidence_count == 0:
        flags.append("no_evidence")
    if assignment_confidence < assignment_confidence_floor:
        flags.append("low_assignment_confidence")
    if inactive_slots:
        flags.append("inactive_slots")
    if low_confidence_slots:
        flags.append("low_confidence_slots")
    if mixed_slots:
        flags.append("mixed_slots")
    if slot_collapse:
        suffix = "unknown" if dominant_slot is None else str(dominant_slot)
        flags.append(f"slot_collapse:{suffix}")
    if object_purity is not None and object_purity < purity_floor:
        flags.append("low_object_purity")
    return tuple(flags)


def _slot_diagnostics(
    mass: float,
    normalized_entropy: float,
    *,
    low_confidence_mass: float,
    mixed_entropy_threshold: float,
) -> tuple[str, ...]:
    flags: list[str] = []
    if mass <= _EPS:
        flags.append("inactive_slot")
    if 0.0 < mass < low_confidence_mass:
        flags.append("low_confidence_slot")
    if mass > _EPS and normalized_entropy >= mixed_entropy_threshold:
        flags.append("mixed_slot")
    return tuple(flags)


def _status(diagnostics: tuple[str, ...]) -> str:
    if "inactive_slot" in diagnostics:
        return "inactive"
    if "mixed_slot" in diagnostics:
        return "mixed"
    if "low_confidence_slot" in diagnostics:
        return "low_confidence"
    return "active"
