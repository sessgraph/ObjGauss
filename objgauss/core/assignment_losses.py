from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.object_state import validate_assignment_matrix

ASSIGNMENT_LOSS_V2_SCHEMA = "objgauss-assignment-loss-v2"
_EPS = 1e-8


@dataclass(frozen=True)
class AssignmentLossV2Term:
    name: str
    value: float
    weight: float
    enabled: bool
    status: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": float(self.value),
            "weight": float(self.weight),
            "enabled": bool(self.enabled),
            "status": self.status,
        }


@dataclass(frozen=True)
class AssignmentLossV2Result:
    total_loss: float
    cluster_loss: float
    entropy_loss: float
    balance_loss: float
    temporal_loss: float
    matching_loss: float
    supervised_loss: float
    gradients: tuple[np.ndarray, ...]
    terms: tuple[AssignmentLossV2Term, ...]
    schema: str = ASSIGNMENT_LOSS_V2_SCHEMA

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "loss_family": "assignment_object_loss_v2",
            "total_loss": float(self.total_loss),
            "losses": {
                "cluster": float(self.cluster_loss),
                "entropy": float(self.entropy_loss),
                "balance": float(self.balance_loss),
                "temporal": float(self.temporal_loss),
                "matching": float(self.matching_loss),
                "supervised": float(self.supervised_loss),
            },
            "terms": [term.as_dict() for term in self.terms],
            "enabled_terms": [term.name for term in self.terms if term.enabled],
            "disabled_terms": [term.name for term in self.terms if not term.enabled],
            "gradient_shapes": [
                [int(value) for value in gradient.shape]
                for gradient in self.gradients
            ],
            "gradient_l2": [
                float(np.linalg.norm(gradient))
                for gradient in self.gradients
            ],
        }


def assignment_loss_v2_breakdown(
    assignments: Sequence[np.ndarray],
    *,
    cluster_costs: Sequence[np.ndarray] | None = None,
    target_assignments: Sequence[np.ndarray | None] | None = None,
    cluster_weight: float = 0.0,
    entropy_weight: float = 0.0,
    balance_weight: float = 0.0,
    supervised_weight: float = 0.0,
) -> AssignmentLossV2Result:
    checked = _validate_assignment_sequence(assignments)
    _validate_weight("cluster_weight", cluster_weight)
    _validate_weight("entropy_weight", entropy_weight)
    _validate_weight("balance_weight", balance_weight)
    _validate_weight("supervised_weight", supervised_weight)

    cluster_loss, cluster_gradients = assignment_cluster_loss_and_gradient(
        checked,
        cluster_costs,
    )
    entropy_loss, entropy_gradients = assignment_entropy_loss_and_gradient(checked)
    balance_loss, balance_gradients = assignment_balance_loss_and_gradient(checked)
    supervised_loss, supervised_gradients = supervised_assignment_loss_and_gradient(
        checked,
        target_assignments,
    )
    gradients = tuple(
        (
            cluster_weight * cluster_gradient
            + entropy_weight * entropy_gradient
            + balance_weight * balance_gradient
            + supervised_weight * supervised_gradient
        ).astype(np.float32, copy=False)
        for cluster_gradient, entropy_gradient, balance_gradient, supervised_gradient in zip(
            cluster_gradients,
            entropy_gradients,
            balance_gradients,
            supervised_gradients,
            strict=True,
        )
    )
    total = (
        cluster_weight * cluster_loss
        + entropy_weight * entropy_loss
        + balance_weight * balance_loss
        + supervised_weight * supervised_loss
    )
    terms = (
        AssignmentLossV2Term(
            name="cluster",
            value=cluster_loss,
            weight=cluster_weight,
            enabled=cluster_costs is not None and cluster_weight > 0,
            status="enabled" if cluster_costs is not None and cluster_weight > 0 else "disabled",
        ),
        AssignmentLossV2Term(
            name="entropy",
            value=entropy_loss,
            weight=entropy_weight,
            enabled=entropy_weight > 0,
            status="enabled" if entropy_weight > 0 else "disabled",
        ),
        AssignmentLossV2Term(
            name="balance",
            value=balance_loss,
            weight=balance_weight,
            enabled=balance_weight > 0,
            status="enabled" if balance_weight > 0 else "disabled",
        ),
        AssignmentLossV2Term(
            name="temporal",
            value=0.0,
            weight=0.0,
            enabled=False,
            status="disabled_pending_temporal_matching",
        ),
        AssignmentLossV2Term(
            name="matching",
            value=0.0,
            weight=0.0,
            enabled=False,
            status="disabled_pending_temporal_matching",
        ),
        AssignmentLossV2Term(
            name="supervised",
            value=supervised_loss,
            weight=supervised_weight,
            enabled=target_assignments is not None and supervised_weight > 0,
            status="enabled" if target_assignments is not None and supervised_weight > 0 else "disabled",
        ),
    )
    return AssignmentLossV2Result(
        total_loss=float(total),
        cluster_loss=float(cluster_loss),
        entropy_loss=float(entropy_loss),
        balance_loss=float(balance_loss),
        temporal_loss=0.0,
        matching_loss=0.0,
        supervised_loss=float(supervised_loss),
        gradients=gradients,
        terms=terms,
    )


def assignment_cluster_loss_and_gradient(
    assignments: Sequence[np.ndarray],
    cluster_costs: Sequence[np.ndarray] | None,
) -> tuple[float, tuple[np.ndarray, ...]]:
    checked = _validate_assignment_sequence(assignments)
    if cluster_costs is None:
        return 0.0, tuple(np.zeros_like(assignment, dtype=np.float32) for assignment in checked)
    costs = _validate_cost_sequence(cluster_costs, checked)
    frame_count = max(len(checked), 1)
    losses: list[float] = []
    gradients: list[np.ndarray] = []
    for assignment, cost in zip(checked, costs, strict=True):
        evidence_count = max(float(assignment.shape[0]), _EPS)
        losses.append(float(np.mean(np.sum(assignment * cost, axis=1))))
        gradients.append((cost / evidence_count / frame_count).astype(np.float32, copy=False))
    return float(np.mean(losses)) if losses else 0.0, tuple(gradients)


def supervised_assignment_loss_and_gradient(
    assignments: Sequence[np.ndarray],
    target_assignments: Sequence[np.ndarray | None] | None,
) -> tuple[float, tuple[np.ndarray, ...]]:
    checked = _validate_assignment_sequence(assignments)
    if target_assignments is None:
        return 0.0, tuple(np.zeros_like(assignment, dtype=np.float32) for assignment in checked)
    targets = tuple(target_assignments)
    if len(targets) != len(checked):
        raise ValueError("target_assignments must have one entry per assignment")
    losses: list[float] = []
    gradients: list[np.ndarray] = []
    frame_count = max(len(checked), 1)
    for assignment, target in zip(checked, targets, strict=True):
        if target is None:
            gradients.append(np.zeros_like(assignment, dtype=np.float32))
            continue
        checked_target = validate_assignment_matrix(target, evidence_count=assignment.shape[0])
        if checked_target.shape[1] != assignment.shape[1]:
            raise ValueError("target_assignment slots must match solver slots")
        clipped = np.clip(assignment, _EPS, 1.0)
        losses.append(float(-np.mean(np.sum(checked_target * np.log(clipped), axis=1))))
        gradients.append(
            (
                -(checked_target / clipped)
                / max(float(assignment.shape[0]), _EPS)
                / frame_count
            ).astype(np.float32, copy=False)
        )
    return float(np.mean(losses)) if losses else 0.0, tuple(gradients)


def assignment_entropy_loss_and_gradient(
    assignments: Sequence[np.ndarray],
) -> tuple[float, tuple[np.ndarray, ...]]:
    checked = _validate_assignment_sequence(assignments)
    losses: list[float] = []
    gradients: list[np.ndarray] = []
    frame_count = max(len(checked), 1)
    for assignment in checked:
        if assignment.shape[1] <= 1:
            gradients.append(np.zeros_like(assignment, dtype=np.float32))
            continue
        clipped = np.clip(assignment, _EPS, 1.0)
        normalizer = np.log(float(assignment.shape[1]))
        entropy = -np.sum(assignment * np.log(clipped), axis=1) / normalizer
        losses.append(float(np.mean(entropy)))
        gradients.append(
            (
                -(np.log(clipped) + 1.0)
                / normalizer
                / max(float(assignment.shape[0]), _EPS)
                / frame_count
            ).astype(np.float32, copy=False)
        )
    return float(np.mean(losses)) if losses else 0.0, tuple(gradients)


def assignment_balance_loss_and_gradient(
    assignments: Sequence[np.ndarray],
) -> tuple[float, tuple[np.ndarray, ...]]:
    checked = _validate_assignment_sequence(assignments)
    losses: list[float] = []
    gradients: list[np.ndarray] = []
    frame_count = max(len(checked), 1)
    for assignment in checked:
        evidence_count = max(float(assignment.shape[0]), _EPS)
        slots = assignment.shape[1]
        mass_fraction = np.sum(assignment, axis=0) / evidence_count
        target = np.full(slots, 1.0 / float(slots), dtype=np.float32)
        delta = mass_fraction - target
        losses.append(float(np.mean(delta ** 2)))
        per_slot = (2.0 / float(slots)) * delta / evidence_count / frame_count
        gradients.append(np.tile(per_slot[None, :], (assignment.shape[0], 1)).astype(np.float32))
    return float(np.mean(losses)) if losses else 0.0, tuple(gradients)


def validate_assignment_loss_v2_summary(payload: dict[str, Any]) -> bool:
    if not isinstance(payload, dict):
        raise TypeError("assignment loss v2 summary must be a dict")
    if payload.get("schema") != ASSIGNMENT_LOSS_V2_SCHEMA:
        raise ValueError(f"unsupported assignment loss schema: {payload.get('schema')}")
    losses = payload.get("losses")
    if not isinstance(losses, dict):
        raise ValueError("assignment loss summary missing losses")
    for key in ("cluster", "entropy", "balance", "temporal", "matching", "supervised"):
        if key not in losses:
            raise ValueError(f"assignment loss summary missing losses.{key}")
        float(losses[key])
    if not isinstance(payload.get("terms"), list):
        raise ValueError("assignment loss summary missing terms")
    if not isinstance(payload.get("gradient_shapes"), list):
        raise ValueError("assignment loss summary missing gradient_shapes")
    return True


def _validate_assignment_sequence(assignments: Sequence[np.ndarray]) -> tuple[np.ndarray, ...]:
    checked = tuple(
        validate_assignment_matrix(assignment)
        for assignment in assignments
    )
    if not checked:
        raise ValueError("at least one assignment matrix is required")
    return checked


def _validate_cost_sequence(
    costs: Sequence[np.ndarray],
    assignments: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, ...]:
    checked = tuple(np.asarray(cost, dtype=np.float32) for cost in costs)
    if len(checked) != len(assignments):
        raise ValueError("cluster_costs must have one matrix per assignment")
    for index, (cost, assignment) in enumerate(zip(checked, assignments, strict=True)):
        if cost.ndim != 2:
            raise ValueError(f"cluster_costs[{index}] must be a 2D matrix")
        if cost.shape != assignment.shape:
            raise ValueError(f"cluster_costs[{index}] shape must match assignment")
        if not np.isfinite(cost).all():
            raise ValueError(f"cluster_costs[{index}] must contain only finite values")
    return checked


def _validate_weight(label: str, value: float) -> None:
    if value < 0:
        raise ValueError(f"{label} must be >= 0")
