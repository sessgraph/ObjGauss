from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_field import field_from_labels
from objgauss.core.object_state import (
    ObjectStateProjection,
    dynamic_k_proposal_report,
    match_object_states,
    object_state_stability_report,
    project_object_states,
    project_object_states_from_field,
)

OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA = "objgauss-object-state-stability-benchmark-v1"

DEFAULT_OBJECT_STATE_BENCHMARK_THRESHOLDS: dict[str, float] = {
    "assignment_confidence_floor": 0.5,
    "purity_floor": 0.8,
    "collapse_mass_fraction": 0.9,
    "mixed_entropy_threshold": 0.8,
    "temporal_drift_warning": 0.03,
    "duplicate_feature_distance": 0.01,
    "duplicate_centroid_distance": 0.05,
}

__all__ = (
    "OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA",
    "DEFAULT_OBJECT_STATE_BENCHMARK_THRESHOLDS",
    "object_state_stability_benchmark",
    "write_object_state_stability_benchmark",
    "validate_object_state_stability_benchmark",
)


def object_state_stability_benchmark(
    *,
    report_id: str = "objectstate-stability-synthetic-v1",
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run deterministic pre-training ObjectState stability pressure cases.

    The suite intentionally includes unhealthy assignments. A case passes when
    the expected diagnostic evidence is emitted; `observed_status` records
    whether the simulated state itself is healthy or warning-worthy.
    """

    limits = {**DEFAULT_OBJECT_STATE_BENCHMARK_THRESHOLDS, **(thresholds or {})}
    cases = [
        _clean_sparse_case(limits),
        _uniform_mixed_case(limits),
        _single_slot_collapse_case(limits),
        _soft_noise_case(limits),
        _slot_permutation_case(limits),
        _temporal_jitter_case(limits),
        _birth_unmatched_case(limits),
        _duplicate_fragment_case(limits),
    ]
    failed = [case for case in cases if case["status"] != "pass"]
    observed_warn = [case for case in cases if case["observed_status"] == "warn"]
    coverage = sorted({mode for case in cases for mode in case["failure_modes"]})
    report = {
        "schema": OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA,
        "report_id": report_id,
        "status": "pass" if not failed else "warn",
        "thresholds": limits,
        "aggregate": {
            "case_count": len(cases),
            "warn_count": len(failed),
            "observed_warn_count": len(observed_warn),
            "failure_mode_coverage": coverage,
        },
        "cases": cases,
        "limitations": [
            "Synthetic dependency-free pressure suite for ObjectState kernel regression.",
            "This is a pre-training stability gate, not a production quality certificate.",
            "Renderer, SAM, CLIP, torch, gsplat, CUDA, and large scene assets are intentionally out of scope.",
        ],
    }
    validate_object_state_stability_benchmark(report)
    return report


def write_object_state_stability_benchmark(
    path: str | Path,
    *,
    report_id: str = "objectstate-stability-synthetic-v1",
    thresholds: dict[str, float] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    report = object_state_stability_benchmark(report_id=report_id, thresholds=thresholds)
    validate_object_state_stability_benchmark(report, strict=strict)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def validate_object_state_stability_benchmark(
    report: dict[str, Any],
    *,
    strict: bool = False,
) -> bool:
    if not isinstance(report, dict):
        raise TypeError("object state stability benchmark must be a dict")
    if report.get("schema") != OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA:
        raise ValueError(f"unsupported object state stability benchmark schema: {report.get('schema')}")
    if report.get("status") not in {"pass", "warn"}:
        raise ValueError("object state stability benchmark status must be pass or warn")
    cases = report.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("object state stability benchmark cases must be a non-empty list")
    for case in cases:
        if not isinstance(case, dict):
            raise ValueError("object state stability benchmark cases must be objects")
        if case.get("status") not in {"pass", "warn"}:
            raise ValueError("object state stability benchmark case status must be pass or warn")
        if case.get("observed_status") not in {"pass", "warn"}:
            raise ValueError("object state stability benchmark observed_status must be pass or warn")
        checks = case.get("checks")
        if not isinstance(checks, list) or not checks:
            raise ValueError("object state stability benchmark case checks must be non-empty")
        for check in checks:
            if not isinstance(check, dict) or not isinstance(check.get("passed"), bool):
                raise ValueError("object state stability benchmark checks must contain boolean passed")
    aggregate = report.get("aggregate")
    if not isinstance(aggregate, dict) or int(aggregate.get("case_count", 0)) != len(cases):
        raise ValueError("object state stability benchmark aggregate case_count is invalid")
    if strict and report.get("status") != "pass":
        raise ValueError("object state stability benchmark strict gate failed")
    return True


def _clean_sparse_case(thresholds: dict[str, float]) -> dict[str, Any]:
    projection = project_object_states_from_field(
        _base_cloud(),
        field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0),
        evidence_features=_base_features(),
    )
    return _case_record(
        name="clean_sparse",
        description="one-hot two-object assignment should remain healthy",
        projection=projection,
        purity_labels=np.array([0, 0, 1, 1], dtype=np.int32),
        thresholds=thresholds,
        expected_diagnostics=(),
        failure_modes=(),
    )


def _uniform_mixed_case(thresholds: dict[str, float]) -> dict[str, Any]:
    assignment = np.full((4, 2), 0.5, dtype=np.float32)
    projection = project_object_states(_base_cloud(), assignment, evidence_features=_base_features())
    return _case_record(
        name="uniform_mixed",
        description="uniform A should expose high entropy, mixed slots, and low purity",
        projection=projection,
        purity_labels=np.array([0, 0, 1, 1], dtype=np.int32),
        thresholds=thresholds,
        expected_diagnostics=("low_assignment_confidence", "mixed_slots", "low_object_purity"),
        failure_modes=("uniform_assignment", "mixed_slots", "low_object_purity"),
    )


def _single_slot_collapse_case(thresholds: dict[str, float]) -> dict[str, Any]:
    assignment = np.array(
        [
            [0.97, 0.03],
            [0.97, 0.03],
            [0.97, 0.03],
            [0.97, 0.03],
        ],
        dtype=np.float32,
    )
    projection = project_object_states(_base_cloud(), assignment, evidence_features=_base_features())
    return _case_record(
        name="single_slot_collapse",
        description="dominant slot should emit collapse and low-confidence diagnostics",
        projection=projection,
        thresholds=thresholds,
        expected_diagnostics=("low_confidence_slots", "slot_collapse:0"),
        failure_modes=("slot_collapse", "low_confidence_slot"),
    )


def _soft_noise_case(thresholds: dict[str, float]) -> dict[str, Any]:
    assignment = np.array(
        [
            [0.60, 0.40],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.40, 0.60],
        ],
        dtype=np.float32,
    )
    projection = project_object_states(_base_cloud(), assignment, evidence_features=_base_features())
    return _case_record(
        name="soft_noise",
        description="ambiguous soft assignment should surface entropy and purity warnings",
        projection=projection,
        purity_labels=np.array([0, 0, 1, 1], dtype=np.int32),
        thresholds=thresholds,
        expected_diagnostics=("low_assignment_confidence", "mixed_slots", "low_object_purity"),
        failure_modes=("soft_assignment_noise", "mixed_slots", "low_object_purity"),
    )


def _slot_permutation_case(thresholds: dict[str, float]) -> dict[str, Any]:
    previous = project_object_states_from_field(
        _base_cloud(),
        field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0),
        evidence_features=_base_features(),
    )
    current = project_object_states(
        _base_cloud(),
        np.array(
            [
                [0.0, 1.0],
                [0.0, 1.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ],
            dtype=np.float32,
        ),
        evidence_features=_base_features(),
    )
    return _case_record(
        name="slot_permutation",
        description="slot id swap should be stable after ObjectState matching",
        projection=current,
        previous_projection=previous,
        purity_labels=np.array([0, 0, 1, 1], dtype=np.int32),
        thresholds=thresholds,
        expected_diagnostics=(),
        expected_temporal_diagnostics=(),
        expected_raw_assignment_jitter_gte=0.9,
        expected_max_temporal_drift_lte=1e-6,
        expected_slot_permutation_resolved=True,
        failure_modes=("slot_permutation", "raw_slot_jitter"),
    )


def _temporal_jitter_case(thresholds: dict[str, float]) -> dict[str, Any]:
    previous = project_object_states_from_field(
        _base_cloud(),
        field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0),
        evidence_features=_base_features(),
    )
    current = project_object_states_from_field(
        _base_cloud(offset=(0.08, 0.0, 0.0)),
        field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0),
        evidence_features=_base_features(),
    )
    return _case_record(
        name="temporal_jitter",
        description="small geometry drift should be reported by temporal matching",
        projection=current,
        previous_projection=previous,
        thresholds=thresholds,
        expected_diagnostics=(),
        expected_temporal_diagnostics=("high_temporal_drift",),
        expected_max_temporal_drift_gte=thresholds["temporal_drift_warning"],
        failure_modes=("temporal_jitter",),
    )


def _birth_unmatched_case(thresholds: dict[str, float]) -> dict[str, Any]:
    previous = project_object_states_from_field(
        _base_cloud(),
        field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0),
        evidence_features=_base_features(),
    )
    current = project_object_states_from_field(
        _three_object_cloud(),
        field_from_labels(np.array([0, 0, 1, 1, 2, 2], dtype=np.int32), slots=3, confidence=1.0),
        evidence_features=_three_object_features(),
    )
    return _case_record(
        name="birth_unmatched",
        description="new current object should remain unmatched and emit birth proposal",
        projection=current,
        previous_projection=previous,
        thresholds=thresholds,
        expected_diagnostics=(),
        expected_temporal_diagnostics=("unmatched_current",),
        expected_dynamic_proposals=("birth_unmatched",),
        failure_modes=("birth_unmatched",),
    )


def _duplicate_fragment_case(thresholds: dict[str, float]) -> dict[str, Any]:
    projection = project_object_states_from_field(
        _duplicate_cloud(),
        field_from_labels(np.array([0, 1, 2], dtype=np.int32), slots=3, confidence=1.0),
        evidence_features=_duplicate_features(),
    )
    return _case_record(
        name="duplicate_fragment",
        description="near-duplicate slots should emit merge proposal without mutating K",
        projection=projection,
        purity_labels=np.array([0, 0, 1], dtype=np.int32),
        thresholds=thresholds,
        expected_diagnostics=(),
        expected_dynamic_proposals=("merge_duplicate",),
        failure_modes=("duplicate_fragment", "merge_proposal"),
    )


def _case_record(
    *,
    name: str,
    description: str,
    projection: ObjectStateProjection,
    thresholds: dict[str, float],
    expected_diagnostics: Sequence[str],
    failure_modes: Sequence[str],
    purity_labels: np.ndarray | None = None,
    previous_projection: ObjectStateProjection | None = None,
    expected_temporal_diagnostics: Sequence[str] | None = None,
    expected_dynamic_proposals: Sequence[str] | None = None,
    expected_raw_assignment_jitter_gte: float | None = None,
    expected_max_temporal_drift_lte: float | None = None,
    expected_max_temporal_drift_gte: float | None = None,
    expected_slot_permutation_resolved: bool | None = None,
) -> dict[str, Any]:
    stability = object_state_stability_report(
        projection,
        purity_labels=purity_labels,
        assignment_confidence_floor=thresholds["assignment_confidence_floor"],
        collapse_mass_fraction=thresholds["collapse_mass_fraction"],
        mixed_entropy_threshold=thresholds["mixed_entropy_threshold"],
        purity_floor=thresholds["purity_floor"],
    )
    temporal = (
        match_object_states(
            previous_projection,
            projection,
            max_cost=None,
            drift_warning_threshold=thresholds["temporal_drift_warning"],
        )
        if previous_projection is not None
        else None
    )
    dynamic_k = dynamic_k_proposal_report(
        projection,
        stability_report=stability,
        temporal_match=temporal,
        duplicate_feature_distance_threshold=thresholds["duplicate_feature_distance"],
        duplicate_centroid_distance_threshold=thresholds["duplicate_centroid_distance"],
    )
    metrics = _case_metrics(projection, stability, previous_projection=previous_projection, temporal=temporal)
    proposal_kinds = tuple(proposal.kind for proposal in dynamic_k.proposals)
    checks = _expectation_checks(
        stability_diagnostics=stability.diagnostics,
        temporal_diagnostics=temporal.diagnostics if temporal is not None else (),
        proposal_kinds=proposal_kinds,
        metrics=metrics,
        expected_diagnostics=expected_diagnostics,
        expected_temporal_diagnostics=expected_temporal_diagnostics,
        expected_dynamic_proposals=expected_dynamic_proposals,
        expected_raw_assignment_jitter_gte=expected_raw_assignment_jitter_gte,
        expected_max_temporal_drift_lte=expected_max_temporal_drift_lte,
        expected_max_temporal_drift_gte=expected_max_temporal_drift_gte,
        expected_slot_permutation_resolved=expected_slot_permutation_resolved,
    )
    observed_warn = bool(
        stability.diagnostics
        or (temporal is not None and temporal.diagnostics)
        or dynamic_k.proposals
    )
    return {
        "name": name,
        "description": description,
        "status": "pass" if all(check["passed"] for check in checks) else "warn",
        "observed_status": "warn" if observed_warn else "pass",
        "failure_modes": list(failure_modes),
        "expected": {
            "stability_diagnostics": list(expected_diagnostics),
            "temporal_diagnostics": list(expected_temporal_diagnostics or ()),
            "dynamic_proposals": list(expected_dynamic_proposals or ()),
        },
        "checks": checks,
        "metrics": metrics,
        "stability": _stability_summary(stability),
        "temporal": _temporal_summary(temporal),
        "dynamic_k": {
            "proposal_count": int(dynamic_k.proposal_count),
            "proposal_kinds": list(proposal_kinds),
            "diagnostics": list(dynamic_k.diagnostics),
        },
    }


def _expectation_checks(
    *,
    stability_diagnostics: Sequence[str],
    temporal_diagnostics: Sequence[str],
    proposal_kinds: Sequence[str],
    metrics: dict[str, Any],
    expected_diagnostics: Sequence[str],
    expected_temporal_diagnostics: Sequence[str] | None,
    expected_dynamic_proposals: Sequence[str] | None,
    expected_raw_assignment_jitter_gte: float | None,
    expected_max_temporal_drift_lte: float | None,
    expected_max_temporal_drift_gte: float | None,
    expected_slot_permutation_resolved: bool | None,
) -> list[dict[str, Any]]:
    checks = [
        _check_subset("stability_diagnostics", expected_diagnostics, stability_diagnostics),
    ]
    if expected_temporal_diagnostics is not None:
        checks.append(
            _check_subset(
                "temporal_diagnostics",
                expected_temporal_diagnostics,
                temporal_diagnostics,
            )
        )
    if expected_dynamic_proposals is not None:
        checks.append(_check_subset("dynamic_proposals", expected_dynamic_proposals, proposal_kinds))
    if expected_raw_assignment_jitter_gte is not None:
        value = float(metrics.get("raw_assignment_jitter") or 0.0)
        checks.append(
            {
                "name": "raw_assignment_jitter_gte",
                "passed": value >= expected_raw_assignment_jitter_gte,
                "value": value,
                "threshold": float(expected_raw_assignment_jitter_gte),
            }
        )
    if expected_max_temporal_drift_lte is not None:
        value = float(metrics.get("max_temporal_drift") or 0.0)
        checks.append(
            {
                "name": "max_temporal_drift_lte",
                "passed": value <= expected_max_temporal_drift_lte,
                "value": value,
                "threshold": float(expected_max_temporal_drift_lte),
            }
        )
    if expected_max_temporal_drift_gte is not None:
        value = float(metrics.get("max_temporal_drift") or 0.0)
        checks.append(
            {
                "name": "max_temporal_drift_gte",
                "passed": value >= expected_max_temporal_drift_gte,
                "value": value,
                "threshold": float(expected_max_temporal_drift_gte),
            }
        )
    if expected_slot_permutation_resolved is not None:
        value = bool(metrics.get("slot_permutation_resolved"))
        checks.append(
            {
                "name": "slot_permutation_resolved",
                "passed": value is expected_slot_permutation_resolved,
                "value": value,
                "expected": expected_slot_permutation_resolved,
            }
        )
    return checks


def _check_subset(name: str, expected: Sequence[str], observed: Sequence[str]) -> dict[str, Any]:
    observed_set = set(observed)
    missing = [item for item in expected if item not in observed_set]
    return {
        "name": name,
        "passed": not missing,
        "expected": list(expected),
        "observed": list(observed),
        "missing": missing,
    }


def _case_metrics(
    projection: ObjectStateProjection,
    stability: Any,
    *,
    previous_projection: ObjectStateProjection | None,
    temporal: Any,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "assignment_confidence": _round6(stability.assignment_confidence),
        "mean_normalized_entropy": _round6(stability.mean_normalized_entropy),
        "effective_slots": _round6(stability.effective_slots),
        "dominant_slot_mass_fraction": _round6(stability.dominant_slot_mass_fraction),
        "object_purity": _round6(stability.object_purity),
        "label_fragmentation": _round6(_label_fragmentation(projection.assignment)),
        "bbox_diagonal_mean": _round6(_bbox_diagonal_mean(projection)),
    }
    if previous_projection is not None:
        raw_jitter = _assignment_jitter(previous_projection.assignment, projection.assignment)
        metrics.update(
            {
                "raw_assignment_jitter": _round6(raw_jitter),
                "mean_temporal_drift": _round6(temporal.mean_temporal_drift if temporal else None),
                "max_temporal_drift": _round6(temporal.max_temporal_drift if temporal else None),
                "matched_identity_count": len(temporal.matches) if temporal is not None else 0,
                "slot_permutation_resolved": bool(
                    raw_jitter is not None
                    and raw_jitter > 0.0
                    and temporal is not None
                    and temporal.max_temporal_drift <= 1e-6
                    and len(temporal.matches) > 0
                ),
                "bbox_drift_mean": _round6(_matched_bbox_drift(previous_projection, projection, temporal)),
            }
        )
    return metrics


def _stability_summary(report: Any) -> dict[str, Any]:
    return {
        "assignment_confidence": _round6(report.assignment_confidence),
        "mean_normalized_entropy": _round6(report.mean_normalized_entropy),
        "effective_slots": _round6(report.effective_slots),
        "inactive_slots": list(report.inactive_slots),
        "low_confidence_slots": list(report.low_confidence_slots),
        "mixed_slots": list(report.mixed_slots),
        "slot_collapse": bool(report.slot_collapse),
        "dominant_slot": report.dominant_slot,
        "dominant_slot_mass_fraction": _round6(report.dominant_slot_mass_fraction),
        "object_purity": _round6(report.object_purity),
        "diagnostics": list(report.diagnostics),
    }


def _temporal_summary(report: Any | None) -> dict[str, Any] | None:
    if report is None:
        return None
    return {
        "previous_count": int(report.previous_count),
        "current_count": int(report.current_count),
        "matches": [
            {
                "previous_id": int(match.previous_id),
                "current_id": int(match.current_id),
                "cost": _round6(match.cost),
                "temporal_drift": _round6(match.temporal_drift),
                "bbox_iou": _round6(match.bbox_iou),
                "feature_distance": _round6(match.feature_distance),
                "mass_delta": _round6(match.mass_delta),
            }
            for match in report.matches
        ],
        "unmatched_previous": list(report.unmatched_previous),
        "unmatched_current": list(report.unmatched_current),
        "ignored_previous": list(report.ignored_previous),
        "ignored_current": list(report.ignored_current),
        "mean_temporal_drift": _round6(report.mean_temporal_drift),
        "max_temporal_drift": _round6(report.max_temporal_drift),
        "diagnostics": list(report.diagnostics),
    }


def _label_fragmentation(assignment: np.ndarray) -> float:
    if assignment.size == 0:
        return 0.0
    labels = np.argmax(assignment, axis=1).astype(np.int64, copy=False)
    total = 0.0
    weighted = 0.0
    for label in np.unique(labels):
        rows = assignment[labels == label]
        mass = float(rows.sum())
        if mass <= 0.0:
            continue
        dominant = float(rows.sum(axis=0).max() / mass)
        weighted += (1.0 - dominant) * mass
        total += mass
    return 0.0 if total <= 0.0 else float(weighted / total)


def _assignment_jitter(previous: np.ndarray, current: np.ndarray) -> float | None:
    if previous.shape != current.shape:
        return None
    return float(np.mean(np.abs(current.astype(np.float32) - previous.astype(np.float32))))


def _bbox_diagonal_mean(projection: ObjectStateProjection) -> float | None:
    values: list[float] = []
    for state in projection.states:
        bbox = np.asarray(state.bbox, dtype=np.float32)
        if bbox.shape == (6,) and np.isfinite(bbox).all():
            values.append(float(np.linalg.norm(np.maximum(bbox[3:] - bbox[:3], 0.0))))
    if not values:
        return None
    return float(np.mean(values))


def _matched_bbox_drift(
    previous: ObjectStateProjection,
    current: ObjectStateProjection,
    temporal: Any | None,
) -> float | None:
    if temporal is None or not temporal.matches:
        return None
    previous_by_id = {int(state.id): state for state in previous.states}
    current_by_id = {int(state.id): state for state in current.states}
    values: list[float] = []
    for match in temporal.matches:
        prev = previous_by_id.get(int(match.previous_id))
        curr = current_by_id.get(int(match.current_id))
        if prev is None or curr is None:
            continue
        if np.isfinite(prev.bbox).all() and np.isfinite(curr.bbox).all():
            values.append(float(np.linalg.norm(curr.bbox - prev.bbox)))
    if not values:
        return None
    return float(np.mean(values))


def _base_cloud(offset: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> GaussianCloud:
    vertices = _vertices(4)
    delta = np.asarray(offset, dtype=np.float32)
    vertices["x"] = np.array([-1.0, -0.8, 0.8, 1.0], dtype=np.float32) + delta[0]
    vertices["y"] = np.array([0.0, 0.1, 0.0, -0.1], dtype=np.float32) + delta[1]
    vertices["z"] = np.zeros(4, dtype=np.float32) + delta[2]
    vertices["red"] = np.array([255, 245, 10, 20], dtype=np.uint8)
    vertices["green"] = np.array([10, 20, 245, 255], dtype=np.uint8)
    vertices["blue"] = np.array([0, 5, 10, 15], dtype=np.uint8)
    vertices["opacity"] = np.ones(4, dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="object-state-stability-benchmark")


def _three_object_cloud() -> GaussianCloud:
    vertices = _vertices(6)
    vertices["x"] = np.array([-1.0, -0.8, 0.8, 1.0, 4.0, 4.2], dtype=np.float32)
    vertices["y"] = np.array([0.0, 0.1, 0.0, -0.1, 0.0, 0.1], dtype=np.float32)
    vertices["z"] = np.zeros(6, dtype=np.float32)
    vertices["red"] = np.array([255, 245, 10, 20, 120, 130], dtype=np.uint8)
    vertices["green"] = np.array([10, 20, 245, 255, 120, 130], dtype=np.uint8)
    vertices["blue"] = np.array([0, 5, 10, 15, 240, 250], dtype=np.uint8)
    vertices["opacity"] = np.ones(6, dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="object-state-stability-benchmark")


def _duplicate_cloud() -> GaussianCloud:
    vertices = _vertices(3)
    vertices["x"] = np.array([0.0, 0.02, 1.0], dtype=np.float32)
    vertices["y"] = np.zeros(3, dtype=np.float32)
    vertices["z"] = np.zeros(3, dtype=np.float32)
    vertices["red"] = np.array([255, 254, 10], dtype=np.uint8)
    vertices["green"] = np.array([10, 11, 245], dtype=np.uint8)
    vertices["blue"] = np.array([0, 0, 10], dtype=np.uint8)
    vertices["opacity"] = np.ones(3, dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="object-state-stability-benchmark")


def _vertices(count: int) -> np.ndarray:
    return np.zeros(
        count,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("opacity", "f4"),
        ],
    )


def _base_features() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _three_object_features() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.1, 0.9, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.1, 0.9],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _duplicate_features() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.999, 0.001],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _round6(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not np.isfinite(numeric):
        return None
    return float(round(numeric, 6))
