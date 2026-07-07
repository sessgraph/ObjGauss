from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from objgauss.core.v2_stability_diagnostics import (
    FailureModeClassifier,
    IdentitySlotObservation,
    SyntheticStabilityDiagnosticsReport,
    V2_STABILITY_DIAGNOSTICS_SCHEMA,
    V2_STABILITY_FAILURE_MODES,
    diagnose_synthetic_stability_fixture,
)
from objgauss.core.v2_stability_foundation import (
    SyntheticStabilityScenarioFixture,
    make_synthetic_stability_scenario_suite,
    validate_synthetic_stability_scenario_fixture,
)

V2_STABILITY_GATE_SCHEMA = "objgauss-v2-stability-gate-v1"
V2_STABILITY_GATE_SUITE_SCHEMA = "objgauss-v2-stability-gate-suite-v1"
V2_STABILITY_GATE_HARD_CHECKS = (
    "expected_slot_consistency_pass",
    "no_slot_swap_pass",
    "identity_no_cross_slot_drift_pass",
    "adversarial_swap_no_exchange_pass",
    "occlusion_recovery_return_pass",
    "no_object_merge_pass",
    "no_background_absorption_pass",
    "diagnostics_failure_reporting_pass",
)
_GATE_STATUS_PASS = "synthetic_stability_gate_pass"
_GATE_STATUS_FAIL = "synthetic_stability_gate_fail"
_SUITE_STATUS_PASS = "synthetic_stability_suite_gate_pass"
_SUITE_STATUS_FAIL = "synthetic_stability_suite_gate_fail"
_SOFT_STATUS_PASS = "soft_pass"
_SOFT_STATUS_WARN = "soft_warn"


@dataclass(frozen=True)
class SyntheticStabilityGateReport:
    fixture: SyntheticStabilityScenarioFixture
    diagnostics: SyntheticStabilityDiagnosticsReport
    hard_gates: dict[str, bool]
    hard_blockers: tuple[str, ...]
    soft_diagnostics: dict[str, Any]
    schema: str = V2_STABILITY_GATE_SCHEMA

    @property
    def passed(self) -> bool:
        return all(bool(value) for value in self.hard_gates.values())

    def as_dict(self) -> dict[str, Any]:
        diagnostics_summary = self.diagnostics.as_dict()
        summary = {
            "schema": self.schema,
            "kind": "synthetic_stability_gate",
            "status": _GATE_STATUS_PASS if self.passed else _GATE_STATUS_FAIL,
            "scenario_id": self.fixture.scenario_id,
            "scenario_kind": self.fixture.scenario_kind,
            "gate_role": "identity_invariant_hard_gate",
            "gate_policy": {
                "hard_gate_source": "synthetic_oracle_identity",
                "hard_checks": list(V2_STABILITY_GATE_HARD_CHECKS),
                "soft_diagnostics_do_not_override_hard_gate": True,
            },
            "hard_gates": {key: bool(value) for key, value in self.hard_gates.items()},
            "hard_blockers": list(self.hard_blockers),
            "soft_diagnostics": self.soft_diagnostics,
            "diagnostics_schema": diagnostics_summary["schema"],
            "diagnostics_status": diagnostics_summary["status"],
            "failure_mode_counts": diagnostics_summary["failure_mode_counts"],
            "failure_modes": diagnostics_summary["failure_modes"],
            "diagnostics": diagnostics_summary,
            "non_goals": {
                "trains_solver": False,
                "uses_renderer_loss": False,
                "uses_rollout_model": False,
                "mutates_dynamic_k": False,
            },
        }
        return validate_synthetic_stability_gate_summary(summary)


@dataclass(frozen=True)
class SyntheticStabilitySuiteGateReport:
    reports: tuple[SyntheticStabilityGateReport, ...]
    schema: str = V2_STABILITY_GATE_SUITE_SCHEMA

    @property
    def passed(self) -> bool:
        return all(report.passed for report in self.reports)

    def as_dict(self) -> dict[str, Any]:
        report_summaries = [report.as_dict() for report in self.reports]
        failed = [summary for summary in report_summaries if summary["status"] == _GATE_STATUS_FAIL]
        summary = {
            "schema": self.schema,
            "kind": "synthetic_stability_suite_gate",
            "status": _SUITE_STATUS_PASS if self.passed else _SUITE_STATUS_FAIL,
            "gate_role": "identity_invariant_hard_gate",
            "fixture_count": len(report_summaries),
            "passed_count": len(report_summaries) - len(failed),
            "failed_count": len(failed),
            "scenario_statuses": [
                {
                    "scenario_id": report["scenario_id"],
                    "scenario_kind": report["scenario_kind"],
                    "status": report["status"],
                    "hard_blockers": report["hard_blockers"],
                }
                for report in report_summaries
            ],
            "aggregate_failure_mode_counts": _aggregate_failure_mode_counts(report_summaries),
            "hard_blockers": sorted(
                {
                    blocker
                    for report in report_summaries
                    for blocker in report["hard_blockers"]
                }
            ),
            "reports": report_summaries,
            "non_goals": {
                "trains_solver": False,
                "uses_renderer_loss": False,
                "uses_rollout_model": False,
                "mutates_dynamic_k": False,
            },
        }
        return validate_synthetic_stability_suite_gate_summary(summary)


def evaluate_synthetic_stability_gate(
    fixture: SyntheticStabilityScenarioFixture,
    *,
    predicted_slots: Sequence[np.ndarray | Sequence[int]] | None = None,
    predicted_assignments: Sequence[np.ndarray] | None = None,
    classifier: FailureModeClassifier | None = None,
    assignment_entropy_warn_threshold: float = 0.6,
    object_purity_warn_threshold: float = 0.8,
    temporal_coherence_warn_threshold: float = 0.8,
) -> SyntheticStabilityGateReport:
    fixture = validate_synthetic_stability_scenario_fixture(fixture)
    if predicted_slots is None and predicted_assignments is None:
        raise ValueError(
            "synthetic stability gate requires explicit predicted_slots "
            "or predicted_assignments"
        )
    _validate_gate_thresholds(
        assignment_entropy_warn_threshold=assignment_entropy_warn_threshold,
        object_purity_warn_threshold=object_purity_warn_threshold,
        temporal_coherence_warn_threshold=temporal_coherence_warn_threshold,
    )
    diagnostics = diagnose_synthetic_stability_fixture(
        fixture,
        predicted_slots=predicted_slots,
        predicted_assignments=predicted_assignments,
        classifier=classifier,
    )
    hard_gates, hard_blockers = _hard_gate_result(fixture, diagnostics)
    soft_diagnostics = _soft_diagnostics(
        fixture,
        diagnostics,
        predicted_slots=predicted_slots,
        predicted_assignments=predicted_assignments,
        assignment_entropy_warn_threshold=assignment_entropy_warn_threshold,
        object_purity_warn_threshold=object_purity_warn_threshold,
        temporal_coherence_warn_threshold=temporal_coherence_warn_threshold,
    )
    return SyntheticStabilityGateReport(
        fixture=fixture,
        diagnostics=diagnostics,
        hard_gates=hard_gates,
        hard_blockers=hard_blockers,
        soft_diagnostics=soft_diagnostics,
    )


def evaluate_synthetic_stability_suite_gate(
    fixtures: Sequence[SyntheticStabilityScenarioFixture] | None = None,
    *,
    predicted_slots_by_fixture: Sequence[Sequence[np.ndarray | Sequence[int]] | None] | None = None,
    predicted_assignments_by_fixture: Sequence[Sequence[np.ndarray] | None] | None = None,
    classifier: FailureModeClassifier | None = None,
    assignment_entropy_warn_threshold: float = 0.6,
    object_purity_warn_threshold: float = 0.8,
    temporal_coherence_warn_threshold: float = 0.8,
) -> SyntheticStabilitySuiteGateReport:
    resolved_fixtures = tuple(
        make_synthetic_stability_scenario_suite()
        if fixtures is None
        else fixtures
    )
    if not resolved_fixtures:
        raise ValueError("fixtures must contain at least one scenario")
    if predicted_slots_by_fixture is not None and predicted_assignments_by_fixture is not None:
        raise ValueError("provide predicted_slots_by_fixture or predicted_assignments_by_fixture, not both")
    if predicted_slots_by_fixture is None and predicted_assignments_by_fixture is None:
        raise ValueError(
            "synthetic stability suite gate requires explicit predicted_slots_by_fixture "
            "or predicted_assignments_by_fixture"
        )
    slot_predictions = _prediction_sequence(predicted_slots_by_fixture, len(resolved_fixtures))
    assignment_predictions = _prediction_sequence(predicted_assignments_by_fixture, len(resolved_fixtures))
    reports = []
    for index, fixture in enumerate(resolved_fixtures):
        reports.append(
            evaluate_synthetic_stability_gate(
                fixture,
                predicted_slots=slot_predictions[index],
                predicted_assignments=assignment_predictions[index],
                classifier=classifier,
                assignment_entropy_warn_threshold=assignment_entropy_warn_threshold,
                object_purity_warn_threshold=object_purity_warn_threshold,
                temporal_coherence_warn_threshold=temporal_coherence_warn_threshold,
            )
        )
    return SyntheticStabilitySuiteGateReport(reports=tuple(reports))


def validate_synthetic_stability_gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("synthetic stability gate summary must be a dict")
    if payload.get("schema") != V2_STABILITY_GATE_SCHEMA:
        raise ValueError(f"unsupported stability gate schema: {payload.get('schema')}")
    if payload.get("kind") != "synthetic_stability_gate":
        raise ValueError("stability gate kind must be synthetic_stability_gate")
    if payload.get("status") not in {_GATE_STATUS_PASS, _GATE_STATUS_FAIL}:
        raise ValueError("stability gate status is unsupported")
    if payload.get("gate_role") != "identity_invariant_hard_gate":
        raise ValueError("stability gate role must be identity_invariant_hard_gate")
    for key in (
        "hard_gates",
        "hard_blockers",
        "soft_diagnostics",
        "diagnostics_schema",
        "failure_mode_counts",
        "failure_modes",
        "diagnostics",
        "non_goals",
    ):
        if key not in payload:
            raise ValueError(f"stability gate summary missing {key}")
    hard_gates = payload["hard_gates"]
    for key in V2_STABILITY_GATE_HARD_CHECKS:
        if key not in hard_gates:
            raise ValueError(f"stability gate hard_gates missing {key}")
        if not isinstance(hard_gates[key], bool):
            raise ValueError(f"stability gate hard_gates[{key}] must be bool")
    expected_status = _GATE_STATUS_PASS if all(hard_gates.values()) else _GATE_STATUS_FAIL
    if payload["status"] != expected_status:
        raise ValueError("stability gate status must match hard gate results")
    if payload["diagnostics_schema"] != V2_STABILITY_DIAGNOSTICS_SCHEMA:
        raise ValueError("stability gate diagnostics schema is unsupported")
    for mode in V2_STABILITY_FAILURE_MODES:
        if mode not in payload["failure_mode_counts"]:
            raise ValueError(f"stability gate failure_mode_counts missing {mode}")
    soft = payload["soft_diagnostics"]
    for key in ("assignment_entropy", "assignment_purity", "temporal_coherence"):
        if key not in soft:
            raise ValueError(f"stability gate soft_diagnostics missing {key}")
        if soft[key].get("participates_in_hard_gate") is not False:
            raise ValueError(f"soft diagnostic {key} must not participate in hard gate")
    non_goals = payload["non_goals"]
    if (
        non_goals.get("trains_solver")
        or non_goals.get("uses_renderer_loss")
        or non_goals.get("uses_rollout_model")
        or non_goals.get("mutates_dynamic_k")
    ):
        raise ValueError("stability gate cannot train, use renderer loss, rollout, or mutate dynamic-K")
    return payload


def validate_synthetic_stability_suite_gate_summary(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError("synthetic stability suite gate summary must be a dict")
    if payload.get("schema") != V2_STABILITY_GATE_SUITE_SCHEMA:
        raise ValueError(f"unsupported stability suite gate schema: {payload.get('schema')}")
    if payload.get("kind") != "synthetic_stability_suite_gate":
        raise ValueError("stability suite gate kind must be synthetic_stability_suite_gate")
    if payload.get("status") not in {_SUITE_STATUS_PASS, _SUITE_STATUS_FAIL}:
        raise ValueError("stability suite gate status is unsupported")
    reports = payload.get("reports")
    if not isinstance(reports, list) or not reports:
        raise ValueError("stability suite gate requires at least one report")
    for report in reports:
        validate_synthetic_stability_gate_summary(report)
    failed_count = sum(1 for report in reports if report["status"] == _GATE_STATUS_FAIL)
    if payload.get("fixture_count") != len(reports):
        raise ValueError("stability suite fixture_count must match reports")
    if payload.get("failed_count") != failed_count:
        raise ValueError("stability suite failed_count must match reports")
    if payload.get("passed_count") != len(reports) - failed_count:
        raise ValueError("stability suite passed_count must match reports")
    expected_status = _SUITE_STATUS_PASS if failed_count == 0 else _SUITE_STATUS_FAIL
    if payload["status"] != expected_status:
        raise ValueError("stability suite status must match report results")
    non_goals = payload.get("non_goals", {})
    if (
        non_goals.get("trains_solver")
        or non_goals.get("uses_renderer_loss")
        or non_goals.get("uses_rollout_model")
        or non_goals.get("mutates_dynamic_k")
    ):
        raise ValueError("stability suite gate cannot train, use renderer loss, rollout, or mutate dynamic-K")
    return payload


def _hard_gate_result(
    fixture: SyntheticStabilityScenarioFixture,
    diagnostics: SyntheticStabilityDiagnosticsReport,
) -> tuple[dict[str, bool], tuple[str, ...]]:
    observations = diagnostics.identity_observations
    counts = _failure_mode_counts(diagnostics.failure_modes)
    occlusion = _occlusion_recovery_summary(fixture, observations)
    expected_mismatches = [obs for obs in observations if not obs.matches_expected]
    failure_reporting_pass = _diagnostics_failure_reporting_pass(diagnostics)
    hard_gates = {
        "expected_slot_consistency_pass": len(expected_mismatches) == 0,
        "no_slot_swap_pass": counts["slot_swap"] == 0,
        "identity_no_cross_slot_drift_pass": counts["identity_fragmentation"] == 0
        and counts["temporal_drift"] == 0,
        "adversarial_swap_no_exchange_pass": fixture.scenario_kind != "adversarial_swap"
        or counts["slot_swap"] == 0,
        "occlusion_recovery_return_pass": bool(occlusion["pass"]),
        "no_object_merge_pass": counts["object_merge"] == 0,
        "no_background_absorption_pass": counts["background_absorption"] == 0,
        "diagnostics_failure_reporting_pass": failure_reporting_pass,
    }
    blockers: list[str] = []
    if expected_mismatches:
        blockers.append("expected_slot_mismatch")
    for mode in V2_STABILITY_FAILURE_MODES:
        if counts[mode] > 0:
            blockers.append(mode)
    if not occlusion["pass"]:
        blockers.append("occlusion_recovery_not_returned_to_expected_slot")
    if not failure_reporting_pass:
        blockers.append("diagnostics_failure_reporting_missing")
    return hard_gates, tuple(sorted(set(blockers)))


def _soft_diagnostics(
    fixture: SyntheticStabilityScenarioFixture,
    diagnostics: SyntheticStabilityDiagnosticsReport,
    *,
    predicted_slots: Sequence[np.ndarray | Sequence[int]] | None,
    predicted_assignments: Sequence[np.ndarray] | None,
    assignment_entropy_warn_threshold: float,
    object_purity_warn_threshold: float,
    temporal_coherence_warn_threshold: float,
) -> dict[str, Any]:
    source = _prediction_source(
        predicted_slots=predicted_slots,
        predicted_assignments=predicted_assignments,
    )
    return {
        "assignment_entropy": _assignment_entropy_summary(
            fixture,
            predicted_assignments=predicted_assignments,
            source=source,
            warn_threshold=assignment_entropy_warn_threshold,
        ),
        "assignment_purity": _assignment_purity_summary(
            diagnostics.identity_observations,
            background_slot=diagnostics.classifier.background_slot,
            warn_threshold=object_purity_warn_threshold,
        ),
        "temporal_coherence": _temporal_coherence_summary(
            diagnostics.identity_observations,
            warn_threshold=temporal_coherence_warn_threshold,
        ),
    }


def _assignment_entropy_summary(
    fixture: SyntheticStabilityScenarioFixture,
    *,
    predicted_assignments: Sequence[np.ndarray] | None,
    source: str,
    warn_threshold: float,
) -> dict[str, Any]:
    if predicted_assignments is None:
        mean_entropy = 0.0
        row_count = sum(observation.evidence.evidence_count for observation in fixture.observations)
    else:
        entropies = []
        row_count = 0
        for frame_index, (observation, assignment) in enumerate(zip(fixture.observations, predicted_assignments)):
            matrix = _normalized_assignment_array(assignment, f"predicted_assignments[{frame_index}]")
            if matrix.shape[0] != observation.evidence.evidence_count:
                raise ValueError("predicted assignment rows must match observation evidence rows")
            if matrix.shape[1] != fixture.world.oracle.slots:
                raise ValueError("predicted assignment columns must match fixture slot count")
            row_count += matrix.shape[0]
            entropies.append(_normalized_entropy(matrix))
        mean_entropy = float(np.mean(np.concatenate(entropies))) if entropies else 0.0
    return {
        "source": source,
        "mean_normalized_entropy": float(mean_entropy),
        "warn_threshold": float(warn_threshold),
        "status": _SOFT_STATUS_PASS if mean_entropy <= warn_threshold else _SOFT_STATUS_WARN,
        "row_count": int(row_count),
        "participates_in_hard_gate": False,
    }


def _assignment_purity_summary(
    observations: Sequence[IdentitySlotObservation],
    *,
    background_slot: int,
    warn_threshold: float,
) -> dict[str, Any]:
    total = 0
    dominant = 0
    for frame_observations in _observations_by_frame(observations).values():
        by_slot: dict[int, list[int]] = {}
        for obs in frame_observations:
            if int(obs.predicted_slot) == int(background_slot):
                continue
            by_slot.setdefault(int(obs.predicted_slot), []).append(int(obs.oracle_object_id))
        for object_ids in by_slot.values():
            total += len(object_ids)
            dominant += max(object_ids.count(object_id) for object_id in set(object_ids))
    purity = float(dominant / total) if total else 0.0
    return {
        "object_purity": purity,
        "warn_threshold": float(warn_threshold),
        "status": _SOFT_STATUS_PASS if purity >= warn_threshold else _SOFT_STATUS_WARN,
        "evaluated_identity_observations": int(total),
        "participates_in_hard_gate": False,
    }


def _temporal_coherence_summary(
    observations: Sequence[IdentitySlotObservation],
    *,
    warn_threshold: float,
) -> dict[str, Any]:
    transition_count = 0
    stable_count = 0
    for object_observations in _observations_by_object(observations).values():
        ordered = sorted(object_observations, key=lambda obs: int(obs.frame_index))
        for previous, current in zip(ordered, ordered[1:]):
            transition_count += 1
            if int(previous.predicted_slot) == int(current.predicted_slot):
                stable_count += 1
    score = float(stable_count / transition_count) if transition_count else 1.0
    return {
        "temporal_coherence": score,
        "warn_threshold": float(warn_threshold),
        "status": _SOFT_STATUS_PASS if score >= warn_threshold else _SOFT_STATUS_WARN,
        "transition_count": int(transition_count),
        "stable_transition_count": int(stable_count),
        "participates_in_hard_gate": False,
    }


def _occlusion_recovery_summary(
    fixture: SyntheticStabilityScenarioFixture,
    observations: Sequence[IdentitySlotObservation],
) -> dict[str, Any]:
    by_key = {
        (int(obs.frame_index), int(obs.oracle_object_id)): obs
        for obs in observations
    }
    recovery_checks = []
    for frame_index in range(1, fixture.world.oracle.frame_count):
        previous_frame = fixture.world.oracle.frames[frame_index - 1]
        current_frame = fixture.world.oracle.frames[frame_index]
        previous_by_id = {
            int(observation.oracle_object_id): observation
            for observation in previous_frame
        }
        for current in current_frame:
            previous = previous_by_id[int(current.oracle_object_id)]
            if previous.visible or not current.visible:
                continue
            observed = by_key.get((frame_index, int(current.oracle_object_id)))
            passed = observed is not None and observed.matches_expected
            recovery_checks.append(
                {
                    "frame_index": int(frame_index),
                    "oracle_object_id": int(current.oracle_object_id),
                    "expected_slot": int(current.expected_slot),
                    "predicted_slot": None if observed is None else int(observed.predicted_slot),
                    "pass": bool(passed),
                }
            )
    return {
        "required": bool(recovery_checks),
        "pass": all(check["pass"] for check in recovery_checks),
        "checks": recovery_checks,
    }


def _diagnostics_failure_reporting_pass(diagnostics: SyntheticStabilityDiagnosticsReport) -> bool:
    counts = _failure_mode_counts(diagnostics.failure_modes)
    for mode in V2_STABILITY_FAILURE_MODES:
        if counts[mode] != sum(1 for event in diagnostics.failure_modes if event.mode == mode):
            return False
    return True


def _failure_mode_counts(events: Sequence[Any]) -> dict[str, int]:
    counts = {mode: 0 for mode in V2_STABILITY_FAILURE_MODES}
    for event in events:
        counts[str(event.mode)] += 1
    return counts


def _aggregate_failure_mode_counts(reports: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts = {mode: 0 for mode in V2_STABILITY_FAILURE_MODES}
    for report in reports:
        for mode in V2_STABILITY_FAILURE_MODES:
            counts[mode] += int(report["failure_mode_counts"][mode])
    return counts


def _prediction_source(
    *,
    predicted_slots: Sequence[np.ndarray | Sequence[int]] | None,
    predicted_assignments: Sequence[np.ndarray] | None,
) -> str:
    if predicted_assignments is not None:
        return "predicted_assignments"
    if predicted_slots is not None:
        return "predicted_slots"
    return "missing_predictions"


def _prediction_sequence(value: Sequence[Any] | None, expected_length: int) -> tuple[Any, ...]:
    if value is None:
        return tuple(None for _ in range(expected_length))
    if len(value) != expected_length:
        raise ValueError("prediction sequence must cover every fixture")
    return tuple(value)


def _observations_by_frame(
    observations: Sequence[IdentitySlotObservation],
) -> dict[int, tuple[IdentitySlotObservation, ...]]:
    grouped: dict[int, list[IdentitySlotObservation]] = {}
    for obs in observations:
        grouped.setdefault(int(obs.frame_index), []).append(obs)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def _observations_by_object(
    observations: Sequence[IdentitySlotObservation],
) -> dict[int, tuple[IdentitySlotObservation, ...]]:
    grouped: dict[int, list[IdentitySlotObservation]] = {}
    for obs in observations:
        grouped.setdefault(int(obs.oracle_object_id), []).append(obs)
    return {key: tuple(value) for key, value in sorted(grouped.items())}


def _normalized_assignment_array(value: np.ndarray, label: str) -> np.ndarray:
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


def _normalized_entropy(assignments: np.ndarray) -> np.ndarray:
    if assignments.shape[1] <= 1:
        return np.zeros(assignments.shape[0], dtype=np.float32)
    clipped = np.clip(assignments, 1e-8, 1.0)
    entropy = -np.sum(clipped * np.log(clipped), axis=1)
    return (entropy / np.log(assignments.shape[1])).astype(np.float32, copy=False)


def _validate_gate_thresholds(
    *,
    assignment_entropy_warn_threshold: float,
    object_purity_warn_threshold: float,
    temporal_coherence_warn_threshold: float,
) -> None:
    for name, value in {
        "assignment_entropy_warn_threshold": assignment_entropy_warn_threshold,
        "object_purity_warn_threshold": object_purity_warn_threshold,
        "temporal_coherence_warn_threshold": temporal_coherence_warn_threshold,
    }.items():
        if not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
