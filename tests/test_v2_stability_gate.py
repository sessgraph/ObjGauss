from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.v2_stability_diagnostics import expected_slots_for_synthetic_fixture
from objgauss.core.v2_stability_foundation import (
    ObservationModelConfig,
    make_synthetic_stability_scenario_fixture,
    make_synthetic_stability_scenario_suite,
)
from objgauss.core.v2_stability_gate import (
    SyntheticStabilityGateReport,
    SyntheticStabilitySuiteGateReport,
    V2_STABILITY_GATE_HARD_CHECKS,
    V2_STABILITY_GATE_SCHEMA,
    V2_STABILITY_GATE_SUITE_SCHEMA,
    evaluate_synthetic_stability_gate,
    evaluate_synthetic_stability_suite_gate,
    validate_synthetic_stability_gate_summary,
    validate_synthetic_stability_suite_gate_summary,
)


def test_clean_synthetic_stability_suite_passes_identity_hard_gate():
    fixtures = make_synthetic_stability_scenario_suite(object_count=2, seed=100)
    predicted = tuple(expected_slots_for_synthetic_fixture(fixture) for fixture in fixtures)

    report = evaluate_synthetic_stability_suite_gate(
        fixtures,
        predicted_slots_by_fixture=predicted,
    )
    payload = report.as_dict()

    assert isinstance(report, SyntheticStabilitySuiteGateReport)
    assert payload["schema"] == V2_STABILITY_GATE_SUITE_SCHEMA
    assert payload["status"] == "synthetic_stability_suite_gate_pass"
    assert payload["fixture_count"] == 4
    assert payload["failed_count"] == 0
    assert payload["aggregate_failure_mode_counts"] == {
        "slot_swap": 0,
        "identity_fragmentation": 0,
        "object_merge": 0,
        "background_absorption": 0,
        "temporal_drift": 0,
    }
    for scenario in payload["scenario_statuses"]:
        assert scenario["status"] == "synthetic_stability_gate_pass"
        assert scenario["hard_blockers"] == []
    assert validate_synthetic_stability_suite_gate_summary(payload) is payload


def test_gate_requires_explicit_candidate_predictions():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        seed=105,
    )

    with pytest.raises(ValueError, match="requires explicit predicted_slots"):
        evaluate_synthetic_stability_gate(fixture)

    with pytest.raises(ValueError, match="requires explicit predicted_slots_by_fixture"):
        evaluate_synthetic_stability_suite_gate((fixture,))


def test_gate_rejects_assignment_slot_count_mismatch():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        frame_count=2,
        seed=106,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=107),
    )
    bad_assignments = tuple(
        np.full((frame.evidence.evidence_count, 3), 1.0 / 3.0, dtype=np.float32)
        for frame in fixture.observations
    )

    with pytest.raises(ValueError, match="columns must match fixture slot count"):
        evaluate_synthetic_stability_gate(
            fixture,
            predicted_assignments=bad_assignments,
        )


def test_adversarial_swap_prediction_fails_identity_hard_gate_with_diagnostics():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="adversarial_swap",
        object_count=2,
        frame_count=2,
        feature_dim=3,
        seed=110,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=111),
    )
    predicted = expected_slots_for_synthetic_fixture(fixture)
    predicted = (
        predicted[0],
        np.asarray([1, 0], dtype=np.int64),
    )

    payload = evaluate_synthetic_stability_gate(
        fixture,
        predicted_slots=predicted,
    ).as_dict()

    assert payload["schema"] == V2_STABILITY_GATE_SCHEMA
    assert payload["status"] == "synthetic_stability_gate_fail"
    assert payload["gate_role"] == "identity_invariant_hard_gate"
    assert payload["hard_gates"]["expected_slot_consistency_pass"] is False
    assert payload["hard_gates"]["no_slot_swap_pass"] is False
    assert payload["hard_gates"]["adversarial_swap_no_exchange_pass"] is False
    assert "expected_slot_mismatch" in payload["hard_blockers"]
    assert "slot_swap" in payload["hard_blockers"]
    assert payload["failure_mode_counts"]["slot_swap"] == 1
    assert payload["diagnostics"]["diagnostic_role"] == "diagnostic_only_not_gate"
    assert validate_synthetic_stability_gate_summary(payload) is payload


def test_occlusion_recovery_must_return_to_expected_slot():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="occlusion_recovery",
        object_count=2,
        frame_count=4,
        feature_dim=4,
        seed=120,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=121),
    )
    predicted = list(expected_slots_for_synthetic_fixture(fixture))
    predicted[-1] = np.asarray([2, 1], dtype=np.int64)

    payload = evaluate_synthetic_stability_gate(
        fixture,
        predicted_slots=tuple(predicted),
    ).as_dict()

    assert payload["status"] == "synthetic_stability_gate_fail"
    assert payload["hard_gates"]["occlusion_recovery_return_pass"] is False
    assert payload["hard_gates"]["expected_slot_consistency_pass"] is False
    assert "occlusion_recovery_not_returned_to_expected_slot" in payload["hard_blockers"]
    assert payload["soft_diagnostics"]["temporal_coherence"]["participates_in_hard_gate"] is False
    occlusion_checks = payload["soft_diagnostics"]["temporal_coherence"]
    assert occlusion_checks["status"] in {"soft_pass", "soft_warn"}


def test_soft_entropy_warning_does_not_fail_identity_hard_gate():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        frame_count=2,
        feature_dim=4,
        seed=130,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=131),
    )
    assignments = tuple(_soft_correct_assignments(frame.expected_slots) for frame in fixture.observations)

    payload = evaluate_synthetic_stability_gate(
        fixture,
        predicted_assignments=assignments,
        assignment_entropy_warn_threshold=0.6,
    ).as_dict()

    assert isinstance(
        evaluate_synthetic_stability_gate(fixture, predicted_assignments=assignments),
        SyntheticStabilityGateReport,
    )
    assert payload["status"] == "synthetic_stability_gate_pass"
    assert all(payload["hard_gates"][key] for key in V2_STABILITY_GATE_HARD_CHECKS)
    entropy = payload["soft_diagnostics"]["assignment_entropy"]
    assert entropy["source"] == "predicted_assignments"
    assert entropy["status"] == "soft_warn"
    assert entropy["mean_normalized_entropy"] > 0.9
    assert entropy["participates_in_hard_gate"] is False
    assert payload["soft_diagnostics"]["assignment_purity"]["status"] == "soft_pass"


def test_gate_rejects_bad_soft_thresholds():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        seed=140,
    )

    with pytest.raises(ValueError, match="assignment_entropy_warn_threshold"):
        evaluate_synthetic_stability_gate(
            fixture,
            predicted_slots=expected_slots_for_synthetic_fixture(fixture),
            assignment_entropy_warn_threshold=1.5,
        )


def _soft_correct_assignments(expected_slots: np.ndarray) -> np.ndarray:
    assignments = np.zeros((expected_slots.shape[0], 2), dtype=np.float32)
    for index, expected_slot in enumerate(expected_slots.astype(int).tolist()):
        if expected_slot == 0:
            assignments[index] = [0.51, 0.49]
        else:
            assignments[index] = [0.49, 0.51]
    return assignments
