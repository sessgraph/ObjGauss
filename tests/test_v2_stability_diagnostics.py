from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.v2_stability_diagnostics import (
    FailureModeClassifier,
    SyntheticStabilityDiagnosticsReport,
    V2_STABILITY_DIAGNOSTICS_SCHEMA,
    V2_STABILITY_FAILURE_MODES,
    diagnose_synthetic_stability_fixture,
    expected_slots_for_synthetic_fixture,
    validate_synthetic_stability_diagnostics_summary,
)
from objgauss.core.v2_stability_foundation import (
    ObservationModelConfig,
    make_synthetic_stability_scenario_fixture,
)


def test_diagnostics_clean_fixture_reports_transition_matrix_and_confusion_graph():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=3,
        frame_count=3,
        seed=10,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=11),
    )
    report = diagnose_synthetic_stability_fixture(
        fixture,
        predicted_slots=expected_slots_for_synthetic_fixture(fixture),
    )
    payload = report.as_dict()

    assert isinstance(report, SyntheticStabilityDiagnosticsReport)
    assert payload["schema"] == V2_STABILITY_DIAGNOSTICS_SCHEMA
    assert payload["status"] == "stability_diagnostics_clean"
    assert payload["diagnostic_role"] == "diagnostic_only_not_gate"
    assert payload["classifier"]["failure_modes"] == list(V2_STABILITY_FAILURE_MODES)
    assert payload["failure_mode_counts"] == {mode: 0 for mode in V2_STABILITY_FAILURE_MODES}
    assert payload["slot_transition_labels"] == [0, 1, 2, -1]
    assert payload["slot_transition_matrix"][0][0] == 2
    assert payload["slot_transition_matrix"][1][1] == 2
    assert payload["slot_transition_matrix"][2][2] == 2
    assert payload["identity_confusion_graph"]["edge_count"] == 9
    assert all(edge["matches_expected"] for edge in payload["identity_confusion_graph"]["edges"])
    assert payload["non_goals"] == {
        "trains_solver": False,
        "uses_renderer_loss": False,
        "acts_as_gate": False,
    }
    assert validate_synthetic_stability_diagnostics_summary(payload) is payload


def test_diagnostics_requires_explicit_candidate_predictions():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        seed=15,
    )

    with pytest.raises(ValueError, match="require explicit predicted_slots"):
        diagnose_synthetic_stability_fixture(fixture)


def test_slot_swap_is_detected_without_changing_oracle_identity():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="adversarial_swap",
        object_count=2,
        frame_count=2,
        feature_dim=3,
        seed=20,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=21),
    )
    predicted = expected_slots_for_synthetic_fixture(fixture)
    predicted = (
        predicted[0],
        np.asarray([1, 0], dtype=np.int64),
    )

    payload = diagnose_synthetic_stability_fixture(fixture, predicted_slots=predicted).as_dict()

    assert payload["status"] == "stability_diagnostics_failures_detected"
    assert payload["failure_mode_counts"]["slot_swap"] == 1
    swap = [event for event in payload["failure_modes"] if event["mode"] == "slot_swap"][0]
    assert swap["frame_index"] == 1
    assert swap["oracle_object_ids"] == [0, 1]
    assert swap["expected_slots"] == [0, 1]
    assert swap["predicted_slots"] == [0, 1]
    assert payload["identity_confusion_graph"]["edges"][2]["oracle_object_id"] == 0
    assert payload["identity_confusion_graph"]["edges"][2]["expected_slot"] == 0
    assert payload["identity_confusion_graph"]["edges"][2]["predicted_slot"] == 1


def test_fragmentation_and_temporal_drift_are_distinguished_from_merge():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="perturbation",
        object_count=2,
        frame_count=3,
        feature_dim=4,
        seed=30,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=31),
    )
    predicted = (
        np.asarray([0, 1], dtype=np.int64),
        np.asarray([2, 1], dtype=np.int64),
        np.asarray([2, 1], dtype=np.int64),
    )

    payload = diagnose_synthetic_stability_fixture(fixture, predicted_slots=predicted).as_dict()

    assert payload["failure_mode_counts"]["identity_fragmentation"] == 1
    assert payload["failure_mode_counts"]["temporal_drift"] == 1
    assert payload["failure_mode_counts"]["object_merge"] == 0
    assert payload["slot_transition_labels"] == [0, 1, 2, -1]
    fragmentation = [
        event for event in payload["failure_modes"] if event["mode"] == "identity_fragmentation"
    ][0]
    assert fragmentation["oracle_object_ids"] == [0]
    assert fragmentation["predicted_slots"] == [0, 2]
    drift = [event for event in payload["failure_modes"] if event["mode"] == "temporal_drift"][0]
    assert drift["frame_index"] == 1
    assert drift["predicted_slots"] == [0, 2]


def test_object_merge_and_background_absorption_are_reported():
    merge_fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=3,
        frame_count=1,
        seed=40,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=41),
    )
    merge_payload = diagnose_synthetic_stability_fixture(
        merge_fixture,
        predicted_slots=(np.asarray([0, 0, 2], dtype=np.int64),),
    ).as_dict()

    assert merge_payload["failure_mode_counts"]["object_merge"] == 1
    merge = [event for event in merge_payload["failure_modes"] if event["mode"] == "object_merge"][0]
    assert merge["oracle_object_ids"] == [0, 1]
    assert merge["expected_slots"] == [0, 1]
    assert merge["predicted_slots"] == [0]

    background_fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        frame_count=1,
        seed=42,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=43),
    )
    background_payload = diagnose_synthetic_stability_fixture(
        background_fixture,
        predicted_slots=(np.asarray([-1, 1], dtype=np.int64),),
    ).as_dict()

    assert background_payload["failure_mode_counts"]["background_absorption"] == 1
    background = [
        event for event in background_payload["failure_modes"] if event["mode"] == "background_absorption"
    ][0]
    assert background["oracle_object_ids"] == [0]
    assert background["predicted_slots"] == [-1]


def test_low_confidence_assignment_is_background_absorption_diagnostic():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        frame_count=1,
        seed=50,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=51),
    )
    assignments = (
        np.asarray(
            [
                [0.55, 0.45],
                [0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )
    payload = diagnose_synthetic_stability_fixture(
        fixture,
        predicted_assignments=assignments,
        classifier=FailureModeClassifier(confidence_floor=0.75),
    ).as_dict()

    assert payload["failure_mode_counts"]["background_absorption"] == 1
    assert payload["identity_observations"][0]["mean_confidence"] == pytest.approx(0.55)
    assert payload["identity_observations"][0]["predicted_slot"] == 0


def test_diagnostics_reject_misaligned_prediction_rows():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        frame_count=1,
        seed=60,
        observation_config=ObservationModelConfig(points_per_object=2, position_jitter=0.0, seed=61),
    )

    with pytest.raises(ValueError, match="predicted slot rows"):
        diagnose_synthetic_stability_fixture(
            fixture,
            predicted_slots=(np.asarray([0, 1], dtype=np.int64),),
        )

    with pytest.raises(ValueError, match="provide predicted_slots or predicted_assignments"):
        diagnose_synthetic_stability_fixture(
            fixture,
            predicted_slots=expected_slots_for_synthetic_fixture(fixture),
            predicted_assignments=(np.ones((4, 2), dtype=np.float32),),
        )
