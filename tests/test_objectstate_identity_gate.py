from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.objectstate_identity_gate import (
    OBJECTSTATE_IDENTITY_DATASET_SCHEMA,
    OBJECTSTATE_IDENTITY_GATE_SCHEMA,
    ObjectStateIdentityGateReport,
    evaluate_objectstate_identity_gate,
    validate_objectstate_identity_gate_summary,
)
from objgauss.core.v2_stability_diagnostics import expected_slots_for_synthetic_fixture
from objgauss.core.v2_stability_foundation import (
    ObservationModelConfig,
    make_synthetic_stability_scenario_fixture,
)


def test_objectstate_identity_gate_passes_clean_candidate_identity_suite():
    fixtures = _identity_suite(seed=500)
    predicted = tuple(expected_slots_for_synthetic_fixture(fixture) for fixture in fixtures)

    report = evaluate_objectstate_identity_gate(
        fixtures,
        predicted_slots_by_fixture=predicted,
    )
    payload = report.as_dict()

    assert isinstance(report, ObjectStateIdentityGateReport)
    assert payload["schema"] == OBJECTSTATE_IDENTITY_GATE_SCHEMA
    assert payload["status"] == "objectstate_identity_gate_pass"
    assert payload["dataset"]["schema"] == OBJECTSTATE_IDENTITY_DATASET_SCHEMA
    assert payload["dataset"]["contract"]["inputs"] == [
        "ObjectInstance",
        "Observation",
        "Transformation",
        "GroundTruthIdentity",
    ]
    assert payload["metrics"]["id_accuracy"] == 1.0
    assert payload["metrics"]["idf1"] == 1.0
    assert payload["metrics"]["embedding_retrieval_recall_at_1"] == 1.0
    assert payload["metrics"]["long_term_drift_rate"] == 0.0
    assert payload["metrics"]["fragmentation_rate"] == 0.0
    assert payload["metrics"]["occlusion_recovery_rate"] == 1.0
    assert payload["metrics"]["contrastive_margin"] > 0.0
    assert payload["hard_blockers"] == []
    first = payload["dataset"]["rows"][0]
    assert first["object_instance"]["lineage_id"].startswith("lineage-")
    assert first["candidate_objectstate"]["predicted_identity"] == (
        first["ground_truth_identity"]["expected_identity"]
    )
    assert validate_objectstate_identity_gate_summary(payload) is payload


def test_objectstate_identity_gate_fails_swapped_candidate_identity():
    fixtures = _identity_suite(seed=520)
    predicted = [list(expected_slots_for_synthetic_fixture(fixture)) for fixture in fixtures]
    adversarial = predicted[-1]
    adversarial[-1] = np.asarray([1, 0], dtype=np.int64)

    payload = evaluate_objectstate_identity_gate(
        fixtures,
        predicted_slots_by_fixture=tuple(tuple(item) for item in predicted),
    ).as_dict()

    assert payload["status"] == "objectstate_identity_gate_fail"
    assert payload["metrics"]["id_accuracy"] < 1.0
    assert payload["metrics"]["idf1"] < 1.0
    assert payload["hard_gates"]["id_accuracy_pass"] is False
    assert payload["hard_gates"]["idf1_pass"] is False
    assert "expected_identity_mismatch" in payload["hard_blockers"]
    assert "id_accuracy_below_threshold" in payload["hard_blockers"]
    assert validate_objectstate_identity_gate_summary(payload) is payload


def test_objectstate_identity_gate_requires_candidate_predictions():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        seed=540,
    )

    with pytest.raises(ValueError, match="requires explicit predicted_slots_by_fixture"):
        evaluate_objectstate_identity_gate((fixture,))


def test_objectstate_identity_gate_rejects_assignment_slot_count_mismatch():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        seed=550,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=551),
    )
    assignments = tuple(
        np.full((frame.evidence.evidence_count, 3), 1.0 / 3.0, dtype=np.float32)
        for frame in fixture.observations
    )

    with pytest.raises(ValueError, match="columns must match fixture slot count"):
        evaluate_objectstate_identity_gate(
            (fixture,),
            predicted_assignments_by_fixture=(assignments,),
        )


def _identity_suite(*, seed: int):
    return tuple(
        make_synthetic_stability_scenario_fixture(
            scenario_kind=scenario_kind,
            object_count=2,
            feature_dim=4,
            seed=seed + index,
            observation_config=ObservationModelConfig(
                points_per_object=1,
                position_jitter=0.0,
                seed=seed + 10 + index,
            ),
        )
        for index, scenario_kind in enumerate(
            ("cross_view", "occlusion_recovery", "perturbation", "adversarial_swap")
        )
    )
