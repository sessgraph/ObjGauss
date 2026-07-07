from __future__ import annotations

import pytest

from objgauss.core.objectstate_predictive_gate import (
    OBJECTSTATE_PREDICTIVE_GATE_SCHEMA,
    ObjectStatePredictiveGateReport,
    evaluate_objectstate_predictive_gate,
    validate_objectstate_predictive_gate_summary,
)
from objgauss.core.v2_stability_foundation import (
    ObservationModelConfig,
    make_synthetic_stability_scenario_fixture,
)


def test_objectstate_predictive_gate_passes_synthetic_velocity_state():
    report = evaluate_objectstate_predictive_gate(_predictive_suite(seed=800), horizon=1)
    payload = report.as_dict()

    assert isinstance(report, ObjectStatePredictiveGateReport)
    assert payload["schema"] == OBJECTSTATE_PREDICTIVE_GATE_SCHEMA
    assert payload["status"] == "objectstate_predictive_gate_pass"
    assert payload["velocity_source"] == "synthetic_world_object_trajectory"
    assert payload["metrics"]["state_ade"] == pytest.approx(0.0, abs=1e-7)
    assert payload["metrics"]["prediction_error_ratio"] == pytest.approx(0.0, abs=1e-6)
    assert payload["metrics"]["identity_consistency_rate"] == 1.0
    assert payload["hard_blockers"] == []
    assert payload["claim_policy"] == {
        "synthetic_smoke_only": True,
        "does_not_claim_real_world_markov_state": True,
        "requires_history_baseline_comparison": True,
        "counterfactual_required_later": True,
    }
    assert validate_objectstate_predictive_gate_summary(payload) is payload


def test_objectstate_predictive_gate_fails_without_velocity_state():
    payload = evaluate_objectstate_predictive_gate(
        _predictive_suite(seed=820),
        horizon=1,
        velocity_scale=0.0,
    ).as_dict()

    assert payload["status"] == "objectstate_predictive_gate_fail"
    assert payload["metrics"]["state_ade"] > 0.0
    assert payload["hard_gates"]["state_ade_pass"] is False
    assert "state_ade_above_threshold" in payload["hard_blockers"]
    assert validate_objectstate_predictive_gate_summary(payload) is payload


def _predictive_suite(*, seed: int):
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
            ("cross_view", "occlusion_recovery", "perturbation")
        )
    )
