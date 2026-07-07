from __future__ import annotations

import pytest

from objgauss.core.objectstate_causal_gate import (
    OBJECTSTATE_ACTION_SCHEMA,
    OBJECTSTATE_CAUSAL_GATE_SCHEMA,
    ObjectStateCausalGateReport,
    evaluate_objectstate_causal_gate,
    validate_objectstate_causal_gate_summary,
)
from objgauss.core.v2_stability_foundation import (
    ObservationModelConfig,
    make_synthetic_stability_scenario_fixture,
)


def test_objectstate_causal_gate_passes_action_conditioned_counterfactuals():
    report = evaluate_objectstate_causal_gate(_causal_suite(seed=900))
    payload = report.as_dict()

    assert isinstance(report, ObjectStateCausalGateReport)
    assert payload["schema"] == OBJECTSTATE_CAUSAL_GATE_SCHEMA
    assert payload["status"] == "objectstate_causal_gate_pass"
    assert payload["action_source"] == "synthetic_controlled_action_oracle"
    assert payload["metrics"]["action_conditioned_ade"] == pytest.approx(0.0, abs=1e-7)
    assert payload["metrics"]["counterfactual_outcome_accuracy"] == 1.0
    assert payload["metrics"]["wrong_direction_rate"] == 0.0
    assert payload["metrics"]["intervention_gain"] > 0.0
    assert payload["metrics"]["action_type_counts"]["push_left"] > 0
    assert payload["metrics"]["action_type_counts"]["push_right"] > 0
    assert payload["rows"][0]["action"]["schema"] == OBJECTSTATE_ACTION_SCHEMA
    assert payload["claim_policy"] == {
        "synthetic_controlled_actions_only": True,
        "does_not_claim_real_world_causality": True,
        "requires_action_conditioned_comparison": True,
        "controlled_real_rows_required_later": True,
    }
    assert validate_objectstate_causal_gate_summary(payload) is payload


def test_objectstate_causal_gate_fails_when_candidate_ignores_action():
    payload = evaluate_objectstate_causal_gate(
        _causal_suite(seed=920),
        candidate_action_scale=0.0,
    ).as_dict()

    assert payload["status"] == "objectstate_causal_gate_fail"
    assert payload["metrics"]["action_conditioned_ade"] > 0.0
    assert payload["metrics"]["counterfactual_outcome_accuracy"] < 1.0
    assert payload["hard_gates"]["action_conditioned_ade_pass"] is False
    assert payload["hard_gates"]["counterfactual_accuracy_pass"] is False
    assert "action_conditioned_ade_above_threshold" in payload["hard_blockers"]
    assert "counterfactual_accuracy_below_threshold" in payload["hard_blockers"]
    assert validate_objectstate_causal_gate_summary(payload) is payload


def _causal_suite(*, seed: int):
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
