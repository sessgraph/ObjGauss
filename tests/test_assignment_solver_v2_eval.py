from __future__ import annotations

import numpy as np

from objgauss.core.assignment_solver_v2 import (
    ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA,
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
    assignment_solver_v2_checkpoint,
    assignment_solver_v2_state_from_checkpoint,
    train_assignment_solver_v2,
    validate_assignment_solver_v2_checkpoint,
)
from objgauss.evaluation.assignment_solver_v2_eval import (
    ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA,
    AssignmentSolverV2StabilityEvalReport,
    evaluate_assignment_solver_v2_stability,
    validate_assignment_solver_v2_stability_eval_summary,
)
from objgauss.datasets.v2_stability_foundation import (
    ObservationModelConfig,
    make_synthetic_stability_scenario_fixture,
)


def test_assignment_solver_v2_eval_requires_loss_and_identity_gate():
    fixtures = _two_object_stability_suite()
    batches = tuple(observation.evidence for fixture in fixtures for observation in fixture.observations)
    result = train_assignment_solver_v2(
        batches,
        initial_state=_swapped_two_slot_state(feature_dim=6),
        iterations=120,
        learning_rate=0.4,
        cluster_weight=0.0,
        entropy_weight=0.0,
        balance_weight=0.0,
        supervised_weight=1.0,
    )

    report = evaluate_assignment_solver_v2_stability(result, fixtures)
    payload = report.as_dict()

    assert isinstance(report, AssignmentSolverV2StabilityEvalReport)
    assert payload["schema"] == ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA
    assert payload["status"] == "assignment_solver_v2_stability_eval_pass"
    assert payload["training_loss"]["loss_decreased"] is True
    assert payload["hard_gate"]["before_status"] == "synthetic_stability_suite_gate_fail"
    assert payload["hard_gate"]["after_status"] == "synthetic_stability_suite_gate_pass"
    assert payload["hard_gate"]["loss_decrease_does_not_override_identity_gate"] is True
    assert "slot_swap" in payload["diagnostics_delta"]["before_hard_blockers"]
    assert payload["diagnostics_delta"]["after_hard_blockers"] == []
    assert payload["diagnostics_delta"]["before_failure_mode_counts"]["slot_swap"] > 0
    assert payload["diagnostics_delta"]["after_failure_mode_counts"]["slot_swap"] == 0
    assert payload["checkpoint_roundtrip"]["pass"] is True
    assert payload["checkpoint"]["schema"] == ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA
    assert payload["non_goals"] == {
        "uses_renderer_loss": False,
        "uses_gpu": False,
        "uses_rollout_model": False,
        "uses_replay_buffer": False,
        "mutates_dynamic_k": False,
    }
    assert validate_assignment_solver_v2_stability_eval_summary(payload) is payload


def test_assignment_solver_v2_checkpoint_roundtrips_final_state():
    fixtures = _two_object_stability_suite()[:1]
    batches = tuple(observation.evidence for fixture in fixtures for observation in fixture.observations)
    result = train_assignment_solver_v2(
        batches,
        initial_state=_swapped_two_slot_state(feature_dim=6),
        iterations=40,
        learning_rate=0.4,
        cluster_weight=0.0,
        entropy_weight=0.0,
        balance_weight=0.0,
        supervised_weight=1.0,
    )

    checkpoint = assignment_solver_v2_checkpoint(result, source="fixture://assignment-solver-v2-eval")
    restored = assignment_solver_v2_state_from_checkpoint(checkpoint)

    assert validate_assignment_solver_v2_checkpoint(checkpoint) is checkpoint
    np.testing.assert_allclose(restored.feature_centers, result.final_state.feature_centers, atol=1e-5)
    np.testing.assert_allclose(restored.position_centers, result.final_state.position_centers, atol=1e-5)
    np.testing.assert_allclose(restored.slot_bias, result.final_state.slot_bias, atol=1e-5)
    assert restored.step == result.final_state.step


def _two_object_stability_suite():
    return tuple(
        make_synthetic_stability_scenario_fixture(
            scenario_kind=scenario_kind,
            object_count=2,
            feature_dim=6,
            seed=300 + index,
            observation_config=ObservationModelConfig(
                points_per_object=2,
                position_jitter=0.0,
                seed=310 + index,
            ),
        )
        for index, scenario_kind in enumerate(
            ("cross_view", "occlusion_recovery", "perturbation", "adversarial_swap")
        )
    )


def _swapped_two_slot_state(*, feature_dim: int) -> AssignmentSolverV2State:
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(slots=2, feature_dim=feature_dim, temperature=0.7),
        feature_centers=np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        position_centers=np.asarray(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        slot_bias=np.zeros(2, dtype=np.float32),
        source="swapped_cost_softmax_assignment_v2_eval_fixture",
    )
