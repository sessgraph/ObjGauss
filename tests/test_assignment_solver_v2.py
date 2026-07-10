from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.assignment_solver_v2 import (
    ASSIGNMENT_SOLVER_V2_COST_TERMS,
    ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA,
    ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
    ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA,
    AssignmentSolverV2Config,
    AssignmentSolverV2Prediction,
    AssignmentSolverV2State,
    AssignmentSolverV2TrainingResult,
    assignment_solver_v2_state_from_dict,
    initialize_assignment_solver_v2,
    predict_assignment_solver_v2,
    train_assignment_solver_v2,
    validate_assignment_solver_v2_config,
    validate_assignment_solver_v2_state,
    validate_assignment_solver_v2_training_summary,
)
from objgauss.datasets.v2_stability_foundation import (
    ObservationModelConfig,
    make_synthetic_stability_scenario_fixture,
)


def test_assignment_solver_v2_trains_cost_softmax_on_synthetic_fixture():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        frame_count=2,
        feature_dim=4,
        seed=201,
        observation_config=ObservationModelConfig(points_per_object=2, position_jitter=0.0, seed=202),
    )
    batches = tuple(observation.evidence for observation in fixture.observations)
    initial_state = _swapped_two_slot_state(feature_dim=4)

    result = train_assignment_solver_v2(
        batches,
        initial_state=initial_state,
        iterations=80,
        learning_rate=0.5,
        cluster_weight=0.0,
        entropy_weight=0.0,
        balance_weight=0.0,
        supervised_weight=1.0,
        record_every=40,
    )
    summary = result.as_dict(include_state_arrays=True, include_assignments=True, include_cost=True)

    assert isinstance(result, AssignmentSolverV2TrainingResult)
    assert summary["schema"] == ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA
    assert summary["solver_family"] == "cost-softmax-assignment-v2"
    assert summary["loss_decreased"] is True
    assert summary["supervised_loss_decreased"] is True
    assert result.final_loss.total_loss < 0.01
    assert result.final_loss.total_loss < result.initial_loss.total_loss
    assert result.final_state.step == 80
    assert summary["renderer_loss"] == "not_used"
    assert summary["dynamic_k"] == "disabled"
    assert summary["non_goals"] == {
        "uses_gpu": False,
        "uses_renderer_loss": False,
        "uses_temporal_matching_loss": False,
        "mutates_dynamic_k": False,
        "uses_slot_attention": False,
        "uses_sinkhorn_or_ot": False,
    }
    assert validate_assignment_solver_v2_training_summary(summary) is summary

    for prediction, observation in zip(result.predictions, fixture.observations, strict=True):
        assert isinstance(prediction, AssignmentSolverV2Prediction)
        assert prediction.schema == ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA
        assert prediction.assignment.shape == observation.evidence.target_assignment.shape
        assert prediction.assignment.argmax(axis=1).tolist() == observation.expected_slots.tolist()


def test_assignment_solver_v2_state_roundtrip_and_prediction_contract():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="perturbation",
        object_count=2,
        frame_count=1,
        feature_dim=4,
        seed=210,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=211),
    )
    batch = fixture.observations[0].evidence
    state = initialize_assignment_solver_v2(slots=2, feature_dim=4, seed=212, scale=0.1)
    payload = state.as_dict(include_arrays=True)
    restored = assignment_solver_v2_state_from_dict(payload)
    prediction = predict_assignment_solver_v2(batch, restored)
    pred_summary = prediction.as_dict(include_assignment=True, include_cost=True)

    assert state.schema == ASSIGNMENT_SOLVER_V2_STATE_SCHEMA
    assert validate_assignment_solver_v2_state(restored).schema == ASSIGNMENT_SOLVER_V2_STATE_SCHEMA
    np.testing.assert_allclose(restored.feature_centers, state.feature_centers, atol=1e-6)
    np.testing.assert_allclose(restored.position_centers, state.position_centers, atol=1e-6)
    np.testing.assert_allclose(restored.slot_bias, state.slot_bias, atol=1e-6)
    assert pred_summary["schema"] == ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA
    assert pred_summary["slots"] == 2
    assert pred_summary["evidence_count"] == batch.evidence_count
    assert len(pred_summary["assignment"]) == batch.evidence_count
    assert len(pred_summary["cost"]) == batch.evidence_count
    np.testing.assert_allclose(prediction.assignment.sum(axis=1), np.ones(batch.evidence_count), atol=1e-6)


def test_assignment_solver_v2_requires_targets_for_supervised_training():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        frame_count=1,
        seed=220,
    )
    batch = fixture.observations[0].evidence
    no_target = type(batch)(
        positions=batch.positions,
        features=batch.features,
        frame_index=batch.frame_index,
        mask_votes=batch.mask_votes,
        track_hints=batch.track_hints,
        target_assignment=None,
        source=batch.source,
    )

    with pytest.raises(ValueError, match="supervised_weight requires"):
        train_assignment_solver_v2(
            [no_target],
            slots=2,
            iterations=2,
            supervised_weight=1.0,
        )


def test_assignment_solver_v2_config_keeps_temporal_matching_and_ot_disabled():
    config = AssignmentSolverV2Config(slots=2, feature_dim=4)

    assert validate_assignment_solver_v2_config(config) is config
    assert config.cost_terms == ASSIGNMENT_SOLVER_V2_COST_TERMS
    assert config.balance_policy == "loss-only-v1"
    assert config.temporal_policy == "disabled"
    assert config.matching_policy == "disabled"

    with pytest.raises(ValueError, match="temporal_policy"):
        validate_assignment_solver_v2_config(
            AssignmentSolverV2Config(
                slots=2,
                feature_dim=4,
                temporal_policy="enabled",
            )
        )


def _swapped_two_slot_state(*, feature_dim: int) -> AssignmentSolverV2State:
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(slots=2, feature_dim=feature_dim, temperature=0.7),
        feature_centers=np.asarray(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
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
        source="swapped_cost_softmax_assignment_v2_fixture",
    )
