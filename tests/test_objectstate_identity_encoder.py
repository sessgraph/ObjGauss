from __future__ import annotations

import pytest

from objgauss.pipelines.objectstate_identity_encoder import (
    OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA,
    OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA,
    ObjectStateIdentityEncoderConfig,
    ObjectStateIdentityEncoderTrainingResult,
    objectstate_identity_encoder_features,
    train_objectstate_identity_encoder,
    validate_objectstate_identity_encoder_training_summary,
)
from objgauss.evaluation.objectstate_identity_gate import evaluate_objectstate_identity_gate
from objgauss.evaluation.v2_stability_diagnostics import expected_slots_for_synthetic_fixture
from objgauss.datasets.v2_stability_foundation import (
    ObservationModelConfig,
    make_synthetic_stability_scenario_fixture,
)


def test_contrastive_identity_encoder_training_decreases_loss():
    gate = evaluate_objectstate_identity_gate(
        _identity_suite(seed=700),
        predicted_slots_by_fixture=_identity_predictions(seed=700),
    )
    rows = gate.rows
    features = objectstate_identity_encoder_features(rows, feature_source="appearance")
    config = ObjectStateIdentityEncoderConfig(
        input_dim=features.shape[1],
        embedding_dim=2,
        margin=0.75,
        learning_rate=0.4,
        weight_decay=0.0,
        feature_source="appearance",
        seed=7,
    )

    result = train_objectstate_identity_encoder(rows, config=config, iterations=80)
    payload = result.as_dict()

    assert isinstance(result, ObjectStateIdentityEncoderTrainingResult)
    assert payload["schema"] == OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA
    assert payload["status"] == "objectstate_identity_encoder_training_pass"
    assert payload["initial_state"]["config"]["schema"] == OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA
    assert payload["loss"]["loss_decreased"] is True
    assert payload["loss"]["negative_loss_decreased"] is True
    assert payload["loss"]["final"]["total_loss"] < payload["loss"]["initial"]["total_loss"]
    assert payload["loss"]["final"]["positive_loss"] < 0.01
    assert payload["loss"]["final"]["active_negative_pair_count"] <= (
        payload["loss"]["initial"]["active_negative_pair_count"]
    )
    assert payload["retrieval"]["final"]["recall_at_1"] >= payload["retrieval"]["initial"]["recall_at_1"]
    assert payload["non_goals"] == {
        "uses_identity_graph": False,
        "uses_replay_buffer": False,
        "uses_diffusion": False,
        "uses_renderer_loss": False,
        "mutates_viewer_defaults": False,
    }
    assert validate_objectstate_identity_encoder_training_summary(payload) is payload


def test_identity_encoder_rejects_too_small_dataset():
    gate = evaluate_objectstate_identity_gate(
        _identity_suite(seed=720)[:1],
        predicted_slots_by_fixture=_identity_predictions(seed=720)[:1],
    )
    single_identity_rows = tuple(row for row in gate.rows if row.oracle_object_id == 0)

    with pytest.raises(ValueError, match="at least two identities"):
        train_objectstate_identity_encoder(single_identity_rows)


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


def _identity_predictions(*, seed: int):
    return tuple(expected_slots_for_synthetic_fixture(fixture) for fixture in _identity_suite(seed=seed))
