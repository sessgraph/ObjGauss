from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.assignment_evidence import AssignmentEvidenceBatch
from objgauss.core.v2_stability_foundation import (
    ObjectIdentityObservation,
    ObjectIdentityOracle,
    ObjectIdentityRecord,
    ObservationModelConfig,
    SyntheticStabilityScenarioFixture,
    SyntheticWorldState,
    V2_STABILITY_FOUNDATION_SCHEMA,
    V2_STABILITY_SCENARIO_FIXTURE_SCHEMA,
    V2_STABILITY_SCENARIO_KINDS,
    V2_SYNTHETIC_OBSERVATION_SCHEMA,
    make_object_identity_oracle,
    make_synthetic_stability_scenario_fixture,
    make_synthetic_stability_scenario_suite,
    make_synthetic_world_state,
    observe_synthetic_world,
    validate_object_identity_oracle,
    validate_synthetic_observation_frame,
    validate_synthetic_stability_scenario_fixture,
    validate_synthetic_world_state,
)


def test_object_identity_oracle_freezes_lineage_visibility_and_slots():
    oracle = make_object_identity_oracle(
        scenario_id="fixture-identity",
        object_count=3,
        frame_count=4,
        occluded_object_ids=[1],
        occluded_frame_indices=[1, 2],
    )
    payload = oracle.as_dict()

    assert payload["schema"] == V2_STABILITY_FOUNDATION_SCHEMA
    assert payload["kind"] == "object_identity_oracle"
    assert payload["identity_source"] == "synthetic_oracle_labels"
    assert payload["object_count"] == 3
    assert payload["frame_count"] == 4
    assert payload["slots"] == 3
    assert payload["has_occlusion"] is True
    assert payload["identities"][1]["oracle_object_id"] == 1
    assert payload["identities"][1]["lineage_id"] == "lineage-0001"
    assert payload["identities"][1]["canonical_slot"] == 1
    assert payload["frames"][1][1]["visible"] is False
    assert payload["frames"][1][1]["expected_slot_relation"] == "occluded"
    assert payload["frames"][3][1]["visible"] is True
    assert payload["frames"][3][1]["expected_slot"] == 1
    assert validate_object_identity_oracle(oracle) == oracle


def test_synthetic_world_state_is_object_level_before_observation():
    world = make_synthetic_world_state(
        scenario_id="fixture-world",
        scenario_kind="occlusion_recovery",
        object_count=2,
        frame_count=3,
        feature_dim=4,
        seed=7,
    )
    payload = world.as_dict()

    assert isinstance(world, SyntheticWorldState)
    assert payload["schema"] == V2_STABILITY_FOUNDATION_SCHEMA
    assert payload["kind"] == "synthetic_world_state"
    assert payload["scenario_kind"] == "occlusion_recovery"
    assert payload["identity_contract"]["object_id_field"] == "oracle_object_id"
    assert payload["identity_contract"]["lineage_field"] == "lineage_id"
    assert payload["frames"][1]["objects"][0]["visible"] is False
    assert payload["frames"][2]["objects"][0]["visible"] is True
    assert payload["frames"][2]["objects"][0]["lineage_id"] == "lineage-0000"
    assert validate_synthetic_world_state(world) == world


def test_observation_model_projects_world_to_assignment_evidence_with_oracle_targets():
    world = make_synthetic_world_state(
        scenario_id="fixture-observation",
        scenario_kind="cross_view",
        object_count=2,
        frame_count=2,
        feature_dim=4,
        seed=2,
    )
    observations = observe_synthetic_world(
        world,
        config=ObservationModelConfig(points_per_object=2, position_jitter=0.02, seed=11),
    )

    assert len(observations) == 2
    first = validate_synthetic_observation_frame(observations[0])
    second = validate_synthetic_observation_frame(observations[1])
    assert first.schema == V2_SYNTHETIC_OBSERVATION_SCHEMA
    assert isinstance(first.evidence, AssignmentEvidenceBatch)
    assert first.evidence.evidence_count == 4
    assert first.evidence.mask_votes is not None
    assert first.evidence.track_hints is not None
    assert first.oracle_object_ids.tolist() == [0, 0, 1, 1]
    assert first.expected_slots.tolist() == [0, 0, 1, 1]
    np.testing.assert_allclose(first.evidence.target_assignment, first.evidence.mask_votes)
    np.testing.assert_array_equal(
        np.argmax(first.evidence.target_assignment, axis=1),
        first.expected_slots,
    )
    assert second.view_id == "camera-view-1"
    assert second.evidence.positions[:, 1].mean() > first.evidence.positions[:, 1].mean()


def test_occlusion_observation_keeps_oracle_identity_but_drops_invisible_evidence():
    world = make_synthetic_world_state(
        scenario_id="fixture-occlusion",
        scenario_kind="occlusion_recovery",
        object_count=2,
        frame_count=3,
        feature_dim=4,
        seed=5,
    )
    observations = observe_synthetic_world(
        world,
        config=ObservationModelConfig(points_per_object=2, seed=13),
    )

    assert world.oracle.frames[1][0].visible is False
    assert world.oracle.frames[2][0].visible is True
    assert observations[1].oracle_object_ids.tolist() == [1, 1]
    assert observations[1].expected_slots.tolist() == [1, 1]
    assert observations[2].oracle_object_ids.tolist() == [0, 0, 1, 1]
    assert observations[2].lineage_ids[0] == "lineage-0000"


def test_synthetic_stability_scenario_suite_covers_required_fixture_contracts():
    fixtures = make_synthetic_stability_scenario_suite(object_count=3, seed=21)

    assert tuple(fixture.scenario_kind for fixture in fixtures) == V2_STABILITY_SCENARIO_KINDS
    for fixture in fixtures:
        checked = validate_synthetic_stability_scenario_fixture(fixture)
        payload = checked.as_dict()

        assert isinstance(checked, SyntheticStabilityScenarioFixture)
        assert payload["schema"] == V2_STABILITY_SCENARIO_FIXTURE_SCHEMA
        assert payload["kind"] == "synthetic_stability_scenario_fixture"
        assert payload["scenario_role"] == "fixture_only_not_final_gate"
        assert payload["identity_source"] == "synthetic_oracle_labels"
        assert payload["expected_slot_source"] == "canonical_oracle_slot"
        assert len(payload["oracle"]["identities"]) == 3
        assert len(payload["visibility_transitions"]) == 3
        assert len(checked.observations) == checked.world.frame_count
        assert [identity.label for identity in checked.world.oracle.identities] == [
            "synthetic-object-0",
            "synthetic-object-1",
            "synthetic-object-2",
        ]
        for observation in checked.observations:
            assert observation.evidence.target_assignment is not None
            target_slots = np.argmax(observation.evidence.target_assignment, axis=1)
            np.testing.assert_array_equal(target_slots, observation.expected_slots)


def test_occlusion_recovery_fixture_records_visible_occluded_transitions():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="occlusion_recovery",
        object_count=3,
        seed=31,
    )
    payload = fixture.as_dict()
    object_zero = payload["visibility_transitions"][0]

    assert fixture.frame_count == 4
    assert [status["visible"] for status in object_zero["statuses"]] == [
        True,
        False,
        False,
        True,
    ]
    assert [item["transition"] for item in object_zero["transitions"]] == [
        "visible_to_occluded",
        "occluded_to_occluded",
        "occluded_to_visible",
    ]
    assert fixture.world.oracle.frames[1][0].expected_slot_relation == "occluded"
    assert 0 not in fixture.observations[1].oracle_object_ids.tolist()
    assert 0 in fixture.observations[-1].oracle_object_ids.tolist()
    assert fixture.observations[-1].expected_slots.tolist().count(0) == (
        fixture.observation_config.points_per_object
    )


def test_cross_view_and_perturbation_fixtures_preserve_slots_under_observation_changes():
    cross_view = make_synthetic_stability_scenario_fixture(
        scenario_kind="cross_view",
        object_count=2,
        seed=41,
        observation_config=ObservationModelConfig(points_per_object=2, position_jitter=0.0, seed=51),
    )
    perturbation = make_synthetic_stability_scenario_fixture(
        scenario_kind="perturbation",
        object_count=2,
        seed=42,
        observation_config=ObservationModelConfig(points_per_object=2, position_jitter=0.0, seed=52),
    )

    assert [frame.view_id for frame in cross_view.world.frames] == [
        "camera-view-0",
        "camera-view-1",
        "camera-view-2",
    ]
    assert (
        cross_view.observations[2].evidence.positions[:, 1].mean()
        > cross_view.observations[0].evidence.positions[:, 1].mean()
    )
    np.testing.assert_array_equal(cross_view.observations[0].expected_slots, [0, 0, 1, 1])
    np.testing.assert_array_equal(cross_view.observations[2].expected_slots, [0, 0, 1, 1])

    first_feature = perturbation.world.frames[0].objects[0].appearance_feature
    later_feature = perturbation.world.frames[1].objects[0].appearance_feature
    assert perturbation.world.frames[1].objects[0].perturbation["kind"] == "appearance_noise"
    assert not np.allclose(first_feature, later_feature)
    np.testing.assert_array_equal(perturbation.observations[1].expected_slots, [0, 0, 1, 1])


def test_adversarial_swap_fixture_swaps_appearance_without_swapping_identity_slots():
    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="adversarial_swap",
        object_count=2,
        frame_count=2,
        feature_dim=4,
        seed=61,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=71),
    )
    first_frame = fixture.world.frames[0]
    swap_frame = fixture.world.frames[1]

    np.testing.assert_array_equal(first_frame.objects[0].appearance_feature, [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(first_frame.objects[1].appearance_feature, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(swap_frame.objects[0].appearance_feature, [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(swap_frame.objects[1].appearance_feature, [1.0, 0.0, 0.0, 0.0])
    assert swap_frame.objects[0].perturbation["kind"] == "appearance_swap"
    assert swap_frame.objects[0].perturbation["swapped_with"] == 1
    assert swap_frame.objects[0].lineage_id == "lineage-0000"
    assert swap_frame.objects[1].lineage_id == "lineage-0001"

    swapped_observation = fixture.observations[1]
    assert swapped_observation.oracle_object_ids.tolist() == [0, 1]
    assert swapped_observation.expected_slots.tolist() == [0, 1]
    np.testing.assert_array_equal(swapped_observation.evidence.features[0], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_array_equal(swapped_observation.evidence.features[1], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(
        np.argmax(swapped_observation.evidence.target_assignment, axis=1),
        [0, 1],
    )


def test_scenario_fixture_observation_batches_are_reproducible_from_seed_and_config():
    config = ObservationModelConfig(points_per_object=3, position_jitter=0.015, seed=81)
    left = make_synthetic_stability_scenario_fixture(
        scenario_kind="perturbation",
        object_count=3,
        seed=82,
        observation_config=config,
    )
    right = make_synthetic_stability_scenario_fixture(
        scenario_kind="perturbation",
        object_count=3,
        seed=82,
        observation_config=config,
    )

    assert left.as_dict() == right.as_dict()
    for left_frame, right_frame in zip(left.observations, right.observations):
        np.testing.assert_allclose(left_frame.evidence.positions, right_frame.evidence.positions)
        np.testing.assert_allclose(left_frame.evidence.features, right_frame.evidence.features)
        np.testing.assert_array_equal(left_frame.oracle_object_ids, right_frame.oracle_object_ids)
        np.testing.assert_array_equal(left_frame.expected_slots, right_frame.expected_slots)


def test_identity_oracle_rejects_slot_and_visibility_contract_violations():
    identities = (
        ObjectIdentityRecord(oracle_object_id=0, lineage_id="lineage-0000", canonical_slot=0, label="a"),
        ObjectIdentityRecord(oracle_object_id=1, lineage_id="lineage-0001", canonical_slot=0, label="b"),
    )
    bad_slots = ObjectIdentityOracle(
        scenario_id="bad-slots",
        identities=identities,
        frames=(
            (
                ObjectIdentityObservation(0, "lineage-0000", 0, True, 0),
                ObjectIdentityObservation(1, "lineage-0001", 0, True, 0),
            ),
        ),
    )
    with pytest.raises(ValueError, match="canonical slots"):
        validate_object_identity_oracle(bad_slots)

    good_identity = (
        ObjectIdentityRecord(oracle_object_id=0, lineage_id="lineage-0000", canonical_slot=0, label="a"),
    )
    bad_visibility = ObjectIdentityOracle(
        scenario_id="bad-visibility",
        identities=good_identity,
        frames=(
            (
                ObjectIdentityObservation(
                    oracle_object_id=0,
                    lineage_id="lineage-0000",
                    frame_index=0,
                    visible=False,
                    expected_slot=0,
                    expected_slot_relation="same_lineage",
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="occluded observation"):
        validate_object_identity_oracle(bad_visibility)
