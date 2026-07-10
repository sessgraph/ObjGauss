from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_field import field_from_labels
from objgauss.core.object_state import (
    DynamicKUpdatePlan,
    bind_object_states_to_artifact,
    dynamic_k_proposal_report,
    dynamic_k_update_plan,
    match_object_states,
    object_state_delivery_summary,
    object_state_projection_summary,
    object_state_stability_report,
    project_object_states,
    project_object_states_from_field,
    validate_assignment_matrix,
)


def test_project_object_states_from_field_pools_sparse_assignment():
    cloud = _cloud()
    field = field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0)
    features = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    projection = project_object_states_from_field(cloud, field, evidence_features=features)

    assert projection.derived_object_ids.tolist() == [0, 0, 1, 1]
    assert projection.slots == 2
    assert projection.evidence_count == 4

    first, second = projection.states
    assert first.id == 1_000_000
    assert first.slot == 0
    assert first.status == "active"
    assert first.diagnostics == ()
    assert first.slot_mass == pytest.approx(2.0)
    assert first.confidence == pytest.approx(1.0)
    assert first.mass_fraction == pytest.approx(0.5)
    assert first.normalized_assignment_entropy == pytest.approx(0.0, abs=1e-5)
    np.testing.assert_allclose(first.centroid, [-0.9, 0.05, 0.0], atol=1e-5)
    np.testing.assert_allclose(first.bbox, [-1.0, 0.0, 0.0, -0.8, 0.1, 0.0], atol=1e-5)
    np.testing.assert_allclose(first.feature, [1.0, 0.0], atol=1e-5)

    assert second.id == 1_000_001
    assert second.slot == 1
    assert second.status == "active"
    np.testing.assert_allclose(second.centroid, [0.9, -0.05, 0.0], atol=1e-5)
    np.testing.assert_allclose(second.bbox, [0.8, -0.1, 0.0, 1.0, 0.0, 0.0], atol=1e-5)
    np.testing.assert_allclose(second.feature, [0.0, 1.0], atol=1e-5)


def test_object_state_projection_summary_preserves_persistent_and_renderer_ids():
    projection = project_object_states_from_field(
        _cloud(),
        field_from_labels(
            np.array([0, 0, 1, 1], dtype=np.int32),
            slots=2,
            confidence=1.0,
        ),
        evidence_features=_features(),
    )

    summary = object_state_projection_summary(projection)

    assert summary["type"] == "ObjectStateProjection"
    assert summary["evidence_count"] == 4
    assert summary["object_state_count"] == 2
    assert summary["active_state_count"] == 2
    assert summary["derived_object_id_source"] == "argmax(A[N,K])"
    assert [state["id"] for state in summary["states"]] == [1_000_000, 1_000_001]
    assert [state["persistent_id"] for state in summary["states"]] == [
        1_000_000,
        1_000_001,
    ]
    assert [state["slot"] for state in summary["states"]] == [0, 1]
    assert [state["object_id"] for state in summary["states"]] == [0, 1]
    assert summary["states"][0]["centroid"] == pytest.approx([-0.9, 0.05, 0.0])


def test_stability_report_marks_sparse_assignment_healthy():
    field = field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0)
    projection = project_object_states_from_field(
        _cloud(),
        field,
        evidence_features=_features(),
    )

    report = object_state_stability_report(projection, purity_labels=np.array([0, 0, 1, 1]))

    assert report.evidence_count == 4
    assert report.slots == 2
    assert report.assignment_confidence == pytest.approx(1.0)
    assert report.mean_normalized_entropy == pytest.approx(0.0, abs=1e-5)
    assert report.effective_slots == pytest.approx(2.0)
    assert report.slot_mass == pytest.approx((2.0, 2.0))
    assert report.slot_mass_fraction == pytest.approx((0.5, 0.5))
    assert report.inactive_slots == ()
    assert report.low_confidence_slots == ()
    assert report.mixed_slots == ()
    assert report.slot_collapse is False
    assert report.dominant_slot in {0, 1}
    assert report.dominant_slot_mass_fraction == pytest.approx(0.5)
    assert report.object_purity == pytest.approx(1.0)
    assert report.per_slot_purity == pytest.approx((1.0, 1.0))
    assert report.diagnostics == ()


def test_assignment_matrix_must_be_row_normalized():
    validate_assignment_matrix(np.array([[0.25, 0.75], [1.0, 0.0]], dtype=np.float32))

    with pytest.raises(ValueError, match="rows must sum to 1"):
        validate_assignment_matrix(np.array([[1.0, 1.0]], dtype=np.float32))
    with pytest.raises(ValueError, match="non-negative"):
        validate_assignment_matrix(np.array([[1.1, -0.1]], dtype=np.float32))
    with pytest.raises(ValueError, match="finite"):
        validate_assignment_matrix(np.array([[np.nan, 1.0]], dtype=np.float32))


def test_uniform_assignment_marks_mixed_slots_and_global_bbox():
    assignment = np.full((4, 2), 0.5, dtype=np.float32)
    features = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    projection = project_object_states(_cloud(), assignment, evidence_features=features)

    assert projection.derived_object_ids.tolist() == [0, 0, 0, 0]
    for state in projection.states:
        assert state.status == "mixed"
        assert state.diagnostics == ("mixed_slot",)
        assert state.slot_mass == pytest.approx(2.0)
        assert state.confidence == pytest.approx(0.5)
        assert state.normalized_assignment_entropy == pytest.approx(1.0)
        np.testing.assert_allclose(state.centroid, [0.0, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(state.bbox, [-1.0, -0.1, 0.0, 1.0, 0.1, 0.0], atol=1e-6)
        np.testing.assert_allclose(state.feature, [0.5, 0.5], atol=1e-6)

    report = object_state_stability_report(projection, purity_labels=np.array([0, 0, 1, 1]))
    assert report.assignment_confidence == pytest.approx(0.0)
    assert report.effective_slots == pytest.approx(2.0)
    assert report.mixed_slots == (0, 1)
    assert report.slot_collapse is False
    assert report.object_purity == pytest.approx(0.5)
    assert report.per_slot_purity == pytest.approx((0.5, 0.5))
    assert report.diagnostics == (
        "low_assignment_confidence",
        "mixed_slots",
        "low_object_purity",
    )


def test_single_dominant_slot_keeps_low_confidence_slot_visible():
    assignment = np.array(
        [
            [0.97, 0.03],
            [0.97, 0.03],
            [0.97, 0.03],
            [0.97, 0.03],
        ],
        dtype=np.float32,
    )

    projection = project_object_states(_cloud(), assignment, evidence_features=_features())

    dominant, weak = projection.states
    assert projection.derived_object_ids.tolist() == [0, 0, 0, 0]
    assert dominant.status == "active"
    assert dominant.slot_mass == pytest.approx(3.88)
    assert dominant.confidence == pytest.approx(0.97)
    assert dominant.normalized_assignment_entropy < 0.2
    assert weak.status == "low_confidence"
    assert weak.diagnostics == ("low_confidence_slot",)
    assert weak.slot_mass == pytest.approx(0.12)
    assert weak.confidence == pytest.approx(0.03)
    assert weak.normalized_assignment_entropy < 0.2

    report = object_state_stability_report(projection)
    assert report.slot_collapse is True
    assert report.dominant_slot == 0
    assert report.dominant_slot_mass_fraction == pytest.approx(0.97)
    assert report.low_confidence_slots == (1,)
    assert report.inactive_slots == ()
    assert report.object_purity is None
    assert report.per_slot_purity == (None, None)
    assert report.diagnostics == ("low_confidence_slots", "slot_collapse:0")


def test_noisy_assignment_remains_deterministic_and_reports_entropy():
    assignment = np.array(
        [
            [0.60, 0.40],
            [0.55, 0.45],
            [0.45, 0.55],
            [0.40, 0.60],
        ],
        dtype=np.float32,
    )

    projection = project_object_states(_cloud(), assignment, evidence_features=_features())

    assert projection.derived_object_ids.tolist() == [0, 0, 1, 1]
    assert [state.status for state in projection.states] == ["mixed", "mixed"]
    assert [state.slot_mass for state in projection.states] == pytest.approx([2.0, 2.0])
    assert [state.confidence for state in projection.states] == pytest.approx([0.5125, 0.5125])
    assert all(state.normalized_assignment_entropy > 0.95 for state in projection.states)
    np.testing.assert_allclose(projection.states[0].centroid, [-0.14, 0.0075, 0.0], atol=1e-5)
    np.testing.assert_allclose(projection.states[1].centroid, [0.14, -0.0075, 0.0], atol=1e-5)

    report = object_state_stability_report(projection, purity_labels=np.array([0, 0, 1, 1]))
    assert report.assignment_confidence < 0.05
    assert report.mixed_slots == (0, 1)
    assert report.object_purity == pytest.approx(0.575)
    assert report.diagnostics == (
        "low_assignment_confidence",
        "mixed_slots",
        "low_object_purity",
    )


def test_empty_assignment_returns_inactive_states_without_crashing():
    assignment = np.empty((0, 2), dtype=np.float32)
    features = np.empty((0, 2), dtype=np.float32)

    projection = project_object_states(_empty_cloud(), assignment, evidence_features=features)

    assert projection.evidence_count == 0
    assert projection.derived_object_ids.tolist() == []
    assert len(projection.states) == 2
    for state in projection.states:
        assert state.status == "inactive"
        assert state.diagnostics == ("inactive_slot",)
        assert state.slot_mass == 0.0
        assert state.mass_fraction == 0.0
        assert state.assignment_entropy == 0.0
        assert np.isnan(state.centroid).all()
        assert np.isnan(state.bbox).all()
        assert np.isnan(state.feature).all()

    report = object_state_stability_report(projection, purity_labels=np.empty(0, dtype=np.int32))
    assert report.assignment_confidence == 0.0
    assert report.mean_normalized_entropy == 0.0
    assert report.effective_slots == 0.0
    assert report.slot_mass == pytest.approx((0.0, 0.0))
    assert report.slot_mass_fraction == pytest.approx((0.0, 0.0))
    assert report.inactive_slots == (0, 1)
    assert report.low_confidence_slots == ()
    assert report.mixed_slots == ()
    assert report.slot_collapse is False
    assert report.dominant_slot is None
    assert report.object_purity is None
    assert report.diagnostics == (
        "no_evidence",
        "low_assignment_confidence",
        "inactive_slots",
    )


def test_stability_report_validates_purity_labels():
    projection = project_object_states(_cloud(), np.full((4, 2), 0.5, dtype=np.float32))

    with pytest.raises(ValueError, match="matching the evidence count"):
        object_state_stability_report(projection, purity_labels=np.array([0, 1], dtype=np.int32))
    with pytest.raises(ValueError, match="non-negative"):
        object_state_stability_report(projection, purity_labels=np.array([0, 0, 1, -1], dtype=np.int32))


def test_temporal_matching_handles_slot_permutation_without_hard_id_equality():
    previous = project_object_states_from_field(
        _cloud(),
        field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0),
        evidence_features=_features(),
    )
    swapped_assignment = np.array(
        [
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
        ],
        dtype=np.float32,
    )
    current = project_object_states(
        _cloud(),
        swapped_assignment,
        evidence_features=_features(),
        previous_state=previous,
        persistent_match_max_cost=0.05,
    )

    report = match_object_states(previous, current, max_cost=0.05)

    assert [state.slot for state in current.states] == [0, 1]
    assert [state.id for state in current.states] == [
        previous.states[1].id,
        previous.states[0].id,
    ]
    assert {(match.previous_id, match.current_id) for match in report.matches} == {
        (previous.states[0].id, previous.states[0].id),
        (previous.states[1].id, previous.states[1].id),
    }
    assert report.unmatched_previous == ()
    assert report.unmatched_current == ()
    assert report.mean_temporal_drift == pytest.approx(0.0)
    assert report.max_temporal_drift == pytest.approx(0.0)
    assert report.diagnostics == ()


def test_temporal_matching_reports_unmatched_current_state_for_birth_policy():
    previous = project_object_states_from_field(
        _cloud(),
        field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0),
        evidence_features=_features(),
    )
    current = project_object_states_from_field(
        _three_object_cloud(),
        field_from_labels(np.array([0, 0, 1, 1, 2, 2], dtype=np.int32), slots=3, confidence=1.0),
        evidence_features=_three_features(),
        previous_state=previous,
        persistent_match_max_cost=0.05,
    )

    report = match_object_states(previous, current, max_cost=0.05)

    assert {(match.previous_id, match.current_id) for match in report.matches} == {
        (previous.states[0].id, previous.states[0].id),
        (previous.states[1].id, previous.states[1].id),
    }
    assert report.unmatched_previous == ()
    assert report.unmatched_current == (current.states[2].id,)
    assert "unmatched_current" in report.diagnostics


def test_object_state_delivery_summary_binds_gaussian_children_and_chunk_metadata():
    projection = project_object_states_from_field(
        _cloud(),
        field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0),
        evidence_features=_features(),
    )
    chunk_index = {
        "schema": "objgauss-chunk-index-v1",
        "objects": [
            {"object_id": 0, "chunk_ids": [0]},
            {"object_id": 1, "chunk_ids": [1]},
        ],
        "chunks": [
            {"chunk_id": 0, "object_id": 0},
            {"chunk_id": 1, "object_id": 1},
        ],
    }

    summary = object_state_delivery_summary(projection, chunk_index=chunk_index)

    assert summary["schema"] == "objgauss-object-state-delivery-binding-v1"
    assert summary["derived_object_id_source"] == "argmax_assignment"
    assert summary["object_id_coverage"] == {
        "field": "object_id",
        "mode": "derived_from_assignment_argmax",
        "has_object_ids": True,
        "object_count": 2,
    }
    assert [
        (
            entry["object_id"],
            entry["persistent_id"],
            entry["gaussian_count"],
            entry["status"],
        )
        for entry in summary["gaussian_children"]
    ] == [
        (0, projection.states[0].id, 2, "active"),
        (1, projection.states[1].id, 2, "active"),
    ]
    assert summary["active_object_ids"] == [0, 1]
    assert summary["active_persistent_ids"] == [
        projection.states[0].id,
        projection.states[1].id,
    ]
    assert summary["chunk_binding"]["compatible"] is True
    assert summary["chunk_binding"]["chunk_ids_by_object"] == {"0": [0], "1": [1]}

    artifact = {
        "role": "object_edit",
        "path": "public/samples/fixture.ply",
        "format": ".ply",
        "delivery_tier": "browser_edit",
        "browser_ready": True,
    }
    bound = bind_object_states_to_artifact(artifact, projection, chunk_index=chunk_index)

    assert bound["object_count"] == 2
    assert bound["object_id_coverage"]["mode"] == "derived_from_assignment_argmax"
    assert bound["object_state_summary"] == summary


def test_dynamic_k_proposals_emit_remove_split_merge_and_birth_without_mutating_state():
    empty_projection = project_object_states(
        _empty_cloud(),
        np.empty((0, 2), dtype=np.float32),
        evidence_features=np.empty((0, 2), dtype=np.float32),
    )
    empty_report = dynamic_k_proposal_report(empty_projection)
    assert [proposal.kind for proposal in empty_report.proposals] == [
        "remove_inactive",
        "remove_inactive",
    ]
    assert all(proposal.action == "proposal_only" for proposal in empty_report.proposals)

    mixed_projection = project_object_states(
        _cloud(),
        np.full((4, 2), 0.5, dtype=np.float32),
        evidence_features=_features(),
    )
    mixed_report = dynamic_k_proposal_report(mixed_projection)
    assert {proposal.kind for proposal in mixed_report.proposals} >= {"split_mixed"}
    assert {proposal.source_ids for proposal in mixed_report.proposals if proposal.kind == "split_mixed"} == {
        (0,),
        (1,),
    }

    duplicate_projection = project_object_states_from_field(
        _duplicate_cloud(),
        field_from_labels(np.array([0, 1, 2], dtype=np.int32), slots=3, confidence=1.0),
        evidence_features=_duplicate_features(),
    )
    duplicate_report = dynamic_k_proposal_report(
        duplicate_projection,
        duplicate_feature_distance_threshold=0.01,
        duplicate_centroid_distance_threshold=0.05,
    )
    assert any(
        proposal.kind == "merge_duplicate" and proposal.source_ids == (0, 1)
        for proposal in duplicate_report.proposals
    )

    previous = project_object_states_from_field(
        _cloud(),
        field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0),
        evidence_features=_features(),
    )
    current = project_object_states_from_field(
        _three_object_cloud(),
        field_from_labels(np.array([0, 0, 1, 1, 2, 2], dtype=np.int32), slots=3, confidence=1.0),
        evidence_features=_three_features(),
    )
    temporal = match_object_states(previous, current, max_cost=0.05)
    birth_report = dynamic_k_proposal_report(current, temporal_match=temporal)
    assert any(
        proposal.kind == "birth_unmatched" and proposal.source_ids == (2,)
        for proposal in birth_report.proposals
    )


def test_dynamic_k_update_plan_applies_gated_epoch_boundary_actions_without_mutating_projection():
    empty_projection = project_object_states(
        _empty_cloud(),
        np.empty((0, 2), dtype=np.float32),
        evidence_features=np.empty((0, 2), dtype=np.float32),
    )
    empty_plan = dynamic_k_update_plan(empty_projection, min_slot_count=1)
    assert isinstance(empty_plan, DynamicKUpdatePlan)
    assert empty_plan.schema == "objgauss-dynamic-k-update-plan-v1"
    assert empty_plan.current_slot_count == 2
    assert empty_plan.next_slot_count == 1
    assert empty_plan.accepted_count == 1
    assert empty_plan.blocked_count == 1
    assert [action.accepted for action in empty_plan.actions] == [True, False]
    assert empty_plan.actions[0].slot_delta == -1
    assert empty_plan.actions[1].reason == "min_slot_count would be violated"
    assert empty_projection.slots == 2

    mixed_projection = project_object_states(
        _cloud(),
        np.full((4, 2), 0.5, dtype=np.float32),
        evidence_features=_features(),
    )
    mixed_plan = dynamic_k_update_plan(mixed_projection, max_slot_count=3)
    split_actions = [action for action in mixed_plan.actions if action.kind == "split_mixed"]
    assert [action.accepted for action in split_actions] == [True, False]
    assert mixed_plan.next_slot_count == 3
    assert "slot_count_increase" in mixed_plan.diagnostics

    duplicate_projection = project_object_states_from_field(
        _duplicate_cloud(),
        field_from_labels(np.array([0, 1, 2], dtype=np.int32), slots=3, confidence=1.0),
        evidence_features=_duplicate_features(),
    )
    duplicate_report = dynamic_k_proposal_report(
        duplicate_projection,
        duplicate_feature_distance_threshold=0.01,
        duplicate_centroid_distance_threshold=0.05,
    )
    duplicate_plan = dynamic_k_update_plan(duplicate_projection, proposal_report=duplicate_report)
    merge_actions = [action for action in duplicate_plan.actions if action.kind == "merge_duplicate"]
    assert len(merge_actions) == 1
    assert merge_actions[0].accepted is True
    assert merge_actions[0].target_id in merge_actions[0].source_ids
    assert duplicate_plan.next_slot_count == 2

    previous = project_object_states_from_field(
        _cloud(),
        field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=1.0),
        evidence_features=_features(),
    )
    current = project_object_states_from_field(
        _three_object_cloud(),
        field_from_labels(np.array([0, 0, 1, 1, 2, 2], dtype=np.int32), slots=3, confidence=1.0),
        evidence_features=_three_features(),
    )
    temporal = match_object_states(previous, current, max_cost=0.05)
    birth_report = dynamic_k_proposal_report(current, temporal_match=temporal)
    birth_plan = dynamic_k_update_plan(current, proposal_report=birth_report)
    birth_actions = [action for action in birth_plan.actions if action.kind == "birth_unmatched"]
    assert len(birth_actions) == 1
    assert birth_actions[0].accepted is True
    assert birth_actions[0].slot_delta == 0
    assert birth_plan.next_slot_count == 3
    assert birth_plan.as_dict()["apply_at"] == "epoch_boundary"


def test_dynamic_k_update_plan_rejects_non_epoch_boundary_updates():
    projection = project_object_states(
        _cloud(),
        np.full((4, 2), 0.5, dtype=np.float32),
        evidence_features=_features(),
    )
    with pytest.raises(ValueError, match="epoch_boundary"):
        dynamic_k_update_plan(projection, apply_at="train_step")


def _cloud() -> GaussianCloud:
    vertices = _vertices(4)
    vertices["x"] = np.array([-1.0, -0.8, 0.8, 1.0], dtype=np.float32)
    vertices["y"] = np.array([0.0, 0.1, 0.0, -0.1], dtype=np.float32)
    vertices["z"] = np.zeros(4, dtype=np.float32)
    vertices["red"] = np.array([255, 245, 10, 20], dtype=np.uint8)
    vertices["green"] = np.array([10, 20, 245, 255], dtype=np.uint8)
    vertices["blue"] = np.array([0, 5, 10, 15], dtype=np.uint8)
    vertices["opacity"] = np.ones(4, dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="ascii")


def _empty_cloud() -> GaussianCloud:
    return GaussianCloud(vertices=_vertices(0), source_format="ascii")


def _three_object_cloud() -> GaussianCloud:
    vertices = _vertices(6)
    vertices["x"] = np.array([-1.0, -0.8, 0.8, 1.0, 4.0, 4.2], dtype=np.float32)
    vertices["y"] = np.array([0.0, 0.1, 0.0, -0.1, 0.0, 0.1], dtype=np.float32)
    vertices["z"] = np.zeros(6, dtype=np.float32)
    vertices["red"] = np.array([255, 245, 10, 20, 120, 130], dtype=np.uint8)
    vertices["green"] = np.array([10, 20, 245, 255, 120, 130], dtype=np.uint8)
    vertices["blue"] = np.array([0, 5, 10, 15, 240, 250], dtype=np.uint8)
    vertices["opacity"] = np.ones(6, dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="ascii")


def _duplicate_cloud() -> GaussianCloud:
    vertices = _vertices(3)
    vertices["x"] = np.array([0.0, 0.02, 1.0], dtype=np.float32)
    vertices["y"] = np.zeros(3, dtype=np.float32)
    vertices["z"] = np.zeros(3, dtype=np.float32)
    vertices["red"] = np.array([255, 254, 10], dtype=np.uint8)
    vertices["green"] = np.array([10, 11, 245], dtype=np.uint8)
    vertices["blue"] = np.array([0, 0, 10], dtype=np.uint8)
    vertices["opacity"] = np.ones(3, dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="ascii")


def _vertices(count: int) -> np.ndarray:
    return np.zeros(
        count,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("opacity", "f4"),
        ],
    )


def _features() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _three_features() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.1, 0.9],
            [0.0, 1.0],
            [0.5, 0.5],
            [0.55, 0.45],
        ],
        dtype=np.float32,
    )


def _duplicate_features() -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0],
            [0.999, 0.001],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
