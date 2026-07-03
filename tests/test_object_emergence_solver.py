from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_emergence_solver import (
    OBJECT_EMERGENCE_ASSIGNMENT_SCHEMA,
    ObjectEmergenceEvidence,
    ObjectEmergenceSolverConfig,
    ObjectEmergenceSolverState,
    evidence_from_gaussian_cloud,
    initialize_object_emergence_solver,
    predict_object_emergence_assignment,
    project_object_emergence_prediction,
    validate_object_emergence_evidence,
)


def test_linear_solver_predicts_normalized_assignment():
    evidence = ObjectEmergenceEvidence(
        positions=np.array(
            [
                [-1.0, 0.0, 0.0],
                [-0.8, 0.1, 0.0],
                [0.9, 0.0, 0.0],
                [1.1, -0.1, 0.0],
            ],
            dtype=np.float32,
        ),
        features=np.array(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.1, 0.9],
            ],
            dtype=np.float32,
        ),
        frame_index=2,
        source="unit-test",
    )
    state = ObjectEmergenceSolverState(
        config=ObjectEmergenceSolverConfig(slots=2, feature_dim=2, temperature=0.5),
        feature_weights=np.array([[4.0, -4.0], [-4.0, 4.0]], dtype=np.float32),
        position_weights=np.zeros((3, 2), dtype=np.float32),
        bias=np.zeros(2, dtype=np.float32),
        step=7,
        source="test_weights",
    )

    prediction = predict_object_emergence_assignment(evidence, state)

    assert prediction.schema == OBJECT_EMERGENCE_ASSIGNMENT_SCHEMA
    assert prediction.assignment.shape == (4, 2)
    np.testing.assert_allclose(prediction.assignment.sum(axis=1), np.ones(4), atol=1e-6)
    assert prediction.top_slots.tolist() == [0, 0, 1, 1]
    assert prediction.confidence.min() > 0.95
    assert prediction.slot_mass.tolist() == pytest.approx([2.0, 2.0], abs=0.02)
    assert prediction.mean_normalized_entropy < 0.1
    assert prediction.diagnostics == ("ok",)
    summary = prediction.as_dict(include_assignment=True)
    assert summary["frame_index"] == 2
    assert summary["solver_step"] == 7
    assert len(summary["assignment"]) == 4


def test_gaussian_cloud_evidence_projects_prediction_to_object_state():
    cloud = _object_cloud()
    evidence = evidence_from_gaussian_cloud(cloud, source="fixture-cloud")
    state = initialize_object_emergence_solver(
        slots=2,
        feature_dim=evidence.feature_dim,
        seed=1,
        scale=0.0,
    )
    state = ObjectEmergenceSolverState(
        config=state.config,
        feature_weights=np.zeros_like(state.feature_weights),
        position_weights=np.array([[-4.0, 4.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        bias=np.zeros(2, dtype=np.float32),
        source="x_axis_split",
    )

    prediction = predict_object_emergence_assignment(evidence, state)
    projection = project_object_emergence_prediction(
        cloud,
        prediction,
        evidence_features=evidence.features,
    )

    assert evidence.source == "fixture-cloud"
    assert evidence.evidence_count == cloud.count
    assert evidence.feature_dim == 7
    assert projection.assignment.shape == (4, 2)
    assert len(projection.states) == 2
    assert projection.states[0].status == "active"
    assert projection.states[1].status == "active"
    assert projection.states[0].centroid[0] < 0
    assert projection.states[1].centroid[0] > 0


def test_solver_abi_validates_shapes_and_targets():
    evidence = ObjectEmergenceEvidence(
        positions=np.zeros((2, 3), dtype=np.float32),
        features=np.zeros((3, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="features rows must match positions"):
        validate_object_emergence_evidence(evidence)

    bad_target = ObjectEmergenceEvidence(
        positions=np.zeros((2, 3), dtype=np.float32),
        features=np.zeros((2, 2), dtype=np.float32),
        target_assignment=np.array([[1.0, 0.0], [0.5, 0.4]], dtype=np.float32),
    )
    with pytest.raises(ValueError, match="assignment rows must sum to 1"):
        validate_object_emergence_evidence(bad_target)

    good = ObjectEmergenceEvidence(
        positions=np.zeros((2, 3), dtype=np.float32),
        features=np.zeros((2, 2), dtype=np.float32),
    )
    state = initialize_object_emergence_solver(slots=2, feature_dim=3, seed=0)
    with pytest.raises(ValueError, match="feature_dim 2 does not match solver feature_dim 3"):
        predict_object_emergence_assignment(good, state)


def _object_cloud() -> GaussianCloud:
    vertices = np.zeros(
        4,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "f4"),
            ("green", "f4"),
            ("blue", "f4"),
            ("opacity", "f4"),
        ],
    )
    vertices["x"] = np.array([-1.0, -0.8, 0.9, 1.1], dtype=np.float32)
    vertices["y"] = np.array([0.0, 0.1, 0.0, -0.1], dtype=np.float32)
    vertices["z"] = 0.0
    vertices["red"] = np.array([0.9, 0.8, 0.1, 0.1], dtype=np.float32)
    vertices["green"] = np.array([0.1, 0.1, 0.8, 0.9], dtype=np.float32)
    vertices["blue"] = 0.2
    vertices["opacity"] = 1.0
    return GaussianCloud(vertices)

