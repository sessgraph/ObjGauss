from __future__ import annotations

import json

import numpy as np
import pytest

from objgauss.cli import main
from objgauss.core.io import write_ply
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_emergence_solver import (
    OBJECT_EMERGENCE_ASSIGNMENT_SCHEMA,
    OBJECT_EMERGENCE_TRAINING_SCHEMA,
    ObjectEmergenceEvidence,
    ObjectEmergenceSolverConfig,
    ObjectEmergenceSolverState,
    evidence_from_gaussian_cloud,
    initialize_object_emergence_solver,
    object_id_targets_from_cloud,
    predict_object_emergence_assignment,
    project_object_emergence_prediction,
    train_object_emergence_solver,
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


def test_train_object_emergence_solver_updates_weights_and_reduces_loss():
    frames = _training_frames()

    result = train_object_emergence_solver(
        frames,
        slots=2,
        iterations=24,
        learning_rate=0.8,
        assignment_weight=1.0,
        entropy_weight=0.01,
        balance_weight=0.05,
        temporal_weight=0.01,
        seed=3,
        record_every=8,
    )

    assert result.schema == OBJECT_EMERGENCE_TRAINING_SCHEMA
    assert result.final_loss.total_loss < result.initial_loss.total_loss
    assert result.final_loss.assignment_loss < result.initial_loss.assignment_loss
    assert result.final_state.step == 24
    assert result.final_state.source == "trained_numpy_finite_difference"
    assert not np.allclose(result.initial_state.feature_weights, result.final_state.feature_weights)
    assert len(result.history) >= 3
    assert len(result.predictions) == 2
    np.testing.assert_allclose(result.predictions[0].assignment.sum(axis=1), np.ones(4), atol=1e-6)
    assert result.predictions[0].confidence.mean() > 0.7

    summary = result.as_dict(include_weights=True, include_assignments=True)
    assert summary["schema"] == OBJECT_EMERGENCE_TRAINING_SCHEMA
    assert summary["loss_decreased"] is True
    assert summary["assignment_loss_decreased"] is True
    assert summary["final_solver_state"]["step"] == 24
    assert "weights" in summary["final_solver_state"]
    assert len(summary["predictions"][0]["assignment"]) == 4


def test_object_id_targets_from_cloud_bind_solver_training_targets():
    cloud = _object_id_cloud()
    targets, mapping = object_id_targets_from_cloud(cloud)
    evidence = evidence_from_gaussian_cloud(cloud, target_assignment=targets)
    result = train_object_emergence_solver(
        [evidence],
        iterations=18,
        learning_rate=0.75,
        assignment_weight=1.0,
        entropy_weight=0.0,
        balance_weight=0.05,
        temporal_weight=0.0,
        seed=4,
    )

    assert mapping == {5: 0, 9: 1}
    assert targets.shape == (4, 2)
    assert result.final_loss.assignment_loss < result.initial_loss.assignment_loss
    assert result.final_state.config.slots == 2


def test_object_emergence_solver_cli_writes_cpu_training_summary(tmp_path, capsys):
    input_path = tmp_path / "object_cloud.ply"
    summary_path = tmp_path / "solver-summary.json"
    write_ply(input_path, _object_id_cloud(), fmt="ascii")

    status = main(
        [
            "training",
            "object-emergence-solver",
            str(input_path),
            "--iterations",
            "8",
            "--learning-rate",
            "0.5",
            "--summary-output",
            str(summary_path),
            "--require-loss-decrease",
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert status == 0
    assert "schema=objgauss-object-emergence-solver-training-v1" in stdout
    assert "gpu_used=false" in stdout
    assert "vram_reserve_gb=1" in stdout
    assert payload["loss_decreased"] is True
    assert payload["assignment_loss_decreased"] is True
    assert payload["gpu_policy"]["uses_gpu"] is False
    assert payload["gpu_policy"]["vram_reserve_gb"] == 1


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

    no_targets = ObjectEmergenceEvidence(
        positions=np.zeros((2, 3), dtype=np.float32),
        features=np.zeros((2, 2), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="assignment_weight requires every evidence frame"):
        train_object_emergence_solver([no_targets], slots=2, assignment_weight=1.0)


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


def _object_id_cloud() -> GaussianCloud:
    cloud = _object_cloud()
    vertices = np.zeros(
        cloud.count,
        dtype=cloud.vertices.dtype.descr + [("object_id", "i4")],
    )
    for name in cloud.fields:
        vertices[name] = cloud.vertices[name]
    vertices["object_id"] = np.array([5, 5, 9, 9], dtype=np.int32)
    return GaussianCloud(vertices)


def _training_frames() -> tuple[ObjectEmergenceEvidence, ...]:
    target = np.array(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )
    positions_0 = np.array(
        [
            [-1.0, 0.0, 0.0],
            [-0.8, 0.1, 0.0],
            [0.9, 0.0, 0.0],
            [1.1, -0.1, 0.0],
        ],
        dtype=np.float32,
    )
    positions_1 = positions_0 + np.array(
        [
            [0.03, 0.01, 0.0],
            [0.03, 0.00, 0.0],
            [-0.02, 0.01, 0.0],
            [-0.02, 0.00, 0.0],
        ],
        dtype=np.float32,
    )
    features_0 = np.array(
        [
            [1.0, 0.0],
            [0.9, 0.1],
            [0.0, 1.0],
            [0.1, 0.9],
        ],
        dtype=np.float32,
    )
    features_1 = np.array(
        [
            [0.95, 0.05],
            [0.85, 0.15],
            [0.05, 0.95],
            [0.15, 0.85],
        ],
        dtype=np.float32,
    )
    return (
        ObjectEmergenceEvidence(
            positions=positions_0,
            features=features_0,
            target_assignment=target,
            frame_index=0,
            source="training-fixture",
        ),
        ObjectEmergenceEvidence(
            positions=positions_1,
            features=features_1,
            target_assignment=target,
            frame_index=1,
            source="training-fixture",
        ),
    )
