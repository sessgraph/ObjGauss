from __future__ import annotations

import json

import numpy as np
import pytest

from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.evaluation.objectstate_model_identity_gate import (
    OBJECTSTATE_MODEL_IDENTITY_BASELINES,
    OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA,
    ObjectStateModelIdentityGateThresholds,
    objectstate_model_identity_gate_summary,
    validate_objectstate_model_identity_gate_summary,
)


def test_model_identity_gate_is_permutation_aware(tmp_path):
    summary = objectstate_model_identity_gate_summary(
        _swap_cloud(frame=0),
        np.asarray([0, 0, 1, 1], dtype=np.int64),
        _swap_cloud(frame=1),
        np.asarray([1, 1, 0, 0], dtype=np.int64),
        _position_solver_state(feature_dim=3),
        output_dir=tmp_path,
        sample_id="identity-swap-smoke",
        frame0_features=_color_features(frame=0),
        frame1_features=_color_features(frame=1),
        thresholds=ObjectStateModelIdentityGateThresholds(
            identity_retrieval_at_1_min=1.0,
            assignment_consistency_min=1.0,
            objectstate_drift_max=0.01,
        ),
    )

    assert summary["schema"] == OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA
    assert summary["status"] == "objectstate_model_identity_gate_pass"
    assert summary["gate_status"] == "pass"
    assert summary["metrics"]["identity_retrieval_at_1"] == pytest.approx(1.0)
    assert summary["metrics"]["identity_margin"] > 0.0
    assert summary["metrics"]["slot_swap_rate"] == pytest.approx(1.0)
    assert summary["metrics"]["assignment_consistency"] == pytest.approx(1.0)
    assert summary["metrics"]["objectstate_drift"] == pytest.approx(0.0, abs=1e-6)
    assert summary["candidate"]["matching"][0]["correct"] is True
    assert set(summary["baselines"]) == set(OBJECTSTATE_MODEL_IDENTITY_BASELINES)
    assert summary["claim_policy"]["uses_permutation_aware_identity_matching"] is True
    assert summary["non_goals"]["uses_hungarian_dependency"] is False
    assert validate_objectstate_model_identity_gate_summary(summary) == summary

    assert (tmp_path / "identity-summary.json").exists()
    assert (tmp_path / "identity-matching.json").exists()
    assert (tmp_path / "objectstate-retrieval.json").exists()
    assert (tmp_path / "identity-pairwise-distances.csv").exists()
    assert (tmp_path / "assignment-t0.ply").exists()
    assert (tmp_path / "assignment-t1.ply").exists()
    assert json.loads(tmp_path.joinpath("identity-summary.json").read_text())["schema"] == (
        OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA
    )


def test_model_identity_gate_requires_two_shared_identities(tmp_path):
    with pytest.raises(ValueError, match="at least two shared physical identities"):
        objectstate_model_identity_gate_summary(
            _swap_cloud(frame=0),
            np.asarray([0, 0, 0, 0], dtype=np.int64),
            _swap_cloud(frame=1),
            np.asarray([0, 0, 0, 0], dtype=np.int64),
            _position_solver_state(feature_dim=3),
            output_dir=tmp_path,
            frame0_features=_color_features(frame=0),
            frame1_features=_color_features(frame=1),
        )


def _position_solver_state(*, feature_dim: int) -> AssignmentSolverV2State:
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(
            slots=2,
            feature_dim=feature_dim,
            temperature=0.25,
            feature_weight=0.0,
            position_weight=1.0,
        ),
        feature_centers=np.zeros((2, feature_dim), dtype=np.float32),
        position_centers=np.asarray(
            [
                [-1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        slot_bias=np.zeros(2, dtype=np.float32),
        source="position_identity_swap_fixture",
    )


def _swap_cloud(*, frame: int) -> GaussianCloud:
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "f4"),
            ("green", "f4"),
            ("blue", "f4"),
            ("opacity", "f4"),
        ]
    )
    vertices = np.zeros(4, dtype=dtype)
    if frame == 0:
        vertices["x"] = [-1.0, -0.9, 0.9, 1.0]
        colors = [(1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0)]
    else:
        vertices["x"] = [-1.0, -0.9, 0.9, 1.0]
        colors = [(0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    vertices["y"] = [0.0, 0.05, 0.0, -0.05]
    vertices["z"] = [0.0, 0.0, 0.0, 0.0]
    vertices["red"] = [item[0] for item in colors]
    vertices["green"] = [item[1] for item in colors]
    vertices["blue"] = [item[2] for item in colors]
    vertices["opacity"] = [1.0, 1.0, 1.0, 1.0]
    return GaussianCloud(vertices)


def _color_features(*, frame: int) -> np.ndarray:
    if frame == 0:
        return np.asarray(
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
    return np.asarray(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
