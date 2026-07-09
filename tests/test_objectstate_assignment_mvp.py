from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.objectstate_assignment_mvp import (
    OBJECTSTATE_ASSIGNMENT_MVP_SCHEMA,
    objectstate_assignment_mvp_summary,
    validate_objectstate_assignment_mvp_summary,
)


def test_objectstate_assignment_mvp_runs_gaussian_to_projection_path():
    cloud = _two_object_cloud()
    state = _x_axis_state(feature_dim=7)
    target = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )

    summary = objectstate_assignment_mvp_summary(
        cloud,
        state,
        target_assignment=target,
        source="unit-test-cloud",
        include_assignment=True,
    )

    assert summary["schema"] == OBJECTSTATE_ASSIGNMENT_MVP_SCHEMA
    assert summary["kind"] == "objectstate_assignment_mvp"
    assert summary["status"] == "objectstate_assignment_mvp_ready"
    assert summary["model_contract"] == (
        "Gaussian / AssignmentEvidence -> A[N,K] -> ObjectStateProjection"
    )
    assert summary["input"]["gaussian_count"] == 4
    assert summary["input"]["feature_dim"] == 7
    assert summary["solver"]["family"] == "cost-softmax-assignment-v2"
    assert summary["solver"]["slots"] == 2
    assert summary["assignment"]["shape"] == [4, 2]
    assert summary["assignment"]["row_normalized"] is True
    assert summary["assignment"]["derived_object_ids"] == [0, 0, 1, 1]
    assert summary["target_metrics"] == pytest.approx(
        {
            "target_slots": 2,
            "mean_best_iou": 1.0,
            "ari": 1.0,
            "purity": 1.0,
        }
    )
    assert summary["projection"]["object_state_count"] == 2
    assert summary["projection"]["active_state_count"] == 2
    assert summary["projection"]["states"][0]["centroid"][0] < 0.0
    assert summary["projection"]["states"][1]["centroid"][0] > 0.0
    assert summary["claim_policy"]["assignment_matrix_is_single_source_of_truth"] is True
    assert summary["non_goals"]["trains_model"] is False
    assert validate_objectstate_assignment_mvp_summary(summary) == summary


def test_objectstate_assignment_mvp_allows_unlabeled_inference_handoff():
    summary = objectstate_assignment_mvp_summary(
        _two_object_cloud(),
        _x_axis_state(feature_dim=7),
        source="unlabeled-cloud",
    )

    assert summary["target_metrics"] is None
    assert summary["projection"]["derived_object_id_source"] == "argmax(A[N,K])"
    assert summary["claim_policy"]["does_not_claim_identity_gate_pass"] is True
    assert summary["claim_policy"]["does_not_claim_world_model"] is True


def test_objectstate_assignment_mvp_rejects_solver_feature_mismatch():
    bad_state = _x_axis_state(feature_dim=6)

    with pytest.raises(ValueError, match="feature_dim"):
        objectstate_assignment_mvp_summary(_two_object_cloud(), bad_state)


def _x_axis_state(*, feature_dim: int) -> AssignmentSolverV2State:
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(
            slots=2,
            feature_dim=feature_dim,
            temperature=0.2,
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
        source="x_axis_assignment_mvp_fixture",
    )


def _two_object_cloud() -> GaussianCloud:
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
    vertices["x"] = [-1.0, -0.8, 0.8, 1.0]
    vertices["y"] = [0.0, 0.05, 0.0, -0.05]
    vertices["z"] = [0.0, 0.0, 0.0, 0.0]
    vertices["red"] = [1.0, 1.0, 0.0, 0.0]
    vertices["green"] = [0.0, 0.0, 0.0, 0.0]
    vertices["blue"] = [0.0, 0.0, 1.0, 1.0]
    vertices["opacity"] = [1.0, 1.0, 1.0, 1.0]
    return GaussianCloud(vertices)
