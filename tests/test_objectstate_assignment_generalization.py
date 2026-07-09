from __future__ import annotations

import json

import numpy as np
import pytest

from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.objectstate_assignment_generalization import (
    OBJECTSTATE_ASSIGNMENT_GENERALIZATION_SCHEMA,
    objectstate_assignment_generalization_summary,
    validate_objectstate_assignment_generalization_summary,
)


def test_assignment_generalization_evaluates_held_out_scene(tmp_path):
    summary = objectstate_assignment_generalization_summary(
        _two_object_cloud(offset=0.0),
        _target_assignment(),
        _two_object_cloud(offset=0.25),
        _target_assignment(),
        output_dir=tmp_path,
        sample_id="generalization-test",
        train_sample_id="train-scene",
        test_sample_id="heldout-scene",
        initial_state=_collapsed_initial_state(feature_dim=7),
        iterations=80,
        learning_rate=0.4,
        assignment_weight=1.0,
        compactness_weight=0.05,
        seed=11,
    )

    assert summary["schema"] == OBJECTSTATE_ASSIGNMENT_GENERALIZATION_SCHEMA
    assert summary["status"] == "objectstate_assignment_generalization_pass"
    assert summary["train_dataset"]["split"] == "train"
    assert summary["test_dataset"]["split"] == "test"
    assert summary["loss"]["loss_decreased"] is True
    assert summary["train_metric_delta"]["ari"] > 0.0
    assert summary["test_metric_delta"]["ari"] > 0.0
    assert summary["test_after_metrics"]["ari"] == pytest.approx(1.0)
    assert summary["test_after_metrics"]["purity"] == pytest.approx(1.0)
    assert summary["generalization_gap"]["ari"] == pytest.approx(0.0)
    assert summary["long_training_gate"]["allowed"] is True
    assert summary["checkpoint"]["roundtrip_ok"] is True
    assert summary["decision"]["next_action"] == "run_identity_gate_handoff"
    assert summary["claim_policy"]["tests_held_out_assignment_sample"] is True
    assert validate_objectstate_assignment_generalization_summary(summary) == summary

    summary_path = tmp_path / "generalization-summary.json"
    checkpoint_path = tmp_path / "assignment-generalization-final-state.json"
    assert summary_path.exists()
    assert checkpoint_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["schema"] == (
        OBJECTSTATE_ASSIGNMENT_GENERALIZATION_SCHEMA
    )


def test_assignment_generalization_requires_matching_train_test_slots(tmp_path):
    with pytest.raises(ValueError, match="same slot count"):
        objectstate_assignment_generalization_summary(
            _two_object_cloud(offset=0.0),
            _target_assignment(),
            _two_object_cloud(offset=0.25),
            np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype=np.float32,
            ),
            output_dir=tmp_path,
        )


def _target_assignment() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _collapsed_initial_state(*, feature_dim: int) -> AssignmentSolverV2State:
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(
            slots=2,
            feature_dim=feature_dim,
            temperature=0.5,
            feature_weight=0.5,
            position_weight=1.0,
        ),
        feature_centers=np.zeros((2, feature_dim), dtype=np.float32),
        position_centers=np.zeros((2, 3), dtype=np.float32),
        slot_bias=np.zeros(2, dtype=np.float32),
        source="collapsed_assignment_generalization_fixture",
    )


def _two_object_cloud(*, offset: float) -> GaussianCloud:
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
    vertices["x"] = [-1.0 - offset, -0.8 - offset, 0.8 + offset, 1.0 + offset]
    vertices["y"] = [0.0, 0.05, 0.0, -0.05]
    vertices["z"] = [0.0, 0.0, 0.0, 0.0]
    vertices["red"] = [1.0, 1.0, 0.0, 0.0]
    vertices["green"] = [0.0, 0.0, 0.0, 0.0]
    vertices["blue"] = [0.0, 0.0, 1.0, 1.0]
    vertices["opacity"] = [1.0, 1.0, 1.0, 1.0]
    return GaussianCloud(vertices)
