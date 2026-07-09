from __future__ import annotations

import json

import numpy as np
import pytest

from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io import read_ply
from objgauss.core.objectstate_assignment_train import (
    OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA,
    OBJECTSTATE_ASSIGNMENT_TRAIN_RUN_SCHEMA,
    objectstate_assignment_train_dataset_summary,
    objectstate_assignment_train_smoke,
    validate_objectstate_assignment_train_dataset_summary,
    validate_objectstate_assignment_train_run_summary,
)


def test_objectstate_assignment_train_smoke_writes_metrics_checkpoint_and_visuals(tmp_path):
    cloud = _two_object_cloud()
    target = _target_assignment()

    summary = objectstate_assignment_train_smoke(
        cloud,
        target,
        output_dir=tmp_path,
        sample_id="assignment-train-smoke-test",
        initial_state=_collapsed_initial_state(feature_dim=7),
        iterations=60,
        learning_rate=0.4,
        assignment_weight=1.0,
        compactness_weight=0.05,
        seed=7,
    )

    assert summary["schema"] == OBJECTSTATE_ASSIGNMENT_TRAIN_RUN_SCHEMA
    assert summary["status"] == "objectstate_assignment_train_smoke_pass"
    assert summary["dataset"]["schema"] == OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA
    assert summary["loss"]["loss_decreased"] is True
    assert summary["loss"]["assignment_loss_decreased"] is True
    assert summary["after_metrics"]["ari"] > summary["before_metrics"]["ari"]
    assert summary["after_metrics"]["purity"] > summary["before_metrics"]["purity"]
    assert summary["after_metrics"]["mean_best_iou"] > summary["before_metrics"]["mean_best_iou"]
    assert summary["long_training_gate"]["allowed"] is True
    assert summary["checkpoint"]["roundtrip_ok"] is True
    assert summary["gate_handoff"]["identity_gate_status"] == "not_run"
    assert summary["claim_policy"]["short_smoke_only"] is True
    assert summary["non_goals"]["uses_gpu"] is False
    assert validate_objectstate_assignment_train_run_summary(summary) == summary

    summary_path = tmp_path / "summary.json"
    checkpoint_path = tmp_path / "assignment-solver-v2-final-state.json"
    before_ply = tmp_path / "assignment-before.ply"
    after_ply = tmp_path / "assignment-after.ply"
    assert summary_path.exists()
    assert checkpoint_path.exists()
    assert before_ply.exists()
    assert after_ply.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["schema"] == (
        OBJECTSTATE_ASSIGNMENT_TRAIN_RUN_SCHEMA
    )
    after_cloud = read_ply(after_ply)
    assert "predicted_object_id" in after_cloud.fields
    assert after_cloud.vertices["predicted_object_id"].tolist() == [0, 0, 1, 1]


def test_assignment_train_dataset_summary_records_supervision_only_contract():
    summary = objectstate_assignment_train_dataset_summary(
        _two_object_cloud(),
        _target_assignment(),
        sample_id="dataset-contract-test",
    )

    assert summary["schema"] == OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA
    assert summary["source_kind"] == "synthetic"
    assert summary["gaussian_count"] == 4
    assert summary["slots"] == 2
    assert summary["target_object_labels"] == [0, 0, 1, 1]
    assert summary["claim_policy"]["target_assignment_is_supervision_only"] is True
    assert validate_objectstate_assignment_train_dataset_summary(summary) == summary


def test_assignment_train_smoke_keeps_iteration_bound_short(tmp_path):
    with pytest.raises(ValueError, match="iterations must stay <= 600"):
        objectstate_assignment_train_smoke(
            _two_object_cloud(),
            _target_assignment(),
            output_dir=tmp_path,
            iterations=601,
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
        source="collapsed_assignment_train_smoke_fixture",
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
