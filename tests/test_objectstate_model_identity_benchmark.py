from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from objgauss.core.assignment_solver_v2 import (
    AssignmentSolverV2Config,
    AssignmentSolverV2State,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.objectstate_model_identity_benchmark import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS,
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA,
    ObjectStateModelIdentityBenchmarkScenario,
    objectstate_model_identity_benchmark_summary,
    validate_objectstate_model_identity_benchmark_summary,
)


def test_model_identity_benchmark_aggregates_required_perturbations(tmp_path):
    scenarios = [
        _scenario("viewpoint"),
        _scenario("dropout"),
        _scenario("occlusion"),
        _scenario("appearance"),
        _scenario("spatial"),
    ]

    summary = objectstate_model_identity_benchmark_summary(
        scenarios,
        _identity_feature_solver_state(),
        output_dir=tmp_path,
        sample_id="identity-benchmark-smoke",
        seed=7,
    )

    assert summary["schema"] == OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA
    assert summary["status"] == "objectstate_model_identity_benchmark_candidate_ready"
    assert summary["num_scenarios"] == len(OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS)
    assert summary["num_pairs"] == 15
    assert summary["long_training_gate"]["status"] == "candidate_ready"
    assert summary["long_training_gate"]["reasons"] == []
    assert all(summary["perturbation_coverage"].values())
    solver = summary["baselines"]["assignment_solver_v2"]["metrics"]
    xyz = summary["baselines"]["xyz_centroid"]["metrics"]
    random = summary["baselines"]["random_assignment"]["metrics"]
    oracle = summary["baselines"]["oracle_target_assignment"]["metrics"]
    assert solver["identity_retrieval_at_1"] == pytest.approx(1.0)
    assert solver["identity_retrieval_at_1"] > xyz["identity_retrieval_at_1"]
    assert solver["occlusion_recovery"] > random["occlusion_recovery"]
    assert oracle["identity_retrieval_at_1"] >= solver["identity_retrieval_at_1"]
    assert summary["perturbation_breakdown"]["dropout"]["num_scenarios"] == 1
    assert summary["claim_policy"]["physical_identity_labels_are_evaluation_only"] is True
    assert summary["non_goals"]["uses_temporal_loss"] is False
    assert validate_objectstate_model_identity_benchmark_summary(summary) == summary

    assert (tmp_path / "identity-benchmark-summary.json").exists()
    assert json.loads(tmp_path.joinpath("identity-benchmark-summary.json").read_text())[
        "schema"
    ] == OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA
    for scenario_result in summary["scenario_results"]:
        assert scenario_result["summary_path"]
        assert Path(scenario_result["summary_path"]).exists()


def test_model_identity_benchmark_blocks_when_required_coverage_is_missing(tmp_path):
    summary = objectstate_model_identity_benchmark_summary(
        [_scenario("viewpoint")],
        _identity_feature_solver_state(),
        output_dir=tmp_path,
        sample_id="identity-benchmark-missing-coverage",
        seed=7,
    )

    assert summary["status"] == "objectstate_model_identity_benchmark_blocked"
    assert summary["long_training_gate"]["status"] == "blocked"
    assert "required_perturbations" in summary["long_training_gate"]["reasons"][0]
    assert summary["perturbation_coverage"]["viewpoint"] is True
    assert summary["perturbation_coverage"]["dropout"] is False


def test_model_identity_benchmark_rejects_duplicate_scenario_ids(tmp_path):
    with pytest.raises(ValueError, match="duplicate model identity benchmark scenario_id"):
        objectstate_model_identity_benchmark_summary(
            [_scenario("viewpoint"), _scenario("viewpoint")],
            _identity_feature_solver_state(),
            output_dir=tmp_path,
        )


def _identity_feature_solver_state() -> AssignmentSolverV2State:
    return AssignmentSolverV2State(
        config=AssignmentSolverV2Config(
            slots=3,
            feature_dim=3,
            temperature=0.15,
            feature_weight=1.0,
            position_weight=0.0,
        ),
        feature_centers=np.eye(3, dtype=np.float32),
        position_centers=np.zeros((3, 3), dtype=np.float32),
        slot_bias=np.zeros(3, dtype=np.float32),
        source="identity_feature_benchmark_fixture",
    )


def _scenario(kind: str) -> ObjectStateModelIdentityBenchmarkScenario:
    labels0, positions0, colors0, features0 = _frame_arrays("base")
    labels1, positions1, colors1, features1 = _frame_arrays(kind)
    return ObjectStateModelIdentityBenchmarkScenario(
        scenario_id=f"{kind}-challenge",
        perturbation_kind=kind,
        frame0_cloud=_cloud(positions0, colors0),
        frame0_identity_labels=labels0,
        frame1_cloud=_cloud(positions1, colors1),
        frame1_identity_labels=labels1,
        frame0_id=f"{kind}:t0",
        frame1_id=f"{kind}:t1",
        frame0_features=features0,
        frame1_features=features1,
        description=f"{kind} synthetic identity challenge",
    )


def _frame_arrays(kind: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    positions = []
    labels = []
    for identity in range(3):
        for part_index, x in enumerate((-1.0, 0.0, 1.0)):
            if kind == "dropout" and part_index == 1:
                continue
            if kind == "occlusion" and identity == 1 and part_index == 2:
                continue
            position = np.asarray([x, identity * 0.01, part_index * 0.02], dtype=np.float32)
            if kind == "viewpoint":
                position = np.asarray([position[2] - 0.02, position[1], -position[0]], dtype=np.float32)
            elif kind == "spatial":
                position = position + np.asarray([0.35, -0.15, 0.08], dtype=np.float32)
            positions.append(position)
            labels.append(identity)
    label_array = np.asarray(labels, dtype=np.int64)
    features = np.eye(3, dtype=np.float32)[label_array]
    colors = features * 0.8 + 0.1
    if kind == "appearance":
        colors = 1.0 - colors * 0.65
    return (
        label_array,
        np.asarray(positions, dtype=np.float32),
        colors.astype(np.float32, copy=False),
        features.astype(np.float32, copy=False),
    )


def _cloud(positions: np.ndarray, colors: np.ndarray) -> GaussianCloud:
    dtype = np.dtype([
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "f4"),
        ("green", "f4"),
        ("blue", "f4"),
        ("opacity", "f4"),
    ])
    vertices = np.zeros(positions.shape[0], dtype=dtype)
    for index, field in enumerate(("x", "y", "z")):
        vertices[field] = positions[:, index]
    for index, field in enumerate(("red", "green", "blue")):
        vertices[field] = colors[:, index]
    vertices["opacity"] = 1.0
    return GaussianCloud(vertices)
