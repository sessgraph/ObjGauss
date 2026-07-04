from __future__ import annotations

import json

import numpy as np

from objgauss.cli import main
from objgauss.core.assignment_evidence import assignment_evidence_sequence_from_trainable_frames
from objgauss.core.assignment_stability import (
    ASSIGNMENT_STABILITY_EVAL_SCHEMA,
    evaluate_assignment_stability,
    validate_assignment_stability_eval,
)
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_emergence_solver import (
    ObjectEmergenceEvidence,
    ObjectEmergenceSolverConfig,
    ObjectEmergenceSolverState,
    object_emergence_solver_checkpoint,
    train_object_emergence_solver,
)
from objgauss.core.trainable_kernel import trainable_kernel_sample_from_cloud
from objgauss.ply import write_ply


def test_evaluate_assignment_stability_reports_temporal_gates():
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=6,
        seed=3,
    )
    result = train_object_emergence_solver(
        _solver_frames(sample),
        slots=sample.slots,
        iterations=3,
        learning_rate=0.4,
        assignment_weight=1.0,
        entropy_weight=0.01,
        balance_weight=0.05,
        temporal_weight=0.0,
        seed=4,
    )
    checkpoint = object_emergence_solver_checkpoint(
        result,
        input_path="fixture://assignment-stability",
        source_gaussians=sample.source_count,
        sampled_gaussians=sample.sampled_count,
        target_source=sample.target_source,
        object_id_mapping=sample.object_id_mapping,
    )
    evidence = assignment_evidence_sequence_from_trainable_frames(
        sample.frames,
        source="assignment_stability_fixture",
    )

    summary = evaluate_assignment_stability(
        evidence,
        checkpoint,
        entropy_threshold=1.0,
        purity_threshold=0.0,
        collapse_mass_fraction=1.0,
        id_stability_threshold=0.0,
        temporal_drift_threshold=10.0,
        solver_temperature=0.5,
    )

    assert summary["schema"] == ASSIGNMENT_STABILITY_EVAL_SCHEMA
    assert summary["checkpoint_schema"] == "objgauss-object-emergence-solver-checkpoint-v1"
    assert summary["status"] == "assignment_stability_eval_pass"
    assert summary["solver"]["step"] == result.final_state.step
    assert summary["solver"]["temperature"] == 0.5
    assert summary["solver"]["temperature_override"] is True
    assert summary["aggregate"]["frame_count"] == 2
    assert summary["aggregate"]["evidence_count"] == 12
    assert summary["temporal"]["pair_count"] == 1
    assert summary["dynamic_k"]["mode"] == "proposal_only"
    assert summary["dynamic_k"]["auto_update"] is False
    assert summary["aggregate"]["id_stability"] >= 0.0
    assert summary["gates"]["entropy_pass"] is True
    assert summary["gates"]["purity_pass"] is True
    assert summary["gates"]["id_stability_pass"] is True
    assert summary["gates"]["temporal_drift_pass"] is True
    assert validate_assignment_stability_eval(summary) == summary


def test_assignment_stability_dynamic_k_reports_split_proposals_without_mutating_k():
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=6,
        seed=3,
    )
    evidence = assignment_evidence_sequence_from_trainable_frames(
        sample.frames,
        source="assignment_stability_uniform_solver_fixture",
    )
    feature_dim = sample.frames[0].features.shape[1]
    state = ObjectEmergenceSolverState(
        config=ObjectEmergenceSolverConfig(slots=sample.slots, feature_dim=feature_dim),
        feature_weights=np.zeros((feature_dim, sample.slots), dtype=np.float32),
        position_weights=np.zeros((3, sample.slots), dtype=np.float32),
        bias=np.zeros(sample.slots, dtype=np.float32),
        source="uniform_assignment_fixture",
    )

    summary = evaluate_assignment_stability(
        evidence,
        state,
        entropy_threshold=1.0,
        purity_threshold=0.0,
        collapse_mass_fraction=1.0,
        id_stability_threshold=0.0,
    )

    assert summary["dynamic_k"]["mode"] == "proposal_only"
    assert summary["dynamic_k"]["auto_update"] is False
    assert summary["dynamic_k"]["checkpoint_k_mutation"] == "forbidden"
    assert "split_mixed" in summary["dynamic_k"]["proposal_kinds"]
    assert summary["dynamic_k"]["proposal_count"] >= 1
    actions = [
        proposal["action"]
        for frame in summary["dynamic_k"]["frames"]
        for proposal in frame["proposals"]
    ]
    assert actions
    assert set(actions) == {"proposal_only"}


def test_eval_assignment_cli_writes_summary(tmp_path, capsys):
    input_path = tmp_path / "objects.ply"
    checkpoint_path = tmp_path / "solver-checkpoint.json"
    summary_path = tmp_path / "assignment-eval.json"
    cloud = _object_cloud()
    write_ply(input_path, cloud, fmt="ascii")
    sample = trainable_kernel_sample_from_cloud(
        cloud,
        frame_count=2,
        max_points=6,
        seed=0,
    )
    result = train_object_emergence_solver(
        _solver_frames(sample),
        slots=sample.slots,
        iterations=2,
        learning_rate=0.4,
        assignment_weight=1.0,
        entropy_weight=0.01,
        balance_weight=0.05,
        temporal_weight=0.0,
        seed=5,
    )
    checkpoint = object_emergence_solver_checkpoint(
        result,
        input_path=str(input_path),
        source_gaussians=sample.source_count,
        sampled_gaussians=sample.sampled_count,
        target_source=sample.target_source,
        object_id_mapping=sample.object_id_mapping,
    )
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    status = main(
        [
            "training",
            "eval-assignment",
            str(input_path),
            "--checkpoint",
            str(checkpoint_path),
            "--max-points",
            "6",
            "--frames",
            "2",
            "--solver-temperature",
            "0.5",
            "--entropy-threshold",
            "1.0",
            "--purity-threshold",
            "0.0",
            "--collapse-mass-fraction",
            "1.0",
            "--id-stability-threshold",
            "0.0",
            "--temporal-drift-threshold",
            "10.0",
            "--summary-output",
            str(summary_path),
            "--require-pass",
        ]
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert status == 0
    assert "schema=objgauss-assignment-stability-eval-v1" in stdout
    assert "eval_status=assignment_stability_eval_pass" in stdout
    assert "solver_temperature=0.5" in stdout
    assert "gate_id_stability_pass=true" in stdout
    assert "gate_temporal_drift_pass=true" in stdout
    assert "dynamic_k_mode=proposal_only" in stdout
    assert "dynamic_k_auto_update=false" in stdout
    assert f"summary={summary_path}" in stdout
    assert summary["schema"] == ASSIGNMENT_STABILITY_EVAL_SCHEMA
    assert summary["checkpoint_schema"] == "objgauss-object-emergence-solver-checkpoint-v1"
    assert summary["temporal"]["pair_count"] == 1
    assert summary["dynamic_k"]["mode"] == "proposal_only"


def _solver_frames(sample) -> list[ObjectEmergenceEvidence]:
    return [
        ObjectEmergenceEvidence(
            positions=frame.positions,
            features=frame.features,
            target_assignment=frame.target_assignment,
            frame_index=index,
            source="assignment_stability_fixture",
        )
        for index, frame in enumerate(sample.frames)
    ]


def _object_cloud() -> GaussianCloud:
    vertices = np.zeros(
        6,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("opacity", "f4"),
            ("object_id", "i4"),
        ],
    )
    vertices["x"] = np.asarray([-1.0, -0.9, -1.1, 1.0, 0.9, 1.1], dtype=np.float32)
    vertices["y"] = np.asarray([0.0, 0.1, -0.1, 0.0, 0.1, -0.1], dtype=np.float32)
    vertices["z"] = 0.0
    vertices["red"] = np.asarray([230, 225, 235, 30, 35, 25], dtype=np.uint8)
    vertices["green"] = np.asarray([30, 25, 35, 210, 220, 205], dtype=np.uint8)
    vertices["blue"] = np.asarray([20, 25, 15, 230, 220, 225], dtype=np.uint8)
    vertices["opacity"] = 1.0
    vertices["object_id"] = np.asarray([5, 5, 5, 9, 9, 9], dtype=np.int32)
    return GaussianCloud(vertices=vertices, source_format="fixture")
