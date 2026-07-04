from __future__ import annotations

import json

import numpy as np

from objgauss.cli import main
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.object_state_eval import (
    OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA,
    evaluate_solver_decoder_object_states,
    validate_objectstate_checkpoint_eval,
)
from objgauss.core.solver_decoder_training import (
    solver_decoder_joint_checkpoint,
    train_solver_decoder_joint,
)
from objgauss.core.trainable_kernel import trainable_kernel_sample_from_cloud
from objgauss.ply import write_ply


def test_evaluate_solver_decoder_object_states_reports_gates():
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=6,
        bind_image_targets=True,
        image_width=8,
        image_height=8,
        seed=3,
    )
    result = train_solver_decoder_joint(
        sample.frames,
        iterations=3,
        solver_learning_rate=0.08,
        decoder_learning_rate=0.6,
        object_weight=0.2,
        seed=4,
    )
    checkpoint = solver_decoder_joint_checkpoint(
        result,
        input_path="fixture://objectstate-eval",
        source_gaussians=sample.source_count,
        sampled_gaussians=sample.sampled_count,
        target_source=sample.target_source,
        assignment_source="object_id_one_hot_targets",
        object_id_mapping=sample.object_id_mapping,
    )

    summary = evaluate_solver_decoder_object_states(
        sample,
        checkpoint,
        entropy_threshold=1.0,
        purity_threshold=0.0,
        collapse_mass_fraction=1.0,
    )
    sharpened = evaluate_solver_decoder_object_states(
        sample,
        checkpoint,
        entropy_threshold=1.0,
        purity_threshold=0.0,
        collapse_mass_fraction=1.0,
        solver_temperature=0.5,
    )

    assert summary["schema"] == OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA
    assert summary["status"] == "objectstate_eval_pass"
    assert summary["solver"]["step"] == result.final_solver_state.step
    assert summary["decoder"]["step"] == result.final_decoder_state.step
    assert summary["aggregate"]["frame_count"] == 2
    assert summary["aggregate"]["evidence_count"] == 12
    assert summary["gates"]["entropy_pass"] is True
    assert summary["gates"]["purity_pass"] is True
    assert summary["frames"][0]["slot_summaries"]
    assert validate_objectstate_checkpoint_eval(summary) == summary
    assert sharpened["solver"]["temperature"] == 0.5
    assert sharpened["solver"]["temperature_override"] is True
    assert (
        sharpened["aggregate"]["mean_normalized_entropy"]
        <= summary["aggregate"]["mean_normalized_entropy"]
    )


def test_eval_objectstate_cli_writes_summary(tmp_path, capsys):
    input_path = tmp_path / "objects.ply"
    checkpoint_path = tmp_path / "joint-checkpoint.json"
    summary_path = tmp_path / "objectstate-eval.json"
    write_ply(input_path, _object_cloud(), fmt="ascii")
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=6,
        bind_image_targets=True,
        image_width=8,
        image_height=8,
        seed=0,
    )
    result = train_solver_decoder_joint(
        sample.frames,
        iterations=2,
        solver_learning_rate=0.08,
        decoder_learning_rate=0.6,
        object_weight=0.2,
        seed=5,
    )
    checkpoint = solver_decoder_joint_checkpoint(
        result,
        input_path=str(input_path),
        source_gaussians=sample.source_count,
        sampled_gaussians=sample.sampled_count,
        target_source=sample.target_source,
        assignment_source="object_id_one_hot_targets",
        object_id_mapping=sample.object_id_mapping,
    )
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    status = main(
        [
            "training",
            "eval-objectstate",
            str(input_path),
            "--checkpoint",
            str(checkpoint_path),
            "--max-points",
            "6",
            "--solver-temperature",
            "0.5",
            "--entropy-threshold",
            "1.0",
            "--purity-threshold",
            "0.0",
            "--collapse-mass-fraction",
            "1.0",
            "--summary-output",
            str(summary_path),
        ]
    )

    stdout = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert status == 0
    assert "schema=objgauss-objectstate-checkpoint-eval-v1" in stdout
    assert "eval_status=objectstate_eval_pass" in stdout
    assert "solver_temperature=0.5" in stdout
    assert "gate_entropy_pass=true" in stdout
    assert f"summary={summary_path}" in stdout
    assert summary["schema"] == OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA
    assert summary["checkpoint_schema"] == "objgauss-solver-decoder-joint-checkpoint-v1"
    assert summary["solver"]["temperature"] == 0.5
    assert summary["solver"]["temperature_override"] is True


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
