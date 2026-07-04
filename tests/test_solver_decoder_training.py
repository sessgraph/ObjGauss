from __future__ import annotations

import json

import numpy as np

from objgauss.cli import main
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.solver_decoder_training import (
    SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA,
    SOLVER_DECODER_JOINT_TRAINING_SCHEMA,
    SolverDecoderJointTrainingResult,
    solver_decoder_joint_checkpoint,
    solver_decoder_joint_states_from_dict,
    train_solver_decoder_joint,
    validate_solver_decoder_joint_checkpoint,
)
from objgauss.core.trainable_kernel import trainable_kernel_sample_from_cloud
from objgauss.core.training_scale import (
    TRAINING_SCALE_PLAN_SCHEMA,
    solver_decoder_training_scale_plan,
    validate_solver_decoder_training_scale_plan,
)
from objgauss.ply import write_ply


def test_solver_decoder_joint_training_updates_solver_and_decoder():
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=6,
        bind_image_targets=True,
        image_width=8,
        image_height=8,
        seed=4,
    )

    result = train_solver_decoder_joint(
        sample.frames,
        iterations=6,
        solver_learning_rate=0.08,
        decoder_learning_rate=0.6,
        object_weight=0.2,
        seed=5,
    )
    payload = result.as_dict(include_weights=True, include_assignments=True)

    assert isinstance(result, SolverDecoderJointTrainingResult)
    assert result.schema == SOLVER_DECODER_JOINT_TRAINING_SCHEMA
    assert result.final_loss.total_loss < result.initial_loss.total_loss
    assert result.final_loss.image_render_loss < result.initial_loss.image_render_loss
    assert result.final_loss.object_loss < result.initial_loss.object_loss
    assert "solver.feature_weights" in payload["trained_fields"]
    assert "decoder.object_colors" in payload["trained_fields"]
    assert "dynamic_k" in payload["frozen_fields"]
    assert payload["renderer_api"]["status"] == "ready"
    assert payload["gpu_policy"]["uses_gpu"] is False
    assert payload["predictions"][0]["assignment"]
    assert not np.allclose(
        result.initial_solver_state.feature_weights,
        result.final_solver_state.feature_weights,
    )
    assert not np.allclose(
        result.initial_decoder_state.object_colors,
        result.final_decoder_state.object_colors,
    )


def test_solver_decoder_joint_checkpoint_roundtrips_and_resumes():
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=6,
        bind_image_targets=True,
        image_width=8,
        image_height=8,
        seed=2,
    )
    first = train_solver_decoder_joint(
        sample.frames,
        iterations=3,
        solver_learning_rate=0.08,
        decoder_learning_rate=0.6,
        object_weight=0.2,
        seed=7,
    )

    checkpoint = solver_decoder_joint_checkpoint(
        first,
        input_path="fixture://objects",
        source_gaussians=sample.source_count,
        sampled_gaussians=sample.sampled_count,
        target_source=sample.target_source,
        assignment_source="object_id_one_hot_targets",
        object_id_mapping=sample.object_id_mapping,
        vram_reserve_gb=1,
    )
    restored_solver, restored_decoder = solver_decoder_joint_states_from_dict(checkpoint)
    resumed = train_solver_decoder_joint(
        sample.frames,
        initial_solver_state=restored_solver,
        initial_decoder_state=restored_decoder,
        iterations=2,
        solver_learning_rate=0.04,
        decoder_learning_rate=0.3,
        object_weight=0.2,
        seed=8,
    )

    assert checkpoint["schema"] == SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA
    assert checkpoint["source"]["object_id_mapping"] == {"5": 0, "9": 1}
    assert checkpoint["solver_state"]["weights"]["bias"] is not None
    assert checkpoint["decoder_state"]["object_colors"] is not None
    assert validate_solver_decoder_joint_checkpoint(checkpoint) == checkpoint
    assert restored_solver.step == first.final_solver_state.step
    assert restored_decoder.step == first.final_decoder_state.step
    np.testing.assert_allclose(restored_solver.bias, first.final_solver_state.bias, atol=1e-6)
    np.testing.assert_allclose(
        restored_decoder.object_colors,
        first.final_decoder_state.object_colors,
        atol=1e-6,
    )
    assert resumed.initial_solver_state.step == first.final_solver_state.step
    assert resumed.initial_decoder_state.step == first.final_decoder_state.step
    assert resumed.final_solver_state.step == first.final_solver_state.step + 2
    assert resumed.final_decoder_state.step == first.final_decoder_state.step + 2


def test_solver_decoder_training_scale_plan_segments_run_outputs(tmp_path):
    plan = solver_decoder_training_scale_plan(
        total_iterations=5,
        checkpoint_every=2,
        loss_log_every=1,
        output_dir=tmp_path / "run",
        image_renderer="gsplat",
        vram_reserve_gb=1,
    )

    assert plan["schema"] == TRAINING_SCALE_PLAN_SCHEMA
    assert plan["segment_count"] == 3
    assert [segment["iterations"] for segment in plan["segments"]] == [2, 2, 1]
    assert plan["gpu_policy"]["preflight_required"] is True
    assert plan["gpu_policy"]["vram_reserve_gb"] == 1
    assert plan["outputs"]["final_checkpoint"].endswith("final-checkpoint.json")
    assert validate_solver_decoder_training_scale_plan(plan) == plan


def test_solver_decoder_mvp_cli_writes_joint_summary(tmp_path, capsys):
    input_path = tmp_path / "objects.ply"
    summary_path = tmp_path / "joint-summary.json"
    write_ply(input_path, _object_cloud(), fmt="ascii")

    status = main(
        [
            "training",
            "solver-decoder-mvp",
            str(input_path),
            "--max-points",
            "4",
            "--image-width",
            "8",
            "--image-height",
            "8",
            "--iterations",
            "4",
            "--solver-learning-rate",
            "0.08",
            "--decoder-learning-rate",
            "0.6",
            "--object-weight",
            "0.2",
            "--summary-output",
            str(summary_path),
            "--require-loss-decrease",
            "--require-image-render-loss-decrease",
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert status == 0
    assert "schema=objgauss-solver-decoder-joint-training-v1" in stdout
    assert "trained_fields=solver.feature_weights,solver.position_weights,solver.bias,decoder.object_colors" in stdout
    assert "frozen_fields=means,quats,scales,opacities,cameras,dynamic_k" in stdout
    assert payload["schema"] == SOLVER_DECODER_JOINT_TRAINING_SCHEMA
    assert payload["loss_decreased"] is True
    assert payload["image_render_loss_decreased"] is True
    assert payload["sample"]["sampled_count"] == 4
    assert payload["assignment_source"] == "object_id_one_hot_targets"


def test_solver_decoder_mvp_cli_writes_and_resumes_joint_checkpoint(tmp_path, capsys):
    input_path = tmp_path / "objects.ply"
    first_summary_path = tmp_path / "joint-summary.json"
    checkpoint_path = tmp_path / "joint-checkpoint.json"
    resumed_summary_path = tmp_path / "resumed-summary.json"
    resumed_checkpoint_path = tmp_path / "resumed-checkpoint.json"
    write_ply(input_path, _object_cloud(), fmt="ascii")

    first_status = main(
        [
            "training",
            "solver-decoder-mvp",
            str(input_path),
            "--max-points",
            "4",
            "--image-width",
            "8",
            "--image-height",
            "8",
            "--iterations",
            "3",
            "--solver-learning-rate",
            "0.08",
            "--decoder-learning-rate",
            "0.6",
            "--object-weight",
            "0.2",
            "--summary-output",
            str(first_summary_path),
            "--checkpoint-output",
            str(checkpoint_path),
        ]
    )
    first_stdout = capsys.readouterr().out

    resumed_status = main(
        [
            "training",
            "solver-decoder-mvp",
            str(input_path),
            "--resume-checkpoint",
            str(checkpoint_path),
            "--max-points",
            "4",
            "--image-width",
            "8",
            "--image-height",
            "8",
            "--iterations",
            "2",
            "--solver-learning-rate",
            "0.04",
            "--decoder-learning-rate",
            "0.3",
            "--object-weight",
            "0.2",
            "--summary-output",
            str(resumed_summary_path),
            "--checkpoint-output",
            str(resumed_checkpoint_path),
        ]
    )
    resumed_stdout = capsys.readouterr().out

    first_summary = json.loads(first_summary_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    resumed_summary = json.loads(resumed_summary_path.read_text(encoding="utf-8"))
    resumed_checkpoint = json.loads(resumed_checkpoint_path.read_text(encoding="utf-8"))
    assert first_status == 0
    assert resumed_status == 0
    assert f"checkpoint={checkpoint_path}" in first_stdout
    assert f"resume_checkpoint={checkpoint_path}" in resumed_stdout
    assert f"checkpoint={resumed_checkpoint_path}" in resumed_stdout
    assert checkpoint["schema"] == SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA
    assert checkpoint["training_schema"] == SOLVER_DECODER_JOINT_TRAINING_SCHEMA
    assert checkpoint["gpu_policy"]["vram_reserve_gb"] == 1
    assert resumed_summary["assignment_source"] == "solver_decoder_joint_checkpoint_resume"
    assert resumed_summary["resume_checkpoint"] == str(checkpoint_path)
    assert resumed_summary["initial_solver_state"]["step"] == first_summary["final_solver_state"]["step"]
    assert resumed_summary["final_solver_state"]["step"] == first_summary["final_solver_state"]["step"] + 2
    assert resumed_summary["initial_decoder_state"]["step"] == first_summary["final_decoder_state"]["step"]
    assert resumed_summary["final_decoder_state"]["step"] == first_summary["final_decoder_state"]["step"] + 2
    assert resumed_checkpoint["source"]["resume_checkpoint"] == str(checkpoint_path)
    assert validate_solver_decoder_joint_checkpoint(resumed_checkpoint) == resumed_checkpoint


def test_solver_decoder_mvp_cli_writes_scaled_run_outputs(tmp_path, capsys):
    input_path = tmp_path / "objects.ply"
    run_dir = tmp_path / "scaled-run"
    write_ply(input_path, _object_cloud(), fmt="ascii")

    status = main(
        [
            "training",
            "solver-decoder-mvp",
            str(input_path),
            "--max-points",
            "4",
            "--image-width",
            "8",
            "--image-height",
            "8",
            "--iterations",
            "4",
            "--checkpoint-every",
            "2",
            "--loss-log-every",
            "1",
            "--solver-learning-rate",
            "0.08",
            "--decoder-learning-rate",
            "0.6",
            "--object-weight",
            "0.2",
            "--run-output-dir",
            str(run_dir),
            "--require-loss-decrease",
            "--require-image-render-loss-decrease",
        ]
    )

    stdout = capsys.readouterr().out
    plan = json.loads((run_dir / "training-scale-plan.json").read_text(encoding="utf-8"))
    final_summary = json.loads((run_dir / "final-summary.json").read_text(encoding="utf-8"))
    final_checkpoint = json.loads((run_dir / "final-checkpoint.json").read_text(encoding="utf-8"))
    boundary = json.loads((run_dir / "renderer-loss-boundary.json").read_text(encoding="utf-8"))
    segment_checkpoint = run_dir / "segments" / "segment-0002-checkpoint.json"
    assert status == 0
    assert f"run_output_dir={run_dir}" in stdout
    assert "training_scale_segments=2" in stdout
    assert "training_scale_total_iterations=4" in stdout
    assert "run_loss_decreased=true" in stdout
    assert plan["schema"] == TRAINING_SCALE_PLAN_SCHEMA
    assert plan["segment_count"] == 2
    assert plan["checkpoint_every"] == 2
    assert plan["loss_log_every"] == 1
    assert final_summary["training_scale"]["segment_count"] == 2
    assert final_summary["run_loss"]["loss_decreased"] is True
    assert final_summary["run_loss"]["image_render_loss_decreased"] is True
    assert final_checkpoint["schema"] == SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA
    assert boundary["status"] == "solver_decoder_joint_training_ready"
    assert segment_checkpoint.exists()


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
    xyz = np.array(
        [
            [-1.0, 0.0, 0.0],
            [-0.9, 0.1, 0.0],
            [-1.1, -0.1, 0.0],
            [0.9, 0.0, 0.0],
            [1.0, 0.1, 0.0],
            [1.1, -0.1, 0.0],
        ],
        dtype=np.float32,
    )
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    vertices["red"][:3] = 235
    vertices["green"][:3] = 48
    vertices["blue"][:3] = 32
    vertices["red"][3:] = 20
    vertices["green"][3:] = 178
    vertices["blue"][3:] = 220
    vertices["opacity"] = 1.0
    vertices["object_id"][:3] = 5
    vertices["object_id"][3:] = 9
    return GaussianCloud(vertices=vertices, source_format="fixture")
