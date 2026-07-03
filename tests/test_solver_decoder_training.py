from __future__ import annotations

import json

import numpy as np

from objgauss.cli import main
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.solver_decoder_training import (
    SOLVER_DECODER_JOINT_TRAINING_SCHEMA,
    SolverDecoderJointTrainingResult,
    train_solver_decoder_joint,
)
from objgauss.core.trainable_kernel import trainable_kernel_sample_from_cloud
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
