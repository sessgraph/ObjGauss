from __future__ import annotations

import json

import numpy as np
import pytest

from objgauss.cli import main
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.gaussian_decoder_training import (
    OBJECT_STATE_GAUSSIAN_DECODER_TRAINING_SCHEMA,
    train_object_state_gaussian_decoder,
)
from objgauss.core.object_emergence_solver import (
    evidence_from_gaussian_cloud,
    object_emergence_solver_checkpoint,
    object_id_targets_from_cloud,
    train_object_emergence_solver,
)
from objgauss.core.trainable_kernel import (
    bind_image_targets_to_frames,
    make_trainable_kernel_mvp_fixture,
)
from objgauss.ply import write_ply


def test_object_state_gaussian_decoder_trains_object_colors_against_image_loss():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    assignment = _fixture_assignment()

    result = train_object_state_gaussian_decoder(
        frames,
        [assignment, assignment],
        iterations=8,
        learning_rate=0.7,
        image_render_weight=1.0,
        seed=11,
    )
    payload = result.as_dict()

    assert result.schema == OBJECT_STATE_GAUSSIAN_DECODER_TRAINING_SCHEMA
    assert result.final_loss.image_render_loss < result.initial_loss.image_render_loss
    assert result.final_loss.total_loss < result.initial_loss.total_loss
    assert payload["decoder_schema"] == "objgauss-object-state-gaussian-decode-v1"
    assert payload["trained_fields"] == ["object_colors"]
    assert "means" in payload["frozen_fields"]
    assert "assignments" in payload["frozen_fields"]
    assert payload["renderer_api"]["status"] == "ready"
    assert payload["gpu_policy"]["uses_gpu"] is False
    assert payload["gpu_policy"]["vram_reserve_gb"] == 1
    assert payload["image_render_loss_decreased"] is True
    assert payload["assignments"][0]["shape"] == [6, 2]


def test_object_state_gaussian_decoder_requires_image_targets_for_image_loss():
    with pytest.raises(ValueError, match="image_render_weight requires"):
        train_object_state_gaussian_decoder(
            make_trainable_kernel_mvp_fixture(),
            [_fixture_assignment(), _fixture_assignment()],
            iterations=2,
            image_render_weight=1.0,
        )


def test_decoder_mvp_cli_writes_summary(tmp_path, capsys):
    input_path = tmp_path / "objects.ply"
    summary_path = tmp_path / "decoder-summary.json"
    write_ply(input_path, _object_cloud(), fmt="ascii")

    status = main(
        [
            "training",
            "decoder-mvp",
            str(input_path),
            "--max-points",
            "4",
            "--image-width",
            "8",
            "--image-height",
            "8",
            "--iterations",
            "8",
            "--learning-rate",
            "0.7",
            "--summary-output",
            str(summary_path),
            "--require-image-render-loss-decrease",
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert status == 0
    assert "schema=objgauss-object-state-gaussian-decoder-training-v1" in stdout
    assert "assignment_source=object_id_one_hot_targets" in stdout
    assert "trained_fields=object_colors" in stdout
    assert "gpu_used=false" in stdout
    assert payload["schema"] == OBJECT_STATE_GAUSSIAN_DECODER_TRAINING_SCHEMA
    assert payload["loss_decreased"] is True
    assert payload["image_render_loss_decreased"] is True
    assert payload["sample"]["sampled_count"] == 4
    assert payload["assignment_source"] == "object_id_one_hot_targets"
    assert payload["trained_fields"] == ["object_colors"]


def test_decoder_mvp_cli_accepts_solver_checkpoint_assignment(tmp_path, capsys):
    input_path = tmp_path / "objects.ply"
    checkpoint_path = tmp_path / "solver-checkpoint.json"
    summary_path = tmp_path / "decoder-summary.json"
    cloud = _object_cloud()
    write_ply(input_path, cloud, fmt="ascii")
    targets, mapping = object_id_targets_from_cloud(cloud)
    solver_result = train_object_emergence_solver(
        [evidence_from_gaussian_cloud(cloud, target_assignment=targets)],
        iterations=2,
        learning_rate=0.4,
        seed=2,
    )
    checkpoint = object_emergence_solver_checkpoint(
        solver_result,
        input_path=str(input_path),
        source_gaussians=cloud.count,
        sampled_gaussians=cloud.count,
        target_source="object_id_one_hot_targets",
        object_id_mapping=mapping,
        vram_reserve_gb=1,
    )
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")

    status = main(
        [
            "training",
            "decoder-mvp",
            str(input_path),
            "--solver-checkpoint",
            str(checkpoint_path),
            "--max-points",
            "4",
            "--image-width",
            "8",
            "--image-height",
            "8",
            "--iterations",
            "2",
            "--learning-rate",
            "0.5",
            "--summary-output",
            str(summary_path),
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert status == 0
    assert "assignment_source=solver_checkpoint" in stdout
    assert payload["assignment_source"] == "solver_checkpoint"
    assert payload["solver_checkpoint"] == str(checkpoint_path)
    assert payload["slots"] == 2


def _fixture_assignment() -> np.ndarray:
    assignment = np.zeros((6, 2), dtype=np.float32)
    assignment[:3, 0] = 1.0
    assignment[3:, 1] = 1.0
    return assignment


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
