from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from objgauss.cli import main
from objgauss.core.gaussian import GaussianCloud
from objgauss.core.gaussian_decoder_training import ObjectStateGaussianDecoderState
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
from objgauss.core.training_tensorboard import (
    TENSORBOARD_SCALAR_EXPORT_SCHEMA,
    write_solver_decoder_tensorboard_events,
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


def test_solver_decoder_joint_checkpoint_preserves_decoder_renderer_field_contract():
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=6,
        bind_image_targets=True,
        image_width=8,
        image_height=8,
        seed=2,
    )
    initial_decoder_state = ObjectStateGaussianDecoderState(
        object_colors=np.array(
            [
                [0.2, 0.3, 0.4],
                [0.6, 0.7, 0.8],
            ],
            dtype=np.float32,
        ),
        object_opacity_logits=np.array([-0.5, 0.5], dtype=np.float32),
        object_scale_log_offsets=np.log(np.array([0.9, 1.1], dtype=np.float32)).astype(np.float32),
        step=9,
        source="fixture_opacity_contract",
    )

    result = train_solver_decoder_joint(
        sample.frames,
        initial_decoder_state=initial_decoder_state,
        iterations=2,
        solver_learning_rate=0.08,
        decoder_learning_rate=0.6,
        object_weight=0.2,
        seed=7,
    )
    checkpoint = solver_decoder_joint_checkpoint(
        result,
        input_path="fixture://objects",
        source_gaussians=sample.source_count,
        sampled_gaussians=sample.sampled_count,
        target_source=sample.target_source,
        assignment_source="object_id_one_hot_targets",
        object_id_mapping=sample.object_id_mapping,
        vram_reserve_gb=1,
    )
    _restored_solver, restored_decoder = solver_decoder_joint_states_from_dict(checkpoint)

    assert checkpoint["decoder_state"]["object_opacity_logits"] == [-0.5, 0.5]
    assert checkpoint["decoder_state"]["opacity_policy"] == "object-opacity-soft-assignment-v1"
    assert checkpoint["decoder_state"]["scale_policy"] == "object-scale-soft-assignment-v1"
    np.testing.assert_allclose(
        checkpoint["decoder_state"]["object_scale_multipliers"],
        [0.9, 1.1],
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.final_decoder_state.object_opacity_logits,
        initial_decoder_state.object_opacity_logits,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        result.final_decoder_state.object_scale_log_offsets,
        initial_decoder_state.object_scale_log_offsets,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        restored_decoder.object_opacity_logits,
        initial_decoder_state.object_opacity_logits,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        restored_decoder.object_scale_log_offsets,
        initial_decoder_state.object_scale_log_offsets,
        atol=1e-6,
    )


def test_solver_decoder_joint_training_updates_decoder_opacity_when_enabled():
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=6,
        bind_image_targets=True,
        image_width=8,
        image_height=8,
        seed=2,
    )
    initial_decoder_state = ObjectStateGaussianDecoderState(
        object_colors=np.array(
            [
                [0.2, 0.3, 0.4],
                [0.6, 0.7, 0.8],
            ],
            dtype=np.float32,
        ),
        object_opacity_logits=np.zeros(2, dtype=np.float32),
        step=0,
        source="fixture_opacity_training",
    )

    result = train_solver_decoder_joint(
        sample.frames,
        initial_decoder_state=initial_decoder_state,
        iterations=4,
        solver_learning_rate=0.08,
        decoder_learning_rate=0.3,
        train_decoder_opacity=True,
        decoder_opacity_learning_rate=1.0,
        object_weight=0.2,
        seed=7,
    )
    payload = result.as_dict()
    checkpoint = solver_decoder_joint_checkpoint(
        result,
        input_path="fixture://objects",
        source_gaussians=sample.source_count,
        sampled_gaussians=sample.sampled_count,
        target_source=sample.target_source,
        assignment_source="object_id_one_hot_targets",
        object_id_mapping=sample.object_id_mapping,
        vram_reserve_gb=1,
    )

    assert result.train_decoder_opacity is True
    assert "decoder.object_opacity_logits" in payload["trained_fields"]
    assert "source_opacities" in payload["frozen_fields"]
    assert payload["decoder_opacity"]["enabled"] is True
    assert payload["renderer_api"]["gradients"]["decoder_opacity_logits_shape"] == [2]
    assert payload["renderer_api"]["gradients"]["decoder_opacity_logits_l2"] > 0.0
    assert not np.allclose(
        result.final_decoder_state.object_opacity_logits,
        initial_decoder_state.object_opacity_logits,
    )
    assert checkpoint["training"]["train_decoder_opacity"] is True
    assert checkpoint["training"]["learning_rates"]["decoder_opacity"] == 1.0
    assert "decoder.object_opacity_logits" in checkpoint["trained_fields"]


def test_solver_decoder_joint_training_can_override_solver_temperature():
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=6,
        bind_image_targets=True,
        image_width=8,
        image_height=8,
        seed=2,
    )

    result = train_solver_decoder_joint(
        sample.frames,
        iterations=2,
        solver_learning_rate=0.08,
        decoder_learning_rate=0.6,
        object_weight=0.2,
        solver_temperature=0.5,
        seed=7,
    )
    checkpoint = solver_decoder_joint_checkpoint(
        result,
        input_path="fixture://objects",
        source_gaussians=sample.source_count,
        sampled_gaussians=sample.sampled_count,
        target_source=sample.target_source,
        assignment_source="object_id_one_hot_targets",
        object_id_mapping=sample.object_id_mapping,
        vram_reserve_gb=1,
    )

    assert result.initial_solver_state.config.temperature == 0.5
    assert result.final_solver_state.config.temperature == 0.5
    assert checkpoint["solver_state"]["config"]["temperature"] == 0.5


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
            "--solver-temperature",
            "0.75",
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
    assert payload["final_solver_state"]["config"]["temperature"] == 0.75
    assert "solver_temperature=0.75" in stdout


def test_solver_decoder_mvp_cli_trains_decoder_opacity_when_enabled(tmp_path, capsys):
    input_path = tmp_path / "objects.ply"
    summary_path = tmp_path / "joint-summary.json"
    checkpoint_path = tmp_path / "joint-checkpoint.json"
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
            "3",
            "--solver-learning-rate",
            "0.08",
            "--decoder-learning-rate",
            "0.4",
            "--train-decoder-opacity",
            "--decoder-opacity-learning-rate",
            "1.0",
            "--decoder-opacity-init-logit",
            "0.0",
            "--object-weight",
            "0.2",
            "--summary-output",
            str(summary_path),
            "--checkpoint-output",
            str(checkpoint_path),
        ]
    )

    stdout = capsys.readouterr().out
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    assert status == 0
    assert "train_decoder_opacity=true" in stdout
    assert "decoder_opacity_learning_rate=1.0" in stdout
    assert "decoder.object_opacity_logits" in stdout
    assert payload["train_decoder_opacity"] is True
    assert payload["decoder_opacity"]["enabled"] is True
    assert payload["renderer_api"]["gradients"]["decoder_opacity_logits_shape"] == [2]
    assert "decoder.object_opacity_logits" in payload["trained_fields"]
    assert "source_opacities" in payload["frozen_fields"]
    assert checkpoint["decoder_state"]["object_opacity_logits"] is not None
    assert checkpoint["training"]["train_decoder_opacity"] is True


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


def test_solver_decoder_tensorboard_export_writes_scalar_tags(tmp_path):
    writer = _FakeTensorBoardWriter(tmp_path / "tb")
    summary = {
        "training_scale": {
            "total_iterations": 4,
            "segments": [
                {
                    "start_iteration": 0,
                    "end_iteration": 2,
                    "initial_total_loss": 0.4,
                    "final_total_loss": 0.3,
                    "initial_image_render_loss": 0.2,
                    "final_image_render_loss": 0.15,
                    "initial_object_loss": 1.0,
                    "final_object_loss": 0.9,
                    "final_entropy_loss": 0.8,
                    "final_balance_loss": 0.01,
                    "final_decoder_opacity_scale_min": 0.91,
                    "final_decoder_opacity_scale_mean": 0.94,
                    "final_decoder_opacity_scale_max": 0.98,
                },
                {
                    "start_iteration": 2,
                    "end_iteration": 4,
                    "initial_total_loss": 0.3,
                    "final_total_loss": 0.25,
                    "initial_image_render_loss": 0.15,
                    "final_image_render_loss": 0.12,
                    "initial_object_loss": 0.9,
                    "final_object_loss": 0.85,
                    "final_entropy_loss": 0.7,
                    "final_balance_loss": 0.02,
                    "final_decoder_opacity_scale_min": 0.92,
                    "final_decoder_opacity_scale_mean": 0.95,
                    "final_decoder_opacity_scale_max": 0.99,
                },
            ],
        },
        "run_loss": {"final_total_loss": 0.25},
    }

    export = write_solver_decoder_tensorboard_events(
        summary,
        tmp_path / "tb",
        writer_factory=lambda _logdir: writer,
    )

    assert export["schema"] == TENSORBOARD_SCALAR_EXPORT_SCHEMA
    assert export["segment_count"] == 2
    assert export["scalar_count"] == len(writer.scalars)
    assert ("loss/total", 0.4, 0) in writer.scalars
    assert ("loss/total", 0.25, 4) in writer.scalars
    assert ("loss/image_render", 0.12, 4) in writer.scalars
    assert ("loss/object", 0.85, 4) in writer.scalars
    assert ("loss/entropy", 0.7, 4) in writer.scalars
    assert ("decoder/opacity_scale_min", 0.92, 4) in writer.scalars
    assert ("decoder/opacity_scale_mean", 0.95, 4) in writer.scalars
    assert ("decoder/opacity_scale_max", 0.99, 4) in writer.scalars
    assert ("run/final_total_loss", 0.25, 4) in writer.scalars
    assert writer.flushed is True
    assert writer.closed is True


def test_solver_decoder_mvp_cli_writes_tensorboard_metadata(tmp_path, capsys, monkeypatch):
    input_path = tmp_path / "objects.ply"
    run_dir = tmp_path / "scaled-run"
    tensorboard_logdir = run_dir / "tensorboard"
    write_ply(input_path, _object_cloud(), fmt="ascii")

    def fake_export(summary, logdir):
        assert summary["training_scale"]["segment_count"] == 2
        Path(logdir).mkdir(parents=True, exist_ok=True)
        (Path(logdir) / "events.out.tfevents.fake").write_text("fake\n", encoding="utf-8")
        return {
            "schema": TENSORBOARD_SCALAR_EXPORT_SCHEMA,
            "kind": "solver_decoder_tensorboard_scalars",
            "logdir": str(logdir),
            "segment_count": 2,
            "scalar_count": 12,
            "tags": ["loss/total"],
        }

    monkeypatch.setattr("objgauss.cli.write_solver_decoder_tensorboard_events", fake_export)
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
            "--tensorboard-logdir",
            str(tensorboard_logdir),
        ]
    )

    stdout = capsys.readouterr().out
    final_summary = json.loads((run_dir / "final-summary.json").read_text(encoding="utf-8"))
    assert status == 0
    assert f"tensorboard_logdir={tensorboard_logdir}" in stdout
    assert "tensorboard_scalar_count=12" in stdout
    assert final_summary["tensorboard"]["schema"] == TENSORBOARD_SCALAR_EXPORT_SCHEMA
    assert final_summary["tensorboard"]["logdir"] == str(tensorboard_logdir)
    assert (tensorboard_logdir / "events.out.tfevents.fake").exists()


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


class _FakeTensorBoardWriter:
    def __init__(self, logdir):
        self.logdir = logdir
        self.scalars = []
        self.flushed = False
        self.closed = False

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalars.append((tag, float(scalar_value), int(global_step)))

    def flush(self):
        self.flushed = True

    def close(self):
        self.closed = True
