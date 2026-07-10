from __future__ import annotations

import pytest
import numpy as np

from objgauss.pipelines.renderer_loss import (
    RENDERER_LOSS_BOUNDARY_SCHEMA,
    RendererLossBoundaryReport,
    renderer_loss_boundary_report,
    validate_renderer_loss_boundary_summary,
)
from objgauss.core.object_emergence_solver import (
    ObjectEmergenceEvidence,
    object_emergence_solver_checkpoint,
    train_object_emergence_solver,
)
from objgauss.pipelines.trainable_kernel import (
    bind_image_targets_to_frames,
    make_trainable_kernel_mvp_fixture,
    train_kernel_mvp,
)
from objgauss.pipelines.gaussian_decoder_training import train_object_state_gaussian_decoder
from objgauss.pipelines.solver_decoder_training import (
    solver_decoder_joint_checkpoint,
    train_solver_decoder_joint,
)
from objgauss.pipelines.training_renderer import evaluate_training_renderer_loss


def test_renderer_loss_boundary_accepts_trainable_kernel_summary():
    result = train_kernel_mvp(
        make_trainable_kernel_mvp_fixture(),
        slots=2,
        iterations=12,
        learning_rate=0.4,
        seed=4,
    )

    report = renderer_loss_boundary_report(result.as_dict())
    payload = report.as_dict()

    assert isinstance(report, RendererLossBoundaryReport)
    assert payload["schema"] == RENDERER_LOSS_BOUNDARY_SCHEMA
    assert payload["status"] == "point_render_smoke_ready"
    assert payload["point_smoke_ready"] is True
    assert payload["current_renderer"] == "cpu-point-rgb-smoke"
    assert payload["target_renderer"] == "differentiable-gaussian-image-renderer"
    assert payload["point_smoke_blockers"] == []
    assert "image_space_targets_not_bound" in payload["upgrade_blockers"]
    assert payload["render_target_contract"]["current"]["kind"] == "point_rgb_rows"
    assert payload["render_target_contract"]["target"]["kind"] == "image_space_render"
    assert payload["decoder_handoff_contract"]["schema"] == "objgauss-decoder-renderer-handoff-v1"
    assert payload["decoder_handoff_contract"]["status"] == "awaiting_solver_checkpoint"
    assert payload["decoder_handoff_contract"]["starts_real_training"] is False
    assert validate_renderer_loss_boundary_summary(payload) is True


def test_renderer_loss_boundary_clears_image_target_blockers_when_bound():
    result = train_kernel_mvp(
        bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8),
        slots=2,
        iterations=12,
        learning_rate=0.4,
        seed=6,
    )

    report = renderer_loss_boundary_report(result.as_dict())
    payload = report.as_dict()

    assert payload["point_smoke_ready"] is True
    assert payload["evidence"]["image_targets_bound"] is True
    assert "image_space_targets_not_bound" not in payload["upgrade_blockers"]
    assert "camera_visibility_policy_not_bound" not in payload["upgrade_blockers"]
    assert "differentiable_gaussian_renderer_not_selected" in payload["upgrade_blockers"]


def test_renderer_loss_boundary_accepts_renderer_api_gradient_path():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    result = train_kernel_mvp(
        frames,
        slots=2,
        iterations=12,
        learning_rate=0.4,
        seed=8,
    )
    summary = result.as_dict()
    summary["renderer_api"] = evaluate_training_renderer_loss(
        frames,
        result.assignments,
        result.decoder_colors,
    ).as_dict()

    report = renderer_loss_boundary_report(summary)
    payload = report.as_dict()

    assert payload["status"] == "renderer_api_ready"
    assert payload["current_renderer"] == "cpu-image-point-splat-differentiable-v1"
    assert payload["evidence"]["renderer_api_ready"] is True
    assert payload["evidence"]["renderer_gradient_path"] == "analytic-color-assignment-gradient-v1"
    assert payload["evidence"]["final_image_render_loss"] >= 0
    assert payload["decoder_handoff_contract"]["status"] == "renderer_api_decoder_smoke_ready"
    assert payload["decoder_handoff_contract"]["starts_real_training"] is False
    assert "renderer_gradient_path_not_defined" not in payload["upgrade_blockers"]
    assert "differentiable_gaussian_renderer_not_selected" not in payload["upgrade_blockers"]
    assert "full_3dgs_renderer_not_selected" in payload["upgrade_blockers"]


def test_renderer_loss_boundary_accepts_full_gsplat_renderer_evidence():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    result = train_kernel_mvp(
        frames,
        slots=2,
        iterations=8,
        learning_rate=0.4,
        seed=10,
    )
    summary = result.as_dict()
    renderer_api = evaluate_training_renderer_loss(
        frames,
        result.assignments,
        result.decoder_colors,
    ).as_dict()
    renderer_api["renderer_name"] = "gsplat-rasterization-v1"
    renderer_api["gradient_path"] = "torch-autograd-gsplat-rasterization-v1"
    summary["renderer_api"] = renderer_api

    payload = renderer_loss_boundary_report(summary).as_dict()

    assert payload["status"] == "full_3dgs_renderer_ready"
    assert payload["current_renderer"] == "gsplat-rasterization-v1"
    assert payload["decoder_handoff_contract"]["status"] == "full_renderer_decoder_ready"
    assert payload["decoder_handoff_contract"]["remaining_before_full_training"] == []
    assert "full_3dgs_renderer_not_selected" not in payload["upgrade_blockers"]
    assert "differentiable_gaussian_renderer_not_selected" not in payload["upgrade_blockers"]
    assert "renderer_gradient_path_not_defined" not in payload["upgrade_blockers"]


def test_renderer_loss_boundary_accepts_object_emergence_solver_checkpoint():
    result = train_object_emergence_solver(
        [
            ObjectEmergenceEvidence(
                positions=np.array(
                    [[-1.0, 0.0, 0.0], [-0.8, 0.0, 0.0], [0.8, 0.0, 0.0], [1.0, 0.0, 0.0]],
                    dtype="float32",
                ),
                features=np.array(
                    [[1.0, 0.0], [0.8, 0.2], [0.2, 0.8], [0.0, 1.0]],
                    dtype="float32",
                ),
                target_assignment=np.array(
                    [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
                    dtype="float32",
                ),
            )
        ],
        iterations=8,
        learning_rate=0.5,
        assignment_weight=1.0,
        entropy_weight=0.0,
        balance_weight=0.0,
        temporal_weight=0.0,
        seed=3,
    )
    checkpoint = object_emergence_solver_checkpoint(
        result,
        input_path="fixture://solver",
        source_gaussians=4,
        sampled_gaussians=4,
        target_source="object_id_one_hot_targets",
    )

    payload = renderer_loss_boundary_report(checkpoint).as_dict()

    assert payload["status"] == "object_emergence_solver_ready"
    assert payload["point_smoke_ready"] is False
    assert "point_render_smoke_not_present" in payload["point_smoke_blockers"]
    assert payload["evidence"]["kind"] == "object_emergence_solver_checkpoint"
    assert payload["evidence"]["solver_loss_decreased"] is True
    assert payload["evidence"]["assignment_loss_decreased"] is True
    assert payload["evidence"]["gpu_used"] is False
    assert payload["evidence"]["vram_reserve_gb"] == 1
    decoder_handoff = payload["decoder_handoff_contract"]
    assert decoder_handoff["schema"] == "objgauss-decoder-renderer-handoff-v1"
    assert decoder_handoff["status"] == "solver_checkpoint_ready"
    assert decoder_handoff["source_evidence"] == "object_emergence_solver_checkpoint"
    assert decoder_handoff["ready_without_gpu"] is True
    assert decoder_handoff["starts_real_training"] is False
    assert "GaussianToken decode" in decoder_handoff["state_chain"]
    assert "renderer_api image_render_loss" in decoder_handoff["state_chain"]
    assert "bind solver checkpoint output to Gaussian decoder parameters" in (
        decoder_handoff["remaining_before_full_training"]
    )
    assert decoder_handoff["object_state_input_contract"]["assignment"] == "float32[N,K] row-normalized"
    assert "decode_gaussian" in decoder_handoff["gaussian_decoder_contract"]["function"]
    assert "solver_checkpoint_not_bound_to_renderer_loss" in payload["upgrade_blockers"]
    assert "solver_checkpoint_not_bound_to_gaussian_decoder" in payload["upgrade_blockers"]


def test_renderer_loss_boundary_accepts_decoder_training_summary():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    assignment = np.zeros((6, 2), dtype=np.float32)
    assignment[:3, 0] = 1.0
    assignment[3:, 1] = 1.0
    result = train_object_state_gaussian_decoder(
        frames,
        [assignment, assignment],
        iterations=8,
        learning_rate=0.7,
        seed=5,
    )

    payload = renderer_loss_boundary_report(result.as_dict()).as_dict()

    assert payload["status"] == "object_state_decoder_training_ready"
    assert payload["point_smoke_ready"] is False
    assert payload["evidence"]["kind"] == "object_state_gaussian_decoder_training"
    assert payload["evidence"]["loss_decreased"] is True
    assert payload["evidence"]["image_render_loss_decreased"] is True
    assert payload["evidence"]["trained_fields"] == ["object_colors"]
    assert payload["decoder_handoff_contract"]["status"] == "decoder_training_ready"
    assert payload["decoder_handoff_contract"]["starts_real_training"] is True
    assert "full_3dgs_renderer_not_selected" in payload["upgrade_blockers"]


def test_renderer_loss_boundary_accepts_solver_decoder_joint_summary():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    target = np.zeros((6, 2), dtype=np.float32)
    target[:3, 0] = 1.0
    target[3:, 1] = 1.0
    frames = tuple(
        type(frame)(
            positions=frame.positions,
            features=frame.features,
            target_rgb=frame.target_rgb,
            target_assignment=target,
            image_target=frame.image_target,
        )
        for frame in frames
    )
    result = train_solver_decoder_joint(
        frames,
        iterations=4,
        solver_learning_rate=0.08,
        decoder_learning_rate=0.6,
        object_weight=0.2,
        seed=3,
    )

    payload = renderer_loss_boundary_report(result.as_dict()).as_dict()

    assert payload["status"] == "solver_decoder_joint_training_ready"
    assert payload["evidence"]["kind"] == "solver_decoder_joint_training"
    assert payload["evidence"]["loss_decreased"] is True
    assert payload["evidence"]["image_render_loss_decreased"] is True
    assert "solver.feature_weights" in payload["evidence"]["trained_fields"]
    assert payload["decoder_handoff_contract"]["status"] == "solver_decoder_joint_training_ready"
    assert payload["decoder_handoff_contract"]["starts_real_training"] is True


def test_renderer_loss_boundary_uses_segmented_run_loss_for_joint_summary():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    target = np.zeros((6, 2), dtype=np.float32)
    target[:3, 0] = 1.0
    target[3:, 1] = 1.0
    frames = tuple(
        type(frame)(
            positions=frame.positions,
            features=frame.features,
            target_rgb=frame.target_rgb,
            target_assignment=target,
            image_target=frame.image_target,
        )
        for frame in frames
    )
    result = train_solver_decoder_joint(
        frames,
        iterations=4,
        solver_learning_rate=0.08,
        decoder_learning_rate=0.6,
        object_weight=0.2,
        seed=9,
    )
    summary = result.as_dict()
    summary["renderer_api"]["renderer_name"] = "gsplat-rasterization-v1"
    summary["renderer_api"]["gradient_path"] = "torch-autograd-gsplat-rasterization-v1"
    summary["final_loss"]["image_render_loss"] = summary["initial_loss"]["image_render_loss"]
    summary["image_render_loss_decreased"] = False
    summary["run_loss"] = {
        "initial_total_loss": summary["initial_loss"]["total_loss"] + 0.1,
        "final_total_loss": summary["final_loss"]["total_loss"],
        "initial_image_render_loss": summary["initial_loss"]["image_render_loss"] + 0.01,
        "final_image_render_loss": summary["final_loss"]["image_render_loss"],
        "initial_object_loss": summary["initial_loss"]["object_loss"] + 0.1,
        "final_object_loss": summary["final_loss"]["object_loss"],
        "loss_decreased": True,
        "image_render_loss_decreased": True,
        "object_loss_decreased": True,
    }

    payload = renderer_loss_boundary_report(summary).as_dict()

    assert payload["status"] == "full_3dgs_solver_decoder_joint_training_ready"
    assert payload["point_smoke_blockers"] == []
    assert payload["evidence"]["loss_delta_source"] == "run_loss"
    assert payload["evidence"]["image_render_loss_decreased"] is True
    assert payload["evidence"]["initial_image_render_loss"] == summary["run_loss"]["initial_image_render_loss"]
    assert payload["evidence"]["segment_initial_image_render_loss"] == summary["initial_loss"]["image_render_loss"]
    assert payload["evidence"]["segment_final_image_render_loss"] == summary["final_loss"]["image_render_loss"]
    assert payload["decoder_handoff_contract"]["status"] == "full_renderer_solver_decoder_joint_training_ready"


def test_renderer_loss_boundary_accepts_solver_decoder_joint_checkpoint():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    target = np.zeros((6, 2), dtype=np.float32)
    target[:3, 0] = 1.0
    target[3:, 1] = 1.0
    frames = tuple(
        type(frame)(
            positions=frame.positions,
            features=frame.features,
            target_rgb=frame.target_rgb,
            target_assignment=target,
            image_target=frame.image_target,
        )
        for frame in frames
    )
    result = train_solver_decoder_joint(
        frames,
        iterations=4,
        solver_learning_rate=0.08,
        decoder_learning_rate=0.6,
        object_weight=0.2,
        seed=6,
    )
    checkpoint = solver_decoder_joint_checkpoint(
        result,
        input_path="fixture://joint",
        source_gaussians=6,
        sampled_gaussians=6,
        target_source="object_id_one_hot_targets",
        assignment_source="object_id_one_hot_targets",
    )

    payload = renderer_loss_boundary_report(checkpoint).as_dict()

    assert payload["status"] == "solver_decoder_joint_training_ready"
    assert payload["evidence"]["kind"] == "solver_decoder_joint_checkpoint"
    assert payload["evidence"]["training_schema"] == "objgauss-solver-decoder-joint-training-v1"
    assert payload["evidence"]["loss_decreased"] is True
    assert payload["evidence"]["image_render_loss_decreased"] is True
    assert payload["evidence"]["source_count"] == 6
    assert payload["decoder_handoff_contract"]["source_evidence"] == "solver_decoder_joint_checkpoint"
    assert payload["decoder_handoff_contract"]["status"] == "solver_decoder_joint_training_ready"
    assert "run resume/load smoke from solver + decoder joint checkpoint" in payload["next_steps"]


def test_renderer_loss_boundary_marks_missing_summary_as_contract_only():
    report = renderer_loss_boundary_report()

    assert report.status == "contract_defined"
    assert report.point_smoke_ready is False
    assert report.point_smoke_blockers == ("missing_kernel_summary",)


def test_renderer_loss_boundary_rejects_incomplete_loss_telemetry():
    payload = {
        "schema": "objgauss-v1-trainable-kernel-mvp-v1",
        "initial_loss": {"total_loss": 1.0, "render_loss": 1.0, "object_loss": 0.0, "temporal_loss": 0.0},
        "final_loss": {"total_loss": 0.5},
    }

    with pytest.raises(ValueError, match="final_loss missing loss fields"):
        renderer_loss_boundary_report(payload)
