from __future__ import annotations

import pytest

from objgauss.core.renderer_loss import (
    RENDERER_LOSS_BOUNDARY_SCHEMA,
    RendererLossBoundaryReport,
    renderer_loss_boundary_report,
    validate_renderer_loss_boundary_summary,
)
from objgauss.core.trainable_kernel import (
    bind_image_targets_to_frames,
    make_trainable_kernel_mvp_fixture,
    train_kernel_mvp,
)
from objgauss.core.training_renderer import evaluate_training_renderer_loss


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
    assert "full_3dgs_renderer_not_selected" not in payload["upgrade_blockers"]
    assert "differentiable_gaussian_renderer_not_selected" not in payload["upgrade_blockers"]
    assert "renderer_gradient_path_not_defined" not in payload["upgrade_blockers"]


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
