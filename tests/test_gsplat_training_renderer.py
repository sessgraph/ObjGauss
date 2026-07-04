from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.gsplat_training_renderer import (
    GSPLAT_GRADIENT_PATH,
    GSPLAT_RENDERER,
    GSPLAT_SYNTHETIC_GAUSSIAN_POLICY,
    build_gsplat_training_input,
    build_gsplat_training_input_from_object_state,
    evaluate_gsplat_training_renderer_loss,
    gsplat_renderer_availability,
)
from objgauss.core.object_state import project_object_states
from objgauss.core.trainable_kernel import (
    bind_image_targets_to_frames,
    make_trainable_kernel_mvp_fixture,
)


def test_gsplat_availability_reports_missing_optional_dependencies():
    availability = gsplat_renderer_availability(_importer=_missing_importer)
    payload = availability.as_dict()

    assert availability.available is False
    assert payload["schema"] == "objgauss-gsplat-training-renderer-availability-v1"
    assert payload["renderer_name"] == GSPLAT_RENDERER
    assert payload["gradient_path"] == GSPLAT_GRADIENT_PATH
    assert "optional_dependency_missing:torch" in payload["blockers"]
    assert "optional_dependency_missing:gsplat" in payload["blockers"]


def test_build_gsplat_training_input_maps_assignment_to_gaussian_state():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    assignment = np.zeros((6, 2), dtype=np.float32)
    assignment[:3, 0] = 1.0
    assignment[3:, 1] = 1.0
    decoder_colors = np.vstack([frames[0].target_rgb[0], frames[0].target_rgb[3]]).astype(np.float32)

    record = build_gsplat_training_input(
        frames[0],
        assignment,
        decoder_colors,
    )
    payload = record.as_dict()

    assert payload["schema"] == "objgauss-gsplat-training-input-v1"
    assert payload["renderer_name"] == GSPLAT_RENDERER
    assert payload["gaussian_policy"] == GSPLAT_SYNTHETIC_GAUSSIAN_POLICY
    assert payload["decoder_schema"] == "objgauss-object-state-gaussian-decode-v1"
    assert payload["object_state_slots"] == 2
    assert payload["shapes"]["means"] == [6, 3]
    assert payload["shapes"]["quats"] == [6, 4]
    assert payload["shapes"]["scales"] == [6, 3]
    assert payload["shapes"]["colors"] == [6, 3]
    assert payload["shapes"]["viewmats"] == [1, 4, 4]
    assert payload["shapes"]["Ks"] == [1, 3, 3]
    assert payload["visibility_coverage"] > 0
    np.testing.assert_allclose(record.colors[:3], np.tile(decoder_colors[0], (3, 1)), atol=1e-6)
    np.testing.assert_allclose(record.colors[3:], np.tile(decoder_colors[1], (3, 1)), atol=1e-6)
    np.testing.assert_allclose(record.quats[:, 0], np.ones(6, dtype=np.float32), atol=1e-6)
    np.testing.assert_allclose(record.opacities, np.ones(6, dtype=np.float32), atol=1e-6)


def test_build_gsplat_training_input_accepts_decoder_opacity_logits():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    assignment = np.zeros((6, 2), dtype=np.float32)
    assignment[:3, 0] = 1.0
    assignment[3:, 1] = 1.0
    decoder_colors = np.vstack([frames[0].target_rgb[0], frames[0].target_rgb[3]]).astype(np.float32)

    record = build_gsplat_training_input(
        frames[0],
        assignment,
        decoder_colors,
        decoder_opacity_logits=np.array([-2.0, 2.0], dtype=np.float32),
        default_opacity=0.8,
    )

    assert record.opacities[:3].mean() < record.opacities[3:].mean()
    assert np.all(record.opacities >= 0.0)
    assert np.all(record.opacities <= 0.8)


def test_build_gsplat_training_input_accepts_decoder_scale_log_offsets():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    assignment = np.zeros((6, 2), dtype=np.float32)
    assignment[:3, 0] = 1.0
    assignment[3:, 1] = 1.0
    decoder_colors = np.vstack([frames[0].target_rgb[0], frames[0].target_rgb[3]]).astype(np.float32)

    record = build_gsplat_training_input(
        frames[0],
        assignment,
        decoder_colors,
        decoder_scale_log_offsets=np.log(np.array([0.8, 1.2], dtype=np.float32)),
        default_scale=0.5,
    )

    assert record.scales[:3].mean() == pytest.approx(0.4, abs=1e-6)
    assert record.scales[3:].mean() == pytest.approx(0.6, abs=1e-6)
    assert np.all(record.scales >= 0.5 * 0.75)
    assert np.all(record.scales <= 0.5 * 1.25)


def test_build_gsplat_training_input_accepts_object_state_projection():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    assignment = np.zeros((6, 2), dtype=np.float32)
    assignment[:3, 0] = 1.0
    assignment[3:, 1] = 1.0
    projection = project_object_states(
        _frame_cloud(frames[0].positions),
        assignment,
        evidence_features=frames[0].features,
    )
    decoder_colors = np.vstack([frames[0].target_rgb[0], frames[0].target_rgb[3]]).astype(np.float32)

    record = build_gsplat_training_input_from_object_state(
        frames[0],
        projection,
        decoder_colors,
    )

    assert record.decoder_schema == "objgauss-object-state-gaussian-decode-v1"
    assert record.object_state_slots == 2
    assert record.gaussian_policy == GSPLAT_SYNTHETIC_GAUSSIAN_POLICY
    np.testing.assert_allclose(record.colors[:3], np.tile(decoder_colors[0], (3, 1)), atol=1e-6)
    np.testing.assert_allclose(record.colors[3:], np.tile(decoder_colors[1], (3, 1)), atol=1e-6)


def test_build_gsplat_training_input_requires_bound_image_target():
    frames = make_trainable_kernel_mvp_fixture()
    assignment = np.full((6, 2), 0.5, dtype=np.float32)
    decoder_colors = np.zeros((2, 3), dtype=np.float32)

    with pytest.raises(ValueError, match="frame.image_target"):
        build_gsplat_training_input(frames[0], assignment, decoder_colors)


def test_evaluate_gsplat_training_renderer_loss_fails_with_clear_blockers_when_unavailable():
    frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=8, height=8)
    assignment = np.zeros((6, 2), dtype=np.float32)
    assignment[:3, 0] = 1.0
    assignment[3:, 1] = 1.0
    decoder_colors = np.vstack([frames[0].target_rgb[0], frames[0].target_rgb[3]]).astype(np.float32)

    with pytest.raises(RuntimeError, match="optional_dependency_missing:torch"):
        evaluate_gsplat_training_renderer_loss(
            frames[:1],
            [assignment],
            decoder_colors,
            _importer=_missing_importer,
        )


def _missing_importer(name: str):
    raise ImportError(name)


def _frame_cloud(positions: np.ndarray):
    vertices = np.zeros(
        positions.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ],
    )
    vertices["x"] = positions[:, 0]
    vertices["y"] = positions[:, 1]
    vertices["z"] = positions[:, 2]
    from objgauss.core.gaussian import GaussianCloud

    return GaussianCloud(vertices=vertices, source_format="fixture")
