from __future__ import annotations

import numpy as np
import pytest

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.io import read_ply
from objgauss.core.real_sample_v2_smoke import (
    REAL_SAMPLE_V2_SMOKE_SCHEMA,
    RealSampleV2SmokeReport,
    evaluate_real_sample_v2_smoke,
    real_sample_v2_smoke_from_cloud,
    validate_real_sample_v2_smoke_summary,
)
from objgauss.core.trainable_kernel import trainable_kernel_sample_from_cloud


def test_real_sample_v2_smoke_runs_object_id_sample_through_renderer_boundary():
    report = real_sample_v2_smoke_from_cloud(
        _object_cloud(),
        sample_source="public/samples/tiny_objects.ply",
        frame_count=2,
        max_points=6,
        image_width=8,
        image_height=8,
        iterations=100,
        learning_rate=0.4,
        seed=4,
    )
    summary = report.as_dict()

    assert isinstance(report, RealSampleV2SmokeReport)
    assert summary["schema"] == REAL_SAMPLE_V2_SMOKE_SCHEMA
    assert summary["status"] == "real_sample_v2_smoke_pass"
    assert summary["sample"]["source_kind"] == "public_sample"
    assert summary["sample"]["target_source"] == "object_id_one_hot_targets"
    assert summary["sample"]["object_id_mapping"] == {"5": 0, "9": 1}
    assert summary["training_loss"]["loss_decreased"] is True
    assert summary["training_loss"]["supervised_loss_decreased"] is True
    assert summary["gates"] == {
        "object_id_targets_bound": True,
        "training_loss_decreased": True,
        "renderer_joint_passed": True,
        "renderer_boundary_ready": True,
        "checkpoint_roundtrip_passed": True,
    }
    assert summary["truth_contract"] == {
        "target_source": "object_id_one_hot_targets",
        "object_id_labels_are_training_targets": True,
        "semantic_ground_truth_claimed": False,
        "fixture_oracle_claimed": False,
    }
    assert summary["renderer_joint"]["status"] == (
        "assignment_v2_renderer_joint_validation_pass"
    )
    assert summary["renderer_boundary"]["status"] == (
        "assignment_v2_renderer_joint_validation_ready"
    )
    assert summary["non_goals"]["uses_fixture_oracle"] is False
    assert summary["non_goals"]["unfreezes_gaussian_geometry"] is False
    assert validate_real_sample_v2_smoke_summary(summary) is summary


def test_real_sample_v2_smoke_binds_missing_image_targets_on_sample_input():
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(),
        frame_count=2,
        max_points=6,
        seed=4,
    )

    report = evaluate_real_sample_v2_smoke(
        sample,
        sample_source="memory://unit-test",
        image_width=8,
        image_height=8,
        iterations=100,
        learning_rate=0.4,
        seed=4,
    )

    assert report.sample.as_dict()["image_target_contract"]["status"] == "image_targets_bound"
    assert report.as_dict()["status"] == "real_sample_v2_smoke_pass"


def test_real_public_sample_smoke_exposes_objectstate_gap():
    cloud = read_ply("public/samples/lego_alpha_v1_objects.ply")

    report = real_sample_v2_smoke_from_cloud(
        cloud,
        sample_source="public/samples/lego_alpha_v1_objects.ply",
        frame_count=2,
        max_points=24,
        image_width=12,
        image_height=12,
        iterations=100,
        learning_rate=0.4,
        seed=4,
    )
    summary = report.as_dict()

    assert summary["status"] == "real_sample_v2_smoke_fail"
    assert summary["training_loss"]["loss_decreased"] is True
    assert summary["renderer_joint"]["loss_decreased"] is True
    assert summary["renderer_joint"]["image_render_loss_decreased"] is True
    assert summary["renderer_joint"]["checkpoint_roundtrip"]["pass"] is True
    assert summary["renderer_joint"]["status"] == (
        "assignment_v2_renderer_joint_validation_fail"
    )
    object_state_eval = summary["renderer_joint"]["object_state_eval"]
    assert object_state_eval["status"] == "objectstate_eval_fail"
    diagnostics = {
        diagnostic
        for frame in object_state_eval["frames"]
        for diagnostic in frame["diagnostics"]
    }
    assert {"low_assignment_confidence", "low_object_purity"} <= diagnostics


def test_real_public_sample_smoke_passes_with_temperature_sharpening():
    cloud = read_ply("public/samples/lego_alpha_v1_objects.ply")

    report = real_sample_v2_smoke_from_cloud(
        cloud,
        sample_source="public/samples/lego_alpha_v1_objects.ply",
        frame_count=2,
        max_points=24,
        image_width=12,
        image_height=12,
        iterations=100,
        learning_rate=0.4,
        solver_temperature=0.5,
        seed=4,
    )
    summary = report.as_dict()

    assert summary["status"] == "real_sample_v2_smoke_pass"
    assert summary["training"]["final_state"]["config"]["temperature"] == 0.5
    assert summary["renderer_joint"]["object_state_eval"]["status"] == "objectstate_eval_pass"
    assert summary["renderer_joint"]["object_state_eval"]["object_purity"] >= 0.8


def test_real_sample_v2_smoke_rejects_feature_pseudo_targets():
    sample = trainable_kernel_sample_from_cloud(
        _object_cloud(include_object_ids=False),
        slots=2,
        frame_count=1,
        max_points=4,
    )

    with pytest.raises(ValueError, match="object_id_one_hot_targets"):
        evaluate_real_sample_v2_smoke(sample)


def _object_cloud(*, include_object_ids: bool = True) -> GaussianCloud:
    fields = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
        ("opacity", "f4"),
    ]
    if include_object_ids:
        fields.append(("object_id", "i4"))
    vertices = np.zeros(6, dtype=np.dtype(fields))
    vertices["x"] = np.array([-1.0, -0.8, -0.7, 0.8, 1.0, 1.2], dtype=np.float32)
    vertices["y"] = np.array([0.0, 0.1, -0.1, 0.0, 0.12, -0.08], dtype=np.float32)
    vertices["z"] = 0.0
    vertices["red"] = np.array([240, 230, 220, 20, 30, 40], dtype=np.uint8)
    vertices["green"] = np.array([30, 35, 40, 220, 230, 225], dtype=np.uint8)
    vertices["blue"] = np.array([25, 30, 35, 220, 230, 235], dtype=np.uint8)
    vertices["opacity"] = 1.0
    if include_object_ids:
        vertices["object_id"] = np.array([5, 5, 5, 9, 9, 9], dtype=np.int32)
    return GaussianCloud(vertices=vertices, source_format="ascii")
