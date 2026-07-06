from __future__ import annotations

import json

import numpy as np

from objgauss.cli import main
from objgauss.core.io import read_ply
from objgauss.core.real_sample_v2_promoted_weights_cross_sample import (
    REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA,
    RealSampleV2PromotedWeightsCrossSampleReport,
    real_sample_v2_promoted_weights_cross_sample_from_cloud,
    validate_real_sample_v2_promoted_weights_cross_sample_summary,
)


def test_real_sample_v2_promoted_weights_cross_sample_records_hard_boundary_regression():
    report = real_sample_v2_promoted_weights_cross_sample_from_cloud(
        read_ply("public/samples/polyhaven_chair_demo_objects.ply"),
        sample_source="public/samples/polyhaven_chair_demo_objects.ply",
        max_points=128,
        solver_temperature=0.35,
        baseline_feature_weight=1.0,
        baseline_position_weight=1.0,
        promoted_feature_weight=2.0,
        promoted_position_weight=1.0,
        frame_count=2,
        image_width=12,
        image_height=12,
        iterations=100,
        learning_rate=0.4,
        seed=4,
        viewer_path="/samples/objgauss-real-sample-v2-promoted-weights-cross-sample.ply",
    )
    summary = report.as_dict()

    assert isinstance(report, RealSampleV2PromotedWeightsCrossSampleReport)
    assert summary["schema"] == REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA
    assert summary["status"] == "real_sample_v2_promoted_weights_cross_sample_diagnostic"
    assert summary["source"]["source_gaussians"] == 50000
    assert summary["source"]["reference_sample"] == "public/samples/lego_alpha_v1_objects.ply"
    assert summary["fixed_target"]["max_points"] == 128
    assert summary["fixed_target"]["solver_temperature"] == 0.35
    assert summary["promotion_policy"]["baseline_feature_weight"] == 1.0
    assert summary["promotion_policy"]["promoted_feature_weight"] == 2.0
    assert summary["promotion_policy"]["uses_target_labels_for_prediction"] is False
    assert summary["promotion_policy"]["mutates_checkpoint"] is False

    baseline = summary["baseline"]
    promoted = summary["promoted"]
    baseline_hard = baseline["projection"]["hard_segmentation"]
    promoted_hard = promoted["projection"]["hard_segmentation"]
    assert baseline["status"] == "real_sample_v2_viewer_preview_pass"
    assert promoted["status"] == "real_sample_v2_viewer_preview_pass"
    assert baseline_hard["mixed_gaussians"] == 3840
    assert promoted_hard["mixed_gaussians"] == 3918
    assert baseline_hard["object_id_counts"] == [
        {"object_id": 0, "count": 15214},
        {"object_id": 1, "count": 4383},
        {"object_id": 2, "count": 7359},
        {"object_id": 3, "count": 9152},
        {"object_id": 4, "count": 7812},
        {"object_id": 5, "count": 6080},
    ]
    assert promoted_hard["object_id_counts"] == [
        {"object_id": 0, "count": 13999},
        {"object_id": 1, "count": 3910},
        {"object_id": 2, "count": 8291},
        {"object_id": 3, "count": 9240},
        {"object_id": 4, "count": 7615},
        {"object_id": 5, "count": 6945},
    ]

    delta = summary["quality_delta"]
    assert delta["mixed_gaussians_delta"] == 78
    assert delta["predicted_object_count_delta"] == 0
    assert delta["direct_slot_match_delta"] < 0.0
    assert delta["object_purity_delta"] > 0.060
    assert delta["assignment_confidence_delta"] > 0.12
    assert delta["mean_normalized_entropy_delta"] < -0.12

    changed = summary["changed_gaussians"]
    assert changed["changed_count"] == 3670
    assert changed["hard_fix_count"] == 1736
    assert changed["hard_regression_count"] == 1814
    assert {"baseline_object_id": 0, "promoted_object_id": 5, "count": 1071} in changed["pairs"]
    assert {"baseline_object_id": 1, "promoted_object_id": 2, "count": 612} in changed["pairs"]

    gate = summary["cross_sample_gate"]
    assert gate["result"] == "diagnostic"
    assert gate["hard_mixed_gaussians_non_regression"] is False
    assert gate["direct_slot_match_non_regression"] is False
    assert gate["soft_purity_non_regression"] is True
    assert summary["recommendation"]["decision"] == "hold_promoted_weights_for_global_default"
    assert summary["recommendation"]["requires_geometry_unfreeze"] is False
    assert summary["recommendation"]["requires_diffusion_replay_or_rollout"] is False
    assert validate_real_sample_v2_promoted_weights_cross_sample_summary(summary) is summary

    promoted_cloud = report.promoted_cloud
    assert promoted_cloud.count == 50000
    assert {
        "baseline_object_id",
        "baseline_assignment_confidence",
        "baseline_assignment_entropy",
        "promotion_changed",
        "promotion_hard_fix",
        "promotion_hard_regression",
    }.issubset(set(promoted_cloud.fields))
    assert int(np.sum(promoted_cloud.vertices["promotion_changed"])) == 3670
    assert int(np.sum(promoted_cloud.vertices["promotion_hard_fix"])) == 1736
    assert int(np.sum(promoted_cloud.vertices["promotion_hard_regression"])) == 1814


def test_real_sample_v2_promoted_weights_cross_sample_cli_writes_diagnostic(tmp_path):
    preview_ply = tmp_path / "promoted-cross-sample.ply"
    summary_path = tmp_path / "promoted-cross-sample-summary.json"

    exit_code = main(
        [
            "training",
            "real-sample-v2-promoted-weights-cross-sample",
            "public/samples/polyhaven_chair_demo_objects.ply",
            "--preview-ply-output",
            str(preview_ply),
            "--summary-output",
            str(summary_path),
            "--viewer-path",
            "/samples/objgauss-real-sample-v2-promoted-weights-cross-sample.ply",
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cloud = read_ply(preview_ply)
    assert summary["status"] == "real_sample_v2_promoted_weights_cross_sample_diagnostic"
    assert summary["quality_delta"]["mixed_gaussians_delta"] == 78
    assert summary["changed_gaussians"]["changed_count"] == 3670
    assert summary["viewer"]["debug_route"] == (
        "/?ply=/samples/objgauss-real-sample-v2-promoted-weights-cross-sample.ply"
    )
    assert cloud.count == 50000
    assert "baseline_object_id" in cloud.fields
    assert "promotion_hard_regression" in cloud.fields
    assert int(np.sum(cloud.vertices["promotion_hard_regression"])) == 1814
