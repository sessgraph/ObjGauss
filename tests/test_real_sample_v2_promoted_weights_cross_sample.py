from __future__ import annotations

import json

import numpy as np

from objgauss.cli import main
from objgauss.core.io import read_ply
from objgauss.pipelines.real_sample_v2_promoted_weights_cross_sample import (
    REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA,
    RealSampleV2PromotedWeightsCrossSampleReport,
    real_sample_v2_promoted_weights_cross_sample_from_cloud,
    validate_real_sample_v2_promoted_weights_cross_sample_summary,
)


def test_real_sample_v2_promoted_weights_cross_sample_records_hard_boundary_regression(
    real_sample_v2_scenarios,
):
    scenario = real_sample_v2_scenarios["weight-regression-a"]
    reference = real_sample_v2_scenarios["weight-fix"]
    report = real_sample_v2_promoted_weights_cross_sample_from_cloud(
        scenario.cloud,
        sample_source=scenario.source,
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
        reference_sample=reference.source,
    )
    summary = report.as_dict()

    assert isinstance(report, RealSampleV2PromotedWeightsCrossSampleReport)
    assert summary["schema"] == REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA
    assert summary["status"] == "real_sample_v2_promoted_weights_cross_sample_diagnostic"
    assert summary["source"]["source_gaussians"] == scenario.cloud.count
    assert summary["source"]["reference_sample"] == reference.source
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
    assert baseline_hard["mixed_gaussians"] == 32
    assert promoted_hard["mixed_gaussians"] == 49
    assert baseline_hard["object_id_counts"] == [
        {"object_id": 0, "count": 224},
        {"object_id": 1, "count": 288},
    ]
    assert promoted_hard["object_id_counts"] == [
        {"object_id": 0, "count": 225},
        {"object_id": 1, "count": 287},
    ]

    delta = summary["quality_delta"]
    assert delta["mixed_gaussians_delta"] == 17
    assert delta["predicted_object_count_delta"] == 0
    assert delta["direct_slot_match_delta"] < 0.0
    assert delta["object_purity_delta"] < 0.0
    assert delta["assignment_confidence_delta"] > 0.0
    assert delta["mean_normalized_entropy_delta"] < 0.0

    changed = summary["changed_gaussians"]
    assert changed["changed_count"] == 17
    assert changed["hard_fix_count"] == 0
    assert changed["hard_regression_count"] == 17
    assert changed["pairs"] == [
        {"baseline_object_id": 0, "promoted_object_id": 1, "count": 8},
        {"baseline_object_id": 1, "promoted_object_id": 0, "count": 9},
    ]

    gate = summary["cross_sample_gate"]
    assert gate["result"] == "diagnostic"
    assert gate["hard_mixed_gaussians_non_regression"] is False
    assert gate["direct_slot_match_non_regression"] is False
    assert gate["soft_purity_non_regression"] is False
    assert summary["recommendation"]["decision"] == "hold_promoted_weights_for_global_default"
    assert summary["recommendation"]["requires_geometry_unfreeze"] is False
    assert summary["recommendation"]["requires_diffusion_replay_or_rollout"] is False
    assert validate_real_sample_v2_promoted_weights_cross_sample_summary(summary) is summary

    promoted_cloud = report.promoted_cloud
    assert promoted_cloud.count == scenario.cloud.count
    assert {
        "baseline_object_id",
        "baseline_assignment_confidence",
        "baseline_assignment_entropy",
        "promotion_changed",
        "promotion_hard_fix",
        "promotion_hard_regression",
    }.issubset(set(promoted_cloud.fields))
    assert int(np.sum(promoted_cloud.vertices["promotion_changed"])) == 17
    assert int(np.sum(promoted_cloud.vertices["promotion_hard_fix"])) == 0
    assert int(np.sum(promoted_cloud.vertices["promotion_hard_regression"])) == 17


def test_real_sample_v2_promoted_weights_cross_sample_cli_writes_diagnostic(
    tmp_path,
    real_sample_v2_scenarios,
):
    scenario = real_sample_v2_scenarios["weight-regression-a"]
    reference = real_sample_v2_scenarios["weight-fix"]
    input_path = scenario.write(tmp_path)
    preview_ply = tmp_path / "promoted-cross-sample.ply"
    summary_path = tmp_path / "promoted-cross-sample-summary.json"

    exit_code = main(
        [
            "training",
            "real-sample-v2-promoted-weights-cross-sample",
            str(input_path),
            "--preview-ply-output",
            str(preview_ply),
            "--summary-output",
            str(summary_path),
            "--viewer-path",
            "/samples/objgauss-real-sample-v2-promoted-weights-cross-sample.ply",
            "--reference-sample",
            reference.source,
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cloud = read_ply(preview_ply)
    assert summary["status"] == "real_sample_v2_promoted_weights_cross_sample_diagnostic"
    assert summary["quality_delta"]["mixed_gaussians_delta"] == 17
    assert summary["changed_gaussians"]["changed_count"] == 17
    assert summary["viewer"]["debug_route"] == (
        "/?ply=/samples/objgauss-real-sample-v2-promoted-weights-cross-sample.ply"
    )
    assert cloud.count == scenario.cloud.count
    assert "baseline_object_id" in cloud.fields
    assert "promotion_hard_regression" in cloud.fields
    assert int(np.sum(cloud.vertices["promotion_hard_regression"])) == 17
