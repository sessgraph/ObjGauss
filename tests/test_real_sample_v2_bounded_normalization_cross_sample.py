from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.pipelines.real_sample_v2_bounded_normalization_cross_sample import (
    REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA,
    RealSampleV2BoundedNormalizationCrossSampleInput,
    RealSampleV2BoundedNormalizationCrossSampleReport,
    real_sample_v2_bounded_normalization_cross_sample_from_clouds,
    validate_real_sample_v2_bounded_normalization_cross_sample_summary,
)


def test_bounded_normalization_cross_sample_passes_with_sample_aware_policy(
    real_sample_v2_scenarios,
):
    fix = real_sample_v2_scenarios["weight-fix"]
    regression_a = real_sample_v2_scenarios["weight-regression-a"]
    regression_b = real_sample_v2_scenarios["weight-regression-b"]
    report = real_sample_v2_bounded_normalization_cross_sample_from_clouds(
        (
            RealSampleV2BoundedNormalizationCrossSampleInput(
                sample_id="weight-fix",
                cloud=fix.cloud,
                sample_source=fix.source,
                viewer_path="/samples/objgauss-real-sample-v2-sample-aware-weight-fix.ply",
            ),
            RealSampleV2BoundedNormalizationCrossSampleInput(
                sample_id="weight-regression-a",
                cloud=regression_a.cloud,
                sample_source=regression_a.source,
                viewer_path="/samples/objgauss-real-sample-v2-sample-aware-regression-a.ply",
            ),
            RealSampleV2BoundedNormalizationCrossSampleInput(
                sample_id="weight-regression-b",
                cloud=regression_b.cloud,
                sample_source=regression_b.source,
                viewer_path="/samples/objgauss-real-sample-v2-sample-aware-regression-b.ply",
            ),
        ),
        min_samples=2,
        max_points=128,
        solver_temperature=0.35,
        frame_count=2,
        image_width=12,
        image_height=12,
        iterations=100,
        learning_rate=0.4,
        seed=4,
    )
    summary = report.as_dict()

    assert isinstance(report, RealSampleV2BoundedNormalizationCrossSampleReport)
    assert summary["schema"] == REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA
    assert summary["status"] == "real_sample_v2_bounded_normalization_cross_sample_pass"
    assert summary["sample_count"] == 3
    assert summary["policy"]["uses_target_labels_for_prediction"] is False
    assert summary["policy"]["uses_target_labels_for_gate"] is True

    aggregate = summary["aggregate"]
    assert aggregate["result"] == "pass"
    assert aggregate["enough_samples"] is True
    assert aggregate["all_sample_policies_pass"] is True
    assert aggregate["selected_policy_counts"] == {
        "baseline": 2,
        "promoted": 1,
    }
    assert aggregate["evidence_normalization_status_counts"] == {
        "not_required_for_selected_policy": 1,
        "required_before_global_weight_promotion": 2,
    }
    assert aggregate["selected_hard_regression_count"] == 0
    assert aggregate["selected_hard_regression_samples"] == []
    assert aggregate["blocked_promoted_sample_count"] == 2
    assert aggregate["blocked_promoted_samples"] == [
        "weight-regression-a",
        "weight-regression-b",
    ]

    rows = {row["sample_id"]: row for row in summary["rows"]}
    fix_row = rows["weight-fix"]
    assert fix_row["source"]["source_gaussians"] == fix.cloud.count
    assert fix_row["selected_policy"]["candidate_name"] == "promoted"
    assert fix_row["selected_metrics"]["mixed_gaussians"] == 0
    assert fix_row["selected_changed_gaussians"]["hard_fix_count"] == 9
    assert fix_row["selected_changed_gaussians"]["hard_regression_count"] == 0
    assert fix_row["promoted_candidate"]["sample_policy_gate"]["eligible_for_sample"] is True

    row_a = rows["weight-regression-a"]
    assert row_a["source"]["source_gaussians"] == regression_a.cloud.count
    assert row_a["selected_policy"]["candidate_name"] == "baseline"
    assert row_a["selected_metrics"]["mixed_gaussians"] == 32
    assert row_a["selected_changed_gaussians"]["hard_regression_count"] == 0
    assert row_a["promoted_candidate"]["sample_policy_gate"]["eligible_for_sample"] is False
    assert row_a["promoted_candidate"]["sample_policy_gate"]["hard_regression_free"] is False
    assert row_a["promoted_candidate"]["sample_policy_gate"]["hard_regression_count"] == 17
    assert row_a["bounded_normalized_candidate"]["sample_policy_gate"]["eligible_for_sample"] is False
    assert row_a["bounded_normalized_candidate"]["sample_policy_gate"]["decision"] == (
        "bounded_evidence_normalization_noop_baseline_fallback"
    )
    assert row_a["evidence_normalization_status"] == "required_before_global_weight_promotion"

    row_b = rows["weight-regression-b"]
    assert row_b["source"]["source_gaussians"] == regression_b.cloud.count
    assert row_b["selected_policy"]["candidate_name"] == "baseline"
    assert row_b["selected_metrics"]["mixed_gaussians"] == 45
    assert row_b["selected_changed_gaussians"]["hard_regression_count"] == 0
    assert row_b["promoted_candidate"]["sample_policy_gate"]["eligible_for_sample"] is False
    assert row_b["promoted_candidate"]["sample_policy_gate"]["hard_regression_free"] is False
    assert row_b["promoted_candidate"]["sample_policy_gate"]["hard_regression_count"] == 17
    assert row_b["bounded_normalized_candidate"]["sample_policy_gate"]["eligible_for_sample"] is False
    assert row_b["bounded_normalized_candidate"]["sample_policy_gate"]["decision"] == (
        "bounded_evidence_normalization_noop_baseline_fallback"
    )
    assert row_b["evidence_normalization_status"] == "required_before_global_weight_promotion"

    assert summary["recommendation"]["decision"] == (
        "sample_aware_bounded_normalization_cross_sample_pass"
    )
    assert summary["recommendation"]["global_default"] == (
        "sample_aware_policy_not_single_weight_default"
    )
    assert summary["recommendation"]["requires_geometry_unfreeze"] is False
    assert summary["recommendation"]["requires_diffusion_replay_or_rollout"] is False
    assert validate_real_sample_v2_bounded_normalization_cross_sample_summary(summary) is summary


def test_bounded_normalization_cross_sample_cli_writes_summary(
    tmp_path,
    real_sample_v2_scenarios,
):
    names = ("weight-fix", "weight-regression-a", "weight-regression-b")
    input_paths = [real_sample_v2_scenarios[name].write(tmp_path) for name in names]
    summary_path = tmp_path / "bounded-normalization-cross-sample-summary.json"

    exit_code = main(
        [
            "training",
            "real-sample-v2-bounded-normalization-cross-sample",
            *(str(path) for path in input_paths),
            "--sample-id",
            names[0],
            "--sample-id",
            names[1],
            "--sample-id",
            names[2],
            "--summary-output",
            str(summary_path),
            "--require-pass",
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "real_sample_v2_bounded_normalization_cross_sample_pass"
    assert summary["aggregate"]["selected_policy_counts"] == {
        "baseline": 2,
        "promoted": 1,
    }
    assert summary["aggregate"]["selected_hard_regression_count"] == 0
    assert summary["aggregate"]["blocked_promoted_samples"] == [names[1], names[2]]
    assert [row["sample_id"] for row in summary["rows"]] == list(names)
