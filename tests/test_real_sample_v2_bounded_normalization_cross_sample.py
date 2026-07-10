from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.io import read_ply
from objgauss.pipelines.real_sample_v2_bounded_normalization_cross_sample import (
    REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA,
    RealSampleV2BoundedNormalizationCrossSampleInput,
    RealSampleV2BoundedNormalizationCrossSampleReport,
    real_sample_v2_bounded_normalization_cross_sample_from_clouds,
    validate_real_sample_v2_bounded_normalization_cross_sample_summary,
)


def test_bounded_normalization_cross_sample_passes_with_sample_aware_policy():
    report = real_sample_v2_bounded_normalization_cross_sample_from_clouds(
        (
            RealSampleV2BoundedNormalizationCrossSampleInput(
                sample_id="lego",
                cloud=read_ply("public/samples/lego_alpha_v1_objects.ply"),
                sample_source="public/samples/lego_alpha_v1_objects.ply",
                viewer_path="/samples/objgauss-real-sample-v2-sample-aware-lego.ply",
            ),
            RealSampleV2BoundedNormalizationCrossSampleInput(
                sample_id="polyhaven",
                cloud=read_ply("public/samples/polyhaven_chair_demo_objects.ply"),
                sample_source="public/samples/polyhaven_chair_demo_objects.ply",
                viewer_path="/samples/objgauss-real-sample-v2-sample-aware-polyhaven.ply",
            ),
            RealSampleV2BoundedNormalizationCrossSampleInput(
                sample_id="nike",
                cloud=read_ply("public/samples/nike_objects.ply"),
                sample_source="public/samples/nike_objects.ply",
                viewer_path="/samples/objgauss-real-sample-v2-sample-aware-nike.ply",
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
    assert aggregate["blocked_promoted_samples"] == ["polyhaven", "nike"]

    rows = {row["sample_id"]: row for row in summary["rows"]}
    lego = rows["lego"]
    assert lego["source"]["source_gaussians"] == 5696
    assert lego["selected_policy"]["candidate_name"] == "promoted"
    assert lego["selected_metrics"]["mixed_gaussians"] == 0
    assert lego["selected_changed_gaussians"]["hard_fix_count"] == 59
    assert lego["selected_changed_gaussians"]["hard_regression_count"] == 0
    assert lego["promoted_candidate"]["sample_policy_gate"]["eligible_for_sample"] is True

    polyhaven = rows["polyhaven"]
    assert polyhaven["source"]["source_gaussians"] == 50000
    assert polyhaven["selected_policy"]["candidate_name"] == "baseline"
    assert polyhaven["selected_metrics"]["mixed_gaussians"] == 3840
    assert polyhaven["selected_changed_gaussians"]["hard_regression_count"] == 0
    assert polyhaven["promoted_candidate"]["sample_policy_gate"]["eligible_for_sample"] is False
    assert polyhaven["promoted_candidate"]["sample_policy_gate"]["hard_regression_free"] is False
    assert polyhaven["promoted_candidate"]["sample_policy_gate"]["hard_regression_count"] == 1814
    assert polyhaven["bounded_normalized_candidate"]["sample_policy_gate"]["eligible_for_sample"] is False
    assert polyhaven["bounded_normalized_candidate"]["sample_policy_gate"]["decision"] == (
        "bounded_evidence_normalization_noop_baseline_fallback"
    )
    assert polyhaven["evidence_normalization_status"] == "required_before_global_weight_promotion"

    nike = rows["nike"]
    assert nike["source"]["source_gaussians"] == 270491
    assert nike["selected_policy"]["candidate_name"] == "baseline"
    assert nike["selected_metrics"]["mixed_gaussians"] == 16721
    assert nike["selected_changed_gaussians"]["hard_regression_count"] == 0
    assert nike["promoted_candidate"]["sample_policy_gate"]["eligible_for_sample"] is False
    assert nike["promoted_candidate"]["sample_policy_gate"]["hard_regression_free"] is False
    assert nike["promoted_candidate"]["sample_policy_gate"]["hard_regression_count"] == 6671
    assert nike["bounded_normalized_candidate"]["sample_policy_gate"]["eligible_for_sample"] is False
    assert nike["bounded_normalized_candidate"]["sample_policy_gate"]["decision"] == (
        "bounded_evidence_normalization_noop_baseline_fallback"
    )
    assert nike["evidence_normalization_status"] == "required_before_global_weight_promotion"

    assert summary["recommendation"]["decision"] == (
        "sample_aware_bounded_normalization_cross_sample_pass"
    )
    assert summary["recommendation"]["global_default"] == (
        "sample_aware_policy_not_single_weight_default"
    )
    assert summary["recommendation"]["requires_geometry_unfreeze"] is False
    assert summary["recommendation"]["requires_diffusion_replay_or_rollout"] is False
    assert validate_real_sample_v2_bounded_normalization_cross_sample_summary(summary) is summary


def test_bounded_normalization_cross_sample_cli_writes_summary(tmp_path):
    summary_path = tmp_path / "bounded-normalization-cross-sample-summary.json"

    exit_code = main(
        [
            "training",
            "real-sample-v2-bounded-normalization-cross-sample",
            "public/samples/lego_alpha_v1_objects.ply",
            "public/samples/polyhaven_chair_demo_objects.ply",
            "public/samples/nike_objects.ply",
            "--sample-id",
            "lego",
            "--sample-id",
            "polyhaven",
            "--sample-id",
            "nike",
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
    assert summary["aggregate"]["blocked_promoted_samples"] == ["polyhaven", "nike"]
    assert [row["sample_id"] for row in summary["rows"]] == ["lego", "polyhaven", "nike"]
