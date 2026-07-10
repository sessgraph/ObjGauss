from __future__ import annotations

import json

import numpy as np
import pytest

from objgauss.cli import main
from objgauss.core.io import read_ply
from objgauss.pipelines.real_sample_v2_sample_aware_weight_policy import (
    REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA,
    RealSampleV2SampleAwareWeightPolicyReport,
    real_sample_v2_sample_aware_weight_policy_from_cloud,
    validate_real_sample_v2_sample_aware_weight_policy_summary,
)


def test_sample_aware_weight_policy_selects_baseline_when_promoted_regresses_polyhaven():
    report = real_sample_v2_sample_aware_weight_policy_from_cloud(
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
        viewer_path="/samples/objgauss-real-sample-v2-sample-aware-polyhaven.ply",
    )
    summary = report.as_dict()

    assert isinstance(report, RealSampleV2SampleAwareWeightPolicyReport)
    assert summary["schema"] == REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA
    assert summary["status"] == "real_sample_v2_sample_aware_weight_policy_pass"
    assert summary["source"]["source_gaussians"] == 50000
    assert summary["policy"]["selection_rule"] == (
        "prefer_hard_boundary_non_regression_over_bounded_normalization_over_baseline"
    )
    assert summary["policy"]["uses_target_labels_for_prediction"] is False
    assert summary["policy"]["uses_target_labels_for_gate"] is True
    assert summary["policy"]["bounded_evidence_normalization"]["schema"] == (
        "objgauss-bounded-evidence-normalization-v1"
    )
    assert summary["policy"]["bounded_evidence_normalization"]["feature_weight_blend"] == 0.0
    assert summary["selected_policy"]["candidate_name"] == "baseline"
    assert summary["selected_policy"]["feature_weight"] == 1.0
    assert summary["selected_policy"]["selection_reason"] == (
        "baseline_safe_fallback"
    )
    assert summary["selected_policy"]["global_default"] == "sample_specific_only"

    baseline, promoted, normalized = summary["candidates"]
    assert baseline["candidate"]["name"] == "baseline"
    assert baseline["sample_policy_gate"]["eligible_for_sample"] is True
    assert baseline["metrics"]["mixed_gaussians"] == 3840
    assert baseline["metrics"]["object_id_counts"] == [
        {"object_id": 0, "count": 15214},
        {"object_id": 1, "count": 4383},
        {"object_id": 2, "count": 7359},
        {"object_id": 3, "count": 9152},
        {"object_id": 4, "count": 7812},
        {"object_id": 5, "count": 6080},
    ]
    assert promoted["candidate"]["name"] == "promoted"
    assert promoted["sample_policy_gate"]["eligible_for_sample"] is False
    assert promoted["sample_policy_gate"]["hard_mixed_gaussians_non_regression"] is False
    assert promoted["sample_policy_gate"]["direct_slot_match_non_regression"] is False
    assert promoted["metrics"]["mixed_gaussians"] == 3918
    assert promoted["delta_vs_baseline"]["mixed_gaussians_delta"] == 78
    assert promoted["delta_vs_baseline"]["direct_slot_match_delta"] < 0.0
    assert promoted["delta_vs_baseline"]["object_purity_delta"] > 0.060
    assert promoted["changed_gaussians"]["changed_count"] == 3670
    assert promoted["changed_gaussians"]["hard_fix_count"] == 1736
    assert promoted["changed_gaussians"]["hard_regression_count"] == 1814
    assert normalized["candidate"]["name"] == "bounded-normalized"
    assert normalized["candidate"]["feature_weight"] == 1.0
    assert normalized["bounded_evidence_normalization"]["reason"] == (
        "hard_regression_not_bounded_by_hard_fix"
    )
    assert normalized["bounded_evidence_normalization"]["hard_safety_blend"] == 0.0
    assert normalized["bounded_evidence_normalization"]["soft_evidence_blend"] > 0.0
    assert normalized["bounded_evidence_normalization"]["bounded_confidence_gain"] > 0.12
    assert normalized["bounded_evidence_normalization"]["bounded_entropy_reduction"] > 0.12
    assert normalized["sample_policy_gate"]["eligible_for_sample"] is False
    assert normalized["sample_policy_gate"]["decision"] == (
        "bounded_evidence_normalization_noop_baseline_fallback"
    )
    assert normalized["sample_policy_gate"]["hard_regression_count"] == 0
    assert normalized["metrics"]["mixed_gaussians"] == baseline["metrics"]["mixed_gaussians"]
    assert normalized["delta_vs_baseline"]["mixed_gaussians_delta"] == 0

    evidence_gate = summary["evidence_normalization_gate"]
    assert evidence_gate["status"] == "required_before_global_weight_promotion"
    assert evidence_gate["requires_evidence_normalization"] is True
    assert evidence_gate["blocked_soft_sharpening_candidates"] == ["promoted"]
    assert evidence_gate["bounded_normalized_candidate"]["eligible_for_sample"] is False
    assert evidence_gate["bounded_normalized_candidate"]["feature_weight_blend"] == 0.0
    assert evidence_gate["requires_geometry_unfreeze"] is False
    assert evidence_gate["requires_diffusion_replay_or_rollout"] is False
    assert validate_real_sample_v2_sample_aware_weight_policy_summary(summary) is summary

    selected_cloud = report.selected_cloud
    assert selected_cloud.count == 50000
    assert {
        "sample_aware_baseline_object_id",
        "sample_aware_baseline_confidence",
        "sample_aware_baseline_entropy",
        "sample_aware_selected_index",
        "sample_aware_changed",
        "sample_aware_hard_fix",
        "sample_aware_hard_regression",
    }.issubset(set(selected_cloud.fields))
    assert np.unique(selected_cloud.vertices["sample_aware_selected_index"]).tolist() == [0]
    assert int(np.sum(selected_cloud.vertices["sample_aware_changed"])) == 0


def test_sample_aware_weight_policy_selects_promoted_for_lego_local_boundary():
    report = real_sample_v2_sample_aware_weight_policy_from_cloud(
        read_ply("public/samples/lego_alpha_v1_objects.ply"),
        sample_source="public/samples/lego_alpha_v1_objects.ply",
        max_points=128,
        solver_temperature=0.35,
        frame_count=2,
        image_width=12,
        image_height=12,
        iterations=100,
        learning_rate=0.4,
        seed=4,
        viewer_path="/samples/objgauss-real-sample-v2-sample-aware-lego.ply",
    )
    summary = report.as_dict()

    assert summary["status"] == "real_sample_v2_sample_aware_weight_policy_pass"
    assert summary["source"]["source_gaussians"] == 5696
    assert summary["selected_policy"]["candidate_name"] == "promoted"
    assert summary["selected_policy"]["feature_weight"] == 2.0
    assert summary["selected_policy"]["selection_reason"] == (
        "candidate_hard_boundary_non_regression"
    )
    assert summary["evidence_normalization_gate"]["status"] == (
        "not_required_for_selected_policy"
    )
    assert summary["evidence_normalization_gate"]["requires_evidence_normalization"] is False

    baseline, promoted, normalized = summary["candidates"]
    assert baseline["metrics"]["mixed_gaussians"] == 59
    assert promoted["sample_policy_gate"]["eligible_for_sample"] is True
    assert promoted["metrics"]["mixed_gaussians"] == 0
    assert promoted["metrics"]["direct_slot_match"] == 1.0
    assert promoted["delta_vs_baseline"]["mixed_gaussians_delta"] == -59
    assert promoted["delta_vs_baseline"]["direct_slot_match_delta"] > 0.0103
    assert promoted["changed_gaussians"]["hard_fix_count"] == 59
    assert promoted["changed_gaussians"]["hard_regression_count"] == 0
    assert normalized["candidate"]["name"] == "bounded-normalized"
    assert normalized["bounded_evidence_normalization"]["feature_weight_blend"] == 1.0
    assert normalized["bounded_evidence_normalization"]["hard_safety_blend"] == 1.0
    assert normalized["bounded_evidence_normalization"]["bounded_entropy_reduction"] > 0.20
    assert normalized["metrics"]["mixed_gaussians"] == 0
    assert normalized["changed_gaussians"]["hard_regression_count"] == 0

    selected_cloud = report.selected_cloud
    assert np.unique(selected_cloud.vertices["sample_aware_selected_index"]).tolist() == [1]
    assert int(np.sum(selected_cloud.vertices["sample_aware_changed"])) == 59
    assert int(np.sum(selected_cloud.vertices["sample_aware_hard_fix"])) == 59
    assert int(np.sum(selected_cloud.vertices["sample_aware_hard_regression"])) == 0
    assert np.bincount(np.asarray(selected_cloud.vertices["object_id"], dtype=np.int64)).tolist() == [
        736,
        581,
        1787,
        2592,
    ]


def test_sample_aware_weight_policy_rejects_plush_when_only_hard_regressing_candidates_pass():
    report = real_sample_v2_sample_aware_weight_policy_from_cloud(
        read_ply("public/samples/plush_objects.ply"),
        sample_source="public/samples/plush_objects.ply",
        max_points=128,
        solver_temperature=0.35,
        frame_count=2,
        image_width=12,
        image_height=12,
        iterations=100,
        learning_rate=0.4,
        seed=4,
        viewer_path="/samples/objgauss-real-sample-v2-sample-aware-plush.ply",
    )

    with pytest.raises(ValueError, match="no sample-aware candidate passed the gate"):
        report.as_dict()


def test_sample_aware_weight_policy_cli_writes_selected_ply_and_summary(tmp_path):
    preview_ply = tmp_path / "sample-aware-polyhaven.ply"
    summary_path = tmp_path / "sample-aware-polyhaven-summary.json"

    exit_code = main(
        [
            "training",
            "real-sample-v2-sample-aware-weight-policy",
            "public/samples/polyhaven_chair_demo_objects.ply",
            "--preview-ply-output",
            str(preview_ply),
            "--summary-output",
            str(summary_path),
            "--viewer-path",
            "/samples/objgauss-real-sample-v2-sample-aware-polyhaven.ply",
            "--require-pass",
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cloud = read_ply(preview_ply)
    assert summary["status"] == "real_sample_v2_sample_aware_weight_policy_pass"
    assert summary["selected_policy"]["candidate_name"] == "baseline"
    assert summary["evidence_normalization_gate"]["requires_evidence_normalization"] is True
    assert summary["evidence_normalization_gate"]["status"] == "required_before_global_weight_promotion"
    assert summary["viewer"]["debug_route"] == (
        "/?ply=/samples/objgauss-real-sample-v2-sample-aware-polyhaven.ply"
    )
    assert cloud.count == 50000
    assert "sample_aware_selected_index" in cloud.fields
    assert "sample_aware_hard_regression" in cloud.fields
    assert np.unique(cloud.vertices["sample_aware_selected_index"]).tolist() == [0]
    assert int(np.sum(cloud.vertices["sample_aware_hard_regression"])) == 0
