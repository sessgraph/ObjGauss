from __future__ import annotations

import json

import numpy as np

from objgauss.cli import main
from objgauss.core.io import read_ply
from objgauss.pipelines.real_sample_v2_weak_boundary_opt import (
    REAL_SAMPLE_V2_WEAK_BOUNDARY_OPT_SCHEMA,
    RealSampleV2WeakBoundaryOptReport,
    real_sample_v2_weak_boundary_opt_from_cloud,
    validate_real_sample_v2_weak_boundary_opt_summary,
)


def test_real_sample_v2_weak_boundary_opt_promotes_cost_weight_candidate(
    real_sample_v2_scenarios,
):
    scenario = real_sample_v2_scenarios["weight-fix"]
    report = real_sample_v2_weak_boundary_opt_from_cloud(
        scenario.cloud,
        sample_source=scenario.source,
        max_points=128,
        solver_temperature=0.35,
        candidate_feature_weight=2.0,
        candidate_position_weight=1.0,
        frame_count=2,
        image_width=12,
        image_height=12,
        iterations=100,
        learning_rate=0.4,
        seed=4,
        viewer_path="/samples/objgauss-real-sample-v2-weak-boundary-opt.ply",
    )
    summary = report.as_dict()

    assert isinstance(report, RealSampleV2WeakBoundaryOptReport)
    assert summary["schema"] == REAL_SAMPLE_V2_WEAK_BOUNDARY_OPT_SCHEMA
    assert summary["status"] == "real_sample_v2_weak_boundary_opt_pass"
    assert summary["fixed_target"]["max_points"] == 128
    assert summary["fixed_target"]["solver_temperature"] == 0.35
    assert summary["fixed_target"]["coverage_scan"] == "disabled"
    assert summary["fixed_target"]["temperature_sharpening_scan"] == "disabled"
    assert summary["candidate_policy"]["feature_weight"] == 2.0
    assert summary["candidate_policy"]["position_weight"] == 1.0
    assert summary["candidate_policy"]["uses_target_labels_for_prediction"] is False

    baseline = summary["baseline"]["global_quality"]
    candidate = summary["candidate"]["global_quality"]
    assert baseline["mixed_gaussians"] == 9
    assert candidate["mixed_gaussians"] == 0
    assert candidate["direct_slot_match"] == 1.0
    assert candidate["hard_argmax_object_purity"] == 1.0
    assert candidate["min_predicted_object_purity"] == 1.0
    assert candidate["min_target_recall"] == 1.0
    assert summary["candidate"]["confusion"]["matrix"] == [
        [256, 0],
        [0, 256],
    ]
    assert summary["quality_delta"]["mixed_gaussians_delta"] == -9
    assert summary["quality_delta"]["direct_slot_match_delta"] > 0.017
    assert summary["quality_delta"]["min_target_recall_delta"] > 0.035

    changed = summary["changed_gaussians"]
    assert changed["changed_count"] == 9
    assert changed["pairs"] == [
        {"baseline_object_id": 1, "candidate_object_id": 0, "count": 9}
    ]
    assert summary["recommendation"]["decision"] == "promote_cost_weight_normalization_candidate"
    assert summary["recommendation"]["action"] == "use_feature_weight_boost_for_next_viewer_preview"
    assert summary["recommendation"]["requires_more_coverage"] is False
    assert summary["recommendation"]["requires_temperature_sharpening"] is False
    assert summary["recommendation"]["requires_geometry_unfreeze"] is False
    assert summary["recommendation"]["requires_diffusion_replay_or_rollout"] is False
    assert validate_real_sample_v2_weak_boundary_opt_summary(summary) is summary
    assert {
        "baseline_object_id",
        "baseline_assignment_confidence",
        "baseline_assignment_entropy",
        "weak_boundary_candidate",
        "boundary_changed",
    }.issubset(set(report.candidate_cloud.fields))
    assert int(np.sum(report.candidate_cloud.vertices["boundary_changed"])) == 9


def test_real_sample_v2_weak_boundary_opt_cli_writes_candidate_ply_and_summary(
    tmp_path,
    real_sample_v2_scenarios,
):
    scenario = real_sample_v2_scenarios["weight-fix"]
    input_path = scenario.write(tmp_path)
    preview_ply = tmp_path / "weak-boundary-opt.ply"
    summary_path = tmp_path / "weak-boundary-opt-summary.json"

    exit_code = main(
        [
            "training",
            "real-sample-v2-weak-boundary-opt",
            str(input_path),
            "--preview-ply-output",
            str(preview_ply),
            "--summary-output",
            str(summary_path),
            "--viewer-path",
            "/samples/objgauss-real-sample-v2-weak-boundary-opt.ply",
            "--require-pass",
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cloud = read_ply(preview_ply)
    assert summary["status"] == "real_sample_v2_weak_boundary_opt_pass"
    assert summary["candidate"]["global_quality"]["mixed_gaussians"] == 0
    assert summary["changed_gaussians"]["changed_count"] == 9
    assert cloud.count == scenario.cloud.count
    assert "baseline_object_id" in cloud.fields
    assert "boundary_changed" in cloud.fields
    assert int(np.sum(cloud.vertices["boundary_changed"])) == 9
    assert np.bincount(np.asarray(cloud.vertices["object_id"], dtype=np.int64)).tolist() == [
        256,
        256,
    ]
