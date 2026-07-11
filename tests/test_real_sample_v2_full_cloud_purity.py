from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.io import read_ply
from objgauss.pipelines.real_sample_v2_full_cloud_purity import (
    REAL_SAMPLE_V2_FULL_CLOUD_PURITY_SCHEMA,
    RealSampleV2FullCloudPurityReport,
    real_sample_v2_full_cloud_purity_from_cloud,
    validate_real_sample_v2_full_cloud_purity_summary,
)


def test_real_sample_v2_full_cloud_purity_selects_128_point_target(
    real_sample_v2_scenarios,
):
    scenario = real_sample_v2_scenarios["coverage-boundary"]
    report = real_sample_v2_full_cloud_purity_from_cloud(
        scenario.cloud,
        sample_source=scenario.source,
        max_point_candidates=(24, 64, 128),
        frame_count=2,
        image_width=12,
        image_height=12,
        iterations=100,
        learning_rate=0.4,
        temperature_candidates=(1.0, 0.75, 0.5, 0.35),
        seed=4,
        viewer_path="/samples/objgauss-real-sample-v2-full-cloud-purity.ply",
    )
    summary = report.as_dict()

    assert isinstance(report, RealSampleV2FullCloudPurityReport)
    assert summary["schema"] == REAL_SAMPLE_V2_FULL_CLOUD_PURITY_SCHEMA
    assert summary["status"] == "real_sample_v2_full_cloud_purity_pass"
    assert summary["candidate_count"] == 3
    assert summary["source"]["source_gaussians"] == scenario.cloud.count
    assert summary["segmentation_target"]["selected_max_points"] == 128
    assert summary["segmentation_target"]["selected_solver_temperature"] == 1.0
    assert summary["viewer"]["debug_route"] == (
        "/?ply=/samples/objgauss-real-sample-v2-full-cloud-purity.ply"
    )

    baseline = summary["baseline_candidate"]
    assert baseline["max_points"] == 24
    assert baseline["quality"]["status"] == "full_cloud_objectstate_preview_quality_diagnostic"
    assert "low_assignment_confidence" in baseline["quality"]["diagnostics"]
    assert baseline["quality"]["assignment_confidence"] < 0.5

    best = summary["best_candidate"]
    assert best["max_points"] == 128
    assert best["sampled_gaussians"] == 128
    assert best["projected_gaussians"] == scenario.cloud.count
    assert best["quality"]["status"] == "full_cloud_objectstate_preview_quality_pass"
    assert best["quality"]["diagnostics"] == []
    assert best["quality"]["object_purity"] >= 0.85
    assert best["quality"]["direct_slot_match"] > 0.98

    recommendation = summary["recommendation"]
    assert recommendation["decision"] == "increase_segmentation_target_coverage"
    assert recommendation["action"] == "set_max_points"
    assert recommendation["max_points"] == 128
    assert recommendation["evidence_normalization"] == "not_required_for_current_sample"
    assert recommendation["requires_geometry_unfreeze"] is False
    assert recommendation["requires_diffusion_replay_or_rollout"] is False
    assert summary["quality_delta"]["purity_delta"] > 0.10
    assert summary["quality_delta"]["confidence_delta"] > 0.30
    assert validate_real_sample_v2_full_cloud_purity_summary(summary) is summary
    assert {
        "object_id",
        "target_object_id",
        "target_slot",
        "assignment_confidence",
        "assignment_entropy",
    }.issubset(set(report.best_candidate.projected_cloud.fields))


def test_real_sample_v2_full_cloud_purity_cli_writes_best_ply_and_summary(
    tmp_path,
    real_sample_v2_scenarios,
):
    scenario = real_sample_v2_scenarios["coverage-boundary"]
    input_path = scenario.write(tmp_path)
    preview_ply = tmp_path / "full-cloud-purity.ply"
    summary_path = tmp_path / "full-cloud-purity-summary.json"

    exit_code = main(
        [
            "training",
            "real-sample-v2-full-cloud-purity",
            str(input_path),
            "--max-point-candidates",
            "24",
            "64",
            "128",
            "--temperature-candidates",
            "1.0",
            "0.75",
            "0.5",
            "0.35",
            "--preview-ply-output",
            str(preview_ply),
            "--summary-output",
            str(summary_path),
            "--viewer-path",
            "/samples/objgauss-real-sample-v2-full-cloud-purity.ply",
            "--require-pass",
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cloud = read_ply(preview_ply)
    assert summary["status"] == "real_sample_v2_full_cloud_purity_pass"
    assert summary["segmentation_target"]["selected_max_points"] == 128
    assert summary["best_candidate"]["quality"]["object_purity"] >= 0.85
    assert cloud.count == scenario.cloud.count
    assert "target_object_id" in cloud.fields
    assert "assignment_confidence" in cloud.fields
