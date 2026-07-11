from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.io import read_ply
from objgauss.pipelines.real_sample_v2_segmentation_quality import (
    REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA,
    RealSampleV2SegmentationQualityReport,
    real_sample_v2_segmentation_quality_from_cloud,
    validate_real_sample_v2_segmentation_quality_summary,
)


def test_real_sample_v2_segmentation_quality_identifies_weak_slot_boundary(
    real_sample_v2_scenarios,
):
    scenario = real_sample_v2_scenarios["weight-fix"]
    report = real_sample_v2_segmentation_quality_from_cloud(
        scenario.cloud,
        sample_source=scenario.source,
        max_points=128,
        frame_count=2,
        image_width=12,
        image_height=12,
        iterations=100,
        learning_rate=0.4,
        temperature_candidates=(1.0, 0.75, 0.5, 0.35),
        seed=4,
        viewer_path="/samples/objgauss-real-sample-v2-segmentation-quality.ply",
    )
    summary = report.as_dict()

    assert isinstance(report, RealSampleV2SegmentationQualityReport)
    assert summary["schema"] == REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA
    assert summary["status"] == "real_sample_v2_segmentation_quality_pass"
    assert summary["source"]["source_gaussians"] == scenario.cloud.count
    assert summary["segmentation_target"]["max_points"] == 128
    assert summary["segmentation_target"]["solver_temperature"] == 1.0
    assert summary["training_quality"]["object_purity"] >= 0.85
    assert summary["viewer"]["debug_route"] == (
        "/?ply=/samples/objgauss-real-sample-v2-segmentation-quality.ply"
    )

    global_quality = summary["global_quality"]
    assert global_quality["direct_slot_match"] > 0.98
    assert global_quality["hard_argmax_object_purity"] > 0.98
    assert global_quality["min_predicted_object_purity"] > 0.96
    assert global_quality["min_target_recall"] > 0.96
    assert global_quality["mixed_gaussians"] == 9
    assert global_quality["diagnostics"] == ["mixed_predicted_object:1"]

    assert summary["confusion"]["matrix"] == [
        [247, 9],
        [0, 256],
    ]
    target_zero = summary["per_target_object"][0]
    assert target_zero["target_slot"] == 0
    assert target_zero["dominant_predicted_object_id"] == 0
    assert target_zero["missed_count"] == 9
    assert target_zero["leakage"] == [{"object_id": 1, "count": 9}]

    predicted_one = summary["per_predicted_object"][1]
    assert predicted_one["object_id"] == 1
    assert predicted_one["mixed_count"] == 9
    assert predicted_one["purity"] > 0.96
    assert "mixed_predicted_object" in predicted_one["diagnostics"]

    recommendation = summary["recommendation"]
    assert recommendation["decision"] == "keep_128_target_and_inspect_weak_boundaries"
    assert recommendation["action"] == "inspect_confusion_slots_before_evidence_normalization"
    assert recommendation["weak_target_slots"] == []
    assert recommendation["mixed_predicted_objects"] == [1]
    assert recommendation["low_confidence_predicted_objects"] == []
    assert recommendation["high_entropy_predicted_objects"] == []
    assert recommendation["requires_more_coverage"] is False
    assert recommendation["requires_geometry_unfreeze"] is False
    assert recommendation["requires_diffusion_replay_or_rollout"] is False
    assert validate_real_sample_v2_segmentation_quality_summary(summary) is summary


def test_real_sample_v2_segmentation_quality_cli_writes_ply_and_summary(
    tmp_path,
    real_sample_v2_scenarios,
):
    scenario = real_sample_v2_scenarios["weight-fix"]
    input_path = scenario.write(tmp_path)
    preview_ply = tmp_path / "segmentation-quality.ply"
    summary_path = tmp_path / "segmentation-quality-summary.json"

    exit_code = main(
        [
            "training",
            "real-sample-v2-segmentation-quality",
            str(input_path),
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
            "/samples/objgauss-real-sample-v2-segmentation-quality.ply",
            "--require-pass",
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cloud = read_ply(preview_ply)
    assert summary["status"] == "real_sample_v2_segmentation_quality_pass"
    assert summary["segmentation_target"]["max_points"] == 128
    assert summary["global_quality"]["mixed_gaussians"] == 9
    assert summary["recommendation"]["mixed_predicted_objects"] == [1]
    assert cloud.count == scenario.cloud.count
    assert "target_slot" in cloud.fields
    assert "assignment_entropy" in cloud.fields
