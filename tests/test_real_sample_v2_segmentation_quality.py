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


def test_real_sample_v2_segmentation_quality_identifies_weak_slot_boundary():
    report = real_sample_v2_segmentation_quality_from_cloud(
        read_ply("public/samples/lego_alpha_v1_objects.ply"),
        sample_source="public/samples/lego_alpha_v1_objects.ply",
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
    assert summary["source"]["source_gaussians"] == 5696
    assert summary["segmentation_target"]["max_points"] == 128
    assert summary["segmentation_target"]["solver_temperature"] == 0.35
    assert summary["training_quality"]["object_purity"] >= 0.85
    assert summary["viewer"]["debug_route"] == (
        "/?ply=/samples/objgauss-real-sample-v2-segmentation-quality.ply"
    )

    global_quality = summary["global_quality"]
    assert global_quality["direct_slot_match"] > 0.989
    assert global_quality["hard_argmax_object_purity"] > 0.989
    assert global_quality["min_predicted_object_purity"] > 0.97
    assert global_quality["min_target_recall"] > 0.91
    assert global_quality["mixed_gaussians"] == 59
    assert "weak_target_recall:1" in global_quality["diagnostics"]
    assert "mixed_predicted_object:2" in global_quality["diagnostics"]
    assert "low_confidence_predicted_object:1" in global_quality["diagnostics"]
    assert "high_entropy_predicted_object:1" in global_quality["diagnostics"]

    assert summary["confusion"]["matrix"] == [
        [729, 0, 0, 7],
        [0, 529, 52, 0],
        [0, 0, 1787, 0],
        [0, 0, 0, 2592],
    ]
    target_one = summary["per_target_object"][1]
    assert target_one["target_slot"] == 1
    assert target_one["dominant_predicted_object_id"] == 1
    assert target_one["missed_count"] == 52
    assert target_one["leakage"] == [{"object_id": 2, "count": 52}]
    assert "weak_target_recall" in target_one["diagnostics"]

    predicted_one = summary["per_predicted_object"][1]
    assert predicted_one["object_id"] == 1
    assert predicted_one["purity"] == 1.0
    assert predicted_one["confidence"]["mean"] < 0.59
    assert predicted_one["entropy"]["mean"] > 0.59
    assert predicted_one["high_entropy_count"] == 431
    assert "low_confidence_predicted_object" in predicted_one["diagnostics"]
    assert "high_entropy_predicted_object" in predicted_one["diagnostics"]

    predicted_two = summary["per_predicted_object"][2]
    assert predicted_two["object_id"] == 2
    assert predicted_two["mixed_count"] == 52
    assert predicted_two["purity"] > 0.97
    assert "mixed_predicted_object" in predicted_two["diagnostics"]

    recommendation = summary["recommendation"]
    assert recommendation["decision"] == "keep_128_target_and_inspect_weak_boundaries"
    assert recommendation["action"] == "inspect_confusion_slots_before_evidence_normalization"
    assert recommendation["weak_target_slots"] == [1]
    assert recommendation["mixed_predicted_objects"] == [2]
    assert recommendation["low_confidence_predicted_objects"] == [1]
    assert recommendation["high_entropy_predicted_objects"] == [1]
    assert recommendation["requires_more_coverage"] is False
    assert recommendation["requires_geometry_unfreeze"] is False
    assert recommendation["requires_diffusion_replay_or_rollout"] is False
    assert validate_real_sample_v2_segmentation_quality_summary(summary) is summary


def test_real_sample_v2_segmentation_quality_cli_writes_ply_and_summary(tmp_path):
    preview_ply = tmp_path / "segmentation-quality.ply"
    summary_path = tmp_path / "segmentation-quality-summary.json"

    exit_code = main(
        [
            "training",
            "real-sample-v2-segmentation-quality",
            "public/samples/lego_alpha_v1_objects.ply",
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
    assert summary["global_quality"]["mixed_gaussians"] == 59
    assert summary["recommendation"]["weak_target_slots"] == [1]
    assert cloud.count == 5696
    assert "target_slot" in cloud.fields
    assert "assignment_entropy" in cloud.fields
