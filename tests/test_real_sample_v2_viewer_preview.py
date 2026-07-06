from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.core.io import read_ply
from objgauss.core.real_sample_v2_viewer_preview import (
    REAL_SAMPLE_V2_VIEWER_PREVIEW_SCHEMA,
    RealSampleV2ViewerPreviewReport,
    real_sample_v2_viewer_preview_from_cloud,
    validate_real_sample_v2_viewer_preview_summary,
)


def test_real_sample_v2_viewer_preview_projects_checkpoint_to_full_cloud():
    report = real_sample_v2_viewer_preview_from_cloud(
        read_ply("public/samples/lego_alpha_v1_objects.ply"),
        sample_source="public/samples/lego_alpha_v1_objects.ply",
        frame_count=2,
        max_points=24,
        image_width=12,
        image_height=12,
        iterations=100,
        learning_rate=0.4,
        temperature_candidates=(1.0, 0.75, 0.5, 0.35),
        seed=4,
        viewer_path="/samples/objgauss-real-sample-v2-viewer-preview.ply",
    )
    summary = report.as_dict()

    assert isinstance(report, RealSampleV2ViewerPreviewReport)
    assert summary["schema"] == REAL_SAMPLE_V2_VIEWER_PREVIEW_SCHEMA
    assert summary["status"] == "real_sample_v2_viewer_preview_pass"
    assert summary["source"]["source_gaussians"] == 5696
    assert summary["projection"]["projected_gaussians"] == 5696
    assert summary["projection"]["exported_gaussians"] == 5696
    assert summary["handoff"]["recommended_solver_temperature"] == 0.5
    assert summary["viewer"]["debug_route"] == (
        "/?ply=/samples/objgauss-real-sample-v2-viewer-preview.ply"
    )
    assert summary["quality"]["status"] == "full_cloud_objectstate_preview_quality_diagnostic"
    assert "low_object_purity" in summary["quality"]["diagnostics"]
    assert summary["quality"]["direct_slot_match"] > 0.89
    assert summary["quality"]["object_purity"] < 0.8
    assert {
        "object_id",
        "target_object_id",
        "target_slot",
        "assignment_confidence",
        "assignment_entropy",
        "red",
        "green",
        "blue",
    }.issubset(set(report.projected_cloud.fields))
    assert validate_real_sample_v2_viewer_preview_summary(summary) is summary


def test_real_sample_v2_viewer_preview_cli_writes_ply_and_summary(tmp_path):
    preview_ply = tmp_path / "viewer-preview.ply"
    summary_path = tmp_path / "viewer-preview-summary.json"

    exit_code = main(
        [
            "training",
            "real-sample-v2-viewer-preview",
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
            "/samples/objgauss-real-sample-v2-viewer-preview.ply",
            "--require-pass",
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    cloud = read_ply(preview_ply)
    assert summary["status"] == "real_sample_v2_viewer_preview_pass"
    assert summary["projection"]["export_object_id"] == "argmax_assignment_slot"
    assert summary["viewer"]["debug_route"] == (
        "/?ply=/samples/objgauss-real-sample-v2-viewer-preview.ply"
    )
    assert cloud.count == summary["projection"]["exported_gaussians"]
    assert "target_object_id" in cloud.fields
    assert "assignment_confidence" in cloud.fields
