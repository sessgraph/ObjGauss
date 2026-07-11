from __future__ import annotations

import json

from objgauss.cli import main
from objgauss.pipelines.real_sample_v2_model_handoff import (
    REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA,
    RealSampleV2ModelHandoffReport,
    real_sample_v2_model_handoff_from_cloud,
    render_real_sample_v2_model_handoff_html,
    validate_real_sample_v2_model_handoff_summary,
)


def test_real_sample_v2_model_handoff_restores_checkpoint_and_builds_preview(
    real_sample_v2_scenarios,
):
    scenario = real_sample_v2_scenarios["temperature-boundary"]
    report = real_sample_v2_model_handoff_from_cloud(
        scenario.cloud,
        sample_source=scenario.source,
        frame_count=2,
        max_points=24,
        image_width=12,
        image_height=12,
        iterations=100,
        learning_rate=0.4,
        temperature_candidates=(1.0, 0.75, 0.5, 0.35),
        seed=4,
    )
    summary = report.as_dict()

    assert isinstance(report, RealSampleV2ModelHandoffReport)
    assert summary["schema"] == REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA
    assert summary["status"] == "real_sample_v2_model_handoff_pass"
    assert summary["recommended_solver_temperature"] == 0.5
    assert summary["model_checkpoint"]["solver_temperature"] == 0.5
    assert summary["restore_validation"]["json_roundtrip_restored"] is True
    assert summary["restore_validation"]["renderer_joint_status"] == (
        "assignment_v2_renderer_joint_validation_pass"
    )
    assert summary["restore_validation"]["object_state_status"] == "objectstate_eval_pass"
    assert summary["effect_preview"]["point_count"] == 24
    assert len(summary["effect_preview"]["panels"]) == 2
    assert summary["effect_preview"]["panels"][0]["label"] == "baseline"
    assert summary["effect_preview"]["panels"][1]["label"] == "trained_temperature_sharpened"
    assert validate_real_sample_v2_model_handoff_summary(summary) is summary

    html = render_real_sample_v2_model_handoff_html(summary)
    assert "<svg" in html
    assert "ObjGauss Real Sample V2 Model Handoff" in html
    assert "trained_temperature_sharpened" in html


def test_real_sample_v2_handoff_cli_writes_summary_checkpoint_and_preview(
    tmp_path,
    real_sample_v2_scenarios,
):
    input_path = real_sample_v2_scenarios["temperature-boundary"].write(tmp_path)
    summary_path = tmp_path / "summary.json"
    checkpoint_path = tmp_path / "checkpoint.json"
    preview_path = tmp_path / "preview.html"

    exit_code = main(
        [
            "training",
            "real-sample-v2-handoff",
            str(input_path),
            "--temperature-candidates",
            "1.0",
            "0.75",
            "0.5",
            "0.35",
            "--summary-output",
            str(summary_path),
            "--checkpoint-output",
            str(checkpoint_path),
            "--preview-output",
            str(preview_path),
            "--require-pass",
        ]
    )

    assert exit_code == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    preview = preview_path.read_text(encoding="utf-8")
    assert summary["status"] == "real_sample_v2_model_handoff_pass"
    assert checkpoint["schema"] == "objgauss-assignment-solver-v2-checkpoint"
    assert checkpoint["solver_state"]["config"]["temperature"] == 0.5
    assert "<svg" in preview
    assert "Baseline purity" in preview
