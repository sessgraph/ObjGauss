from __future__ import annotations

from objgauss.core.io import read_ply
from objgauss.core.real_sample_v2_diagnostics import (
    REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA,
    RealSampleV2DiagnosticsReport,
    real_sample_v2_diagnostics_from_cloud,
    validate_real_sample_v2_diagnostics_summary,
)


def test_real_sample_v2_diagnostics_selects_temperature_sharpening():
    cloud = read_ply("public/samples/lego_alpha_v1_objects.ply")

    report = real_sample_v2_diagnostics_from_cloud(
        cloud,
        sample_source="public/samples/lego_alpha_v1_objects.ply",
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

    assert isinstance(report, RealSampleV2DiagnosticsReport)
    assert summary["schema"] == REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA
    assert summary["status"] == "real_sample_v2_diagnostics_pass"
    assert summary["baseline"]["renderer_joint_status"] == (
        "assignment_v2_renderer_joint_validation_fail"
    )
    assert summary["baseline"]["object_state_metrics"]["status"] == "objectstate_eval_fail"
    assert "low_assignment_confidence" in summary["baseline"]["diagnostics"]
    assert "low_object_purity" in summary["baseline"]["diagnostics"]
    assert summary["best_temperature"] == 0.5
    assert summary["best_candidate"]["renderer_joint_status"] == (
        "assignment_v2_renderer_joint_validation_pass"
    )
    assert summary["best_candidate"]["object_state_metrics"]["status"] == (
        "objectstate_eval_pass"
    )
    assert summary["best_candidate"]["object_state_metrics"]["object_purity"] >= 0.8
    assert summary["best_checkpoint"]["solver_temperature"] == 0.5
    assert summary["best_checkpoint"]["solver_state"]["step"] == 100
    assert "arrays" in summary["best_checkpoint"]["solver_state"]
    assert summary["failure_breakdown"]["confidence_delta"] > 0.0
    assert summary["failure_breakdown"]["purity_delta"] > 0.0
    assert summary["recommendation"] == {
        "decision": "temperature_sharpening_sufficient",
        "action": "set_solver_temperature",
        "solver_temperature": 0.5,
        "selection_policy": "highest_temperature_candidate_that_passes_objectstate_gate",
        "evidence_normalization": "not_required_for_current_smoke",
        "requires_geometry_unfreeze": False,
        "requires_diffusion_replay_or_rollout": False,
    }
    assert summary["non_goals"]["implements_evidence_normalization"] is False
    assert summary["non_goals"]["unfreezes_gaussian_geometry"] is False
    assert validate_real_sample_v2_diagnostics_summary(summary) is summary
