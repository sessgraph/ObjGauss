from __future__ import annotations

import json

import pytest

from objgauss.evaluation.objectstate_model_identity_ablation import (
    DEFAULT_OBJECTSTATE_MODEL_IDENTITY_ABLATION_POLICIES,
    OBJECTSTATE_MODEL_IDENTITY_ABLATION_SCHEMA,
    objectstate_model_identity_ablation_summary,
    validate_objectstate_model_identity_ablation_summary,
)
from objgauss.evaluation.objectstate_model_identity_benchmark import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS,
)
from objgauss.evaluation.objectstate_model_identity_benchmark_report import (
    OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES,
    objectstate_model_identity_benchmark_report_scenarios,
)
from objgauss.evaluation.objectstate_model_identity_gate import OBJECTSTATE_MODEL_IDENTITY_BASELINES


def test_model_identity_ablation_compares_report_ladder_feature_policies(tmp_path):
    artifact_dir = tmp_path / "identity-ablation-artifacts"
    summary = objectstate_model_identity_ablation_summary(
        tmp_path,
        artifact_dir=artifact_dir,
        sample_id="identity-ablation-test",
        seed=11,
    )

    assert summary["schema"] == OBJECTSTATE_MODEL_IDENTITY_ABLATION_SCHEMA
    assert summary["status"] == "objectstate_model_identity_ablation_teacher_evidence_indicated"
    assert summary["scenario_count"] == 15
    assert summary["identity_pair_count"] == 60
    assert summary["difficulty_levels"] == list(
        OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES
    )
    assert summary["perturbation_kinds"] == list(OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS)
    assert [policy["policy"] for policy in summary["policies"]] == list(
        DEFAULT_OBJECTSTATE_MODEL_IDENTITY_ABLATION_POLICIES
    )
    assert validate_objectstate_model_identity_ablation_summary(summary) == summary

    variants = {variant["policy"]: variant for variant in summary["variants"]}
    assert set(variants) == set(DEFAULT_OBJECTSTATE_MODEL_IDENTITY_ABLATION_POLICIES)
    for variant in variants.values():
        assert set(variant["baseline_metrics"]) == set(OBJECTSTATE_MODEL_IDENTITY_BASELINES)
        assert set(variant["perturbation_breakdown"]) == set(
            OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS
        )
        assert variant["scenario_count"] == 15
        assert variant["identity_pair_count"] == 60
        assert variant["artifact_refs"]["raw_benchmark_summary"]

    semantic = variants["semantic"]["assignment_solver_metrics"]
    native = variants["xyz_rgb_opacity"]["assignment_solver_metrics"]
    assert variants["semantic"]["long_training_gate"]["scoped_to_policy"] == "semantic"
    assert variants["semantic"]["long_training_gate"]["candidate_ready_is_policy_scoped"] is True
    assert variants["xyz_rgb_opacity"]["long_training_gate"]["scoped_to_policy"] == "xyz_rgb_opacity"
    assert variants["xyz_rgb_opacity"]["long_training_gate"]["scope"]["native_gaussian_evidence_only"] is True
    assert semantic["identity_retrieval_at_1"] == pytest.approx(1.0)
    assert semantic["identity_margin"] > 0.0
    assert semantic["identity_retrieval_at_1"] > native["identity_retrieval_at_1"]

    ranking = summary["policy_ranking"]
    assert ranking["minimum_sufficient_evidence"]["policy"] == "semantic"
    assert ranking["minimum_sufficient_evidence"]["uses_semantic"] is True
    assert ranking["minimum_native_candidate"]["found"] is False
    assert ranking["semantic_candidate_policy_count"] == 1
    assert ranking["raw_candidate_policy_count"] > ranking["candidate_policy_count"]
    assert "rgb" in ranking["perturbation_rejected_policies"]
    assert variants["rgb"]["long_training_gate"]["status"] == "candidate_ready"
    assert variants["rgb"]["perturbation_breakdown"]["appearance"][
        "assignment_solver_metrics"
    ]["identity_retrieval_at_1"] < 0.95

    diagnostics = summary["shortcut_diagnostics"]
    assert diagnostics["teacher_evidence_layer_indicated"] is True
    assert "native_gaussian_attributes_not_sufficient_without_semantic_evidence" in diagnostics["notes"]
    assert summary["next_stage_gate"]["teacher_evidence_layer_contract_recommended"] is True
    assert summary["next_stage_gate"]["native_long_training_gate"] == "blocked"
    assert summary["next_stage_gate"]["semantic_long_training_gate"] == "candidate_ready"
    assert summary["next_stage_gate"]["long_training_allowed"] is False

    summary_path = tmp_path / "identity-ablation-summary.json"
    assert summary["artifact_refs"]["identity_ablation_summary"] == str(summary_path)
    assert summary["artifact_refs"]["identity_ablation_artifacts"] == str(artifact_dir)
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["schema"] == (
        OBJECTSTATE_MODEL_IDENTITY_ABLATION_SCHEMA
    )


def test_model_identity_ablation_requires_the_report_scenario_ladder(tmp_path):
    scenarios = objectstate_model_identity_benchmark_report_scenarios()

    with pytest.raises(ValueError, match="15 report scenarios"):
        objectstate_model_identity_ablation_summary(
            tmp_path,
            scenarios=scenarios[:1],
            policies=("xyz",),
        )
