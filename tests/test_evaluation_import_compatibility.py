from __future__ import annotations

import objgauss.core.baseline_comparison as legacy_baseline_comparison
import objgauss.core.emergence as legacy_emergence
import objgauss.core.assignment_stability as legacy_assignment_stability
import objgauss.core.assignment_solver_v2 as canonical_assignment_solver
import objgauss.core.assignment_solver_v2_eval as legacy_assignment_solver_eval
import objgauss.core.object_state_eval as legacy_object_state_eval
import objgauss.core.object_state_benchmark as legacy_object_state_benchmark
import objgauss.core.projection as legacy_projection
import objgauss.core.objectstate_causal_gate as legacy_causal_gate
import objgauss.core.objectstate_controlled_identity_eval as legacy_identity
import objgauss.core.objectstate_controlled_intervention_eval as legacy_intervention
import objgauss.core.objectstate_controlled_prediction_eval as legacy_prediction
import objgauss.core.objectstate_controlled_real_rows as legacy_controlled_real_rows
import objgauss.core.objectstate_controlled_real_identity_eval as legacy_controlled_real_identity
import objgauss.core.objectstate_controlled_real_prediction_eval as legacy_controlled_real_prediction
import objgauss.core.objectstate_controlled_real_readiness_audit as legacy_controlled_real_readiness
import objgauss.core.objectstate_bop_reality_rows as legacy_bop_reality_rows
import objgauss.core.objectstate_identity_prediction_adapter as legacy_identity_adapter
import objgauss.core.objectstate_identity_gate as legacy_identity_gate
import objgauss.core.objectstate_predictive_gate as legacy_predictive_gate
import objgauss.core.objectstate_reality_gate as legacy_reality
import objgauss.core.objectstate_model_identity_benchmark as legacy_identity_benchmark
import objgauss.core.objectstate_model_identity_benchmark_report as legacy_identity_benchmark_report
import objgauss.core.objectstate_teacher_evidence_leakage_audit as legacy_teacher_audit
import objgauss.core.objectstate_model_identity_gate as legacy_model_identity_gate
import objgauss.core.objectstate_model_identity_ablation as legacy_model_identity_ablation
import objgauss.core.objectstate_public_interaction_reality_rows as legacy_public_interaction_rows
import objgauss.core.objectstate_real_identity_rows as legacy_real_identity_rows
import objgauss.core.objectstate_real_intervention_rows as legacy_real_intervention_rows
import objgauss.core.objectstate_real_prediction_rows as legacy_real_prediction_rows
import objgauss.core.objectstate_reality_public_rows as legacy_reality_public_rows
import objgauss.core.objectstate_reality_row_ledger as legacy_reality_row_ledger
import objgauss.core.v2_stability_diagnostics as legacy_v2_diagnostics
import objgauss.core.v2_stability_gate as legacy_v2_gate
import objgauss.mask_voting as legacy_mask_voting
import objgauss.evaluation.objectstate_controlled_identity_eval as canonical_identity
import objgauss.evaluation.baseline_comparison as canonical_baseline_comparison
import objgauss.evaluation.emergence as canonical_emergence
import objgauss.evaluation.assignment_stability as canonical_assignment_stability
import objgauss.evaluation.assignment_solver_v2_eval as canonical_assignment_solver_eval
import objgauss.evaluation.object_state_eval as canonical_object_state_eval
import objgauss.evaluation.object_state_benchmark as canonical_object_state_benchmark
import objgauss.evaluation.objectstate_causal_gate as canonical_causal_gate
import objgauss.evaluation.objectstate_controlled_intervention_eval as canonical_intervention
import objgauss.evaluation.objectstate_controlled_prediction_eval as canonical_prediction
import objgauss.datasets.objectstate_controlled_real_manifest as canonical_controlled_real_manifest
import objgauss.evaluation.objectstate_controlled_real_rows as canonical_controlled_real_rows
import objgauss.evaluation.objectstate_controlled_real_identity_eval as canonical_controlled_real_identity
import objgauss.evaluation.objectstate_controlled_real_prediction_eval as canonical_controlled_real_prediction
import objgauss.evaluation.objectstate_controlled_real_readiness_audit as canonical_controlled_real_readiness
import objgauss.evaluation.objectstate_bop_reality_rows as canonical_bop_reality_rows
import objgauss.evaluation.objectstate_identity_prediction_adapter as canonical_identity_adapter
import objgauss.evaluation.objectstate_identity_gate as canonical_identity_gate
import objgauss.evaluation.objectstate_predictive_gate as canonical_predictive_gate
import objgauss.evaluation.objectstate_reality_gate as canonical_reality
import objgauss.evaluation.objectstate_model_identity_benchmark as canonical_identity_benchmark
import objgauss.evaluation.objectstate_model_identity_benchmark_report as canonical_identity_benchmark_report
import objgauss.evaluation.objectstate_teacher_evidence_leakage_audit as canonical_teacher_audit
import objgauss.evaluation.objectstate_model_identity_gate as canonical_model_identity_gate
import objgauss.evaluation.objectstate_model_identity_ablation as canonical_model_identity_ablation
import objgauss.evaluation.objectstate_public_interaction_reality_rows as canonical_public_interaction_rows
import objgauss.evaluation.objectstate_real_identity_rows as canonical_real_identity_rows
import objgauss.evaluation.objectstate_real_intervention_rows as canonical_real_intervention_rows
import objgauss.evaluation.objectstate_real_prediction_rows as canonical_real_prediction_rows
import objgauss.evaluation.objectstate_reality_public_rows as canonical_reality_public_rows
import objgauss.evaluation.objectstate_reality_row_ledger as canonical_reality_row_ledger
import objgauss.evaluation.v2_stability_diagnostics as canonical_v2_diagnostics
import objgauss.evaluation.v2_stability_gate as canonical_v2_gate
import objgauss.evaluation.mask_vote_quality as canonical_mask_vote_quality
import objgauss.pipelines.objectstate_identity_prediction_adapter as pipeline_identity_adapter


def test_legacy_evaluation_imports_preserve_canonical_object_identity():
    module_pairs = (
        (legacy_baseline_comparison, canonical_baseline_comparison),
        (legacy_emergence, canonical_emergence),
        (legacy_assignment_stability, canonical_assignment_stability),
        (legacy_object_state_eval, canonical_object_state_eval),
        (legacy_object_state_benchmark, canonical_object_state_benchmark),
        (legacy_v2_diagnostics, canonical_v2_diagnostics),
        (legacy_v2_gate, canonical_v2_gate),
        (legacy_identity_gate, canonical_identity_gate),
        (legacy_predictive_gate, canonical_predictive_gate),
        (legacy_causal_gate, canonical_causal_gate),
        (legacy_identity, canonical_identity),
        (legacy_intervention, canonical_intervention),
        (legacy_prediction, canonical_prediction),
        (legacy_controlled_real_readiness, canonical_controlled_real_readiness),
        (legacy_controlled_real_identity, canonical_controlled_real_identity),
        (legacy_controlled_real_prediction, canonical_controlled_real_prediction),
        (legacy_identity_adapter, canonical_identity_adapter),
        (legacy_identity_benchmark, canonical_identity_benchmark),
        (legacy_identity_benchmark_report, canonical_identity_benchmark_report),
        (legacy_teacher_audit, canonical_teacher_audit),
        (legacy_model_identity_gate, canonical_model_identity_gate),
        (legacy_model_identity_ablation, canonical_model_identity_ablation),
        (legacy_public_interaction_rows, canonical_public_interaction_rows),
        (legacy_bop_reality_rows, canonical_bop_reality_rows),
        (legacy_real_identity_rows, canonical_real_identity_rows),
        (legacy_real_prediction_rows, canonical_real_prediction_rows),
        (legacy_real_intervention_rows, canonical_real_intervention_rows),
        (legacy_reality_public_rows, canonical_reality_public_rows),
        (legacy_reality_row_ledger, canonical_reality_row_ledger),
        (legacy_reality, canonical_reality),
    )

    for legacy_module, canonical_module in module_pairs:
        assert legacy_module.__all__
        for export_name in legacy_module.__all__:
            assert getattr(legacy_module, export_name) is getattr(
                canonical_module, export_name
            )


def test_mask_vote_quality_compatibility_surface_and_identity_are_exact():
    assert set(canonical_mask_vote_quality.__all__) == {"mask_vote_quality_check"}
    assert (
        legacy_projection.mask_vote_quality_check
        is canonical_mask_vote_quality.mask_vote_quality_check
    )
    assert (
        legacy_mask_voting.mask_vote_quality_check
        is canonical_mask_vote_quality.mask_vote_quality_check
    )


def test_legacy_controlled_real_rows_split_preserves_canonical_object_identity():
    manifest_exports = {
        "OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA",
        "read_objectstate_controlled_real_manifest",
        "validate_objectstate_controlled_real_manifest",
    }
    assert legacy_controlled_real_rows.__all__
    for export_name in legacy_controlled_real_rows.__all__:
        canonical_module = (
            canonical_controlled_real_manifest
            if export_name in manifest_exports
            else canonical_controlled_real_rows
        )
        assert getattr(legacy_controlled_real_rows, export_name) is getattr(
            canonical_module, export_name
        )


def test_evaluation_identity_adapter_path_preserves_pipeline_identity():
    assert set(canonical_identity_adapter.__all__) == {
        "objectstate_identity_predictions_from_trainable_artifact",
        "read_trainable_kernel_identity_source",
    }
    for export_name in canonical_identity_adapter.__all__:
        assert getattr(canonical_identity_adapter, export_name) is getattr(
            pipeline_identity_adapter, export_name
        )


def test_assignment_and_checkpoint_evaluation_compatibility_surfaces_are_exact():
    assignment_stability_exports = {
        "ASSIGNMENT_STABILITY_EVAL_SCHEMA",
        "evaluate_assignment_stability",
        "validate_assignment_stability_eval",
    }
    object_state_eval_exports = {
        "OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA",
        "evaluate_solver_decoder_object_states",
        "validate_objectstate_checkpoint_eval",
    }
    assert set(legacy_assignment_stability.__all__) == assignment_stability_exports
    assert set(canonical_assignment_stability.__all__) == assignment_stability_exports
    assert set(legacy_object_state_eval.__all__) == object_state_eval_exports
    assert set(canonical_object_state_eval.__all__) == object_state_eval_exports


def test_identity_benchmark_compatibility_surfaces_are_exact():
    assert set(legacy_identity_benchmark.__all__) == {
        "OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_SCHEMA",
        "OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_PERTURBATIONS",
        "ObjectStateModelIdentityBenchmarkThresholds",
        "ObjectStateModelIdentityBenchmarkScenario",
        "objectstate_model_identity_benchmark_summary",
        "validate_objectstate_model_identity_benchmark_summary",
    }
    assert set(legacy_identity_benchmark_report.__all__) == {
        "OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_SCHEMA",
        "OBJECTSTATE_MODEL_IDENTITY_BENCHMARK_REPORT_DIFFICULTIES",
        "write_objectstate_model_identity_benchmark_report",
        "objectstate_model_identity_benchmark_report_scenarios",
        "objectstate_model_identity_benchmark_report_difficulty_by_scenario",
        "validate_objectstate_model_identity_benchmark_report_summary",
    }


def test_teacher_audit_compatibility_surface_is_exact():
    assert set(legacy_teacher_audit.__all__) == {
        "OBJECTSTATE_TEACHER_EVIDENCE_LEAKAGE_AUDIT_SCHEMA",
        "TEACHER_EVIDENCE_LEAKAGE_AUDIT_CHECKS",
        "TeacherEvidenceLeakageAuditThresholds",
        "objectstate_teacher_evidence_leakage_audit_summary",
        "teacher_evidence_scenarios_from_audit",
        "validate_objectstate_teacher_evidence_leakage_audit_summary",
    }


def test_model_identity_gate_compatibility_surface_is_exact():
    assert set(legacy_model_identity_gate.__all__) == {
        "OBJECTSTATE_MODEL_IDENTITY_GATE_SCHEMA",
        "OBJECTSTATE_MODEL_IDENTITY_BASELINES",
        "ObjectStateModelIdentityGateThresholds",
        "objectstate_model_identity_gate_summary",
        "validate_objectstate_model_identity_gate_summary",
    }


def test_model_identity_ablation_compatibility_surface_is_exact():
    assert set(legacy_model_identity_ablation.__all__) == {
        "OBJECTSTATE_MODEL_IDENTITY_ABLATION_SCHEMA",
        "DEFAULT_OBJECTSTATE_MODEL_IDENTITY_ABLATION_POLICIES",
        "objectstate_model_identity_ablation_summary",
        "validate_objectstate_model_identity_ablation_summary",
    }


def test_public_interaction_rows_compatibility_surface_is_exact():
    assert set(legacy_public_interaction_rows.__all__) == {
        "OBJECTSTATE_PUBLIC_INTERACTION_REALITY_ROWS_SCHEMA",
        "read_objectstate_public_interaction_handoff_summary",
        "objectstate_public_interaction_reality_rows_from_handoff",
        "objectstate_public_interaction_reality_rows_summary",
        "validate_objectstate_public_interaction_reality_rows_summary",
    }


def test_reality_row_adapter_compatibility_surfaces_are_exact():
    assert set(legacy_bop_reality_rows.__all__) == {
        "OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA",
        "read_objectstate_bop_local_row_summary",
        "objectstate_bop_reality_rows_from_summary",
        "objectstate_bop_reality_rows_summary",
        "validate_objectstate_bop_reality_rows_summary",
    }
    assert set(legacy_real_identity_rows.__all__) == {
        "OBJECTSTATE_REAL_IDENTITY_ROWS_SCHEMA",
        "read_objectstate_real_identity_rows_summary",
        "objectstate_real_identity_rows_summary",
        "objectstate_real_identity_rows_from_bundle",
        "validate_objectstate_real_identity_rows_summary",
    }
    assert set(legacy_real_prediction_rows.__all__) == {
        "OBJECTSTATE_REAL_PREDICTION_ROWS_SCHEMA",
        "read_objectstate_real_prediction_rows_summary",
        "objectstate_real_prediction_rows_summary",
        "objectstate_real_prediction_rows_from_bundle",
        "validate_objectstate_real_prediction_rows_summary",
    }
    assert set(legacy_real_intervention_rows.__all__) == {
        "OBJECTSTATE_REAL_INTERVENTION_ROWS_SCHEMA",
        "read_objectstate_real_intervention_rows_summary",
        "objectstate_real_intervention_rows_summary",
        "objectstate_real_intervention_rows_from_bundle",
        "validate_objectstate_real_intervention_rows_summary",
    }
    assert set(legacy_reality_public_rows.__all__) == {
        "OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA",
        "ObjectStateRealityPublicArtifact",
        "default_objectstate_reality_public_artifacts",
        "objectstate_reality_rows_from_public_artifacts",
        "evaluate_public_artifact_reality_gate",
        "objectstate_reality_public_rows_summary",
        "validate_objectstate_reality_public_rows_summary",
        "validate_objectstate_reality_public_artifact",
    }
    assert set(legacy_reality_row_ledger.__all__) == {
        "OBJECTSTATE_REALITY_ROW_LEDGER_SCHEMA",
        "read_objectstate_reality_row_summary",
        "objectstate_reality_rows_from_summary",
        "objectstate_reality_row_ledger",
        "validate_objectstate_reality_row_ledger_summary",
    }


def test_controlled_real_evaluation_compatibility_surfaces_are_exact():
    assert set(legacy_controlled_real_readiness.__all__) == {
        "OBJECTSTATE_CONTROLLED_REAL_READINESS_AUDIT_SCHEMA",
        "CONTROLLED_REAL_READINESS_BLOCK_REASONS",
        "objectstate_controlled_real_readiness_audit_from_file",
        "objectstate_controlled_real_readiness_audit",
        "objectstate_controlled_real_readiness_markdown",
        "objectstate_controlled_real_readiness_breakdown_csv",
        "validate_objectstate_controlled_real_readiness_audit",
    }
    assert set(legacy_controlled_real_identity.__all__) == {
        "OBJECTSTATE_CONTROLLED_REAL_IDENTITY_EVAL_SCHEMA",
        "OBJECTSTATE_CONTROLLED_REAL_IDENTITY_TEACHER_EVIDENCE_SCHEMA",
        "OBJECTSTATE_CONTROLLED_REAL_IDENTITY_BASELINES",
        "objectstate_controlled_real_identity_eval_from_files",
        "read_objectstate_controlled_real_identity_teacher_evidence",
        "objectstate_controlled_real_identity_eval",
        "objectstate_controlled_real_identity_report",
        "objectstate_controlled_real_identity_accounting_csv",
        "objectstate_controlled_real_identity_pairwise_csv",
        "objectstate_controlled_real_identity_artifact_manifest",
        "validate_objectstate_controlled_real_identity_teacher_evidence",
        "validate_objectstate_controlled_real_identity_eval",
    }
    assert set(legacy_controlled_real_prediction.__all__) == {
        "OBJECTSTATE_CONTROLLED_REAL_PREDICTION_EVAL_SCHEMA",
        "objectstate_controlled_real_prediction_eval_from_files",
        "read_objectstate_controlled_real_identity_eval",
        "objectstate_controlled_real_prediction_eval",
        "objectstate_controlled_real_prediction_report",
        "objectstate_controlled_real_prediction_accounting_csv",
        "objectstate_controlled_real_prediction_errors_csv",
        "objectstate_controlled_real_prediction_artifact_manifest",
        "validate_objectstate_controlled_real_prediction_eval",
    }


def test_baseline_comparison_compatibility_surface_is_exact():
    assert set(legacy_baseline_comparison.__all__) == {
        "DEVELOPMENT_STAGE_NOTICE",
        "compare_baseline_candidates",
        "render_comparison_markdown",
        "write_comparison_markdown",
    }


def test_emergence_compatibility_surface_is_exact():
    assert set(legacy_emergence.__all__) == {
        "object_emergence_metrics",
        "object_emergence_curve",
        "mask_proxy_occlusion_delta",
        "write_emergence_curve_csv",
        "adjusted_rand_index",
    }


def test_object_state_benchmark_compatibility_surface_is_exact():
    assert set(legacy_object_state_benchmark.__all__) == {
        "OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA",
        "DEFAULT_OBJECT_STATE_BENCHMARK_THRESHOLDS",
        "object_state_stability_benchmark",
        "write_object_state_stability_benchmark",
        "validate_object_state_stability_benchmark",
    }


def test_v2_stability_evaluation_compatibility_surfaces_are_exact():
    assert set(legacy_v2_diagnostics.__all__) == {
        "V2_STABILITY_DIAGNOSTICS_SCHEMA",
        "V2_STABILITY_FAILURE_MODES",
        "IdentitySlotObservation",
        "FailureModeEvent",
        "FailureModeClassifier",
        "SyntheticStabilityDiagnosticsReport",
        "diagnose_synthetic_stability_fixture",
        "expected_slots_for_synthetic_fixture",
        "validate_synthetic_stability_diagnostics_summary",
    }
    assert set(legacy_v2_gate.__all__) == {
        "V2_STABILITY_GATE_SCHEMA",
        "V2_STABILITY_GATE_SUITE_SCHEMA",
        "V2_STABILITY_GATE_HARD_CHECKS",
        "SyntheticStabilityGateReport",
        "SyntheticStabilitySuiteGateReport",
        "evaluate_synthetic_stability_gate",
        "evaluate_synthetic_stability_suite_gate",
        "validate_synthetic_stability_gate_summary",
        "validate_synthetic_stability_suite_gate_summary",
    }


def test_objectstate_synthetic_gate_compatibility_surfaces_are_exact():
    assert set(legacy_identity_gate.__all__) == {
        "OBJECTSTATE_IDENTITY_GATE_SCHEMA",
        "OBJECTSTATE_IDENTITY_DATASET_SCHEMA",
        "ObjectStateIdentityGateThresholds",
        "ObjectStateIdentityRow",
        "ObjectStateIdentityGateReport",
        "evaluate_objectstate_identity_gate",
        "validate_objectstate_identity_gate_summary",
        "validate_objectstate_identity_gate_thresholds",
        "validate_objectstate_identity_row",
    }
    assert set(legacy_predictive_gate.__all__) == {
        "OBJECTSTATE_PREDICTIVE_GATE_SCHEMA",
        "ObjectStatePredictiveGateThresholds",
        "ObjectStatePredictiveRow",
        "ObjectStatePredictiveGateReport",
        "evaluate_objectstate_predictive_gate",
        "validate_objectstate_predictive_gate_summary",
        "validate_objectstate_predictive_gate_thresholds",
        "validate_objectstate_predictive_row",
    }
    assert set(legacy_causal_gate.__all__) == {
        "OBJECTSTATE_CAUSAL_GATE_SCHEMA",
        "OBJECTSTATE_ACTION_SCHEMA",
        "OBJECTSTATE_CAUSAL_ACTIONS",
        "ObjectStateAction",
        "ObjectStateCausalGateThresholds",
        "ObjectStateCausalRow",
        "ObjectStateCausalGateReport",
        "evaluate_objectstate_causal_gate",
        "validate_objectstate_causal_gate_summary",
        "validate_objectstate_causal_gate_thresholds",
        "validate_objectstate_action",
        "validate_objectstate_causal_row",
    }


def test_assignment_solver_eval_split_compatibility_surface_is_exact():
    checkpoint_exports = {
        "ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA",
        "assignment_solver_v2_checkpoint",
        "assignment_solver_v2_state_from_checkpoint",
        "validate_assignment_solver_v2_checkpoint",
    }
    evaluator_exports = {
        "ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA",
        "AssignmentSolverV2StabilityEvalReport",
        "evaluate_assignment_solver_v2_stability",
        "validate_assignment_solver_v2_stability_eval_summary",
    }
    assert set(legacy_assignment_solver_eval.__all__) == (
        checkpoint_exports | evaluator_exports
    )
    assert set(canonical_assignment_solver_eval.__all__) == evaluator_exports
    for name in checkpoint_exports:
        assert getattr(legacy_assignment_solver_eval, name) is getattr(
            canonical_assignment_solver, name
        )
    for name in evaluator_exports:
        assert getattr(legacy_assignment_solver_eval, name) is getattr(
            canonical_assignment_solver_eval, name
        )
