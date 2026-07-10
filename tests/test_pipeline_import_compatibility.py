from __future__ import annotations

import objgauss.core.objectstate_controlled_capture_handoff as legacy_capture_handoff
import objgauss.core.object_field as legacy_core_object_field
import objgauss.core.real_sample_v2_bounded_normalization_cross_sample as legacy_real_bounded
import objgauss.core.real_sample_v2_diagnostics as legacy_real_diagnostics
import objgauss.core.real_sample_v2_full_cloud_purity as legacy_real_purity
import objgauss.core.real_sample_v2_model_handoff as legacy_real_handoff
import objgauss.core.real_sample_v2_promoted_weights_cross_sample as legacy_real_promoted
import objgauss.core.real_sample_v2_sample_aware_weight_policy as legacy_real_weight_policy
import objgauss.core.real_sample_v2_segmentation_quality as legacy_real_segmentation
import objgauss.core.real_sample_v2_smoke as legacy_real_smoke
import objgauss.core.real_sample_v2_viewer_preview as legacy_real_viewer
import objgauss.core.real_sample_v2_weak_boundary_opt as legacy_real_weak_boundary
import objgauss.core.core_model_validation as legacy_core_model_validation
import objgauss.core.objectstate_controlled_identity_handoff as legacy_identity_handoff
import objgauss.core.objectstate_controlled_identity_evidence_package as legacy_identity_package
import objgauss.core.objectstate_controlled_identity_bundle_handoff as legacy_identity_bundle_handoff
import objgauss.core.objectstate_controlled_prediction_baseline as legacy_prediction_baseline
import objgauss.core.objectstate_controlled_prediction_evidence_package as legacy_prediction_package
import objgauss.core.objectstate_controlled_reality_candidate_template as legacy_candidate_template
import objgauss.core.objectstate_controlled_reality_bundle_handoff as legacy_reality_bundle_handoff
import objgauss.core.objectstate_controlled_reality_bundle_readiness as legacy_reality_bundle_readiness
import objgauss.core.objectstate_controlled_reality_evidence_package as legacy_reality_package
import objgauss.core.objectstate_transition_prediction_candidates as legacy_transition_prediction
import objgauss.core.objectstate_transition_intervention_candidates as legacy_transition_intervention
import objgauss.core.objectstate_transition_reality_handoff as legacy_transition_handoff
import objgauss.core.objectstate_transition_reality_evidence_package as legacy_transition_package
import objgauss.core.objectstate_phase1_evidence_ledger as legacy_phase1_ledger
import objgauss.core.objectstate_identity_prediction_adapter as legacy_identity_adapter
import objgauss.core.objectstate_identity_encoder as legacy_identity_encoder
import objgauss.core.clip_scoring as legacy_core_clip_scoring
import objgauss.clip_scoring as legacy_clip_scoring
import objgauss.core.projection as legacy_core_projection
import objgauss.mask_voting as legacy_mask_voting
import objgauss.core.semantic_slots as legacy_core_semantic_slots
import objgauss.semantic_slots as legacy_semantic_slots
import objgauss.core.trainable_kernel as legacy_trainable_kernel
import objgauss.core.training_renderer as legacy_training_renderer
import objgauss.core.gsplat_training_renderer as legacy_gsplat_training_renderer
import objgauss.core.gaussian_decoder_training as legacy_gaussian_decoder_training
import objgauss.core.solver_decoder_training as legacy_solver_decoder_training
import objgauss.core.assignment_v2_renderer_validation as legacy_assignment_renderer_validation
import objgauss.core.renderer_loss as legacy_renderer_loss
import objgauss.core.trainable_artifact as legacy_trainable_artifact
import objgauss.core.trainable_quality as legacy_trainable_quality
import objgauss.core.training_scale as legacy_training_scale
import objgauss.core.training_tensorboard as legacy_training_tensorboard
import objgauss.core.objectstate_temporal_assignment as legacy_temporal_assignment
import objgauss.core.objectstate_assignment_long_smoke as legacy_assignment_long_smoke
import objgauss.core.objectstate_temporal_assignment_contract as legacy_temporal_contract
import objgauss.core.objectstate_assignment_long_smoke_contract as legacy_long_smoke_contract
import objgauss.core.objectstate_assignment_mvp as legacy_assignment_mvp
import objgauss.core.objectstate_assignment_train as legacy_assignment_train
import objgauss.core.objectstate_assignment_generalization as legacy_assignment_generalization
import objgauss.core.objectstate_assignment_ablation as legacy_assignment_ablation
import objgauss.core.objectstate_bop_identity_handoff as legacy_bop_identity_handoff
import objgauss.core.objectstate_bop_identity_route_audit as legacy_bop_identity_route_audit
import objgauss.core.objectstate_bop_baseline_local_row_handoff as legacy_bop_baseline_local_row_handoff
import objgauss.core.objectstate_bop_baseline_candidate as legacy_bop_baseline_candidate
import objgauss.core.objectstate_bop_cross_sample_ledger as legacy_bop_cross_sample_ledger
import objgauss.core.objectstate_bop_local_row_handoff as legacy_bop_local_row_handoff
import objgauss.core.objectstate_bop_local_row_batch_handoff as legacy_bop_batch_handoff
import objgauss.core.objectstate_bop_local_row_batch_readiness as legacy_bop_batch_readiness
import objgauss.core.objectstate_bop_prediction_baseline_handoff as legacy_bop_prediction_handoff
import objgauss.core.objectstate_bop_phase1_route_audit as legacy_bop_phase1_route_audit
import objgauss.core.objectstate_bop_phase1_local_row_readiness as legacy_bop_local_readiness
import objgauss.core.objectstate_bop_rgbd_baseline_local_row_handoff as legacy_bop_rgbd_handoff
import objgauss.core.objectstate_bop_rgbd_gaussian_export as legacy_bop_rgbd_export
import objgauss.core.objectstate_bop_candidate_artifact_template as legacy_bop_candidate_template
import objgauss.core.objectstate_bop_gaussian_evidence_preflight as legacy_bop_gaussian_preflight
import objgauss.core.objectstate_bop_phase1_authoring_progress as legacy_bop_authoring_progress
import objgauss.core.objectstate_bop_real_evidence_bundle as legacy_bop_real_bundle
import objgauss.core.objectstate_public_dataset_candidates as legacy_public_candidates
import objgauss.core.objectstate_public_interaction_workspace as legacy_public_workspace
import objgauss.core.objectstate_real_evidence_bundle_ledger as legacy_real_bundle_ledger
import objgauss.core.objectstate_real_evidence_bundle_ledger_audit as legacy_real_bundle_ledger_audit
import objgauss.object_field as legacy_object_field
import objgauss.pipelines.objectstate_controlled_capture_handoff as canonical_capture_handoff
import objgauss.pipelines.real_sample_v2_bounded_normalization_cross_sample as canonical_real_bounded
import objgauss.pipelines.real_sample_v2_diagnostics as canonical_real_diagnostics
import objgauss.pipelines.real_sample_v2_full_cloud_purity as canonical_real_purity
import objgauss.pipelines.real_sample_v2_model_handoff as canonical_real_handoff
import objgauss.pipelines.real_sample_v2_promoted_weights_cross_sample as canonical_real_promoted
import objgauss.pipelines.real_sample_v2_sample_aware_weight_policy as canonical_real_weight_policy
import objgauss.pipelines.real_sample_v2_segmentation_quality as canonical_real_segmentation
import objgauss.pipelines.real_sample_v2_smoke as canonical_real_smoke
import objgauss.pipelines.real_sample_v2_viewer_preview as canonical_real_viewer
import objgauss.pipelines.real_sample_v2_weak_boundary_opt as canonical_real_weak_boundary
import objgauss.pipelines.core_model_validation as canonical_core_model_validation
import objgauss.pipelines.objectstate_controlled_identity_handoff as canonical_identity_handoff
import objgauss.pipelines.objectstate_controlled_identity_evidence_package as canonical_identity_package
import objgauss.pipelines.objectstate_controlled_identity_bundle_handoff as canonical_identity_bundle_handoff
import objgauss.pipelines.objectstate_controlled_prediction_baseline as canonical_prediction_baseline
import objgauss.pipelines.objectstate_controlled_prediction_evidence_package as canonical_prediction_package
import objgauss.pipelines.objectstate_controlled_reality_candidate_template as canonical_candidate_template
import objgauss.pipelines.objectstate_controlled_reality_bundle_handoff as canonical_reality_bundle_handoff
import objgauss.pipelines.objectstate_controlled_reality_bundle_readiness as canonical_reality_bundle_readiness
import objgauss.pipelines.objectstate_controlled_reality_evidence_package as canonical_reality_package
import objgauss.pipelines.objectstate_transition_prediction_candidates as canonical_transition_prediction
import objgauss.pipelines.objectstate_transition_intervention_candidates as canonical_transition_intervention
import objgauss.pipelines.objectstate_transition_reality_handoff as canonical_transition_handoff
import objgauss.pipelines.objectstate_transition_reality_evidence_package as canonical_transition_package
import objgauss.pipelines.objectstate_phase1_evidence_ledger as canonical_phase1_ledger
import objgauss.pipelines.objectstate_identity_prediction_adapter as canonical_identity_adapter
import objgauss.pipelines.objectstate_identity_encoder as canonical_identity_encoder
import objgauss.pipelines.clip_scoring as canonical_clip_scoring
import objgauss.evaluation.mask_vote_quality as canonical_mask_vote_quality
import objgauss.pipelines.mask_voting as canonical_mask_voting
import objgauss.pipelines.semantic_slots as canonical_semantic_slots
import objgauss.pipelines.trainable_kernel as canonical_trainable_kernel
import objgauss.pipelines.training_renderer as canonical_training_renderer
import objgauss.pipelines.gsplat_training_renderer as canonical_gsplat_training_renderer
import objgauss.pipelines.gaussian_decoder_training as canonical_gaussian_decoder_training
import objgauss.pipelines.solver_decoder_training as canonical_solver_decoder_training
import objgauss.pipelines.assignment_v2_renderer_validation as canonical_assignment_renderer_validation
import objgauss.pipelines.renderer_loss as canonical_renderer_loss
import objgauss.pipelines.trainable_artifact as canonical_trainable_artifact
import objgauss.pipelines.trainable_quality as canonical_trainable_quality
import objgauss.pipelines.training_scale as canonical_training_scale
import objgauss.pipelines.training_tensorboard as canonical_training_tensorboard
import objgauss.pipelines.objectstate_temporal_assignment as canonical_temporal_assignment
import objgauss.pipelines.objectstate_assignment_long_smoke as canonical_assignment_long_smoke
import objgauss.pipelines.objectstate_temporal_assignment_contract as canonical_temporal_contract
import objgauss.pipelines.objectstate_assignment_long_smoke_contract as canonical_long_smoke_contract
import objgauss.pipelines.objectstate_assignment_mvp as canonical_assignment_mvp
import objgauss.pipelines.objectstate_assignment_train as canonical_assignment_train
import objgauss.pipelines.objectstate_assignment_generalization as canonical_assignment_generalization
import objgauss.pipelines.objectstate_assignment_ablation as canonical_assignment_ablation
import objgauss.pipelines.objectstate_bop_identity_handoff as canonical_bop_identity_handoff
import objgauss.pipelines.objectstate_bop_identity_route_audit as canonical_bop_identity_route_audit
import objgauss.pipelines.objectstate_bop_baseline_local_row_handoff as canonical_bop_baseline_local_row_handoff
import objgauss.pipelines.objectstate_bop_baseline_candidate as canonical_bop_baseline_candidate
import objgauss.pipelines.objectstate_bop_cross_sample_ledger as canonical_bop_cross_sample_ledger
import objgauss.pipelines.objectstate_bop_local_row_handoff as canonical_bop_local_row_handoff
import objgauss.pipelines.objectstate_bop_local_row_batch_handoff as canonical_bop_batch_handoff
import objgauss.pipelines.objectstate_bop_local_row_batch_readiness as canonical_bop_batch_readiness
import objgauss.pipelines.objectstate_bop_prediction_baseline_handoff as canonical_bop_prediction_handoff
import objgauss.pipelines.objectstate_bop_phase1_route_audit as canonical_bop_phase1_route_audit
import objgauss.pipelines.objectstate_bop_phase1_local_row_readiness as canonical_bop_local_readiness
import objgauss.pipelines.objectstate_bop_rgbd_baseline_local_row_handoff as canonical_bop_rgbd_handoff
import objgauss.pipelines.objectstate_bop_rgbd_gaussian_export as canonical_bop_rgbd_export
import objgauss.pipelines.objectstate_bop_candidate_artifact_template as canonical_bop_candidate_template
import objgauss.pipelines.objectstate_bop_gaussian_evidence_preflight as canonical_bop_gaussian_preflight
import objgauss.pipelines.objectstate_bop_phase1_authoring_progress as canonical_bop_authoring_progress
import objgauss.pipelines.objectstate_bop_real_evidence_bundle as canonical_bop_real_bundle
import objgauss.pipelines.objectstate_public_dataset_candidates as canonical_public_candidates
import objgauss.pipelines.objectstate_public_interaction_workspace as canonical_public_workspace
import objgauss.pipelines.objectstate_real_evidence_bundle_ledger as canonical_real_bundle_ledger
import objgauss.pipelines.objectstate_real_evidence_bundle_ledger_audit as canonical_real_bundle_ledger_audit
import objgauss.pipelines.json_io as canonical_json_io


def test_legacy_capture_handoff_preserves_canonical_object_identity():
    assert legacy_capture_handoff.__all__
    for name in legacy_capture_handoff.__all__:
        assert getattr(legacy_capture_handoff, name) is getattr(
            canonical_capture_handoff, name
        )


def test_legacy_json_io_aliases_preserve_exact_surface_and_object_identity():
    assert set(canonical_json_io.__all__) == {"write_json"}
    assert legacy_core_object_field.write_json is canonical_json_io.write_json
    assert legacy_object_field.write_json is canonical_json_io.write_json


def test_legacy_controlled_identity_handoff_preserves_object_identity():
    assert set(legacy_identity_handoff.__all__) == {
        "OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA",
        "OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA",
        "OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA",
        "objectstate_controlled_identity_handoff",
        "validate_objectstate_controlled_identity_handoff_summary",
    }
    for name in legacy_identity_handoff.__all__:
        assert getattr(legacy_identity_handoff, name) is getattr(
            canonical_identity_handoff, name
        )


def test_legacy_controlled_identity_package_preserves_object_identity():
    assert set(legacy_identity_package.__all__) == {
        "OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA",
        "objectstate_controlled_identity_evidence_package",
        "validate_objectstate_controlled_identity_evidence_package_summary",
    }
    for name in legacy_identity_package.__all__:
        assert getattr(legacy_identity_package, name) is getattr(
            canonical_identity_package, name
        )


def test_legacy_controlled_identity_bundle_handoff_preserves_object_identity():
    assert set(legacy_identity_bundle_handoff.__all__) == {
        "OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA",
        "objectstate_controlled_identity_bundle_handoff",
        "validate_objectstate_controlled_identity_bundle_handoff_summary",
    }
    for name in legacy_identity_bundle_handoff.__all__:
        assert getattr(legacy_identity_bundle_handoff, name) is getattr(
            canonical_identity_bundle_handoff, name
        )


def test_legacy_controlled_prediction_baseline_preserves_object_identity():
    assert set(legacy_prediction_baseline.__all__) == {
        "OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA",
        "write_objectstate_controlled_prediction_baseline_candidates",
        "validate_objectstate_controlled_prediction_baseline_summary",
    }
    for name in legacy_prediction_baseline.__all__:
        assert getattr(legacy_prediction_baseline, name) is getattr(
            canonical_prediction_baseline, name
        )


def test_legacy_controlled_prediction_package_preserves_object_identity():
    assert set(legacy_prediction_package.__all__) == {
        "OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA",
        "objectstate_controlled_prediction_evidence_package",
        "validate_objectstate_controlled_prediction_evidence_package_summary",
    }
    for name in legacy_prediction_package.__all__:
        assert getattr(legacy_prediction_package, name) is getattr(
            canonical_prediction_package, name
        )


def test_legacy_controlled_candidate_template_preserves_object_identity():
    assert set(legacy_candidate_template.__all__) == {
        "OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA",
        "OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA",
        "OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA",
        "OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA",
        "OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA",
        "write_objectstate_controlled_reality_candidate_templates",
        "write_objectstate_controlled_reality_candidate_templates_from_manifest",
        "finalize_objectstate_controlled_reality_candidate_templates",
        "finalize_objectstate_controlled_prediction_candidate_template",
        "validate_objectstate_controlled_reality_candidate_template_summary",
        "validate_objectstate_controlled_reality_candidate_finalize_summary",
        "validate_objectstate_controlled_prediction_candidate_finalize_summary",
        "validate_objectstate_controlled_prediction_candidates_template",
        "validate_objectstate_controlled_intervention_candidates_template",
    }
    for name in legacy_candidate_template.__all__:
        assert getattr(legacy_candidate_template, name) is getattr(
            canonical_candidate_template, name
        )


def test_legacy_reality_bundle_handoff_preserves_object_identity():
    assert set(legacy_reality_bundle_handoff.__all__) == {
        "OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA",
        "objectstate_controlled_reality_bundle_handoff",
        "validate_objectstate_controlled_reality_bundle_handoff_summary",
    }
    for name in legacy_reality_bundle_handoff.__all__:
        assert getattr(legacy_reality_bundle_handoff, name) is getattr(
            canonical_reality_bundle_handoff, name
        )


def test_legacy_reality_bundle_readiness_preserves_object_identity():
    assert set(legacy_reality_bundle_readiness.__all__) == {
        "OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_READINESS_SCHEMA",
        "objectstate_controlled_reality_bundle_readiness",
        "validate_objectstate_controlled_reality_bundle_readiness_summary",
    }
    for name in legacy_reality_bundle_readiness.__all__:
        assert getattr(legacy_reality_bundle_readiness, name) is getattr(
            canonical_reality_bundle_readiness, name
        )


def test_legacy_reality_package_preserves_object_identity():
    assert set(legacy_reality_package.__all__) == {
        "OBJECTSTATE_CONTROLLED_REALITY_EVIDENCE_PACKAGE_SCHEMA",
        "objectstate_controlled_reality_evidence_package",
        "validate_objectstate_controlled_reality_evidence_package_summary",
    }
    for name in legacy_reality_package.__all__:
        assert getattr(legacy_reality_package, name) is getattr(
            canonical_reality_package, name
        )


def test_legacy_transition_prediction_candidates_preserve_object_identity():
    assert set(legacy_transition_prediction.__all__) == {
        "OBJECTSTATE_TRANSITION_PREDICTION_CANDIDATES_SCHEMA",
        "OBJECTSTATE_TRANSITION_PREDICTION_POLICIES",
        "objectstate_transition_prediction_candidates",
        "objectstate_transition_prediction_candidates_summary",
        "write_objectstate_transition_prediction_candidates",
        "validate_objectstate_transition_prediction_candidates_summary",
    }
    for name in legacy_transition_prediction.__all__:
        assert getattr(legacy_transition_prediction, name) is getattr(
            canonical_transition_prediction, name
        )


def test_legacy_transition_intervention_candidates_preserve_object_identity():
    assert set(legacy_transition_intervention.__all__) == {
        "OBJECTSTATE_TRANSITION_INTERVENTION_CANDIDATES_SCHEMA",
        "OBJECTSTATE_TRANSITION_INTERVENTION_POLICIES",
        "objectstate_transition_intervention_candidates",
        "objectstate_transition_intervention_candidates_summary",
        "write_objectstate_transition_intervention_candidates",
        "validate_objectstate_transition_intervention_candidates_summary",
    }
    for name in legacy_transition_intervention.__all__:
        assert getattr(legacy_transition_intervention, name) is getattr(
            canonical_transition_intervention, name
        )


def test_legacy_transition_handoff_preserves_object_identity():
    assert set(legacy_transition_handoff.__all__) == {
        "OBJECTSTATE_TRANSITION_REALITY_HANDOFF_SCHEMA",
        "objectstate_transition_reality_handoff",
        "write_objectstate_transition_reality_handoff",
        "validate_objectstate_transition_reality_handoff_summary",
    }
    for name in legacy_transition_handoff.__all__:
        assert getattr(legacy_transition_handoff, name) is getattr(
            canonical_transition_handoff, name
        )


def test_legacy_transition_package_preserves_object_identity():
    assert set(legacy_transition_package.__all__) == {
        "OBJECTSTATE_TRANSITION_REALITY_EVIDENCE_PACKAGE_SCHEMA",
        "objectstate_transition_reality_evidence_package",
        "validate_objectstate_transition_reality_evidence_package_summary",
    }
    for name in legacy_transition_package.__all__:
        assert getattr(legacy_transition_package, name) is getattr(
            canonical_transition_package, name
        )


def test_legacy_phase1_ledger_preserves_object_identity():
    assert set(legacy_phase1_ledger.__all__) == {
        "OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA",
        "IDENTITY_EVIDENCE_PACKAGE_SUMMARY_FILENAMES",
        "PREDICTION_EVIDENCE_PACKAGE_SUMMARY_FILENAMES",
        "REALITY_EVIDENCE_PACKAGE_SUMMARY_FILENAMES",
        "TRANSITION_REALITY_EVIDENCE_PACKAGE_SUMMARY_FILENAMES",
        "objectstate_phase1_evidence_ledger",
        "validate_objectstate_phase1_evidence_ledger_summary",
    }
    for name in legacy_phase1_ledger.__all__:
        assert getattr(legacy_phase1_ledger, name) is getattr(
            canonical_phase1_ledger, name
        )


def test_legacy_identity_adapter_preserves_pipeline_object_identity():
    assert set(legacy_identity_adapter.__all__) == {
        "objectstate_identity_predictions_from_trainable_artifact",
        "read_trainable_kernel_identity_source",
    }
    for name in legacy_identity_adapter.__all__:
        assert getattr(legacy_identity_adapter, name) is getattr(
            canonical_identity_adapter, name
        )


def test_legacy_trainable_artifact_preserves_pipeline_object_identity():
    assert set(legacy_trainable_artifact.__all__) == {
        "TRAINABLE_KERNEL_MODEL_ARTIFACT_SCHEMA",
        "trainable_kernel_model_artifact",
        "write_trainable_kernel_model_artifact",
        "validate_trainable_kernel_model_artifact",
    }
    for name in legacy_trainable_artifact.__all__:
        assert getattr(legacy_trainable_artifact, name) is getattr(
            canonical_trainable_artifact, name
        )


def test_legacy_temporal_assignment_preserves_pipeline_object_identity():
    assert set(legacy_temporal_assignment.__all__) == {
        "OBJECTSTATE_TEMPORAL_ASSIGNMENT_SCHEMA",
        "objectstate_temporal_assignment_summary",
        "validate_objectstate_temporal_assignment_summary",
    }
    for name in legacy_temporal_assignment.__all__:
        assert getattr(legacy_temporal_assignment, name) is getattr(
            canonical_temporal_assignment, name
        )


def test_legacy_assignment_long_smoke_preserves_pipeline_object_identity():
    assert set(legacy_assignment_long_smoke.__all__) == {
        "OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SCHEMA",
        "objectstate_assignment_long_smoke_summary",
        "validate_objectstate_assignment_long_smoke_summary",
    }
    for name in legacy_assignment_long_smoke.__all__:
        assert getattr(legacy_assignment_long_smoke, name) is getattr(
            canonical_assignment_long_smoke, name
        )


def test_legacy_temporal_contract_preserves_pipeline_object_identity():
    assert set(legacy_temporal_contract.__all__) == {
        "OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SCHEMA",
        "OBJECTSTATE_TEMPORAL_ASSIGNMENT_CONTRACT_SUMMARY_SCHEMA",
        "OBJECTSTATE_TEMPORAL_ASSIGNMENT_REQUIRED_POLICY",
        "OBJECTSTATE_TEMPORAL_ASSIGNMENT_INPUTS",
        "OBJECTSTATE_TEMPORAL_ASSIGNMENT_LOSS_TERMS",
        "OBJECTSTATE_TEMPORAL_ASSIGNMENT_METRICS",
        "ObjectStateTemporalAssignmentContractThresholds",
        "objectstate_temporal_assignment_contract_summary",
        "validate_objectstate_temporal_assignment_contract_summary",
    }
    for name in legacy_temporal_contract.__all__:
        assert getattr(legacy_temporal_contract, name) is getattr(
            canonical_temporal_contract, name
        )


def test_legacy_long_smoke_contract_preserves_pipeline_object_identity():
    assert set(legacy_long_smoke_contract.__all__) == {
        "OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SCHEMA",
        "OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_CONTRACT_SUMMARY_SCHEMA",
        "OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_REQUIRED_POLICY",
        "OBJECTSTATE_ASSIGNMENT_LONG_SMOKE_SUCCESS_CRITERIA",
        "ObjectStateAssignmentLongSmokeContractThresholds",
        "objectstate_assignment_long_smoke_contract_summary",
        "validate_objectstate_assignment_long_smoke_contract_summary",
    }
    for name in legacy_long_smoke_contract.__all__:
        assert getattr(legacy_long_smoke_contract, name) is getattr(
            canonical_long_smoke_contract, name
        )


def test_legacy_bop_identity_handoff_preserves_canonical_object_identity():
    assert set(legacy_bop_identity_handoff.__all__) == {
        "OBJECTSTATE_BOP_IDENTITY_HANDOFF_SCHEMA",
        "objectstate_bop_identity_handoff",
        "validate_objectstate_bop_identity_handoff_summary",
    }
    for name in legacy_bop_identity_handoff.__all__:
        assert getattr(legacy_bop_identity_handoff, name) is getattr(
            canonical_bop_identity_handoff, name
        )


def test_legacy_bop_identity_route_audit_preserves_object_identity():
    assert set(legacy_bop_identity_route_audit.__all__) == {
        "OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA",
        "objectstate_bop_identity_route_audit",
        "validate_objectstate_bop_identity_route_audit_summary",
    }
    for name in legacy_bop_identity_route_audit.__all__:
        assert getattr(legacy_bop_identity_route_audit, name) is getattr(
            canonical_bop_identity_route_audit, name
        )


def test_legacy_bop_phase1_route_audit_preserves_object_identity():
    assert set(legacy_bop_phase1_route_audit.__all__) == {
        "OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA",
        "objectstate_bop_phase1_route_audit",
        "validate_objectstate_bop_phase1_route_audit_summary",
    }
    for name in legacy_bop_phase1_route_audit.__all__:
        assert getattr(legacy_bop_phase1_route_audit, name) is getattr(
            canonical_bop_phase1_route_audit, name
        )


def test_legacy_bop_local_readiness_preserves_object_identity():
    assert set(legacy_bop_local_readiness.__all__) == {
        "OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA",
        "objectstate_bop_phase1_local_row_readiness",
        "validate_objectstate_bop_phase1_local_row_readiness_summary",
    }
    for name in legacy_bop_local_readiness.__all__:
        assert getattr(legacy_bop_local_readiness, name) is getattr(
            canonical_bop_local_readiness, name
        )


def test_legacy_bop_prediction_handoff_preserves_canonical_object_identity():
    assert set(legacy_bop_prediction_handoff.__all__) == {
        "OBJECTSTATE_BOP_PREDICTION_BASELINE_HANDOFF_SCHEMA",
        "objectstate_bop_prediction_baseline_handoff",
        "validate_objectstate_bop_prediction_baseline_handoff_summary",
    }
    for name in legacy_bop_prediction_handoff.__all__:
        assert getattr(legacy_bop_prediction_handoff, name) is getattr(
            canonical_bop_prediction_handoff, name
        )


def test_legacy_bop_local_row_handoff_preserves_canonical_object_identity():
    assert set(legacy_bop_local_row_handoff.__all__) == {
        "OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA",
        "objectstate_bop_local_row_handoff",
        "validate_objectstate_bop_local_row_handoff_summary",
    }
    for name in legacy_bop_local_row_handoff.__all__:
        assert getattr(legacy_bop_local_row_handoff, name) is getattr(
            canonical_bop_local_row_handoff, name
        )


def test_legacy_bop_batch_handoff_preserves_canonical_object_identity():
    handoff_names = {
        "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_HANDOFF_SCHEMA",
        "objectstate_bop_local_row_batch_handoff",
        "validate_objectstate_bop_local_row_batch_handoff_summary",
    }
    assert set(legacy_bop_batch_handoff.__all__) == {
        "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA",
        "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_HANDOFF_SCHEMA",
        "read_objectstate_bop_local_row_batch_spec",
        "objectstate_bop_local_row_batch_handoff",
        "validate_objectstate_bop_local_row_batch_spec",
        "validate_objectstate_bop_local_row_batch_handoff_summary",
    }
    for name in handoff_names:
        assert getattr(legacy_bop_batch_handoff, name) is getattr(
            canonical_bop_batch_handoff, name
        )


def test_legacy_bop_batch_readiness_preserves_object_identity():
    assert set(legacy_bop_batch_readiness.__all__) == {
        "OBJECTSTATE_BOP_LOCAL_ROW_BATCH_READINESS_SCHEMA",
        "objectstate_bop_local_row_batch_readiness",
        "validate_objectstate_bop_local_row_batch_readiness_summary",
    }
    for name in legacy_bop_batch_readiness.__all__:
        assert getattr(legacy_bop_batch_readiness, name) is getattr(
            canonical_bop_batch_readiness, name
        )


def test_legacy_bop_baseline_local_row_handoff_preserves_object_identity():
    assert set(legacy_bop_baseline_local_row_handoff.__all__) == {
        "OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA",
        "objectstate_bop_baseline_local_row_handoff",
        "validate_objectstate_bop_baseline_local_row_handoff_summary",
    }
    for name in legacy_bop_baseline_local_row_handoff.__all__:
        assert getattr(legacy_bop_baseline_local_row_handoff, name) is getattr(
            canonical_bop_baseline_local_row_handoff, name
        )


def test_legacy_bop_baseline_candidate_preserves_object_identity():
    assert set(legacy_bop_baseline_candidate.__all__) == {
        "OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA",
        "write_objectstate_bop_gaussian_centroid_baseline_candidate",
        "validate_objectstate_bop_baseline_candidate_summary",
    }
    for name in legacy_bop_baseline_candidate.__all__:
        assert getattr(legacy_bop_baseline_candidate, name) is getattr(
            canonical_bop_baseline_candidate, name
        )


def test_legacy_bop_cross_sample_ledger_preserves_object_identity():
    assert set(legacy_bop_cross_sample_ledger.__all__) == {
        "OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA",
        "objectstate_bop_cross_sample_ledger",
        "validate_objectstate_bop_cross_sample_ledger_summary",
    }
    for name in legacy_bop_cross_sample_ledger.__all__:
        assert getattr(legacy_bop_cross_sample_ledger, name) is getattr(
            canonical_bop_cross_sample_ledger, name
        )


def test_legacy_bop_rgbd_handoff_preserves_canonical_object_identity():
    assert set(legacy_bop_rgbd_handoff.__all__) == {
        "OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA",
        "objectstate_bop_rgbd_baseline_local_row_handoff",
        "validate_objectstate_bop_rgbd_baseline_local_row_handoff_summary",
    }
    for name in legacy_bop_rgbd_handoff.__all__:
        assert getattr(legacy_bop_rgbd_handoff, name) is getattr(
            canonical_bop_rgbd_handoff, name
        )


def test_legacy_bop_rgbd_export_preserves_canonical_object_identity():
    assert set(legacy_bop_rgbd_export.__all__) == {
        "OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA",
        "objectstate_bop_rgbd_gaussian_export",
        "validate_objectstate_bop_rgbd_gaussian_export_summary",
    }
    for name in legacy_bop_rgbd_export.__all__:
        assert getattr(legacy_bop_rgbd_export, name) is getattr(
            canonical_bop_rgbd_export, name
        )


def test_legacy_bop_candidate_template_preserves_object_identity():
    assert set(legacy_bop_candidate_template.__all__) == {
        "OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA",
        "OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA",
        "OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_FINALIZE_SCHEMA",
        "write_objectstate_bop_candidate_artifact_template",
        "finalize_objectstate_bop_candidate_artifact_template",
        "validate_objectstate_bop_candidate_artifact_template",
        "validate_objectstate_bop_candidate_artifact_finalize_summary",
        "validate_objectstate_bop_candidate_artifact_template_summary",
    }
    for name in legacy_bop_candidate_template.__all__:
        assert getattr(legacy_bop_candidate_template, name) is getattr(
            canonical_bop_candidate_template, name
        )


def test_legacy_bop_authoring_progress_preserves_object_identity():
    assert set(legacy_bop_authoring_progress.__all__) == {
        "OBJECTSTATE_BOP_PHASE1_AUTHORING_PROGRESS_SCHEMA",
        "objectstate_bop_phase1_authoring_progress",
        "validate_objectstate_bop_phase1_authoring_progress_summary",
    }
    for name in legacy_bop_authoring_progress.__all__:
        assert getattr(legacy_bop_authoring_progress, name) is getattr(
            canonical_bop_authoring_progress, name
        )


def test_legacy_public_candidates_preserve_object_identity():
    assert set(legacy_public_candidates.__all__) == {
        "OBJECTSTATE_PUBLIC_DATASET_CANDIDATES_SCHEMA",
        "OBJECTSTATE_PUBLIC_DATASET_CANDIDATE_SCHEMA",
        "OBJECTSTATE_PUBLIC_INTERACTION_ROUTE_AUDIT_SCHEMA",
        "OBJECTSTATE_PUBLIC_DATASET_GATE_KINDS",
        "OBJECTSTATE_PUBLIC_DATASET_COVERAGE_VALUES",
        "OBJECTSTATE_PUBLIC_DATASET_SOURCE_KINDS",
        "ObjectStatePublicDatasetCandidate",
        "default_objectstate_public_dataset_candidates",
        "objectstate_public_dataset_candidates_audit",
        "objectstate_public_dataset_candidates_markdown",
        "objectstate_public_interaction_route_audit",
        "objectstate_public_interaction_route_markdown",
        "validate_objectstate_public_dataset_candidates_audit",
        "validate_objectstate_public_interaction_route_audit",
        "validate_objectstate_public_dataset_candidate",
    }
    for name in legacy_public_candidates.__all__:
        assert getattr(legacy_public_candidates, name) is getattr(
            canonical_public_candidates, name
        )


def test_legacy_public_workspace_preserves_object_identity():
    assert set(legacy_public_workspace.__all__) == {
        "OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_SCHEMA",
        "OBJECTSTATE_PUBLIC_INTERACTION_WORKSPACE_PROGRESS_SCHEMA",
        "OBJECTSTATE_PUBLIC_INTERACTION_CLIP_CSV_ADAPTER_SCHEMA",
        "OBJECTSTATE_PUBLIC_INTERACTION_CLIP_CSV_HEADER",
        "write_objectstate_public_interaction_workspace",
        "write_objectstate_public_interaction_clip_csv_bundle",
        "objectstate_public_interaction_workspace_progress",
        "validate_objectstate_public_interaction_workspace_summary",
        "validate_objectstate_public_interaction_workspace_progress_summary",
        "validate_objectstate_public_interaction_clip_csv_adapter_summary",
    }
    for name in legacy_public_workspace.__all__:
        assert getattr(legacy_public_workspace, name) is getattr(
            canonical_public_workspace, name
        )


def test_legacy_bop_real_bundle_preserves_object_identity():
    assert set(legacy_bop_real_bundle.__all__) == {
        "OBJECTSTATE_BOP_REAL_EVIDENCE_BUNDLE_ADAPTER_SCHEMA",
        "read_objectstate_bop_real_evidence_bundle_adapter_summary",
        "objectstate_bop_real_evidence_bundle_adapter_summary_from_files",
        "objectstate_bop_real_evidence_bundle_adapter_summary",
        "objectstate_bop_real_evidence_bundle_from_summaries",
        "validate_objectstate_bop_real_evidence_bundle_adapter_summary",
    }
    for name in legacy_bop_real_bundle.__all__:
        assert getattr(legacy_bop_real_bundle, name) is getattr(
            canonical_bop_real_bundle, name
        )


def test_legacy_real_bundle_ledger_chain_preserves_object_identity():
    assert set(legacy_real_bundle_ledger.__all__) == {
        "OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_SCHEMA",
        "write_objectstate_real_evidence_bundle_ledger",
        "validate_objectstate_real_evidence_bundle_ledger_summary",
    }
    assert set(legacy_real_bundle_ledger_audit.__all__) == {
        "OBJECTSTATE_REAL_EVIDENCE_BUNDLE_LEDGER_PACKAGE_AUDIT_SCHEMA",
        "objectstate_real_evidence_bundle_ledger_package_audit",
        "validate_objectstate_real_evidence_bundle_ledger_package_audit",
    }
    for legacy_module, canonical_module in (
        (legacy_real_bundle_ledger, canonical_real_bundle_ledger),
        (legacy_real_bundle_ledger_audit, canonical_real_bundle_ledger_audit),
    ):
        for name in legacy_module.__all__:
            assert getattr(legacy_module, name) is getattr(canonical_module, name)


def test_legacy_assignment_experiment_chain_preserves_object_identity():
    assert set(legacy_assignment_train.__all__) == {
        "OBJECTSTATE_ASSIGNMENT_TRAIN_DATASET_SCHEMA",
        "OBJECTSTATE_ASSIGNMENT_TRAIN_RUN_SCHEMA",
        "objectstate_assignment_train_dataset_summary",
        "objectstate_assignment_train_smoke",
        "validate_objectstate_assignment_train_dataset_summary",
        "validate_objectstate_assignment_train_run_summary",
    }
    assert set(legacy_assignment_generalization.__all__) == {
        "OBJECTSTATE_ASSIGNMENT_GENERALIZATION_SCHEMA",
        "objectstate_assignment_generalization_summary",
        "validate_objectstate_assignment_generalization_summary",
    }
    assert set(legacy_assignment_ablation.__all__) == {
        "OBJECTSTATE_ASSIGNMENT_ABLATION_SCHEMA",
        "DEFAULT_ASSIGNMENT_ABLATION_POLICIES",
        "objectstate_assignment_ablation_summary",
        "validate_objectstate_assignment_ablation_summary",
    }
    for legacy_module, canonical_module in (
        (legacy_assignment_train, canonical_assignment_train),
        (legacy_assignment_generalization, canonical_assignment_generalization),
        (legacy_assignment_ablation, canonical_assignment_ablation),
    ):
        for name in legacy_module.__all__:
            assert getattr(legacy_module, name) is getattr(canonical_module, name)


def test_legacy_bop_gaussian_preflight_preserves_object_identity():
    assert set(legacy_bop_gaussian_preflight.__all__) == {
        "OBJECTSTATE_BOP_GAUSSIAN_EVIDENCE_PREFLIGHT_SCHEMA",
        "objectstate_bop_gaussian_evidence_preflight",
        "validate_objectstate_bop_gaussian_evidence_preflight_summary",
    }
    for name in legacy_bop_gaussian_preflight.__all__:
        assert getattr(legacy_bop_gaussian_preflight, name) is getattr(
            canonical_bop_gaussian_preflight, name
        )


def test_legacy_assignment_mvp_preserves_object_identity():
    assert set(legacy_assignment_mvp.__all__) == {
        "OBJECTSTATE_ASSIGNMENT_MVP_SCHEMA",
        "objectstate_assignment_mvp_summary",
        "validate_objectstate_assignment_mvp_summary",
    }
    for name in legacy_assignment_mvp.__all__:
        assert getattr(legacy_assignment_mvp, name) is getattr(
            canonical_assignment_mvp, name
        )


def test_legacy_identity_encoder_preserves_object_identity():
    assert set(legacy_identity_encoder.__all__) == {
        "OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA",
        "OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA",
        "ObjectStateIdentityEncoderConfig",
        "ObjectStateIdentityEncoderState",
        "ObjectStateIdentityContrastiveLoss",
        "ObjectStateIdentityEncoderTrainingResult",
        "train_objectstate_identity_encoder",
        "initialize_objectstate_identity_encoder_state",
        "objectstate_identity_encoder_features",
        "validate_objectstate_identity_encoder_config",
        "validate_objectstate_identity_encoder_state",
        "validate_objectstate_identity_encoder_training_summary",
    }
    for name in legacy_identity_encoder.__all__:
        assert getattr(legacy_identity_encoder, name) is getattr(
            canonical_identity_encoder, name
        )


def test_semantic_slots_compatibility_preserves_exact_surface_and_identity():
    expected = {
        "DEFAULT_SLOT_BACKGROUND_LABELS",
        "SlotAlignmentResult",
        "align_mask_manifest_slots",
    }
    for module in (
        legacy_core_semantic_slots,
        legacy_semantic_slots,
        canonical_semantic_slots,
    ):
        assert set(module.__all__) == expected
        for name in expected:
            assert getattr(module, name) is getattr(canonical_semantic_slots, name)


def test_clip_scoring_compatibility_preserves_exact_surface_and_identity():
    expected = {
        "CLIP_LABEL_PRESETS",
        "DEFAULT_BACKGROUND_LABELS",
        "DEFAULT_PROMPT_TEMPLATES",
        "ClipMaskScorer",
        "ClipScoringResult",
        "HashClipMaskScorer",
        "TransformersClipMaskScorer",
        "read_clip_labels",
        "score_mask_manifest_with_clip",
    }
    for module in (
        legacy_core_clip_scoring,
        legacy_clip_scoring,
        canonical_clip_scoring,
    ):
        assert set(module.__all__) == expected
        for name in expected:
            assert getattr(module, name) is getattr(canonical_clip_scoring, name)


def test_mask_voting_compatibility_preserves_exact_surfaces_and_identity():
    historical_surface = (
        "MaskTrainingResult",
        "MaskVoteResult",
        "Projection",
        "depth_visibility_diagnostic",
        "mask_vote_quality_audit",
        "mask_vote_quality_check",
        "mask_vote_targets",
        "project_points",
        "projection_loss",
        "train_object_field_from_votes",
        "training_summary",
        "vote_masks_to_gaussians",
    )
    core_surface = (
        "MaskVoteResult",
        "Projection",
        "mask_vote_quality_audit",
        "mask_vote_targets",
        "project_points",
        "projection_loss",
    )
    pipeline_surface = (
        "MaskTrainingResult",
        "vote_masks_to_gaussians",
        "depth_visibility_diagnostic",
        "train_object_field_from_votes",
        "training_summary",
    )
    evaluation_surface = ("mask_vote_quality_check",)

    assert legacy_core_projection.__all__ == historical_surface
    assert legacy_mask_voting.__all__ == historical_surface
    assert canonical_mask_voting.__all__ == pipeline_surface

    for name in core_surface:
        assert getattr(legacy_mask_voting, name) is getattr(legacy_core_projection, name)
    for name in pipeline_surface:
        canonical = getattr(canonical_mask_voting, name)
        assert getattr(legacy_core_projection, name) is canonical
        assert getattr(legacy_mask_voting, name) is canonical
    for name in evaluation_surface:
        canonical = getattr(canonical_mask_vote_quality, name)
        assert getattr(legacy_core_projection, name) is canonical
        assert getattr(legacy_mask_voting, name) is canonical


def test_legacy_training_renderer_chain_preserves_exact_surfaces_and_identity():
    expected_surfaces = (
        (
            legacy_trainable_kernel,
            canonical_trainable_kernel,
            {
                "TRAINABLE_KERNEL_MVP_SCHEMA",
                "TRAINABLE_IMAGE_TARGET_CONTRACT_SCHEMA",
                "TRAINABLE_IMAGE_TARGET_SCHEMA",
                "TRAINING_IMAGE_RENDERER_POINT",
                "TRAINING_IMAGE_RENDERER_GSPLAT",
                "TRAINING_IMAGE_RENDERERS",
                "TrainableKernelCamera",
                "TrainableKernelImageTarget",
                "TrainableKernelFrame",
                "TrainableKernelLoss",
                "TrainableKernelResult",
                "TrainableKernelSample",
                "train_kernel_mvp",
                "train_kernel_mvp_from_cloud",
                "trainable_kernel_sample_from_cloud",
                "make_trainable_kernel_mvp_fixture",
                "bind_image_targets_to_frames",
                "make_trainable_image_target",
                "image_target_contract_summary",
                "validate_trainable_image_target",
                "validate_image_target_contract_summary",
            },
        ),
        (
            legacy_training_renderer,
            canonical_training_renderer,
            {
                "TRAINING_RENDERER_API_SCHEMA",
                "CPU_IMAGE_SPLAT_RENDERER",
                "CPU_IMAGE_SPLAT_GRADIENT_PATH",
                "TrainingRendererFrameLoss",
                "TrainingRendererLossResult",
                "evaluate_training_renderer_loss",
                "validate_training_renderer_summary",
            },
        ),
        (
            legacy_gsplat_training_renderer,
            canonical_gsplat_training_renderer,
            {
                "GSPLAT_RENDERER",
                "GSPLAT_GRADIENT_PATH",
                "GSPLAT_AVAILABILITY_SCHEMA",
                "GSPLAT_TRAINING_INPUT_SCHEMA",
                "GSPLAT_SYNTHETIC_GAUSSIAN_POLICY",
                "GsplatRendererAvailability",
                "GsplatTrainingInput",
                "gsplat_renderer_availability",
                "build_gsplat_training_input",
                "build_gsplat_training_input_from_object_state",
                "evaluate_gsplat_training_renderer_loss",
            },
        ),
        (
            legacy_gaussian_decoder_training,
            canonical_gaussian_decoder_training,
            {
                "OBJECT_STATE_GAUSSIAN_DECODER_STATE_SCHEMA",
                "OBJECT_STATE_GAUSSIAN_DECODER_TRAINING_SCHEMA",
                "ObjectStateGaussianDecoderState",
                "ObjectStateGaussianDecoderLoss",
                "ObjectStateGaussianDecoderTrainingResult",
                "initialize_object_state_gaussian_decoder",
                "train_object_state_gaussian_decoder",
                "validate_object_state_gaussian_decoder_state",
                "object_state_gaussian_decoder_state_from_dict",
            },
        ),
        (
            legacy_solver_decoder_training,
            canonical_solver_decoder_training,
            {
                "SOLVER_DECODER_JOINT_TRAINING_SCHEMA",
                "SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA",
                "SolverDecoderJointLoss",
                "SolverDecoderJointTrainingResult",
                "train_solver_decoder_joint",
                "solver_decoder_joint_checkpoint",
                "validate_solver_decoder_joint_checkpoint",
                "solver_decoder_joint_states_from_dict",
            },
        ),
        (
            legacy_assignment_renderer_validation,
            canonical_assignment_renderer_validation,
            {
                "ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA",
                "AssignmentV2RendererJointValidationReport",
                "evaluate_assignment_v2_renderer_joint",
                "validate_assignment_v2_renderer_joint_summary",
            },
        ),
        (
            legacy_renderer_loss,
            canonical_renderer_loss,
            {
                "RENDERER_LOSS_BOUNDARY_SCHEMA",
                "RendererLossBoundaryReport",
                "renderer_loss_boundary_report",
                "validate_renderer_loss_boundary_summary",
            },
        ),
    )
    for legacy_module, canonical_module, expected in expected_surfaces:
        assert set(legacy_module.__all__) == expected
        assert set(canonical_module.__all__) == expected
        for name in expected:
            assert getattr(legacy_module, name) is getattr(canonical_module, name)


def test_legacy_small_training_orchestration_preserves_object_identity():
    expected_surfaces = (
        (
            legacy_core_model_validation,
            canonical_core_model_validation,
            {
                "CORE_MODEL_TRAIN_VALIDATE_SCHEMA",
                "CoreModelTrainValidateReport",
                "core_model_train_validate_report",
                "validate_core_model_train_validate_summary",
            },
        ),
        (
            legacy_trainable_quality,
            canonical_trainable_quality,
            {
                "TRAINABLE_QUALITY_REPORT_SCHEMA",
                "trainable_quality_report",
                "write_trainable_quality_report",
                "validate_trainable_quality_report",
            },
        ),
        (
            legacy_training_scale,
            canonical_training_scale,
            {
                "TRAINING_SCALE_PLAN_SCHEMA",
                "solver_decoder_training_scale_plan",
                "validate_solver_decoder_training_scale_plan",
            },
        ),
        (
            legacy_training_tensorboard,
            canonical_training_tensorboard,
            {
                "TENSORBOARD_SCALAR_EXPORT_SCHEMA",
                "ScalarWriter",
                "write_solver_decoder_tensorboard_events",
            },
        ),
    )
    for legacy_module, canonical_module, expected in expected_surfaces:
        assert set(legacy_module.__all__) == expected
        for name in legacy_module.__all__:
            assert getattr(legacy_module, name) is getattr(canonical_module, name)


def test_legacy_real_sample_v2_pipeline_preserves_exact_surfaces_and_identity():
    expected_surfaces = (
        (
            legacy_real_smoke,
            canonical_real_smoke,
            {
                "REAL_SAMPLE_V2_SMOKE_SCHEMA",
                "RealSampleV2SmokeReport",
                "real_sample_v2_smoke_from_cloud",
                "evaluate_real_sample_v2_smoke",
                "validate_real_sample_v2_smoke_summary",
            },
        ),
        (
            legacy_real_diagnostics,
            canonical_real_diagnostics,
            {
                "REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA",
                "RealSampleV2DiagnosticsReport",
                "real_sample_v2_diagnostics_from_cloud",
                "evaluate_real_sample_v2_diagnostics",
                "validate_real_sample_v2_diagnostics_summary",
            },
        ),
        (
            legacy_real_handoff,
            canonical_real_handoff,
            {
                "REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA",
                "REAL_SAMPLE_V2_EFFECT_PREVIEW_SCHEMA",
                "RealSampleV2ModelHandoffReport",
                "real_sample_v2_model_handoff_from_cloud",
                "evaluate_real_sample_v2_model_handoff",
                "validate_real_sample_v2_model_handoff_summary",
                "validate_real_sample_v2_effect_preview",
                "render_real_sample_v2_model_handoff_html",
            },
        ),
        (
            legacy_real_viewer,
            canonical_real_viewer,
            {
                "REAL_SAMPLE_V2_VIEWER_PREVIEW_SCHEMA",
                "REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT",
                "REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT",
                "RealSampleV2ViewerPreviewReport",
                "real_sample_v2_viewer_preview_from_cloud",
                "real_sample_v2_viewer_preview_from_handoff",
                "validate_real_sample_v2_viewer_preview_summary",
            },
        ),
        (
            legacy_real_purity,
            canonical_real_purity,
            {
                "REAL_SAMPLE_V2_FULL_CLOUD_PURITY_SCHEMA",
                "RealSampleV2FullCloudPurityReport",
                "real_sample_v2_full_cloud_purity_from_cloud",
                "validate_real_sample_v2_full_cloud_purity_summary",
            },
        ),
        (
            legacy_real_segmentation,
            canonical_real_segmentation,
            {
                "REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA",
                "RealSampleV2SegmentationQualityReport",
                "real_sample_v2_segmentation_quality_from_cloud",
                "real_sample_v2_segmentation_quality_from_purity_report",
                "real_sample_v2_segmentation_quality_from_projected_cloud",
                "validate_real_sample_v2_segmentation_quality_summary",
            },
        ),
        (
            legacy_real_weak_boundary,
            canonical_real_weak_boundary,
            {
                "REAL_SAMPLE_V2_WEAK_BOUNDARY_OPT_SCHEMA",
                "RealSampleV2WeakBoundaryOptReport",
                "real_sample_v2_weak_boundary_opt_from_cloud",
                "validate_real_sample_v2_weak_boundary_opt_summary",
            },
        ),
        (
            legacy_real_weight_policy,
            canonical_real_weight_policy,
            {
                "REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA",
                "RealSampleV2SampleAwareWeightPolicyReport",
                "real_sample_v2_sample_aware_weight_policy_from_cloud",
                "validate_real_sample_v2_sample_aware_weight_policy_summary",
            },
        ),
        (
            legacy_real_bounded,
            canonical_real_bounded,
            {
                "REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA",
                "RealSampleV2BoundedNormalizationCrossSampleInput",
                "RealSampleV2BoundedNormalizationCrossSampleReport",
                "real_sample_v2_bounded_normalization_cross_sample_from_clouds",
                "validate_real_sample_v2_bounded_normalization_cross_sample_summary",
            },
        ),
        (
            legacy_real_promoted,
            canonical_real_promoted,
            {
                "REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA",
                "RealSampleV2PromotedWeightsCrossSampleReport",
                "real_sample_v2_promoted_weights_cross_sample_from_cloud",
                "validate_real_sample_v2_promoted_weights_cross_sample_summary",
            },
        ),
    )
    for legacy_module, canonical_module, expected in expected_surfaces:
        assert set(legacy_module.__all__) == expected
        for name in legacy_module.__all__:
            assert getattr(legacy_module, name) is getattr(canonical_module, name)
