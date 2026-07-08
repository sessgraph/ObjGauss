from __future__ import annotations

import numpy as np

from objgauss.core import (
    GaussianCloud,
    DynamicKProposalReport,
    DynamicKUpdatePlan,
    GsplatRendererAvailability,
    GsplatTrainingInput,
    ObjectState,
    ObjectStateAction,
    ObjectStateCausalGateReport,
    ObjectStateCausalGateThresholds,
    ObjectStateCausalRow,
    ASSIGNMENT_MVP_TRAINING_SCHEMA,
    ASSIGNMENT_SOLVER_V2_COST_TERMS,
    ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA,
    ASSIGNMENT_SOLVER_V2_STATE_SCHEMA,
    ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA,
    ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA,
    ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA,
    ASSIGNMENT_STABILITY_EVAL_SCHEMA,
    ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA,
    CORE_MODEL_TRAIN_VALIDATE_SCHEMA,
    REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA,
    REAL_SAMPLE_V2_EFFECT_PREVIEW_SCHEMA,
    REAL_SAMPLE_V2_FULL_CLOUD_PURITY_SCHEMA,
    REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA,
    REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA,
    REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA,
    REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA,
    REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA,
    REAL_SAMPLE_V2_SMOKE_SCHEMA,
    REAL_SAMPLE_V2_WEAK_BOUNDARY_OPT_SCHEMA,
    AssignmentEvidenceBatch,
    AssignmentV2RendererJointValidationReport,
    AssignmentSolverV2Config,
    AssignmentSolverV2Prediction,
    AssignmentSolverV2State,
    AssignmentSolverV2StabilityEvalReport,
    AssignmentSolverV2TrainingResult,
    CoreModelTrainValidateReport,
    FailureModeClassifier,
    FailureModeEvent,
    ObjectStateGaussianDecode,
    ObjectStateGaussianDecoderTrainingResult,
    ObjectStateGaussianDecoderState,
    ObjectStateIdentityEncoderConfig,
    ObjectStateIdentityEncoderState,
    ObjectStateIdentityEncoderTrainingResult,
    ObjectStateIdentityGateReport,
    ObjectStateIdentityGateThresholds,
    ObjectStateIdentityRow,
    ObjectStatePredictiveGateReport,
    ObjectStatePredictiveGateThresholds,
    ObjectStatePredictiveRow,
    ObjectStateRealityGateReport,
    ObjectStateRealityGateThresholds,
    ObjectStateRealityPublicArtifact,
    ObjectStateRealityRow,
    OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_ACCEPTANCE_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA,
    OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA,
    OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA,
    OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
    OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
    OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA,
    OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA,
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA,
    OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SUMMARY_SCHEMA,
    OBJECTSTATE_BOP_IDENTITY_HANDOFF_SCHEMA,
    OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA,
    OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA,
    OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA,
    OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA,
    OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA,
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA,
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_HANDOFF_SCHEMA,
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_AUTHORING_SCHEMA,
    OBJECTSTATE_BOP_LOCAL_ROW_BATCH_READINESS_SCHEMA,
    OBJECTSTATE_BOP_PREDICTION_BASELINE_HANDOFF_SCHEMA,
    OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA,
    OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA,
    OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA,
    OBJECTSTATE_BOP_PHASE1_SUBSET_SELECTOR_SCHEMA,
    OBJECTSTATE_BOP_GAUSSIAN_EVIDENCE_PREFLIGHT_SCHEMA,
    OBJECTSTATE_BOP_RGBD_GAUSSIAN_EXPORT_SCHEMA,
    OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA,
    OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SCHEMA,
    OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_TEMPLATE_SUMMARY_SCHEMA,
    OBJECTSTATE_BOP_CANDIDATE_ARTIFACT_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA,
    OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_READINESS_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA,
    OBJECTSTATE_CONTROLLED_REALITY_EVIDENCE_PACKAGE_SCHEMA,
    OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
    OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA,
    OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA,
    ObjectStateControlledIdentityThresholds,
    ObjectStateControlledInterventionThresholds,
    ObjectStateControlledPredictionThresholds,
    OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA,
    OBJECTSTATE_ACTION_SCHEMA,
    OBJECTSTATE_CAUSAL_ACTIONS,
    OBJECTSTATE_CAUSAL_GATE_SCHEMA,
    OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA,
    OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA,
    OBJECTSTATE_IDENTITY_DATASET_SCHEMA,
    OBJECTSTATE_IDENTITY_GATE_SCHEMA,
    OBJECTSTATE_PREDICTIVE_GATE_SCHEMA,
    OBJECTSTATE_REALITY_EVIDENCE_KINDS,
    OBJECTSTATE_REALITY_GATE_SCHEMA,
    OBJECTSTATE_REALITY_ROW_SCHEMA,
    OBJECTSTATE_REALITY_ROW_STATUSES,
    OBJECTSTATE_REALITY_SOURCE_KINDS,
    OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA,
    SolverDecoderJointTrainingResult,
    ObjectEmergenceAssignmentPrediction,
    ObjectEmergenceEvidence,
    ObjectEmergenceSolverTrainingResult,
    ObjectEmergenceSolverState,
    ObjectIdentityOracle,
    ObservationModelConfig,
    IdentitySlotObservation,
    ObjectStabilityReport,
    ObjectTemporalMatchReport,
    RendererLossBoundaryReport,
    RealSampleV2DiagnosticsReport,
    RealSampleV2FullCloudPurityReport,
    RealSampleV2ModelHandoffReport,
    RealSampleV2BoundedNormalizationCrossSampleInput,
    RealSampleV2BoundedNormalizationCrossSampleReport,
    RealSampleV2PromotedWeightsCrossSampleReport,
    RealSampleV2SampleAwareWeightPolicyReport,
    RealSampleV2SegmentationQualityReport,
    RealSampleV2SmokeReport,
    RealSampleV2WeakBoundaryOptReport,
    SyntheticObservationFrame,
    SyntheticStabilityScenarioFixture,
    SyntheticStabilityDiagnosticsReport,
    SyntheticStabilityGateReport,
    SyntheticStabilitySuiteGateReport,
    SyntheticWorldState,
    TrainableKernelCamera,
    TrainableKernelImageTarget,
    TrainableKernelResult,
    TrainableKernelSample,
    TENSORBOARD_SCALAR_EXPORT_SCHEMA,
    TRAINING_SCALE_PLAN_SCHEMA,
    TrainingRendererLossResult,
    V2_STABILITY_FOUNDATION_SCHEMA,
    V2_STABILITY_DIAGNOSTICS_SCHEMA,
    V2_STABILITY_FAILURE_MODES,
    V2_STABILITY_GATE_HARD_CHECKS,
    V2_STABILITY_GATE_SCHEMA,
    V2_STABILITY_GATE_SUITE_SCHEMA,
    V2_STABILITY_SCENARIO_FIXTURE_SCHEMA,
    V2_STABILITY_SCENARIO_KINDS,
    V2_SYNTHETIC_OBSERVATION_SCHEMA,
    append_or_replace_property,
    assignment_balance_loss_and_gradient,
    assignment_cluster_loss_and_gradient,
    assignment_evidence_from_trainable_frame,
    assignment_evidence_sequence_from_trainable_frames,
    assignment_entropy_loss_and_gradient,
    assignment_loss_v2_breakdown,
    assignment_mvp_training_summary,
    assignment_solver_v2_checkpoint,
    assignment_solver_v2_state_from_dict,
    assignment_solver_v2_state_from_checkpoint,
    attach_object_aware_lod_metadata,
    attach_quantization_metadata,
    assign_object_ids,
    bind_image_targets_to_frames,
    bind_object_states_to_artifact,
    build_chunk_index,
    build_gsplat_training_input,
    build_gsplat_training_input_from_object_state,
    cluster_features,
    core_model_train_validate_report,
    decode_gaussian_from_object_state,
    diagnose_synthetic_stability_fixture,
    evaluate_objectstate_identity_gate,
    evaluate_objectstate_causal_gate,
    evaluate_objectstate_predictive_gate,
    evaluate_objectstate_reality_gate,
    evaluate_public_artifact_reality_gate,
    evaluate_controlled_real_manifest_reality_gate,
    evaluate_synthetic_stability_gate,
    evaluate_synthetic_stability_suite_gate,
    finalize_objectstate_controlled_prediction_candidate_template,
    finalize_objectstate_controlled_reality_candidate_templates,
    write_objectstate_controlled_prediction_baseline_candidates,
    dynamic_k_proposal_report,
    dynamic_k_update_plan,
    evaluate_assignment_stability,
    evaluate_assignment_solver_v2_stability,
    evaluate_assignment_v2_renderer_joint,
    evaluate_real_sample_v2_diagnostics,
    evaluate_real_sample_v2_model_handoff,
    evaluate_real_sample_v2_smoke,
    initialize_assignment_solver_v2,
    expected_slots_for_synthetic_fixture,
    evaluate_solver_decoder_object_states,
    evaluate_training_renderer_loss,
    evaluate_gsplat_training_renderer_loss,
    gsplat_renderer_availability,
    evidence_from_gaussian_cloud,
    initialize_object_field,
    initialize_object_emergence_solver,
    initialize_object_state_gaussian_decoder,
    image_target_contract_summary,
    initialize_objectstate_identity_encoder_state,
    default_objectstate_reality_public_artifacts,
    make_object_identity_oracle,
    make_synthetic_stability_scenario_fixture,
    make_synthetic_stability_scenario_suite,
    make_synthetic_world_state,
    make_trainable_image_target,
    make_trainable_kernel_mvp_fixture,
    match_object_states,
    object_state_delivery_summary,
    objectstate_reality_blocked_rows_markdown,
    objectstate_reality_public_rows_summary,
    objectstate_reality_rows_from_public_artifacts,
    objectstate_controlled_capture_summary,
    objectstate_controlled_capture_bundle_acceptance_summary,
    objectstate_controlled_capture_bundle_readiness,
    objectstate_controlled_capture_environment,
    objectstate_controlled_capture_file_audit,
    objectstate_controlled_capture_import_summary,
    objectstate_controlled_capture_manifest_from_bundle,
    objectstate_controlled_capture_missing_files_markdown,
    write_objectstate_controlled_capture_bundle_template,
    objectstate_controlled_real_manifest_from_capture_manifest,
    evaluate_objectstate_controlled_identity_predictions,
    evaluate_objectstate_controlled_intervention_candidates,
    evaluate_objectstate_controlled_prediction_candidates,
    objectstate_controlled_identity_handoff,
    objectstate_controlled_reality_bundle_handoff,
    objectstate_controlled_reality_bundle_readiness,
    objectstate_controlled_reality_evidence_package,
    objectstate_controlled_identity_evidence_package,
    objectstate_controlled_prediction_evidence_package,
    objectstate_phase1_evidence_ledger,
    objectstate_bop_prediction_baseline_handoff,
    objectstate_bop_identity_handoff,
    objectstate_bop_local_row_handoff,
    objectstate_bop_baseline_local_row_handoff,
    objectstate_bop_rgbd_baseline_local_row_handoff,
    objectstate_bop_reality_rows_summary,
    objectstate_bop_reality_rows_from_summary,
    read_objectstate_bop_local_row_summary,
    objectstate_bop_cross_sample_ledger,
    objectstate_bop_local_row_batch_handoff,
    objectstate_bop_local_row_batch_spec_authoring,
    objectstate_bop_local_row_batch_readiness,
    read_objectstate_bop_local_row_batch_spec,
    objectstate_bop_capture_condition_sidecar_summary,
    objectstate_bop_phase1_route_audit,
    objectstate_bop_identity_route_audit,
    objectstate_bop_phase1_local_row_readiness,
    objectstate_bop_phase1_subset_selector,
    objectstate_bop_gaussian_evidence_preflight,
    objectstate_bop_rgbd_gaussian_export,
    write_objectstate_bop_gaussian_centroid_baseline_candidate,
    write_objectstate_bop_candidate_artifact_template,
    finalize_objectstate_bop_candidate_artifact_template,
    write_objectstate_controlled_reality_candidate_templates,
    write_objectstate_controlled_reality_candidate_templates_from_manifest,
    objectstate_identity_predictions_from_trainable_artifact,
    objectstate_controlled_real_rows_summary,
    objectstate_reality_rows_from_controlled_real_manifest,
    object_state_stability_report,
    object_id_targets_from_cloud,
    object_emergence_solver_checkpoint,
    objectstate_identity_encoder_features,
    object_emergence_solver_state_from_dict,
    object_scale_multipliers_from_log_offsets,
    object_state_gaussian_decoder_state_from_dict,
    observe_synthetic_world,
    predict_object_emergence_assignment,
    predict_assignment_solver_v2,
    project_object_emergence_prediction,
    project_object_states,
    project_object_states_from_field,
    read_ply,
    read_trainable_kernel_identity_source,
    real_sample_v2_diagnostics_from_cloud,
    real_sample_v2_full_cloud_purity_from_cloud,
    real_sample_v2_model_handoff_from_cloud,
    real_sample_v2_bounded_normalization_cross_sample_from_clouds,
    real_sample_v2_promoted_weights_cross_sample_from_cloud,
    real_sample_v2_sample_aware_weight_policy_from_cloud,
    real_sample_v2_segmentation_quality_from_cloud,
    real_sample_v2_segmentation_quality_from_projected_cloud,
    real_sample_v2_segmentation_quality_from_purity_report,
    real_sample_v2_weak_boundary_opt_from_cloud,
    real_sample_v2_smoke_from_cloud,
    render_real_sample_v2_model_handoff_html,
    renderer_loss_boundary_report,
    solver_decoder_training_scale_plan,
    train_kernel_mvp,
    train_objectstate_identity_encoder,
    train_object_emergence_solver,
    train_assignment_solver_v2,
    train_object_state_gaussian_decoder,
    train_solver_decoder_joint,
    train_kernel_mvp_from_cloud,
    trainable_kernel_model_artifact,
    trainable_kernel_sample_from_cloud,
    validate_image_target_contract_summary,
    validate_assignment_loss_v2_summary,
    validate_assignment_evidence_summary,
    validate_assignment_stability_eval,
    validate_assignment_solver_v2_config,
    validate_assignment_solver_v2_checkpoint,
    validate_assignment_solver_v2_state,
    validate_assignment_solver_v2_stability_eval_summary,
    validate_assignment_solver_v2_training_summary,
    validate_assignment_v2_renderer_joint_summary,
    validate_core_model_train_validate_summary,
    validate_real_sample_v2_diagnostics_summary,
    validate_real_sample_v2_effect_preview,
    validate_real_sample_v2_full_cloud_purity_summary,
    validate_real_sample_v2_model_handoff_summary,
    validate_real_sample_v2_bounded_normalization_cross_sample_summary,
    validate_real_sample_v2_promoted_weights_cross_sample_summary,
    validate_real_sample_v2_sample_aware_weight_policy_summary,
    validate_real_sample_v2_segmentation_quality_summary,
    validate_real_sample_v2_smoke_summary,
    validate_real_sample_v2_weak_boundary_opt_summary,
    validate_object_emergence_evidence,
    validate_object_emergence_solver_checkpoint,
    validate_object_identity_oracle,
    validate_object_state_gaussian_decoder_state,
    validate_objectstate_checkpoint_eval,
    validate_objectstate_causal_gate_summary,
    validate_objectstate_reality_gate_summary,
    validate_objectstate_reality_public_rows_summary,
    validate_objectstate_controlled_capture_manifest,
    validate_objectstate_controlled_capture_bundle_acceptance_summary,
    validate_objectstate_controlled_capture_bundle_readiness_summary,
    validate_objectstate_controlled_capture_environment_summary,
    validate_objectstate_controlled_capture_bundle_template_summary,
    validate_objectstate_controlled_capture_file_audit_summary,
    validate_objectstate_controlled_capture_summary,
    validate_objectstate_controlled_capture_import_summary,
    validate_objectstate_controlled_identity_eval_summary,
    validate_objectstate_controlled_identity_handoff_summary,
    validate_objectstate_controlled_identity_predictions,
    validate_objectstate_controlled_identity_thresholds,
    validate_objectstate_controlled_intervention_candidates,
    validate_objectstate_controlled_intervention_candidates_template,
    validate_objectstate_controlled_intervention_eval_summary,
    validate_objectstate_controlled_prediction_candidates,
    validate_objectstate_bop_capture_condition_sidecar,
    validate_objectstate_bop_capture_condition_sidecar_summary,
    validate_objectstate_bop_prediction_baseline_handoff_summary,
    validate_objectstate_bop_identity_handoff_summary,
    validate_objectstate_bop_local_row_handoff_summary,
    validate_objectstate_bop_baseline_local_row_handoff_summary,
    validate_objectstate_bop_rgbd_baseline_local_row_handoff_summary,
    validate_objectstate_bop_reality_rows_summary,
    validate_objectstate_bop_cross_sample_ledger_summary,
    validate_objectstate_bop_local_row_batch_spec,
    validate_objectstate_bop_local_row_batch_handoff_summary,
    validate_objectstate_bop_phase1_route_audit_summary,
    validate_objectstate_bop_identity_route_audit_summary,
    validate_objectstate_bop_phase1_local_row_readiness_summary,
    validate_objectstate_bop_phase1_subset_selector_summary,
    validate_objectstate_bop_gaussian_evidence_preflight_summary,
    validate_objectstate_bop_rgbd_gaussian_export_summary,
    validate_objectstate_bop_baseline_candidate_summary,
    validate_objectstate_bop_candidate_artifact_template,
    validate_objectstate_bop_candidate_artifact_template_summary,
    validate_objectstate_bop_candidate_artifact_finalize_summary,
    validate_objectstate_controlled_identity_evidence_package_summary,
    validate_objectstate_phase1_evidence_ledger_summary,
    validate_objectstate_controlled_prediction_baseline_summary,
    validate_objectstate_controlled_prediction_candidate_finalize_summary,
    validate_objectstate_controlled_prediction_candidates_template,
    validate_objectstate_controlled_prediction_evidence_package_summary,
    validate_objectstate_controlled_prediction_eval_summary,
    validate_objectstate_controlled_reality_bundle_handoff_summary,
    validate_objectstate_controlled_reality_bundle_readiness_summary,
    validate_objectstate_controlled_reality_candidate_finalize_summary,
    validate_objectstate_controlled_reality_candidate_template_summary,
    validate_objectstate_controlled_reality_evidence_package_summary,
    validate_objectstate_controlled_real_manifest,
    validate_objectstate_controlled_real_rows_summary,
    validate_observation_model_config,
    validate_solver_decoder_joint_checkpoint,
    validate_solver_decoder_training_scale_plan,
    validate_synthetic_observation_frame,
    validate_objectstate_identity_encoder_training_summary,
    validate_objectstate_identity_gate_summary,
    validate_objectstate_predictive_gate_summary,
    validate_synthetic_stability_diagnostics_summary,
    validate_synthetic_stability_gate_summary,
    validate_synthetic_stability_suite_gate_summary,
    validate_synthetic_stability_scenario_fixture,
    validate_synthetic_world_state,
    validate_renderer_loss_boundary_summary,
    validate_trainable_kernel_model_artifact,
    validate_trainable_image_target,
    validate_training_renderer_summary,
    write_ogc_payload,
    write_ply,
    write_quantized_ogc_payload,
    write_solver_decoder_tensorboard_events,
    write_trainable_kernel_model_artifact,
    solver_decoder_joint_checkpoint,
    solver_decoder_joint_states_from_dict,
    supervised_assignment_loss_and_gradient,
)
from objgauss.core.features import extract_features
from objgauss.core.object_field import field_from_labels
from objgauss.core.objects import apply_object_colors, filter_objects
from objgauss.baseline_comparison import compare_baseline_candidates as historical_compare_baselines
from objgauss.clip_scoring import score_mask_manifest_with_clip as historical_score_clip
from objgauss.emergence import object_emergence_metrics as historical_emergence_metrics
from objgauss.gaussians import GaussianCloud as HistoricalGaussianCloud
from objgauss.chunk_index import build_chunk_index as historical_build_chunk_index
from objgauss.lod import attach_object_aware_lod_metadata as historical_attach_lod
from objgauss.ogc_payload import write_ogc_payload as historical_write_ogc_payload
from objgauss.quantization import attach_quantization_metadata as historical_attach_quantization
from objgauss.quantization import write_quantized_ogc_payload as historical_write_quantized_ogc_payload
from objgauss.mask_voting import project_points as historical_project_points
from objgauss.masks import validate_mask_manifest as historical_validate_masks
from objgauss.object_field import ObjectField as HistoricalObjectField
from objgauss.ply import read_ply as historical_read_ply
from objgauss.segment import assign_object_ids as historical_assign_object_ids
from objgauss.semantic_slots import align_mask_manifest_slots as historical_align_slots


def test_core_namespace_reuses_existing_gaussian_model():
    assert GaussianCloud is HistoricalGaussianCloud
    assert GaussianCloud.__module__ == "objgauss.core.gaussian"


def test_historical_paths_are_core_wrappers():
    from objgauss.core.io_ply import read_ply as core_read_ply
    from objgauss.core.masks import validate_mask_manifest as core_validate_masks
    from objgauss.core.object_field import ObjectField as CoreObjectField
    from objgauss.core.objects import assign_object_ids as core_assign_object_ids
    from objgauss.core.chunk_index import build_chunk_index as core_build_chunk_index
    from objgauss.core.lod import attach_object_aware_lod_metadata as core_attach_lod
    from objgauss.core.ogc_payload import write_ogc_payload as core_write_ogc_payload
    from objgauss.core.quantization import attach_quantization_metadata as core_attach_quantization
    from objgauss.core.quantization import write_quantized_ogc_payload as core_write_quantized_ogc_payload
    from objgauss.core.projection import project_points as core_project_points
    from objgauss.core.semantic_slots import align_mask_manifest_slots as core_align_slots
    from objgauss.core.clip_scoring import score_mask_manifest_with_clip as core_score_clip
    from objgauss.core.baseline_comparison import compare_baseline_candidates as core_compare_baselines
    from objgauss.core.emergence import object_emergence_metrics as core_emergence_metrics

    assert historical_read_ply is core_read_ply
    assert historical_assign_object_ids is core_assign_object_ids
    assert historical_build_chunk_index is core_build_chunk_index
    assert historical_attach_lod is core_attach_lod
    assert historical_attach_quantization is core_attach_quantization
    assert historical_write_quantized_ogc_payload is core_write_quantized_ogc_payload
    assert historical_write_ogc_payload is core_write_ogc_payload
    assert HistoricalObjectField is CoreObjectField
    assert historical_project_points is core_project_points
    assert historical_validate_masks is core_validate_masks
    assert historical_align_slots is core_align_slots
    assert historical_score_clip is core_score_clip
    assert historical_compare_baselines is core_compare_baselines
    assert historical_emergence_metrics is core_emergence_metrics


def test_core_namespace_supports_minimal_object_workflow(tmp_path):
    cloud = _tiny_cloud()
    features = extract_features(cloud)
    clustering = cluster_features(features, clusters=2, seed=7, max_iter=20)

    labeled = assign_object_ids(cloud, clustering.labels)
    colored = apply_object_colors(labeled)
    assert "object_id" in colored.fields

    output = tmp_path / "core_objects.ply"
    write_ply(output, colored, fmt="ascii")
    loaded = read_ply(output)

    assert loaded.count == cloud.count
    assert set(np.unique(loaded.vertices["object_id"])) == {0, 1}
    assert 0 < filter_objects(loaded, {0}, mode="remove").count < loaded.count


def test_core_namespace_exposes_object_field_kernel():
    field = field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2)

    assert field.gaussian_count == 4
    assert field.slots == 2
    assert field.labels().tolist() == [0, 0, 1, 1]
    projection = project_object_states_from_field(_tiny_cloud(), field)
    assert isinstance(projection.states[0], ObjectState)
    assert projection.derived_object_ids.tolist() == [0, 0, 1, 1]
    decoded = decode_gaussian_from_object_state(
        np.column_stack(
            [
                _tiny_cloud().vertices["x"],
                _tiny_cloud().vertices["y"],
                _tiny_cloud().vertices["z"],
            ]
        ),
        projection,
        np.asarray([[0.9, 0.1, 0.1], [0.1, 0.8, 0.7]], dtype=np.float32),
    )
    assert isinstance(decoded, ObjectStateGaussianDecode)
    assert decoded.as_dict()["schema"] == "objgauss-object-state-gaussian-decode-v1"
    report = object_state_stability_report(projection)
    assert isinstance(report, ObjectStabilityReport)
    assert report.evidence_count == 4
    assert report.slots == 2
    temporal = match_object_states(projection, projection)
    assert isinstance(temporal, ObjectTemporalMatchReport)
    summary = object_state_delivery_summary(projection)
    assert summary["schema"] == "objgauss-object-state-delivery-binding-v1"
    bound = bind_object_states_to_artifact({"role": "object_edit"}, projection)
    assert bound["object_state_summary"] == summary
    proposals = dynamic_k_proposal_report(projection)
    assert isinstance(proposals, DynamicKProposalReport)
    update_plan = dynamic_k_update_plan(projection)
    assert isinstance(update_plan, DynamicKUpdatePlan)
    assert update_plan.apply_at == "epoch_boundary"

    initialized = initialize_object_field(_tiny_cloud(), slots=2, seed=3, max_iter=10)
    assert initialized.field.gaussian_count == 4
    assert initialized.field.slots == 2
    np.testing.assert_allclose(
        object_scale_multipliers_from_log_offsets(np.zeros(2, dtype=np.float32)),
        np.ones(2, dtype=np.float32),
        atol=1e-6,
    )


def test_core_namespace_exposes_object_emergence_solver_abi():
    evidence = evidence_from_gaussian_cloud(_tiny_cloud(), source="namespace-test")
    assert isinstance(evidence, ObjectEmergenceEvidence)
    assert validate_object_emergence_evidence(evidence)[0].shape == (4, 3)
    state = initialize_object_emergence_solver(
        slots=2,
        feature_dim=evidence.feature_dim,
        seed=4,
        scale=0.0,
    )
    assert isinstance(state, ObjectEmergenceSolverState)
    state = ObjectEmergenceSolverState(
        config=state.config,
        feature_weights=np.zeros_like(state.feature_weights),
        position_weights=np.array([[-3.0, 3.0], [0.0, 0.0], [0.0, 0.0]], dtype=np.float32),
        bias=np.zeros(2, dtype=np.float32),
        source="namespace-x-axis-split",
    )
    prediction = predict_object_emergence_assignment(evidence, state)
    assert isinstance(prediction, ObjectEmergenceAssignmentPrediction)
    assert prediction.assignment.shape == (4, 2)
    np.testing.assert_allclose(prediction.assignment.sum(axis=1), np.ones(4), atol=1e-6)
    projection = project_object_emergence_prediction(
        _tiny_cloud(),
        prediction,
        evidence_features=evidence.features,
    )
    assert isinstance(projection.states[0], ObjectState)
    assert projection.states[0].status == "active"
    targets, mapping = object_id_targets_from_cloud(_tiny_object_cloud())
    train_evidence = evidence_from_gaussian_cloud(_tiny_object_cloud(), target_assignment=targets)
    training = train_object_emergence_solver(
        [train_evidence],
        iterations=2,
        learning_rate=0.25,
        assignment_weight=1.0,
        entropy_weight=0.0,
        balance_weight=0.0,
        temporal_weight=0.0,
        seed=2,
    )
    assert isinstance(training, ObjectEmergenceSolverTrainingResult)
    checkpoint = object_emergence_solver_checkpoint(
        training,
        input_path="fixture://namespace",
        source_gaussians=_tiny_object_cloud().count,
        sampled_gaussians=_tiny_object_cloud().count,
        target_source="object_id_one_hot_targets",
        object_id_mapping=mapping,
    )
    assert validate_object_emergence_solver_checkpoint(checkpoint) == checkpoint
    restored = object_emergence_solver_state_from_dict(checkpoint)
    assert isinstance(restored, ObjectEmergenceSolverState)
    assert restored.step == training.final_state.step
    assert mapping == {0: 0, 1: 1}


def test_core_namespace_exposes_v2_stability_foundation_contract():
    assert V2_STABILITY_FOUNDATION_SCHEMA == "objgauss-v2-stability-foundation-v1"
    assert V2_SYNTHETIC_OBSERVATION_SCHEMA == "objgauss-v2-synthetic-observation-v1"
    assert V2_STABILITY_SCENARIO_FIXTURE_SCHEMA == "objgauss-v2-stability-scenario-fixture-v1"
    assert V2_STABILITY_SCENARIO_KINDS == (
        "cross_view",
        "occlusion_recovery",
        "perturbation",
        "adversarial_swap",
    )

    oracle = make_object_identity_oracle(
        scenario_id="namespace-v2-stability",
        object_count=2,
        frame_count=2,
    )
    assert isinstance(oracle, ObjectIdentityOracle)
    assert validate_object_identity_oracle(oracle) is oracle

    world = make_synthetic_world_state(
        scenario_id="namespace-v2-stability",
        scenario_kind="cross_view",
        object_count=2,
        frame_count=2,
        feature_dim=3,
        seed=3,
    )
    assert isinstance(world, SyntheticWorldState)
    assert validate_synthetic_world_state(world) is world

    config = ObservationModelConfig(points_per_object=1, position_jitter=0.0)
    assert validate_observation_model_config(config) is config
    observations = observe_synthetic_world(world, config=config)
    assert isinstance(observations[0], SyntheticObservationFrame)
    assert validate_synthetic_observation_frame(observations[0]) is observations[0]
    assert observations[0].oracle_object_ids.tolist() == [0, 1]
    assert observations[0].expected_slots.tolist() == [0, 1]

    fixture = make_synthetic_stability_scenario_fixture(
        scenario_kind="adversarial_swap",
        object_count=2,
        frame_count=2,
        feature_dim=3,
        seed=5,
        observation_config=ObservationModelConfig(points_per_object=1, position_jitter=0.0, seed=6),
    )
    assert isinstance(fixture, SyntheticStabilityScenarioFixture)
    assert validate_synthetic_stability_scenario_fixture(fixture).schema == (
        V2_STABILITY_SCENARIO_FIXTURE_SCHEMA
    )
    assert fixture.observations[1].oracle_object_ids.tolist() == [0, 1]
    assert fixture.observations[1].expected_slots.tolist() == [0, 1]
    suite = make_synthetic_stability_scenario_suite(object_count=2, seed=7)
    assert tuple(item.scenario_kind for item in suite) == V2_STABILITY_SCENARIO_KINDS

    assert V2_STABILITY_DIAGNOSTICS_SCHEMA == "objgauss-v2-stability-diagnostics-v1"
    assert "slot_swap" in V2_STABILITY_FAILURE_MODES
    predicted = expected_slots_for_synthetic_fixture(fixture)
    diagnostics = diagnose_synthetic_stability_fixture(
        fixture,
        predicted_slots=(predicted[0], np.asarray([1, 0], dtype=np.int64)),
        classifier=FailureModeClassifier(),
    )
    assert isinstance(diagnostics, SyntheticStabilityDiagnosticsReport)
    summary = diagnostics.as_dict()
    assert validate_synthetic_stability_diagnostics_summary(summary) is summary
    assert summary["failure_mode_counts"]["slot_swap"] == 1
    assert summary["diagnostic_role"] == "diagnostic_only_not_gate"
    first_observation = diagnostics.identity_observations[0]
    assert isinstance(first_observation, IdentitySlotObservation)
    assert isinstance(diagnostics.failure_modes[0], FailureModeEvent)

    assert V2_STABILITY_GATE_SCHEMA == "objgauss-v2-stability-gate-v1"
    assert V2_STABILITY_GATE_SUITE_SCHEMA == "objgauss-v2-stability-gate-suite-v1"
    assert "expected_slot_consistency_pass" in V2_STABILITY_GATE_HARD_CHECKS
    gate = evaluate_synthetic_stability_gate(fixture, predicted_slots=predicted)
    assert isinstance(gate, SyntheticStabilityGateReport)
    gate_summary = gate.as_dict()
    assert validate_synthetic_stability_gate_summary(gate_summary) is gate_summary
    assert gate_summary["status"] == "synthetic_stability_gate_pass"
    suite_predictions = tuple(expected_slots_for_synthetic_fixture(item) for item in suite)
    suite_gate = evaluate_synthetic_stability_suite_gate(
        suite,
        predicted_slots_by_fixture=suite_predictions,
    )
    assert isinstance(suite_gate, SyntheticStabilitySuiteGateReport)
    suite_summary = suite_gate.as_dict()
    assert validate_synthetic_stability_suite_gate_summary(suite_summary) is suite_summary
    assert suite_summary["status"] == "synthetic_stability_suite_gate_pass"

    assert OBJECTSTATE_IDENTITY_GATE_SCHEMA == "objgauss-objectstate-identity-gate-v1"
    assert OBJECTSTATE_IDENTITY_DATASET_SCHEMA == "objgauss-objectstate-identity-dataset-v1"
    identity_gate = evaluate_objectstate_identity_gate(
        suite,
        predicted_slots_by_fixture=suite_predictions,
    )
    assert isinstance(identity_gate, ObjectStateIdentityGateReport)
    assert isinstance(identity_gate.thresholds, ObjectStateIdentityGateThresholds)
    assert isinstance(identity_gate.rows[0], ObjectStateIdentityRow)
    identity_summary = identity_gate.as_dict()
    assert validate_objectstate_identity_gate_summary(identity_summary) is identity_summary
    assert identity_summary["status"] == "objectstate_identity_gate_pass"
    assert identity_summary["metrics"]["idf1"] == 1.0

    assert OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA == (
        "objgauss-objectstate-identity-encoder-training-v1"
    )
    assert OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA == (
        "objgauss-objectstate-identity-encoder-state-v1"
    )
    encoder_features = objectstate_identity_encoder_features(identity_gate.rows)
    encoder_config = ObjectStateIdentityEncoderConfig(
        input_dim=encoder_features.shape[1],
        embedding_dim=2,
        learning_rate=0.4,
        weight_decay=0.0,
        seed=8,
    )
    encoder_state = initialize_objectstate_identity_encoder_state(encoder_config)
    assert isinstance(encoder_state, ObjectStateIdentityEncoderState)
    encoder_result = train_objectstate_identity_encoder(
        identity_gate.rows,
        config=encoder_config,
        iterations=20,
    )
    assert isinstance(encoder_result, ObjectStateIdentityEncoderTrainingResult)
    encoder_summary = encoder_result.as_dict()
    assert validate_objectstate_identity_encoder_training_summary(encoder_summary) is encoder_summary
    assert encoder_summary["schema"] == OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA

    assert OBJECTSTATE_PREDICTIVE_GATE_SCHEMA == "objgauss-objectstate-predictive-gate-v1"
    predictive_report = evaluate_objectstate_predictive_gate(suite)
    assert isinstance(predictive_report, ObjectStatePredictiveGateReport)
    assert isinstance(predictive_report.thresholds, ObjectStatePredictiveGateThresholds)
    assert isinstance(predictive_report.rows[0], ObjectStatePredictiveRow)
    predictive_summary = predictive_report.as_dict()
    assert validate_objectstate_predictive_gate_summary(predictive_summary) is predictive_summary
    assert predictive_summary["schema"] == OBJECTSTATE_PREDICTIVE_GATE_SCHEMA

    assert OBJECTSTATE_CAUSAL_GATE_SCHEMA == "objgauss-objectstate-causal-gate-v1"
    assert OBJECTSTATE_ACTION_SCHEMA == "objgauss-objectstate-action-v1"
    assert "push_left" in OBJECTSTATE_CAUSAL_ACTIONS
    causal_report = evaluate_objectstate_causal_gate(suite)
    assert isinstance(causal_report, ObjectStateCausalGateReport)
    assert isinstance(causal_report.thresholds, ObjectStateCausalGateThresholds)
    assert isinstance(causal_report.rows[0], ObjectStateCausalRow)
    assert isinstance(causal_report.rows[0].action, ObjectStateAction)
    causal_summary = causal_report.as_dict()
    assert validate_objectstate_causal_gate_summary(causal_summary) is causal_summary
    assert causal_summary["schema"] == OBJECTSTATE_CAUSAL_GATE_SCHEMA

    assert OBJECTSTATE_REALITY_GATE_SCHEMA == "objgauss-objectstate-reality-gate-v1"
    assert OBJECTSTATE_REALITY_ROW_SCHEMA == "objgauss-objectstate-real-public-row-v1"
    assert "identity" in OBJECTSTATE_REALITY_EVIDENCE_KINDS
    assert "controlled_real" in OBJECTSTATE_REALITY_SOURCE_KINDS
    assert "blocked" in OBJECTSTATE_REALITY_ROW_STATUSES
    reality_rows = (
        ObjectStateRealityRow(
            row_id="namespace-identity",
            sample_id="namespace-controlled-cup",
            source_kind="controlled_real",
            evidence_kind="identity",
            status="pass",
            object_category="cup",
            scenario="cross_view_reappearance",
            observation_modalities=("rgb", "gaussian"),
            artifact_refs=("datasets/namespace-controlled-cup/identity.json",),
            metrics={
                "idf1": 1.0,
                "fragmentation_rate": 0.0,
                "swap_rate": 0.0,
                "identity_collapse": False,
            },
            has_identity_gt=True,
            has_pose_gt=True,
            has_action_gt=False,
            has_timestamp=True,
        ),
        ObjectStateRealityRow(
            row_id="namespace-prediction",
            sample_id="namespace-controlled-cup",
            source_kind="controlled_real",
            evidence_kind="prediction",
            status="pass",
            object_category="cup",
            scenario="short_horizon_pose",
            observation_modalities=("rgb", "gaussian"),
            artifact_refs=("datasets/namespace-controlled-cup/pose.json",),
            metrics={
                "state_ade": 0.04,
                "history_ade": 0.03,
                "prediction_gap_vs_history_model": 0.01,
            },
            has_identity_gt=True,
            has_pose_gt=True,
            has_action_gt=False,
            has_timestamp=True,
        ),
        ObjectStateRealityRow(
            row_id="namespace-intervention",
            sample_id="namespace-controlled-cup",
            source_kind="controlled_real",
            evidence_kind="intervention",
            status="pass",
            object_category="cup",
            scenario="push_left",
            observation_modalities=("rgb", "gaussian"),
            artifact_refs=("datasets/namespace-controlled-cup/action.json",),
            metrics={
                "action_conditioned_ade": 0.02,
                "counterfactual_outcome_accuracy": 1.0,
                "wrong_direction_rate": 0.0,
            },
            has_identity_gt=True,
            has_pose_gt=True,
            has_action_gt=True,
            has_timestamp=True,
        ),
    )
    reality_report = evaluate_objectstate_reality_gate(
        reality_rows,
        synthetic_smoke_passed=True,
    )
    assert isinstance(reality_report, ObjectStateRealityGateReport)
    assert isinstance(reality_report.thresholds, ObjectStateRealityGateThresholds)
    assert isinstance(reality_report.rows[0], ObjectStateRealityRow)
    reality_summary = reality_report.as_dict()
    assert validate_objectstate_reality_gate_summary(reality_summary) is reality_summary
    assert reality_summary["schema"] == OBJECTSTATE_REALITY_GATE_SCHEMA
    assert reality_summary["status"] == "objectstate_reality_gate_pass"
    assert objectstate_reality_blocked_rows_markdown(reality_report) == (
        "No blocked ObjectState reality rows.\n"
    )

    assert OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA == (
        "objgauss-objectstate-public-artifact-rows-v1"
    )
    public_artifacts = default_objectstate_reality_public_artifacts()
    assert isinstance(public_artifacts[0], ObjectStateRealityPublicArtifact)
    public_rows = objectstate_reality_rows_from_public_artifacts(public_artifacts[:1])
    assert len(public_rows) == 3
    assert public_rows[0].status == "blocked"
    public_report = evaluate_public_artifact_reality_gate(public_artifacts[:1])
    assert isinstance(public_report, ObjectStateRealityGateReport)
    public_summary = objectstate_reality_public_rows_summary(public_artifacts[:1])
    assert validate_objectstate_reality_public_rows_summary(public_summary) is public_summary
    assert public_summary["schema"] == OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA
    assert public_summary["gate"]["status"] == "objectstate_reality_gate_fail"
    assert public_summary["claim_policy"]["object_id_is_not_identity_ground_truth"] is True

    assert OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA == (
        "objgauss-objectstate-controlled-capture-manifest-v1"
    )
    assert OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA == (
        "objgauss-objectstate-controlled-capture-summary-v1"
    )
    capture_manifest = {
        "schema": OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "namespace-controlled-capture",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "identity_reappearance",
            "fps": 30.0,
            "capture_device": "fixture-camera",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": ["outputs/controlled-real/namespace/capture.json"],
            "license": "local controlled capture",
        },
        "objects": [{"object_id": "cup-001", "category": "cup"}],
        "actions": [
            {
                "action_id": "push-right-001",
                "action_type": "push_right",
                "object_id": "cup-001",
                "start_timestamp": 0.0,
                "end_timestamp": 0.033333,
                "actor": "namespace-fixture",
                "vector": [0.01, 0.0, 0.0],
            }
        ],
        "frames": [
            {
                "frame_id": "frame-000000",
                "timestamp": 0.0,
                "observation": {
                    "rgb": "rgb/000000.png",
                    "gaussian": "gaussians/000000.ply",
                },
                "action_id": "push-right-001",
                "objects": [
                    {
                        "object_id": "cup-001",
                        "pose": {
                            "position": [0.1, 0.2, 0.3],
                            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                ],
            },
            {
                "frame_id": "frame-000001",
                "timestamp": 0.033333,
                "observation": {
                    "rgb": "rgb/000001.png",
                    "gaussian": "gaussians/000001.ply",
                },
                "objects": [
                    {
                        "object_id": "cup-001",
                        "pose": {
                            "position": [0.11, 0.2, 0.3],
                            "rotation_xyzw": [0.0, 0.0, 0.0, 1.0],
                        },
                    }
                ],
            },
        ],
    }
    assert validate_objectstate_controlled_capture_manifest(capture_manifest)[
        "schema"
    ] == OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA
    capture_summary = objectstate_controlled_capture_summary(capture_manifest)
    assert validate_objectstate_controlled_capture_summary(capture_summary) is capture_summary
    assert capture_summary["schema"] == OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA
    assert OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA == (
        "objgauss-objectstate-controlled-capture-file-audit-v1"
    )
    assert OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA == (
        "objgauss-objectstate-controlled-capture-import-v1"
    )
    assert OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_ACCEPTANCE_SCHEMA == (
        "objgauss-objectstate-controlled-capture-bundle-acceptance-v1"
    )
    assert OBJECTSTATE_CONTROLLED_CAPTURE_ENVIRONMENT_SCHEMA == (
        "objgauss-objectstate-controlled-capture-environment-v1"
    )
    assert objectstate_controlled_capture_file_audit is not None
    assert objectstate_controlled_capture_environment is not None
    assert objectstate_controlled_capture_bundle_acceptance_summary is not None
    assert objectstate_controlled_capture_import_summary is not None
    assert objectstate_controlled_capture_manifest_from_bundle is not None
    assert objectstate_controlled_capture_missing_files_markdown([]).endswith(
        "no missing files |"
    )
    assert validate_objectstate_controlled_capture_file_audit_summary is not None
    assert validate_objectstate_controlled_capture_environment_summary is not None
    assert validate_objectstate_controlled_capture_bundle_acceptance_summary is not None
    assert validate_objectstate_controlled_capture_import_summary is not None
    assert capture_summary["readiness"]["identity_stage_ready"] is True
    capture_seed = objectstate_controlled_real_manifest_from_capture_manifest(
        capture_manifest
    )
    assert capture_seed["schema"] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    assert capture_seed["evidence_rows"][0]["status"] == "blocked"
    assert OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA == (
        "objgauss-objectstate-controlled-identity-predictions-v1"
    )
    assert OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA == (
        "objgauss-objectstate-controlled-identity-eval-v1"
    )
    assert OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA == (
        "objgauss-objectstate-controlled-identity-handoff-v1"
    )
    assert OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA == (
        "objgauss-objectstate-controlled-candidate-artifact-file-audit-v1"
    )
    assert OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA == (
        "objgauss-objectstate-controlled-identity-scenario-audit-v1"
    )
    identity_predictions = {
        "schema": OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA,
        "sample_id": "namespace-controlled-capture",
        "candidate": {
            "candidate_id": "namespace-identity-candidate",
            "source": "namespace fixture",
            "artifact_refs": ["outputs/controlled-real/namespace/objectstates.json"],
            "identity_evidence": {
                "reconstruction_noise_robustness": 1.0,
                "reconstruction_noise_variant_count": 2,
                "source": "namespace repeated Gaussian reconstruction noise variants",
            },
        },
        "predictions": [
            {
                "frame_id": "frame-000000",
                "object_id": "cup-001",
                "predicted_identity": "cup-track",
            },
            {
                "frame_id": "frame-000001",
                "object_id": "cup-001",
                "predicted_identity": "cup-track",
            },
        ],
    }
    assert validate_objectstate_controlled_identity_predictions(identity_predictions)[
        "schema"
    ] == OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA
    identity_thresholds = ObjectStateControlledIdentityThresholds()
    assert validate_objectstate_controlled_identity_thresholds(identity_thresholds)[
        "min_idf1"
    ] == 0.95
    identity_summary = evaluate_objectstate_controlled_identity_predictions(
        capture_manifest,
        identity_predictions,
        thresholds=identity_thresholds,
    )
    assert validate_objectstate_controlled_identity_eval_summary(
        identity_summary
    ) is identity_summary
    assert identity_summary["schema"] == OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA
    assert identity_summary["status"] == "objectstate_controlled_identity_eval_pass"
    assert identity_summary["controlled_real_manifest"]["evidence_rows"][0][
        "status"
    ] == "pass"
    assert OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA == (
        "objgauss-objectstate-controlled-prediction-candidates-v1"
    )
    assert OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_TEMPLATE_SCHEMA == (
        "objgauss-objectstate-controlled-prediction-candidates-template-v1"
    )
    assert OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA == (
        "objgauss-objectstate-controlled-prediction-eval-v1"
    )
    prediction_candidates = {
        "schema": OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA,
        "sample_id": "namespace-controlled-capture",
        "candidate": {
            "candidate_id": "namespace-prediction-candidate",
            "source": "namespace fixture",
            "artifact_refs": ["outputs/controlled-real/namespace/predictions.json"],
        },
        "predictions": [
            {
                "source_frame_id": "frame-000000",
                "target_frame_id": "frame-000001",
                "object_id": "cup-001",
                "predicted_position": [0.11, 0.2, 0.3],
                "history_baseline_position": [0.1, 0.2, 0.3],
            }
        ],
    }
    assert validate_objectstate_controlled_prediction_candidates(
        prediction_candidates
    )["schema"] == OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA
    prediction_thresholds = ObjectStateControlledPredictionThresholds()
    prediction_summary = evaluate_objectstate_controlled_prediction_candidates(
        capture_manifest,
        prediction_candidates,
        thresholds=prediction_thresholds,
    )
    assert validate_objectstate_controlled_prediction_eval_summary(
        prediction_summary
    ) is prediction_summary
    assert prediction_summary["schema"] == OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA
    assert prediction_summary["status"] == "objectstate_controlled_prediction_eval_pass"
    assert prediction_summary["controlled_real_manifest"]["evidence_rows"][1][
        "status"
    ] == "pass"
    assert OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA == (
        "objgauss-objectstate-controlled-intervention-candidates-v1"
    )
    assert OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_TEMPLATE_SCHEMA == (
        "objgauss-objectstate-controlled-intervention-candidates-template-v1"
    )
    assert OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA == (
        "objgauss-objectstate-controlled-intervention-eval-v1"
    )
    intervention_candidates = {
        "schema": OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA,
        "sample_id": "namespace-controlled-capture",
        "candidate": {
            "candidate_id": "namespace-intervention-candidate",
            "source": "namespace fixture",
            "artifact_refs": ["outputs/controlled-real/namespace/interventions.json"],
        },
        "interventions": [
            {
                "source_frame_id": "frame-000000",
                "target_frame_id": "frame-000001",
                "object_id": "cup-001",
                "action_id": "push-right-001",
                "action_conditioned_position": [0.11, 0.2, 0.3],
                "no_action_baseline_position": [0.1, 0.2, 0.3],
            }
        ],
    }
    assert validate_objectstate_controlled_intervention_candidates(
        intervention_candidates
    )["schema"] == OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA
    intervention_thresholds = ObjectStateControlledInterventionThresholds()
    intervention_summary = evaluate_objectstate_controlled_intervention_candidates(
        capture_manifest,
        intervention_candidates,
        thresholds=intervention_thresholds,
    )
    assert validate_objectstate_controlled_intervention_eval_summary(
        intervention_summary
    ) is intervention_summary
    assert (
        intervention_summary["schema"]
        == OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA
    )
    assert (
        intervention_summary["status"]
        == "objectstate_controlled_intervention_eval_pass"
    )
    assert intervention_summary["controlled_real_manifest"]["evidence_rows"][2][
        "status"
    ] == "pass"
    assert OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_HANDOFF_SCHEMA == (
        "objgauss-objectstate-controlled-reality-bundle-handoff-v1"
    )
    assert OBJECTSTATE_CONTROLLED_REALITY_BUNDLE_READINESS_SCHEMA == (
        "objgauss-objectstate-controlled-reality-bundle-readiness-v1"
    )
    assert OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_TEMPLATE_SCHEMA == (
        "objgauss-objectstate-controlled-reality-candidate-template-v1"
    )
    assert OBJECTSTATE_CONTROLLED_REALITY_CANDIDATE_FINALIZE_SCHEMA == (
        "objgauss-objectstate-controlled-reality-candidate-finalize-v1"
    )
    assert OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATE_FINALIZE_SCHEMA == (
        "objgauss-objectstate-controlled-prediction-candidate-finalize-v1"
    )
    assert OBJECTSTATE_CONTROLLED_PREDICTION_BASELINE_SCHEMA == (
        "objgauss-objectstate-controlled-prediction-baseline-candidates-v1"
    )
    assert OBJECTSTATE_BOP_PREDICTION_BASELINE_HANDOFF_SCHEMA == (
        "objgauss-objectstate-bop-prediction-baseline-handoff-v1"
    )
    assert OBJECTSTATE_BOP_PHASE1_ROUTE_AUDIT_SCHEMA == (
        "objgauss-objectstate-bop-phase1-route-audit-v1"
    )
    assert OBJECTSTATE_CONTROLLED_PREDICTION_EVIDENCE_PACKAGE_SCHEMA == (
        "objgauss-objectstate-controlled-prediction-evidence-package-v1"
    )
    assert OBJECTSTATE_CONTROLLED_IDENTITY_EVIDENCE_PACKAGE_SCHEMA == (
        "objgauss-objectstate-controlled-identity-evidence-package-v1"
    )
    assert OBJECTSTATE_CONTROLLED_REALITY_EVIDENCE_PACKAGE_SCHEMA == (
        "objgauss-objectstate-controlled-reality-evidence-package-v1"
    )
    assert OBJECTSTATE_PHASE1_EVIDENCE_LEDGER_SCHEMA == (
        "objgauss-objectstate-phase1-evidence-ledger-v1"
    )
    assert OBJECTSTATE_BOP_IDENTITY_ROUTE_AUDIT_SCHEMA == (
        "objgauss-objectstate-bop-identity-route-audit-v1"
    )
    assert OBJECTSTATE_BOP_PHASE1_LOCAL_ROW_READINESS_SCHEMA == (
        "objgauss-objectstate-bop-phase1-local-row-readiness-v1"
    )
    assert OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SCHEMA == (
        "objgauss-objectstate-bop-capture-condition-sidecar-v1"
    )
    assert OBJECTSTATE_BOP_CAPTURE_CONDITION_SIDECAR_SUMMARY_SCHEMA == (
        "objgauss-objectstate-bop-capture-condition-sidecar-summary-v1"
    )
    assert OBJECTSTATE_BOP_IDENTITY_HANDOFF_SCHEMA == (
        "objgauss-objectstate-bop-identity-handoff-v1"
    )
    assert OBJECTSTATE_BOP_LOCAL_ROW_HANDOFF_SCHEMA == (
        "objgauss-objectstate-bop-local-row-handoff-v1"
    )
    assert OBJECTSTATE_BOP_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA == (
        "objgauss-objectstate-bop-baseline-local-row-handoff-v1"
    )
    assert OBJECTSTATE_BOP_RGBD_BASELINE_LOCAL_ROW_HANDOFF_SCHEMA == (
        "objgauss-objectstate-bop-rgbd-baseline-local-row-handoff-v1"
    )
    assert OBJECTSTATE_BOP_REALITY_ROWS_SCHEMA == (
        "objgauss-objectstate-bop-reality-rows-v1"
    )
    assert OBJECTSTATE_BOP_CROSS_SAMPLE_LEDGER_SCHEMA == (
        "objgauss-objectstate-bop-cross-sample-ledger-v1"
    )
    assert OBJECTSTATE_BOP_LOCAL_ROW_BATCH_SPEC_SCHEMA == (
        "objgauss-objectstate-bop-local-row-batch-spec-v1"
    )
    assert OBJECTSTATE_BOP_LOCAL_ROW_BATCH_HANDOFF_SCHEMA == (
        "objgauss-objectstate-bop-local-row-batch-handoff-v1"
    )
    assert OBJECTSTATE_BOP_BASELINE_CANDIDATE_SCHEMA == (
        "objgauss-objectstate-bop-baseline-candidate-v1"
    )
    assert objectstate_controlled_reality_bundle_handoff is not None
    assert objectstate_controlled_reality_bundle_readiness is not None
    assert objectstate_controlled_reality_evidence_package is not None
    assert objectstate_controlled_identity_evidence_package is not None
    assert objectstate_controlled_prediction_evidence_package is not None
    assert objectstate_phase1_evidence_ledger is not None
    assert write_objectstate_controlled_reality_candidate_templates is not None
    assert write_objectstate_controlled_reality_candidate_templates_from_manifest is not None
    assert finalize_objectstate_controlled_reality_candidate_templates is not None
    assert finalize_objectstate_controlled_prediction_candidate_template is not None
    assert write_objectstate_controlled_prediction_baseline_candidates is not None
    assert objectstate_bop_prediction_baseline_handoff is not None
    assert objectstate_bop_identity_handoff is not None
    assert objectstate_bop_local_row_handoff is not None
    assert objectstate_bop_baseline_local_row_handoff is not None
    assert objectstate_bop_rgbd_baseline_local_row_handoff is not None
    assert objectstate_bop_reality_rows_summary is not None
    assert objectstate_bop_reality_rows_from_summary is not None
    assert read_objectstate_bop_local_row_summary is not None
    assert objectstate_bop_cross_sample_ledger is not None
    assert objectstate_bop_local_row_batch_handoff is not None
    assert read_objectstate_bop_local_row_batch_spec is not None
    assert objectstate_bop_capture_condition_sidecar_summary is not None
    assert objectstate_bop_phase1_route_audit is not None
    assert objectstate_bop_identity_route_audit is not None
    assert objectstate_bop_phase1_local_row_readiness is not None
    assert write_objectstate_bop_gaussian_centroid_baseline_candidate is not None
    assert validate_objectstate_bop_capture_condition_sidecar is not None
    assert validate_objectstate_bop_capture_condition_sidecar_summary is not None
    assert validate_objectstate_controlled_reality_bundle_handoff_summary is not None
    assert validate_objectstate_controlled_reality_bundle_readiness_summary is not None
    assert validate_objectstate_controlled_reality_candidate_template_summary is not None
    assert validate_objectstate_controlled_reality_candidate_finalize_summary is not None
    assert validate_objectstate_controlled_prediction_candidate_finalize_summary is not None
    assert validate_objectstate_controlled_prediction_baseline_summary is not None
    assert validate_objectstate_bop_prediction_baseline_handoff_summary is not None
    assert validate_objectstate_bop_identity_handoff_summary is not None
    assert validate_objectstate_bop_local_row_handoff_summary is not None
    assert validate_objectstate_bop_baseline_local_row_handoff_summary is not None
    assert validate_objectstate_bop_rgbd_baseline_local_row_handoff_summary is not None
    assert validate_objectstate_bop_reality_rows_summary is not None
    assert validate_objectstate_bop_cross_sample_ledger_summary is not None
    assert validate_objectstate_bop_local_row_batch_spec is not None
    assert validate_objectstate_bop_local_row_batch_handoff_summary is not None
    assert validate_objectstate_bop_phase1_route_audit_summary is not None
    assert validate_objectstate_bop_identity_route_audit_summary is not None
    assert validate_objectstate_bop_phase1_local_row_readiness_summary is not None
    assert validate_objectstate_bop_baseline_candidate_summary is not None
    assert validate_objectstate_controlled_identity_evidence_package_summary is not None
    assert validate_objectstate_phase1_evidence_ledger_summary is not None
    assert validate_objectstate_controlled_prediction_evidence_package_summary is not None
    assert (
        validate_objectstate_controlled_reality_evidence_package_summary
        is not None
    )
    assert validate_objectstate_controlled_prediction_candidates_template is not None
    assert validate_objectstate_controlled_intervention_candidates_template is not None
    assert objectstate_identity_predictions_from_trainable_artifact is not None
    assert objectstate_controlled_identity_handoff is not None
    assert validate_objectstate_controlled_identity_handoff_summary is not None
    assert read_trainable_kernel_identity_source is not None

    assert OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA == (
        "objgauss-objectstate-controlled-real-manifest-v1"
    )
    assert OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA == (
        "objgauss-objectstate-controlled-real-rows-v1"
    )
    controlled_manifest = {
        "schema": OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA,
        "sample": {
            "sample_id": "namespace-controlled-cup",
            "source_kind": "controlled_real",
            "object_category": "cup",
            "scenario": "identity_reappearance",
            "observation_modalities": ["rgb", "gaussian"],
            "artifact_refs": ["outputs/controlled-real/namespace/manifest.json"],
            "license": "local controlled capture",
        },
        "ground_truth": {
            "identity": True,
            "pose": False,
            "action": False,
            "timestamp": True,
        },
        "evidence_rows": [
            {
                "evidence_kind": "identity",
                "status": "pass",
                "metrics": {
                    "idf1": 1.0,
                    "fragmentation_rate": 0.0,
                    "swap_rate": 0.0,
                    "identity_collapse": False,
                },
            },
            {
                "evidence_kind": "prediction",
                "status": "blocked",
                "metrics": {},
                "block_reason": "missing pose tracks",
            },
        ],
    }
    assert validate_objectstate_controlled_real_manifest(controlled_manifest)[
        "schema"
    ] == OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA
    controlled_rows = objectstate_reality_rows_from_controlled_real_manifest(
        controlled_manifest
    )
    assert controlled_rows[0].status == "pass"
    controlled_report = evaluate_controlled_real_manifest_reality_gate(
        controlled_manifest,
        thresholds=ObjectStateRealityGateThresholds(
            require_prediction_pass_row=False,
            require_intervention_pass_row=False,
        ),
    )
    assert controlled_report.as_dict()["status"] == "objectstate_reality_gate_pass"
    controlled_summary = objectstate_controlled_real_rows_summary(controlled_manifest)
    assert validate_objectstate_controlled_real_rows_summary(controlled_summary) is controlled_summary
    assert controlled_summary["schema"] == OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA
    assert controlled_summary["pass_row_count"] == 1


def test_core_namespace_exposes_property_append_helper():
    cloud = _tiny_cloud()
    values = np.array([0, 0, 1, 1], dtype=np.int32)
    vertices = append_or_replace_property(cloud.vertices, "object_id", values, "i4")

    assert "object_id" in vertices.dtype.names
    assert vertices["object_id"].tolist() == [0, 0, 1, 1]


def test_core_namespace_exposes_chunk_index_builder():
    cloud = _tiny_cloud()
    vertices = append_or_replace_property(
        cloud.vertices,
        "object_id",
        np.array([0, 0, 1, 1], dtype=np.int32),
        "i4",
    )
    result = build_chunk_index(GaussianCloud(vertices=vertices), chunk_size_target=2)

    assert result.index["schema"] == "objgauss-chunk-index-v1"
    assert result.index["chunk_size_target"] == 2
    assert [chunk["object_id"] for chunk in result.index["chunks"]] == [0, 1]
    assert attach_object_aware_lod_metadata is not None
    assert attach_quantization_metadata is not None
    assert write_quantized_ogc_payload is not None


def test_core_namespace_exposes_trainable_kernel_mvp():
    result = train_kernel_mvp(
        make_trainable_kernel_mvp_fixture(),
        slots=2,
        iterations=4,
        learning_rate=0.25,
    )

    assert isinstance(result, TrainableKernelResult)
    assert result.schema == "objgauss-v1-trainable-kernel-mvp-v1"
    assert result.frame_count == 2
    sample = trainable_kernel_sample_from_cloud(_tiny_object_cloud(), frame_count=1)
    assert isinstance(sample, TrainableKernelSample)
    sample_result, sample_again = train_kernel_mvp_from_cloud(
        _tiny_object_cloud(),
        iterations=2,
        frame_count=1,
    )
    assert isinstance(sample_result, TrainableKernelResult)
    assert sample_again.target_source == "object_id_one_hot_targets"
    renderer_report = renderer_loss_boundary_report(sample_result.as_dict())
    assert isinstance(renderer_report, RendererLossBoundaryReport)
    assert validate_renderer_loss_boundary_summary(renderer_report.as_dict()) is True
    image_target = make_trainable_image_target(make_trainable_kernel_mvp_fixture()[0], width=6, height=5)
    assert isinstance(image_target, TrainableKernelImageTarget)
    assert isinstance(image_target.camera, TrainableKernelCamera)
    assert validate_trainable_image_target(image_target) is True
    bound_frames = bind_image_targets_to_frames(make_trainable_kernel_mvp_fixture(), width=6, height=5)
    image_contract = image_target_contract_summary(tuple(frame.image_target for frame in bound_frames))
    assert image_contract["status"] == "image_targets_bound"
    assert validate_image_target_contract_summary(image_contract) is True
    scale_plan = solver_decoder_training_scale_plan(
        total_iterations=3,
        checkpoint_every=2,
        loss_log_every=1,
        output_dir="fixture://scaled-run",
        image_renderer="point",
    )
    assert scale_plan["schema"] == TRAINING_SCALE_PLAN_SCHEMA
    assert validate_solver_decoder_training_scale_plan(scale_plan) == scale_plan
    assert TENSORBOARD_SCALAR_EXPORT_SCHEMA == "objgauss-tensorboard-scalar-export-v1"
    assert write_solver_decoder_tensorboard_events is not None
    assert OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA == "objgauss-objectstate-checkpoint-eval-v1"
    assert evaluate_solver_decoder_object_states is not None
    assert validate_objectstate_checkpoint_eval is not None
    assert assignment_loss_v2_breakdown is not None
    assert AssignmentEvidenceBatch is not None
    assert ASSIGNMENT_MVP_TRAINING_SCHEMA == "objgauss-assignment-mvp-training-v1"
    assert assignment_mvp_training_summary is not None
    assert ASSIGNMENT_STABILITY_EVAL_SCHEMA == "objgauss-assignment-stability-eval-v1"
    assert evaluate_assignment_stability is not None
    assert validate_assignment_stability_eval is not None
    assert assignment_evidence_from_trainable_frame is not None
    assert assignment_evidence_sequence_from_trainable_frames is not None
    assert assignment_cluster_loss_and_gradient is not None
    assert assignment_entropy_loss_and_gradient is not None
    assert assignment_balance_loss_and_gradient is not None
    assert supervised_assignment_loss_and_gradient is not None
    assert ASSIGNMENT_SOLVER_V2_STATE_SCHEMA == "objgauss-assignment-solver-state-v2"
    assert ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA == "objgauss-assignment-prediction-v2"
    assert ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA == "objgauss-assignment-solver-v2-training-v1"
    assert ASSIGNMENT_SOLVER_V2_COST_TERMS == ("feature", "position", "slot_bias")
    assignment = np.full((6, 2), 0.5, dtype=np.float32)
    loss_summary = assignment_loss_v2_breakdown([assignment], entropy_weight=0.1).as_dict()
    assert validate_assignment_loss_v2_summary(loss_summary) is True
    evidence_batch = assignment_evidence_from_trainable_frame(bound_frames[0])
    evidence_summary = evidence_batch.as_dict()
    assert validate_assignment_evidence_summary(evidence_summary) is True
    solver_v2 = initialize_assignment_solver_v2(slots=2, feature_dim=bound_frames[0].features.shape[1], seed=2)
    assert isinstance(solver_v2, AssignmentSolverV2State)
    assert isinstance(solver_v2.config, AssignmentSolverV2Config)
    assert validate_assignment_solver_v2_config(solver_v2.config) is solver_v2.config
    solver_v2 = validate_assignment_solver_v2_state(solver_v2)
    restored_solver_v2 = assignment_solver_v2_state_from_dict(solver_v2.as_dict(include_arrays=True))
    np.testing.assert_allclose(restored_solver_v2.feature_centers, solver_v2.feature_centers, atol=1e-6)
    solver_v2_prediction = predict_assignment_solver_v2(evidence_batch, restored_solver_v2)
    assert isinstance(solver_v2_prediction, AssignmentSolverV2Prediction)
    solver_v2_training = train_assignment_solver_v2(
        [evidence_batch],
        initial_state=restored_solver_v2,
        iterations=1,
        learning_rate=0.1,
        cluster_weight=0.0,
        entropy_weight=0.1,
        balance_weight=0.0,
        supervised_weight=0.0,
    )
    assert isinstance(solver_v2_training, AssignmentSolverV2TrainingResult)
    solver_v2_summary = solver_v2_training.as_dict()
    assert validate_assignment_solver_v2_training_summary(solver_v2_summary) is solver_v2_summary
    assert ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA == "objgauss-assignment-solver-v2-checkpoint"
    assert ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA == (
        "objgauss-assignment-solver-v2-stability-eval-v1"
    )
    solver_v2_checkpoint = assignment_solver_v2_checkpoint(
        solver_v2_training,
        source="fixture://namespace",
    )
    assert validate_assignment_solver_v2_checkpoint(solver_v2_checkpoint) is solver_v2_checkpoint
    restored_solver_v2_checkpoint = assignment_solver_v2_state_from_checkpoint(solver_v2_checkpoint)
    assert isinstance(restored_solver_v2_checkpoint, AssignmentSolverV2State)
    assert restored_solver_v2_checkpoint.step == solver_v2_training.final_state.step
    assert AssignmentSolverV2StabilityEvalReport is not None
    assert evaluate_assignment_solver_v2_stability is not None
    assert validate_assignment_solver_v2_stability_eval_summary is not None
    assert ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA == (
        "objgauss-assignment-v2-render-joint-validation-v1"
    )
    assert AssignmentV2RendererJointValidationReport is not None
    assert evaluate_assignment_v2_renderer_joint is not None
    assert validate_assignment_v2_renderer_joint_summary is not None
    assert CORE_MODEL_TRAIN_VALIDATE_SCHEMA == "objgauss-core-model-train-validate-v1"
    assert CoreModelTrainValidateReport is not None
    assert core_model_train_validate_report is not None
    assert validate_core_model_train_validate_summary is not None
    assert REAL_SAMPLE_V2_SMOKE_SCHEMA == "objgauss-real-sample-v2-smoke-v1"
    assert RealSampleV2SmokeReport is not None
    assert real_sample_v2_smoke_from_cloud is not None
    assert evaluate_real_sample_v2_smoke is not None
    assert validate_real_sample_v2_smoke_summary is not None
    assert REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA == "objgauss-real-sample-v2-diagnostics-v1"
    assert RealSampleV2DiagnosticsReport is not None
    assert real_sample_v2_diagnostics_from_cloud is not None
    assert evaluate_real_sample_v2_diagnostics is not None
    assert validate_real_sample_v2_diagnostics_summary is not None
    assert REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA == "objgauss-real-sample-v2-model-handoff-v1"
    assert REAL_SAMPLE_V2_EFFECT_PREVIEW_SCHEMA == "objgauss-real-sample-v2-effect-preview-v1"
    assert RealSampleV2ModelHandoffReport is not None
    assert real_sample_v2_model_handoff_from_cloud is not None
    assert evaluate_real_sample_v2_model_handoff is not None
    assert render_real_sample_v2_model_handoff_html is not None
    assert validate_real_sample_v2_model_handoff_summary is not None
    assert validate_real_sample_v2_effect_preview is not None
    assert REAL_SAMPLE_V2_FULL_CLOUD_PURITY_SCHEMA == (
        "objgauss-real-sample-v2-full-cloud-purity-v1"
    )
    assert RealSampleV2FullCloudPurityReport is not None
    assert real_sample_v2_full_cloud_purity_from_cloud is not None
    assert validate_real_sample_v2_full_cloud_purity_summary is not None
    assert REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA == (
        "objgauss-real-sample-v2-segmentation-quality-v1"
    )
    assert RealSampleV2SegmentationQualityReport is not None
    assert real_sample_v2_segmentation_quality_from_cloud is not None
    assert real_sample_v2_segmentation_quality_from_projected_cloud is not None
    assert real_sample_v2_segmentation_quality_from_purity_report is not None
    assert validate_real_sample_v2_segmentation_quality_summary is not None
    assert REAL_SAMPLE_V2_WEAK_BOUNDARY_OPT_SCHEMA == (
        "objgauss-real-sample-v2-weak-boundary-opt-v1"
    )
    assert RealSampleV2WeakBoundaryOptReport is not None
    assert real_sample_v2_weak_boundary_opt_from_cloud is not None
    assert validate_real_sample_v2_weak_boundary_opt_summary is not None
    assert REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA == (
        "objgauss-real-sample-v2-promoted-weights-cross-sample-v1"
    )
    assert RealSampleV2PromotedWeightsCrossSampleReport is not None
    assert real_sample_v2_promoted_weights_cross_sample_from_cloud is not None
    assert validate_real_sample_v2_promoted_weights_cross_sample_summary is not None
    assert REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA == (
        "objgauss-real-sample-v2-bounded-normalization-cross-sample-v1"
    )
    assert RealSampleV2BoundedNormalizationCrossSampleInput is not None
    assert RealSampleV2BoundedNormalizationCrossSampleReport is not None
    assert real_sample_v2_bounded_normalization_cross_sample_from_clouds is not None
    assert validate_real_sample_v2_bounded_normalization_cross_sample_summary is not None
    assert REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA == (
        "objgauss-real-sample-v2-sample-aware-weight-policy-v1"
    )
    assert RealSampleV2SampleAwareWeightPolicyReport is not None
    assert real_sample_v2_sample_aware_weight_policy_from_cloud is not None
    assert validate_real_sample_v2_sample_aware_weight_policy_summary is not None
    renderer_result = evaluate_training_renderer_loss(
        bound_frames[:1],
        [assignment],
        np.asarray([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]], dtype=np.float32),
    )
    assert isinstance(renderer_result, TrainingRendererLossResult)
    assert validate_training_renderer_summary(renderer_result.as_dict()) is True
    gsplat_availability = gsplat_renderer_availability(_importer=_missing_importer)
    assert isinstance(gsplat_availability, GsplatRendererAvailability)
    assert gsplat_availability.available is False
    gsplat_input = build_gsplat_training_input(
        bound_frames[0],
        assignment,
        np.asarray([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]], dtype=np.float32),
    )
    assert isinstance(gsplat_input, GsplatTrainingInput)
    object_assignment = np.zeros((bound_frames[0].positions.shape[0], 2), dtype=np.float32)
    object_assignment[:3, 0] = 1.0
    object_assignment[3:, 1] = 1.0
    object_projection = project_object_states(
        _frame_cloud_from_positions(bound_frames[0].positions),
        object_assignment,
        evidence_features=bound_frames[0].features,
    )
    object_state_input = build_gsplat_training_input_from_object_state(
        bound_frames[0],
        object_projection,
        np.asarray([[0.2, 0.3, 0.4], [0.6, 0.7, 0.8]], dtype=np.float32),
    )
    assert object_state_input.decoder_schema == "objgauss-object-state-gaussian-decode-v1"
    decoder_state = initialize_object_state_gaussian_decoder(slots=2, seed=1)
    assert validate_object_state_gaussian_decoder_state(decoder_state) is decoder_state
    restored_decoder_state = object_state_gaussian_decoder_state_from_dict(decoder_state.as_dict())
    assert isinstance(restored_decoder_state, ObjectStateGaussianDecoderState)
    np.testing.assert_allclose(restored_decoder_state.object_colors, decoder_state.object_colors, atol=1e-6)
    decoder_result = train_object_state_gaussian_decoder(
        bound_frames[:1],
        [object_assignment],
        initial_state=decoder_state,
        iterations=1,
        learning_rate=0.2,
    )
    assert isinstance(decoder_result, ObjectStateGaussianDecoderTrainingResult)
    assert decoder_result.as_dict()["trained_fields"] == ["object_colors"]
    joint_frame = type(bound_frames[0])(
        positions=bound_frames[0].positions,
        features=bound_frames[0].features,
        target_rgb=bound_frames[0].target_rgb,
        target_assignment=object_assignment,
        image_target=bound_frames[0].image_target,
    )
    joint_result = train_solver_decoder_joint(
        [joint_frame],
        iterations=1,
        solver_learning_rate=0.05,
        decoder_learning_rate=0.2,
        object_weight=0.1,
    )
    assert isinstance(joint_result, SolverDecoderJointTrainingResult)
    assert "decoder.object_colors" in joint_result.as_dict()["trained_fields"]
    joint_checkpoint = solver_decoder_joint_checkpoint(
        joint_result,
        input_path="fixture://namespace-joint",
        source_gaussians=bound_frames[0].positions.shape[0],
        sampled_gaussians=bound_frames[0].positions.shape[0],
        target_source="object_id_one_hot_targets",
        assignment_source="object_id_one_hot_targets",
    )
    assert validate_solver_decoder_joint_checkpoint(joint_checkpoint) == joint_checkpoint
    restored_solver, restored_joint_decoder = solver_decoder_joint_states_from_dict(joint_checkpoint)
    assert isinstance(restored_solver, ObjectEmergenceSolverState)
    assert isinstance(restored_joint_decoder, ObjectStateGaussianDecoderState)
    assert evaluate_gsplat_training_renderer_loss is not None
    artifact = trainable_kernel_model_artifact(
        result,
        input_path="fixture://namespace",
        renderer_api=renderer_result.as_dict(),
    )
    assert validate_trainable_kernel_model_artifact(artifact) is True
    assert write_trainable_kernel_model_artifact is not None


def _tiny_cloud() -> GaussianCloud:
    vertices = np.zeros(
        4,
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
            ("opacity", "f4"),
        ],
    )
    vertices["x"] = np.array([-1.0, -0.8, 0.8, 1.0], dtype=np.float32)
    vertices["y"] = np.array([0.0, 0.1, 0.0, -0.1], dtype=np.float32)
    vertices["z"] = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    vertices["red"] = np.array([240, 230, 20, 30], dtype=np.uint8)
    vertices["green"] = np.array([20, 30, 230, 240], dtype=np.uint8)
    vertices["blue"] = np.array([20, 30, 20, 30], dtype=np.uint8)
    vertices["opacity"] = np.ones(4, dtype=np.float32)
    return GaussianCloud(vertices=vertices, source_format="ascii")


def _tiny_object_cloud() -> GaussianCloud:
    cloud = _tiny_cloud()
    vertices = append_or_replace_property(
        cloud.vertices,
        "object_id",
        np.array([0, 0, 1, 1], dtype=np.int32),
        "i4",
    )
    return GaussianCloud(vertices=vertices, source_format="ascii")


def _frame_cloud_from_positions(frame_positions: np.ndarray) -> GaussianCloud:
    xyz = np.asarray(frame_positions, dtype=np.float32)
    vertices = np.zeros(
        xyz.shape[0],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
        ],
    )
    vertices["x"] = xyz[:, 0]
    vertices["y"] = xyz[:, 1]
    vertices["z"] = xyz[:, 2]
    return GaussianCloud(vertices=vertices, source_format="frame_fixture")


def _missing_importer(name: str):
    raise ImportError(name)
