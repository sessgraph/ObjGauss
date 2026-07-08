"""Stable core algorithm entry points for ObjGauss.

The namespace is intentionally lazy. Compatibility wrappers such as
`objgauss.gaussians` import specific core submodules during package
initialization, so eagerly importing every core domain here can create circular
imports while the migration is in progress.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "AssignmentEvidenceBatch": (
        "objgauss.core.assignment_evidence",
        "AssignmentEvidenceBatch",
    ),
    "AssignmentSolverV2Config": (
        "objgauss.core.assignment_solver_v2",
        "AssignmentSolverV2Config",
    ),
    "AssignmentSolverV2LossRecord": (
        "objgauss.core.assignment_solver_v2",
        "AssignmentSolverV2LossRecord",
    ),
    "AssignmentSolverV2Prediction": (
        "objgauss.core.assignment_solver_v2",
        "AssignmentSolverV2Prediction",
    ),
    "AssignmentSolverV2State": (
        "objgauss.core.assignment_solver_v2",
        "AssignmentSolverV2State",
    ),
    "AssignmentSolverV2TrainingResult": (
        "objgauss.core.assignment_solver_v2",
        "AssignmentSolverV2TrainingResult",
    ),
    "AssignmentSolverV2StabilityEvalReport": (
        "objgauss.core.assignment_solver_v2_eval",
        "AssignmentSolverV2StabilityEvalReport",
    ),
    "AssignmentV2RendererJointValidationReport": (
        "objgauss.core.assignment_v2_renderer_validation",
        "AssignmentV2RendererJointValidationReport",
    ),
    "CoreModelTrainValidateReport": (
        "objgauss.core.core_model_validation",
        "CoreModelTrainValidateReport",
    ),
    "RealSampleV2SmokeReport": (
        "objgauss.core.real_sample_v2_smoke",
        "RealSampleV2SmokeReport",
    ),
    "RealSampleV2DiagnosticsReport": (
        "objgauss.core.real_sample_v2_diagnostics",
        "RealSampleV2DiagnosticsReport",
    ),
    "RealSampleV2ModelHandoffReport": (
        "objgauss.core.real_sample_v2_model_handoff",
        "RealSampleV2ModelHandoffReport",
    ),
    "RealSampleV2ViewerPreviewReport": (
        "objgauss.core.real_sample_v2_viewer_preview",
        "RealSampleV2ViewerPreviewReport",
    ),
    "RealSampleV2FullCloudPurityReport": (
        "objgauss.core.real_sample_v2_full_cloud_purity",
        "RealSampleV2FullCloudPurityReport",
    ),
    "RealSampleV2SegmentationQualityReport": (
        "objgauss.core.real_sample_v2_segmentation_quality",
        "RealSampleV2SegmentationQualityReport",
    ),
    "RealSampleV2WeakBoundaryOptReport": (
        "objgauss.core.real_sample_v2_weak_boundary_opt",
        "RealSampleV2WeakBoundaryOptReport",
    ),
    "RealSampleV2PromotedWeightsCrossSampleReport": (
        "objgauss.core.real_sample_v2_promoted_weights_cross_sample",
        "RealSampleV2PromotedWeightsCrossSampleReport",
    ),
    "RealSampleV2SampleAwareWeightPolicyReport": (
        "objgauss.core.real_sample_v2_sample_aware_weight_policy",
        "RealSampleV2SampleAwareWeightPolicyReport",
    ),
    "RealSampleV2BoundedNormalizationCrossSampleInput": (
        "objgauss.core.real_sample_v2_bounded_normalization_cross_sample",
        "RealSampleV2BoundedNormalizationCrossSampleInput",
    ),
    "RealSampleV2BoundedNormalizationCrossSampleReport": (
        "objgauss.core.real_sample_v2_bounded_normalization_cross_sample",
        "RealSampleV2BoundedNormalizationCrossSampleReport",
    ),
    "ObjectIdentityObservation": (
        "objgauss.core.v2_stability_foundation",
        "ObjectIdentityObservation",
    ),
    "ObjectStateIdentityGateReport": (
        "objgauss.core.objectstate_identity_gate",
        "ObjectStateIdentityGateReport",
    ),
    "ObjectStateIdentityEncoderConfig": (
        "objgauss.core.objectstate_identity_encoder",
        "ObjectStateIdentityEncoderConfig",
    ),
    "ObjectStateIdentityEncoderState": (
        "objgauss.core.objectstate_identity_encoder",
        "ObjectStateIdentityEncoderState",
    ),
    "ObjectStateIdentityEncoderTrainingResult": (
        "objgauss.core.objectstate_identity_encoder",
        "ObjectStateIdentityEncoderTrainingResult",
    ),
    "ObjectStateIdentityGateThresholds": (
        "objgauss.core.objectstate_identity_gate",
        "ObjectStateIdentityGateThresholds",
    ),
    "ObjectStateIdentityRow": (
        "objgauss.core.objectstate_identity_gate",
        "ObjectStateIdentityRow",
    ),
    "ObjectStatePredictiveGateReport": (
        "objgauss.core.objectstate_predictive_gate",
        "ObjectStatePredictiveGateReport",
    ),
    "ObjectStatePredictiveGateThresholds": (
        "objgauss.core.objectstate_predictive_gate",
        "ObjectStatePredictiveGateThresholds",
    ),
    "ObjectStatePredictiveRow": (
        "objgauss.core.objectstate_predictive_gate",
        "ObjectStatePredictiveRow",
    ),
    "ObjectStateRealityGateReport": (
        "objgauss.core.objectstate_reality_gate",
        "ObjectStateRealityGateReport",
    ),
    "ObjectStateRealityGateThresholds": (
        "objgauss.core.objectstate_reality_gate",
        "ObjectStateRealityGateThresholds",
    ),
    "ObjectStateRealityRow": (
        "objgauss.core.objectstate_reality_gate",
        "ObjectStateRealityRow",
    ),
    "ObjectStateRealityPublicArtifact": (
        "objgauss.core.objectstate_reality_public_rows",
        "ObjectStateRealityPublicArtifact",
    ),
    "OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA": (
        "objgauss.core.objectstate_controlled_capture",
        "OBJECTSTATE_CONTROLLED_CAPTURE_MANIFEST_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA": (
        "objgauss.core.objectstate_controlled_capture",
        "OBJECTSTATE_CONTROLLED_CAPTURE_SUMMARY_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA": (
        "objgauss.core.objectstate_controlled_capture_files",
        "OBJECTSTATE_CONTROLLED_CAPTURE_FILE_AUDIT_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA": (
        "objgauss.core.objectstate_controlled_capture_import",
        "OBJECTSTATE_CONTROLLED_CAPTURE_IMPORT_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_ACCEPTANCE_SCHEMA": (
        "objgauss.core.objectstate_controlled_capture_import",
        "OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_ACCEPTANCE_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA": (
        "objgauss.core.objectstate_controlled_capture_template",
        "OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_TEMPLATE_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA": (
        "objgauss.core.objectstate_controlled_capture_bundle_readiness",
        "OBJECTSTATE_CONTROLLED_CAPTURE_BUNDLE_READINESS_SCHEMA",
    ),
    "objectstate_controlled_capture_bundle_acceptance_summary": (
        "objgauss.core.objectstate_controlled_capture_import",
        "objectstate_controlled_capture_bundle_acceptance_summary",
    ),
    "objectstate_controlled_capture_import_summary": (
        "objgauss.core.objectstate_controlled_capture_import",
        "objectstate_controlled_capture_import_summary",
    ),
    "objectstate_controlled_capture_manifest_from_bundle": (
        "objgauss.core.objectstate_controlled_capture_import",
        "objectstate_controlled_capture_manifest_from_bundle",
    ),
    "write_objectstate_controlled_capture_bundle_template": (
        "objgauss.core.objectstate_controlled_capture_template",
        "write_objectstate_controlled_capture_bundle_template",
    ),
    "objectstate_controlled_capture_bundle_readiness": (
        "objgauss.core.objectstate_controlled_capture_bundle_readiness",
        "objectstate_controlled_capture_bundle_readiness",
    ),
    "validate_objectstate_controlled_capture_import_summary": (
        "objgauss.core.objectstate_controlled_capture_import",
        "validate_objectstate_controlled_capture_import_summary",
    ),
    "validate_objectstate_controlled_capture_bundle_acceptance_summary": (
        "objgauss.core.objectstate_controlled_capture_import",
        "validate_objectstate_controlled_capture_bundle_acceptance_summary",
    ),
    "validate_objectstate_controlled_capture_bundle_template_summary": (
        "objgauss.core.objectstate_controlled_capture_template",
        "validate_objectstate_controlled_capture_bundle_template_summary",
    ),
    "validate_objectstate_controlled_capture_bundle_readiness_summary": (
        "objgauss.core.objectstate_controlled_capture_bundle_readiness",
        "validate_objectstate_controlled_capture_bundle_readiness_summary",
    ),
    "OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA": (
        "objgauss.core.objectstate_controlled_identity_eval",
        "OBJECTSTATE_CONTROLLED_IDENTITY_EVAL_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA": (
        "objgauss.core.objectstate_controlled_identity_handoff",
        "OBJECTSTATE_CONTROLLED_IDENTITY_HANDOFF_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA": (
        "objgauss.core.objectstate_controlled_identity_bundle_handoff",
        "OBJECTSTATE_CONTROLLED_IDENTITY_BUNDLE_HANDOFF_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA": (
        "objgauss.core.objectstate_controlled_identity_handoff",
        "OBJECTSTATE_CONTROLLED_CANDIDATE_ARTIFACT_FILE_AUDIT_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA": (
        "objgauss.core.objectstate_controlled_identity_handoff",
        "OBJECTSTATE_CONTROLLED_IDENTITY_SCENARIO_AUDIT_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA": (
        "objgauss.core.objectstate_controlled_identity_eval",
        "OBJECTSTATE_CONTROLLED_IDENTITY_PREDICTIONS_SCHEMA",
    ),
    "ObjectStateControlledIdentityThresholds": (
        "objgauss.core.objectstate_controlled_identity_eval",
        "ObjectStateControlledIdentityThresholds",
    ),
    "OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA": (
        "objgauss.core.objectstate_controlled_prediction_eval",
        "OBJECTSTATE_CONTROLLED_PREDICTION_CANDIDATES_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA": (
        "objgauss.core.objectstate_controlled_prediction_eval",
        "OBJECTSTATE_CONTROLLED_PREDICTION_EVAL_SCHEMA",
    ),
    "ObjectStateControlledPredictionThresholds": (
        "objgauss.core.objectstate_controlled_prediction_eval",
        "ObjectStateControlledPredictionThresholds",
    ),
    "OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA": (
        "objgauss.core.objectstate_controlled_intervention_eval",
        "OBJECTSTATE_CONTROLLED_INTERVENTION_CANDIDATES_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA": (
        "objgauss.core.objectstate_controlled_intervention_eval",
        "OBJECTSTATE_CONTROLLED_INTERVENTION_EVAL_SCHEMA",
    ),
    "ObjectStateControlledInterventionThresholds": (
        "objgauss.core.objectstate_controlled_intervention_eval",
        "ObjectStateControlledInterventionThresholds",
    ),
    "evaluate_objectstate_controlled_intervention_candidates": (
        "objgauss.core.objectstate_controlled_intervention_eval",
        "evaluate_objectstate_controlled_intervention_candidates",
    ),
    "validate_objectstate_controlled_intervention_candidates": (
        "objgauss.core.objectstate_controlled_intervention_eval",
        "validate_objectstate_controlled_intervention_candidates",
    ),
    "validate_objectstate_controlled_intervention_eval_summary": (
        "objgauss.core.objectstate_controlled_intervention_eval",
        "validate_objectstate_controlled_intervention_eval_summary",
    ),
    "evaluate_objectstate_controlled_prediction_candidates": (
        "objgauss.core.objectstate_controlled_prediction_eval",
        "evaluate_objectstate_controlled_prediction_candidates",
    ),
    "validate_objectstate_controlled_prediction_candidates": (
        "objgauss.core.objectstate_controlled_prediction_eval",
        "validate_objectstate_controlled_prediction_candidates",
    ),
    "validate_objectstate_controlled_prediction_eval_summary": (
        "objgauss.core.objectstate_controlled_prediction_eval",
        "validate_objectstate_controlled_prediction_eval_summary",
    ),
    "OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA": (
        "objgauss.core.objectstate_controlled_real_rows",
        "OBJECTSTATE_CONTROLLED_REAL_MANIFEST_SCHEMA",
    ),
    "OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA": (
        "objgauss.core.objectstate_controlled_real_rows",
        "OBJECTSTATE_CONTROLLED_REAL_ROWS_SCHEMA",
    ),
    "ObjectIdentityOracle": (
        "objgauss.core.v2_stability_foundation",
        "ObjectIdentityOracle",
    ),
    "ObjectIdentityRecord": (
        "objgauss.core.v2_stability_foundation",
        "ObjectIdentityRecord",
    ),
    "FailureModeClassifier": (
        "objgauss.core.v2_stability_diagnostics",
        "FailureModeClassifier",
    ),
    "FailureModeEvent": (
        "objgauss.core.v2_stability_diagnostics",
        "FailureModeEvent",
    ),
    "GaussianCloud": ("objgauss.core.gaussian", "GaussianCloud"),
    "IdentitySlotObservation": (
        "objgauss.core.v2_stability_diagnostics",
        "IdentitySlotObservation",
    ),
    "ObjectField": ("objgauss.core.object_field", "ObjectField"),
    "ObservationModelConfig": (
        "objgauss.core.v2_stability_foundation",
        "ObservationModelConfig",
    ),
    "ObjectStateGaussianDecode": (
        "objgauss.core.gaussian_decoder",
        "ObjectStateGaussianDecode",
    ),
    "ObjectStateGaussianDecoderLoss": (
        "objgauss.core.gaussian_decoder_training",
        "ObjectStateGaussianDecoderLoss",
    ),
    "ObjectStateGaussianDecoderState": (
        "objgauss.core.gaussian_decoder_training",
        "ObjectStateGaussianDecoderState",
    ),
    "ObjectStateGaussianDecoderTrainingResult": (
        "objgauss.core.gaussian_decoder_training",
        "ObjectStateGaussianDecoderTrainingResult",
    ),
    "SolverDecoderJointLoss": (
        "objgauss.core.solver_decoder_training",
        "SolverDecoderJointLoss",
    ),
    "SolverDecoderJointTrainingResult": (
        "objgauss.core.solver_decoder_training",
        "SolverDecoderJointTrainingResult",
    ),
    "SyntheticObservationFrame": (
        "objgauss.core.v2_stability_foundation",
        "SyntheticObservationFrame",
    ),
    "SyntheticStabilityScenarioFixture": (
        "objgauss.core.v2_stability_foundation",
        "SyntheticStabilityScenarioFixture",
    ),
    "SyntheticStabilityDiagnosticsReport": (
        "objgauss.core.v2_stability_diagnostics",
        "SyntheticStabilityDiagnosticsReport",
    ),
    "SyntheticStabilityGateReport": (
        "objgauss.core.v2_stability_gate",
        "SyntheticStabilityGateReport",
    ),
    "SyntheticStabilitySuiteGateReport": (
        "objgauss.core.v2_stability_gate",
        "SyntheticStabilitySuiteGateReport",
    ),
    "SyntheticWorldFrame": (
        "objgauss.core.v2_stability_foundation",
        "SyntheticWorldFrame",
    ),
    "SyntheticWorldObject": (
        "objgauss.core.v2_stability_foundation",
        "SyntheticWorldObject",
    ),
    "SyntheticWorldState": (
        "objgauss.core.v2_stability_foundation",
        "SyntheticWorldState",
    ),
    "solver_decoder_joint_checkpoint": (
        "objgauss.core.solver_decoder_training",
        "solver_decoder_joint_checkpoint",
    ),
    "ObjectState": ("objgauss.core.object_state", "ObjectState"),
    "ObjectStateAction": (
        "objgauss.core.objectstate_causal_gate",
        "ObjectStateAction",
    ),
    "ObjectStateCausalGateReport": (
        "objgauss.core.objectstate_causal_gate",
        "ObjectStateCausalGateReport",
    ),
    "ObjectStateCausalGateThresholds": (
        "objgauss.core.objectstate_causal_gate",
        "ObjectStateCausalGateThresholds",
    ),
    "ObjectStateCausalRow": (
        "objgauss.core.objectstate_causal_gate",
        "ObjectStateCausalRow",
    ),
    "OBJECTSTATE_REALITY_EVIDENCE_KINDS": (
        "objgauss.core.objectstate_reality_gate",
        "OBJECTSTATE_REALITY_EVIDENCE_KINDS",
    ),
    "OBJECTSTATE_REALITY_GATE_SCHEMA": (
        "objgauss.core.objectstate_reality_gate",
        "OBJECTSTATE_REALITY_GATE_SCHEMA",
    ),
    "OBJECTSTATE_REALITY_ROW_SCHEMA": (
        "objgauss.core.objectstate_reality_gate",
        "OBJECTSTATE_REALITY_ROW_SCHEMA",
    ),
    "OBJECTSTATE_REALITY_ROW_STATUSES": (
        "objgauss.core.objectstate_reality_gate",
        "OBJECTSTATE_REALITY_ROW_STATUSES",
    ),
    "OBJECTSTATE_REALITY_SOURCE_KINDS": (
        "objgauss.core.objectstate_reality_gate",
        "OBJECTSTATE_REALITY_SOURCE_KINDS",
    ),
    "OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA": (
        "objgauss.core.objectstate_reality_public_rows",
        "OBJECTSTATE_REALITY_PUBLIC_ROWS_SCHEMA",
    ),
    "ObjectStateMatch": ("objgauss.core.object_state", "ObjectStateMatch"),
    "ObjectStateProjection": ("objgauss.core.object_state", "ObjectStateProjection"),
    "ObjectStabilityReport": ("objgauss.core.object_state", "ObjectStabilityReport"),
    "OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA": (
        "objgauss.core.object_state_eval",
        "OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA",
    ),
    "ObjectTemporalMatchReport": ("objgauss.core.object_state", "ObjectTemporalMatchReport"),
    "DynamicKProposal": ("objgauss.core.object_state", "DynamicKProposal"),
    "DynamicKProposalReport": ("objgauss.core.object_state", "DynamicKProposalReport"),
    "DynamicKUpdateAction": ("objgauss.core.object_state", "DynamicKUpdateAction"),
    "DynamicKUpdatePlan": ("objgauss.core.object_state", "DynamicKUpdatePlan"),
    "ObjectEmergenceAssignmentPrediction": (
        "objgauss.core.object_emergence_solver",
        "ObjectEmergenceAssignmentPrediction",
    ),
    "ObjectEmergenceEvidence": (
        "objgauss.core.object_emergence_solver",
        "ObjectEmergenceEvidence",
    ),
    "ObjectEmergenceSolverConfig": (
        "objgauss.core.object_emergence_solver",
        "ObjectEmergenceSolverConfig",
    ),
    "ObjectEmergenceSolverLoss": (
        "objgauss.core.object_emergence_solver",
        "ObjectEmergenceSolverLoss",
    ),
    "ObjectEmergenceSolverState": (
        "objgauss.core.object_emergence_solver",
        "ObjectEmergenceSolverState",
    ),
    "ObjectEmergenceSolverTrainingResult": (
        "objgauss.core.object_emergence_solver",
        "ObjectEmergenceSolverTrainingResult",
    ),
    "ASSIGNMENT_MVP_TRAINING_SCHEMA": (
        "objgauss.core.object_emergence_solver",
        "ASSIGNMENT_MVP_TRAINING_SCHEMA",
    ),
    "ASSIGNMENT_STABILITY_EVAL_SCHEMA": (
        "objgauss.core.assignment_stability",
        "ASSIGNMENT_STABILITY_EVAL_SCHEMA",
    ),
    "ASSIGNMENT_SOLVER_V2_COST_TERMS": (
        "objgauss.core.assignment_solver_v2",
        "ASSIGNMENT_SOLVER_V2_COST_TERMS",
    ),
    "ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA": (
        "objgauss.core.assignment_solver_v2",
        "ASSIGNMENT_SOLVER_V2_PREDICTION_SCHEMA",
    ),
    "ASSIGNMENT_SOLVER_V2_STATE_SCHEMA": (
        "objgauss.core.assignment_solver_v2",
        "ASSIGNMENT_SOLVER_V2_STATE_SCHEMA",
    ),
    "ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA": (
        "objgauss.core.assignment_solver_v2",
        "ASSIGNMENT_SOLVER_V2_TRAINING_SCHEMA",
    ),
    "ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA": (
        "objgauss.core.assignment_solver_v2_eval",
        "ASSIGNMENT_SOLVER_V2_CHECKPOINT_SCHEMA",
    ),
    "ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA": (
        "objgauss.core.assignment_solver_v2_eval",
        "ASSIGNMENT_SOLVER_V2_STABILITY_EVAL_SCHEMA",
    ),
    "ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA": (
        "objgauss.core.assignment_v2_renderer_validation",
        "ASSIGNMENT_V2_RENDER_JOINT_VALIDATION_SCHEMA",
    ),
    "CORE_MODEL_TRAIN_VALIDATE_SCHEMA": (
        "objgauss.core.core_model_validation",
        "CORE_MODEL_TRAIN_VALIDATE_SCHEMA",
    ),
    "REAL_SAMPLE_V2_SMOKE_SCHEMA": (
        "objgauss.core.real_sample_v2_smoke",
        "REAL_SAMPLE_V2_SMOKE_SCHEMA",
    ),
    "REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA": (
        "objgauss.core.real_sample_v2_diagnostics",
        "REAL_SAMPLE_V2_DIAGNOSTICS_SCHEMA",
    ),
    "REAL_SAMPLE_V2_EFFECT_PREVIEW_SCHEMA": (
        "objgauss.core.real_sample_v2_model_handoff",
        "REAL_SAMPLE_V2_EFFECT_PREVIEW_SCHEMA",
    ),
    "REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA": (
        "objgauss.core.real_sample_v2_model_handoff",
        "REAL_SAMPLE_V2_MODEL_HANDOFF_SCHEMA",
    ),
    "REAL_SAMPLE_V2_FULL_CLOUD_PURITY_SCHEMA": (
        "objgauss.core.real_sample_v2_full_cloud_purity",
        "REAL_SAMPLE_V2_FULL_CLOUD_PURITY_SCHEMA",
    ),
    "REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA": (
        "objgauss.core.real_sample_v2_segmentation_quality",
        "REAL_SAMPLE_V2_SEGMENTATION_QUALITY_SCHEMA",
    ),
    "REAL_SAMPLE_V2_WEAK_BOUNDARY_OPT_SCHEMA": (
        "objgauss.core.real_sample_v2_weak_boundary_opt",
        "REAL_SAMPLE_V2_WEAK_BOUNDARY_OPT_SCHEMA",
    ),
    "REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA": (
        "objgauss.core.real_sample_v2_promoted_weights_cross_sample",
        "REAL_SAMPLE_V2_PROMOTED_WEIGHTS_CROSS_SAMPLE_SCHEMA",
    ),
    "REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA": (
        "objgauss.core.real_sample_v2_sample_aware_weight_policy",
        "REAL_SAMPLE_V2_SAMPLE_AWARE_WEIGHT_POLICY_SCHEMA",
    ),
    "REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA": (
        "objgauss.core.real_sample_v2_bounded_normalization_cross_sample",
        "REAL_SAMPLE_V2_BOUNDED_NORMALIZATION_CROSS_SAMPLE_SCHEMA",
    ),
    "V2_STABILITY_FOUNDATION_SCHEMA": (
        "objgauss.core.v2_stability_foundation",
        "V2_STABILITY_FOUNDATION_SCHEMA",
    ),
    "V2_SYNTHETIC_OBSERVATION_SCHEMA": (
        "objgauss.core.v2_stability_foundation",
        "V2_SYNTHETIC_OBSERVATION_SCHEMA",
    ),
    "V2_STABILITY_SCENARIO_FIXTURE_SCHEMA": (
        "objgauss.core.v2_stability_foundation",
        "V2_STABILITY_SCENARIO_FIXTURE_SCHEMA",
    ),
    "V2_STABILITY_SCENARIO_KINDS": (
        "objgauss.core.v2_stability_foundation",
        "V2_STABILITY_SCENARIO_KINDS",
    ),
    "V2_STABILITY_DIAGNOSTICS_SCHEMA": (
        "objgauss.core.v2_stability_diagnostics",
        "V2_STABILITY_DIAGNOSTICS_SCHEMA",
    ),
    "V2_STABILITY_FAILURE_MODES": (
        "objgauss.core.v2_stability_diagnostics",
        "V2_STABILITY_FAILURE_MODES",
    ),
    "V2_STABILITY_GATE_HARD_CHECKS": (
        "objgauss.core.v2_stability_gate",
        "V2_STABILITY_GATE_HARD_CHECKS",
    ),
    "V2_STABILITY_GATE_SCHEMA": (
        "objgauss.core.v2_stability_gate",
        "V2_STABILITY_GATE_SCHEMA",
    ),
    "V2_STABILITY_GATE_SUITE_SCHEMA": (
        "objgauss.core.v2_stability_gate",
        "V2_STABILITY_GATE_SUITE_SCHEMA",
    ),
    "assignment_mvp_training_summary": (
        "objgauss.core.object_emergence_solver",
        "assignment_mvp_training_summary",
    ),
    "object_emergence_solver_checkpoint": (
        "objgauss.core.object_emergence_solver",
        "object_emergence_solver_checkpoint",
    ),
    "OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA": (
        "objgauss.core.object_state_benchmark",
        "OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA",
    ),
    "OBJECTSTATE_ACTION_SCHEMA": (
        "objgauss.core.objectstate_causal_gate",
        "OBJECTSTATE_ACTION_SCHEMA",
    ),
    "OBJECTSTATE_CAUSAL_ACTIONS": (
        "objgauss.core.objectstate_causal_gate",
        "OBJECTSTATE_CAUSAL_ACTIONS",
    ),
    "OBJECTSTATE_CAUSAL_GATE_SCHEMA": (
        "objgauss.core.objectstate_causal_gate",
        "OBJECTSTATE_CAUSAL_GATE_SCHEMA",
    ),
    "OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA": (
        "objgauss.core.objectstate_identity_encoder",
        "OBJECTSTATE_IDENTITY_ENCODER_STATE_SCHEMA",
    ),
    "OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA": (
        "objgauss.core.objectstate_identity_encoder",
        "OBJECTSTATE_IDENTITY_ENCODER_TRAINING_SCHEMA",
    ),
    "OBJECTSTATE_IDENTITY_DATASET_SCHEMA": (
        "objgauss.core.objectstate_identity_gate",
        "OBJECTSTATE_IDENTITY_DATASET_SCHEMA",
    ),
    "OBJECTSTATE_IDENTITY_GATE_SCHEMA": (
        "objgauss.core.objectstate_identity_gate",
        "OBJECTSTATE_IDENTITY_GATE_SCHEMA",
    ),
    "OBJECTSTATE_PREDICTIVE_GATE_SCHEMA": (
        "objgauss.core.objectstate_predictive_gate",
        "OBJECTSTATE_PREDICTIVE_GATE_SCHEMA",
    ),
    "TrainableKernelCamera": ("objgauss.core.trainable_kernel", "TrainableKernelCamera"),
    "TrainableKernelFrame": ("objgauss.core.trainable_kernel", "TrainableKernelFrame"),
    "TrainableKernelImageTarget": (
        "objgauss.core.trainable_kernel",
        "TrainableKernelImageTarget",
    ),
    "TrainableKernelLoss": ("objgauss.core.trainable_kernel", "TrainableKernelLoss"),
    "TrainableKernelResult": ("objgauss.core.trainable_kernel", "TrainableKernelResult"),
    "TrainableKernelSample": ("objgauss.core.trainable_kernel", "TrainableKernelSample"),
    "TRAINABLE_QUALITY_REPORT_SCHEMA": (
        "objgauss.core.trainable_quality",
        "TRAINABLE_QUALITY_REPORT_SCHEMA",
    ),
    "GsplatRendererAvailability": (
        "objgauss.core.gsplat_training_renderer",
        "GsplatRendererAvailability",
    ),
    "GsplatTrainingInput": (
        "objgauss.core.gsplat_training_renderer",
        "GsplatTrainingInput",
    ),
    "TrainingRendererFrameLoss": (
        "objgauss.core.training_renderer",
        "TrainingRendererFrameLoss",
    ),
    "TrainingRendererLossResult": (
        "objgauss.core.training_renderer",
        "TrainingRendererLossResult",
    ),
    "TRAINING_SCALE_PLAN_SCHEMA": (
        "objgauss.core.training_scale",
        "TRAINING_SCALE_PLAN_SCHEMA",
    ),
    "TENSORBOARD_SCALAR_EXPORT_SCHEMA": (
        "objgauss.core.training_tensorboard",
        "TENSORBOARD_SCALAR_EXPORT_SCHEMA",
    ),
    "RendererLossBoundaryReport": ("objgauss.core.renderer_loss", "RendererLossBoundaryReport"),
    "align_mask_manifest_slots": ("objgauss.core.semantics", "align_mask_manifest_slots"),
    "append_or_replace_property": ("objgauss.core.io", "append_or_replace_property"),
    "assign_object_ids": ("objgauss.core.objects", "assign_object_ids"),
    "attach_hard_labels": ("objgauss.core.object_field", "attach_hard_labels"),
    "attach_object_aware_lod_metadata": ("objgauss.core.lod", "attach_object_aware_lod_metadata"),
    "attach_quantization_metadata": ("objgauss.core.quantization", "attach_quantization_metadata"),
    "build_chunk_index": ("objgauss.core.chunk_index", "build_chunk_index"),
    "build_gsplat_training_input": (
        "objgauss.core.gsplat_training_renderer",
        "build_gsplat_training_input",
    ),
    "build_gsplat_training_input_from_object_state": (
        "objgauss.core.gsplat_training_renderer",
        "build_gsplat_training_input_from_object_state",
    ),
    "cluster_features": ("objgauss.core.clustering", "cluster_features"),
    "compare_baseline_candidates": ("objgauss.core.evaluation", "compare_baseline_candidates"),
    "filter_objects": ("objgauss.core.objects", "filter_objects"),
    "initialize_object_field": ("objgauss.core.object_field", "initialize_object_field"),
    "bind_image_targets_to_frames": (
        "objgauss.core.trainable_kernel",
        "bind_image_targets_to_frames",
    ),
    "bind_object_states_to_artifact": ("objgauss.core.object_state", "bind_object_states_to_artifact"),
    "dynamic_k_proposal_report": ("objgauss.core.object_state", "dynamic_k_proposal_report"),
    "dynamic_k_update_plan": ("objgauss.core.object_state", "dynamic_k_update_plan"),
    "decode_gaussian_from_object_state": (
        "objgauss.core.gaussian_decoder",
        "decode_gaussian_from_object_state",
    ),
    "diagnose_synthetic_stability_fixture": (
        "objgauss.core.v2_stability_diagnostics",
        "diagnose_synthetic_stability_fixture",
    ),
    "evidence_from_gaussian_cloud": (
        "objgauss.core.object_emergence_solver",
        "evidence_from_gaussian_cloud",
    ),
    "evaluate_training_renderer_loss": (
        "objgauss.core.training_renderer",
        "evaluate_training_renderer_loss",
    ),
    "evaluate_gsplat_training_renderer_loss": (
        "objgauss.core.gsplat_training_renderer",
        "evaluate_gsplat_training_renderer_loss",
    ),
    "gsplat_renderer_availability": (
        "objgauss.core.gsplat_training_renderer",
        "gsplat_renderer_availability",
    ),
    "initialize_object_emergence_solver": (
        "objgauss.core.object_emergence_solver",
        "initialize_object_emergence_solver",
    ),
    "initialize_object_state_gaussian_decoder": (
        "objgauss.core.gaussian_decoder_training",
        "initialize_object_state_gaussian_decoder",
    ),
    "make_trainable_kernel_mvp_fixture": (
        "objgauss.core.trainable_kernel",
        "make_trainable_kernel_mvp_fixture",
    ),
    "make_trainable_image_target": (
        "objgauss.core.trainable_kernel",
        "make_trainable_image_target",
    ),
    "make_object_identity_oracle": (
        "objgauss.core.v2_stability_foundation",
        "make_object_identity_oracle",
    ),
    "make_synthetic_world_state": (
        "objgauss.core.v2_stability_foundation",
        "make_synthetic_world_state",
    ),
    "make_synthetic_stability_scenario_fixture": (
        "objgauss.core.v2_stability_foundation",
        "make_synthetic_stability_scenario_fixture",
    ),
    "make_synthetic_stability_scenario_suite": (
        "objgauss.core.v2_stability_foundation",
        "make_synthetic_stability_scenario_suite",
    ),
    "match_object_states": ("objgauss.core.object_state", "match_object_states"),
    "object_emergence_metrics": ("objgauss.core.evaluation", "object_emergence_metrics"),
    "object_state_delivery_summary": ("objgauss.core.object_state", "object_state_delivery_summary"),
    "object_state_stability_benchmark": (
        "objgauss.core.object_state_benchmark",
        "object_state_stability_benchmark",
    ),
    "object_state_stability_report": ("objgauss.core.object_state", "object_state_stability_report"),
    "evaluate_solver_decoder_object_states": (
        "objgauss.core.object_state_eval",
        "evaluate_solver_decoder_object_states",
    ),
    "evaluate_assignment_stability": (
        "objgauss.core.assignment_stability",
        "evaluate_assignment_stability",
    ),
    "evaluate_synthetic_stability_gate": (
        "objgauss.core.v2_stability_gate",
        "evaluate_synthetic_stability_gate",
    ),
    "evaluate_synthetic_stability_suite_gate": (
        "objgauss.core.v2_stability_gate",
        "evaluate_synthetic_stability_suite_gate",
    ),
    "evaluate_objectstate_identity_gate": (
        "objgauss.core.objectstate_identity_gate",
        "evaluate_objectstate_identity_gate",
    ),
    "evaluate_objectstate_causal_gate": (
        "objgauss.core.objectstate_causal_gate",
        "evaluate_objectstate_causal_gate",
    ),
    "evaluate_objectstate_predictive_gate": (
        "objgauss.core.objectstate_predictive_gate",
        "evaluate_objectstate_predictive_gate",
    ),
    "evaluate_objectstate_reality_gate": (
        "objgauss.core.objectstate_reality_gate",
        "evaluate_objectstate_reality_gate",
    ),
    "default_objectstate_reality_public_artifacts": (
        "objgauss.core.objectstate_reality_public_rows",
        "default_objectstate_reality_public_artifacts",
    ),
    "evaluate_public_artifact_reality_gate": (
        "objgauss.core.objectstate_reality_public_rows",
        "evaluate_public_artifact_reality_gate",
    ),
    "evaluate_controlled_real_manifest_reality_gate": (
        "objgauss.core.objectstate_controlled_real_rows",
        "evaluate_controlled_real_manifest_reality_gate",
    ),
    "objectstate_controlled_capture_summary": (
        "objgauss.core.objectstate_controlled_capture",
        "objectstate_controlled_capture_summary",
    ),
    "objectstate_controlled_real_manifest_from_capture_manifest": (
        "objgauss.core.objectstate_controlled_capture",
        "objectstate_controlled_real_manifest_from_capture_manifest",
    ),
    "evaluate_objectstate_controlled_identity_predictions": (
        "objgauss.core.objectstate_controlled_identity_eval",
        "evaluate_objectstate_controlled_identity_predictions",
    ),
    "initialize_objectstate_identity_encoder_state": (
        "objgauss.core.objectstate_identity_encoder",
        "initialize_objectstate_identity_encoder_state",
    ),
    "objectstate_identity_encoder_features": (
        "objgauss.core.objectstate_identity_encoder",
        "objectstate_identity_encoder_features",
    ),
    "train_objectstate_identity_encoder": (
        "objgauss.core.objectstate_identity_encoder",
        "train_objectstate_identity_encoder",
    ),
    "validate_objectstate_identity_gate_summary": (
        "objgauss.core.objectstate_identity_gate",
        "validate_objectstate_identity_gate_summary",
    ),
    "validate_objectstate_predictive_gate_summary": (
        "objgauss.core.objectstate_predictive_gate",
        "validate_objectstate_predictive_gate_summary",
    ),
    "validate_objectstate_identity_encoder_training_summary": (
        "objgauss.core.objectstate_identity_encoder",
        "validate_objectstate_identity_encoder_training_summary",
    ),
    "validate_objectstate_causal_gate_summary": (
        "objgauss.core.objectstate_causal_gate",
        "validate_objectstate_causal_gate_summary",
    ),
    "objectstate_reality_blocked_rows_markdown": (
        "objgauss.core.objectstate_reality_gate",
        "objectstate_reality_blocked_rows_markdown",
    ),
    "objectstate_reality_public_rows_summary": (
        "objgauss.core.objectstate_reality_public_rows",
        "objectstate_reality_public_rows_summary",
    ),
    "objectstate_reality_rows_from_public_artifacts": (
        "objgauss.core.objectstate_reality_public_rows",
        "objectstate_reality_rows_from_public_artifacts",
    ),
    "objectstate_controlled_real_rows_summary": (
        "objgauss.core.objectstate_controlled_real_rows",
        "objectstate_controlled_real_rows_summary",
    ),
    "objectstate_reality_rows_from_controlled_real_manifest": (
        "objgauss.core.objectstate_controlled_real_rows",
        "objectstate_reality_rows_from_controlled_real_manifest",
    ),
    "read_objectstate_controlled_real_manifest": (
        "objgauss.core.objectstate_controlled_real_rows",
        "read_objectstate_controlled_real_manifest",
    ),
    "read_objectstate_controlled_capture_manifest": (
        "objgauss.core.objectstate_controlled_capture",
        "read_objectstate_controlled_capture_manifest",
    ),
    "read_objectstate_controlled_identity_predictions": (
        "objgauss.core.objectstate_controlled_identity_eval",
        "read_objectstate_controlled_identity_predictions",
    ),
    "read_trainable_kernel_identity_source": (
        "objgauss.core.objectstate_identity_prediction_adapter",
        "read_trainable_kernel_identity_source",
    ),
    "objectstate_identity_predictions_from_trainable_artifact": (
        "objgauss.core.objectstate_identity_prediction_adapter",
        "objectstate_identity_predictions_from_trainable_artifact",
    ),
    "objectstate_controlled_identity_handoff": (
        "objgauss.core.objectstate_controlled_identity_handoff",
        "objectstate_controlled_identity_handoff",
    ),
    "objectstate_controlled_identity_bundle_handoff": (
        "objgauss.core.objectstate_controlled_identity_bundle_handoff",
        "objectstate_controlled_identity_bundle_handoff",
    ),
    "objectstate_controlled_capture_file_audit": (
        "objgauss.core.objectstate_controlled_capture_files",
        "objectstate_controlled_capture_file_audit",
    ),
    "objectstate_controlled_capture_missing_files_markdown": (
        "objgauss.core.objectstate_controlled_capture_files",
        "objectstate_controlled_capture_missing_files_markdown",
    ),
    "validate_objectstate_controlled_capture_manifest": (
        "objgauss.core.objectstate_controlled_capture",
        "validate_objectstate_controlled_capture_manifest",
    ),
    "validate_objectstate_controlled_capture_summary": (
        "objgauss.core.objectstate_controlled_capture",
        "validate_objectstate_controlled_capture_summary",
    ),
    "validate_objectstate_controlled_capture_file_audit_summary": (
        "objgauss.core.objectstate_controlled_capture_files",
        "validate_objectstate_controlled_capture_file_audit_summary",
    ),
    "validate_objectstate_controlled_identity_eval_summary": (
        "objgauss.core.objectstate_controlled_identity_eval",
        "validate_objectstate_controlled_identity_eval_summary",
    ),
    "validate_objectstate_controlled_identity_handoff_summary": (
        "objgauss.core.objectstate_controlled_identity_handoff",
        "validate_objectstate_controlled_identity_handoff_summary",
    ),
    "validate_objectstate_controlled_identity_bundle_handoff_summary": (
        "objgauss.core.objectstate_controlled_identity_bundle_handoff",
        "validate_objectstate_controlled_identity_bundle_handoff_summary",
    ),
    "validate_objectstate_controlled_identity_predictions": (
        "objgauss.core.objectstate_controlled_identity_eval",
        "validate_objectstate_controlled_identity_predictions",
    ),
    "validate_objectstate_controlled_identity_thresholds": (
        "objgauss.core.objectstate_controlled_identity_eval",
        "validate_objectstate_controlled_identity_thresholds",
    ),
    "validate_objectstate_controlled_real_manifest": (
        "objgauss.core.objectstate_controlled_real_rows",
        "validate_objectstate_controlled_real_manifest",
    ),
    "validate_objectstate_controlled_real_rows_summary": (
        "objgauss.core.objectstate_controlled_real_rows",
        "validate_objectstate_controlled_real_rows_summary",
    ),
    "validate_objectstate_reality_gate_summary": (
        "objgauss.core.objectstate_reality_gate",
        "validate_objectstate_reality_gate_summary",
    ),
    "validate_objectstate_reality_public_rows_summary": (
        "objgauss.core.objectstate_reality_public_rows",
        "validate_objectstate_reality_public_rows_summary",
    ),
    "evaluate_assignment_solver_v2_stability": (
        "objgauss.core.assignment_solver_v2_eval",
        "evaluate_assignment_solver_v2_stability",
    ),
    "evaluate_assignment_v2_renderer_joint": (
        "objgauss.core.assignment_v2_renderer_validation",
        "evaluate_assignment_v2_renderer_joint",
    ),
    "evaluate_real_sample_v2_smoke": (
        "objgauss.core.real_sample_v2_smoke",
        "evaluate_real_sample_v2_smoke",
    ),
    "evaluate_real_sample_v2_diagnostics": (
        "objgauss.core.real_sample_v2_diagnostics",
        "evaluate_real_sample_v2_diagnostics",
    ),
    "evaluate_real_sample_v2_model_handoff": (
        "objgauss.core.real_sample_v2_model_handoff",
        "evaluate_real_sample_v2_model_handoff",
    ),
    "core_model_train_validate_report": (
        "objgauss.core.core_model_validation",
        "core_model_train_validate_report",
    ),
    "initialize_assignment_solver_v2": (
        "objgauss.core.assignment_solver_v2",
        "initialize_assignment_solver_v2",
    ),
    "expected_slots_for_synthetic_fixture": (
        "objgauss.core.v2_stability_diagnostics",
        "expected_slots_for_synthetic_fixture",
    ),
    "object_id_targets_from_cloud": (
        "objgauss.core.object_emergence_solver",
        "object_id_targets_from_cloud",
    ),
    "predict_object_emergence_assignment": (
        "objgauss.core.object_emergence_solver",
        "predict_object_emergence_assignment",
    ),
    "predict_assignment_solver_v2": (
        "objgauss.core.assignment_solver_v2",
        "predict_assignment_solver_v2",
    ),
    "project_object_emergence_prediction": (
        "objgauss.core.object_emergence_solver",
        "project_object_emergence_prediction",
    ),
    "project_object_states": ("objgauss.core.object_state", "project_object_states"),
    "project_object_states_from_field": ("objgauss.core.object_state", "project_object_states_from_field"),
    "project_points": ("objgauss.core.projection", "project_points"),
    "read_ply": ("objgauss.core.io", "read_ply"),
    "real_sample_v2_smoke_from_cloud": (
        "objgauss.core.real_sample_v2_smoke",
        "real_sample_v2_smoke_from_cloud",
    ),
    "real_sample_v2_diagnostics_from_cloud": (
        "objgauss.core.real_sample_v2_diagnostics",
        "real_sample_v2_diagnostics_from_cloud",
    ),
    "real_sample_v2_model_handoff_from_cloud": (
        "objgauss.core.real_sample_v2_model_handoff",
        "real_sample_v2_model_handoff_from_cloud",
    ),
    "real_sample_v2_viewer_preview_from_cloud": (
        "objgauss.core.real_sample_v2_viewer_preview",
        "real_sample_v2_viewer_preview_from_cloud",
    ),
    "real_sample_v2_viewer_preview_from_handoff": (
        "objgauss.core.real_sample_v2_viewer_preview",
        "real_sample_v2_viewer_preview_from_handoff",
    ),
    "real_sample_v2_full_cloud_purity_from_cloud": (
        "objgauss.core.real_sample_v2_full_cloud_purity",
        "real_sample_v2_full_cloud_purity_from_cloud",
    ),
    "real_sample_v2_segmentation_quality_from_cloud": (
        "objgauss.core.real_sample_v2_segmentation_quality",
        "real_sample_v2_segmentation_quality_from_cloud",
    ),
    "real_sample_v2_segmentation_quality_from_projected_cloud": (
        "objgauss.core.real_sample_v2_segmentation_quality",
        "real_sample_v2_segmentation_quality_from_projected_cloud",
    ),
    "real_sample_v2_segmentation_quality_from_purity_report": (
        "objgauss.core.real_sample_v2_segmentation_quality",
        "real_sample_v2_segmentation_quality_from_purity_report",
    ),
    "real_sample_v2_weak_boundary_opt_from_cloud": (
        "objgauss.core.real_sample_v2_weak_boundary_opt",
        "real_sample_v2_weak_boundary_opt_from_cloud",
    ),
    "real_sample_v2_promoted_weights_cross_sample_from_cloud": (
        "objgauss.core.real_sample_v2_promoted_weights_cross_sample",
        "real_sample_v2_promoted_weights_cross_sample_from_cloud",
    ),
    "real_sample_v2_sample_aware_weight_policy_from_cloud": (
        "objgauss.core.real_sample_v2_sample_aware_weight_policy",
        "real_sample_v2_sample_aware_weight_policy_from_cloud",
    ),
    "real_sample_v2_bounded_normalization_cross_sample_from_clouds": (
        "objgauss.core.real_sample_v2_bounded_normalization_cross_sample",
        "real_sample_v2_bounded_normalization_cross_sample_from_clouds",
    ),
    "render_real_sample_v2_model_handoff_html": (
        "objgauss.core.real_sample_v2_model_handoff",
        "render_real_sample_v2_model_handoff_html",
    ),
    "read_splat": ("objgauss.core.io", "read_splat"),
    "renderer_loss_boundary_report": (
        "objgauss.core.renderer_loss",
        "renderer_loss_boundary_report",
    ),
    "score_mask_manifest_with_clip": ("objgauss.core.semantics", "score_mask_manifest_with_clip"),
    "solver_decoder_training_scale_plan": (
        "objgauss.core.training_scale",
        "solver_decoder_training_scale_plan",
    ),
    "image_target_contract_summary": (
        "objgauss.core.trainable_kernel",
        "image_target_contract_summary",
    ),
    "train_kernel_mvp": ("objgauss.core.trainable_kernel", "train_kernel_mvp"),
    "train_object_emergence_solver": (
        "objgauss.core.object_emergence_solver",
        "train_object_emergence_solver",
    ),
    "train_assignment_solver_v2": (
        "objgauss.core.assignment_solver_v2",
        "train_assignment_solver_v2",
    ),
    "train_kernel_mvp_from_cloud": (
        "objgauss.core.trainable_kernel",
        "train_kernel_mvp_from_cloud",
    ),
    "train_object_state_gaussian_decoder": (
        "objgauss.core.gaussian_decoder_training",
        "train_object_state_gaussian_decoder",
    ),
    "train_solver_decoder_joint": (
        "objgauss.core.solver_decoder_training",
        "train_solver_decoder_joint",
    ),
    "trainable_kernel_model_artifact": (
        "objgauss.core.trainable_artifact",
        "trainable_kernel_model_artifact",
    ),
    "trainable_quality_report": (
        "objgauss.core.trainable_quality",
        "trainable_quality_report",
    ),
    "trainable_kernel_sample_from_cloud": (
        "objgauss.core.trainable_kernel",
        "trainable_kernel_sample_from_cloud",
    ),
    "validate_mask_manifest": ("objgauss.core.masks", "validate_mask_manifest"),
    "validate_assignment_matrix": ("objgauss.core.object_state", "validate_assignment_matrix"),
    "validate_object_emergence_evidence": (
        "objgauss.core.object_emergence_solver",
        "validate_object_emergence_evidence",
    ),
    "validate_object_emergence_solver_state": (
        "objgauss.core.object_emergence_solver",
        "validate_object_emergence_solver_state",
    ),
    "object_emergence_solver_state_from_dict": (
        "objgauss.core.object_emergence_solver",
        "object_emergence_solver_state_from_dict",
    ),
    "assignment_solver_v2_state_from_dict": (
        "objgauss.core.assignment_solver_v2",
        "assignment_solver_v2_state_from_dict",
    ),
    "assignment_solver_v2_checkpoint": (
        "objgauss.core.assignment_solver_v2_eval",
        "assignment_solver_v2_checkpoint",
    ),
    "assignment_solver_v2_state_from_checkpoint": (
        "objgauss.core.assignment_solver_v2_eval",
        "assignment_solver_v2_state_from_checkpoint",
    ),
    "assignment_balance_loss_and_gradient": (
        "objgauss.core.assignment_losses",
        "assignment_balance_loss_and_gradient",
    ),
    "assignment_evidence_from_object_emergence": (
        "objgauss.core.assignment_evidence",
        "assignment_evidence_from_object_emergence",
    ),
    "assignment_evidence_from_trainable_frame": (
        "objgauss.core.assignment_evidence",
        "assignment_evidence_from_trainable_frame",
    ),
    "assignment_evidence_sequence_from_trainable_frames": (
        "objgauss.core.assignment_evidence",
        "assignment_evidence_sequence_from_trainable_frames",
    ),
    "assignment_cluster_loss_and_gradient": (
        "objgauss.core.assignment_losses",
        "assignment_cluster_loss_and_gradient",
    ),
    "assignment_entropy_loss_and_gradient": (
        "objgauss.core.assignment_losses",
        "assignment_entropy_loss_and_gradient",
    ),
    "assignment_loss_v2_breakdown": (
        "objgauss.core.assignment_losses",
        "assignment_loss_v2_breakdown",
    ),
    "supervised_assignment_loss_and_gradient": (
        "objgauss.core.assignment_losses",
        "supervised_assignment_loss_and_gradient",
    ),
    "object_state_gaussian_decoder_state_from_dict": (
        "objgauss.core.gaussian_decoder_training",
        "object_state_gaussian_decoder_state_from_dict",
    ),
    "object_scale_multipliers_from_log_offsets": (
        "objgauss.core.gaussian_decoder",
        "object_scale_multipliers_from_log_offsets",
    ),
    "observe_synthetic_world": (
        "objgauss.core.v2_stability_foundation",
        "observe_synthetic_world",
    ),
    "solver_decoder_joint_states_from_dict": (
        "objgauss.core.solver_decoder_training",
        "solver_decoder_joint_states_from_dict",
    ),
    "validate_object_emergence_solver_checkpoint": (
        "objgauss.core.object_emergence_solver",
        "validate_object_emergence_solver_checkpoint",
    ),
    "validate_assignment_loss_v2_summary": (
        "objgauss.core.assignment_losses",
        "validate_assignment_loss_v2_summary",
    ),
    "validate_assignment_evidence_batch": (
        "objgauss.core.assignment_evidence",
        "validate_assignment_evidence_batch",
    ),
    "validate_assignment_evidence_summary": (
        "objgauss.core.assignment_evidence",
        "validate_assignment_evidence_summary",
    ),
    "validate_assignment_stability_eval": (
        "objgauss.core.assignment_stability",
        "validate_assignment_stability_eval",
    ),
    "validate_assignment_solver_v2_config": (
        "objgauss.core.assignment_solver_v2",
        "validate_assignment_solver_v2_config",
    ),
    "validate_assignment_solver_v2_state": (
        "objgauss.core.assignment_solver_v2",
        "validate_assignment_solver_v2_state",
    ),
    "validate_assignment_solver_v2_training_summary": (
        "objgauss.core.assignment_solver_v2",
        "validate_assignment_solver_v2_training_summary",
    ),
    "validate_assignment_solver_v2_checkpoint": (
        "objgauss.core.assignment_solver_v2_eval",
        "validate_assignment_solver_v2_checkpoint",
    ),
    "validate_assignment_solver_v2_stability_eval_summary": (
        "objgauss.core.assignment_solver_v2_eval",
        "validate_assignment_solver_v2_stability_eval_summary",
    ),
    "validate_assignment_v2_renderer_joint_summary": (
        "objgauss.core.assignment_v2_renderer_validation",
        "validate_assignment_v2_renderer_joint_summary",
    ),
    "validate_core_model_train_validate_summary": (
        "objgauss.core.core_model_validation",
        "validate_core_model_train_validate_summary",
    ),
    "validate_real_sample_v2_smoke_summary": (
        "objgauss.core.real_sample_v2_smoke",
        "validate_real_sample_v2_smoke_summary",
    ),
    "validate_real_sample_v2_diagnostics_summary": (
        "objgauss.core.real_sample_v2_diagnostics",
        "validate_real_sample_v2_diagnostics_summary",
    ),
    "validate_real_sample_v2_effect_preview": (
        "objgauss.core.real_sample_v2_model_handoff",
        "validate_real_sample_v2_effect_preview",
    ),
    "validate_real_sample_v2_model_handoff_summary": (
        "objgauss.core.real_sample_v2_model_handoff",
        "validate_real_sample_v2_model_handoff_summary",
    ),
    "validate_real_sample_v2_viewer_preview_summary": (
        "objgauss.core.real_sample_v2_viewer_preview",
        "validate_real_sample_v2_viewer_preview_summary",
    ),
    "validate_real_sample_v2_full_cloud_purity_summary": (
        "objgauss.core.real_sample_v2_full_cloud_purity",
        "validate_real_sample_v2_full_cloud_purity_summary",
    ),
    "validate_real_sample_v2_segmentation_quality_summary": (
        "objgauss.core.real_sample_v2_segmentation_quality",
        "validate_real_sample_v2_segmentation_quality_summary",
    ),
    "validate_real_sample_v2_weak_boundary_opt_summary": (
        "objgauss.core.real_sample_v2_weak_boundary_opt",
        "validate_real_sample_v2_weak_boundary_opt_summary",
    ),
    "validate_real_sample_v2_promoted_weights_cross_sample_summary": (
        "objgauss.core.real_sample_v2_promoted_weights_cross_sample",
        "validate_real_sample_v2_promoted_weights_cross_sample_summary",
    ),
    "validate_real_sample_v2_sample_aware_weight_policy_summary": (
        "objgauss.core.real_sample_v2_sample_aware_weight_policy",
        "validate_real_sample_v2_sample_aware_weight_policy_summary",
    ),
    "validate_real_sample_v2_bounded_normalization_cross_sample_summary": (
        "objgauss.core.real_sample_v2_bounded_normalization_cross_sample",
        "validate_real_sample_v2_bounded_normalization_cross_sample_summary",
    ),
    "validate_solver_decoder_joint_checkpoint": (
        "objgauss.core.solver_decoder_training",
        "validate_solver_decoder_joint_checkpoint",
    ),
    "validate_object_state_gaussian_decoder_state": (
        "objgauss.core.gaussian_decoder_training",
        "validate_object_state_gaussian_decoder_state",
    ),
    "validate_object_state_stability_benchmark": (
        "objgauss.core.object_state_benchmark",
        "validate_object_state_stability_benchmark",
    ),
    "validate_object_identity_oracle": (
        "objgauss.core.v2_stability_foundation",
        "validate_object_identity_oracle",
    ),
    "validate_observation_model_config": (
        "objgauss.core.v2_stability_foundation",
        "validate_observation_model_config",
    ),
    "validate_objectstate_checkpoint_eval": (
        "objgauss.core.object_state_eval",
        "validate_objectstate_checkpoint_eval",
    ),
    "validate_image_target_contract_summary": (
        "objgauss.core.trainable_kernel",
        "validate_image_target_contract_summary",
    ),
    "validate_trainable_image_target": (
        "objgauss.core.trainable_kernel",
        "validate_trainable_image_target",
    ),
    "validate_training_renderer_summary": (
        "objgauss.core.training_renderer",
        "validate_training_renderer_summary",
    ),
    "validate_solver_decoder_training_scale_plan": (
        "objgauss.core.training_scale",
        "validate_solver_decoder_training_scale_plan",
    ),
    "validate_trainable_kernel_model_artifact": (
        "objgauss.core.trainable_artifact",
        "validate_trainable_kernel_model_artifact",
    ),
    "validate_trainable_quality_report": (
        "objgauss.core.trainable_quality",
        "validate_trainable_quality_report",
    ),
    "validate_renderer_loss_boundary_summary": (
        "objgauss.core.renderer_loss",
        "validate_renderer_loss_boundary_summary",
    ),
    "validate_synthetic_observation_frame": (
        "objgauss.core.v2_stability_foundation",
        "validate_synthetic_observation_frame",
    ),
    "validate_synthetic_stability_scenario_fixture": (
        "objgauss.core.v2_stability_foundation",
        "validate_synthetic_stability_scenario_fixture",
    ),
    "validate_synthetic_stability_diagnostics_summary": (
        "objgauss.core.v2_stability_diagnostics",
        "validate_synthetic_stability_diagnostics_summary",
    ),
    "validate_synthetic_stability_gate_summary": (
        "objgauss.core.v2_stability_gate",
        "validate_synthetic_stability_gate_summary",
    ),
    "validate_synthetic_stability_suite_gate_summary": (
        "objgauss.core.v2_stability_gate",
        "validate_synthetic_stability_suite_gate_summary",
    ),
    "validate_synthetic_world_state": (
        "objgauss.core.v2_stability_foundation",
        "validate_synthetic_world_state",
    ),
    "vote_masks_to_gaussians": ("objgauss.core.projection", "vote_masks_to_gaussians"),
    "write_ogc_payload": ("objgauss.core.ogc_payload", "write_ogc_payload"),
    "write_ply": ("objgauss.core.io", "write_ply"),
    "write_solver_decoder_tensorboard_events": (
        "objgauss.core.training_tensorboard",
        "write_solver_decoder_tensorboard_events",
    ),
    "write_object_state_stability_benchmark": (
        "objgauss.core.object_state_benchmark",
        "write_object_state_stability_benchmark",
    ),
    "write_quantized_ogc_payload": ("objgauss.core.quantization", "write_quantized_ogc_payload"),
    "write_trainable_kernel_model_artifact": (
        "objgauss.core.trainable_artifact",
        "write_trainable_kernel_model_artifact",
    ),
    "write_trainable_quality_report": (
        "objgauss.core.trainable_quality",
        "write_trainable_quality_report",
    ),
    "write_splat": ("objgauss.core.io", "write_splat"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as error:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from error
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value
