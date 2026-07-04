"""Stable core algorithm entry points for ObjGauss.

The namespace is intentionally lazy. Compatibility wrappers such as
`objgauss.gaussians` import specific core submodules during package
initialization, so eagerly importing every core domain here can create circular
imports while the migration is in progress.
"""

from importlib import import_module
from typing import Any

_EXPORTS = {
    "GaussianCloud": ("objgauss.core.gaussian", "GaussianCloud"),
    "ObjectField": ("objgauss.core.object_field", "ObjectField"),
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
    "solver_decoder_joint_checkpoint": (
        "objgauss.core.solver_decoder_training",
        "solver_decoder_joint_checkpoint",
    ),
    "ObjectState": ("objgauss.core.object_state", "ObjectState"),
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
    "object_emergence_solver_checkpoint": (
        "objgauss.core.object_emergence_solver",
        "object_emergence_solver_checkpoint",
    ),
    "OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA": (
        "objgauss.core.object_state_benchmark",
        "OBJECT_STATE_STABILITY_BENCHMARK_SCHEMA",
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
    "object_id_targets_from_cloud": (
        "objgauss.core.object_emergence_solver",
        "object_id_targets_from_cloud",
    ),
    "predict_object_emergence_assignment": (
        "objgauss.core.object_emergence_solver",
        "predict_object_emergence_assignment",
    ),
    "project_object_emergence_prediction": (
        "objgauss.core.object_emergence_solver",
        "project_object_emergence_prediction",
    ),
    "project_object_states": ("objgauss.core.object_state", "project_object_states"),
    "project_object_states_from_field": ("objgauss.core.object_state", "project_object_states_from_field"),
    "project_points": ("objgauss.core.projection", "project_points"),
    "read_ply": ("objgauss.core.io", "read_ply"),
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
    "object_state_gaussian_decoder_state_from_dict": (
        "objgauss.core.gaussian_decoder_training",
        "object_state_gaussian_decoder_state_from_dict",
    ),
    "solver_decoder_joint_states_from_dict": (
        "objgauss.core.solver_decoder_training",
        "solver_decoder_joint_states_from_dict",
    ),
    "validate_object_emergence_solver_checkpoint": (
        "objgauss.core.object_emergence_solver",
        "validate_object_emergence_solver_checkpoint",
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
