from __future__ import annotations

import numpy as np

from objgauss.core import (
    GaussianCloud,
    DynamicKProposalReport,
    DynamicKUpdatePlan,
    GsplatRendererAvailability,
    GsplatTrainingInput,
    ObjectState,
    AssignmentEvidenceBatch,
    ObjectStateGaussianDecode,
    ObjectStateGaussianDecoderTrainingResult,
    ObjectStateGaussianDecoderState,
    OBJECTSTATE_CHECKPOINT_EVAL_SCHEMA,
    SolverDecoderJointTrainingResult,
    ObjectEmergenceAssignmentPrediction,
    ObjectEmergenceEvidence,
    ObjectEmergenceSolverTrainingResult,
    ObjectEmergenceSolverState,
    ObjectStabilityReport,
    ObjectTemporalMatchReport,
    RendererLossBoundaryReport,
    TrainableKernelCamera,
    TrainableKernelImageTarget,
    TrainableKernelResult,
    TrainableKernelSample,
    TENSORBOARD_SCALAR_EXPORT_SCHEMA,
    TRAINING_SCALE_PLAN_SCHEMA,
    TrainingRendererLossResult,
    append_or_replace_property,
    assignment_balance_loss_and_gradient,
    assignment_cluster_loss_and_gradient,
    assignment_evidence_from_trainable_frame,
    assignment_evidence_sequence_from_trainable_frames,
    assignment_entropy_loss_and_gradient,
    assignment_loss_v2_breakdown,
    attach_object_aware_lod_metadata,
    attach_quantization_metadata,
    assign_object_ids,
    bind_image_targets_to_frames,
    bind_object_states_to_artifact,
    build_chunk_index,
    build_gsplat_training_input,
    build_gsplat_training_input_from_object_state,
    cluster_features,
    decode_gaussian_from_object_state,
    dynamic_k_proposal_report,
    dynamic_k_update_plan,
    evaluate_solver_decoder_object_states,
    evaluate_training_renderer_loss,
    evaluate_gsplat_training_renderer_loss,
    gsplat_renderer_availability,
    evidence_from_gaussian_cloud,
    initialize_object_field,
    initialize_object_emergence_solver,
    initialize_object_state_gaussian_decoder,
    image_target_contract_summary,
    make_trainable_image_target,
    make_trainable_kernel_mvp_fixture,
    match_object_states,
    object_state_delivery_summary,
    object_state_stability_report,
    object_id_targets_from_cloud,
    object_emergence_solver_checkpoint,
    object_emergence_solver_state_from_dict,
    object_scale_multipliers_from_log_offsets,
    object_state_gaussian_decoder_state_from_dict,
    predict_object_emergence_assignment,
    project_object_emergence_prediction,
    project_object_states,
    project_object_states_from_field,
    read_ply,
    renderer_loss_boundary_report,
    solver_decoder_training_scale_plan,
    train_kernel_mvp,
    train_object_emergence_solver,
    train_object_state_gaussian_decoder,
    train_solver_decoder_joint,
    train_kernel_mvp_from_cloud,
    trainable_kernel_model_artifact,
    trainable_kernel_sample_from_cloud,
    validate_image_target_contract_summary,
    validate_assignment_loss_v2_summary,
    validate_assignment_evidence_summary,
    validate_object_emergence_evidence,
    validate_object_emergence_solver_checkpoint,
    validate_object_state_gaussian_decoder_state,
    validate_objectstate_checkpoint_eval,
    validate_solver_decoder_joint_checkpoint,
    validate_solver_decoder_training_scale_plan,
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
    assert assignment_evidence_from_trainable_frame is not None
    assert assignment_evidence_sequence_from_trainable_frames is not None
    assert assignment_cluster_loss_and_gradient is not None
    assert assignment_entropy_loss_and_gradient is not None
    assert assignment_balance_loss_and_gradient is not None
    assert supervised_assignment_loss_and_gradient is not None
    assignment = np.full((6, 2), 0.5, dtype=np.float32)
    loss_summary = assignment_loss_v2_breakdown([assignment], entropy_weight=0.1).as_dict()
    assert validate_assignment_loss_v2_summary(loss_summary) is True
    evidence_summary = assignment_evidence_from_trainable_frame(bound_frames[0]).as_dict()
    assert validate_assignment_evidence_summary(evidence_summary) is True
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
