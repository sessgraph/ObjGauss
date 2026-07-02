from __future__ import annotations

import numpy as np

from objgauss.core import (
    GaussianCloud,
    DynamicKProposalReport,
    ObjectState,
    ObjectStabilityReport,
    ObjectTemporalMatchReport,
    RendererLossBoundaryReport,
    TrainableKernelResult,
    TrainableKernelSample,
    append_or_replace_property,
    attach_object_aware_lod_metadata,
    attach_quantization_metadata,
    assign_object_ids,
    bind_object_states_to_artifact,
    build_chunk_index,
    cluster_features,
    dynamic_k_proposal_report,
    initialize_object_field,
    make_trainable_kernel_mvp_fixture,
    match_object_states,
    object_state_delivery_summary,
    object_state_stability_report,
    project_object_states_from_field,
    read_ply,
    renderer_loss_boundary_report,
    train_kernel_mvp,
    train_kernel_mvp_from_cloud,
    trainable_kernel_sample_from_cloud,
    validate_renderer_loss_boundary_summary,
    write_ogc_payload,
    write_ply,
    write_quantized_ogc_payload,
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

    initialized = initialize_object_field(_tiny_cloud(), slots=2, seed=3, max_iter=10)
    assert initialized.field.gaussian_count == 4
    assert initialized.field.slots == 2


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
