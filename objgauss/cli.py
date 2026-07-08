from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np

from objgauss.assets import list_assets, pull_asset
from objgauss.baseline_comparison import (
    compare_baseline_candidates,
    write_comparison_markdown,
)
from objgauss.clustering import cluster_features, summarize_labels
from objgauss.clip_scoring import (
    CLIP_LABEL_PRESETS,
    read_clip_labels,
    score_mask_manifest_with_clip,
)
from objgauss.demo import build_v1_closure_demo, verify_v1_closure_demo
from objgauss.emergence_benchmark import run_emergence_benchmark
from objgauss.emergence import (
    object_emergence_curve,
    object_emergence_metrics,
    write_emergence_curve_csv,
)
from objgauss.emergence_report import (
    load_emergence_curve,
    write_emergence_curve_report,
)
from objgauss.features import extract_features
from objgauss.goal_audit import audit_v1_goal
from objgauss.mask_voting import (
    depth_visibility_diagnostic,
    mask_vote_quality_audit,
    train_object_field_from_votes,
    training_summary,
    vote_masks_to_gaussians,
)
from objgauss.lego_verify import verify_lego_alpha_closure_demo
from objgauss.masks import (
    build_nerf_alpha_fgbg_mask_manifest,
    build_nerf_alpha_mask_manifest,
    build_nerf_rgba_color_mask_manifest,
    build_nerf_sam_mask_manifest,
    split_mask_manifest,
    validate_mask_manifest,
)
from objgauss.nerf_proxy import build_lego_alpha_closure_demo
from objgauss.object_field import (
    attach_hard_labels,
    cloud_positions_for_metrics,
    initialize_object_field,
    inspect_nerf_dataset,
    load_object_field,
    object_field_metrics,
    save_object_field,
    write_json,
)
from objgauss.ply import read_ply, write_ply
from objgauss.render_probe import load_render_probe_frames
from objgauss.segment import (
    apply_object_colors,
    assign_object_ids,
    filter_objects,
    parse_object_ids,
)
from objgauss.semantic_demo import (
    build_plush_semantic_closure_demo,
    verify_plush_semantic_closure_demo,
)
from objgauss.sample_bundle import write_sample_bundle
from objgauss.semantic_slots import align_mask_manifest_slots
from objgauss.splat import read_splat
from objgauss.training import register_training_output
from objgauss.core.trainable_kernel import (
    make_trainable_kernel_mvp_fixture,
    train_kernel_mvp,
    train_kernel_mvp_from_cloud,
    trainable_kernel_sample_from_cloud,
)
from objgauss.core.object_emergence_solver import (
    ObjectEmergenceEvidence,
    assignment_mvp_training_summary,
    evidence_from_gaussian_cloud,
    object_emergence_solver_checkpoint,
    object_emergence_solver_state_from_dict,
    object_id_targets_from_cloud,
    predict_object_emergence_assignment,
    train_object_emergence_solver,
    validate_object_emergence_solver_checkpoint,
)
from objgauss.core.assignment_evidence import (
    assignment_evidence_from_object_emergence,
    assignment_evidence_sequence_from_trainable_frames,
)
from objgauss.core.assignment_stability import evaluate_assignment_stability
from objgauss.core.gaussian_decoder_training import train_object_state_gaussian_decoder
from objgauss.core.solver_decoder_training import (
    SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA,
    solver_decoder_joint_checkpoint,
    solver_decoder_joint_states_from_dict,
    train_solver_decoder_joint,
    validate_solver_decoder_joint_checkpoint,
)
from objgauss.core.renderer_loss import renderer_loss_boundary_report
from objgauss.core.training_scale import solver_decoder_training_scale_plan
from objgauss.core.training_tensorboard import write_solver_decoder_tensorboard_events
from objgauss.core.object_state_eval import evaluate_solver_decoder_object_states
from objgauss.core.real_sample_v2_model_handoff import (
    real_sample_v2_model_handoff_from_cloud,
    render_real_sample_v2_model_handoff_html,
)
from objgauss.core.real_sample_v2_viewer_preview import (
    REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT,
    REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT,
    real_sample_v2_viewer_preview_from_cloud,
)
from objgauss.core.real_sample_v2_full_cloud_purity import (
    real_sample_v2_full_cloud_purity_from_cloud,
)
from objgauss.core.real_sample_v2_segmentation_quality import (
    real_sample_v2_segmentation_quality_from_cloud,
)
from objgauss.core.real_sample_v2_weak_boundary_opt import (
    real_sample_v2_weak_boundary_opt_from_cloud,
)
from objgauss.core.real_sample_v2_promoted_weights_cross_sample import (
    real_sample_v2_promoted_weights_cross_sample_from_cloud,
)
from objgauss.core.real_sample_v2_sample_aware_weight_policy import (
    real_sample_v2_sample_aware_weight_policy_from_cloud,
)
from objgauss.core.real_sample_v2_bounded_normalization_cross_sample import (
    RealSampleV2BoundedNormalizationCrossSampleInput,
    real_sample_v2_bounded_normalization_cross_sample_from_clouds,
)
from objgauss.core.training_renderer import evaluate_training_renderer_loss
from objgauss.core.gsplat_training_renderer import evaluate_gsplat_training_renderer_loss
from objgauss.core.trainable_artifact import write_trainable_kernel_model_artifact
from objgauss.core.trainable_quality import write_trainable_quality_report
from objgauss.core.object_state_benchmark import write_object_state_stability_benchmark
from objgauss.core.objectstate_controlled_capture import (
    objectstate_controlled_capture_summary,
    read_objectstate_controlled_capture_manifest,
)
from objgauss.core.objectstate_controlled_capture_template import (
    write_objectstate_controlled_capture_bundle_template,
)
from objgauss.core.objectstate_controlled_capture_bundle_readiness import (
    objectstate_controlled_capture_bundle_readiness,
)
from objgauss.core.objectstate_controlled_capture_environment import (
    objectstate_controlled_capture_environment,
)
from objgauss.core.objectstate_controlled_capture_import import (
    objectstate_controlled_capture_bundle_acceptance_summary,
    objectstate_controlled_capture_import_summary,
)
from objgauss.core.objectstate_controlled_capture_files import (
    objectstate_controlled_capture_file_audit,
)
from objgauss.core.objectstate_controlled_identity_eval import (
    ObjectStateControlledIdentityThresholds,
    evaluate_objectstate_controlled_identity_predictions,
    read_objectstate_controlled_identity_predictions,
)
from objgauss.core.objectstate_controlled_prediction_eval import (
    ObjectStateControlledPredictionThresholds,
    evaluate_objectstate_controlled_prediction_candidates,
    read_objectstate_controlled_prediction_candidates,
)
from objgauss.core.objectstate_controlled_intervention_eval import (
    ObjectStateControlledInterventionThresholds,
    evaluate_objectstate_controlled_intervention_candidates,
    read_objectstate_controlled_intervention_candidates,
)
from objgauss.core.objectstate_controlled_identity_handoff import (
    objectstate_controlled_identity_handoff,
)
from objgauss.core.objectstate_controlled_identity_bundle_handoff import (
    objectstate_controlled_identity_bundle_handoff,
)
from objgauss.core.objectstate_controlled_reality_bundle_handoff import (
    objectstate_controlled_reality_bundle_handoff,
)
from objgauss.core.objectstate_controlled_reality_bundle_readiness import (
    objectstate_controlled_reality_bundle_readiness,
)
from objgauss.core.objectstate_controlled_reality_candidate_template import (
    finalize_objectstate_controlled_prediction_candidate_template,
    finalize_objectstate_controlled_reality_candidate_templates,
    write_objectstate_controlled_reality_candidate_templates_from_manifest,
    write_objectstate_controlled_reality_candidate_templates,
)
from objgauss.core.objectstate_controlled_prediction_baseline import (
    write_objectstate_controlled_prediction_baseline_candidates,
)
from objgauss.core.objectstate_controlled_reality_evidence_package import (
    objectstate_controlled_reality_evidence_package,
)
from objgauss.core.objectstate_controlled_prediction_evidence_package import (
    objectstate_controlled_prediction_evidence_package,
)
from objgauss.core.objectstate_controlled_identity_evidence_package import (
    objectstate_controlled_identity_evidence_package,
)
from objgauss.core.objectstate_phase1_evidence_ledger import (
    objectstate_phase1_evidence_ledger,
)
from objgauss.core.objectstate_public_dataset_candidates import (
    objectstate_public_dataset_candidates_audit,
    objectstate_public_dataset_candidates_markdown,
)
from objgauss.core.objectstate_bop_phase1_subset_selector import (
    objectstate_bop_phase1_subset_selector,
)
from objgauss.core.objectstate_bop_gaussian_evidence_preflight import (
    objectstate_bop_gaussian_evidence_preflight,
)
from objgauss.core.objectstate_bop_rgbd_gaussian_export import (
    objectstate_bop_rgbd_gaussian_export,
)
from objgauss.core.objectstate_bop_candidate_artifact_template import (
    finalize_objectstate_bop_candidate_artifact_template,
    write_objectstate_bop_candidate_artifact_template,
)
from objgauss.core.objectstate_bop_capture_adapter import (
    objectstate_bop_capture_acceptance_summary,
    objectstate_bop_capture_adapter_summary,
    objectstate_bop_capture_condition_sidecar_summary,
)
from objgauss.core.objectstate_bop_prediction_baseline_handoff import (
    objectstate_bop_prediction_baseline_handoff,
)
from objgauss.core.objectstate_bop_identity_handoff import (
    objectstate_bop_identity_handoff,
)
from objgauss.core.objectstate_bop_local_row_handoff import (
    objectstate_bop_local_row_handoff,
)
from objgauss.core.objectstate_bop_cross_sample_ledger import (
    objectstate_bop_cross_sample_ledger,
)
from objgauss.core.objectstate_bop_local_row_batch_handoff import (
    objectstate_bop_local_row_batch_handoff,
)
from objgauss.core.objectstate_bop_local_row_batch_spec import (
    objectstate_bop_local_row_batch_spec_authoring,
)
from objgauss.core.objectstate_bop_local_row_batch_readiness import (
    objectstate_bop_local_row_batch_readiness,
)
from objgauss.core.objectstate_bop_phase1_route_audit import (
    objectstate_bop_phase1_route_audit,
)
from objgauss.core.objectstate_bop_identity_route_audit import (
    objectstate_bop_identity_route_audit,
)
from objgauss.core.objectstate_bop_phase1_local_row_readiness import (
    objectstate_bop_phase1_local_row_readiness,
)
from objgauss.core.objectstate_identity_prediction_adapter import (
    objectstate_identity_predictions_from_trainable_artifact,
    read_trainable_kernel_identity_source,
)
from objgauss.core.objectstate_controlled_real_rows import (
    objectstate_controlled_real_rows_summary,
    read_objectstate_controlled_real_manifest,
)
from objgauss.core.objectstate_reality_gate import ObjectStateRealityGateThresholds
from objgauss.model_manifest import (
    manifest_from_trainable_kernel_model_artifact,
    write_model_artifact_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        args.handler(args)
    except Exception as exc:
        parser.exit(2, f"objgauss: error: {exc}\n")
    return 0


def _cluster(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    features = extract_features(
        cloud,
        spatial_weight=args.spatial_weight,
        color_weight=args.color_weight,
        opacity_weight=args.opacity_weight,
        normalize=not args.no_normalize,
    )
    result = cluster_features(
        features,
        clusters=args.clusters,
        seed=args.seed,
        max_iter=args.max_iter,
    )
    cloud = assign_object_ids(cloud, result.labels)
    if args.colorize:
        cloud = apply_object_colors(cloud, rewrite_sh=args.rewrite_sh)
    write_ply(args.output, cloud, fmt=_output_format(args))

    print(f"clustered {cloud.count} gaussians into {args.clusters} objects")
    print(f"backend={result.backend} inertia={result.inertia:.4f}")
    _print_summary(result.labels)


def _colorize(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    cloud = apply_object_colors(
        cloud,
        object_id_field=args.object_id_field,
        rewrite_sh=args.rewrite_sh,
    )
    write_ply(args.output, cloud, fmt=_output_format(args))
    print(f"wrote object-colored PLY with {cloud.count} gaussians")


def _filter(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    ids = parse_object_ids(args.ids)
    before = cloud.count
    cloud = filter_objects(
        cloud,
        ids,
        mode=args.mode,
        object_id_field=args.object_id_field,
    )
    write_ply(args.output, cloud, fmt=_output_format(args))
    print(f"{args.mode} ids={sorted(ids)}: {before} -> {cloud.count} gaussians")


def _stats(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    print(f"gaussians={cloud.count}")
    print(f"fields={','.join(cloud.fields)}")
    if args.object_id_field in cloud.fields:
        _print_summary(cloud.vertices[args.object_id_field])
    else:
        print(f"no {args.object_id_field!r} property found")


def _convert_splat(args: argparse.Namespace) -> None:
    cloud = read_splat(args.input)
    write_ply(args.output, cloud, fmt="ascii" if args.ascii else "binary_little_endian")
    print(f"converted {cloud.count} splats to PLY")


def _assets_list(args: argparse.Namespace) -> None:
    assets = list_assets()
    if args.pullable:
        assets = tuple(asset for asset in assets if asset.pull_pipeline)
    for asset in assets:
        mode = "pull" if asset.pull_pipeline else "manual"
        local = asset.local_path or "-"
        use_cases = ",".join(asset.use_cases) if asset.use_cases else "-"
        print(
            f"{asset.id}\t{asset.name}\t{asset.category}\t"
            f"{asset.status}\t{asset.pipeline_stage}\t{use_cases}\t{mode}\t{local}"
        )


def _assets_pull(args: argparse.Namespace) -> None:
    result = pull_asset(
        args.asset_id,
        raw_dir=args.raw_dir,
        converted_dir=args.converted_dir,
        public_dir=args.public_dir,
        training_dir=args.training_dir,
        clusters=args.clusters,
        force=args.force,
    )
    print(f"asset={result.asset.id} name={result.asset.name}")
    print(f"raw={result.raw_path}")
    if result.converted_path:
        print(f"converted={result.converted_path}")
    if result.output_path:
        print(f"output={result.output_path}")
    if result.raw_public_path:
        print(f"viewer_splat={result.raw_public_path}")
    if result.training_path:
        print(f"training={result.training_path}")
    if result.manifest_path:
        print(f"manifest={result.manifest_path}")
    if result.downloaded_files:
        print(f"files={len(result.downloaded_files)}")
    if result.gaussian_count is not None:
        print(f"gaussians={result.gaussian_count}")
    for label, count in result.object_counts:
        print(f"object_id={label} count={count}")


def _object_field_init(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    result = initialize_object_field(
        cloud,
        slots=args.slots,
        seed=args.seed,
        max_iter=args.max_iter,
        confidence=args.confidence,
        spatial_weight=args.spatial_weight,
        color_weight=args.color_weight,
        opacity_weight=args.opacity_weight,
        normalize=not args.no_normalize,
    )
    save_object_field(args.output, result.field)
    metrics = object_field_metrics(
        result.field,
        positions_xyz=cloud_positions_for_metrics(cloud) if args.smoothness else None,
        neighbors=args.neighbors,
        max_smooth_points=args.max_smooth_points,
    )

    print(f"object_field={args.output}")
    print(f"gaussians={result.field.gaussian_count} slots={result.field.slots}")
    print(f"backend={result.clustering.backend} inertia={result.clustering.inertia:.4f}")
    _print_metrics(metrics)
    _print_summary(result.field.labels())

    if args.ply_output:
        labeled = attach_hard_labels(cloud, result.field)
        if args.colorize:
            labeled = apply_object_colors(labeled, rewrite_sh=args.rewrite_sh)
        write_ply(args.ply_output, labeled, fmt=_output_format(args))
        print(f"ply={args.ply_output}")


def _object_field_export(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    field = load_object_field(args.field)
    labels = _labels_with_unknown_policy(field, args)
    labeled = attach_hard_labels(
        cloud,
        field,
        object_id_field=args.object_id_field,
        min_confidence=args.min_confidence,
        unknown_label=args.unknown_object_id,
    )
    if args.colorize:
        labeled = apply_object_colors(
            labeled,
            object_id_field=args.object_id_field,
            rewrite_sh=args.rewrite_sh,
        )
    write_ply(args.output, labeled, fmt=_output_format(args))
    print(f"exported {cloud.count} gaussians from {args.field} to {args.output}")
    _print_unknown_policy(field, labels, args)
    _print_summary(labels)


def _object_field_stats(args: argparse.Namespace) -> None:
    field = load_object_field(args.field)
    metrics = object_field_metrics(field)
    labels = _labels_with_unknown_policy(field, args)
    print(f"gaussians={field.gaussian_count}")
    print(f"slots={field.slots}")
    _print_metrics(metrics)
    _print_unknown_policy(field, labels, args)
    _print_summary(labels)


def _object_field_emergence(args: argparse.Namespace) -> None:
    field = load_object_field(args.field)
    positions_xyz = None
    if args.cloud:
        cloud = read_ply(args.cloud)
        if cloud.count != field.gaussian_count:
            raise ValueError(
                f"field has {field.gaussian_count} gaussians for cloud with {cloud.count}"
            )
        positions_xyz = cloud_positions_for_metrics(cloud)
    reference = load_object_field(args.reference) if args.reference else None
    metrics = object_emergence_metrics(
        field,
        positions_xyz=positions_xyz,
        reference=reference,
    )

    assignment = metrics["assignment"]
    spatial = metrics["spatial"]
    stability = metrics["stability"]
    score = metrics["object_emergence_score"]

    print(f"gaussians={metrics['gaussians']}")
    print(f"slots={metrics['slots']}")
    print(f"assignment_confidence={assignment['assignment_confidence']:.6f}")
    print(f"mean_normalized_entropy={assignment['mean_normalized_entropy']:.6f}")
    print(f"effective_slots={assignment['effective_slots']:.6f}")
    print(f"low_entropy_fraction={assignment['low_entropy_fraction']:.6f}")
    print(f"high_entropy_fraction={assignment['high_entropy_fraction']:.6f}")
    if spatial:
        print(f"spatial_compactness_score={spatial['compactness_score']:.6f}")
        print(f"spatial_overall_normalized_compactness={spatial['overall_normalized_compactness']:.6f}")
    if stability:
        print(f"stability_ari={stability['adjusted_rand_index']:.6f}")
        print(f"matched_label_agreement={stability['matched_label_agreement']:.6f}")
    if score["score"] is not None:
        print(f"object_emergence_score={score['score']:.6f}")
    else:
        print("object_emergence_score=None")
    print(f"object_emergence_complete={str(score['complete']).lower()}")
    print(f"missing_components={','.join(score['missing_components']) or '-'}")

    if args.output:
        write_json(args.output, metrics)
        print(f"summary={args.output}")


def _object_field_emergence_curve(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    field = load_object_field(args.field)
    if field.gaussian_count != cloud.count:
        raise ValueError(
            f"field has {field.gaussian_count} gaussians for cloud with {cloud.count}"
        )
    votes = vote_masks_to_gaussians(
        cloud,
        args.masks,
        slots=field.slots,
        max_frames=args.max_frames,
    )
    heldout_votes = None
    if args.heldout_masks:
        heldout_votes = vote_masks_to_gaussians(
            cloud,
            args.heldout_masks,
            slots=field.slots,
            max_frames=args.heldout_max_frames,
        )
    render_frames = None
    heldout_render_frames = None
    if not args.no_render_occlusion:
        render_frames = load_render_probe_frames(
            args.masks,
            max_frames=args.max_frames,
            max_size=args.render_size,
        )
        if args.heldout_masks:
            heldout_render_frames = load_render_probe_frames(
                args.heldout_masks,
                max_frames=args.heldout_max_frames,
                max_size=args.render_size,
            )
    curve = object_emergence_curve(
        field,
        votes,
        positions_xyz=cloud_positions_for_metrics(cloud),
        cloud=cloud,
        render_frames=render_frames,
        heldout_vote_result=heldout_votes,
        heldout_render_frames=heldout_render_frames,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        eval_every=args.eval_every,
    )
    write_json(args.output, curve)
    if args.csv_output:
        write_emergence_curve_csv(args.csv_output, curve)

    points = curve["points"]
    first = points[0]
    final = points[-1]
    final_occlusion = final["mask_proxy_occlusion_delta"]
    final_render_occlusion = final.get("render_occlusion_delta")
    final_heldout = final.get("heldout")
    print(f"curve={args.output}")
    if args.csv_output:
        print(f"csv={args.csv_output}")
    print(f"occlusion_delta_kind={curve['occlusion_delta_kind']}")
    print(f"points={len(points)}")
    print(f"initial_projection_loss={first['projection_loss']:.6f}")
    print(f"final_projection_loss={final['projection_loss']:.6f}")
    print(f"final_assignment_confidence={final['assignment_confidence']:.6f}")
    print(f"final_ari_to_initial={final['ari_to_initial']:.6f}")
    print(f"final_spatial_compactness_score={final['spatial_compactness_score']:.6f}")
    print(
        "final_mask_proxy_occlusion_mean_delta_loss="
        f"{final_occlusion['mean_delta_loss']:.6f}"
    )
    if final_render_occlusion:
        print(
            "final_render_occlusion_mean_delta_l1="
            f"{final_render_occlusion['mean_delta_l1']:.6f}"
        )
        print(
            "final_render_occlusion_mean_relative_delta_l1="
            f"{final_render_occlusion['mean_relative_delta_l1']:.6f}"
        )
        print(
            "final_render_occlusion_effect_score="
            f"{final_render_occlusion['occlusion_effect_score']:.6f}"
        )
    if isinstance(final_heldout, dict):
        print(
            "final_heldout_projection_loss="
            f"{_format_optional_float(final_heldout.get('projection_loss'))}"
        )
        print(f"final_heldout_supervised_gaussians={final_heldout['supervised_gaussians']}")
        heldout_render = final_heldout.get("render_occlusion_delta")
        if isinstance(heldout_render, dict):
            print(
                "final_heldout_render_occlusion_effect_score="
                f"{heldout_render['occlusion_effect_score']:.6f}"
            )


def _object_field_emergence_report(args: argparse.Namespace) -> None:
    if args.label and len(args.label) != len(args.curves):
        raise ValueError("--label count must match curve JSON count")
    curves = [
        load_emergence_curve(
            path,
            label=args.label[index] if args.label else None,
        )
        for index, path in enumerate(args.curves)
    ]
    summary = write_emergence_curve_report(
        args.output,
        curves,
        title=args.title,
    )
    print(f"report={summary['output']}")
    print(f"curves={summary['curves']}")
    print(f"charts={summary['charts']}")
    print(f"metrics={','.join(summary['metrics'])}")


def _object_field_emergence_benchmark(args: argparse.Namespace) -> None:
    summary = run_emergence_benchmark(
        args.manifest,
        output_dir=args.output_dir,
        report_path=args.report,
        summary_path=args.summary,
        strict=args.strict,
    )
    print(f"benchmark={args.manifest}")
    print(f"output_dir={summary['output_dir']}")
    print(f"summary={summary['summary_path']}")
    print(f"report={summary['report']['output']}")
    print(f"failure_report={summary['failure_report']}")
    print(f"scenes={len(summary['scenes'])}")
    print(f"passed={str(summary['passed']).lower()}")
    for scene in summary["scenes"]:
        print(
            f"scene={scene['id']} passed={str(scene['passed']).lower()} "
            f"points={scene['points']} "
            f"projection_loss={scene['initial_projection_loss']:.6f}->{scene['final_projection_loss']:.6f} "
            f"render_occlusion_effect={scene['final_render_occlusion_effect_score']:.6f}"
        )
        heldout = scene.get("heldout")
        if isinstance(heldout, dict):
            initial = heldout.get("initial_projection_loss")
            final = heldout.get("final_projection_loss")
            effect = heldout.get("render_occlusion_effect_score")
            print(
                f"heldout_scene={scene['id']} "
                f"supervised_gaussians={heldout['supervised_gaussians']} "
                f"projection_loss={_format_optional_float(initial)}->{_format_optional_float(final)} "
                f"render_occlusion_effect={_format_optional_float(effect)}"
            )


def _object_field_inspect_nerf(args: argparse.Namespace) -> None:
    summary = inspect_nerf_dataset(args.dataset)
    print(f"dataset={summary.root}")
    print(f"frames={summary.total_frames}")
    print(f"missing_images={summary.missing_images}")
    print(f"invalid_transforms={summary.invalid_transforms}")
    for split in summary.splits:
        print(
            f"split={split.name} frames={split.frames} "
            f"missing_images={split.missing_images} "
            f"invalid_transforms={split.invalid_transforms}"
        )
    if args.output:
        write_json(args.output, summary.as_dict())
        print(f"manifest={args.output}")


def _object_field_vote_masks(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    field = load_object_field(args.field)
    if field.gaussian_count != cloud.count:
        raise ValueError(
            f"field has {field.gaussian_count} gaussians for cloud with {cloud.count}"
        )
    votes = vote_masks_to_gaussians(
        cloud,
        args.masks,
        slots=field.slots,
        max_frames=args.max_frames,
        background_slot=args.background_slot,
        background_weight=args.background_weight,
        visibility_mode=args.visibility_mode,
        depth_tolerance=args.depth_tolerance,
    )
    result = train_object_field_from_votes(
        field,
        votes,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
    )
    save_object_field(args.output, result.field)
    print(f"object_field={args.output}")
    print(f"frames={votes.frames}")
    print(f"visibility_mode={votes.visibility_mode}")
    print(f"projected={votes.projected}")
    if votes.visibility_mode == "depth-buffer":
        print(f"raw_projected={votes.raw_projected}")
        print(f"depth_culled={votes.depth_culled}")
        print(f"depth_culled_matched={votes.depth_culled_matched}")
    print(f"matched={votes.matched}")
    if votes.background_slot is not None:
        print(f"background_slot={votes.background_slot}")
        print(f"background_matched={votes.background_matched}")
    print(f"supervised_gaussians={result.supervised_gaussians}")
    vote_quality = mask_vote_quality_audit(votes)
    conflict = vote_quality["vote_conflict"]
    print(f"supervised_fraction={vote_quality['supervised_fraction']:.6f}")
    print(f"vote_conflict_gaussians={conflict['gaussians']}")
    print(f"vote_conflict_fraction={conflict['fraction']:.6f}")
    print(f"vote_target_entropy={conflict['normalized_target_entropy']:.6f}")
    print(f"initial_loss={result.initial_loss:.6f}")
    print(f"final_loss={result.final_loss:.6f}")
    _print_metrics(object_field_metrics(result.field))
    labels = _labels_with_unknown_policy(result.field, args)
    _print_unknown_policy(result.field, labels, args)
    _print_summary(labels)

    if args.summary_output:
        write_json(args.summary_output, training_summary(result))
        print(f"summary={args.summary_output}")

    if args.ply_output:
        labeled = attach_hard_labels(
            cloud,
            result.field,
            min_confidence=args.min_confidence,
            unknown_label=args.unknown_object_id,
        )
        if args.colorize:
            labeled = apply_object_colors(labeled, rewrite_sh=args.rewrite_sh)
        write_ply(args.ply_output, labeled, fmt=_output_format(args))
        print(f"ply={args.ply_output}")


def _object_field_vote_diagnostics(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    summary = depth_visibility_diagnostic(
        cloud,
        args.masks,
        slots=args.slots,
        max_frames=args.max_frames,
        background_slot=args.background_slot,
        background_weight=args.background_weight,
        depth_tolerance=args.depth_tolerance,
    )
    baseline = summary["baseline"]
    depth_aware = summary["depth_aware"]
    deltas = summary["deltas"]
    print("diagnostic=objgauss-depth-visibility-vote-diagnostic-v1")
    print(f"frames={baseline['frames']}")
    print(f"baseline_conflict_fraction={baseline['vote_conflict_fraction']:.6f}")
    print(f"depth_conflict_fraction={depth_aware['vote_conflict_fraction']:.6f}")
    print(f"conflict_fraction_reduction={deltas['vote_conflict_fraction_reduction']:.6f}")
    print(f"baseline_slot_balance={baseline['slot_balance_score']:.6f}")
    print(f"depth_slot_balance={depth_aware['slot_balance_score']:.6f}")
    print(f"slot_balance_delta={deltas['slot_balance_score_delta']:.6f}")
    print(f"baseline_supervised_fraction={baseline['supervised_fraction']:.6f}")
    print(f"depth_supervised_fraction={depth_aware['supervised_fraction']:.6f}")
    print(f"depth_culled_matched={deltas['depth_culled_matched']}")
    print(f"recommendation={summary['recommendation']}")
    if args.output:
        write_json(args.output, summary)
        print(f"summary={args.output}")


def _masks_from_nerf_alpha(args: argparse.Namespace) -> None:
    result = build_nerf_alpha_mask_manifest(
        args.dataset,
        output=args.output,
        split=args.split,
        max_frames=args.max_frames,
        slot=args.slot,
        label=args.label,
        threshold=args.threshold,
    )
    print(f"manifest={result.manifest_path}")
    print(f"frames={result.frames}")
    print(f"masks={result.masks}")
    print(f"width={result.width}")
    print(f"height={result.height}")
    print(f"foreground_pixels={result.foreground_pixels}")


def _masks_from_nerf_alpha_fgbg(args: argparse.Namespace) -> None:
    result = build_nerf_alpha_fgbg_mask_manifest(
        args.dataset,
        output=args.output,
        split=args.split,
        max_frames=args.max_frames,
        foreground_threshold=args.foreground_threshold,
        background_threshold=args.background_threshold,
        foreground_slot=args.foreground_slot,
        background_slot=args.background_slot,
        foreground_confidence=args.foreground_confidence,
        background_confidence=args.background_confidence,
    )
    print(f"manifest={result.manifest_path}")
    print(f"frames={result.frames}")
    print(f"masks={result.masks}")
    print(f"width={result.width}")
    print(f"height={result.height}")
    print(f"foreground_pixels={result.foreground_pixels}")
    print(f"background_pixels={result.background_pixels}")
    print(f"ignore_pixels={result.ignore_pixels}")


def _masks_from_nerf_rgba_colors(args: argparse.Namespace) -> None:
    result = build_nerf_rgba_color_mask_manifest(
        args.dataset,
        output=args.output,
        split=args.split,
        max_frames=args.max_frames,
        alpha_threshold=args.alpha_threshold,
    )
    print(f"manifest={result.manifest_path}")
    print(f"frames={result.frames}")
    print(f"masks={result.masks}")
    print(f"width={result.width}")
    print(f"height={result.height}")
    print(f"foreground_pixels={result.foreground_pixels}")
    for slot in result.slot_pixel_counts:
        print(f"slot={slot['slot']} label={slot['label']} pixels={slot['count']}")


def _masks_from_nerf_sam(args: argparse.Namespace) -> None:
    result = build_nerf_sam_mask_manifest(
        args.dataset,
        output=args.output,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        split=args.split,
        max_frames=args.max_frames,
        max_masks_per_frame=args.max_masks_per_frame,
        min_area=args.min_area,
        max_area_fraction=args.max_area_fraction,
        max_image_size=args.max_image_size,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )
    print(f"manifest={result.manifest_path}")
    print(f"frames={result.frames}")
    print(f"masks={result.masks}")
    print(f"width={result.width}")
    print(f"height={result.height}")
    print(f"mask_pixels={result.mask_pixels}")
    print(f"slots={result.slots}")


def _masks_split_manifest(args: argparse.Namespace) -> None:
    result = split_mask_manifest(
        args.source,
        train_output=args.train_output,
        heldout_output=args.heldout_output,
        heldout_every=args.heldout_every,
        heldout_offset=args.heldout_offset,
    )
    print(f"train_manifest={result.train_manifest_path}")
    print(f"heldout_manifest={result.heldout_manifest_path}")
    print(f"source_frames={result.source_frames}")
    print(f"train_frames={result.train_frames}")
    print(f"heldout_frames={result.heldout_frames}")
    print(f"train_masks={result.train_masks}")
    print(f"heldout_masks={result.heldout_masks}")


def _masks_validate(args: argparse.Namespace) -> None:
    result = validate_mask_manifest(
        args.manifest,
        dataset=args.dataset,
        max_overlap_fraction=args.max_overlap_fraction,
        max_mask_area_fraction=args.max_mask_area_fraction,
        allow_empty=args.allow_empty,
    )
    print(f"manifest={result.manifest_path}")
    print(f"passed={str(result.passed).lower()}")
    print(f"frames={result.frames}")
    print(f"masks={result.masks}")
    print(f"slots={','.join(str(slot) for slot in result.slots) if result.slots else '-'}")
    print(f"errors={len(result.errors)}")
    print(f"warnings={len(result.warnings)}")
    for index, frame in enumerate(result.frame_stats[: args.max_report_frames]):
        print(
            f"frame={frame['frame_index']} "
            f"overlap_pixels={frame['overlap_pixels']} "
            f"overlap_fraction={frame['overlap_fraction']:.6f} "
            f"ignore_pixels={frame['ignore_pixels'] if frame['ignore_pixels'] is not None else '-'}"
        )
        for mask in frame["masks"]:
            print(
                f"frame={frame['frame_index']} slot={mask['slot']} "
                f"pixels={mask['pixels']} fraction={mask['fraction']:.6f}"
            )
    for error in result.errors:
        print(f"error={error}")
    for warning in result.warnings:
        print(f"warning={warning}")
    if args.summary_output:
        write_json(args.summary_output, result.as_dict())
        print(f"summary={args.summary_output}")
    if not result.passed and args.strict:
        raise ValueError("mask manifest validation failed")


def _masks_score_clip(args: argparse.Namespace) -> None:
    labels = read_clip_labels(
        args.labels or [],
        labels_file=args.labels_file,
        presets=args.label_preset or [],
    )
    result = score_mask_manifest_with_clip(
        args.manifest,
        output=args.output,
        labels=labels,
        dataset=args.dataset,
        backend=args.backend,
        model=args.model,
        device=args.device,
        max_frames=args.max_frames,
        max_masks=args.max_masks,
        crop_padding=args.crop_padding,
        background_fill=args.background_fill,
        prompt_templates=args.prompt_template,
        background_labels=args.background_labels,
        min_unique_top_labels=args.min_unique_top_labels,
        max_top_label_fraction=args.max_top_label_fraction,
        max_background_label_fraction=args.max_background_label_fraction,
        overwrite_scores=args.overwrite_scores,
    )
    quality = result.naming_quality
    print(f"scored_manifest={result.output_manifest}")
    print(f"backend={result.backend}")
    print(f"model={result.model}")
    print(f"labels={len(result.labels)}")
    print(f"prompt_templates={len(result.prompt_templates)}")
    print(f"background_fill={result.background_fill}")
    print(f"frames={result.frames}")
    print(f"masks={result.masks}")
    print(f"scored_masks={result.scored_masks}")
    print(f"cached_masks={result.cached_masks}")
    print(f"named_masks={result.named_masks}")
    print(f"naming_quality={'passed' if quality.get('passed') else 'failed'}")
    print(f"top_label_counts={quality.get('top_label_counts', {})}")
    if quality.get("blockers"):
        print(f"quality_blockers={quality['blockers']}")
    if args.summary_output:
        write_json(args.summary_output, result.as_dict())
        print(f"summary={args.summary_output}")
    if args.require_naming_quality and not quality.get("passed"):
        raise ValueError(f"CLIP naming quality gate failed: {quality.get('blockers', [])}")


def _masks_align_slots(args: argparse.Namespace) -> None:
    cloud = read_ply(args.cloud)
    result = align_mask_manifest_slots(
        cloud,
        args.manifest,
        output=args.output,
        min_iou=args.min_iou,
        min_shared_gaussians=args.min_shared_gaussians,
        max_slots=args.max_slots,
        max_frames=args.max_frames,
        min_mask_area=args.min_mask_area,
        min_mask_area_fraction=args.min_mask_area_fraction,
        exclude_top_labels=args.exclude_top_labels,
        exclude_background_top_labels=args.exclude_background_top_labels,
        background_labels=args.background_labels,
        min_named_slots=args.min_named_slots,
        min_unique_slot_labels=args.min_unique_slot_labels,
        max_slot_label_fraction=args.max_slot_label_fraction,
        max_background_slot_fraction=args.max_background_slot_fraction,
        foreground_only_slot_names=args.foreground_only_slot_names,
        unique_slot_names=args.unique_slot_names,
        slot_name_diversity_penalty=args.slot_name_diversity_penalty,
        min_slot_support_gaussians=args.min_slot_support_gaussians,
        min_slot_support_ratio=args.min_slot_support_ratio,
        min_balanced_slots=args.min_balanced_slots,
        recover_foreground_coverage=args.recover_foreground_coverage,
    )
    quality = result.slot_naming_quality
    filters = result.record_filters
    rebalance = result.slot_rebalance
    print(f"aligned_manifest={result.output_manifest}")
    print(f"frames={result.frames}")
    print(f"masks={result.masks}")
    print(f"aligned_slots={result.aligned_slots}")
    print(f"remapped_masks={result.remapped_masks}")
    print(f"dropped_masks={result.dropped_masks}")
    print(f"named_slots={result.named_slots}")
    print(f"filtered_low_area={filters.get('filtered_low_area', 0)}")
    print(f"filtered_top_label={filters.get('filtered_top_label', 0)}")
    print(f"slot_naming_quality={'passed' if quality.get('passed') else 'failed'}")
    print(f"slot_label_counts={quality.get('slot_label_counts', {})}")
    print(f"foreground_only_slot_names={str(args.foreground_only_slot_names).lower()}")
    print(f"unique_slot_names={str(args.unique_slot_names).lower()}")
    print(f"slot_name_diversity_penalty={args.slot_name_diversity_penalty:.6f}")
    print(f"dropped_unbalanced_slots={rebalance.get('dropped_slots', 0)}")
    print(f"dropped_unbalanced_masks={rebalance.get('dropped_masks', 0)}")
    print(f"support_balance_score={float(rebalance.get('support_balance_score', 0.0)):.6f}")
    recovery = result.foreground_coverage_recovery
    print(f"foreground_coverage_recovery={'enabled' if recovery.get('enabled') else 'disabled'}")
    print(f"coverage_recovered_masks={recovery.get('recovered_masks', 0)}")
    print(f"coverage_recovered_gaussians={recovery.get('recovered_gaussian_support', 0)}")
    if quality.get("blockers"):
        print(f"slot_quality_blockers={quality['blockers']}")
    for cluster in result.clusters:
        print(
            f"slot={cluster['slot']} "
            f"label={cluster['semantic_label']} "
            f"source={cluster['semantic_name_source']} "
            f"masks={cluster['mask_count']} "
            f"frames={cluster['frame_count']} "
            f"support_gaussians={cluster['support_gaussians']}"
        )
    if args.require_slot_quality and not quality.get("passed"):
        raise ValueError(f"slot naming quality gate failed: {quality.get('blockers', [])}")


def _masks_compare_baselines(args: argparse.Namespace) -> None:
    summary = compare_baseline_candidates(
        args.candidate,
        min_supervised_fraction=args.min_supervised_fraction,
        max_vote_conflict_fraction=args.max_vote_conflict_fraction,
        min_slot_balance_score=args.min_slot_balance_score,
        min_object_active_slots=args.min_object_active_slots,
        object_id_field=args.object_id_field,
    )
    write_json(args.output, summary)
    if args.markdown_output:
        write_comparison_markdown(args.markdown_output, summary)
    policy = summary["promotion_policy"]
    print(f"comparison_summary={args.output}")
    if args.markdown_output:
        print(f"comparison_markdown={args.markdown_output}")
    print(f"candidates={summary['candidate_count']}")
    print(f"promotion_policy={policy['status']}")
    print(f"recommended_candidate={policy.get('recommended_candidate') or '-'}")
    if policy.get("blockers"):
        print(f"promotion_blockers={policy['blockers']}")
    for candidate in summary["candidates"]:
        promotion = candidate["promotion"]
        print(
            f"candidate={candidate['name']} "
            f"promotion={promotion['status']} "
            f"blockers={promotion.get('blockers', [])}"
        )
    if args.require_promotion_ready and policy["status"] != "promote":
        raise ValueError(f"semantic promotion policy failed: {policy.get('blockers', [])}")


def _demo_v1_closure(args: argparse.Namespace) -> None:
    result = build_v1_closure_demo(
        input_ply=args.input,
        splat_path=args.splat,
        output_dir=args.output_dir,
        public_dir=None if args.no_public_copy else args.public_dir,
        image_size=args.image_size,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
    )
    print(f"manifest={result.manifest_path}")
    print(f"mask_manifest={result.mask_manifest_path}")
    print(f"initial_field={result.initial_field_path}")
    print(f"trained_field={result.trained_field_path}")
    print(f"output_ply={result.output_ply_path}")
    if result.public_ply_path:
        print(f"public_ply={result.public_ply_path}")
    print(f"gaussians={result.gaussian_count}")
    print(f"objects={result.object_count}")
    print(f"supervised_gaussians={result.supervised_gaussians}")
    print(f"initial_loss={result.initial_loss:.6f}")
    print(f"final_loss={result.final_loss:.6f}")


def _demo_verify_v1_closure(args: argparse.Namespace) -> None:
    result = verify_v1_closure_demo(
        args.manifest,
        asset_library_path=args.asset_library,
        require_public_copy=not args.no_require_public_copy,
    )
    print(f"manifest={result.manifest_path}")
    print(f"passed={str(result.passed).lower()}")
    for key, value in result.summary.items():
        print(f"{key}={value}")
    for check in result.checks:
        status = "pass" if check["passed"] else "fail"
        print(f"check={check['name']} status={status} detail={check['detail']}")
    if not result.passed:
        raise ValueError("v1 closure verification failed")


def _demo_plush_semantic_closure(args: argparse.Namespace) -> None:
    result = build_plush_semantic_closure_demo(
        input_ply=args.input,
        splat_path=args.splat,
        output_dir=args.output_dir,
        public_dir=None if args.no_public_copy else args.public_dir,
        image_size=args.image_size,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
    )
    print(f"manifest={result.manifest_path}")
    print(f"mask_manifest={result.mask_manifest_path}")
    print(f"initial_field={result.initial_field_path}")
    print(f"trained_field={result.trained_field_path}")
    print(f"output_ply={result.output_ply_path}")
    if result.public_ply_path:
        print(f"public_ply={result.public_ply_path}")
    if result.public_splat_path:
        print(f"public_splat={result.public_splat_path}")
    print(f"gaussians={result.gaussian_count}")
    print(f"slots={result.slot_count}")
    print(f"objects={result.object_count}")
    print(f"supervised_gaussians={result.supervised_gaussians}")
    print(f"initial_loss={result.initial_loss:.6f}")
    print(f"final_loss={result.final_loss:.6f}")


def _demo_verify_plush_semantic_closure(args: argparse.Namespace) -> None:
    result = verify_plush_semantic_closure_demo(
        args.manifest,
        asset_library_path=args.asset_library,
        require_public_copy=not args.no_require_public_copy,
        min_views=args.min_views,
    )
    print(f"manifest={result.manifest_path}")
    print(f"passed={str(result.passed).lower()}")
    for key, value in result.summary.items():
        print(f"{key}={value}")
    for check in result.checks:
        status = "pass" if check["passed"] else "fail"
        print(f"check={check['name']} status={status} detail={check['detail']}")
    if not result.passed:
        raise ValueError("Plush semantic closure verification failed")


def _demo_lego_alpha_closure(args: argparse.Namespace) -> None:
    result = build_lego_alpha_closure_demo(
        dataset=args.dataset,
        output_dir=args.output_dir,
        public_dir=None if args.no_public_copy else args.public_dir,
        split=args.split,
        max_frames=args.max_frames,
        sample_stride=args.sample_stride,
        depth=args.depth,
        alpha_threshold=args.alpha_threshold,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
    )
    print(f"manifest={result.manifest_path}")
    print(f"mask_manifest={result.mask_manifest_path}")
    print(f"raw_ply={result.raw_ply_path}")
    print(f"splat={result.splat_path}")
    print(f"trained_field={result.trained_field_path}")
    print(f"output_ply={result.output_ply_path}")
    if result.public_ply_path:
        print(f"public_ply={result.public_ply_path}")
    if result.public_splat_path:
        print(f"public_splat={result.public_splat_path}")
    print(f"gaussians={result.gaussian_count}")
    print(f"objects={result.object_count}")
    print(f"supervised_gaussians={result.supervised_gaussians}")
    print(f"initial_loss={result.initial_loss:.6f}")
    print(f"final_loss={result.final_loss:.6f}")


def _demo_verify_lego_alpha_closure(args: argparse.Namespace) -> None:
    result = verify_lego_alpha_closure_demo(
        args.manifest,
        asset_library_path=args.asset_library,
        require_public_copy=not args.no_require_public_copy,
        min_frames=args.min_frames,
    )
    print(f"manifest={result.manifest_path}")
    print(f"passed={str(result.passed).lower()}")
    for key, value in result.summary.items():
        print(f"{key}={value}")
    for check in result.checks:
        status = "pass" if check["passed"] else "fail"
        print(f"check={check['name']} status={status} detail={check['detail']}")
    if not result.passed:
        raise ValueError("Lego alpha closure verification failed")


def _demo_audit_v1_goal(args: argparse.Namespace) -> None:
    result = audit_v1_goal(
        v1_manifest=args.v1_manifest,
        lego_manifest=args.lego_manifest,
        semantic_manifest=args.semantic_manifest,
        trained_manifest=args.trained_manifest,
        asset_library_path=args.asset_library,
    )
    print(f"passed={str(result.passed).lower()}")
    for key, value in result.summary.items():
        if isinstance(value, list):
            print(f"{key}={','.join(str(item) for item in value) if value else '-'}")
        else:
            print(f"{key}={value}")
    for check in result.checks:
        status = "pass" if check.passed else "fail"
        print(f"check={check.name} status={status} detail={check.detail}")
    if not result.passed and not args.allow_incomplete:
        raise ValueError("ObjGauss v1 goal audit is incomplete")


def _training_register_output(args: argparse.Namespace) -> None:
    result = register_training_output(
        args.input,
        output_dir=args.output_dir,
        asset_id=args.asset_id,
        dataset=args.dataset,
        masks=args.masks,
        slots=args.slots,
        public_dir=None if args.no_public_copy else args.public_dir,
        public_name=args.public_name,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        colorize=not args.no_colorize,
        object_min_confidence=args.object_min_confidence,
        unknown_object_id=args.unknown_object_id,
        background_slot=args.background_slot,
        background_weight=args.background_weight,
    )
    print(f"manifest={result.manifest_path}")
    print(f"gaussian_ply={result.gaussian_ply_path}")
    print(f"splat={result.splat_path}")
    if result.object_field_path:
        print(f"object_field={result.object_field_path}")
    if result.object_ply_path:
        print(f"object_ply={result.object_ply_path}")
    if result.public_splat_path:
        print(f"public_splat={result.public_splat_path}")
    if result.public_object_ply_path:
        print(f"public_object_ply={result.public_object_ply_path}")
    print(f"gaussians={result.gaussian_count}")
    if result.slots is not None:
        print(f"slots={result.slots}")
    if result.supervised_gaussians is not None:
        print(f"supervised_gaussians={result.supervised_gaussians}")
    if result.background_slot is not None:
        print(f"background_slot={result.background_slot}")
    if result.background_matched is not None:
        print(f"background_matched={result.background_matched}")
    if result.unknown_object_id is not None:
        print(f"unknown_object_id={result.unknown_object_id}")
    if result.unknown_gaussians is not None:
        print(f"unknown_gaussians={result.unknown_gaussians}")
    if result.initial_loss is not None and result.final_loss is not None:
        print(f"initial_loss={result.initial_loss:.6f}")
        print(f"final_loss={result.final_loss:.6f}")


def _training_kernel_mvp(args: argparse.Namespace) -> None:
    result = train_kernel_mvp(
        make_trainable_kernel_mvp_fixture(),
        slots=args.slots,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        render_weight=args.render_weight,
        image_render_weight=args.image_render_weight,
        object_weight=args.object_weight,
        temporal_weight=args.temporal_weight,
        image_renderer=args.image_renderer,
        seed=args.seed,
        record_every=args.record_every,
    )
    summary = result.as_dict()
    print(f"schema={summary['schema']}")
    print(f"frames={summary['frame_count']}")
    print(f"slots={summary['slots']}")
    print(f"iterations={summary['iterations']}")
    print(f"image_renderer={summary['image_renderer']}")
    print(f"initial_total_loss={result.initial_loss.total_loss:.6f}")
    print(f"final_total_loss={result.final_loss.total_loss:.6f}")
    print(f"initial_render_loss={result.initial_loss.render_loss:.6f}")
    print(f"final_render_loss={result.final_loss.render_loss:.6f}")
    print(f"initial_image_render_loss={result.initial_loss.image_render_loss:.6f}")
    print(f"final_image_render_loss={result.final_loss.image_render_loss:.6f}")
    print(f"final_object_loss={result.final_loss.object_loss:.6f}")
    print(f"final_temporal_loss={result.final_loss.temporal_loss:.6f}")
    print(f"loss_decreased={str(summary['loss_decreased']).lower()}")
    print(f"render_loss_decreased={str(summary['render_loss_decreased']).lower()}")
    print(f"image_render_loss_decreased={str(summary['image_render_loss_decreased']).lower()}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.require_loss_decrease and not summary["loss_decreased"]:
        raise ValueError("trainable kernel MVP loss did not decrease")


def _training_kernel_sample(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    result, sample = train_kernel_mvp_from_cloud(
        cloud,
        slots=args.slots,
        frame_count=args.frames,
        max_points=args.max_points,
        object_id_field=args.object_id_field,
        temporal_offset=args.temporal_offset,
        bind_image_targets=args.bind_image_targets,
        image_width=args.image_width,
        image_height=args.image_height,
        point_radius=args.point_radius,
        visibility_policy=args.visibility_policy,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        render_weight=args.render_weight,
        image_render_weight=args.image_render_weight,
        object_weight=args.object_weight,
        temporal_weight=args.temporal_weight,
        image_renderer=args.image_renderer,
        seed=args.seed,
        record_every=args.record_every,
    )
    summary = {
        **result.as_dict(),
        "sample": sample.as_dict(),
        "input": str(args.input),
    }
    if summary["image_target_contract"]["status"] == "image_targets_bound":
        renderer_api = _evaluate_training_renderer_api(
            sample.frames,
            result.assignments,
            result.decoder_colors,
            image_renderer=args.image_renderer,
        )
        summary["renderer_api"] = renderer_api.as_dict()
    print(f"schema={summary['schema']}")
    print(f"input={args.input}")
    print(f"source_gaussians={sample.source_count}")
    print(f"sampled_gaussians={sample.sampled_count}")
    print(f"frames={summary['frame_count']}")
    print(f"slots={summary['slots']}")
    print(f"target_source={sample.target_source}")
    print(f"render_target_mode={summary['render_target_mode']}")
    print(f"image_renderer={summary['image_renderer']}")
    image_contract = summary["image_target_contract"]
    print(f"image_targets_status={image_contract['status']}")
    print(f"image_targets_bound={image_contract['targets_bound']}")
    if image_contract["visibility_policies"]:
        print(f"visibility_policy={image_contract['visibility_policies'][0]}")
    renderer_api_summary = summary.get("renderer_api")
    if isinstance(renderer_api_summary, dict):
        print(f"renderer_api_status={renderer_api_summary['status']}")
        print(f"renderer_name={renderer_api_summary['renderer_name']}")
        print(f"renderer_gradient_path={renderer_api_summary['gradient_path']}")
        print(f"image_render_loss={renderer_api_summary['image_render_loss']:.6f}")
    print(f"initial_total_loss={result.initial_loss.total_loss:.6f}")
    print(f"final_total_loss={result.final_loss.total_loss:.6f}")
    print(f"initial_render_loss={result.initial_loss.render_loss:.6f}")
    print(f"final_render_loss={result.final_loss.render_loss:.6f}")
    print(f"initial_image_render_loss={result.initial_loss.image_render_loss:.6f}")
    print(f"final_image_render_loss={result.final_loss.image_render_loss:.6f}")
    print(f"final_object_loss={result.final_loss.object_loss:.6f}")
    print(f"final_temporal_loss={result.final_loss.temporal_loss:.6f}")
    print(f"loss_decreased={str(summary['loss_decreased']).lower()}")
    print(f"render_loss_decreased={str(summary['render_loss_decreased']).lower()}")
    print(f"image_render_loss_decreased={str(summary['image_render_loss_decreased']).lower()}")
    model_artifact = None
    if args.model_output:
        model_artifact = write_trainable_kernel_model_artifact(
            args.model_output,
            result,
            sample=sample,
            input_path=args.input,
            renderer_api=summary.get("renderer_api") if isinstance(summary.get("renderer_api"), dict) else None,
        )
        summary["model_artifact"] = {
            "schema": model_artifact["schema"],
            "path": str(args.model_output),
            "kind": model_artifact["kind"],
        }
        print(f"model_artifact_schema={model_artifact['schema']}")
        print(f"model_artifact={args.model_output}")
    if args.quality_report_output:
        if model_artifact is None:
            raise ValueError("--quality-report-output requires --model-output")
        quality_report = write_trainable_quality_report(
            args.quality_report_output,
            model_artifact,
            report_id=args.quality_report_id,
            source={
                "type": "trainable_kernel_model_artifact",
                "artifact": str(args.model_output),
            },
        )
        summary["quality_report"] = {
            "schema": quality_report["schema"],
            "path": str(args.quality_report_output),
            "status": quality_report["status"],
            "gate_count": len(quality_report["gates"]),
        }
        print(f"quality_report_schema={quality_report['schema']}")
        print(f"quality_report={args.quality_report_output}")
        print(f"quality_report_status={quality_report['status']}")
    if args.manifest_output:
        if not args.model_output:
            raise ValueError("--manifest-output requires --model-output")
        artifact_route = _manifest_relative_path(args.model_output, args.manifest_output)
        quality_report_route = (
            _manifest_relative_path(args.quality_report_output, args.manifest_output)
            if args.quality_report_output
            else None
        )
        model_manifest = manifest_from_trainable_kernel_model_artifact(
            args.model_output,
            artifact_path=artifact_route,
            quality_report_path=quality_report_route,
            quality_report_file=args.quality_report_output if args.quality_report_output else None,
            manifest_id=args.manifest_id,
            asset_id=args.manifest_asset_id,
            name=args.manifest_name,
            license=args.manifest_license,
        )
        write_model_artifact_manifest(args.manifest_output, model_manifest)
        summary["model_artifact_manifest"] = {
            "schema": model_manifest["schema"],
            "path": str(args.manifest_output),
            "manifest_id": model_manifest["manifest_id"],
            "artifact_path": artifact_route,
            "quality_report_path": quality_report_route,
        }
        print(f"model_artifact_manifest_schema={model_manifest['schema']}")
        print(f"model_artifact_manifest={args.manifest_output}")
        print(f"model_artifact_manifest_asset={model_manifest['asset_id']}")
        print(f"model_artifact_manifest_trainable={artifact_route}")
        if quality_report_route:
            print(f"model_artifact_manifest_quality={quality_report_route}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.require_loss_decrease and not summary["loss_decreased"]:
        raise ValueError("trainable kernel sample loss did not decrease")


def _training_decoder_mvp(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    sample = trainable_kernel_sample_from_cloud(
        cloud,
        slots=args.slots,
        frame_count=args.frames,
        max_points=args.max_points,
        object_id_field=args.object_id_field,
        temporal_offset=args.temporal_offset,
        bind_image_targets=True,
        image_width=args.image_width,
        image_height=args.image_height,
        point_radius=args.point_radius,
        visibility_policy=args.visibility_policy,
        seed=args.seed,
    )
    assignments, assignment_source = _decoder_training_assignments_from_args(args, sample)
    result = train_object_state_gaussian_decoder(
        sample.frames,
        assignments,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        render_weight=args.render_weight,
        image_render_weight=args.image_render_weight,
        object_weight=args.object_weight,
        temporal_weight=args.temporal_weight,
        image_renderer=args.image_renderer,
        gaussian_scale=args.gaussian_scale,
        gaussian_opacity=args.gaussian_opacity,
        seed=args.seed,
        record_every=args.record_every,
        vram_reserve_gb=1,
    )
    summary = {
        **result.as_dict(),
        "input": str(args.input),
        "sample": sample.as_dict(),
        "assignment_source": assignment_source,
        "solver_checkpoint": str(args.solver_checkpoint) if args.solver_checkpoint else None,
    }
    print(f"schema={summary['schema']}")
    print(f"input={args.input}")
    print(f"source_gaussians={sample.source_count}")
    print(f"sampled_gaussians={sample.sampled_count}")
    print(f"frames={summary['frame_count']}")
    print(f"slots={summary['slots']}")
    print(f"iterations={summary['iterations']}")
    print(f"assignment_source={assignment_source}")
    print(f"image_renderer={summary['image_renderer']}")
    print(f"gaussian_scale={summary['gaussian_policy']['default_scale']}")
    print(f"gaussian_opacity={summary['gaussian_policy']['default_opacity']}")
    print(f"initial_total_loss={result.initial_loss.total_loss:.6f}")
    print(f"final_total_loss={result.final_loss.total_loss:.6f}")
    print(f"initial_render_loss={result.initial_loss.render_loss:.6f}")
    print(f"final_render_loss={result.final_loss.render_loss:.6f}")
    print(f"initial_image_render_loss={result.initial_loss.image_render_loss:.6f}")
    print(f"final_image_render_loss={result.final_loss.image_render_loss:.6f}")
    print(f"final_object_loss={result.final_loss.object_loss:.6f}")
    print(f"final_temporal_loss={result.final_loss.temporal_loss:.6f}")
    print(f"loss_decreased={str(summary['loss_decreased']).lower()}")
    print(f"image_render_loss_decreased={str(summary['image_render_loss_decreased']).lower()}")
    print(f"trained_fields={','.join(summary['trained_fields'])}")
    print(f"frozen_fields={','.join(summary['frozen_fields'])}")
    print(f"gpu_used={str(summary['gpu_policy']['uses_gpu']).lower()}")
    print(f"vram_reserve_gb={summary['gpu_policy']['vram_reserve_gb']}")
    renderer_api = summary.get("renderer_api")
    if isinstance(renderer_api, dict):
        print(f"renderer_api_status={renderer_api['status']}")
        print(f"renderer_name={renderer_api['renderer_name']}")
        print(f"renderer_gradient_path={renderer_api['gradient_path']}")
        print(f"renderer_image_render_loss={renderer_api['image_render_loss']:.6f}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.require_loss_decrease and not summary["loss_decreased"]:
        raise ValueError("decoder MVP total loss did not decrease")
    if args.require_image_render_loss_decrease and not summary["image_render_loss_decreased"]:
        raise ValueError("decoder MVP image render loss did not decrease")


def _decoder_training_assignments_from_args(args: argparse.Namespace, sample):
    if args.solver_checkpoint:
        checkpoint = json.loads(args.solver_checkpoint.read_text(encoding="utf-8"))
        validate_object_emergence_solver_checkpoint(checkpoint)
        solver_state = object_emergence_solver_state_from_dict(checkpoint)
        assignments = []
        for frame_index, frame in enumerate(sample.frames):
            prediction = predict_object_emergence_assignment(
                ObjectEmergenceEvidence(
                    positions=frame.positions,
                    features=frame.features,
                    frame_index=frame_index,
                    source=f"solver_checkpoint:{args.solver_checkpoint}",
                ),
                solver_state,
            )
            assignments.append(prediction.assignment)
        return tuple(assignments), "solver_checkpoint"
    if not all(frame.target_assignment is not None for frame in sample.frames):
        raise ValueError("decoder-mvp requires object_id targets or --solver-checkpoint")
    return tuple(frame.target_assignment for frame in sample.frames), sample.target_source


def _training_solver_decoder_mvp(args: argparse.Namespace) -> None:
    if args.resume_checkpoint and args.solver_checkpoint:
        raise ValueError("--resume-checkpoint cannot be combined with --solver-checkpoint")
    if args.checkpoint_every and not args.run_output_dir:
        raise ValueError("--checkpoint-every requires --run-output-dir")
    if args.tensorboard_logdir and not args.run_output_dir:
        raise ValueError("--tensorboard-logdir requires --run-output-dir")
    if args.train_decoder_opacity and args.freeze_decoder_opacity:
        raise ValueError("--train-decoder-opacity cannot be combined with --freeze-decoder-opacity")
    if args.train_decoder_scale and args.freeze_decoder_scale:
        raise ValueError("--train-decoder-scale cannot be combined with --freeze-decoder-scale")
    record_every = _solver_decoder_record_every(args)
    cloud = read_ply(args.input)
    sample = trainable_kernel_sample_from_cloud(
        cloud,
        slots=args.slots,
        frame_count=args.frames,
        max_points=args.max_points,
        object_id_field=args.object_id_field,
        temporal_offset=args.temporal_offset,
        bind_image_targets=True,
        image_width=args.image_width,
        image_height=args.image_height,
        point_radius=args.point_radius,
        visibility_policy=args.visibility_policy,
        seed=args.seed,
    )
    initial_solver_state = None
    initial_decoder_state = None
    assignment_source = sample.target_source
    if args.resume_checkpoint:
        checkpoint = json.loads(args.resume_checkpoint.read_text(encoding="utf-8"))
        validate_solver_decoder_joint_checkpoint(checkpoint)
        initial_solver_state, initial_decoder_state = solver_decoder_joint_states_from_dict(checkpoint)
        assignment_source = "solver_decoder_joint_checkpoint_resume"
    elif args.solver_checkpoint:
        checkpoint = json.loads(args.solver_checkpoint.read_text(encoding="utf-8"))
        initial_solver_state, assignment_source = _solver_decoder_initial_solver_from_checkpoint(checkpoint)
    if args.run_output_dir:
        result, summary, checkpoint_output = _run_solver_decoder_scaled(
            args,
            sample,
            initial_solver_state=initial_solver_state,
            initial_decoder_state=initial_decoder_state,
            assignment_source=assignment_source,
            record_every=record_every,
        )
    else:
        result = _train_solver_decoder_segment(
            args,
            sample,
            initial_solver_state=initial_solver_state,
            initial_decoder_state=initial_decoder_state,
            iterations=args.iterations,
            record_every=record_every,
        )
        summary = _solver_decoder_summary_from_result(
            args,
            sample,
            result,
            assignment_source=assignment_source,
        )
        checkpoint_output = _solver_decoder_checkpoint_from_result(
            args,
            sample,
            result,
            assignment_source=assignment_source,
        )
    print(f"schema={summary['schema']}")
    print(f"input={args.input}")
    print(f"source_gaussians={sample.source_count}")
    print(f"sampled_gaussians={sample.sampled_count}")
    print(f"frames={summary['frame_count']}")
    print(f"slots={summary['slots']}")
    print(f"iterations={summary['iterations']}")
    print(f"assignment_source={assignment_source}")
    if args.resume_checkpoint:
        print(f"resume_checkpoint={args.resume_checkpoint}")
    if args.run_output_dir:
        print(f"run_output_dir={args.run_output_dir}")
        print(f"training_scale_segments={summary['training_scale']['segment_count']}")
        print(f"training_scale_total_iterations={summary['training_scale']['total_iterations']}")
        print(f"training_scale_checkpoint_every={summary['training_scale']['checkpoint_every']}")
    if "tensorboard" in summary:
        print(f"tensorboard_logdir={summary['tensorboard']['logdir']}")
        print(f"tensorboard_scalar_count={summary['tensorboard']['scalar_count']}")
    print(f"image_renderer={summary['image_renderer']}")
    print(f"gaussian_scale={summary['gaussian_policy']['default_scale']}")
    print(f"gaussian_opacity={summary['gaussian_policy']['default_opacity']}")
    print(f"solver_temperature={summary['final_solver_state']['config']['temperature']}")
    print(f"solver_learning_rate={summary['learning_rates']['solver']}")
    print(f"decoder_learning_rate={summary['learning_rates']['decoder']}")
    print(f"decoder_opacity_learning_rate={summary['learning_rates']['decoder_opacity']}")
    print(f"decoder_scale_learning_rate={summary['learning_rates']['decoder_scale']}")
    print(f"train_solver={str(summary['train_solver']).lower()}")
    print(f"train_decoder_colors={str(summary['train_decoder_colors']).lower()}")
    print(f"train_decoder_opacity={str(summary['train_decoder_opacity']).lower()}")
    print(f"train_decoder_scale={str(summary['train_decoder_scale']).lower()}")
    print(f"initial_total_loss={result.initial_loss.total_loss:.6f}")
    print(f"final_total_loss={result.final_loss.total_loss:.6f}")
    print(f"initial_image_render_loss={result.initial_loss.image_render_loss:.6f}")
    print(f"final_image_render_loss={result.final_loss.image_render_loss:.6f}")
    print(f"initial_object_loss={result.initial_loss.object_loss:.6f}")
    print(f"final_object_loss={result.final_loss.object_loss:.6f}")
    print(f"final_entropy_loss={result.final_loss.entropy_loss:.6f}")
    print(f"final_balance_loss={result.final_loss.balance_loss:.6f}")
    print(f"loss_decreased={str(summary['loss_decreased']).lower()}")
    print(f"image_render_loss_decreased={str(summary['image_render_loss_decreased']).lower()}")
    print(f"object_loss_decreased={str(summary['object_loss_decreased']).lower()}")
    if "run_loss" in summary:
        print(f"run_initial_total_loss={summary['run_loss']['initial_total_loss']:.6f}")
        print(f"run_final_total_loss={summary['run_loss']['final_total_loss']:.6f}")
        print(f"run_loss_decreased={str(summary['run_loss']['loss_decreased']).lower()}")
    assignment_stability = summary.get("assignment_stability")
    if isinstance(assignment_stability, dict):
        after_aggregate = assignment_stability["after"]["aggregate"]
        print(f"assignment_stability_status={assignment_stability['status']}")
        print(f"assignment_stability_before_status={assignment_stability['before_status']}")
        print(f"assignment_stability_after_status={assignment_stability['after_status']}")
        print(f"assignment_stability_degraded={str(assignment_stability['status_degraded']).lower()}")
        print(
            "assignment_stability_after_entropy="
            f"{after_aggregate['max_mean_normalized_entropy']:.6f}"
        )
        print(
            "assignment_stability_after_purity="
            f"{_format_optional_float(after_aggregate['object_purity'])}"
        )
        print(
            "assignment_stability_after_id_stability="
            f"{after_aggregate['id_stability']:.6f}"
        )
    run_assignment_stability = summary.get("run_assignment_stability")
    if isinstance(run_assignment_stability, dict):
        print(f"run_assignment_stability_status={run_assignment_stability['status']}")
        print(f"run_assignment_stability_degraded={str(run_assignment_stability['status_degraded']).lower()}")
    print(f"trained_fields={','.join(summary['trained_fields'])}")
    print(f"frozen_fields={','.join(summary['frozen_fields'])}")
    decoder_opacity = summary.get("decoder_opacity")
    if isinstance(decoder_opacity, dict):
        print(f"decoder_opacity_enabled={str(decoder_opacity['enabled']).lower()}")
        print(f"decoder_opacity_scale_min={_format_optional_float(decoder_opacity['scale_min'])}")
        print(f"decoder_opacity_scale_mean={_format_optional_float(decoder_opacity['scale_mean'])}")
        print(f"decoder_opacity_scale_max={_format_optional_float(decoder_opacity['scale_max'])}")
    decoder_scale = summary.get("decoder_scale")
    if isinstance(decoder_scale, dict):
        print(f"decoder_scale_enabled={str(decoder_scale['enabled']).lower()}")
        print(f"decoder_scale_multiplier_min={_format_optional_float(decoder_scale['multiplier_min'])}")
        print(f"decoder_scale_multiplier_mean={_format_optional_float(decoder_scale['multiplier_mean'])}")
        print(f"decoder_scale_multiplier_max={_format_optional_float(decoder_scale['multiplier_max'])}")
    print(f"gpu_used={str(summary['gpu_policy']['uses_gpu']).lower()}")
    print(f"vram_reserve_gb={summary['gpu_policy']['vram_reserve_gb']}")
    renderer_api = summary.get("renderer_api")
    if isinstance(renderer_api, dict):
        print(f"renderer_api_status={renderer_api['status']}")
        print(f"renderer_name={renderer_api['renderer_name']}")
        print(f"renderer_gradient_path={renderer_api['gradient_path']}")
        print(f"renderer_image_render_loss={renderer_api['image_render_loss']:.6f}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.checkpoint_output:
        write_json(args.checkpoint_output, checkpoint_output)
        print(f"checkpoint={args.checkpoint_output}")
    loss_decreased = summary.get("run_loss", {}).get("loss_decreased", summary["loss_decreased"])
    image_loss_decreased = summary.get("run_loss", {}).get(
        "image_render_loss_decreased",
        summary["image_render_loss_decreased"],
    )
    if args.require_loss_decrease and not loss_decreased:
        raise ValueError("solver-decoder MVP total loss did not decrease")
    if args.require_image_render_loss_decrease and not image_loss_decreased:
        raise ValueError("solver-decoder MVP image render loss did not decrease")
    assignment_gate = summary.get("run_assignment_stability") or summary.get("assignment_stability")
    if (
        args.require_assignment_stability_not_degrade
        and isinstance(assignment_gate, dict)
        and assignment_gate.get("status_degraded")
    ):
        raise ValueError("solver-decoder MVP assignment stability degraded")


def _training_eval_objectstate(args: argparse.Namespace) -> None:
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    validate_solver_decoder_joint_checkpoint(checkpoint)
    cloud = read_ply(args.input)
    sample = trainable_kernel_sample_from_cloud(
        cloud,
        slots=args.slots or _checkpoint_solver_slots(checkpoint),
        frame_count=args.frames or _checkpoint_frame_count(checkpoint),
        max_points=args.max_points or _checkpoint_sampled_gaussians(checkpoint),
        object_id_field=args.object_id_field,
        temporal_offset=args.temporal_offset,
        bind_image_targets=False,
        seed=args.seed,
    )
    summary = evaluate_solver_decoder_object_states(
        sample,
        checkpoint,
        entropy_threshold=args.entropy_threshold,
        purity_threshold=args.purity_threshold,
        collapse_mass_fraction=args.collapse_mass_fraction,
        assignment_confidence_floor=args.assignment_confidence_floor,
        solver_temperature=args.solver_temperature,
    )
    aggregate = summary["aggregate"]
    gates = summary["gates"]
    print(f"schema={summary['schema']}")
    print(f"input={args.input}")
    print(f"checkpoint={args.checkpoint}")
    print(f"sampled_gaussians={sample.sampled_count}")
    print(f"frames={sample.as_dict()['frame_count']}")
    print(f"slots={sample.slots}")
    print(f"solver_step={summary['solver']['step']}")
    print(f"solver_temperature={summary['solver']['temperature']}")
    print(f"decoder_step={summary['decoder']['step']}")
    print(f"eval_status={summary['status']}")
    print(f"mean_normalized_entropy={aggregate['mean_normalized_entropy']:.6f}")
    print(f"max_mean_normalized_entropy={aggregate['max_mean_normalized_entropy']:.6f}")
    print(f"assignment_confidence={aggregate['assignment_confidence']:.6f}")
    print(f"effective_slots={aggregate['effective_slots']:.6f}")
    print(f"max_dominant_slot_mass_fraction={aggregate['max_dominant_slot_mass_fraction']:.6f}")
    print(f"slot_collapse={str(aggregate['slot_collapse']).lower()}")
    print(f"object_purity={_format_optional_float(aggregate['object_purity'])}")
    print(f"temporal_mean_drift={aggregate['temporal_mean_drift']:.6f}")
    print(f"gate_entropy_pass={_format_optional_bool(gates['entropy_pass'])}")
    print(f"gate_entropy_borderline={_format_optional_bool(gates['entropy_borderline'])}")
    print(f"gate_no_collapse_pass={_format_optional_bool(gates['no_collapse_pass'])}")
    print(f"gate_purity_pass={_format_optional_bool(gates['purity_pass'])}")
    print(f"trained_fields={','.join(summary['trained_fields'])}")
    print(f"frozen_fields={','.join(summary['frozen_fields'])}")
    print(f"gpu_used={str(summary['gpu_policy']['uses_gpu']).lower()}")
    print(f"vram_reserve_gb={summary['gpu_policy']['vram_reserve_gb']}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.require_pass and summary["status"] != "objectstate_eval_pass":
        raise ValueError("ObjectState checkpoint eval did not pass")


def _training_real_sample_v2_handoff(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    temperatures = (
        tuple(args.temperature_candidates)
        if args.temperature_candidates is not None
        else (1.0, 0.75, 0.5, 0.35, 0.25)
    )
    report = real_sample_v2_model_handoff_from_cloud(
        cloud,
        sample_source=str(args.input),
        object_id_field=args.object_id_field,
        slots=args.slots,
        frame_count=args.frames,
        max_points=args.max_points,
        temporal_offset=args.temporal_offset,
        image_width=args.image_width,
        image_height=args.image_height,
        point_radius=args.point_radius,
        visibility_policy=args.visibility_policy,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        temperature_candidates=temperatures,
        baseline_temperature=args.baseline_temperature,
        image_renderer=args.image_renderer,
        vram_reserve_gb=args.vram_reserve_gb,
    )
    summary = report.as_dict()
    print(f"schema={summary['schema']}")
    print(f"status={summary['status']}")
    print(f"input={args.input}")
    print(f"source_gaussians={summary['sample']['source_count']}")
    print(f"sampled_gaussians={summary['sample']['sampled_count']}")
    print(f"recommended_solver_temperature={summary['recommended_solver_temperature']}")
    print(f"restore_renderer_joint_status={summary['restore_validation']['renderer_joint_status']}")
    print(f"restore_object_state_status={summary['restore_validation']['object_state_status']}")
    best_metrics = summary["training_effect"]["best_candidate"]["object_state_metrics"]
    print(f"best_entropy={best_metrics['mean_normalized_entropy']:.6f}")
    print(f"best_confidence={best_metrics['assignment_confidence']:.6f}")
    print(f"best_purity={best_metrics['object_purity']:.6f}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.checkpoint_output:
        write_json(args.checkpoint_output, report.checkpoint)
        print(f"checkpoint={args.checkpoint_output}")
    if args.preview_output:
        args.preview_output.parent.mkdir(parents=True, exist_ok=True)
        args.preview_output.write_text(
            render_real_sample_v2_model_handoff_html(summary),
            encoding="utf-8",
        )
        print(f"preview={args.preview_output}")
    if args.require_pass and summary["status"] != "real_sample_v2_model_handoff_pass":
        raise ValueError("real sample v2 model handoff did not pass")


def _training_real_sample_v2_viewer_preview(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    temperatures = (
        tuple(args.temperature_candidates)
        if args.temperature_candidates is not None
        else (1.0, 0.75, 0.5, 0.35, 0.25)
    )
    report = real_sample_v2_viewer_preview_from_cloud(
        cloud,
        sample_source=str(args.input),
        object_id_field=args.object_id_field,
        slots=args.slots,
        frame_count=args.frames,
        max_points=args.max_points,
        temporal_offset=args.temporal_offset,
        image_width=args.image_width,
        image_height=args.image_height,
        point_radius=args.point_radius,
        visibility_policy=args.visibility_policy,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        temperature_candidates=temperatures,
        baseline_temperature=args.baseline_temperature,
        image_renderer=args.image_renderer,
        vram_reserve_gb=args.vram_reserve_gb,
        assignment_feature_weight=args.assignment_feature_weight,
        assignment_position_weight=args.assignment_position_weight,
        rewrite_sh=args.rewrite_sh,
        viewer_path=args.viewer_path,
    )
    summary = report.as_dict()
    quality = summary["quality"]
    weight_policy = summary["assignment_weight_policy"]
    hard_segmentation = summary["projection"]["hard_segmentation"]
    print(f"schema={summary['schema']}")
    print(f"status={summary['status']}")
    print(f"input={args.input}")
    print(f"source_gaussians={summary['source']['source_gaussians']}")
    print(f"projected_gaussians={summary['projection']['projected_gaussians']}")
    print(f"predicted_object_count={summary['projection']['predicted_object_count']}")
    print(f"recommended_solver_temperature={summary['handoff']['recommended_solver_temperature']}")
    print(f"assignment_feature_weight={weight_policy['promoted_feature_weight']}")
    print(f"assignment_position_weight={weight_policy['promoted_position_weight']}")
    print(f"assignment_weight_promotion_applied={str(weight_policy['applied']).lower()}")
    print(f"mixed_gaussians={hard_segmentation['mixed_gaussians']}")
    print(
        "object_id_counts="
        + ",".join(
            f"{item['object_id']}:{item['count']}"
            for item in hard_segmentation["object_id_counts"]
        )
    )
    print(f"quality_status={quality['status']}")
    print(f"full_cloud_entropy={quality['mean_normalized_entropy']:.6f}")
    print(f"full_cloud_confidence={quality['assignment_confidence']:.6f}")
    print(f"full_cloud_purity={_format_optional_float(quality['object_purity'])}")
    print(f"direct_slot_match={quality['direct_slot_match']:.6f}")
    if quality["diagnostics"]:
        print(f"quality_diagnostics={','.join(quality['diagnostics'])}")
    else:
        print("quality_diagnostics=none")
    write_ply(args.preview_ply_output, report.projected_cloud, fmt=_output_format(args))
    print(f"preview_ply={args.preview_ply_output}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if summary["viewer"]["debug_route"]:
        print(f"viewer_route={summary['viewer']['debug_route']}")
    if args.require_pass and summary["status"] != "real_sample_v2_viewer_preview_pass":
        raise ValueError("real sample v2 viewer preview did not pass")


def _training_real_sample_v2_full_cloud_purity(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    temperatures = (
        tuple(args.temperature_candidates)
        if args.temperature_candidates is not None
        else (1.0, 0.75, 0.5, 0.35, 0.25)
    )
    report = real_sample_v2_full_cloud_purity_from_cloud(
        cloud,
        sample_source=str(args.input),
        object_id_field=args.object_id_field,
        slots=args.slots,
        max_point_candidates=tuple(args.max_point_candidates),
        frame_count=args.frames,
        temporal_offset=args.temporal_offset,
        image_width=args.image_width,
        image_height=args.image_height,
        point_radius=args.point_radius,
        visibility_policy=args.visibility_policy,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        temperature_candidates=temperatures,
        baseline_temperature=args.baseline_temperature,
        image_renderer=args.image_renderer,
        vram_reserve_gb=args.vram_reserve_gb,
        rewrite_sh=args.rewrite_sh,
        viewer_path=args.viewer_path,
    )
    summary = report.as_dict()
    best = summary["best_candidate"]
    quality = best["quality"]
    delta = summary["quality_delta"]
    recommendation = summary["recommendation"]
    print(f"schema={summary['schema']}")
    print(f"status={summary['status']}")
    print(f"input={args.input}")
    print(f"source_gaussians={summary['source']['source_gaussians']}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"selected_max_points={summary['segmentation_target']['selected_max_points']}")
    print(f"selected_solver_temperature={summary['segmentation_target']['selected_solver_temperature']}")
    print(f"best_quality_status={quality['status']}")
    print(f"best_full_cloud_entropy={quality['mean_normalized_entropy']:.6f}")
    print(f"best_full_cloud_confidence={quality['assignment_confidence']:.6f}")
    print(f"best_full_cloud_purity={_format_optional_float(quality['object_purity'])}")
    print(f"best_direct_slot_match={quality['direct_slot_match']:.6f}")
    print(f"purity_delta={_format_optional_float(delta['purity_delta'])}")
    print(f"direct_slot_match_delta={delta['direct_slot_match_delta']:.6f}")
    print(f"recommendation_decision={recommendation['decision']}")
    print(f"recommendation_action={recommendation['action']}")
    print(f"recommendation_max_points={recommendation['max_points']}")
    for candidate in summary["coverage_sweep"]:
        candidate_quality = candidate["quality"]
        print(
            "candidate="
            f"max_points:{candidate['max_points']},"
            f"temperature:{candidate['solver_temperature']},"
            f"status:{candidate_quality['status']},"
            f"entropy:{candidate_quality['mean_normalized_entropy']:.6f},"
            f"confidence:{candidate_quality['assignment_confidence']:.6f},"
            f"purity:{_format_optional_float(candidate_quality['object_purity'])},"
            f"direct_slot_match:{candidate_quality['direct_slot_match']:.6f},"
            f"diagnostics:{','.join(candidate_quality['diagnostics']) or 'none'}"
        )
    write_ply(args.preview_ply_output, report.best_candidate.projected_cloud, fmt=_output_format(args))
    print(f"preview_ply={args.preview_ply_output}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if summary["viewer"]["debug_route"]:
        print(f"viewer_route={summary['viewer']['debug_route']}")
    if args.require_pass and summary["status"] != "real_sample_v2_full_cloud_purity_pass":
        raise ValueError("real sample v2 full-cloud purity did not pass")


def _training_real_sample_v2_segmentation_quality(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    temperatures = (
        tuple(args.temperature_candidates)
        if args.temperature_candidates is not None
        else (1.0, 0.75, 0.5, 0.35, 0.25)
    )
    report = real_sample_v2_segmentation_quality_from_cloud(
        cloud,
        sample_source=str(args.input),
        object_id_field=args.object_id_field,
        slots=args.slots,
        max_points=args.max_points,
        frame_count=args.frames,
        temporal_offset=args.temporal_offset,
        image_width=args.image_width,
        image_height=args.image_height,
        point_radius=args.point_radius,
        visibility_policy=args.visibility_policy,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        temperature_candidates=temperatures,
        baseline_temperature=args.baseline_temperature,
        image_renderer=args.image_renderer,
        vram_reserve_gb=args.vram_reserve_gb,
        rewrite_sh=args.rewrite_sh,
        viewer_path=args.viewer_path,
    )
    summary = report.as_dict()
    quality = summary["global_quality"]
    recommendation = summary["recommendation"]
    print(f"schema={summary['schema']}")
    print(f"status={summary['status']}")
    print(f"input={args.input}")
    print(f"source_gaussians={summary['source']['source_gaussians']}")
    print(f"max_points={summary['segmentation_target']['max_points']}")
    print(f"solver_temperature={summary['segmentation_target']['solver_temperature']}")
    print(f"direct_slot_match={quality['direct_slot_match']:.6f}")
    print(f"hard_argmax_object_purity={quality['hard_argmax_object_purity']:.6f}")
    print(f"min_predicted_object_purity={quality['min_predicted_object_purity']:.6f}")
    print(f"min_target_recall={quality['min_target_recall']:.6f}")
    print(f"mixed_gaussians={quality['mixed_gaussians']}")
    print(f"quality_diagnostics={','.join(quality['diagnostics']) or 'none'}")
    for row in summary["confusion"]["rows"]:
        counts = ",".join(
            f"{item['object_id']}:{item['count']}" for item in row["predicted_counts"]
        )
        print(f"confusion_row=target_slot:{row['target_slot']},counts:{counts},total:{row['total']}")
    for predicted in summary["per_predicted_object"]:
        print(
            "predicted_object="
            f"id:{predicted['object_id']},"
            f"count:{predicted['gaussian_count']},"
            f"purity:{predicted['purity']:.6f},"
            f"mixed:{predicted['mixed_count']},"
            f"confidence_mean:{predicted['confidence']['mean']:.6f},"
            f"entropy_mean:{predicted['entropy']['mean']:.6f},"
            f"diagnostics:{','.join(predicted['diagnostics']) or 'none'}"
        )
    print(f"recommendation_decision={recommendation['decision']}")
    print(f"recommendation_action={recommendation['action']}")
    print(f"weak_target_slots={','.join(str(value) for value in recommendation['weak_target_slots']) or 'none'}")
    print(
        "mixed_predicted_objects="
        f"{','.join(str(value) for value in recommendation['mixed_predicted_objects']) or 'none'}"
    )
    write_ply(args.preview_ply_output, report.projected_cloud, fmt=_output_format(args))
    print(f"preview_ply={args.preview_ply_output}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if summary["viewer"]["debug_route"]:
        print(f"viewer_route={summary['viewer']['debug_route']}")
    if args.require_pass and summary["status"] != "real_sample_v2_segmentation_quality_pass":
        raise ValueError("real sample v2 segmentation quality did not pass")


def _training_real_sample_v2_weak_boundary_opt(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    report = real_sample_v2_weak_boundary_opt_from_cloud(
        cloud,
        sample_source=str(args.input),
        object_id_field=args.object_id_field,
        slots=args.slots,
        max_points=args.max_points,
        solver_temperature=args.solver_temperature,
        candidate_feature_weight=args.candidate_feature_weight,
        candidate_position_weight=args.candidate_position_weight,
        frame_count=args.frames,
        temporal_offset=args.temporal_offset,
        image_width=args.image_width,
        image_height=args.image_height,
        point_radius=args.point_radius,
        visibility_policy=args.visibility_policy,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        baseline_temperature=args.baseline_temperature,
        image_renderer=args.image_renderer,
        vram_reserve_gb=args.vram_reserve_gb,
        rewrite_sh=args.rewrite_sh,
        viewer_path=args.viewer_path,
    )
    summary = report.as_dict()
    baseline_quality = summary["baseline"]["global_quality"]
    candidate_quality = summary["candidate"]["global_quality"]
    delta = summary["quality_delta"]
    changed = summary["changed_gaussians"]
    recommendation = summary["recommendation"]
    print(f"schema={summary['schema']}")
    print(f"status={summary['status']}")
    print(f"input={args.input}")
    print(f"source_gaussians={summary['source']['source_gaussians']}")
    print(f"max_points={summary['fixed_target']['max_points']}")
    print(f"solver_temperature={summary['fixed_target']['solver_temperature']}")
    print(f"candidate_feature_weight={summary['candidate_policy']['feature_weight']}")
    print(f"candidate_position_weight={summary['candidate_policy']['position_weight']}")
    print(f"baseline_direct_slot_match={baseline_quality['direct_slot_match']:.6f}")
    print(f"candidate_direct_slot_match={candidate_quality['direct_slot_match']:.6f}")
    print(f"direct_slot_match_delta={delta['direct_slot_match_delta']:.6f}")
    print(f"baseline_mixed_gaussians={baseline_quality['mixed_gaussians']}")
    print(f"candidate_mixed_gaussians={candidate_quality['mixed_gaussians']}")
    print(f"mixed_gaussians_delta={delta['mixed_gaussians_delta']}")
    print(f"baseline_min_target_recall={baseline_quality['min_target_recall']:.6f}")
    print(f"candidate_min_target_recall={candidate_quality['min_target_recall']:.6f}")
    print(f"min_target_recall_delta={delta['min_target_recall_delta']:.6f}")
    print(f"changed_gaussians={changed['changed_count']}")
    for pair in changed["pairs"]:
        print(
            "changed_pair="
            f"baseline:{pair['baseline_object_id']},"
            f"candidate:{pair['candidate_object_id']},"
            f"count:{pair['count']}"
        )
    print(f"recommendation_decision={recommendation['decision']}")
    print(f"recommendation_action={recommendation['action']}")
    write_ply(args.preview_ply_output, report.candidate_cloud, fmt=_output_format(args))
    print(f"preview_ply={args.preview_ply_output}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if summary["viewer"]["debug_route"]:
        print(f"viewer_route={summary['viewer']['debug_route']}")
    if args.require_pass and summary["status"] != "real_sample_v2_weak_boundary_opt_pass":
        raise ValueError("real sample v2 weak-boundary optimization did not pass")


def _training_real_sample_v2_promoted_weights_cross_sample(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    report = real_sample_v2_promoted_weights_cross_sample_from_cloud(
        cloud,
        sample_source=str(args.input),
        object_id_field=args.object_id_field,
        slots=args.slots,
        max_points=args.max_points,
        solver_temperature=args.solver_temperature,
        baseline_feature_weight=args.baseline_feature_weight,
        baseline_position_weight=args.baseline_position_weight,
        promoted_feature_weight=args.promoted_feature_weight,
        promoted_position_weight=args.promoted_position_weight,
        frame_count=args.frames,
        temporal_offset=args.temporal_offset,
        image_width=args.image_width,
        image_height=args.image_height,
        point_radius=args.point_radius,
        visibility_policy=args.visibility_policy,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        baseline_temperature=args.baseline_temperature,
        image_renderer=args.image_renderer,
        vram_reserve_gb=args.vram_reserve_gb,
        rewrite_sh=args.rewrite_sh,
        viewer_path=args.viewer_path,
        reference_sample=args.reference_sample,
    )
    summary = report.as_dict()
    baseline_quality = summary["baseline"]["quality"]
    baseline_hard = summary["baseline"]["projection"]["hard_segmentation"]
    promoted_quality = summary["promoted"]["quality"]
    promoted_hard = summary["promoted"]["projection"]["hard_segmentation"]
    delta = summary["quality_delta"]
    changed = summary["changed_gaussians"]
    recommendation = summary["recommendation"]
    print(f"schema={summary['schema']}")
    print(f"status={summary['status']}")
    print(f"input={args.input}")
    print(f"source_gaussians={summary['source']['source_gaussians']}")
    print(f"max_points={summary['fixed_target']['max_points']}")
    print(f"solver_temperature={summary['fixed_target']['solver_temperature']}")
    print(f"baseline_feature_weight={summary['promotion_policy']['baseline_feature_weight']}")
    print(f"baseline_position_weight={summary['promotion_policy']['baseline_position_weight']}")
    print(f"promoted_feature_weight={summary['promotion_policy']['promoted_feature_weight']}")
    print(f"promoted_position_weight={summary['promotion_policy']['promoted_position_weight']}")
    print(f"baseline_mixed_gaussians={baseline_hard['mixed_gaussians']}")
    print(f"promoted_mixed_gaussians={promoted_hard['mixed_gaussians']}")
    print(f"mixed_gaussians_delta={delta['mixed_gaussians_delta']}")
    print(f"baseline_direct_slot_match={baseline_quality['direct_slot_match']:.6f}")
    print(f"promoted_direct_slot_match={promoted_quality['direct_slot_match']:.6f}")
    print(f"direct_slot_match_delta={delta['direct_slot_match_delta']:.6f}")
    print(f"baseline_object_purity={_format_optional_float(baseline_quality['object_purity'])}")
    print(f"promoted_object_purity={_format_optional_float(promoted_quality['object_purity'])}")
    print(f"object_purity_delta={_format_optional_float(delta['object_purity_delta'])}")
    print(f"baseline_confidence={baseline_quality['assignment_confidence']:.6f}")
    print(f"promoted_confidence={promoted_quality['assignment_confidence']:.6f}")
    print(f"assignment_confidence_delta={delta['assignment_confidence_delta']:.6f}")
    print(f"baseline_entropy={baseline_quality['mean_normalized_entropy']:.6f}")
    print(f"promoted_entropy={promoted_quality['mean_normalized_entropy']:.6f}")
    print(f"mean_normalized_entropy_delta={delta['mean_normalized_entropy_delta']:.6f}")
    print(f"changed_gaussians={changed['changed_count']}")
    print(f"hard_fix_count={changed['hard_fix_count']}")
    print(f"hard_regression_count={changed['hard_regression_count']}")
    for pair in changed["pairs"]:
        print(
            "changed_pair="
            f"baseline:{pair['baseline_object_id']},"
            f"promoted:{pair['promoted_object_id']},"
            f"count:{pair['count']}"
        )
    print(f"recommendation_decision={recommendation['decision']}")
    print(f"recommendation_action={recommendation['action']}")
    write_ply(args.preview_ply_output, report.promoted_cloud, fmt=_output_format(args))
    print(f"preview_ply={args.preview_ply_output}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if summary["viewer"]["debug_route"]:
        print(f"viewer_route={summary['viewer']['debug_route']}")
    if (
        args.require_pass
        and summary["status"] != "real_sample_v2_promoted_weights_cross_sample_pass"
    ):
        raise ValueError("real sample v2 promoted weights cross-sample check did not pass")


def _training_real_sample_v2_sample_aware_weight_policy(args: argparse.Namespace) -> None:
    cloud = read_ply(args.input)
    report = real_sample_v2_sample_aware_weight_policy_from_cloud(
        cloud,
        sample_source=str(args.input),
        object_id_field=args.object_id_field,
        slots=args.slots,
        max_points=args.max_points,
        solver_temperature=args.solver_temperature,
        baseline_feature_weight=args.baseline_feature_weight,
        baseline_position_weight=args.baseline_position_weight,
        promoted_feature_weight=args.promoted_feature_weight,
        promoted_position_weight=args.promoted_position_weight,
        frame_count=args.frames,
        temporal_offset=args.temporal_offset,
        image_width=args.image_width,
        image_height=args.image_height,
        point_radius=args.point_radius,
        visibility_policy=args.visibility_policy,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        baseline_temperature=args.baseline_temperature,
        image_renderer=args.image_renderer,
        vram_reserve_gb=args.vram_reserve_gb,
        rewrite_sh=args.rewrite_sh,
        viewer_path=args.viewer_path,
    )
    summary = report.as_dict()
    selected = summary["selected_policy"]
    evidence_gate = summary["evidence_normalization_gate"]
    print(f"schema={summary['schema']}")
    print(f"status={summary['status']}")
    print(f"input={args.input}")
    print(f"source_gaussians={summary['source']['source_gaussians']}")
    print(f"max_points={summary['fixed_target']['max_points']}")
    print(f"solver_temperature={summary['fixed_target']['solver_temperature']}")
    print(f"selected_candidate={selected['candidate_name']}")
    print(f"selected_feature_weight={selected['feature_weight']}")
    print(f"selected_position_weight={selected['position_weight']}")
    print(f"selection_reason={selected['selection_reason']}")
    print(f"evidence_normalization_status={evidence_gate['status']}")
    print(
        "evidence_normalization_required="
        f"{str(evidence_gate['requires_evidence_normalization']).lower()}"
    )
    for candidate in summary["candidates"]:
        metrics = candidate["metrics"]
        delta = candidate["delta_vs_baseline"]
        gate = candidate["sample_policy_gate"]
        print(
            "candidate="
            f"name:{candidate['candidate']['name']},"
            f"feature_weight:{candidate['candidate']['feature_weight']},"
            f"position_weight:{candidate['candidate']['position_weight']},"
            f"eligible:{str(gate['eligible_for_sample']).lower()},"
            f"mixed:{metrics['mixed_gaussians']},"
            f"direct:{metrics['direct_slot_match']:.6f},"
            f"purity:{_format_optional_float(metrics['object_purity'])},"
            f"confidence:{metrics['assignment_confidence']:.6f},"
            f"entropy:{metrics['mean_normalized_entropy']:.6f},"
            f"mixed_delta:{delta['mixed_gaussians_delta']},"
            f"direct_delta:{delta['direct_slot_match_delta']:.6f},"
            f"confidence_delta:{delta['assignment_confidence_delta']:.6f},"
            f"hard_fix:{gate['hard_fix_count']},"
            f"hard_regression:{gate['hard_regression_count']}"
        )
    write_ply(args.preview_ply_output, report.selected_cloud, fmt=_output_format(args))
    print(f"preview_ply={args.preview_ply_output}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if summary["viewer"]["debug_route"]:
        print(f"viewer_route={summary['viewer']['debug_route']}")
    if args.require_pass and summary["status"] != "real_sample_v2_sample_aware_weight_policy_pass":
        raise ValueError("real sample v2 sample-aware weight policy did not pass")


def _training_real_sample_v2_bounded_normalization_cross_sample(
    args: argparse.Namespace,
) -> None:
    sample_ids = list(args.sample_ids or [])
    viewer_paths = list(args.viewer_paths or [])
    if sample_ids and len(sample_ids) != len(args.inputs):
        raise ValueError("--sample-id must be provided once per input when used")
    if viewer_paths and len(viewer_paths) != len(args.inputs):
        raise ValueError("--viewer-path must be provided once per input when used")

    samples = []
    for index, input_path in enumerate(args.inputs):
        sample_id = sample_ids[index] if sample_ids else input_path.stem
        viewer_path = viewer_paths[index] if viewer_paths else None
        samples.append(
            RealSampleV2BoundedNormalizationCrossSampleInput(
                sample_id=sample_id,
                cloud=read_ply(input_path),
                sample_source=str(input_path),
                object_id_field=args.object_id_field,
                slots=args.slots,
                viewer_path=viewer_path,
            )
        )

    report = real_sample_v2_bounded_normalization_cross_sample_from_clouds(
        samples,
        min_samples=args.min_samples,
        max_points=args.max_points,
        solver_temperature=args.solver_temperature,
        baseline_feature_weight=args.baseline_feature_weight,
        baseline_position_weight=args.baseline_position_weight,
        promoted_feature_weight=args.promoted_feature_weight,
        promoted_position_weight=args.promoted_position_weight,
        frame_count=args.frames,
        temporal_offset=args.temporal_offset,
        image_width=args.image_width,
        image_height=args.image_height,
        point_radius=args.point_radius,
        visibility_policy=args.visibility_policy,
        seed=args.seed,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        baseline_temperature=args.baseline_temperature,
        image_renderer=args.image_renderer,
        vram_reserve_gb=args.vram_reserve_gb,
        rewrite_sh=args.rewrite_sh,
    )
    summary = report.as_dict()
    aggregate = summary["aggregate"]
    recommendation = summary["recommendation"]
    print(f"schema={summary['schema']}")
    print(f"status={summary['status']}")
    print(f"sample_count={summary['sample_count']}")
    print(f"min_samples={summary['min_samples']}")
    print(f"aggregate_result={aggregate['result']}")
    print(f"selected_policy_counts={aggregate['selected_policy_counts']}")
    print(f"blocked_promoted_sample_count={aggregate['blocked_promoted_sample_count']}")
    print(f"selected_hard_regression_count={aggregate['selected_hard_regression_count']}")
    print(f"recommendation_decision={recommendation['decision']}")
    print(f"recommendation_action={recommendation['action']}")
    for row in summary["rows"]:
        selected = row["selected_policy"]
        metrics = row["selected_metrics"]
        changed = row["selected_changed_gaussians"]
        promoted_gate = row["promoted_candidate"]["sample_policy_gate"]
        print(
            "sample="
            f"id:{row['sample_id']},"
            f"source_gaussians:{row['source']['source_gaussians']},"
            f"selected:{selected['candidate_name']},"
            f"feature_weight:{selected['feature_weight']},"
            f"position_weight:{selected['position_weight']},"
            f"mixed:{metrics['mixed_gaussians']},"
            f"direct:{metrics['direct_slot_match']:.6f},"
            f"purity:{_format_optional_float(metrics['object_purity'])},"
            f"hard_fix:{changed['hard_fix_count']},"
            f"hard_regression:{changed['hard_regression_count']},"
            f"promoted_eligible:{str(promoted_gate['eligible_for_sample']).lower()},"
            f"promoted_hard_regression:{promoted_gate['hard_regression_count']},"
            f"evidence_status:{row['evidence_normalization_status']}"
        )
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.require_pass and summary["status"] != "real_sample_v2_bounded_normalization_cross_sample_pass":
        raise ValueError("real sample v2 bounded normalization cross-sample gate did not pass")


def _training_eval_assignment(args: argparse.Namespace) -> None:
    checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
    cloud = read_ply(args.input)
    sample = trainable_kernel_sample_from_cloud(
        cloud,
        slots=args.slots or _checkpoint_solver_slots(checkpoint),
        frame_count=args.frames,
        max_points=args.max_points or _checkpoint_sampled_gaussians(checkpoint),
        object_id_field=args.object_id_field,
        temporal_offset=args.temporal_offset,
        bind_image_targets=False,
        seed=args.seed,
    )
    evidence_batches = assignment_evidence_sequence_from_trainable_frames(
        sample.frames,
        source="eval_assignment_trainable_frame",
    )
    summary = evaluate_assignment_stability(
        evidence_batches,
        checkpoint,
        entropy_threshold=args.entropy_threshold,
        purity_threshold=args.purity_threshold,
        collapse_mass_fraction=args.collapse_mass_fraction,
        assignment_confidence_floor=args.assignment_confidence_floor,
        id_stability_threshold=args.id_stability_threshold,
        temporal_drift_threshold=args.temporal_drift_threshold,
        solver_temperature=args.solver_temperature,
    )
    aggregate = summary["aggregate"]
    gates = summary["gates"]
    print(f"schema={summary['schema']}")
    print(f"input={args.input}")
    print(f"checkpoint={args.checkpoint}")
    print(f"sampled_gaussians={sample.sampled_count}")
    print(f"frames={sample.as_dict()['frame_count']}")
    print(f"slots={sample.slots}")
    print(f"solver_step={summary['solver']['step']}")
    print(f"solver_temperature={summary['solver']['temperature']}")
    print(f"eval_status={summary['status']}")
    print(f"mean_normalized_entropy={aggregate['mean_normalized_entropy']:.6f}")
    print(f"max_mean_normalized_entropy={aggregate['max_mean_normalized_entropy']:.6f}")
    print(f"assignment_confidence={aggregate['assignment_confidence']:.6f}")
    print(f"effective_slots={aggregate['effective_slots']:.6f}")
    print(f"max_dominant_slot_mass_fraction={aggregate['max_dominant_slot_mass_fraction']:.6f}")
    print(f"slot_collapse={str(aggregate['slot_collapse']).lower()}")
    print(f"object_purity={_format_optional_float(aggregate['object_purity'])}")
    print(f"temporal_mean_drift={aggregate['temporal_mean_drift']:.6f}")
    print(f"temporal_max_drift={aggregate['temporal_max_drift']:.6f}")
    print(f"id_stability={aggregate['id_stability']:.6f}")
    dynamic_k = summary["dynamic_k"]
    print(f"dynamic_k_mode={dynamic_k['mode']}")
    print(f"dynamic_k_auto_update={str(dynamic_k['auto_update']).lower()}")
    print(f"dynamic_k_proposal_count={dynamic_k['proposal_count']}")
    print(f"dynamic_k_proposal_kinds={','.join(dynamic_k['proposal_kinds'])}")
    print(f"gate_entropy_pass={_format_optional_bool(gates['entropy_pass'])}")
    print(f"gate_entropy_borderline={_format_optional_bool(gates['entropy_borderline'])}")
    print(f"gate_no_collapse_pass={_format_optional_bool(gates['no_collapse_pass'])}")
    print(f"gate_purity_pass={_format_optional_bool(gates['purity_pass'])}")
    print(f"gate_id_stability_pass={_format_optional_bool(gates['id_stability_pass'])}")
    print(f"gate_temporal_drift_pass={_format_optional_bool(gates['temporal_drift_pass'])}")
    print(f"gpu_used={str(summary['gpu_policy']['uses_gpu']).lower()}")
    print(f"vram_reserve_gb={summary['gpu_policy']['vram_reserve_gb']}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.require_pass and summary["status"] != "assignment_stability_eval_pass":
        raise ValueError("assignment stability eval did not pass")


def _checkpoint_solver_slots(checkpoint: dict[str, object]) -> int:
    solver = checkpoint.get("solver_state")
    if not isinstance(solver, dict):
        raise ValueError("checkpoint missing solver_state")
    config = solver.get("config")
    if not isinstance(config, dict) or "slots" not in config:
        raise ValueError("checkpoint solver_state missing config.slots")
    return int(config["slots"])


def _checkpoint_frame_count(checkpoint: dict[str, object]) -> int:
    contract = checkpoint.get("image_target_contract")
    if isinstance(contract, dict) and contract.get("frame_count") is not None:
        return int(contract["frame_count"])
    return 2


def _checkpoint_sampled_gaussians(checkpoint: dict[str, object]) -> int | None:
    source = checkpoint.get("source")
    if isinstance(source, dict) and source.get("sampled_gaussians") is not None:
        return int(source["sampled_gaussians"])
    return None


def _format_optional_float(value: object) -> str:
    return "none" if value is None else f"{float(value):.6f}"


def _format_optional_bool(value: object) -> str:
    return "none" if value is None else str(bool(value)).lower()


def _solver_decoder_record_every(args: argparse.Namespace) -> int | None:
    if args.loss_log_every is not None and args.record_every is not None:
        if int(args.loss_log_every) != int(args.record_every):
            raise ValueError("--loss-log-every and --record-every must match when both are provided")
    return args.loss_log_every if args.loss_log_every is not None else args.record_every


def _train_solver_decoder_segment(
    args: argparse.Namespace,
    sample,
    *,
    initial_solver_state,
    initial_decoder_state,
    iterations: int,
    record_every: int | None,
):
    return train_solver_decoder_joint(
        sample.frames,
        slots=sample.slots,
        initial_solver_state=initial_solver_state,
        initial_decoder_state=initial_decoder_state,
        iterations=iterations,
        solver_learning_rate=args.solver_learning_rate,
        decoder_learning_rate=args.decoder_learning_rate,
        train_solver=not args.freeze_solver,
        train_decoder_colors=not args.freeze_decoder_colors,
        train_decoder_opacity=args.train_decoder_opacity and not args.freeze_decoder_opacity,
        decoder_opacity_learning_rate=args.decoder_opacity_learning_rate,
        decoder_opacity_init_logit=args.decoder_opacity_init_logit,
        train_decoder_scale=args.train_decoder_scale and not args.freeze_decoder_scale,
        decoder_scale_learning_rate=args.decoder_scale_learning_rate,
        decoder_scale_init_log_offset=args.decoder_scale_init_log_offset,
        image_render_weight=args.image_render_weight,
        object_weight=args.object_weight,
        entropy_weight=args.entropy_weight,
        balance_weight=args.balance_weight,
        temporal_weight=args.temporal_weight,
        image_renderer=args.image_renderer,
        gaussian_scale=args.gaussian_scale,
        gaussian_opacity=args.gaussian_opacity,
        solver_temperature=args.solver_temperature,
        seed=args.seed,
        record_every=record_every,
        vram_reserve_gb=args.vram_reserve_gb,
    )


def _solver_decoder_summary_from_result(
    args: argparse.Namespace,
    sample,
    result,
    *,
    assignment_source: str,
    training_scale: dict[str, object] | None = None,
) -> dict[str, object]:
    summary = {
        **result.as_dict(
            include_weights=args.include_weights,
            include_assignments=args.include_assignments,
        ),
        "input": str(args.input),
        "sample": sample.as_dict(),
        "assignment_source": assignment_source,
        "solver_checkpoint": str(args.solver_checkpoint) if args.solver_checkpoint else None,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
    }
    summary["assignment_stability"] = _solver_decoder_assignment_stability_gate(
        sample,
        before_solver_state=result.initial_solver_state,
        after_solver_state=result.final_solver_state,
    )
    if training_scale is not None:
        summary["training_scale"] = training_scale
    return summary


def _solver_decoder_assignment_stability_gate(
    sample,
    *,
    before_solver_state,
    after_solver_state,
) -> dict[str, object]:
    evidence_batches = assignment_evidence_sequence_from_trainable_frames(
        sample.frames,
        source="solver_decoder_assignment_stability_gate",
    )
    before = evaluate_assignment_stability(evidence_batches, before_solver_state)
    after = evaluate_assignment_stability(evidence_batches, after_solver_state)
    before_rank = _assignment_stability_status_rank(before["status"])
    after_rank = _assignment_stability_status_rank(after["status"])
    degraded = after_rank < before_rank
    return {
        "schema": "objgauss-solver-decoder-assignment-stability-gate-v1",
        "kind": "solver_decoder_assignment_stability_gate",
        "status": "assignment_stability_gate_degraded"
        if degraded
        else "assignment_stability_gate_ok",
        "status_degraded": bool(degraded),
        "before_status": before["status"],
        "after_status": after["status"],
        "deltas": _assignment_stability_deltas(before, after),
        "before": before,
        "after": after,
    }


def _assignment_stability_status_rank(status: object) -> int:
    return {
        "assignment_stability_eval_pass": 2,
        "assignment_stability_eval_borderline": 1,
        "assignment_stability_eval_fail": 0,
    }.get(str(status), -1)


def _assignment_stability_deltas(
    before: dict[str, object],
    after: dict[str, object],
) -> dict[str, object]:
    before_aggregate = before["aggregate"]
    after_aggregate = after["aggregate"]
    return {
        "max_mean_normalized_entropy": (
            float(after_aggregate["max_mean_normalized_entropy"])
            - float(before_aggregate["max_mean_normalized_entropy"])
        ),
        "assignment_confidence": (
            float(after_aggregate["assignment_confidence"])
            - float(before_aggregate["assignment_confidence"])
        ),
        "object_purity": _optional_float_delta(
            before_aggregate.get("object_purity"),
            after_aggregate.get("object_purity"),
        ),
        "id_stability": (
            float(after_aggregate["id_stability"])
            - float(before_aggregate["id_stability"])
        ),
        "temporal_max_drift": (
            float(after_aggregate["temporal_max_drift"])
            - float(before_aggregate["temporal_max_drift"])
        ),
    }


def _optional_float_delta(before: object, after: object) -> float | None:
    if before is None or after is None:
        return None
    return float(after) - float(before)


def _solver_decoder_checkpoint_from_result(
    args: argparse.Namespace,
    sample,
    result,
    *,
    assignment_source: str,
    resume_checkpoint: str | None = None,
) -> dict[str, object]:
    return solver_decoder_joint_checkpoint(
        result,
        input_path=str(args.input),
        source_gaussians=sample.source_count,
        sampled_gaussians=sample.sampled_count,
        target_source=sample.target_source,
        assignment_source=assignment_source,
        object_id_mapping=sample.object_id_mapping,
        solver_checkpoint=str(args.solver_checkpoint) if args.solver_checkpoint else None,
        resume_checkpoint=resume_checkpoint or (str(args.resume_checkpoint) if args.resume_checkpoint else None),
        vram_reserve_gb=args.vram_reserve_gb,
    )


def _solver_decoder_opacity_segment_metrics(result) -> dict[str, object]:
    summary = result.as_dict()
    decoder_opacity = summary.get("decoder_opacity")
    if not isinstance(decoder_opacity, dict) or not decoder_opacity.get("enabled"):
        return {}
    return {
        "final_decoder_opacity_scale_min": decoder_opacity["scale_min"],
        "final_decoder_opacity_scale_mean": decoder_opacity["scale_mean"],
        "final_decoder_opacity_scale_max": decoder_opacity["scale_max"],
    }


def _solver_decoder_scale_segment_metrics(result) -> dict[str, object]:
    summary = result.as_dict()
    decoder_scale = summary.get("decoder_scale")
    if not isinstance(decoder_scale, dict) or not decoder_scale.get("enabled"):
        return {}
    return {
        "final_decoder_scale_multiplier_min": decoder_scale["multiplier_min"],
        "final_decoder_scale_multiplier_mean": decoder_scale["multiplier_mean"],
        "final_decoder_scale_multiplier_max": decoder_scale["multiplier_max"],
    }


def _run_solver_decoder_scaled(
    args: argparse.Namespace,
    sample,
    *,
    initial_solver_state,
    initial_decoder_state,
    assignment_source: str,
    record_every: int | None,
):
    plan = solver_decoder_training_scale_plan(
        total_iterations=args.iterations,
        checkpoint_every=args.checkpoint_every,
        loss_log_every=record_every,
        output_dir=args.run_output_dir,
        image_renderer=args.image_renderer,
        vram_reserve_gb=args.vram_reserve_gb,
    )
    write_json(Path(plan["outputs"]["plan"]), plan)
    current_solver = initial_solver_state
    current_decoder = initial_decoder_state
    segment_records: list[dict[str, object]] = []
    first_result = None
    final_result = None
    final_summary = None
    final_checkpoint = None
    previous_checkpoint_path = str(args.resume_checkpoint) if args.resume_checkpoint else None
    for segment in plan["segments"]:
        segment_source = (
            assignment_source if not segment_records else "solver_decoder_joint_checkpoint_segment_resume"
        )
        result = _train_solver_decoder_segment(
            args,
            sample,
            initial_solver_state=current_solver,
            initial_decoder_state=current_decoder,
            iterations=int(segment["iterations"]),
            record_every=record_every,
        )
        summary = _solver_decoder_summary_from_result(
            args,
            sample,
            result,
            assignment_source=segment_source,
            training_scale={
                "plan_schema": plan["schema"],
                "mode": "segmented",
                "segment_id": segment["segment_id"],
                "segment_index": segment["index"],
                "segment_count": plan["segment_count"],
                "total_iterations": plan["total_iterations"],
                "checkpoint_every": plan["checkpoint_every"],
                "loss_log_every": plan["loss_log_every"],
                "run_output_dir": plan["outputs"]["root"],
            },
        )
        checkpoint = _solver_decoder_checkpoint_from_result(
            args,
            sample,
            result,
            assignment_source=segment_source,
            resume_checkpoint=previous_checkpoint_path,
        )
        write_json(Path(segment["summary_path"]), summary)
        write_json(Path(segment["checkpoint_path"]), checkpoint)
        segment_records.append(
            {
                "segment_id": segment["segment_id"],
                "start_iteration": segment["start_iteration"],
                "end_iteration": segment["end_iteration"],
                "iterations": segment["iterations"],
                "summary_path": segment["summary_path"],
                "checkpoint_path": segment["checkpoint_path"],
                "initial_total_loss": result.initial_loss.total_loss,
                "final_total_loss": result.final_loss.total_loss,
                "initial_image_render_loss": result.initial_loss.image_render_loss,
                "final_image_render_loss": result.final_loss.image_render_loss,
                "initial_object_loss": result.initial_loss.object_loss,
                "final_object_loss": result.final_loss.object_loss,
                "final_entropy_loss": result.final_loss.entropy_loss,
                "final_balance_loss": result.final_loss.balance_loss,
                **_solver_decoder_opacity_segment_metrics(result),
                **_solver_decoder_scale_segment_metrics(result),
            }
        )
        first_result = first_result or result
        final_result = result
        final_summary = summary
        final_checkpoint = checkpoint
        current_solver = result.final_solver_state
        current_decoder = result.final_decoder_state
        previous_checkpoint_path = str(segment["checkpoint_path"])
    if first_result is None or final_result is None or final_summary is None or final_checkpoint is None:
        raise ValueError("scaled solver-decoder training did not produce any segments")
    run_loss = {
        "initial_total_loss": first_result.initial_loss.total_loss,
        "final_total_loss": final_result.final_loss.total_loss,
        "initial_image_render_loss": first_result.initial_loss.image_render_loss,
        "final_image_render_loss": final_result.final_loss.image_render_loss,
        "initial_object_loss": first_result.initial_loss.object_loss,
        "final_object_loss": final_result.final_loss.object_loss,
        "loss_decreased": final_result.final_loss.total_loss < first_result.initial_loss.total_loss,
        "image_render_loss_decreased": (
            final_result.final_loss.image_render_loss < first_result.initial_loss.image_render_loss
        ),
        "object_loss_decreased": final_result.final_loss.object_loss < first_result.initial_loss.object_loss,
    }
    final_summary["training_scale"] = {
        "plan_schema": plan["schema"],
        "mode": "segmented",
        "segment_count": plan["segment_count"],
        "total_iterations": plan["total_iterations"],
        "checkpoint_every": plan["checkpoint_every"],
        "loss_log_every": plan["loss_log_every"],
        "run_output_dir": plan["outputs"]["root"],
        "segments": segment_records,
    }
    final_summary["run_loss"] = run_loss
    final_summary["run_assignment_stability"] = _solver_decoder_assignment_stability_gate(
        sample,
        before_solver_state=first_result.initial_solver_state,
        after_solver_state=final_result.final_solver_state,
    )
    if args.tensorboard_logdir:
        final_summary["tensorboard"] = write_solver_decoder_tensorboard_events(
            final_summary,
            args.tensorboard_logdir,
        )
    boundary = renderer_loss_boundary_report(final_checkpoint).as_dict()
    write_json(Path(plan["outputs"]["final_summary"]), final_summary)
    write_json(Path(plan["outputs"]["final_checkpoint"]), final_checkpoint)
    write_json(Path(plan["outputs"]["renderer_loss_boundary"]), boundary)
    return final_result, final_summary, final_checkpoint


def _solver_decoder_initial_solver_from_checkpoint(
    checkpoint: dict[str, object],
):
    if checkpoint.get("schema") == SOLVER_DECODER_JOINT_CHECKPOINT_SCHEMA:
        validate_solver_decoder_joint_checkpoint(checkpoint)
        solver_state, _decoder_state = solver_decoder_joint_states_from_dict(checkpoint)
        return solver_state, "solver_decoder_joint_checkpoint_solver_state"
    validate_object_emergence_solver_checkpoint(checkpoint)
    return object_emergence_solver_state_from_dict(checkpoint), "solver_checkpoint_initial_state"


def _training_object_emergence_solver(args: argparse.Namespace) -> None:
    source_cloud = read_ply(args.input)
    sampled_cloud = _sample_training_cloud(
        source_cloud,
        max_points=args.max_points,
        object_id_field=args.object_id_field,
        seed=args.seed,
    )
    targets, mapping = object_id_targets_from_cloud(
        sampled_cloud,
        object_id_field=args.object_id_field,
        slots=args.slots,
    )
    evidence = evidence_from_gaussian_cloud(
        sampled_cloud,
        target_assignment=targets,
        source=f"object_id_field:{args.object_id_field}",
    )
    assignment_evidence = assignment_evidence_from_object_emergence(evidence)
    result = train_object_emergence_solver(
        [evidence],
        slots=args.slots,
        iterations=args.iterations,
        learning_rate=args.learning_rate,
        assignment_weight=args.assignment_weight,
        entropy_weight=args.entropy_weight,
        balance_weight=args.balance_weight,
        temporal_weight=args.temporal_weight,
        finite_difference_epsilon=args.finite_difference_epsilon,
        seed=args.seed,
        record_every=args.record_every,
    )
    summary = {
        **result.as_dict(include_weights=args.include_weights),
        "input": str(args.input),
        "source_gaussians": source_cloud.count,
        "sampled_gaussians": sampled_cloud.count,
        "target_source": f"{args.object_id_field}_one_hot_targets",
        "object_id_mapping": {str(object_id): slot for object_id, slot in mapping.items()},
        "assignment_mvp": assignment_mvp_training_summary(
            result,
            [assignment_evidence],
        ),
        "gpu_policy": {
            "uses_gpu": False,
            "full_renderer_training": "suspended_until_torch_gsplat_cuda_available",
            "vram_reserve_gb": 1,
        },
    }
    checkpoint = object_emergence_solver_checkpoint(
        result,
        input_path=str(args.input),
        source_gaussians=source_cloud.count,
        sampled_gaussians=sampled_cloud.count,
        target_source=f"{args.object_id_field}_one_hot_targets",
        object_id_mapping=mapping,
        vram_reserve_gb=1,
    )
    print(f"schema={summary['schema']}")
    print(f"input={args.input}")
    print(f"source_gaussians={source_cloud.count}")
    print(f"sampled_gaussians={sampled_cloud.count}")
    print(f"slots={result.final_state.config.slots}")
    print(f"iterations={result.iterations}")
    print(f"initial_total_loss={result.initial_loss.total_loss:.6f}")
    print(f"final_total_loss={result.final_loss.total_loss:.6f}")
    print(f"initial_assignment_loss={result.initial_loss.assignment_loss:.6f}")
    print(f"final_assignment_loss={result.final_loss.assignment_loss:.6f}")
    print(f"loss_decreased={str(summary['loss_decreased']).lower()}")
    print(f"assignment_loss_decreased={str(summary['assignment_loss_decreased']).lower()}")
    print(f"assignment_mvp_schema={summary['assignment_mvp']['schema']}")
    print(f"assignment_mvp_loss_decreased={str(summary['assignment_mvp']['loss_decreased']).lower()}")
    print(
        "assignment_mvp_supervised_loss_decreased="
        f"{str(summary['assignment_mvp']['supervised_loss_decreased']).lower()}"
    )
    print("gpu_used=false")
    print("vram_reserve_gb=1")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.checkpoint_output:
        write_json(args.checkpoint_output, checkpoint)
        print(f"checkpoint={args.checkpoint_output}")
    if args.require_loss_decrease and not summary["loss_decreased"]:
        raise ValueError("object emergence solver loss did not decrease")


def _sample_training_cloud(
    cloud,
    *,
    max_points: int | None,
    object_id_field: str,
    seed: int,
):
    if max_points is None or cloud.count <= max_points:
        return cloud
    if max_points < 1:
        raise ValueError("max_points must be >= 1")
    rng = np.random.default_rng(seed)
    if object_id_field not in cloud.fields:
        selected = np.sort(rng.choice(cloud.count, size=max_points, replace=False))
        return cloud.with_vertices(cloud.vertices[selected])
    object_ids = np.asarray(cloud.vertices[object_id_field])
    selected: list[int] = []
    unique_ids = tuple(np.unique(object_ids))
    per_object = max(1, max_points // max(1, len(unique_ids)))
    for object_id in unique_ids:
        indices = np.flatnonzero(object_ids == object_id)
        take = min(indices.shape[0], per_object)
        selected.extend(rng.choice(indices, size=take, replace=False).astype(int).tolist())
    remaining = max_points - len(selected)
    if remaining > 0:
        pool = np.setdiff1d(np.arange(cloud.count), np.asarray(selected, dtype=np.int64), assume_unique=False)
        if pool.size:
            selected.extend(rng.choice(pool, size=min(remaining, pool.size), replace=False).astype(int).tolist())
    selected_indices = np.asarray(sorted(set(selected))[:max_points], dtype=np.int64)
    return cloud.with_vertices(cloud.vertices[selected_indices])


def _evaluate_training_renderer_api(
    frames,
    assignments,
    decoder_colors,
    *,
    image_renderer: str,
):
    if image_renderer == "point":
        return evaluate_training_renderer_loss(frames, assignments, decoder_colors)
    if image_renderer == "gsplat":
        return evaluate_gsplat_training_renderer_loss(frames, assignments, decoder_colors)
    raise ValueError("image_renderer must be one of: point, gsplat")


def _manifest_relative_path(artifact_path: Path, manifest_path: Path) -> str:
    return Path(
        os.path.relpath(
            Path(artifact_path).resolve(),
            Path(manifest_path).parent.resolve(),
        )
    ).as_posix()


def _training_renderer_loss_contract(args: argparse.Namespace) -> None:
    kernel_summary = None
    if args.kernel_summary:
        kernel_summary = json.loads(args.kernel_summary.read_text(encoding="utf-8"))
    report = renderer_loss_boundary_report(
        kernel_summary,
        target_renderer=args.target_renderer,
    )
    summary = report.as_dict()
    print(f"schema={summary['schema']}")
    print(f"status={summary['status']}")
    print(f"current_renderer={summary['current_renderer']}")
    print(f"target_renderer={summary['target_renderer']}")
    print(f"point_smoke_ready={str(summary['point_smoke_ready']).lower()}")
    print(f"point_smoke_blockers={summary['point_smoke_blockers']}")
    print(f"upgrade_blockers={summary['upgrade_blockers']}")
    decoder_handoff = summary["decoder_handoff_contract"]
    print(f"decoder_handoff_status={decoder_handoff['status']}")
    print(f"decoder_handoff_starts_real_training={str(decoder_handoff['starts_real_training']).lower()}")
    evidence = summary["evidence"]
    if evidence.get("kind") == "trainable_kernel_summary":
        print(f"evidence_target_source={evidence.get('target_source')}")
        print(f"evidence_initial_render_loss={evidence['initial_render_loss']:.6f}")
        print(f"evidence_final_render_loss={evidence['final_render_loss']:.6f}")
        print(f"evidence_initial_image_render_loss={evidence['initial_image_render_loss']:.6f}")
        print(f"evidence_final_image_render_loss={evidence['final_image_render_loss']:.6f}")
        if evidence.get("renderer_api_ready"):
            print(f"evidence_renderer_name={evidence.get('renderer_name')}")
            print(f"evidence_renderer_gradient_path={evidence.get('renderer_gradient_path')}")
            print(f"evidence_image_render_loss={evidence['image_render_loss']:.6f}")
    if evidence.get("kind") in {"object_emergence_solver_training", "object_emergence_solver_checkpoint"}:
        print(f"evidence_kind={evidence.get('kind')}")
        print(f"evidence_target_source={evidence.get('target_source')}")
        print(f"evidence_initial_total_loss={evidence['initial_total_loss']:.6f}")
        print(f"evidence_final_total_loss={evidence['final_total_loss']:.6f}")
        print(f"evidence_initial_assignment_loss={evidence['initial_assignment_loss']:.6f}")
        print(f"evidence_final_assignment_loss={evidence['final_assignment_loss']:.6f}")
        print(f"evidence_gpu_used={str(evidence.get('gpu_used')).lower()}")
        print(f"evidence_vram_reserve_gb={evidence.get('vram_reserve_gb')}")
    if evidence.get("kind") == "object_state_gaussian_decoder_training":
        print(f"evidence_kind={evidence.get('kind')}")
        print(f"evidence_target_source={evidence.get('target_source')}")
        print(f"evidence_initial_total_loss={evidence['initial_total_loss']:.6f}")
        print(f"evidence_final_total_loss={evidence['final_total_loss']:.6f}")
        print(f"evidence_initial_image_render_loss={evidence['initial_image_render_loss']:.6f}")
        print(f"evidence_final_image_render_loss={evidence['final_image_render_loss']:.6f}")
        print(f"evidence_loss_decreased={str(evidence.get('loss_decreased')).lower()}")
        print(f"evidence_image_render_loss_decreased={str(evidence.get('image_render_loss_decreased')).lower()}")
        print(f"evidence_trained_fields={','.join(evidence.get('trained_fields', []))}")
        print(f"evidence_gpu_used={str(evidence.get('gpu_used')).lower()}")
        print(f"evidence_vram_reserve_gb={evidence.get('vram_reserve_gb')}")
    if evidence.get("kind") == "solver_decoder_joint_training":
        print(f"evidence_kind={evidence.get('kind')}")
        print(f"evidence_target_source={evidence.get('target_source')}")
        print(f"evidence_initial_total_loss={evidence['initial_total_loss']:.6f}")
        print(f"evidence_final_total_loss={evidence['final_total_loss']:.6f}")
        print(f"evidence_initial_image_render_loss={evidence['initial_image_render_loss']:.6f}")
        print(f"evidence_final_image_render_loss={evidence['final_image_render_loss']:.6f}")
        print(f"evidence_initial_object_loss={evidence['initial_object_loss']:.6f}")
        print(f"evidence_final_object_loss={evidence['final_object_loss']:.6f}")
        print(f"evidence_loss_decreased={str(evidence.get('loss_decreased')).lower()}")
        print(f"evidence_image_render_loss_decreased={str(evidence.get('image_render_loss_decreased')).lower()}")
        print(f"evidence_object_loss_decreased={str(evidence.get('object_loss_decreased')).lower()}")
        print(f"evidence_trained_fields={','.join(evidence.get('trained_fields', []))}")
        print(f"evidence_gpu_used={str(evidence.get('gpu_used')).lower()}")
        print(f"evidence_vram_reserve_gb={evidence.get('vram_reserve_gb')}")
    if args.output:
        write_json(args.output, summary)
        print(f"summary={args.output}")
    if args.require_point_smoke_ready and not report.point_smoke_ready:
        raise ValueError(f"renderer loss boundary point smoke is not ready: {report.point_smoke_blockers}")


def _training_write_sample_bundle(args: argparse.Namespace) -> None:
    result = write_sample_bundle(
        output=args.output,
        sample_id=args.sample_id,
        asset_id=args.asset_id,
        dataset=args.dataset,
        masks=args.masks,
        training_manifest=args.training_manifest,
        split=args.split,
    )
    print(f"sample={result.sample_path}")
    print(f"sample_id={result.sample_id}")
    print(f"asset_id={result.asset_id}")
    print(f"image_count={result.image_count}")
    print(f"mask_frame_count={result.mask_frame_count}")
    if result.gaussian_count is not None:
        print(f"gaussians={result.gaussian_count}")
    if result.object_field_gaussian_count is not None:
        print(f"object_field_gaussians={result.object_field_gaussian_count}")
    if result.slot_count is not None:
        print(f"slots={result.slot_count}")


def _object_state_stability_benchmark(args: argparse.Namespace) -> None:
    report = write_object_state_stability_benchmark(
        args.output,
        report_id=args.report_id,
        strict=args.strict,
    )
    aggregate = report["aggregate"]
    print(f"schema={report['schema']}")
    print(f"status={report['status']}")
    print(f"cases={aggregate['case_count']}")
    print(f"observed_warn_count={aggregate['observed_warn_count']}")
    print(f"warn_count={aggregate['warn_count']}")
    print(f"output={args.output}")


def _object_state_controlled_real_gate(args: argparse.Namespace) -> None:
    manifest = read_objectstate_controlled_real_manifest(args.manifest)
    summary = objectstate_controlled_real_rows_summary(
        manifest,
        synthetic_smoke_passed=not args.synthetic_smoke_failed,
        thresholds=_controlled_real_gate_thresholds(args),
    )
    gate = summary["gate"]
    sample = summary["sample"]
    print(f"schema={summary['schema']}")
    print(f"manifest={args.manifest}")
    print(f"sample_id={sample['sample_id']}")
    print(f"object_category={sample['object_category']}")
    print(f"scenario={sample['scenario']}")
    print(f"gate_status={gate['status']}")
    print(f"row_count={summary['row_count']}")
    print(f"pass_rows={summary['pass_row_count']}")
    print(f"fail_rows={summary['fail_row_count']}")
    print(f"blocked_rows={summary['blocked_row_count']}")
    if gate["hard_blockers"]:
        print(f"hard_blockers={','.join(gate['hard_blockers'])}")
    else:
        print("hard_blockers=none")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.blocked_rows_output:
        args.blocked_rows_output.parent.mkdir(parents=True, exist_ok=True)
        args.blocked_rows_output.write_text(
            summary["blocked_rows_markdown"],
            encoding="utf-8",
        )
        print(f"blocked_rows={args.blocked_rows_output}")
    if args.require_pass and gate["status"] != "objectstate_reality_gate_pass":
        raise ValueError("controlled real ObjectState reality gate did not pass")


def _object_state_init_controlled_capture_bundle(args: argparse.Namespace) -> None:
    summary = write_objectstate_controlled_capture_bundle_template(
        args.bundle_root,
        sample_id=args.sample_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        capture_device=args.capture_device,
        license_text=args.license_text,
        objects=_controlled_capture_template_objects(args.objects),
        force=args.force,
    )
    print(f"schema={summary['schema']}")
    print(f"bundle_root={summary['root']}")
    print(f"sample_id={summary['sample']['sample_id']}")
    print(f"object_row_count={summary['object_row_count']}")
    for key, path in summary["files"].items():
        print(f"{key}={path}")
    for key, path in summary["directories"].items():
        print(f"{key}_dir={path}")
    for key, command in summary["next_commands"].items():
        print(f"{key}_command={command}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")


def _object_state_audit_controlled_capture_bundle_readiness(
    args: argparse.Namespace,
) -> None:
    summary = objectstate_controlled_capture_bundle_readiness(
        args.bundle_root,
        sample_json=args.sample_json,
        objects_csv=args.objects_csv,
        frames_csv=args.frames_csv,
        annotations_csv=args.annotations_csv,
        actions_csv=args.actions_csv,
        require_prediction_ready=args.require_prediction_ready,
        require_intervention_ready=args.require_intervention_ready,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
        candidate_artifact=args.candidate_artifact,
        require_candidate_artifact=args.require_candidate_artifact,
        min_candidate_artifact_bytes=args.min_candidate_artifact_bytes,
        min_identity_scenario_frames=args.min_identity_scenario_frames,
        min_occlusion_fraction=args.min_occlusion_fraction,
        min_view_conditions=args.min_view_conditions,
        min_lighting_conditions=args.min_lighting_conditions,
        min_camera_motion_m=args.min_camera_motion_m,
    )
    readiness = summary["readiness"]
    print(f"schema={summary['schema']}")
    print(f"bundle_root={summary['root']}")
    print(f"readiness_status={summary['status']}")
    print(f"capture_bundle_ready={str(readiness['capture_bundle_ready']).lower()}")
    print(
        "identity_bundle_handoff_ready="
        f"{str(readiness['identity_bundle_handoff_ready']).lower()}"
    )
    print(f"layout_ready={str(readiness['layout_ready']).lower()}")
    print(f"sample_metadata_ready={str(readiness['sample_metadata_ready']).lower()}")
    print(f"csv_headers_ready={str(readiness['csv_headers_ready']).lower()}")
    print(f"identity_stage_ready={str(readiness['identity_stage_ready']).lower()}")
    print(f"capture_files_ready={str(readiness['capture_files_ready']).lower()}")
    print(
        "identity_scenario_ready="
        f"{str(readiness['identity_scenario_ready']).lower()}"
    )
    print(
        "candidate_artifact_ready="
        f"{str(readiness['candidate_artifact_ready']).lower()}"
    )
    for key, value in summary["row_counts"].items():
        print(f"{key}_rows={value}")
    print(f"hard_blockers={len(summary['hard_blockers'])}")
    for blocker in summary["hard_blockers"]:
        print(f"blocker={blocker}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if (
        args.require_ready
        and summary["status"]
        != "objectstate_controlled_capture_bundle_readiness_ready"
    ):
        raise ValueError("controlled capture bundle readiness did not pass")


def _object_state_audit_controlled_capture_environment(
    args: argparse.Namespace,
) -> None:
    summary = objectstate_controlled_capture_environment(dev_root=args.dev_root)
    readiness = summary["readiness"]
    devices = summary["devices"]
    commands = summary["commands"]
    print(f"schema={summary['schema']}")
    print(f"environment_status={summary['status']}")
    print(f"dev_root={summary['dev_root']}")
    print(f"video_devices={len(devices['video_devices'])}")
    for device in devices["video_devices"]:
        print(f"video_device={device}")
    print(f"rgb_capture_ready={str(readiness['rgb_capture_ready']).lower()}")
    print(
        "gaussian_reconstruction_ready="
        f"{str(readiness['gaussian_reconstruction_ready']).lower()}"
    )
    print(
        "controlled_capture_environment_ready="
        f"{str(readiness['controlled_capture_environment_ready']).lower()}"
    )
    for key in (
        "ffmpeg",
        "v4l2_ctl",
        "colmap",
        "ns_process_data",
        "ns_train",
        "ns_export",
    ):
        record = commands[key]
        print(f"command.{key}.available={str(record['available']).lower()}")
        if record["path"]:
            print(f"command.{key}.path={record['path']}")
    print(
        "python.cv2.available="
        f"{str(summary['python_modules']['cv2']['available']).lower()}"
    )
    print(f"hard_blockers={len(summary['hard_blockers'])}")
    for blocker in summary["hard_blockers"]:
        print(f"blocker={blocker}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if (
        args.require_ready
        and not readiness["controlled_capture_environment_ready"]
    ):
        raise ValueError("controlled capture environment is not ready")


def _object_state_audit_public_dataset_candidates(
    args: argparse.Namespace,
) -> None:
    summary = objectstate_public_dataset_candidates_audit()
    readiness = summary["readiness"]
    print(f"schema={summary['schema']}")
    print(f"status={summary['status']}")
    print(f"candidate_count={summary['candidate_count']}")
    print(f"recommended_first={summary['recommended_first']}")
    print(f"recommended_action_candidate={summary['recommended_action_candidate']}")
    print(
        "direct_phase1_ready="
        f"{str(readiness['has_direct_phase1_ready_dataset']).lower()}"
    )
    print(
        "direct_gaussian_evidence="
        f"{str(readiness['has_direct_gaussian_evidence']).lower()}"
    )
    print(f"hard_blockers={len(summary['hard_blockers'])}")
    for blocker in summary["hard_blockers"]:
        print(f"blocker={blocker}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.markdown_output:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            objectstate_public_dataset_candidates_markdown(summary),
            encoding="utf-8",
        )
        print(f"markdown={args.markdown_output}")
    if args.require_ready and not readiness["has_direct_phase1_ready_dataset"]:
        raise ValueError("no public dataset candidate is directly Phase 1 ready")


def _object_state_select_bop_phase1_subset(args: argparse.Namespace) -> None:
    summary = objectstate_bop_phase1_subset_selector(
        args.dataset_root,
        dataset_id=args.dataset_id,
        output_root=args.output_root,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license_text,
        rgb_dir=args.rgb_dir,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        max_depth=args.max_depth,
        max_scene_candidates=args.max_scene_candidates,
        min_frames=args.min_frames,
        min_objects=args.min_objects,
        min_persistent_objects=args.min_persistent_objects,
    )
    print(f"schema={summary['schema']}")
    print(f"bop_phase1_subset_selector_status={summary['status']}")
    print(f"dataset_root={summary['dataset_root']}")
    print(f"scene_candidates={summary['row_counts']['scene_candidates']}")
    print(f"ready_candidates={summary['row_counts']['ready_candidates']}")
    recommended = summary["recommended"]
    if recommended:
        print(f"recommended_scene_root={recommended['scene_root']}")
        print(f"recommended_sample_id={recommended['sample_id']}")
    for gate, passed in summary["readiness"].items():
        print(f"readiness.{gate}={str(passed).lower()}")
    for blocker in summary["hard_blockers"]:
        print(f"blocker={blocker}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    for command in summary["next_commands"]:
        print(f"next_command={command}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.batch_samples_csv_template_output:
        ready_count = _write_bop_batch_samples_csv_template(
            args.batch_samples_csv_template_output,
            summary,
            dataset_id=args.dataset_id,
            object_category=args.object_category,
            scenario=args.scenario,
            artifact_root=args.batch_sample_artifact_root,
            max_frames=args.max_frames,
            frame_step=args.frame_step,
        )
        print(f"batch_samples_csv_template={args.batch_samples_csv_template_output}")
        print(f"batch_samples_csv_template_rows={ready_count}")
    if args.require_ready and summary["recommended"] is None:
        raise ValueError("no BOP Phase 1 subset candidate is ready")


def _write_bop_batch_samples_csv_template(
    output: Path,
    summary: Mapping[str, object],
    *,
    dataset_id: str,
    object_category: str,
    scenario: str,
    artifact_root: Path,
    max_frames: int | None,
    frame_step: int,
) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for candidate in summary["candidates"]:  # type: ignore[index]
        if not candidate["readiness"]["phase1_seed_ready"]:
            continue
        sample_id = candidate["sample_id"]
        sample_root = artifact_root / sample_id
        rows.append(
            {
                "sample_id": sample_id,
                "scene_root": _path_for_csv(candidate["scene_root"], output.parent),
                "candidate_artifact": _path_for_csv(
                    sample_root / "objectstates.json",
                    output.parent,
                ),
                "condition_sidecar": _path_for_csv(
                    sample_root / "bop-condition-sidecar.json",
                    output.parent,
                ),
                "output_root": f"samples/{sample_id}",
                "dataset_id": dataset_id,
                "object_category": object_category,
                "scenario": scenario,
                "max_frames": "" if max_frames is None else str(max_frames),
                "frame_step": str(frame_step),
            }
        )
    fieldnames = [
        "sample_id",
        "scene_root",
        "candidate_artifact",
        "condition_sidecar",
        "output_root",
        "dataset_id",
        "object_category",
        "scenario",
        "max_frames",
        "frame_step",
    ]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _path_for_csv(value: str | Path, csv_root: Path) -> str:
    path = Path(value)
    absolute = path if path.is_absolute() else Path.cwd() / path
    return os.path.relpath(absolute, csv_root)


def _object_state_audit_bop_gaussian_evidence(args: argparse.Namespace) -> None:
    summary = objectstate_bop_gaussian_evidence_preflight(
        args.scene_root,
        sample_id=args.sample_id,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license_text,
        rgb_dir=args.rgb_dir,
        gaussian_dir=args.gaussian_dir,
        condition_sidecar=args.condition_sidecar,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
        dev_root=args.dev_root,
    )
    readiness = summary["readiness"]
    print(f"schema={summary['schema']}")
    print(f"bop_gaussian_evidence_status={summary['status']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"selected_frames={summary['row_counts']['selected_frames']}")
    print(f"expected_gaussian_files={summary['row_counts']['expected_gaussian_files']}")
    print(f"missing_gaussian_files={summary['row_counts']['missing_gaussian_files']}")
    print(
        "phase1_gaussian_evidence_ready="
        f"{str(readiness['phase1_gaussian_evidence_ready']).lower()}"
    )
    print(
        "gaussian_reconstruction_tools_ready="
        f"{str(readiness['gaussian_reconstruction_tools_ready']).lower()}"
    )
    for blocker in summary["hard_blockers"]:
        print(f"blocker={blocker}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    for command in summary["next_commands"]:
        print(f"next_command={command}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.missing_gaussians_output:
        args.missing_gaussians_output.parent.mkdir(parents=True, exist_ok=True)
        args.missing_gaussians_output.write_text(
            summary["missing_gaussian_files_markdown"],
            encoding="utf-8",
        )
        print(f"missing_gaussians_markdown={args.missing_gaussians_output}")
    if args.require_ready and not readiness["phase1_gaussian_evidence_ready"]:
        raise ValueError("BOP Gaussian evidence is not ready")


def _object_state_export_bop_rgbd_gaussian_evidence(
    args: argparse.Namespace,
) -> None:
    summary = objectstate_bop_rgbd_gaussian_export(
        args.scene_root,
        sample_id=args.sample_id,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license_text,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        gaussian_dir=args.gaussian_dir,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        pixel_stride=args.pixel_stride,
        max_points_per_frame=args.max_points_per_frame,
        min_depth_m=args.min_depth_m,
        max_depth_m=args.max_depth_m,
        overwrite=args.overwrite,
        ply_format=args.ply_format,
    )
    readiness = summary["readiness"]
    print(f"schema={summary['schema']}")
    print(f"bop_rgbd_gaussian_export_status={summary['status']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"selected_frames={summary['row_counts']['selected_frames']}")
    print(f"exported_frames={summary['row_counts']['exported_frames']}")
    print(f"missing_depth_files={summary['row_counts']['missing_depth_files']}")
    print(f"total_vertices={summary['row_counts']['total_vertices']}")
    print(
        "phase1_gaussian_evidence_written="
        f"{str(readiness['phase1_gaussian_evidence_written']).lower()}"
    )
    for blocker in summary["hard_blockers"]:
        print(f"blocker={blocker}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.require_ready and not readiness["phase1_gaussian_evidence_written"]:
        raise ValueError("BOP RGB-D Gaussian evidence export is not ready")


def _object_state_init_bop_condition_sidecar(args: argparse.Namespace) -> None:
    summary = objectstate_bop_capture_condition_sidecar_summary(
        args.scene_root,
        condition_csv=args.condition_csv,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        default_lighting_id=args.default_lighting_id,
        min_view_conditions=args.min_view_conditions,
        min_lighting_conditions=args.min_lighting_conditions,
        min_camera_motion_m=args.min_camera_motion_m,
    )
    readiness = summary["readiness"]
    coverage = summary["coverage"]
    write_json(args.output, summary["sidecar"])
    if args.condition_csv_template_output:
        _write_condition_csv_template(
            args.condition_csv_template_output,
            summary["condition_csv_template"],
        )
    print(f"schema={summary['schema']}")
    print(f"bop_condition_sidecar_status={summary['status']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"sidecar={args.output}")
    if args.condition_csv_template_output:
        print(f"condition_csv_template={args.condition_csv_template_output}")
    print(f"selected_frames={summary['row_counts']['selected_frames']}")
    print(f"csv_condition_rows={summary['row_counts']['csv_condition_rows']}")
    print(f"view_condition_count={coverage['view_condition_count']}")
    print(f"lighting_condition_count={coverage['lighting_condition_count']}")
    print(f"camera_pose_count={coverage['camera_pose_count']}")
    print(
        "max_camera_translation_m="
        f"{coverage['max_camera_translation_m']:.6f}"
    )
    for gate, passed in readiness.items():
        print(f"readiness.{gate}={str(passed).lower()}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.require_identity_ready and not readiness["identity_scenario_metadata_ready"]:
        raise ValueError("BOP condition sidecar is not identity-scenario ready")


def _write_condition_csv_template(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "frame_id",
        "view_id",
        "lighting_id",
        "camera_x",
        "camera_y",
        "camera_z",
        "camera_qx",
        "camera_qy",
        "camera_qz",
        "camera_qw",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def _object_state_import_bop_capture_scene(args: argparse.Namespace) -> None:
    summary = objectstate_bop_capture_adapter_summary(
        args.scene_root,
        sample_id=args.sample_id,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license_text,
        rgb_dir=args.rgb_dir,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        include_gaussian_refs=args.include_gaussian_refs,
        gaussian_dir=args.gaussian_dir,
        condition_sidecar=args.condition_sidecar,
    )
    readiness = summary["readiness"]
    print(f"schema={summary['schema']}")
    print(f"bop_adapter_status={summary['status']}")
    print(f"scene_root={summary['source']['scene_root']}")
    print(f"sample_id={summary['manifest']['sample']['sample_id']}")
    print(f"dataset_id={summary['source']['dataset_id']}")
    print(f"frames={summary['row_counts']['frames']}")
    print(f"objects={summary['row_counts']['objects']}")
    print(f"annotations={summary['row_counts']['annotations']}")
    print(f"identity_stage_ready={str(readiness['identity_stage_ready']).lower()}")
    print(f"prediction_stage_ready={str(readiness['prediction_stage_ready']).lower()}")
    print(f"intervention_stage_ready={str(readiness['intervention_stage_ready']).lower()}")
    print(
        "real_gaussian_reconstruction_present="
        f"{str(readiness['real_gaussian_reconstruction_present']).lower()}"
    )
    print(f"hard_blockers={len(summary['hard_blockers'])}")
    for blocker in summary["hard_blockers"]:
        print(f"blocker={blocker}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    write_json(args.output, summary["manifest"])
    print(f"manifest={args.output}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.controlled_real_output:
        write_json(args.controlled_real_output, summary["controlled_real_manifest_seed"])
        print(f"controlled_real={args.controlled_real_output}")
    if args.require_identity_ready and not readiness["identity_stage_ready"]:
        raise ValueError("BOP capture adapter output is not identity-stage ready")
    if args.require_prediction_ready and not readiness["prediction_stage_ready"]:
        raise ValueError("BOP capture adapter output is not prediction-stage ready")


def _object_state_accept_bop_capture_scene(args: argparse.Namespace) -> None:
    summary = objectstate_bop_capture_acceptance_summary(
        args.scene_root,
        sample_id=args.sample_id,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license_text,
        rgb_dir=args.rgb_dir,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        include_gaussian_refs=args.include_gaussian_refs,
        gaussian_dir=args.gaussian_dir,
        condition_sidecar=args.condition_sidecar,
        require_gaussian_files=args.require_gaussian_files,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
    )
    readiness = summary["readiness"]
    file_audit = summary["file_audit"]
    print(f"schema={summary['schema']}")
    print(f"bop_acceptance_status={summary['status']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"frames={summary['adapter']['row_counts']['frames']}")
    print(f"objects={summary['adapter']['row_counts']['objects']}")
    print(f"capture_file_audit_pass={str(readiness['capture_file_audit_pass']).lower()}")
    print(f"rgb_files_present={str(readiness['rgb_files_present']).lower()}")
    print(f"gaussian_files_present={str(readiness['gaussian_files_present']).lower()}")
    print(
        "phase1_gaussian_evidence_ready="
        f"{str(readiness['phase1_gaussian_evidence_ready']).lower()}"
    )
    print(f"missing_files={len(file_audit['missing_files'])}")
    print(f"hard_blockers={len(summary['hard_blockers'])}")
    for blocker in summary["hard_blockers"]:
        print(f"blocker={blocker}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    write_json(args.output, summary["manifest"])
    print(f"manifest={args.output}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.file_audit_output:
        write_json(args.file_audit_output, file_audit)
        print(f"file_audit={args.file_audit_output}")
    if args.missing_files_output:
        args.missing_files_output.parent.mkdir(parents=True, exist_ok=True)
        args.missing_files_output.write_text(
            file_audit["missing_files_markdown"],
            encoding="utf-8",
        )
        print(f"missing_files_markdown={args.missing_files_output}")
    if args.controlled_real_output:
        write_json(args.controlled_real_output, summary["controlled_real_manifest_seed"])
        print(f"controlled_real={args.controlled_real_output}")
    if args.require_pass and summary["status"] != "objectstate_bop_capture_acceptance_pass":
        raise ValueError("BOP capture scene acceptance did not pass")


def _object_state_init_bop_objectstate_artifact_template(
    args: argparse.Namespace,
) -> None:
    summary = write_objectstate_bop_candidate_artifact_template(
        args.scene_root,
        output=args.output,
        sample_id=args.sample_id,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license,
        rgb_dir=args.rgb_dir,
        gaussian_dir=args.gaussian_dir,
        condition_sidecar=args.condition_sidecar,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        candidate_id=args.candidate_id,
        candidate_source=args.candidate_source,
        target_artifact_path=args.target_artifact_path,
        force=args.force,
    )
    readiness = summary["readiness"]
    counts = summary["row_counts"]
    print(f"schema={summary['schema']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"output={summary['output']}")
    print(f"target_artifact={summary['target_artifact']}")
    print(f"bop_objectstate_artifact_template_status={summary['status']}")
    print(f"frames={counts['frames']}")
    print(f"state_placeholders={counts['state_placeholders']}")
    for gate, passed in readiness.items():
        print(f"readiness.{gate}={str(passed).lower()}")
    print(f"bop_acceptance_status={summary['acceptance']['status']}")
    print(
        "template_not_valid_for_identity_route="
        f"{str(readiness['draft_not_valid_for_identity_route']).lower()}"
    )
    for command in summary["next_commands"]:
        print(f"next_command={command}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")


def _object_state_finalize_bop_objectstate_artifact_template(
    args: argparse.Namespace,
) -> None:
    summary = finalize_objectstate_bop_candidate_artifact_template(
        args.template,
        output=args.output,
        scene_root=args.scene_root,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license,
        rgb_dir=args.rgb_dir,
        gaussian_dir=args.gaussian_dir,
        condition_sidecar=args.condition_sidecar,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        gt_leakage_tolerance=args.gt_leakage_tolerance,
        reconstruction_noise_robustness=args.reconstruction_noise_robustness,
        reconstruction_noise_variant_count=args.reconstruction_noise_variant_count,
        force=args.force,
    )
    readiness = summary["readiness"]
    counts = summary["row_counts"]
    print(f"schema={summary['schema']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"template={summary['template']}")
    print(f"output={summary['output']}")
    print(f"bop_objectstate_artifact_finalize_status={summary['status']}")
    print(f"frames={counts['frames']}")
    print(f"states={counts['states']}")
    for gate, passed in readiness.items():
        print(f"readiness.{gate}={str(passed).lower()}")
    leakage = summary["pose_gt_leakage_guard"]
    print(
        "pose_gt_min_distance="
        f"{float(leakage['min_distance_to_pose_gt']):.9f}"
    )
    print(f"bop_acceptance_status={summary['acceptance']['status']}")
    for command in summary["next_commands"]:
        print(f"next_command={command}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")


def _object_state_init_controlled_reality_candidates(
    args: argparse.Namespace,
) -> None:
    summary = write_objectstate_controlled_reality_candidate_templates(
        args.bundle_root,
        output_dir=args.output_dir,
        sample_json=args.sample_json,
        objects_csv=args.objects_csv,
        frames_csv=args.frames_csv,
        annotations_csv=args.annotations_csv,
        actions_csv=args.actions_csv,
        candidate_id=args.candidate_id,
        candidate_source=args.candidate_source,
        artifact_ref=args.artifact_ref,
        force=args.force,
    )
    print(f"schema={summary['schema']}")
    print(f"bundle_root={summary['bundle_root']}")
    print(f"output_dir={summary['output_dir']}")
    print(f"sample_id={summary['sample']['sample_id']}")
    print(f"prediction_drafts={summary['row_counts']['prediction_drafts']}")
    print(f"intervention_drafts={summary['row_counts']['intervention_drafts']}")
    for key, path in summary["files"].items():
        print(f"{key}={path}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    for key, command in summary["next_commands"].items():
        print(f"{key}_command={command}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")


def _object_state_init_controlled_reality_candidates_from_manifest(
    args: argparse.Namespace,
) -> None:
    summary = write_objectstate_controlled_reality_candidate_templates_from_manifest(
        args.capture_manifest,
        output_dir=args.output_dir,
        candidate_id=args.candidate_id,
        candidate_source=args.candidate_source,
        artifact_ref=args.artifact_ref,
        force=args.force,
    )
    print(f"schema={summary['schema']}")
    print(f"capture_manifest={summary['capture_manifest']}")
    print(f"output_dir={summary['output_dir']}")
    print(f"sample_id={summary['sample']['sample_id']}")
    print(f"prediction_drafts={summary['row_counts']['prediction_drafts']}")
    print(f"intervention_drafts={summary['row_counts']['intervention_drafts']}")
    for key, path in summary["files"].items():
        print(f"{key}={path}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    for key, command in summary["next_commands"].items():
        print(f"{key}_command={command}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")


def _object_state_finalize_controlled_reality_candidates(
    args: argparse.Namespace,
) -> None:
    summary = finalize_objectstate_controlled_reality_candidate_templates(
        args.prediction_template,
        args.intervention_template,
        output_dir=args.output_dir,
        bundle_root=args.bundle_root,
        force=args.force,
    )
    print(f"schema={summary['schema']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"output_dir={args.output_dir}")
    print(f"prediction_candidates={summary['files']['prediction_candidates']}")
    print(f"intervention_candidates={summary['files']['intervention_candidates']}")
    print(
        "prediction_candidate_count="
        f"{summary['row_counts']['prediction_candidates']}"
    )
    print(
        "intervention_candidate_count="
        f"{summary['row_counts']['intervention_candidates']}"
    )
    for key, command in summary["next_commands"].items():
        print(f"{key}_command={command}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")


def _object_state_finalize_controlled_prediction_candidates(
    args: argparse.Namespace,
) -> None:
    summary = finalize_objectstate_controlled_prediction_candidate_template(
        args.prediction_template,
        output_dir=args.output_dir,
        capture_manifest=args.capture_manifest,
        force=args.force,
    )
    print(f"schema={summary['schema']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"output_dir={args.output_dir}")
    print(f"prediction_candidates={summary['files']['prediction_candidates']}")
    print(
        "prediction_candidate_count="
        f"{summary['row_counts']['prediction_candidates']}"
    )
    for key, command in summary["next_commands"].items():
        print(f"{key}_command={command}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")


def _object_state_generate_controlled_prediction_baseline_candidates(
    args: argparse.Namespace,
) -> None:
    summary = write_objectstate_controlled_prediction_baseline_candidates(
        args.capture_manifest,
        args.prediction_template,
        output_dir=args.output_dir,
        policy=args.policy,
        candidate_id=args.candidate_id,
        candidate_source=args.candidate_source,
        artifact_ref=args.artifact_ref,
        confidence=args.confidence,
        force=args.force,
    )
    print(f"schema={summary['schema']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"output_dir={summary['output_dir']}")
    print(f"policy={summary['policy']['name']}")
    print(
        "prediction_candidate_count="
        f"{summary['row_counts']['prediction_candidates']}"
    )
    print(
        "constant_velocity_rows="
        f"{summary['row_counts']['constant_velocity_rows']}"
    )
    print(f"hold_rows={summary['row_counts']['hold_rows']}")
    for key, path in summary["files"].items():
        print(f"{key}={path}")
    for key, command in summary["next_commands"].items():
        print(f"{key}_command={command}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")


def _controlled_capture_template_objects(
    object_specs: list[str] | None,
) -> list[dict[str, str]]:
    objects = []
    for spec in object_specs or ():
        parts = spec.split(":", 2)
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(
                "--object must use object_id:category or object_id:category:label"
            )
        item = {
            "object_id": parts[0],
            "category": parts[1],
        }
        if len(parts) == 3 and parts[2]:
            item["instance_label"] = parts[2]
        objects.append(item)
    return objects


def _object_state_import_controlled_capture_bundle(args: argparse.Namespace) -> None:
    summary = objectstate_controlled_capture_import_summary(
        args.bundle_root,
        sample_json=args.sample_json,
        objects_csv=args.objects_csv,
        frames_csv=args.frames_csv,
        annotations_csv=args.annotations_csv,
        actions_csv=args.actions_csv,
    )
    manifest = summary["manifest"]
    capture_summary = summary["capture_summary"]
    sample = manifest["sample"]
    readiness = capture_summary["readiness"]
    write_json(args.output, manifest)
    print(f"schema={summary['schema']}")
    print(f"bundle_root={args.bundle_root}")
    print(f"sample_id={sample['sample_id']}")
    print(f"frames={summary['row_counts']['frames']}")
    print(f"objects={summary['row_counts']['objects']}")
    print(f"annotations={summary['row_counts']['annotations']}")
    print(f"actions={summary['row_counts']['actions']}")
    print(f"identity_stage_ready={str(readiness['identity_stage_ready']).lower()}")
    print(f"prediction_stage_ready={str(readiness['prediction_stage_ready']).lower()}")
    print(f"intervention_stage_ready={str(readiness['intervention_stage_ready']).lower()}")
    print(
        "real_gaussian_reconstruction_present="
        f"{str(readiness['real_gaussian_reconstruction_present']).lower()}"
    )
    print(f"output={args.output}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.controlled_real_output:
        write_json(
            args.controlled_real_output,
            capture_summary["controlled_real_manifest_seed"],
        )
        print(f"controlled_real_manifest={args.controlled_real_output}")
    if args.require_identity_ready and not readiness["identity_stage_ready"]:
        raise ValueError("imported controlled capture bundle is not identity-stage ready")
    if args.require_prediction_ready and not readiness["prediction_stage_ready"]:
        raise ValueError("imported controlled capture bundle is not prediction-stage ready")
    if args.require_intervention_ready and not readiness["intervention_stage_ready"]:
        raise ValueError("imported controlled capture bundle is not intervention-stage ready")


def _object_state_accept_controlled_capture_bundle(args: argparse.Namespace) -> None:
    summary = objectstate_controlled_capture_bundle_acceptance_summary(
        args.bundle_root,
        sample_json=args.sample_json,
        objects_csv=args.objects_csv,
        frames_csv=args.frames_csv,
        annotations_csv=args.annotations_csv,
        actions_csv=args.actions_csv,
        require_identity_ready=not args.no_require_identity_ready,
        require_prediction_ready=args.require_prediction_ready,
        require_intervention_ready=args.require_intervention_ready,
        require_gaussian_files=not args.no_require_gaussian_files,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
    )
    import_summary = summary["import_summary"]
    file_audit = summary["capture_file_audit"]
    manifest = import_summary["manifest"]
    capture_summary = import_summary["capture_summary"]
    readiness = capture_summary["readiness"]
    gates = summary["acceptance_gates"]
    write_json(args.output, manifest)
    print(f"schema={summary['schema']}")
    print(f"bundle_root={args.bundle_root}")
    print(f"sample_id={manifest['sample']['sample_id']}")
    print(f"acceptance_status={summary['status']}")
    print(f"identity_stage_ready={str(readiness['identity_stage_ready']).lower()}")
    print(f"prediction_stage_ready={str(readiness['prediction_stage_ready']).lower()}")
    print(f"intervention_stage_ready={str(readiness['intervention_stage_ready']).lower()}")
    print(
        "capture_file_audit_status="
        f"{file_audit['status']}"
    )
    print(
        "capture_bundle_files_ready="
        f"{str(file_audit['readiness']['capture_bundle_files_ready']).lower()}"
    )
    print(
        "acceptance_gates="
        + ",".join(f"{key}:{str(value).lower()}" for key, value in gates.items())
    )
    print(f"missing_files={len(file_audit['missing_files'])}")
    print(f"output={args.output}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.import_summary_output:
        write_json(args.import_summary_output, import_summary)
        print(f"import_summary={args.import_summary_output}")
    if args.file_audit_output:
        write_json(args.file_audit_output, file_audit)
        print(f"file_audit={args.file_audit_output}")
    if args.missing_files_output:
        args.missing_files_output.parent.mkdir(parents=True, exist_ok=True)
        args.missing_files_output.write_text(
            file_audit["missing_files_markdown"],
            encoding="utf-8",
        )
        print(f"missing_files_markdown={args.missing_files_output}")
    if args.controlled_real_output:
        write_json(
            args.controlled_real_output,
            capture_summary["controlled_real_manifest_seed"],
        )
        print(f"controlled_real_manifest={args.controlled_real_output}")
    if (
        args.require_pass
        and summary["status"]
        != "objectstate_controlled_capture_bundle_acceptance_pass"
    ):
        raise ValueError("controlled capture bundle acceptance did not pass")


def _object_state_validate_controlled_capture(args: argparse.Namespace) -> None:
    manifest = read_objectstate_controlled_capture_manifest(args.manifest)
    summary = objectstate_controlled_capture_summary(manifest)
    sample = summary["sample"]
    readiness = summary["readiness"]
    coverage = summary["observation_coverage"]
    print(f"schema={summary['schema']}")
    print(f"manifest={args.manifest}")
    print(f"sample_id={sample['sample_id']}")
    print(f"object_category={sample['object_category']}")
    print(f"scenario={sample['scenario']}")
    print(f"frames={summary['frame_count']}")
    print(f"objects={summary['object_count']}")
    print(f"actions={summary['action_count']}")
    print(f"rgb_frames={coverage['rgb_frames']}")
    print(f"gaussian_frames={coverage['gaussian_frames']}")
    print(f"identity_stage_ready={str(readiness['identity_stage_ready']).lower()}")
    print(f"prediction_stage_ready={str(readiness['prediction_stage_ready']).lower()}")
    print(f"intervention_stage_ready={str(readiness['intervention_stage_ready']).lower()}")
    print(
        "real_gaussian_reconstruction_present="
        f"{str(readiness['real_gaussian_reconstruction_present']).lower()}"
    )
    print(f"issues={len(summary['issues'])}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.controlled_real_output:
        write_json(args.controlled_real_output, summary["controlled_real_manifest_seed"])
        print(f"controlled_real_manifest={args.controlled_real_output}")
    if args.require_identity_ready and not readiness["identity_stage_ready"]:
        raise ValueError("controlled capture manifest is not identity-stage ready")
    if args.require_prediction_ready and not readiness["prediction_stage_ready"]:
        raise ValueError("controlled capture manifest is not prediction-stage ready")
    if args.require_intervention_ready and not readiness["intervention_stage_ready"]:
        raise ValueError("controlled capture manifest is not intervention-stage ready")


def _object_state_audit_controlled_capture_files(args: argparse.Namespace) -> None:
    manifest = read_objectstate_controlled_capture_manifest(args.manifest)
    root = args.root if args.root is not None else args.manifest.parent
    summary = objectstate_controlled_capture_file_audit(
        manifest,
        root=root,
        require_gaussian_files=not args.no_require_gaussian_files,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
    )
    counts = summary["file_counts"]
    readiness = summary["readiness"]
    print(f"schema={summary['schema']}")
    print(f"manifest={args.manifest}")
    print(f"root={summary['root']}")
    print(f"sample_id={summary['sample']['sample_id']}")
    print(f"file_audit_status={summary['status']}")
    print(f"rgb_valid={counts['rgb']['valid']}/{counts['rgb']['referenced']}")
    print(
        "gaussian_valid="
        f"{counts['gaussian']['valid']}/{counts['gaussian']['referenced']}"
    )
    print(
        "artifact_refs_valid="
        f"{counts['artifact_refs']['valid']}/{counts['artifact_refs']['referenced']}"
    )
    print(f"frame_formats_valid={str(readiness['frame_formats_valid']).lower()}")
    print(f"missing_files={len(summary['missing_files'])}")
    print(f"capture_bundle_files_ready={str(readiness['capture_bundle_files_ready']).lower()}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.missing_files_output:
        args.missing_files_output.parent.mkdir(parents=True, exist_ok=True)
        args.missing_files_output.write_text(
            summary["missing_files_markdown"],
            encoding="utf-8",
        )
        print(f"missing_files_markdown={args.missing_files_output}")
    if args.require_pass and summary["status"] != "objectstate_controlled_capture_file_audit_pass":
        raise ValueError("controlled capture file audit did not pass")


def _object_state_eval_controlled_identity(args: argparse.Namespace) -> None:
    capture = read_objectstate_controlled_capture_manifest(args.capture_manifest)
    predictions = read_objectstate_controlled_identity_predictions(args.predictions)
    summary = evaluate_objectstate_controlled_identity_predictions(
        capture,
        predictions,
        thresholds=ObjectStateControlledIdentityThresholds(
            min_idf1=args.min_idf1,
            min_track_retrieval_recall_at_1=args.min_track_retrieval_recall_at_1,
            max_fragmentation_rate=args.max_fragmentation_rate,
            max_long_term_drift_rate=args.max_long_term_drift_rate,
            max_swap_rate=args.max_swap_rate,
            min_reconstruction_noise_robustness=(
                args.min_reconstruction_noise_robustness
            ),
            min_reconstruction_noise_variants=args.min_reconstruction_noise_variants,
            require_no_identity_collapse=not args.allow_identity_collapse,
        ),
    )
    metrics = summary["metrics"]
    print(f"schema={summary['schema']}")
    print(f"capture={args.capture_manifest}")
    print(f"predictions={args.predictions}")
    print(f"sample_id={summary['sample']['sample_id']}")
    print(f"candidate_id={summary['candidate']['candidate_id']}")
    print(f"identity_eval_status={summary['status']}")
    print(f"idf1={metrics['idf1']:.6f}")
    print(f"track_retrieval_recall_at_1={metrics['track_retrieval_recall_at_1']:.6f}")
    print(f"long_term_drift_rate={metrics['long_term_drift_rate']:.6f}")
    print(f"fragmentation_rate={metrics['fragmentation_rate']:.6f}")
    print(f"swap_rate={metrics['swap_rate']:.6f}")
    print(f"identity_collapse={str(metrics['identity_collapse']).lower()}")
    noise_robustness = metrics["reconstruction_noise_robustness"]
    noise_text = f"{noise_robustness:.6f}" if noise_robustness is not None else "missing"
    print(f"reconstruction_noise_robustness={noise_text}")
    print(
        "reconstruction_noise_variant_count="
        f"{metrics['reconstruction_noise_variant_count']}"
    )
    print(f"track_coverage={metrics['track_coverage']:.6f}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.controlled_real_output:
        write_json(args.controlled_real_output, summary["controlled_real_manifest"])
        print(f"controlled_real_manifest={args.controlled_real_output}")
    if args.require_pass and summary["status"] != "objectstate_controlled_identity_eval_pass":
        raise ValueError("controlled identity eval did not pass")


def _object_state_eval_controlled_prediction(args: argparse.Namespace) -> None:
    capture = read_objectstate_controlled_capture_manifest(args.capture_manifest)
    candidates = read_objectstate_controlled_prediction_candidates(args.predictions)
    summary = evaluate_objectstate_controlled_prediction_candidates(
        capture,
        candidates,
        thresholds=ObjectStateControlledPredictionThresholds(
            max_state_ade=args.max_state_ade,
            max_prediction_gap_vs_history_model=(
                args.max_prediction_gap_vs_history_model
            ),
            max_error_ratio_vs_history_model=args.max_error_ratio_vs_history_model,
            min_prediction_count=args.min_prediction_count,
        ),
    )
    metrics = summary["metrics"]
    print(f"schema={summary['schema']}")
    print(f"capture={args.capture_manifest}")
    print(f"predictions={args.predictions}")
    print(f"sample_id={summary['sample']['sample_id']}")
    print(f"candidate_id={summary['candidate']['candidate_id']}")
    print(f"prediction_eval_status={summary['status']}")
    print(f"state_ade={metrics['state_ade']:.6f}")
    print(f"history_ade={metrics['history_ade']:.6f}")
    print(
        "prediction_gap_vs_history_model="
        f"{metrics['prediction_gap_vs_history_model']:.6f}"
    )
    print(f"error_ratio_vs_history_model={metrics['error_ratio_vs_history_model']:.6f}")
    print(f"prediction_count={metrics['prediction_count']}")
    print(f"mean_horizon_seconds={metrics['mean_horizon_seconds']:.6f}")
    print(f"max_horizon_seconds={metrics['max_horizon_seconds']:.6f}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.controlled_real_output:
        write_json(args.controlled_real_output, summary["controlled_real_manifest"])
        print(f"controlled_real_manifest={args.controlled_real_output}")
    if (
        args.require_pass
        and summary["status"] != "objectstate_controlled_prediction_eval_pass"
    ):
        raise ValueError("controlled prediction eval did not pass")


def _object_state_eval_controlled_intervention(args: argparse.Namespace) -> None:
    capture = read_objectstate_controlled_capture_manifest(args.capture_manifest)
    candidates = read_objectstate_controlled_intervention_candidates(args.interventions)
    summary = evaluate_objectstate_controlled_intervention_candidates(
        capture,
        candidates,
        thresholds=ObjectStateControlledInterventionThresholds(
            max_action_conditioned_ade=args.max_action_conditioned_ade,
            min_counterfactual_outcome_accuracy=(
                args.min_counterfactual_outcome_accuracy
            ),
            max_wrong_direction_rate=args.max_wrong_direction_rate,
            min_intervention_gain=args.min_intervention_gain,
            min_intervention_count=args.min_intervention_count,
        ),
    )
    metrics = summary["metrics"]
    print(f"schema={summary['schema']}")
    print(f"capture={args.capture_manifest}")
    print(f"interventions={args.interventions}")
    print(f"sample_id={summary['sample']['sample_id']}")
    print(f"candidate_id={summary['candidate']['candidate_id']}")
    print(f"intervention_eval_status={summary['status']}")
    print(f"action_conditioned_ade={metrics['action_conditioned_ade']:.6f}")
    print(f"no_action_ade={metrics['no_action_ade']:.6f}")
    print(f"intervention_gain={metrics['intervention_gain']:.6f}")
    print(
        "counterfactual_outcome_accuracy="
        f"{metrics['counterfactual_outcome_accuracy']:.6f}"
    )
    print(f"wrong_direction_rate={metrics['wrong_direction_rate']:.6f}")
    print(f"intervention_count={metrics['intervention_count']}")
    print(f"mean_horizon_seconds={metrics['mean_horizon_seconds']:.6f}")
    print(f"max_horizon_seconds={metrics['max_horizon_seconds']:.6f}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if args.controlled_real_output:
        write_json(args.controlled_real_output, summary["controlled_real_manifest"])
        print(f"controlled_real_manifest={args.controlled_real_output}")
    if (
        args.require_pass
        and summary["status"] != "objectstate_controlled_intervention_eval_pass"
    ):
        raise ValueError("controlled intervention eval did not pass")


def _object_state_export_identity_predictions(args: argparse.Namespace) -> None:
    capture = read_objectstate_controlled_capture_manifest(args.capture_manifest)
    artifact = read_trainable_kernel_identity_source(args.trainable_artifact)
    artifact_refs = (
        tuple(args.artifact_refs)
        if args.artifact_refs
        else (str(args.trainable_artifact),)
    )
    predictions = objectstate_identity_predictions_from_trainable_artifact(
        capture,
        artifact,
        candidate_id=args.candidate_id,
        source=args.source,
        artifact_refs=artifact_refs,
        max_centroid_distance=args.max_centroid_distance,
    )
    write_json(args.output, predictions)
    print(f"schema={predictions['schema']}")
    print(f"capture={args.capture_manifest}")
    print(f"trainable_artifact={args.trainable_artifact}")
    print(f"sample_id={predictions['sample_id']}")
    print(f"candidate_id={predictions['candidate']['candidate_id']}")
    print(f"prediction_count={len(predictions['predictions'])}")
    print(f"output={args.output}")


def _object_state_controlled_identity_handoff(args: argparse.Namespace) -> None:
    capture = read_objectstate_controlled_capture_manifest(args.capture_manifest)
    artifact = read_trainable_kernel_identity_source(args.trainable_artifact)
    capture_root = (
        args.capture_root
        if args.capture_root is not None
        else args.capture_manifest.parent
    )
    artifact_refs = (
        tuple(args.artifact_refs)
        if args.artifact_refs
        else (str(args.trainable_artifact),)
    )
    summary = objectstate_controlled_identity_handoff(
        capture,
        artifact,
        candidate_id=args.candidate_id,
        source=args.source,
        artifact_refs=artifact_refs,
        max_centroid_distance=args.max_centroid_distance,
        identity_thresholds=ObjectStateControlledIdentityThresholds(
            min_idf1=args.min_idf1,
            min_track_retrieval_recall_at_1=args.min_track_retrieval_recall_at_1,
            max_fragmentation_rate=args.max_fragmentation_rate,
            max_long_term_drift_rate=args.max_long_term_drift_rate,
            max_swap_rate=args.max_swap_rate,
            min_reconstruction_noise_robustness=(
                args.min_reconstruction_noise_robustness
            ),
            min_reconstruction_noise_variants=args.min_reconstruction_noise_variants,
            require_no_identity_collapse=not args.allow_identity_collapse,
        ),
        synthetic_smoke_passed=not args.synthetic_smoke_failed,
        min_real_or_public_rows=args.min_real_or_public_rows,
        capture_root=capture_root,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
        candidate_artifact_path=args.trainable_artifact,
        min_candidate_artifact_bytes=args.min_candidate_artifact_bytes,
        hash_candidate_artifact=args.hash_candidate_artifact,
        min_identity_scenario_frames=args.min_identity_scenario_frames,
        min_occlusion_fraction=args.min_occlusion_fraction,
        min_view_conditions=args.min_view_conditions,
        min_lighting_conditions=args.min_lighting_conditions,
        min_camera_motion_m=args.min_camera_motion_m,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    capture_manifest_path = output_dir / "capture-manifest.json"
    capture_file_audit_path = output_dir / "capture-file-audit.json"
    capture_missing_files_path = output_dir / "capture-missing-files.md"
    candidate_artifact_file_audit_path = output_dir / "candidate-artifact-file-audit.json"
    identity_scenario_audit_path = output_dir / "identity-scenario-audit.json"
    predictions_path = output_dir / "identity-predictions.json"
    identity_eval_path = output_dir / "identity-eval-summary.json"
    controlled_real_path = output_dir / "controlled-real.json"
    controlled_real_summary_path = output_dir / "controlled-real-summary.json"
    blocked_rows_path = output_dir / "blocked-rows.md"
    handoff_path = output_dir / "handoff-summary.json"
    identity_evidence_package_path = (
        output_dir / "identity-evidence-package-summary.json"
    )
    write_json(capture_manifest_path, capture)
    write_json(capture_file_audit_path, summary["capture_file_audit"])
    capture_missing_files_path.write_text(
        summary["capture_file_audit"]["missing_files_markdown"],
        encoding="utf-8",
    )
    write_json(
        candidate_artifact_file_audit_path,
        summary["candidate_artifact_file_audit"],
    )
    write_json(identity_scenario_audit_path, summary["identity_scenario_audit"])
    write_json(predictions_path, summary["identity_predictions"])
    write_json(identity_eval_path, summary["identity_eval"])
    write_json(controlled_real_path, summary["controlled_real_manifest"])
    write_json(controlled_real_summary_path, summary["controlled_real_summary"])
    blocked_rows_path.write_text(
        summary["controlled_real_summary"]["blocked_rows_markdown"],
        encoding="utf-8",
    )
    write_json(handoff_path, summary)
    identity_evidence_package = objectstate_controlled_identity_evidence_package(
        output_dir
    )
    write_json(identity_evidence_package_path, identity_evidence_package)
    metrics = summary["identity_eval"]["metrics"]
    gate = summary["controlled_real_summary"]["gate"]
    scenario_coverage = summary["identity_scenario_audit"]["scenario_coverage"]
    print(f"schema={summary['schema']}")
    print(f"capture={args.capture_manifest}")
    print(f"capture_root={capture_root}")
    print(f"trainable_artifact={args.trainable_artifact}")
    print(f"output_dir={output_dir}")
    print(f"sample_id={summary['sample']['sample_id']}")
    print(f"candidate_id={summary['candidate']['candidate_id']}")
    print(f"handoff_status={summary['status']}")
    print(f"capture_file_audit_status={summary['capture_file_audit']['status']}")
    print(
        "candidate_artifact_file_audit_status="
        f"{summary['candidate_artifact_file_audit']['status']}"
    )
    print(
        "candidate_artifact_ref_match="
        f"{str(summary['candidate_artifact_ref_match']['matches']).lower()}"
    )
    print(f"identity_scenario_audit_status={summary['identity_scenario_audit']['status']}")
    print(
        "identity_scenario_view_conditions="
        f"{scenario_coverage['view_condition_count']}"
    )
    print(
        "identity_scenario_lighting_conditions="
        f"{scenario_coverage['lighting_condition_count']}"
    )
    print(
        "identity_scenario_max_camera_translation_m="
        f"{scenario_coverage['max_camera_translation_m']:.6f}"
    )
    print(f"identity_eval_status={summary['identity_eval']['status']}")
    print(f"identity_gate_status={gate['status']}")
    print(f"idf1={metrics['idf1']:.6f}")
    print(f"track_retrieval_recall_at_1={metrics['track_retrieval_recall_at_1']:.6f}")
    print(f"long_term_drift_rate={metrics['long_term_drift_rate']:.6f}")
    print(f"fragmentation_rate={metrics['fragmentation_rate']:.6f}")
    print(f"swap_rate={metrics['swap_rate']:.6f}")
    print(
        "identity_evidence_package_status="
        f"{identity_evidence_package['status']}"
    )
    print(
        "identity_evidence_package_reviewable="
        f"{str(identity_evidence_package['status'].endswith('_reviewable')).lower()}"
    )
    noise_robustness = metrics["reconstruction_noise_robustness"]
    noise_text = f"{noise_robustness:.6f}" if noise_robustness is not None else "missing"
    print(f"reconstruction_noise_robustness={noise_text}")
    print(
        "reconstruction_noise_variant_count="
        f"{metrics['reconstruction_noise_variant_count']}"
    )
    print(f"blocked_rows={summary['controlled_real_summary']['blocked_row_count']}")
    print(f"capture_manifest={capture_manifest_path}")
    print(f"capture_file_audit={capture_file_audit_path}")
    print(f"capture_missing_files={capture_missing_files_path}")
    print(f"candidate_artifact_file_audit={candidate_artifact_file_audit_path}")
    print(f"identity_scenario_audit={identity_scenario_audit_path}")
    print(f"predictions={predictions_path}")
    print(f"identity_eval={identity_eval_path}")
    print(f"controlled_real_manifest={controlled_real_path}")
    print(f"controlled_real_summary={controlled_real_summary_path}")
    print(f"blocked_rows_markdown={blocked_rows_path}")
    print(f"handoff_summary={handoff_path}")
    print(f"identity_evidence_package_summary={identity_evidence_package_path}")
    if args.require_pass and summary["status"] != "objectstate_controlled_identity_handoff_pass":
        raise ValueError("controlled identity handoff did not pass")


def _object_state_controlled_identity_bundle_handoff(args: argparse.Namespace) -> None:
    artifact = read_trainable_kernel_identity_source(args.trainable_artifact)
    artifact_refs = (
        tuple(args.artifact_refs)
        if args.artifact_refs
        else (str(args.trainable_artifact),)
    )
    summary = objectstate_controlled_identity_bundle_handoff(
        args.bundle_root,
        artifact,
        sample_json=args.sample_json,
        objects_csv=args.objects_csv,
        frames_csv=args.frames_csv,
        annotations_csv=args.annotations_csv,
        actions_csv=args.actions_csv,
        require_prediction_ready=args.require_prediction_ready,
        require_intervention_ready=args.require_intervention_ready,
        candidate_id=args.candidate_id,
        source=args.source,
        artifact_refs=artifact_refs,
        max_centroid_distance=args.max_centroid_distance,
        identity_thresholds=ObjectStateControlledIdentityThresholds(
            min_idf1=args.min_idf1,
            min_track_retrieval_recall_at_1=args.min_track_retrieval_recall_at_1,
            max_fragmentation_rate=args.max_fragmentation_rate,
            max_long_term_drift_rate=args.max_long_term_drift_rate,
            max_swap_rate=args.max_swap_rate,
            min_reconstruction_noise_robustness=(
                args.min_reconstruction_noise_robustness
            ),
            min_reconstruction_noise_variants=args.min_reconstruction_noise_variants,
            require_no_identity_collapse=not args.allow_identity_collapse,
        ),
        synthetic_smoke_passed=not args.synthetic_smoke_failed,
        min_real_or_public_rows=args.min_real_or_public_rows,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
        candidate_artifact_path=args.trainable_artifact,
        min_candidate_artifact_bytes=args.min_candidate_artifact_bytes,
        hash_candidate_artifact=args.hash_candidate_artifact,
        min_identity_scenario_frames=args.min_identity_scenario_frames,
        min_occlusion_fraction=args.min_occlusion_fraction,
        min_view_conditions=args.min_view_conditions,
        min_lighting_conditions=args.min_lighting_conditions,
        min_camera_motion_m=args.min_camera_motion_m,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    acceptance = summary["capture_bundle_acceptance"]
    import_summary = acceptance["import_summary"]
    bundle_file_audit = acceptance["capture_file_audit"]
    handoff = summary["identity_handoff"]

    capture_manifest_path = output_dir / "capture-manifest.json"
    bundle_acceptance_path = output_dir / "bundle-acceptance-summary.json"
    bundle_import_path = output_dir / "bundle-import-summary.json"
    bundle_file_audit_path = output_dir / "bundle-file-audit.json"
    bundle_missing_files_path = output_dir / "bundle-missing-files.md"
    controlled_real_seed_path = output_dir / "controlled-real-seed.json"
    capture_file_audit_path = output_dir / "capture-file-audit.json"
    capture_missing_files_path = output_dir / "capture-missing-files.md"
    candidate_artifact_file_audit_path = output_dir / "candidate-artifact-file-audit.json"
    identity_scenario_audit_path = output_dir / "identity-scenario-audit.json"
    predictions_path = output_dir / "identity-predictions.json"
    identity_eval_path = output_dir / "identity-eval-summary.json"
    controlled_real_path = output_dir / "controlled-real.json"
    controlled_real_summary_path = output_dir / "controlled-real-summary.json"
    blocked_rows_path = output_dir / "blocked-rows.md"
    handoff_path = output_dir / "handoff-summary.json"
    bundle_handoff_path = output_dir / "bundle-handoff-summary.json"
    identity_evidence_package_path = (
        output_dir / "identity-evidence-package-summary.json"
    )

    write_json(capture_manifest_path, import_summary["manifest"])
    write_json(bundle_acceptance_path, acceptance)
    write_json(bundle_import_path, import_summary)
    write_json(bundle_file_audit_path, bundle_file_audit)
    bundle_missing_files_path.write_text(
        bundle_file_audit["missing_files_markdown"],
        encoding="utf-8",
    )
    write_json(
        controlled_real_seed_path,
        import_summary["capture_summary"]["controlled_real_manifest_seed"],
    )
    write_json(capture_file_audit_path, handoff["capture_file_audit"])
    capture_missing_files_path.write_text(
        handoff["capture_file_audit"]["missing_files_markdown"],
        encoding="utf-8",
    )
    write_json(
        candidate_artifact_file_audit_path,
        handoff["candidate_artifact_file_audit"],
    )
    write_json(identity_scenario_audit_path, handoff["identity_scenario_audit"])
    write_json(predictions_path, handoff["identity_predictions"])
    write_json(identity_eval_path, handoff["identity_eval"])
    write_json(controlled_real_path, handoff["controlled_real_manifest"])
    write_json(controlled_real_summary_path, handoff["controlled_real_summary"])
    blocked_rows_path.write_text(
        handoff["controlled_real_summary"]["blocked_rows_markdown"],
        encoding="utf-8",
    )
    write_json(handoff_path, handoff)
    write_json(bundle_handoff_path, summary)
    identity_evidence_package = objectstate_controlled_identity_evidence_package(
        output_dir
    )
    write_json(identity_evidence_package_path, identity_evidence_package)

    metrics = handoff["identity_eval"]["metrics"]
    gate = handoff["controlled_real_summary"]["gate"]
    scenario_coverage = handoff["identity_scenario_audit"]["scenario_coverage"]
    print(f"schema={summary['schema']}")
    print(f"bundle_root={args.bundle_root}")
    print(f"trainable_artifact={args.trainable_artifact}")
    print(f"output_dir={output_dir}")
    print(f"sample_id={summary['sample']['sample_id']}")
    print(f"candidate_id={summary['candidate']['candidate_id']}")
    print(f"bundle_handoff_status={summary['status']}")
    print(f"acceptance_status={acceptance['status']}")
    print(f"handoff_status={handoff['status']}")
    print(f"bundle_file_audit_status={bundle_file_audit['status']}")
    print(f"capture_file_audit_status={handoff['capture_file_audit']['status']}")
    print(
        "candidate_artifact_file_audit_status="
        f"{handoff['candidate_artifact_file_audit']['status']}"
    )
    print(
        "candidate_artifact_ref_match="
        f"{str(handoff['candidate_artifact_ref_match']['matches']).lower()}"
    )
    print(
        "identity_scenario_audit_status="
        f"{handoff['identity_scenario_audit']['status']}"
    )
    print(
        "identity_scenario_view_conditions="
        f"{scenario_coverage['view_condition_count']}"
    )
    print(
        "identity_scenario_lighting_conditions="
        f"{scenario_coverage['lighting_condition_count']}"
    )
    print(
        "identity_scenario_max_camera_translation_m="
        f"{scenario_coverage['max_camera_translation_m']:.6f}"
    )
    print(f"identity_eval_status={handoff['identity_eval']['status']}")
    print(f"identity_gate_status={gate['status']}")
    print(f"idf1={metrics['idf1']:.6f}")
    print(f"track_retrieval_recall_at_1={metrics['track_retrieval_recall_at_1']:.6f}")
    print(f"long_term_drift_rate={metrics['long_term_drift_rate']:.6f}")
    print(f"fragmentation_rate={metrics['fragmentation_rate']:.6f}")
    print(f"swap_rate={metrics['swap_rate']:.6f}")
    print(
        "identity_evidence_package_status="
        f"{identity_evidence_package['status']}"
    )
    print(
        "identity_evidence_package_reviewable="
        f"{str(identity_evidence_package['status'].endswith('_reviewable')).lower()}"
    )
    noise_robustness = metrics["reconstruction_noise_robustness"]
    noise_text = f"{noise_robustness:.6f}" if noise_robustness is not None else "missing"
    print(f"reconstruction_noise_robustness={noise_text}")
    print(
        "reconstruction_noise_variant_count="
        f"{metrics['reconstruction_noise_variant_count']}"
    )
    print(f"blocked_rows={handoff['controlled_real_summary']['blocked_row_count']}")
    print(f"capture_manifest={capture_manifest_path}")
    print(f"bundle_acceptance_summary={bundle_acceptance_path}")
    print(f"bundle_import_summary={bundle_import_path}")
    print(f"bundle_file_audit={bundle_file_audit_path}")
    print(f"bundle_missing_files={bundle_missing_files_path}")
    print(f"controlled_real_seed={controlled_real_seed_path}")
    print(f"capture_file_audit={capture_file_audit_path}")
    print(f"capture_missing_files={capture_missing_files_path}")
    print(f"candidate_artifact_file_audit={candidate_artifact_file_audit_path}")
    print(f"identity_scenario_audit={identity_scenario_audit_path}")
    print(f"predictions={predictions_path}")
    print(f"identity_eval={identity_eval_path}")
    print(f"controlled_real_manifest={controlled_real_path}")
    print(f"controlled_real_summary={controlled_real_summary_path}")
    print(f"blocked_rows_markdown={blocked_rows_path}")
    print(f"handoff_summary={handoff_path}")
    print(f"bundle_handoff_summary={bundle_handoff_path}")
    print(f"identity_evidence_package_summary={identity_evidence_package_path}")
    if (
        args.require_pass
        and summary["status"]
        != "objectstate_controlled_identity_bundle_handoff_pass"
    ):
        raise ValueError("controlled identity bundle handoff did not pass")


def _object_state_controlled_reality_bundle_handoff(args: argparse.Namespace) -> None:
    artifact = read_trainable_kernel_identity_source(args.trainable_artifact)
    predictions = read_objectstate_controlled_prediction_candidates(
        args.prediction_candidates
    )
    interventions = read_objectstate_controlled_intervention_candidates(
        args.intervention_candidates
    )
    artifact_refs = (
        tuple(args.artifact_refs)
        if args.artifact_refs
        else (str(args.trainable_artifact),)
    )
    summary = objectstate_controlled_reality_bundle_handoff(
        args.bundle_root,
        artifact,
        predictions,
        interventions,
        sample_json=args.sample_json,
        objects_csv=args.objects_csv,
        frames_csv=args.frames_csv,
        annotations_csv=args.annotations_csv,
        actions_csv=args.actions_csv,
        candidate_id=args.candidate_id,
        source=args.source,
        artifact_refs=artifact_refs,
        max_centroid_distance=args.max_centroid_distance,
        identity_thresholds=ObjectStateControlledIdentityThresholds(
            min_idf1=args.min_idf1,
            min_track_retrieval_recall_at_1=args.min_track_retrieval_recall_at_1,
            max_fragmentation_rate=args.max_fragmentation_rate,
            max_long_term_drift_rate=args.max_long_term_drift_rate,
            max_swap_rate=args.max_swap_rate,
            min_reconstruction_noise_robustness=(
                args.min_reconstruction_noise_robustness
            ),
            min_reconstruction_noise_variants=args.min_reconstruction_noise_variants,
            require_no_identity_collapse=not args.allow_identity_collapse,
        ),
        prediction_thresholds=ObjectStateControlledPredictionThresholds(
            max_state_ade=args.max_state_ade,
            max_prediction_gap_vs_history_model=(
                args.max_prediction_gap_vs_history_model
            ),
            max_error_ratio_vs_history_model=args.max_error_ratio_vs_history_model,
            min_prediction_count=args.min_prediction_count,
        ),
        intervention_thresholds=ObjectStateControlledInterventionThresholds(
            max_action_conditioned_ade=args.max_action_conditioned_ade,
            min_counterfactual_outcome_accuracy=(
                args.min_counterfactual_outcome_accuracy
            ),
            max_wrong_direction_rate=args.max_wrong_direction_rate,
            min_intervention_gain=args.min_intervention_gain,
            min_intervention_count=args.min_intervention_count,
        ),
        synthetic_smoke_passed=not args.synthetic_smoke_failed,
        min_real_or_public_rows=args.min_real_or_public_rows,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
        candidate_artifact_path=args.trainable_artifact,
        min_candidate_artifact_bytes=args.min_candidate_artifact_bytes,
        hash_candidate_artifact=args.hash_candidate_artifact,
        min_identity_scenario_frames=args.min_identity_scenario_frames,
        min_occlusion_fraction=args.min_occlusion_fraction,
        min_view_conditions=args.min_view_conditions,
        min_lighting_conditions=args.min_lighting_conditions,
        min_camera_motion_m=args.min_camera_motion_m,
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    identity_bundle = summary["identity_bundle_handoff"]
    acceptance = identity_bundle["capture_bundle_acceptance"]
    import_summary = acceptance["import_summary"]
    bundle_file_audit = acceptance["capture_file_audit"]
    identity_handoff = summary["identity_handoff"]
    prediction_eval = summary["prediction_eval"]
    intervention_eval = summary["intervention_eval"]
    controlled_real_summary = summary["controlled_real_summary"]

    capture_manifest_path = output_dir / "capture-manifest.json"
    bundle_acceptance_path = output_dir / "bundle-acceptance-summary.json"
    bundle_import_path = output_dir / "bundle-import-summary.json"
    bundle_file_audit_path = output_dir / "bundle-file-audit.json"
    bundle_missing_files_path = output_dir / "bundle-missing-files.md"
    controlled_real_seed_path = output_dir / "controlled-real-seed.json"
    capture_file_audit_path = output_dir / "capture-file-audit.json"
    capture_missing_files_path = output_dir / "capture-missing-files.md"
    candidate_artifact_file_audit_path = (
        output_dir / "candidate-artifact-file-audit.json"
    )
    identity_scenario_audit_path = output_dir / "identity-scenario-audit.json"
    identity_predictions_path = output_dir / "identity-predictions.json"
    identity_eval_path = output_dir / "identity-eval-summary.json"
    identity_handoff_path = output_dir / "identity-handoff-summary.json"
    prediction_eval_path = output_dir / "prediction-eval-summary.json"
    intervention_eval_path = output_dir / "intervention-eval-summary.json"
    controlled_real_path = output_dir / "controlled-real.json"
    controlled_real_summary_path = output_dir / "controlled-real-summary.json"
    blocked_rows_path = output_dir / "blocked-rows.md"
    reality_handoff_path = output_dir / "reality-bundle-handoff-summary.json"

    write_json(capture_manifest_path, import_summary["manifest"])
    write_json(bundle_acceptance_path, acceptance)
    write_json(bundle_import_path, import_summary)
    write_json(bundle_file_audit_path, bundle_file_audit)
    bundle_missing_files_path.write_text(
        bundle_file_audit["missing_files_markdown"],
        encoding="utf-8",
    )
    write_json(
        controlled_real_seed_path,
        import_summary["capture_summary"]["controlled_real_manifest_seed"],
    )
    write_json(capture_file_audit_path, identity_handoff["capture_file_audit"])
    capture_missing_files_path.write_text(
        identity_handoff["capture_file_audit"]["missing_files_markdown"],
        encoding="utf-8",
    )
    write_json(
        candidate_artifact_file_audit_path,
        identity_handoff["candidate_artifact_file_audit"],
    )
    write_json(identity_scenario_audit_path, identity_handoff["identity_scenario_audit"])
    write_json(identity_predictions_path, summary["identity_predictions"])
    write_json(identity_eval_path, summary["identity_eval"])
    write_json(identity_handoff_path, identity_handoff)
    write_json(prediction_eval_path, prediction_eval)
    write_json(intervention_eval_path, intervention_eval)
    write_json(controlled_real_path, summary["controlled_real_manifest"])
    write_json(controlled_real_summary_path, controlled_real_summary)
    blocked_rows_path.write_text(
        controlled_real_summary["blocked_rows_markdown"],
        encoding="utf-8",
    )
    write_json(reality_handoff_path, summary)

    identity_metrics = summary["identity_eval"]["metrics"]
    prediction_metrics = prediction_eval["metrics"]
    intervention_metrics = intervention_eval["metrics"]
    full_gate = controlled_real_summary["gate"]
    print(f"schema={summary['schema']}")
    print(f"bundle_root={args.bundle_root}")
    print(f"trainable_artifact={args.trainable_artifact}")
    print(f"prediction_candidates={args.prediction_candidates}")
    print(f"intervention_candidates={args.intervention_candidates}")
    print(f"output_dir={output_dir}")
    print(f"sample_id={summary['sample']['sample_id']}")
    print(f"reality_bundle_handoff_status={summary['status']}")
    print(f"acceptance_status={acceptance['status']}")
    print(f"identity_handoff_status={identity_handoff['status']}")
    print(f"identity_eval_status={summary['identity_eval']['status']}")
    print(f"prediction_eval_status={prediction_eval['status']}")
    print(f"intervention_eval_status={intervention_eval['status']}")
    print(f"full_reality_gate_status={full_gate['status']}")
    print(f"idf1={identity_metrics['idf1']:.6f}")
    print(f"state_ade={prediction_metrics['state_ade']:.6f}")
    print(f"history_ade={prediction_metrics['history_ade']:.6f}")
    print(
        "prediction_gap_vs_history_model="
        f"{prediction_metrics['prediction_gap_vs_history_model']:.6f}"
    )
    print(
        "action_conditioned_ade="
        f"{intervention_metrics['action_conditioned_ade']:.6f}"
    )
    print(
        "counterfactual_outcome_accuracy="
        f"{intervention_metrics['counterfactual_outcome_accuracy']:.6f}"
    )
    print(f"wrong_direction_rate={intervention_metrics['wrong_direction_rate']:.6f}")
    print(f"pass_rows={controlled_real_summary['pass_row_count']}")
    print(f"fail_rows={controlled_real_summary['fail_row_count']}")
    print(f"blocked_rows={controlled_real_summary['blocked_row_count']}")
    print(f"capture_manifest={capture_manifest_path}")
    print(f"bundle_acceptance_summary={bundle_acceptance_path}")
    print(f"bundle_import_summary={bundle_import_path}")
    print(f"bundle_file_audit={bundle_file_audit_path}")
    print(f"bundle_missing_files={bundle_missing_files_path}")
    print(f"controlled_real_seed={controlled_real_seed_path}")
    print(f"capture_file_audit={capture_file_audit_path}")
    print(f"capture_missing_files={capture_missing_files_path}")
    print(f"candidate_artifact_file_audit={candidate_artifact_file_audit_path}")
    print(f"identity_scenario_audit={identity_scenario_audit_path}")
    print(f"identity_predictions={identity_predictions_path}")
    print(f"identity_eval={identity_eval_path}")
    print(f"identity_handoff_summary={identity_handoff_path}")
    print(f"prediction_eval={prediction_eval_path}")
    print(f"intervention_eval={intervention_eval_path}")
    print(f"controlled_real_manifest={controlled_real_path}")
    print(f"controlled_real_summary={controlled_real_summary_path}")
    print(f"blocked_rows_markdown={blocked_rows_path}")
    print(f"reality_bundle_handoff_summary={reality_handoff_path}")
    if (
        args.require_pass
        and summary["status"]
        != "objectstate_controlled_reality_bundle_handoff_pass"
    ):
        raise ValueError("controlled reality bundle handoff did not pass")


def _object_state_audit_controlled_reality_bundle_readiness(
    args: argparse.Namespace,
) -> None:
    summary = objectstate_controlled_reality_bundle_readiness(
        args.bundle_root,
        args.trainable_artifact,
        args.prediction_candidates,
        args.intervention_candidates,
        sample_json=args.sample_json,
        objects_csv=args.objects_csv,
        frames_csv=args.frames_csv,
        annotations_csv=args.annotations_csv,
        actions_csv=args.actions_csv,
        max_centroid_distance=args.max_centroid_distance,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
        min_candidate_artifact_bytes=args.min_candidate_artifact_bytes,
        min_identity_scenario_frames=args.min_identity_scenario_frames,
        min_occlusion_fraction=args.min_occlusion_fraction,
        min_view_conditions=args.min_view_conditions,
        min_lighting_conditions=args.min_lighting_conditions,
        min_camera_motion_m=args.min_camera_motion_m,
    )
    if args.summary_output:
        write_json(args.summary_output, summary)
    readiness = summary["readiness"]
    print(f"schema={summary['schema']}")
    print(f"bundle_root={args.bundle_root}")
    print(f"trainable_artifact={args.trainable_artifact}")
    print(f"prediction_candidates={args.prediction_candidates}")
    print(f"intervention_candidates={args.intervention_candidates}")
    print(f"readiness_status={summary['status']}")
    print(
        "full_reality_handoff_ready="
        f"{str(readiness['full_reality_handoff_ready']).lower()}"
    )
    print(
        "capture_bundle_ready="
        f"{str(readiness['capture_bundle_ready']).lower()}"
    )
    print(
        "identity_bundle_handoff_ready="
        f"{str(readiness['identity_bundle_handoff_ready']).lower()}"
    )
    print(
        "trainable_artifact_schema_ready="
        f"{str(readiness['trainable_artifact_schema_ready']).lower()}"
    )
    print(
        "trainable_artifact_binding_ready="
        f"{str(readiness['trainable_artifact_binding_ready']).lower()}"
    )
    print(
        "prediction_candidates_schema_ready="
        f"{str(readiness['prediction_candidates_schema_ready']).lower()}"
    )
    print(
        "prediction_candidates_binding_ready="
        f"{str(readiness['prediction_candidates_binding_ready']).lower()}"
    )
    print(
        "intervention_candidates_schema_ready="
        f"{str(readiness['intervention_candidates_schema_ready']).lower()}"
    )
    print(
        "intervention_candidates_binding_ready="
        f"{str(readiness['intervention_candidates_binding_ready']).lower()}"
    )
    print(f"hard_blocker_count={len(summary['hard_blockers'])}")
    for blocker in summary["hard_blockers"]:
        print(f"hard_blocker={blocker}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    if args.summary_output:
        print(f"summary={args.summary_output}")
    if args.require_ready and not readiness["full_reality_handoff_ready"]:
        raise ValueError("controlled reality bundle is not full-handoff ready")


def _object_state_audit_controlled_reality_evidence_package(
    args: argparse.Namespace,
) -> None:
    summary = objectstate_controlled_reality_evidence_package(
        args.package_root,
        candidate_dir=args.candidate_dir,
        handoff_dir=args.handoff_dir,
    )
    if args.summary_output:
        write_json(args.summary_output, summary)
    row_accounting = summary["row_accounting"]
    print(f"schema={summary['schema']}")
    print(f"package_root={args.package_root}")
    print(f"candidate_dir={summary['candidate_dir']}")
    print(f"handoff_dir={summary['handoff_dir']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"evidence_package_status={summary['status']}")
    print(
        "reviewable="
        f"{str(summary['status'].endswith('_reviewable')).lower()}"
    )
    print(
        "full_reality_handoff_ready="
        f"{str(summary['readiness']['full_reality_handoff_ready']).lower()}"
    )
    print(f"handoff_status={summary['handoff']['status']}")
    print(
        "full_reality_gate_status="
        f"{summary['handoff']['full_reality_gate_status']}"
    )
    print(f"pass_rows={row_accounting['pass_row_count']}")
    print(f"fail_rows={row_accounting['fail_row_count']}")
    print(f"blocked_rows={row_accounting['blocked_row_count']}")
    print(f"issue_count={len(summary['issues'])}")
    for gate, passed in summary["reviewability_gates"].items():
        print(f"reviewability_gate.{gate}={str(passed).lower()}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        print(f"summary={args.summary_output}")
    if (
        args.require_reviewable
        and summary["status"]
        != "objectstate_controlled_reality_evidence_package_reviewable"
    ):
        raise ValueError("controlled reality evidence package is not reviewable")


def _object_state_audit_controlled_prediction_evidence_package(
    args: argparse.Namespace,
) -> None:
    summary = objectstate_controlled_prediction_evidence_package(
        args.package_root,
        candidate_dir=args.candidate_dir,
        capture_manifest=args.capture_manifest,
        acceptance_summary=args.acceptance_summary,
        file_audit=args.file_audit,
        missing_files_markdown=args.missing_files_markdown,
    )
    if args.summary_output:
        write_json(args.summary_output, summary)
    prediction = summary["prediction"]
    print(f"schema={summary['schema']}")
    print(f"package_root={args.package_root}")
    print(f"candidate_dir={summary['candidate_dir']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"prediction_evidence_status={summary['status']}")
    print(
        "reviewable="
        f"{str(summary['status'].endswith('_reviewable')).lower()}"
    )
    print(f"bop_acceptance_status={summary['acceptance']['status']}")
    print(
        "phase1_gaussian_evidence_ready="
        f"{str(summary['acceptance']['phase1_gaussian_evidence_ready']).lower()}"
    )
    print(f"prediction_eval_status={prediction['prediction_eval_status']}")
    print(f"prediction_candidate_count={prediction['prediction_candidate_count']}")
    print(f"prediction_row_status={prediction['prediction_row_status']}")
    print(f"issue_count={len(summary['issues'])}")
    for gate, passed in summary["reviewability_gates"].items():
        print(f"reviewability_gate.{gate}={str(passed).lower()}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        print(f"summary={args.summary_output}")
    if (
        args.require_reviewable
        and summary["status"]
        != "objectstate_controlled_prediction_evidence_package_reviewable"
    ):
        raise ValueError("controlled prediction evidence package is not reviewable")


def _object_state_audit_controlled_identity_evidence_package(
    args: argparse.Namespace,
) -> None:
    summary = objectstate_controlled_identity_evidence_package(
        args.package_root,
        capture_manifest=args.capture_manifest,
        capture_file_audit=args.capture_file_audit,
        capture_missing_files_markdown=args.capture_missing_files_markdown,
        candidate_artifact_file_audit=args.candidate_artifact_file_audit,
        identity_scenario_audit=args.identity_scenario_audit,
        identity_predictions=args.identity_predictions,
        identity_eval=args.identity_eval,
        controlled_real=args.controlled_real,
        controlled_real_summary=args.controlled_real_summary,
        blocked_rows_markdown=args.blocked_rows_markdown,
        handoff_summary=args.handoff_summary,
    )
    if args.summary_output:
        write_json(args.summary_output, summary)
    identity = summary["identity"]
    evidence = summary["evidence"]
    print(f"schema={summary['schema']}")
    print(f"package_root={summary['package_root']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"identity_evidence_status={summary['status']}")
    print(
        "reviewable="
        f"{str(summary['status'].endswith('_reviewable')).lower()}"
    )
    print(
        "capture_file_audit_pass="
        f"{str(evidence['capture_file_audit_pass']).lower()}"
    )
    print(
        "candidate_artifact_file_audit_pass="
        f"{str(evidence['candidate_artifact_file_audit_pass']).lower()}"
    )
    print(
        "candidate_artifact_ref_match="
        f"{str(evidence['candidate_artifact_ref_match']).lower()}"
    )
    print(
        "identity_scenario_audit_pass="
        f"{str(evidence['identity_scenario_audit_pass']).lower()}"
    )
    print(f"identity_eval_status={identity['identity_eval_status']}")
    print(f"identity_prediction_count={identity['identity_prediction_count']}")
    print(f"identity_row_status={identity['identity_row_status']}")
    print(f"issue_count={len(summary['issues'])}")
    for gate, passed in summary["reviewability_gates"].items():
        print(f"reviewability_gate.{gate}={str(passed).lower()}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        print(f"summary={args.summary_output}")
    if (
        args.require_reviewable
        and summary["status"]
        != "objectstate_controlled_identity_evidence_package_reviewable"
    ):
        raise ValueError("controlled identity evidence package is not reviewable")


def _object_state_audit_phase1_evidence_ledger(args: argparse.Namespace) -> None:
    summary = objectstate_phase1_evidence_ledger(
        identity_summaries=tuple(args.identity_summary),
        prediction_summaries=tuple(args.prediction_summary),
        reality_summaries=tuple(args.reality_summary),
        discover_roots=tuple(args.discover_root),
        max_depth=args.max_depth,
    )
    if args.summary_output:
        write_json(args.summary_output, summary)
    stage_summary = summary["stage_summary"]
    discovery = summary["discovery"]
    phase_gates = summary["phase1_evidence_gates"]
    sample_ids = ",".join(summary["sample_scope"]["sample_ids"])
    print(f"schema={summary['schema']}")
    print(f"ledger_status={summary['status']}")
    print(f"maturity={summary['maturity']}")
    print(
        "reviewable="
        f"{str(summary['status'].endswith('_reviewable')).lower()}"
    )
    print(f"sample_ids={sample_ids}")
    print(f"discover_roots={len(discovery['roots'])}")
    print(
        "discovered_identity_summaries="
        f"{discovery['identity_summary_count']}"
    )
    print(
        "discovered_prediction_summaries="
        f"{discovery['prediction_summary_count']}"
    )
    print(
        "discovered_reality_summaries="
        f"{discovery['reality_summary_count']}"
    )
    print(f"identity_packages={stage_summary['identity']['package_count']}")
    print(f"identity_reviewable={stage_summary['identity']['reviewable_count']}")
    print(f"identity_pass_rows={stage_summary['identity']['pass_row_count']}")
    print(f"identity_fail_rows={stage_summary['identity']['fail_row_count']}")
    print(f"prediction_packages={stage_summary['prediction']['package_count']}")
    print(
        "prediction_reviewable="
        f"{stage_summary['prediction']['reviewable_count']}"
    )
    print(f"prediction_pass_rows={stage_summary['prediction']['pass_row_count']}")
    print(f"prediction_fail_rows={stage_summary['prediction']['fail_row_count']}")
    print(f"full_reality_packages={stage_summary['full_reality']['package_count']}")
    print(
        "full_reality_reviewable="
        f"{stage_summary['full_reality']['reviewable_count']}"
    )
    print(
        "full_reality_pass_rows="
        f"{stage_summary['full_reality']['pass_row_count']}"
    )
    print(
        "full_reality_fail_rows="
        f"{stage_summary['full_reality']['fail_row_count']}"
    )
    print(
        "full_reality_blocked_rows="
        f"{stage_summary['full_reality']['blocked_row_count']}"
    )
    for gate, passed in phase_gates.items():
        print(f"phase1_gate.{gate}={str(passed).lower()}")
    for issue in discovery["issues"]:
        print(f"discovery_issue={issue}")
    print(f"issue_count={len(summary['issues'])}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        print(f"summary={args.summary_output}")
    if (
        args.require_reviewable
        and summary["status"] != "objectstate_phase1_evidence_ledger_reviewable"
    ):
        raise ValueError("phase1 evidence ledger is not reviewable")


def _object_state_audit_bop_cross_sample_ledger(args: argparse.Namespace) -> None:
    summary = objectstate_bop_cross_sample_ledger(
        local_row_summaries=tuple(args.local_row_summary),
        discover_roots=tuple(args.discover_root),
        max_depth=args.max_depth,
        min_reviewable_samples=args.min_reviewable_samples,
        min_scene_or_category_coverage=args.min_scene_or_category_coverage,
    )
    if args.summary_output:
        write_json(args.summary_output, summary)
    if args.table_output:
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        args.table_output.write_text(
            summary["sample_table_markdown"],
            encoding="utf-8",
        )
    sample_summary = summary["sample_summary"]
    candidate_gate = summary["candidate_gate"]
    discovery = summary["discovery"]
    print(f"schema={summary['schema']}")
    print(f"bop_cross_sample_ledger_status={summary['status']}")
    print(f"maturity={summary['maturity']}")
    print(
        "candidate_cross_sample_ready="
        f"{str(candidate_gate['candidate_cross_sample_ready']).lower()}"
    )
    print(f"summary_count={sample_summary['summary_count']}")
    print(f"valid_summary_count={sample_summary['valid_summary_count']}")
    print(f"sample_count={sample_summary['sample_count']}")
    print(f"reviewable_samples={sample_summary['reviewable_sample_count']}")
    print(
        "identity_prediction_reviewable_samples="
        f"{sample_summary['identity_prediction_reviewable_sample_count']}"
    )
    print(
        "identity_pass_samples="
        f"{sample_summary['identity_pass_sample_count']}"
    )
    print(
        "prediction_pass_samples="
        f"{sample_summary['prediction_pass_sample_count']}"
    )
    print(f"scene_root_count={sample_summary['scene_root_count']}")
    print(f"object_category_count={sample_summary['object_category_count']}")
    print(f"scenario_count={sample_summary['scenario_count']}")
    print(f"discover_roots={len(discovery['roots'])}")
    print(f"discovered_local_row_summaries={discovery['local_row_summary_count']}")
    for gate, passed in summary["audit_gates"].items():
        print(f"audit_gate.{gate}={str(passed).lower()}")
    for gate, passed in candidate_gate["gates"].items():
        print(f"candidate_gate.{gate}={str(passed).lower()}")
    print(f"issue_count={len(summary['issues'])}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        print(f"summary={args.summary_output}")
    if args.table_output:
        print(f"sample_table={args.table_output}")
    if (
        args.require_reviewable
        and summary["status"] != "objectstate_bop_cross_sample_ledger_reviewable"
    ):
        raise ValueError("BOP cross-sample ledger is not reviewable")
    if (
        args.require_candidate_ready
        and not candidate_gate["candidate_cross_sample_ready"]
    ):
        raise ValueError("BOP cross-sample candidate gate is not ready")


def _object_state_bop_prediction_baseline_handoff(args: argparse.Namespace) -> None:
    summary = objectstate_bop_prediction_baseline_handoff(
        args.scene_root,
        output_root=args.output_root,
        sample_id=args.sample_id,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license,
        rgb_dir=args.rgb_dir,
        gaussian_dir=args.gaussian_dir,
        condition_sidecar=args.condition_sidecar,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        policy=args.policy,
        candidate_id=args.candidate_id,
        candidate_source=args.candidate_source,
        artifact_ref=args.artifact_ref,
        confidence=args.confidence,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
        prediction_thresholds=ObjectStateControlledPredictionThresholds(
            max_state_ade=args.max_state_ade,
            max_prediction_gap_vs_history_model=(
                args.max_prediction_gap_vs_history_model
            ),
            max_error_ratio_vs_history_model=args.max_error_ratio_vs_history_model,
            min_prediction_count=args.min_prediction_count,
        ),
        force=args.force,
    )
    readiness = summary["readiness"]
    prediction_eval = summary["prediction_eval_summary"]
    prediction = summary["prediction_evidence_package"]["prediction"]
    print(f"schema={summary['schema']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"output_root={summary['output_root']}")
    print(f"candidate_dir={summary['candidate_dir']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"bop_prediction_baseline_handoff_status={summary['status']}")
    print(
        "bop_acceptance_pass="
        f"{str(readiness['bop_acceptance_pass']).lower()}"
    )
    print(
        "phase1_gaussian_evidence_ready="
        f"{str(readiness['phase1_gaussian_evidence_ready']).lower()}"
    )
    print(
        "baseline_candidates_ready="
        f"{str(readiness['baseline_candidates_ready']).lower()}"
    )
    print(f"prediction_eval_status={prediction_eval['status']}")
    print(
        "prediction_evidence_package_reviewable="
        f"{str(readiness['prediction_evidence_package_reviewable']).lower()}"
    )
    print(
        "phase1_evidence_ledger_prediction_reviewable="
        f"{str(readiness['phase1_evidence_ledger_prediction_reviewable']).lower()}"
    )
    print(f"prediction_candidate_count={prediction['prediction_candidate_count']}")
    print(f"prediction_row_status={prediction['prediction_row_status']}")
    for key, path in summary["files"].items():
        print(f"{key}={path}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if (
        args.require_reviewable
        and summary["status"]
        != "objectstate_bop_prediction_baseline_handoff_reviewable"
    ):
        raise ValueError("BOP prediction baseline handoff is not reviewable")


def _object_state_bop_identity_handoff(args: argparse.Namespace) -> None:
    summary = objectstate_bop_identity_handoff(
        args.scene_root,
        output_root=args.output_root,
        sample_id=args.sample_id,
        candidate_artifact=args.candidate_artifact,
        identity_dir=args.identity_dir,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license,
        rgb_dir=args.rgb_dir,
        gaussian_dir=args.gaussian_dir,
        condition_sidecar=args.condition_sidecar,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        candidate_id=args.candidate_id,
        candidate_source=args.candidate_source,
        max_centroid_distance=args.max_centroid_distance,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
        min_candidate_artifact_bytes=args.min_candidate_artifact_bytes,
        hash_candidate_artifact=args.hash_candidate_artifact,
        min_identity_scenario_frames=args.min_identity_scenario_frames,
        min_occlusion_fraction=args.min_occlusion_fraction,
        min_view_conditions=args.min_view_conditions,
        min_lighting_conditions=args.min_lighting_conditions,
        min_camera_motion_m=args.min_camera_motion_m,
        identity_thresholds=ObjectStateControlledIdentityThresholds(
            min_idf1=args.min_idf1,
            min_track_retrieval_recall_at_1=args.min_track_retrieval_recall_at_1,
            max_fragmentation_rate=args.max_fragmentation_rate,
            max_long_term_drift_rate=args.max_long_term_drift_rate,
            max_swap_rate=args.max_swap_rate,
            min_reconstruction_noise_robustness=(
                args.min_reconstruction_noise_robustness
            ),
            min_reconstruction_noise_variants=args.min_reconstruction_noise_variants,
            require_no_identity_collapse=not args.allow_identity_collapse,
        ),
        synthetic_smoke_passed=not args.synthetic_smoke_failed,
        min_real_or_public_rows=args.min_real_or_public_rows,
        force=args.force,
    )
    gates = summary["reviewability_gates"]
    pass_gates = summary["pass_gates"]
    handoff = summary["identity_handoff"]
    identity_eval = handoff["identity_eval"]
    metrics = identity_eval["metrics"]
    evidence = summary["identity_evidence_package"]
    ledger = summary["phase1_evidence_ledger_summary"]
    print(f"schema={summary['schema']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"output_root={summary['output_root']}")
    print(f"identity_dir={summary['identity_dir']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"candidate_artifact={summary['candidate']['artifact_ref']}")
    print(f"bop_identity_handoff_status={summary['status']}")
    for gate, passed in gates.items():
        print(f"reviewability.{gate}={str(passed).lower()}")
    for gate, passed in pass_gates.items():
        print(f"pass.{gate}={str(passed).lower()}")
    print(f"identity_handoff_status={handoff['status']}")
    print(f"identity_eval_status={identity_eval['status']}")
    print(f"idf1={metrics['idf1']:.6f}")
    print(
        "track_retrieval_recall_at_1="
        f"{metrics['track_retrieval_recall_at_1']:.6f}"
    )
    print(f"long_term_drift_rate={metrics['long_term_drift_rate']:.6f}")
    print(f"fragmentation_rate={metrics['fragmentation_rate']:.6f}")
    print(f"swap_rate={metrics['swap_rate']:.6f}")
    print(f"identity_evidence_package_status={evidence['status']}")
    print(f"phase1_evidence_ledger_maturity={ledger['maturity']}")
    for key, path in summary["files"].items():
        print(f"{key}={path}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if (
        args.require_reviewable
        and summary["status"] != "objectstate_bop_identity_handoff_reviewable"
    ):
        raise ValueError("BOP identity handoff is not reviewable")
    if args.require_pass and not summary["pass_gates"]["identity_handoff_pass"]:
        raise ValueError("BOP identity handoff did not pass")


def _object_state_bop_local_row_handoff(args: argparse.Namespace) -> None:
    summary = objectstate_bop_local_row_handoff(
        args.scene_root,
        output_root=args.output_root,
        sample_id=args.sample_id,
        candidate_artifact=args.candidate_artifact,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license,
        rgb_dir=args.rgb_dir,
        gaussian_dir=args.gaussian_dir,
        condition_sidecar=args.condition_sidecar,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        identity_candidate_id=args.identity_candidate_id,
        identity_candidate_source=args.identity_candidate_source,
        max_centroid_distance=args.max_centroid_distance,
        prediction_policy=args.prediction_policy,
        prediction_candidate_id=args.prediction_candidate_id,
        prediction_candidate_source=args.prediction_candidate_source,
        prediction_confidence=args.prediction_confidence,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
        min_candidate_artifact_bytes=args.min_candidate_artifact_bytes,
        hash_candidate_artifact=args.hash_candidate_artifact,
        min_identity_scenario_frames=args.min_identity_scenario_frames,
        min_occlusion_fraction=args.min_occlusion_fraction,
        min_view_conditions=args.min_view_conditions,
        min_lighting_conditions=args.min_lighting_conditions,
        min_camera_motion_m=args.min_camera_motion_m,
        identity_thresholds=ObjectStateControlledIdentityThresholds(
            min_idf1=args.min_idf1,
            min_track_retrieval_recall_at_1=args.min_track_retrieval_recall_at_1,
            max_fragmentation_rate=args.max_fragmentation_rate,
            max_long_term_drift_rate=args.max_long_term_drift_rate,
            max_swap_rate=args.max_swap_rate,
            min_reconstruction_noise_robustness=(
                args.min_reconstruction_noise_robustness
            ),
            min_reconstruction_noise_variants=args.min_reconstruction_noise_variants,
            require_no_identity_collapse=not args.allow_identity_collapse,
        ),
        prediction_thresholds=ObjectStateControlledPredictionThresholds(
            max_state_ade=args.max_state_ade,
            max_prediction_gap_vs_history_model=(
                args.max_prediction_gap_vs_history_model
            ),
            max_error_ratio_vs_history_model=args.max_error_ratio_vs_history_model,
            min_prediction_count=args.min_prediction_count,
        ),
        synthetic_smoke_passed=not args.synthetic_smoke_failed,
        min_real_or_public_rows=args.min_real_or_public_rows,
        force=args.force,
    )
    reviewability = summary["reviewability_gates"]
    pass_gates = summary["pass_gates"]
    ledger = summary["phase1_evidence_ledger_summary"]
    print(f"schema={summary['schema']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"output_root={summary['output_root']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"candidate_artifact={summary['candidate_artifact']}")
    print(f"bop_local_row_handoff_status={summary['status']}")
    for gate, passed in reviewability.items():
        print(f"reviewability.{gate}={str(passed).lower()}")
    for gate, passed in pass_gates.items():
        print(f"pass.{gate}={str(passed).lower()}")
    print(f"phase1_evidence_ledger_maturity={ledger['maturity']}")
    print(
        "identity_handoff_status="
        f"{summary['identity_handoff']['identity_handoff']['status']}"
    )
    print(
        "prediction_eval_status="
        f"{summary['prediction_handoff']['prediction_eval_summary']['status']}"
    )
    for key, path in summary["files"].items():
        print(f"{key}={path}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if (
        args.require_reviewable
        and summary["status"] != "objectstate_bop_local_row_handoff_reviewable"
    ):
        raise ValueError("BOP local row handoff is not reviewable")
    if args.require_pass and not all(summary["pass_gates"].values()):
        raise ValueError("BOP local row handoff did not pass")


def _object_state_bop_local_row_batch_handoff(args: argparse.Namespace) -> None:
    summary = objectstate_bop_local_row_batch_handoff(
        args.batch_spec,
        output_root=args.output_root,
        min_reviewable_samples=args.min_reviewable_samples,
        min_scene_or_category_coverage=args.min_scene_or_category_coverage,
        force=args.force,
    )
    if args.summary_output:
        write_json(args.summary_output, summary)
    if args.table_output:
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        args.table_output.write_text(
            summary["cross_sample_ledger"]["sample_table_markdown"],
            encoding="utf-8",
        )
    batch = summary["batch"]
    row_counts = summary["row_counts"]
    print(f"schema={summary['schema']}")
    print(f"batch_spec={args.batch_spec}")
    print(f"batch_id={batch['batch_id']}")
    print(f"output_root={batch['output_root']}")
    print(f"bop_local_row_batch_handoff_status={summary['status']}")
    print(f"samples={row_counts['samples']}")
    print(f"reviewable_samples={row_counts['reviewable_samples']}")
    print(
        "identity_prediction_reviewable_samples="
        f"{row_counts['identity_prediction_reviewable_samples']}"
    )
    print(f"identity_pass_samples={row_counts['identity_pass_samples']}")
    print(f"prediction_pass_samples={row_counts['prediction_pass_samples']}")
    for gate, passed in summary["reviewability_gates"].items():
        print(f"reviewability.{gate}={str(passed).lower()}")
    for gate, passed in summary["pass_gates"].items():
        print(f"pass.{gate}={str(passed).lower()}")
    print(
        "cross_sample_ledger_status="
        f"{summary['cross_sample_ledger']['status']}"
    )
    print(
        "cross_sample_ledger_maturity="
        f"{summary['cross_sample_ledger']['maturity']}"
    )
    print(
        "candidate_cross_sample_ready="
        f"{str(summary['pass_gates']['candidate_cross_sample_ready']).lower()}"
    )
    for key, path in summary["files"].items():
        print(f"{key}={path}")
    for record in summary["sample_records"]:
        print(
            "sample="
            f"{record['sample_id']} status={record['status']} "
            f"identity_pass={str(record['pass_gates']['identity_handoff_pass']).lower()} "
            f"prediction_pass={str(record['pass_gates']['prediction_eval_pass']).lower()}"
        )
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        print(f"summary={args.summary_output}")
    if args.table_output:
        print(f"sample_table={args.table_output}")
    if (
        args.require_reviewable
        and summary["status"]
        != "objectstate_bop_local_row_batch_handoff_reviewable"
    ):
        raise ValueError("BOP local row batch handoff is not reviewable")
    if (
        args.require_candidate_ready
        and not summary["pass_gates"]["candidate_cross_sample_ready"]
    ):
        raise ValueError("BOP local row batch candidate gate is not ready")


def _object_state_init_bop_local_row_batch_spec(args: argparse.Namespace) -> None:
    summary = objectstate_bop_local_row_batch_spec_authoring(
        args.samples_csv,
        output=args.output,
        batch_id=args.batch_id,
        batch_output_root=args.batch_output_root,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license_text,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        gaussian_dir=args.gaussian_dir,
        relative_paths=not args.absolute_paths,
    )
    if args.summary_output:
        write_json(args.summary_output, summary)
    row_counts = summary["row_counts"]
    print(f"schema={summary['schema']}")
    print(f"bop_local_row_batch_spec_authoring_status={summary['status']}")
    print(f"samples_csv={summary['samples_csv']}")
    print(f"output={summary['output']}")
    print(f"samples={row_counts['samples']}")
    print(f"scene_roots_present={row_counts['scene_roots_present']}")
    print(f"candidate_artifacts_present={row_counts['candidate_artifacts_present']}")
    print(f"condition_sidecars_declared={row_counts['condition_sidecars_declared']}")
    print(f"condition_sidecars_present={row_counts['condition_sidecars_present']}")
    for gate, passed in summary["readiness"].items():
        print(f"readiness.{gate}={str(passed).lower()}")
    for command in summary["next_commands"]:
        print(f"next_command={command}")
    for blocker in summary["hard_blockers"]:
        print(f"hard_blocker={blocker}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        print(f"summary={args.summary_output}")
    if (
        args.require_inputs
        and summary["status"]
        != "objectstate_bop_local_row_batch_spec_authoring_ready"
    ):
        raise ValueError("BOP local row batch spec inputs are not ready")


def _object_state_audit_bop_local_row_batch_readiness(
    args: argparse.Namespace,
) -> None:
    summary = objectstate_bop_local_row_batch_readiness(
        args.batch_spec,
        output_root=args.output_root,
        min_reviewable_samples=args.min_reviewable_samples,
        min_scene_or_category_coverage=args.min_scene_or_category_coverage,
    )
    if args.summary_output:
        write_json(args.summary_output, summary)
    if args.table_output:
        args.table_output.parent.mkdir(parents=True, exist_ok=True)
        args.table_output.write_text(
            summary["sample_table_markdown"],
            encoding="utf-8",
        )
    batch = summary["batch"]
    sample_summary = summary["sample_summary"]
    coverage = summary["coverage"]
    print(f"schema={summary['schema']}")
    print(f"batch_spec={args.batch_spec}")
    print(f"batch_id={batch['batch_id']}")
    print(f"output_root={batch['output_root']}")
    print(f"bop_local_row_batch_readiness_status={summary['status']}")
    print(f"samples={sample_summary['samples']}")
    print(
        "ready_for_local_row_handoff_samples="
        f"{sample_summary['ready_for_local_row_handoff_samples']}"
    )
    print(
        "identity_prediction_reviewable_samples="
        f"{sample_summary['identity_prediction_reviewable_samples']}"
    )
    print(
        "ready_or_reviewable_samples="
        f"{sample_summary['ready_or_reviewable_samples']}"
    )
    print(f"object_category_count={coverage['object_category_count']}")
    print(f"scenario_count={coverage['scenario_count']}")
    for gate, passed in summary["readiness_gates"].items():
        print(f"readiness.{gate}={str(passed).lower()}")
    print(f"hard_blocker_count={len(summary['hard_blockers'])}")
    for blocker in summary["hard_blockers"]:
        print(f"hard_blocker={blocker}")
    print(f"next_action_count={len(summary['next_actions'])}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    for record in summary["sample_records"]:
        print(
            "sample="
            f"{record['sample_id']} status={record['status']} "
            f"ready={str(record['ready_for_local_row_handoff']).lower()} "
            f"reviewable={str(record['identity_prediction_reviewable']).lower()}"
        )
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        print(f"summary={args.summary_output}")
    if args.table_output:
        print(f"sample_table={args.table_output}")
    if (
        args.require_ready
        and summary["status"] != "objectstate_bop_local_row_batch_readiness_ready"
    ):
        raise ValueError("BOP local row batch readiness is not ready")


def _object_state_audit_bop_phase1_route(args: argparse.Namespace) -> None:
    summary = objectstate_bop_phase1_route_audit(
        args.scene_root,
        output_root=args.output_root,
        sample_id=args.sample_id,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license,
        rgb_dir=args.rgb_dir,
        gaussian_dir=args.gaussian_dir,
        condition_sidecar=args.condition_sidecar,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
    )
    readiness = summary["readiness"]
    print(f"schema={summary['schema']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"output_root={summary['output_root']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"bop_phase1_route_status={summary['status']}")
    for gate, passed in readiness.items():
        print(f"readiness.{gate}={str(passed).lower()}")
    print(f"hard_blocker_count={len(summary['hard_blockers'])}")
    for blocker in summary["hard_blockers"]:
        print(f"hard_blocker={blocker}")
    print(f"next_action_count={len(summary['next_actions'])}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    for key, path in summary["files"].items():
        print(f"{key}={path}")
    print(f"issue_count={len(summary['issues'])}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if (
        args.require_prediction_reviewable
        and summary["status"]
        != "objectstate_bop_phase1_route_audit_prediction_reviewable"
    ):
        raise ValueError("BOP Phase 1 route is not prediction-reviewable")


def _object_state_audit_bop_identity_route(args: argparse.Namespace) -> None:
    summary = objectstate_bop_identity_route_audit(
        args.scene_root,
        output_root=args.output_root,
        sample_id=args.sample_id,
        candidate_artifact=args.candidate_artifact,
        identity_dir=args.identity_dir,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license,
        rgb_dir=args.rgb_dir,
        gaussian_dir=args.gaussian_dir,
        condition_sidecar=args.condition_sidecar,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
        min_identity_scenario_frames=args.min_identity_scenario_frames,
        min_occlusion_fraction=args.min_occlusion_fraction,
        min_view_conditions=args.min_view_conditions,
        min_lighting_conditions=args.min_lighting_conditions,
        min_camera_motion_m=args.min_camera_motion_m,
    )
    readiness = summary["readiness"]
    print(f"schema={summary['schema']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"output_root={summary['output_root']}")
    print(f"identity_dir={summary['identity_dir']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"bop_identity_route_status={summary['status']}")
    for gate, passed in readiness.items():
        print(f"readiness.{gate}={str(passed).lower()}")
    scenario = summary["identity_scenario_metadata_audit"]
    print(f"identity_scenario_metadata_status={scenario['status']}")
    for gate, passed in scenario["readiness"].items():
        print(f"identity_scenario.{gate}={str(passed).lower()}")
    print(f"hard_blocker_count={len(summary['hard_blockers'])}")
    for blocker in summary["hard_blockers"]:
        print(f"hard_blocker={blocker}")
    print(f"next_action_count={len(summary['next_actions'])}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    for key, path in summary["files"].items():
        print(f"{key}={path}")
    print(f"issue_count={len(summary['issues'])}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if (
        args.require_identity_reviewable
        and summary["status"]
        != "objectstate_bop_identity_route_audit_identity_reviewable"
    ):
        raise ValueError("BOP identity route is not identity-reviewable")


def _object_state_audit_bop_phase1_local_row(args: argparse.Namespace) -> None:
    summary = objectstate_bop_phase1_local_row_readiness(
        args.scene_root,
        output_root=args.output_root,
        sample_id=args.sample_id,
        candidate_artifact=args.candidate_artifact,
        identity_dir=args.identity_dir,
        dataset_id=args.dataset_id,
        object_category=args.object_category,
        scenario=args.scenario,
        fps=args.fps,
        license_text=args.license,
        rgb_dir=args.rgb_dir,
        depth_dir=args.depth_dir,
        gaussian_dir=args.gaussian_dir,
        condition_sidecar=args.condition_sidecar,
        max_frames=args.max_frames,
        frame_step=args.frame_step,
        check_artifact_refs=args.check_artifact_refs,
        min_rgb_bytes=args.min_rgb_bytes,
        min_gaussian_bytes=args.min_gaussian_bytes,
        require_frame_formats=not args.no_require_frame_formats,
        hash_files=args.hash_files,
        min_identity_scenario_frames=args.min_identity_scenario_frames,
        min_occlusion_fraction=args.min_occlusion_fraction,
        min_view_conditions=args.min_view_conditions,
        min_lighting_conditions=args.min_lighting_conditions,
        min_camera_motion_m=args.min_camera_motion_m,
    )
    readiness = summary["readiness"]
    routes = summary["routes"]
    print(f"schema={summary['schema']}")
    print(f"scene_root={summary['scene_root']}")
    print(f"output_root={summary['output_root']}")
    print(f"sample_id={summary['sample_id']}")
    print(f"bop_phase1_local_row_status={summary['status']}")
    print(f"blocking_stage={summary['blocking_stage']}")
    print(f"identity_route_status={routes['identity']['status']}")
    print(f"prediction_route_status={routes['prediction']['status']}")
    for gate, passed in readiness.items():
        print(f"readiness.{gate}={str(passed).lower()}")
    rgbd_hint = summary["rgbd_gaussian_export_hint"]
    print(
        "rgbd_export_candidate="
        f"{str(rgbd_hint['rgbd_export_candidate']).lower()}"
    )
    print(f"rgbd_depth_files_present={rgbd_hint['depth_files_present']}")
    print(f"rgbd_missing_depth_files={rgbd_hint['missing_depth_files']}")
    print(f"rgbd_missing_gaussian_files={rgbd_hint['missing_gaussian_files']}")
    if rgbd_hint["recommended_command"]:
        print(f"rgbd_export_command={rgbd_hint['recommended_command']}")
    print(f"hard_blocker_count={len(summary['hard_blockers'])}")
    for blocker in summary["hard_blockers"]:
        print(f"hard_blocker={blocker}")
    print(f"next_action_count={len(summary['next_actions'])}")
    for action in summary["next_actions"]:
        print(f"next_action={action}")
    for key, path in summary["files"].items():
        print(f"{key}={path}")
    print(f"issue_count={len(summary['issues'])}")
    for issue in summary["issues"]:
        print(f"issue={issue}")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
    if (
        args.require_any_reviewable
        and not summary["readiness"]["phase1_has_any_reviewable_evidence"]
    ):
        raise ValueError("BOP Phase 1 local row has no reviewable evidence")
    if (
        args.require_identity_prediction_reviewable
        and not summary["readiness"]["phase1_identity_prediction_reviewable"]
    ):
        raise ValueError(
            "BOP Phase 1 local row is not identity+prediction-reviewable"
        )


def _controlled_real_gate_thresholds(
    args: argparse.Namespace,
) -> ObjectStateRealityGateThresholds:
    return ObjectStateRealityGateThresholds(
        min_real_or_public_rows=args.min_real_or_public_rows,
        require_identity_pass_row=True,
        require_prediction_pass_row=not args.identity_only,
        require_intervention_pass_row=not args.identity_only,
        fail_on_failed_rows=True,
    )


def _print_summary(labels: np.ndarray) -> None:
    for label, count in summarize_labels(labels):
        print(f"object_id={label} count={count}")


def _labels_with_unknown_policy(field, args: argparse.Namespace) -> np.ndarray:
    return field.labels(
        min_confidence=getattr(args, "min_confidence", None),
        unknown_label=getattr(args, "unknown_object_id", None),
    )


def _print_unknown_policy(field, labels: np.ndarray, args: argparse.Namespace) -> None:
    min_confidence = getattr(args, "min_confidence", None)
    if min_confidence is None:
        return
    unknown_label = getattr(args, "unknown_object_id", None)
    unknown = field.slots if unknown_label is None else int(unknown_label)
    print(f"min_confidence={float(min_confidence):.6f}")
    print(f"unknown_object_id={unknown}")
    print(f"unknown_gaussians={int(np.count_nonzero(labels == unknown))}")


def _print_metrics(metrics) -> None:
    print(f"entropy={metrics.entropy:.6f}")
    print(f"normalized_entropy={metrics.normalized_entropy:.6f}")
    print(f"sharpness={metrics.sharpness:.6f}")
    print(f"active_slots={metrics.active_slots}")
    if metrics.smoothness is not None:
        print(f"smoothness={metrics.smoothness:.6f}")


def _output_format(args: argparse.Namespace) -> str | None:
    if args.ascii:
        return "ascii"
    return None


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="objgauss",
        description="Object-aware clustering tools for Gaussian PLY exports.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cluster = subparsers.add_parser("cluster", help="assign object_id labels")
    _add_io_args(cluster)
    cluster.add_argument("--clusters", "-k", type=int, required=True)
    cluster.add_argument("--seed", type=int, default=0)
    cluster.add_argument("--max-iter", type=int, default=100)
    cluster.add_argument("--spatial-weight", type=float, default=1.0)
    cluster.add_argument("--color-weight", type=float, default=0.5)
    cluster.add_argument("--opacity-weight", type=float, default=0.2)
    cluster.add_argument("--no-normalize", action="store_true")
    cluster.add_argument("--colorize", action="store_true")
    cluster.add_argument("--rewrite-sh", action="store_true")
    cluster.set_defaults(handler=_cluster)

    colorize = subparsers.add_parser("colorize", help="add RGB object colors")
    _add_io_args(colorize)
    colorize.add_argument("--object-id-field", default="object_id")
    colorize.add_argument("--rewrite-sh", action="store_true")
    colorize.set_defaults(handler=_colorize)

    filter_parser = subparsers.add_parser("filter", help="keep or remove objects")
    _add_io_args(filter_parser)
    filter_parser.add_argument("--ids", required=True, help="comma-separated object ids")
    filter_parser.add_argument("--mode", choices=("keep", "remove"), required=True)
    filter_parser.add_argument("--object-id-field", default="object_id")
    filter_parser.set_defaults(handler=_filter)

    stats = subparsers.add_parser("stats", help="print object_id counts")
    stats.add_argument("input", type=Path)
    stats.add_argument("--object-id-field", default="object_id")
    stats.set_defaults(handler=_stats)

    convert_splat = subparsers.add_parser(
        "convert-splat",
        help="convert antimatter15/cakewalk .splat to PLY",
    )
    _add_io_args(convert_splat)
    convert_splat.set_defaults(handler=_convert_splat)

    assets = subparsers.add_parser("assets", help="manage ObjGauss asset sources")
    asset_subparsers = assets.add_subparsers(dest="asset_command", required=True)

    assets_list = asset_subparsers.add_parser("list", help="list registered assets")
    assets_list.add_argument("--pullable", action="store_true", help="show automated assets only")
    assets_list.set_defaults(handler=_assets_list)

    assets_pull = asset_subparsers.add_parser("pull", help="download and localize an asset")
    assets_pull.add_argument("asset_id")
    assets_pull.add_argument("--raw-dir", type=Path, default=Path("outputs/assets/raw"))
    assets_pull.add_argument(
        "--converted-dir",
        type=Path,
        default=Path("outputs/assets/converted"),
    )
    assets_pull.add_argument("--public-dir", type=Path, default=Path("public"))
    assets_pull.add_argument(
        "--training-dir",
        type=Path,
        default=Path("outputs/assets/training"),
    )
    assets_pull.add_argument("--clusters", type=int)
    assets_pull.add_argument("--force", action="store_true")
    assets_pull.set_defaults(handler=_assets_pull)

    object_state = subparsers.add_parser(
        "object-state",
        help="run ObjectState kernel diagnostics",
    )
    object_state_subparsers = object_state.add_subparsers(
        dest="object_state_command",
        required=True,
    )
    state_benchmark = object_state_subparsers.add_parser(
        "stability-benchmark",
        help="run the deterministic pre-training ObjectState stability benchmark",
    )
    state_benchmark.add_argument("--output", "-o", required=True, type=Path)
    state_benchmark.add_argument(
        "--report-id",
        default="objectstate-stability-synthetic-v1",
    )
    state_benchmark.add_argument(
        "--strict",
        action="store_true",
        help="fail if the benchmark suite does not satisfy its expected diagnostics",
    )
    state_benchmark.set_defaults(handler=_object_state_stability_benchmark)
    controlled_real_gate = object_state_subparsers.add_parser(
        "controlled-real-gate",
        help="evaluate a controlled real ObjectState evidence manifest",
    )
    controlled_real_gate.add_argument("manifest", type=Path)
    controlled_real_gate.add_argument("--summary-output", type=Path)
    controlled_real_gate.add_argument("--blocked-rows-output", type=Path)
    controlled_real_gate.add_argument(
        "--identity-only",
        action="store_true",
        help=(
            "run the Stage 1 identity-state gate without requiring prediction "
            "or intervention pass rows"
        ),
    )
    controlled_real_gate.add_argument(
        "--synthetic-smoke-failed",
        action="store_true",
        help="mark the synthetic prerequisite smoke gate as failed",
    )
    controlled_real_gate.add_argument("--min-real-or-public-rows", type=int, default=1)
    controlled_real_gate.add_argument("--require-pass", action="store_true")
    controlled_real_gate.set_defaults(handler=_object_state_controlled_real_gate)
    init_controlled_capture = object_state_subparsers.add_parser(
        "init-controlled-capture-bundle",
        help="create a local controlled tabletop capture bundle skeleton",
    )
    init_controlled_capture.add_argument("bundle_root", type=Path)
    init_controlled_capture.add_argument("--sample-id", required=True)
    init_controlled_capture.add_argument(
        "--object-category",
        default="controlled_tabletop",
    )
    init_controlled_capture.add_argument(
        "--scenario",
        default="cross_view_occlusion_reappearance",
    )
    init_controlled_capture.add_argument("--fps", type=float, default=30.0)
    init_controlled_capture.add_argument(
        "--capture-device",
        default="controlled-camera",
    )
    init_controlled_capture.add_argument(
        "--license",
        dest="license_text",
        default="local controlled capture; not public release",
    )
    init_controlled_capture.add_argument(
        "--object",
        action="append",
        dest="objects",
        help="object_id:category or object_id:category:label; repeat per object",
    )
    init_controlled_capture.add_argument("--summary-output", type=Path)
    init_controlled_capture.add_argument(
        "--force",
        action="store_true",
        help="overwrite template files if they already exist",
    )
    init_controlled_capture.set_defaults(
        handler=_object_state_init_controlled_capture_bundle
    )
    audit_controlled_capture_environment = object_state_subparsers.add_parser(
        "audit-controlled-capture-environment",
        help=(
            "audit local camera and reconstruction tool availability before "
            "controlled real capture"
        ),
    )
    audit_controlled_capture_environment.add_argument(
        "--summary-output",
        type=Path,
    )
    audit_controlled_capture_environment.add_argument(
        "--dev-root",
        type=Path,
        default=Path("/dev"),
        help="device root to scan for video*/media* devices",
    )
    audit_controlled_capture_environment.add_argument(
        "--require-ready",
        action="store_true",
        help="fail unless RGB capture and Gaussian reconstruction tools are ready",
    )
    audit_controlled_capture_environment.set_defaults(
        handler=_object_state_audit_controlled_capture_environment
    )
    audit_public_dataset_candidates = object_state_subparsers.add_parser(
        "audit-public-dataset-candidates",
        help=(
            "audit public controlled pose/action dataset candidates for the "
            "ObjectState reality gate"
        ),
    )
    audit_public_dataset_candidates.add_argument("--summary-output", type=Path)
    audit_public_dataset_candidates.add_argument("--markdown-output", type=Path)
    audit_public_dataset_candidates.add_argument(
        "--require-ready",
        action="store_true",
        help="fail because current public candidates still require adapters and local Gaussian evidence",
    )
    audit_public_dataset_candidates.set_defaults(
        handler=_object_state_audit_public_dataset_candidates
    )
    select_bop_phase1_subset = object_state_subparsers.add_parser(
        "select-bop-phase1-subset",
        help=(
            "scan a local BOP dataset or split root and recommend a Phase 1 "
            "scene seed"
        ),
    )
    select_bop_phase1_subset.add_argument("dataset_root", type=Path)
    select_bop_phase1_subset.add_argument("--summary-output", type=Path)
    select_bop_phase1_subset.add_argument(
        "--batch-samples-csv-template-output",
        type=Path,
        help=(
            "optional CSV template for init-bop-local-row-batch-spec; "
            "includes ready scanned scenes only"
        ),
    )
    select_bop_phase1_subset.add_argument(
        "--batch-sample-artifact-root",
        type=Path,
        default=Path("outputs/captures"),
        help=(
            "root used in the CSV template for per-sample objectstates.json "
            "and bop-condition-sidecar.json placeholders"
        ),
    )
    select_bop_phase1_subset.add_argument("--dataset-id", default="bop-ycbv")
    select_bop_phase1_subset.add_argument("--output-root", type=Path)
    select_bop_phase1_subset.add_argument("--object-category", default="bop_objects")
    select_bop_phase1_subset.add_argument("--scenario", default="bop_pose_sequence")
    select_bop_phase1_subset.add_argument("--fps", type=float, default=30.0)
    select_bop_phase1_subset.add_argument(
        "--license-text",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    select_bop_phase1_subset.add_argument("--rgb-dir", default="rgb")
    select_bop_phase1_subset.add_argument("--max-frames", type=int)
    select_bop_phase1_subset.add_argument("--frame-step", type=int, default=1)
    select_bop_phase1_subset.add_argument("--max-depth", type=int, default=3)
    select_bop_phase1_subset.add_argument("--max-scene-candidates", type=int, default=20)
    select_bop_phase1_subset.add_argument("--min-frames", type=int, default=3)
    select_bop_phase1_subset.add_argument("--min-objects", type=int, default=1)
    select_bop_phase1_subset.add_argument(
        "--min-persistent-objects",
        type=int,
        default=1,
        help="minimum objects that must appear in at least two selected frames",
    )
    select_bop_phase1_subset.add_argument(
        "--require-ready",
        action="store_true",
        help="fail unless at least one BOP scene is ready as a Phase 1 seed",
    )
    select_bop_phase1_subset.set_defaults(
        handler=_object_state_select_bop_phase1_subset
    )
    audit_bop_gaussian_evidence = object_state_subparsers.add_parser(
        "audit-bop-gaussian-evidence",
        help=(
            "audit expected per-frame Gaussian evidence for a local BOP scene "
            "before Phase 1 handoff"
        ),
    )
    audit_bop_gaussian_evidence.add_argument("scene_root", type=Path)
    audit_bop_gaussian_evidence.add_argument("--summary-output", type=Path)
    audit_bop_gaussian_evidence.add_argument(
        "--missing-gaussians-output",
        type=Path,
    )
    audit_bop_gaussian_evidence.add_argument("--sample-id", required=True)
    audit_bop_gaussian_evidence.add_argument("--dataset-id", default="bop-ycbv")
    audit_bop_gaussian_evidence.add_argument("--object-category", default="bop_objects")
    audit_bop_gaussian_evidence.add_argument("--scenario", default="bop_pose_sequence")
    audit_bop_gaussian_evidence.add_argument("--fps", type=float, default=30.0)
    audit_bop_gaussian_evidence.add_argument(
        "--license-text",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    audit_bop_gaussian_evidence.add_argument("--rgb-dir", default="rgb")
    audit_bop_gaussian_evidence.add_argument("--gaussian-dir", default="gaussians")
    audit_bop_gaussian_evidence.add_argument(
        "--condition-sidecar",
        type=Path,
        help="optional JSON sidecar with explicit per-frame view, lighting, and camera_pose metadata",
    )
    audit_bop_gaussian_evidence.add_argument("--max-frames", type=int)
    audit_bop_gaussian_evidence.add_argument("--frame-step", type=int, default=1)
    audit_bop_gaussian_evidence.add_argument("--min-rgb-bytes", type=int, default=1)
    audit_bop_gaussian_evidence.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
    )
    audit_bop_gaussian_evidence.add_argument(
        "--no-require-frame-formats",
        action="store_true",
    )
    audit_bop_gaussian_evidence.add_argument("--hash-files", action="store_true")
    audit_bop_gaussian_evidence.add_argument(
        "--dev-root",
        type=Path,
        default=Path("/dev"),
        help="device root passed through to the reconstruction environment preflight",
    )
    audit_bop_gaussian_evidence.add_argument(
        "--require-ready",
        action="store_true",
        help="fail unless every selected frame has valid Gaussian evidence",
    )
    audit_bop_gaussian_evidence.set_defaults(
        handler=_object_state_audit_bop_gaussian_evidence
    )
    export_bop_rgbd_gaussian_evidence = object_state_subparsers.add_parser(
        "export-bop-rgbd-gaussian-evidence",
        help=(
            "export local BOP RGB-D depth backprojection PLY files as "
            "per-frame Gaussian evidence seeds"
        ),
    )
    export_bop_rgbd_gaussian_evidence.add_argument("scene_root", type=Path)
    export_bop_rgbd_gaussian_evidence.add_argument("--summary-output", type=Path)
    export_bop_rgbd_gaussian_evidence.add_argument("--sample-id", required=True)
    export_bop_rgbd_gaussian_evidence.add_argument("--dataset-id", default="bop-ycbv")
    export_bop_rgbd_gaussian_evidence.add_argument(
        "--object-category",
        default="bop_objects",
    )
    export_bop_rgbd_gaussian_evidence.add_argument(
        "--scenario",
        default="bop_pose_sequence",
    )
    export_bop_rgbd_gaussian_evidence.add_argument("--fps", type=float, default=30.0)
    export_bop_rgbd_gaussian_evidence.add_argument(
        "--license-text",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    export_bop_rgbd_gaussian_evidence.add_argument("--rgb-dir", default="rgb")
    export_bop_rgbd_gaussian_evidence.add_argument("--depth-dir", default="depth")
    export_bop_rgbd_gaussian_evidence.add_argument(
        "--gaussian-dir",
        default="gaussians",
    )
    export_bop_rgbd_gaussian_evidence.add_argument("--max-frames", type=int)
    export_bop_rgbd_gaussian_evidence.add_argument(
        "--frame-step",
        type=int,
        default=1,
    )
    export_bop_rgbd_gaussian_evidence.add_argument(
        "--pixel-stride",
        type=int,
        default=1,
    )
    export_bop_rgbd_gaussian_evidence.add_argument(
        "--max-points-per-frame",
        type=int,
        default=50_000,
    )
    export_bop_rgbd_gaussian_evidence.add_argument(
        "--min-depth-m",
        type=float,
        default=0.0,
    )
    export_bop_rgbd_gaussian_evidence.add_argument("--max-depth-m", type=float)
    export_bop_rgbd_gaussian_evidence.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing gaussians/<frame>.ply outputs",
    )
    export_bop_rgbd_gaussian_evidence.add_argument(
        "--ply-format",
        choices=("ascii", "binary_little_endian", "binary_big_endian"),
        default="binary_little_endian",
    )
    export_bop_rgbd_gaussian_evidence.add_argument(
        "--require-ready",
        action="store_true",
        help="fail unless all selected RGB-D frames produced Gaussian evidence PLY files",
    )
    export_bop_rgbd_gaussian_evidence.set_defaults(
        handler=_object_state_export_bop_rgbd_gaussian_evidence
    )
    init_bop_condition_sidecar = object_state_subparsers.add_parser(
        "init-bop-condition-sidecar",
        help=(
            "write a BOP condition sidecar from selected scene frames and an "
            "optional condition CSV"
        ),
    )
    init_bop_condition_sidecar.add_argument("scene_root", type=Path)
    init_bop_condition_sidecar.add_argument("--output", "-o", required=True, type=Path)
    init_bop_condition_sidecar.add_argument("--summary-output", type=Path)
    init_bop_condition_sidecar.add_argument(
        "--condition-csv",
        type=Path,
        help=(
            "CSV with frame_id, view_id, lighting_id, camera_x/y/z, "
            "camera_qx/qy/qz/qw columns"
        ),
    )
    init_bop_condition_sidecar.add_argument(
        "--condition-csv-template-output",
        type=Path,
        help="write a fillable condition CSV template for the selected BOP frames",
    )
    init_bop_condition_sidecar.add_argument("--max-frames", type=int)
    init_bop_condition_sidecar.add_argument("--frame-step", type=int, default=1)
    init_bop_condition_sidecar.add_argument(
        "--default-lighting-id",
        default="bop-default",
        help="lighting id used for frames not overridden by --condition-csv",
    )
    init_bop_condition_sidecar.add_argument(
        "--min-view-conditions",
        type=int,
        default=2,
    )
    init_bop_condition_sidecar.add_argument(
        "--min-lighting-conditions",
        type=int,
        default=2,
    )
    init_bop_condition_sidecar.add_argument(
        "--min-camera-motion-m",
        type=float,
        default=0.01,
    )
    init_bop_condition_sidecar.add_argument(
        "--require-identity-ready",
        action="store_true",
        help="fail unless the generated sidecar satisfies identity metadata coverage",
    )
    init_bop_condition_sidecar.set_defaults(
        handler=_object_state_init_bop_condition_sidecar
    )
    import_bop_capture_scene = object_state_subparsers.add_parser(
        "import-bop-capture-scene",
        help=(
            "convert a local BOP scene folder into an ObjGauss controlled "
            "capture manifest seed"
        ),
    )
    import_bop_capture_scene.add_argument("scene_root", type=Path)
    import_bop_capture_scene.add_argument("--output", "-o", required=True, type=Path)
    import_bop_capture_scene.add_argument("--summary-output", type=Path)
    import_bop_capture_scene.add_argument("--controlled-real-output", type=Path)
    import_bop_capture_scene.add_argument("--sample-id", required=True)
    import_bop_capture_scene.add_argument("--dataset-id", default="bop-ycbv")
    import_bop_capture_scene.add_argument("--object-category", default="bop_objects")
    import_bop_capture_scene.add_argument("--scenario", default="bop_pose_sequence")
    import_bop_capture_scene.add_argument("--fps", type=float, default=30.0)
    import_bop_capture_scene.add_argument(
        "--license-text",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    import_bop_capture_scene.add_argument("--rgb-dir", default="rgb")
    import_bop_capture_scene.add_argument(
        "--condition-sidecar",
        type=Path,
        help="optional JSON sidecar with explicit per-frame view, lighting, and camera_pose metadata",
    )
    import_bop_capture_scene.add_argument("--max-frames", type=int)
    import_bop_capture_scene.add_argument("--frame-step", type=int, default=1)
    import_bop_capture_scene.add_argument(
        "--include-gaussian-refs",
        action="store_true",
        help="include expected per-frame gaussians/<frame>.ply refs in the manifest",
    )
    import_bop_capture_scene.add_argument("--gaussian-dir", default="gaussians")
    import_bop_capture_scene.add_argument("--require-identity-ready", action="store_true")
    import_bop_capture_scene.add_argument("--require-prediction-ready", action="store_true")
    import_bop_capture_scene.set_defaults(handler=_object_state_import_bop_capture_scene)
    accept_bop_capture_scene = object_state_subparsers.add_parser(
        "accept-bop-capture-scene",
        help=(
            "convert a local BOP scene and run controlled capture file audit "
            "before handoff"
        ),
    )
    accept_bop_capture_scene.add_argument("scene_root", type=Path)
    accept_bop_capture_scene.add_argument("--output", "-o", required=True, type=Path)
    accept_bop_capture_scene.add_argument("--summary-output", type=Path)
    accept_bop_capture_scene.add_argument("--file-audit-output", type=Path)
    accept_bop_capture_scene.add_argument("--missing-files-output", type=Path)
    accept_bop_capture_scene.add_argument("--controlled-real-output", type=Path)
    accept_bop_capture_scene.add_argument("--sample-id", required=True)
    accept_bop_capture_scene.add_argument("--dataset-id", default="bop-ycbv")
    accept_bop_capture_scene.add_argument("--object-category", default="bop_objects")
    accept_bop_capture_scene.add_argument("--scenario", default="bop_pose_sequence")
    accept_bop_capture_scene.add_argument("--fps", type=float, default=30.0)
    accept_bop_capture_scene.add_argument(
        "--license-text",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    accept_bop_capture_scene.add_argument("--rgb-dir", default="rgb")
    accept_bop_capture_scene.add_argument(
        "--condition-sidecar",
        type=Path,
        help="optional JSON sidecar with explicit per-frame view, lighting, and camera_pose metadata",
    )
    accept_bop_capture_scene.add_argument("--max-frames", type=int)
    accept_bop_capture_scene.add_argument("--frame-step", type=int, default=1)
    accept_bop_capture_scene.add_argument(
        "--include-gaussian-refs",
        action="store_true",
        help="include expected per-frame gaussians/<frame>.ply refs in the manifest",
    )
    accept_bop_capture_scene.add_argument("--gaussian-dir", default="gaussians")
    accept_bop_capture_scene.add_argument(
        "--require-gaussian-files",
        action="store_true",
        help="require Gaussian files for each selected BOP frame",
    )
    accept_bop_capture_scene.add_argument(
        "--check-artifact-refs",
        action="store_true",
        help="also require sample artifact refs such as scene_gt.json to exist",
    )
    accept_bop_capture_scene.add_argument("--min-rgb-bytes", type=int, default=1)
    accept_bop_capture_scene.add_argument("--min-gaussian-bytes", type=int, default=1)
    accept_bop_capture_scene.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    accept_bop_capture_scene.add_argument("--hash-files", action="store_true")
    accept_bop_capture_scene.add_argument("--require-pass", action="store_true")
    accept_bop_capture_scene.set_defaults(handler=_object_state_accept_bop_capture_scene)
    init_bop_objectstate_artifact_template = object_state_subparsers.add_parser(
        "init-bop-objectstate-artifact-template",
        help=(
            "write a draft-only BOP ObjectState candidate artifact template "
            "after RGB/Gaussian evidence is staged"
        ),
    )
    init_bop_objectstate_artifact_template.add_argument("scene_root", type=Path)
    init_bop_objectstate_artifact_template.add_argument(
        "--output",
        "-o",
        required=True,
        type=Path,
    )
    init_bop_objectstate_artifact_template.add_argument(
        "--summary-output",
        type=Path,
    )
    init_bop_objectstate_artifact_template.add_argument("--sample-id", required=True)
    init_bop_objectstate_artifact_template.add_argument(
        "--dataset-id",
        default="bop-ycbv",
    )
    init_bop_objectstate_artifact_template.add_argument(
        "--object-category",
        default="bop_objects",
    )
    init_bop_objectstate_artifact_template.add_argument(
        "--scenario",
        default="bop_pose_sequence",
    )
    init_bop_objectstate_artifact_template.add_argument(
        "--fps",
        type=float,
        default=30.0,
    )
    init_bop_objectstate_artifact_template.add_argument(
        "--license",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    init_bop_objectstate_artifact_template.add_argument("--rgb-dir", default="rgb")
    init_bop_objectstate_artifact_template.add_argument(
        "--gaussian-dir",
        default="gaussians",
    )
    init_bop_objectstate_artifact_template.add_argument(
        "--condition-sidecar",
        type=Path,
        help="optional JSON sidecar with explicit per-frame view, lighting, and camera_pose metadata",
    )
    init_bop_objectstate_artifact_template.add_argument("--max-frames", type=int)
    init_bop_objectstate_artifact_template.add_argument(
        "--frame-step",
        type=int,
        default=1,
    )
    init_bop_objectstate_artifact_template.add_argument(
        "--candidate-id",
        default="bop-objectstate-candidate",
    )
    init_bop_objectstate_artifact_template.add_argument(
        "--candidate-source",
        default="local-objectstate-model-output",
    )
    init_bop_objectstate_artifact_template.add_argument(
        "--target-artifact-path",
        type=Path,
        help="intended final trainable ObjectState artifact path; defaults next to --output",
    )
    init_bop_objectstate_artifact_template.add_argument(
        "--force",
        action="store_true",
        help="overwrite the draft template if it already exists",
    )
    init_bop_objectstate_artifact_template.set_defaults(
        handler=_object_state_init_bop_objectstate_artifact_template
    )
    finalize_bop_objectstate_artifact_template = object_state_subparsers.add_parser(
        "finalize-bop-objectstate-artifact-template",
        help=(
            "finalize a filled BOP ObjectState candidate template into the "
            "trainable artifact schema accepted by identity route audits"
        ),
    )
    finalize_bop_objectstate_artifact_template.add_argument("template", type=Path)
    finalize_bop_objectstate_artifact_template.add_argument(
        "--output",
        "-o",
        type=Path,
        help="target objectstates.json path; defaults to template target_artifact",
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--summary-output",
        type=Path,
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--scene-root",
        type=Path,
        help="override scene_root stored in the template",
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--dataset-id",
        default="bop-ycbv",
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--object-category",
        default="bop_objects",
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--scenario",
        default="bop_pose_sequence",
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--fps",
        type=float,
        default=30.0,
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--license",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    finalize_bop_objectstate_artifact_template.add_argument("--rgb-dir", default="rgb")
    finalize_bop_objectstate_artifact_template.add_argument(
        "--gaussian-dir",
        default="gaussians",
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--condition-sidecar",
        type=Path,
        help="optional JSON sidecar with explicit per-frame view, lighting, and camera_pose metadata",
    )
    finalize_bop_objectstate_artifact_template.add_argument("--max-frames", type=int)
    finalize_bop_objectstate_artifact_template.add_argument(
        "--frame-step",
        type=int,
        default=1,
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--gt-leakage-tolerance",
        type=float,
        default=1e-9,
        help="reject candidate centroids within this distance of BOP pose GT",
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--reconstruction-noise-robustness",
        type=float,
        help="optional real model robustness evidence to store in identity_evidence",
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--reconstruction-noise-variant-count",
        type=int,
        help="optional robustness variant count paired with --reconstruction-noise-robustness",
    )
    finalize_bop_objectstate_artifact_template.add_argument(
        "--force",
        action="store_true",
        help="overwrite the finalized artifact if it already exists",
    )
    finalize_bop_objectstate_artifact_template.set_defaults(
        handler=_object_state_finalize_bop_objectstate_artifact_template
    )
    audit_bop_phase1_route = object_state_subparsers.add_parser(
        "audit-bop-phase1-route",
        help=(
            "read-only audit of a local BOP scene and output root on the path "
            "to Phase 1 prediction evidence"
        ),
    )
    audit_bop_phase1_route.add_argument("scene_root", type=Path)
    audit_bop_phase1_route.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    audit_bop_phase1_route.add_argument("--summary-output", type=Path)
    audit_bop_phase1_route.add_argument("--sample-id", required=True)
    audit_bop_phase1_route.add_argument("--dataset-id", default="bop-ycbv")
    audit_bop_phase1_route.add_argument("--object-category", default="bop_objects")
    audit_bop_phase1_route.add_argument("--scenario", default="bop_pose_sequence")
    audit_bop_phase1_route.add_argument("--fps", type=float, default=30.0)
    audit_bop_phase1_route.add_argument(
        "--license",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    audit_bop_phase1_route.add_argument("--rgb-dir", default="rgb")
    audit_bop_phase1_route.add_argument("--gaussian-dir", default="gaussians")
    audit_bop_phase1_route.add_argument(
        "--condition-sidecar",
        type=Path,
        help="optional JSON sidecar with explicit per-frame view, lighting, and camera_pose metadata",
    )
    audit_bop_phase1_route.add_argument("--max-frames", type=int)
    audit_bop_phase1_route.add_argument("--frame-step", type=int, default=1)
    audit_bop_phase1_route.add_argument(
        "--check-artifact-refs",
        action="store_true",
        help="also require sample artifact refs such as scene_gt.json to exist",
    )
    audit_bop_phase1_route.add_argument("--min-rgb-bytes", type=int, default=1)
    audit_bop_phase1_route.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
    )
    audit_bop_phase1_route.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    audit_bop_phase1_route.add_argument("--hash-files", action="store_true")
    audit_bop_phase1_route.add_argument(
        "--require-prediction-reviewable",
        action="store_true",
        help="fail unless existing outputs expose reviewable prediction evidence",
    )
    audit_bop_phase1_route.set_defaults(
        handler=_object_state_audit_bop_phase1_route
    )
    audit_bop_identity_route = object_state_subparsers.add_parser(
        "audit-bop-identity-route",
        help=(
            "read-only audit of a local BOP scene and output root on the path "
            "to Phase 1 identity evidence"
        ),
    )
    audit_bop_identity_route.add_argument("scene_root", type=Path)
    audit_bop_identity_route.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    audit_bop_identity_route.add_argument("--summary-output", type=Path)
    audit_bop_identity_route.add_argument("--sample-id", required=True)
    audit_bop_identity_route.add_argument("--candidate-artifact", type=Path)
    audit_bop_identity_route.add_argument("--identity-dir", default="identity-handoff")
    audit_bop_identity_route.add_argument("--dataset-id", default="bop-ycbv")
    audit_bop_identity_route.add_argument("--object-category", default="bop_objects")
    audit_bop_identity_route.add_argument("--scenario", default="bop_pose_sequence")
    audit_bop_identity_route.add_argument("--fps", type=float, default=30.0)
    audit_bop_identity_route.add_argument(
        "--license",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    audit_bop_identity_route.add_argument("--rgb-dir", default="rgb")
    audit_bop_identity_route.add_argument("--gaussian-dir", default="gaussians")
    audit_bop_identity_route.add_argument(
        "--condition-sidecar",
        type=Path,
        help="optional JSON sidecar with explicit per-frame view, lighting, and camera_pose metadata",
    )
    audit_bop_identity_route.add_argument("--max-frames", type=int)
    audit_bop_identity_route.add_argument("--frame-step", type=int, default=1)
    audit_bop_identity_route.add_argument(
        "--check-artifact-refs",
        action="store_true",
        help="also require sample artifact refs such as scene_gt.json to exist",
    )
    audit_bop_identity_route.add_argument("--min-rgb-bytes", type=int, default=1)
    audit_bop_identity_route.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
    )
    audit_bop_identity_route.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    audit_bop_identity_route.add_argument("--hash-files", action="store_true")
    audit_bop_identity_route.add_argument(
        "--min-identity-scenario-frames",
        type=int,
        default=3,
    )
    audit_bop_identity_route.add_argument(
        "--min-occlusion-fraction",
        type=float,
        default=0.5,
    )
    audit_bop_identity_route.add_argument(
        "--min-view-conditions",
        type=int,
        default=2,
    )
    audit_bop_identity_route.add_argument(
        "--min-lighting-conditions",
        type=int,
        default=2,
    )
    audit_bop_identity_route.add_argument(
        "--min-camera-motion-m",
        type=float,
        default=0.01,
    )
    audit_bop_identity_route.add_argument(
        "--require-identity-reviewable",
        action="store_true",
        help="fail unless existing outputs expose reviewable identity evidence",
    )
    audit_bop_identity_route.set_defaults(
        handler=_object_state_audit_bop_identity_route
    )
    audit_bop_phase1_local_row = object_state_subparsers.add_parser(
        "audit-bop-phase1-local-row",
        help=(
            "read-only combined audit of a local BOP scene for Phase 1 "
            "identity and prediction row readiness"
        ),
    )
    audit_bop_phase1_local_row.add_argument("scene_root", type=Path)
    audit_bop_phase1_local_row.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    audit_bop_phase1_local_row.add_argument("--summary-output", type=Path)
    audit_bop_phase1_local_row.add_argument("--sample-id", required=True)
    audit_bop_phase1_local_row.add_argument("--candidate-artifact", type=Path)
    audit_bop_phase1_local_row.add_argument("--identity-dir", default="identity-handoff")
    audit_bop_phase1_local_row.add_argument("--dataset-id", default="bop-ycbv")
    audit_bop_phase1_local_row.add_argument("--object-category", default="bop_objects")
    audit_bop_phase1_local_row.add_argument("--scenario", default="bop_pose_sequence")
    audit_bop_phase1_local_row.add_argument("--fps", type=float, default=30.0)
    audit_bop_phase1_local_row.add_argument(
        "--license",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    audit_bop_phase1_local_row.add_argument("--rgb-dir", default="rgb")
    audit_bop_phase1_local_row.add_argument(
        "--depth-dir",
        default="depth",
        help="depth directory used only for read-only RGB-D export hints",
    )
    audit_bop_phase1_local_row.add_argument("--gaussian-dir", default="gaussians")
    audit_bop_phase1_local_row.add_argument(
        "--condition-sidecar",
        type=Path,
        help="optional JSON sidecar with explicit per-frame view, lighting, and camera_pose metadata",
    )
    audit_bop_phase1_local_row.add_argument("--max-frames", type=int)
    audit_bop_phase1_local_row.add_argument("--frame-step", type=int, default=1)
    audit_bop_phase1_local_row.add_argument(
        "--check-artifact-refs",
        action="store_true",
        help="also require sample artifact refs such as scene_gt.json to exist",
    )
    audit_bop_phase1_local_row.add_argument("--min-rgb-bytes", type=int, default=1)
    audit_bop_phase1_local_row.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
    )
    audit_bop_phase1_local_row.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    audit_bop_phase1_local_row.add_argument("--hash-files", action="store_true")
    audit_bop_phase1_local_row.add_argument(
        "--min-identity-scenario-frames",
        type=int,
        default=3,
    )
    audit_bop_phase1_local_row.add_argument(
        "--min-occlusion-fraction",
        type=float,
        default=0.5,
    )
    audit_bop_phase1_local_row.add_argument(
        "--min-view-conditions",
        type=int,
        default=2,
    )
    audit_bop_phase1_local_row.add_argument(
        "--min-lighting-conditions",
        type=int,
        default=2,
    )
    audit_bop_phase1_local_row.add_argument(
        "--min-camera-motion-m",
        type=float,
        default=0.01,
    )
    audit_bop_phase1_local_row.add_argument(
        "--require-any-reviewable",
        action="store_true",
        help="fail unless identity or prediction evidence is already reviewable",
    )
    audit_bop_phase1_local_row.add_argument(
        "--require-identity-prediction-reviewable",
        action="store_true",
        help="fail unless both identity and prediction evidence are reviewable",
    )
    audit_bop_phase1_local_row.set_defaults(
        handler=_object_state_audit_bop_phase1_local_row
    )
    bop_prediction_baseline_handoff = object_state_subparsers.add_parser(
        "bop-prediction-baseline-handoff",
        help=(
            "run BOP acceptance, baseline prediction candidate generation, "
            "prediction eval, and prediction evidence package audit"
        ),
    )
    bop_prediction_baseline_handoff.add_argument("scene_root", type=Path)
    bop_prediction_baseline_handoff.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    bop_prediction_baseline_handoff.add_argument("--summary-output", type=Path)
    bop_prediction_baseline_handoff.add_argument("--sample-id", required=True)
    bop_prediction_baseline_handoff.add_argument("--dataset-id", default="bop-ycbv")
    bop_prediction_baseline_handoff.add_argument(
        "--object-category",
        default="bop_objects",
    )
    bop_prediction_baseline_handoff.add_argument(
        "--scenario",
        default="bop_pose_sequence",
    )
    bop_prediction_baseline_handoff.add_argument("--fps", type=float, default=30.0)
    bop_prediction_baseline_handoff.add_argument(
        "--license",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    bop_prediction_baseline_handoff.add_argument("--rgb-dir", default="rgb")
    bop_prediction_baseline_handoff.add_argument("--gaussian-dir", default="gaussians")
    bop_prediction_baseline_handoff.add_argument(
        "--condition-sidecar",
        type=Path,
        help="optional JSON sidecar with explicit per-frame view, lighting, and camera_pose metadata",
    )
    bop_prediction_baseline_handoff.add_argument("--max-frames", type=int)
    bop_prediction_baseline_handoff.add_argument("--frame-step", type=int, default=1)
    bop_prediction_baseline_handoff.add_argument(
        "--policy",
        choices=("constant_velocity", "hold"),
        default="constant_velocity",
    )
    bop_prediction_baseline_handoff.add_argument(
        "--candidate-id",
        default="bop-constant-velocity-baseline",
    )
    bop_prediction_baseline_handoff.add_argument(
        "--candidate-source",
        default="controlled-prediction-baseline",
    )
    bop_prediction_baseline_handoff.add_argument("--artifact-ref")
    bop_prediction_baseline_handoff.add_argument(
        "--confidence",
        type=float,
        default=0.5,
    )
    bop_prediction_baseline_handoff.add_argument(
        "--check-artifact-refs",
        action="store_true",
        help="also require sample artifact refs such as scene_gt.json to exist",
    )
    bop_prediction_baseline_handoff.add_argument("--min-rgb-bytes", type=int, default=1)
    bop_prediction_baseline_handoff.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
    )
    bop_prediction_baseline_handoff.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    bop_prediction_baseline_handoff.add_argument("--hash-files", action="store_true")
    bop_prediction_baseline_handoff.add_argument(
        "--max-state-ade",
        type=float,
        default=0.05,
    )
    bop_prediction_baseline_handoff.add_argument(
        "--max-prediction-gap-vs-history-model",
        type=float,
        default=0.02,
    )
    bop_prediction_baseline_handoff.add_argument(
        "--max-error-ratio-vs-history-model",
        type=float,
        default=1.25,
    )
    bop_prediction_baseline_handoff.add_argument(
        "--min-prediction-count",
        type=int,
        default=1,
    )
    bop_prediction_baseline_handoff.add_argument(
        "--force",
        action="store_true",
        help="overwrite handoff output files if they already exist",
    )
    bop_prediction_baseline_handoff.add_argument(
        "--require-reviewable",
        action="store_true",
        help="fail unless the generated evidence package is reviewable",
    )
    bop_prediction_baseline_handoff.set_defaults(
        handler=_object_state_bop_prediction_baseline_handoff
    )
    bop_identity_handoff = object_state_subparsers.add_parser(
        "bop-identity-handoff",
        help=(
            "run BOP acceptance, trainable ObjectState identity eval, identity "
            "evidence package audit, and Phase 1 evidence ledger"
        ),
    )
    bop_identity_handoff.add_argument("scene_root", type=Path)
    bop_identity_handoff.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    bop_identity_handoff.add_argument("--summary-output", type=Path)
    bop_identity_handoff.add_argument("--sample-id", required=True)
    bop_identity_handoff.add_argument(
        "--candidate-artifact",
        required=True,
        type=Path,
        help="finalized trainable ObjectState artifact to evaluate",
    )
    bop_identity_handoff.add_argument(
        "--identity-dir",
        type=Path,
        help="identity package directory; defaults to <output-root>/identity-handoff",
    )
    bop_identity_handoff.add_argument("--dataset-id", default="bop-ycbv")
    bop_identity_handoff.add_argument(
        "--object-category",
        default="bop_objects",
    )
    bop_identity_handoff.add_argument(
        "--scenario",
        default="bop_pose_sequence",
    )
    bop_identity_handoff.add_argument("--fps", type=float, default=30.0)
    bop_identity_handoff.add_argument(
        "--license",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    bop_identity_handoff.add_argument("--rgb-dir", default="rgb")
    bop_identity_handoff.add_argument("--gaussian-dir", default="gaussians")
    bop_identity_handoff.add_argument(
        "--condition-sidecar",
        type=Path,
        help="optional JSON sidecar with explicit per-frame view, lighting, and camera_pose metadata",
    )
    bop_identity_handoff.add_argument("--max-frames", type=int)
    bop_identity_handoff.add_argument("--frame-step", type=int, default=1)
    bop_identity_handoff.add_argument("--candidate-id")
    bop_identity_handoff.add_argument(
        "--candidate-source",
        default="trainable_kernel_objectstate_nearest_pose_adapter",
    )
    bop_identity_handoff.add_argument("--max-centroid-distance", type=float)
    bop_identity_handoff.add_argument("--min-idf1", type=float, default=0.95)
    bop_identity_handoff.add_argument(
        "--min-track-retrieval-recall-at-1",
        type=float,
        default=0.95,
    )
    bop_identity_handoff.add_argument(
        "--max-fragmentation-rate",
        type=float,
        default=0.05,
    )
    bop_identity_handoff.add_argument(
        "--max-long-term-drift-rate",
        type=float,
        default=0.05,
    )
    bop_identity_handoff.add_argument("--max-swap-rate", type=float, default=0.0)
    bop_identity_handoff.add_argument(
        "--min-reconstruction-noise-robustness",
        type=float,
        default=0.95,
    )
    bop_identity_handoff.add_argument(
        "--min-reconstruction-noise-variants",
        type=int,
        default=2,
    )
    bop_identity_handoff.add_argument("--allow-identity-collapse", action="store_true")
    bop_identity_handoff.add_argument(
        "--check-artifact-refs",
        action="store_true",
        help="also require sample artifact refs such as scene_gt.json to exist",
    )
    bop_identity_handoff.add_argument("--min-rgb-bytes", type=int, default=1)
    bop_identity_handoff.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
    )
    bop_identity_handoff.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    bop_identity_handoff.add_argument("--hash-files", action="store_true")
    bop_identity_handoff.add_argument(
        "--min-candidate-artifact-bytes",
        type=int,
        default=1,
    )
    bop_identity_handoff.add_argument(
        "--hash-candidate-artifact",
        action="store_true",
    )
    bop_identity_handoff.add_argument(
        "--min-identity-scenario-frames",
        type=int,
        default=3,
    )
    bop_identity_handoff.add_argument(
        "--min-occlusion-fraction",
        type=float,
        default=0.5,
    )
    bop_identity_handoff.add_argument(
        "--min-view-conditions",
        type=int,
        default=2,
    )
    bop_identity_handoff.add_argument(
        "--min-lighting-conditions",
        type=int,
        default=2,
    )
    bop_identity_handoff.add_argument(
        "--min-camera-motion-m",
        type=float,
        default=0.01,
    )
    bop_identity_handoff.add_argument(
        "--synthetic-smoke-failed",
        action="store_true",
        help="mark the synthetic prerequisite smoke gate as failed",
    )
    bop_identity_handoff.add_argument(
        "--min-real-or-public-rows",
        type=int,
        default=1,
    )
    bop_identity_handoff.add_argument(
        "--force",
        action="store_true",
        help="overwrite handoff output files if they already exist",
    )
    bop_identity_handoff.add_argument(
        "--require-reviewable",
        action="store_true",
        help="fail unless the generated identity evidence package is reviewable",
    )
    bop_identity_handoff.add_argument(
        "--require-pass",
        action="store_true",
        help="fail unless the identity handoff metric gate passes",
    )
    bop_identity_handoff.set_defaults(
        handler=_object_state_bop_identity_handoff
    )
    bop_local_row_handoff = object_state_subparsers.add_parser(
        "bop-local-row-handoff",
        help=(
            "run BOP identity handoff and prediction baseline handoff, then "
            "write a merged Phase 1 evidence ledger"
        ),
    )
    bop_local_row_handoff.add_argument("scene_root", type=Path)
    bop_local_row_handoff.add_argument(
        "--output-root",
        required=True,
        type=Path,
    )
    bop_local_row_handoff.add_argument("--summary-output", type=Path)
    bop_local_row_handoff.add_argument("--sample-id", required=True)
    bop_local_row_handoff.add_argument(
        "--candidate-artifact",
        required=True,
        type=Path,
        help="finalized trainable ObjectState artifact for identity evidence",
    )
    bop_local_row_handoff.add_argument("--dataset-id", default="bop-ycbv")
    bop_local_row_handoff.add_argument(
        "--object-category",
        default="bop_objects",
    )
    bop_local_row_handoff.add_argument(
        "--scenario",
        default="bop_pose_sequence",
    )
    bop_local_row_handoff.add_argument("--fps", type=float, default=30.0)
    bop_local_row_handoff.add_argument(
        "--license",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    bop_local_row_handoff.add_argument("--rgb-dir", default="rgb")
    bop_local_row_handoff.add_argument("--gaussian-dir", default="gaussians")
    bop_local_row_handoff.add_argument(
        "--condition-sidecar",
        type=Path,
        help="optional JSON sidecar with explicit per-frame view, lighting, and camera_pose metadata",
    )
    bop_local_row_handoff.add_argument("--max-frames", type=int)
    bop_local_row_handoff.add_argument("--frame-step", type=int, default=1)
    bop_local_row_handoff.add_argument("--identity-candidate-id")
    bop_local_row_handoff.add_argument(
        "--identity-candidate-source",
        default="trainable_kernel_objectstate_nearest_pose_adapter",
    )
    bop_local_row_handoff.add_argument("--max-centroid-distance", type=float)
    bop_local_row_handoff.add_argument(
        "--prediction-policy",
        choices=("constant_velocity", "hold"),
        default="constant_velocity",
    )
    bop_local_row_handoff.add_argument(
        "--prediction-candidate-id",
        default="bop-constant-velocity-baseline",
    )
    bop_local_row_handoff.add_argument(
        "--prediction-candidate-source",
        default="controlled-prediction-baseline",
    )
    bop_local_row_handoff.add_argument(
        "--prediction-confidence",
        type=float,
        default=0.5,
    )
    bop_local_row_handoff.add_argument("--min-idf1", type=float, default=0.95)
    bop_local_row_handoff.add_argument(
        "--min-track-retrieval-recall-at-1",
        type=float,
        default=0.95,
    )
    bop_local_row_handoff.add_argument(
        "--max-fragmentation-rate",
        type=float,
        default=0.05,
    )
    bop_local_row_handoff.add_argument(
        "--max-long-term-drift-rate",
        type=float,
        default=0.05,
    )
    bop_local_row_handoff.add_argument("--max-swap-rate", type=float, default=0.0)
    bop_local_row_handoff.add_argument(
        "--min-reconstruction-noise-robustness",
        type=float,
        default=0.95,
    )
    bop_local_row_handoff.add_argument(
        "--min-reconstruction-noise-variants",
        type=int,
        default=2,
    )
    bop_local_row_handoff.add_argument("--allow-identity-collapse", action="store_true")
    bop_local_row_handoff.add_argument(
        "--max-state-ade",
        type=float,
        default=0.05,
    )
    bop_local_row_handoff.add_argument(
        "--max-prediction-gap-vs-history-model",
        type=float,
        default=0.02,
    )
    bop_local_row_handoff.add_argument(
        "--max-error-ratio-vs-history-model",
        type=float,
        default=1.25,
    )
    bop_local_row_handoff.add_argument(
        "--min-prediction-count",
        type=int,
        default=1,
    )
    bop_local_row_handoff.add_argument(
        "--check-artifact-refs",
        action="store_true",
        help="also require sample artifact refs such as scene_gt.json to exist",
    )
    bop_local_row_handoff.add_argument("--min-rgb-bytes", type=int, default=1)
    bop_local_row_handoff.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
    )
    bop_local_row_handoff.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    bop_local_row_handoff.add_argument("--hash-files", action="store_true")
    bop_local_row_handoff.add_argument(
        "--min-candidate-artifact-bytes",
        type=int,
        default=1,
    )
    bop_local_row_handoff.add_argument(
        "--hash-candidate-artifact",
        action="store_true",
    )
    bop_local_row_handoff.add_argument(
        "--min-identity-scenario-frames",
        type=int,
        default=3,
    )
    bop_local_row_handoff.add_argument(
        "--min-occlusion-fraction",
        type=float,
        default=0.5,
    )
    bop_local_row_handoff.add_argument(
        "--min-view-conditions",
        type=int,
        default=2,
    )
    bop_local_row_handoff.add_argument(
        "--min-lighting-conditions",
        type=int,
        default=2,
    )
    bop_local_row_handoff.add_argument(
        "--min-camera-motion-m",
        type=float,
        default=0.01,
    )
    bop_local_row_handoff.add_argument(
        "--synthetic-smoke-failed",
        action="store_true",
        help="mark the synthetic prerequisite smoke gate as failed",
    )
    bop_local_row_handoff.add_argument(
        "--min-real-or-public-rows",
        type=int,
        default=1,
    )
    bop_local_row_handoff.add_argument(
        "--force",
        action="store_true",
        help="overwrite handoff output files if they already exist",
    )
    bop_local_row_handoff.add_argument(
        "--require-reviewable",
        action="store_true",
        help="fail unless identity and prediction evidence are reviewable",
    )
    bop_local_row_handoff.add_argument(
        "--require-pass",
        action="store_true",
        help="fail unless both identity and prediction metric gates pass",
    )
    bop_local_row_handoff.set_defaults(
        handler=_object_state_bop_local_row_handoff
    )
    init_bop_local_row_batch_spec = object_state_subparsers.add_parser(
        "init-bop-local-row-batch-spec",
        help=(
            "write a BOP local-row batch spec from a CSV of scene roots and "
            "candidate artifacts"
        ),
    )
    init_bop_local_row_batch_spec.add_argument(
        "--samples-csv",
        required=True,
        type=Path,
        help=(
            "CSV with sample_id, scene_root, candidate_artifact and optional "
            "condition_sidecar/output_root/options columns"
        ),
    )
    init_bop_local_row_batch_spec.add_argument(
        "--output",
        required=True,
        type=Path,
        help="batch spec JSON to write",
    )
    init_bop_local_row_batch_spec.add_argument(
        "--summary-output",
        type=Path,
        help="optional authoring summary JSON",
    )
    init_bop_local_row_batch_spec.add_argument(
        "--batch-id",
        default="bop-local-row-batch",
    )
    init_bop_local_row_batch_spec.add_argument(
        "--batch-output-root",
        default="bop-local-row-batch",
        help="output_root value written under the batch spec's batch object",
    )
    init_bop_local_row_batch_spec.add_argument("--dataset-id", default="bop-ycbv")
    init_bop_local_row_batch_spec.add_argument("--object-category", default="bop_objects")
    init_bop_local_row_batch_spec.add_argument("--scenario", default="bop_pose_sequence")
    init_bop_local_row_batch_spec.add_argument("--fps", type=float, default=30.0)
    init_bop_local_row_batch_spec.add_argument(
        "--license-text",
        default=(
            "BOP dataset terms; verify source dataset license before redistribution"
        ),
    )
    init_bop_local_row_batch_spec.add_argument("--rgb-dir", default="rgb")
    init_bop_local_row_batch_spec.add_argument("--depth-dir", default="depth")
    init_bop_local_row_batch_spec.add_argument("--gaussian-dir", default="gaussians")
    init_bop_local_row_batch_spec.add_argument(
        "--absolute-paths",
        action="store_true",
        help="write absolute/input paths instead of paths relative to the batch spec",
    )
    init_bop_local_row_batch_spec.add_argument(
        "--require-inputs",
        action="store_true",
        help="fail unless scene roots, candidate artifacts, and declared sidecars exist",
    )
    init_bop_local_row_batch_spec.set_defaults(
        handler=_object_state_init_bop_local_row_batch_spec
    )
    audit_bop_local_row_batch_readiness = object_state_subparsers.add_parser(
        "audit-bop-local-row-batch-readiness",
        help=(
            "read-only preflight for a BOP local-row batch spec before running "
            "batch handoff"
        ),
    )
    audit_bop_local_row_batch_readiness.add_argument(
        "batch_spec",
        type=Path,
        help="JSON batch spec using objgauss-objectstate-bop-local-row-batch-spec-v1",
    )
    audit_bop_local_row_batch_readiness.add_argument(
        "--output-root",
        type=Path,
        help="override the batch output root declared in the spec",
    )
    audit_bop_local_row_batch_readiness.add_argument("--summary-output", type=Path)
    audit_bop_local_row_batch_readiness.add_argument(
        "--table-output",
        type=Path,
        help="optional Markdown output for the batch readiness sample table",
    )
    audit_bop_local_row_batch_readiness.add_argument(
        "--min-reviewable-samples",
        default=3,
        type=int,
        help="minimum ready or reviewable samples before batch handoff",
    )
    audit_bop_local_row_batch_readiness.add_argument(
        "--min-scene-or-category-coverage",
        default=3,
        type=int,
        help="minimum sample, scene, category, or scenario coverage for readiness",
    )
    audit_bop_local_row_batch_readiness.add_argument(
        "--require-ready",
        action="store_true",
        help="fail unless all batch readiness gates pass",
    )
    audit_bop_local_row_batch_readiness.set_defaults(
        handler=_object_state_audit_bop_local_row_batch_readiness
    )
    bop_local_row_batch_handoff = object_state_subparsers.add_parser(
        "bop-local-row-batch-handoff",
        help=(
            "run multiple BOP local-row handoffs from a batch spec, then write "
            "a cross-sample Phase 1 ledger"
        ),
    )
    bop_local_row_batch_handoff.add_argument(
        "batch_spec",
        type=Path,
        help="JSON batch spec using objgauss-objectstate-bop-local-row-batch-spec-v1",
    )
    bop_local_row_batch_handoff.add_argument(
        "--output-root",
        type=Path,
        help="override the batch output root declared in the spec",
    )
    bop_local_row_batch_handoff.add_argument("--summary-output", type=Path)
    bop_local_row_batch_handoff.add_argument(
        "--table-output",
        type=Path,
        help="optional Markdown output for the generated cross-sample sample table",
    )
    bop_local_row_batch_handoff.add_argument(
        "--min-reviewable-samples",
        default=3,
        type=int,
        help="minimum reviewable samples for candidate cross-sample readiness",
    )
    bop_local_row_batch_handoff.add_argument(
        "--min-scene-or-category-coverage",
        default=3,
        type=int,
        help="minimum sample, scene, category, or scenario coverage for readiness",
    )
    bop_local_row_batch_handoff.add_argument(
        "--force",
        action="store_true",
        help="overwrite batch and per-sample handoff outputs if they already exist",
    )
    bop_local_row_batch_handoff.add_argument(
        "--require-reviewable",
        action="store_true",
        help="fail unless all batch local-row outputs are reviewable",
    )
    bop_local_row_batch_handoff.add_argument(
        "--require-candidate-ready",
        action="store_true",
        help="fail unless cross-sample candidate coverage thresholds are met",
    )
    bop_local_row_batch_handoff.set_defaults(
        handler=_object_state_bop_local_row_batch_handoff
    )
    audit_bop_cross_sample_ledger = object_state_subparsers.add_parser(
        "audit-bop-cross-sample-ledger",
        help=(
            "summarize existing BOP local-row handoff summaries as a "
            "cross-sample Phase 1 identity/prediction evidence table"
        ),
    )
    audit_bop_cross_sample_ledger.add_argument(
        "--local-row-summary",
        action="append",
        default=[],
        type=Path,
        help="BOP local-row handoff summary JSON; may be repeated",
    )
    audit_bop_cross_sample_ledger.add_argument(
        "--discover-root",
        action="append",
        default=[],
        type=Path,
        help="root directory to scan for bop-local-row-handoff-summary.json files",
    )
    audit_bop_cross_sample_ledger.add_argument(
        "--max-depth",
        default=4,
        type=int,
        help="maximum subdirectory depth to scan below each discovery root",
    )
    audit_bop_cross_sample_ledger.add_argument(
        "--min-reviewable-samples",
        default=3,
        type=int,
        help="minimum unique reviewable samples for candidate cross-sample readiness",
    )
    audit_bop_cross_sample_ledger.add_argument(
        "--min-scene-or-category-coverage",
        default=3,
        type=int,
        help="minimum sample, scene, category, or scenario coverage for candidate readiness",
    )
    audit_bop_cross_sample_ledger.add_argument("--summary-output", type=Path)
    audit_bop_cross_sample_ledger.add_argument(
        "--table-output",
        type=Path,
        help="optional Markdown output for the cross-sample evidence table",
    )
    audit_bop_cross_sample_ledger.add_argument(
        "--require-reviewable",
        action="store_true",
        help="fail unless all supplied/discovered summaries are present and valid",
    )
    audit_bop_cross_sample_ledger.add_argument(
        "--require-candidate-ready",
        action="store_true",
        help="fail unless cross-sample candidate coverage thresholds are met",
    )
    audit_bop_cross_sample_ledger.set_defaults(
        handler=_object_state_audit_bop_cross_sample_ledger
    )
    audit_controlled_capture_bundle_readiness = object_state_subparsers.add_parser(
        "audit-controlled-capture-bundle-readiness",
        help="audit staged controlled capture bundle readiness before import/handoff",
    )
    audit_controlled_capture_bundle_readiness.add_argument("bundle_root", type=Path)
    audit_controlled_capture_bundle_readiness.add_argument("--summary-output", type=Path)
    audit_controlled_capture_bundle_readiness.add_argument("--sample-json", default="sample.json")
    audit_controlled_capture_bundle_readiness.add_argument("--objects-csv", default="objects.csv")
    audit_controlled_capture_bundle_readiness.add_argument("--frames-csv", default="frames.csv")
    audit_controlled_capture_bundle_readiness.add_argument(
        "--annotations-csv",
        default="annotations.csv",
    )
    audit_controlled_capture_bundle_readiness.add_argument("--actions-csv", default="actions.csv")
    audit_controlled_capture_bundle_readiness.add_argument(
        "--require-prediction-ready",
        action="store_true",
        help="also require prediction-stage capture readiness",
    )
    audit_controlled_capture_bundle_readiness.add_argument(
        "--require-intervention-ready",
        action="store_true",
        help="also require intervention-stage action readiness",
    )
    audit_controlled_capture_bundle_readiness.add_argument("--min-rgb-bytes", type=int, default=1)
    audit_controlled_capture_bundle_readiness.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
    )
    audit_controlled_capture_bundle_readiness.add_argument("--hash-files", action="store_true")
    audit_controlled_capture_bundle_readiness.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    audit_controlled_capture_bundle_readiness.add_argument(
        "--candidate-artifact",
        type=Path,
        help="optional candidate ObjectState artifact to include in handoff readiness",
    )
    audit_controlled_capture_bundle_readiness.add_argument(
        "--require-candidate-artifact",
        action="store_true",
        help="require candidate artifact readiness for overall status",
    )
    audit_controlled_capture_bundle_readiness.add_argument(
        "--min-candidate-artifact-bytes",
        type=int,
        default=1,
    )
    audit_controlled_capture_bundle_readiness.add_argument(
        "--min-identity-scenario-frames",
        type=int,
        default=3,
    )
    audit_controlled_capture_bundle_readiness.add_argument(
        "--min-occlusion-fraction",
        type=float,
        default=0.5,
    )
    audit_controlled_capture_bundle_readiness.add_argument(
        "--min-view-conditions",
        type=int,
        default=2,
    )
    audit_controlled_capture_bundle_readiness.add_argument(
        "--min-lighting-conditions",
        type=int,
        default=2,
    )
    audit_controlled_capture_bundle_readiness.add_argument(
        "--min-camera-motion-m",
        type=float,
        default=0.01,
    )
    audit_controlled_capture_bundle_readiness.add_argument("--require-ready", action="store_true")
    audit_controlled_capture_bundle_readiness.set_defaults(
        handler=_object_state_audit_controlled_capture_bundle_readiness
    )
    audit_controlled_reality_bundle_readiness = object_state_subparsers.add_parser(
        "audit-controlled-reality-bundle-readiness",
        help=(
            "audit controlled bundle, trainable artifact, prediction candidates "
            "and intervention candidates before full reality handoff"
        ),
    )
    audit_controlled_reality_bundle_readiness.add_argument("bundle_root", type=Path)
    audit_controlled_reality_bundle_readiness.add_argument(
        "trainable_artifact",
        type=Path,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "prediction_candidates",
        type=Path,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "intervention_candidates",
        type=Path,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--summary-output",
        type=Path,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--sample-json",
        default="sample.json",
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--objects-csv",
        default="objects.csv",
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--frames-csv",
        default="frames.csv",
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--annotations-csv",
        default="annotations.csv",
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--actions-csv",
        default="actions.csv",
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--max-centroid-distance",
        type=float,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--min-rgb-bytes",
        type=int,
        default=1,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--hash-files",
        action="store_true",
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--min-candidate-artifact-bytes",
        type=int,
        default=1,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--min-identity-scenario-frames",
        type=int,
        default=3,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--min-occlusion-fraction",
        type=float,
        default=0.5,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--min-view-conditions",
        type=int,
        default=2,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--min-lighting-conditions",
        type=int,
        default=2,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--min-camera-motion-m",
        type=float,
        default=0.01,
    )
    audit_controlled_reality_bundle_readiness.add_argument(
        "--require-ready",
        action="store_true",
    )
    audit_controlled_reality_bundle_readiness.set_defaults(
        handler=_object_state_audit_controlled_reality_bundle_readiness
    )
    init_controlled_reality_candidates = object_state_subparsers.add_parser(
        "init-controlled-reality-candidates",
        help=(
            "create draft prediction/intervention candidate templates from a "
            "controlled capture bundle"
        ),
    )
    init_controlled_reality_candidates.add_argument("bundle_root", type=Path)
    init_controlled_reality_candidates.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    init_controlled_reality_candidates.add_argument(
        "--sample-json",
        default="sample.json",
    )
    init_controlled_reality_candidates.add_argument(
        "--objects-csv",
        default="objects.csv",
    )
    init_controlled_reality_candidates.add_argument(
        "--frames-csv",
        default="frames.csv",
    )
    init_controlled_reality_candidates.add_argument(
        "--annotations-csv",
        default="annotations.csv",
    )
    init_controlled_reality_candidates.add_argument(
        "--actions-csv",
        default="actions.csv",
    )
    init_controlled_reality_candidates.add_argument(
        "--candidate-id",
        default="TODO_CANDIDATE_ID",
    )
    init_controlled_reality_candidates.add_argument(
        "--candidate-source",
        default="TODO_CANDIDATE_SOURCE",
    )
    init_controlled_reality_candidates.add_argument(
        "--artifact-ref",
        default="TODO_CANDIDATE_ARTIFACT_REF",
    )
    init_controlled_reality_candidates.add_argument("--summary-output", type=Path)
    init_controlled_reality_candidates.add_argument(
        "--force",
        action="store_true",
        help="overwrite candidate template files if they already exist",
    )
    init_controlled_reality_candidates.set_defaults(
        handler=_object_state_init_controlled_reality_candidates
    )
    init_controlled_reality_candidates_from_manifest = (
        object_state_subparsers.add_parser(
            "init-controlled-reality-candidates-from-manifest",
            help=(
                "create draft prediction/intervention candidate templates from a "
                "controlled capture manifest"
            ),
        )
    )
    init_controlled_reality_candidates_from_manifest.add_argument(
        "capture_manifest",
        type=Path,
    )
    init_controlled_reality_candidates_from_manifest.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    init_controlled_reality_candidates_from_manifest.add_argument(
        "--candidate-id",
        default="TODO_CANDIDATE_ID",
    )
    init_controlled_reality_candidates_from_manifest.add_argument(
        "--candidate-source",
        default="TODO_CANDIDATE_SOURCE",
    )
    init_controlled_reality_candidates_from_manifest.add_argument(
        "--artifact-ref",
        default="TODO_CANDIDATE_ARTIFACT_REF",
    )
    init_controlled_reality_candidates_from_manifest.add_argument(
        "--summary-output",
        type=Path,
    )
    init_controlled_reality_candidates_from_manifest.add_argument(
        "--force",
        action="store_true",
        help="overwrite candidate template files if they already exist",
    )
    init_controlled_reality_candidates_from_manifest.set_defaults(
        handler=_object_state_init_controlled_reality_candidates_from_manifest
    )
    finalize_controlled_reality_candidates = object_state_subparsers.add_parser(
        "finalize-controlled-reality-candidates",
        help=(
            "convert filled prediction/intervention candidate templates into "
            "evaluator-ready JSON"
        ),
    )
    finalize_controlled_reality_candidates.add_argument(
        "prediction_template",
        type=Path,
    )
    finalize_controlled_reality_candidates.add_argument(
        "intervention_template",
        type=Path,
    )
    finalize_controlled_reality_candidates.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    finalize_controlled_reality_candidates.add_argument(
        "--bundle-root",
        type=Path,
        help="optional bundle path used only when printing next commands",
    )
    finalize_controlled_reality_candidates.add_argument("--summary-output", type=Path)
    finalize_controlled_reality_candidates.add_argument(
        "--force",
        action="store_true",
        help="overwrite evaluator-ready candidate files if they already exist",
    )
    finalize_controlled_reality_candidates.set_defaults(
        handler=_object_state_finalize_controlled_reality_candidates
    )
    finalize_controlled_prediction_candidates = object_state_subparsers.add_parser(
        "finalize-controlled-prediction-candidates",
        help=(
            "convert a filled prediction candidate template into "
            "evaluator-ready JSON"
        ),
    )
    finalize_controlled_prediction_candidates.add_argument(
        "prediction_template",
        type=Path,
    )
    finalize_controlled_prediction_candidates.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    finalize_controlled_prediction_candidates.add_argument(
        "--capture-manifest",
        type=Path,
        help="optional capture manifest path used when printing the eval command",
    )
    finalize_controlled_prediction_candidates.add_argument(
        "--summary-output",
        type=Path,
    )
    finalize_controlled_prediction_candidates.add_argument(
        "--force",
        action="store_true",
        help="overwrite evaluator-ready prediction candidates if they already exist",
    )
    finalize_controlled_prediction_candidates.set_defaults(
        handler=_object_state_finalize_controlled_prediction_candidates
    )
    generate_controlled_prediction_baseline_candidates = (
        object_state_subparsers.add_parser(
            "generate-controlled-prediction-baseline-candidates",
            help=(
                "fill a prediction candidate template with a hold or "
                "constant-velocity baseline and finalize it for eval"
            ),
        )
    )
    generate_controlled_prediction_baseline_candidates.add_argument(
        "capture_manifest",
        type=Path,
    )
    generate_controlled_prediction_baseline_candidates.add_argument(
        "prediction_template",
        type=Path,
    )
    generate_controlled_prediction_baseline_candidates.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    generate_controlled_prediction_baseline_candidates.add_argument(
        "--policy",
        choices=("constant_velocity", "hold"),
        default="constant_velocity",
    )
    generate_controlled_prediction_baseline_candidates.add_argument(
        "--candidate-id",
        default="controlled-prediction-baseline-constant-velocity",
    )
    generate_controlled_prediction_baseline_candidates.add_argument(
        "--candidate-source",
    )
    generate_controlled_prediction_baseline_candidates.add_argument(
        "--artifact-ref",
        default="generated-controlled-prediction-baseline",
    )
    generate_controlled_prediction_baseline_candidates.add_argument(
        "--confidence",
        type=float,
        default=0.5,
    )
    generate_controlled_prediction_baseline_candidates.add_argument(
        "--summary-output",
        type=Path,
    )
    generate_controlled_prediction_baseline_candidates.add_argument(
        "--force",
        action="store_true",
        help="overwrite generated baseline candidate files if they already exist",
    )
    generate_controlled_prediction_baseline_candidates.set_defaults(
        handler=_object_state_generate_controlled_prediction_baseline_candidates
    )
    audit_controlled_reality_evidence_package = object_state_subparsers.add_parser(
        "audit-controlled-reality-evidence-package",
        help=(
            "audit the local full Phase 1 controlled reality evidence package "
            "after candidate finalization and full handoff"
        ),
    )
    audit_controlled_reality_evidence_package.add_argument(
        "package_root",
        type=Path,
    )
    audit_controlled_reality_evidence_package.add_argument(
        "--candidate-dir",
        default="reality-candidates",
        help="candidate output directory, relative to package_root unless absolute",
    )
    audit_controlled_reality_evidence_package.add_argument(
        "--handoff-dir",
        default="reality-handoff",
        help="handoff output directory, relative to package_root unless absolute",
    )
    audit_controlled_reality_evidence_package.add_argument(
        "--summary-output",
        type=Path,
    )
    audit_controlled_reality_evidence_package.add_argument(
        "--require-reviewable",
        action="store_true",
        help="fail unless the evidence package is complete and reviewable",
    )
    audit_controlled_reality_evidence_package.set_defaults(
        handler=_object_state_audit_controlled_reality_evidence_package
    )
    audit_controlled_prediction_evidence_package = object_state_subparsers.add_parser(
        "audit-controlled-prediction-evidence-package",
        help=(
            "audit a local prediction-only Phase 1 evidence package after BOP "
            "acceptance, candidate finalization, and prediction eval"
        ),
    )
    audit_controlled_prediction_evidence_package.add_argument(
        "package_root",
        type=Path,
    )
    audit_controlled_prediction_evidence_package.add_argument(
        "--candidate-dir",
        default="reality-candidates",
        help="candidate output directory, relative to package_root unless absolute",
    )
    audit_controlled_prediction_evidence_package.add_argument(
        "--capture-manifest",
        default="capture-manifest.json",
        help="capture manifest path, relative to package_root unless absolute",
    )
    audit_controlled_prediction_evidence_package.add_argument(
        "--acceptance-summary",
        default="bop-acceptance-summary.json",
        help="BOP acceptance summary path, relative to package_root unless absolute",
    )
    audit_controlled_prediction_evidence_package.add_argument(
        "--file-audit",
        default="bop-file-audit.json",
        help="capture file audit path, relative to package_root unless absolute",
    )
    audit_controlled_prediction_evidence_package.add_argument(
        "--missing-files-markdown",
        default="bop-missing-files.md",
        help="missing-files markdown path, relative to package_root unless absolute",
    )
    audit_controlled_prediction_evidence_package.add_argument(
        "--summary-output",
        type=Path,
    )
    audit_controlled_prediction_evidence_package.add_argument(
        "--require-reviewable",
        action="store_true",
        help="fail unless the prediction evidence package is complete and reviewable",
    )
    audit_controlled_prediction_evidence_package.set_defaults(
        handler=_object_state_audit_controlled_prediction_evidence_package
    )
    audit_controlled_identity_evidence_package = object_state_subparsers.add_parser(
        "audit-controlled-identity-evidence-package",
        help=(
            "audit a local identity-only Phase 1 evidence package after "
            "controlled identity handoff"
        ),
    )
    audit_controlled_identity_evidence_package.add_argument(
        "package_root",
        type=Path,
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--capture-manifest",
        default="capture-manifest.json",
        help="capture manifest path, relative to package_root unless absolute",
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--capture-file-audit",
        default="capture-file-audit.json",
        help="capture file audit path, relative to package_root unless absolute",
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--capture-missing-files-markdown",
        default="capture-missing-files.md",
        help="capture missing-files markdown path, relative to package_root unless absolute",
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--candidate-artifact-file-audit",
        default="candidate-artifact-file-audit.json",
        help="candidate artifact audit path, relative to package_root unless absolute",
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--identity-scenario-audit",
        default="identity-scenario-audit.json",
        help="identity scenario audit path, relative to package_root unless absolute",
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--identity-predictions",
        default="identity-predictions.json",
        help="identity predictions path, relative to package_root unless absolute",
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--identity-eval",
        default="identity-eval-summary.json",
        help="identity eval path, relative to package_root unless absolute",
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--controlled-real",
        default="controlled-real.json",
        help="controlled-real manifest path, relative to package_root unless absolute",
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--controlled-real-summary",
        default="controlled-real-summary.json",
        help="controlled-real summary path, relative to package_root unless absolute",
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--blocked-rows-markdown",
        default="blocked-rows.md",
        help="blocked rows markdown path, relative to package_root unless absolute",
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--handoff-summary",
        default="handoff-summary.json",
        help="handoff summary path, relative to package_root unless absolute",
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--summary-output",
        type=Path,
    )
    audit_controlled_identity_evidence_package.add_argument(
        "--require-reviewable",
        action="store_true",
        help="fail unless the identity evidence package is complete and reviewable",
    )
    audit_controlled_identity_evidence_package.set_defaults(
        handler=_object_state_audit_controlled_identity_evidence_package
    )
    audit_phase1_evidence_ledger = object_state_subparsers.add_parser(
        "audit-phase1-evidence-ledger",
        help=(
            "summarize existing identity, prediction, and full reality "
            "evidence package summaries without running evaluators"
        ),
    )
    audit_phase1_evidence_ledger.add_argument(
        "--identity-summary",
        action="append",
        default=[],
        type=Path,
        help="identity evidence package summary JSON; may be repeated",
    )
    audit_phase1_evidence_ledger.add_argument(
        "--prediction-summary",
        action="append",
        default=[],
        type=Path,
        help="prediction evidence package summary JSON; may be repeated",
    )
    audit_phase1_evidence_ledger.add_argument(
        "--reality-summary",
        action="append",
        default=[],
        type=Path,
        help="full reality evidence package summary JSON; may be repeated",
    )
    audit_phase1_evidence_ledger.add_argument(
        "--discover-root",
        action="append",
        default=[],
        type=Path,
        help="root directory to scan for standard Phase 1 summary filenames",
    )
    audit_phase1_evidence_ledger.add_argument(
        "--max-depth",
        default=4,
        type=int,
        help="maximum subdirectory depth to scan below each discovery root",
    )
    audit_phase1_evidence_ledger.add_argument(
        "--summary-output",
        type=Path,
    )
    audit_phase1_evidence_ledger.add_argument(
        "--require-reviewable",
        action="store_true",
        help="fail unless the ledger inputs are present and schema-valid",
    )
    audit_phase1_evidence_ledger.set_defaults(
        handler=_object_state_audit_phase1_evidence_ledger
    )
    import_controlled_capture = object_state_subparsers.add_parser(
        "import-controlled-capture-bundle",
        help="build a controlled capture manifest from local bundle CSV files",
    )
    import_controlled_capture.add_argument("bundle_root", type=Path)
    import_controlled_capture.add_argument("--output", "-o", required=True, type=Path)
    import_controlled_capture.add_argument("--summary-output", type=Path)
    import_controlled_capture.add_argument("--controlled-real-output", type=Path)
    import_controlled_capture.add_argument("--sample-json", default="sample.json")
    import_controlled_capture.add_argument("--objects-csv", default="objects.csv")
    import_controlled_capture.add_argument("--frames-csv", default="frames.csv")
    import_controlled_capture.add_argument("--annotations-csv", default="annotations.csv")
    import_controlled_capture.add_argument("--actions-csv", default="actions.csv")
    import_controlled_capture.add_argument("--require-identity-ready", action="store_true")
    import_controlled_capture.add_argument("--require-prediction-ready", action="store_true")
    import_controlled_capture.add_argument("--require-intervention-ready", action="store_true")
    import_controlled_capture.set_defaults(
        handler=_object_state_import_controlled_capture_bundle
    )
    accept_controlled_capture = object_state_subparsers.add_parser(
        "accept-controlled-capture-bundle",
        help="import and file-audit a controlled capture bundle before identity handoff",
    )
    accept_controlled_capture.add_argument("bundle_root", type=Path)
    accept_controlled_capture.add_argument("--output", "-o", required=True, type=Path)
    accept_controlled_capture.add_argument("--summary-output", type=Path)
    accept_controlled_capture.add_argument("--import-summary-output", type=Path)
    accept_controlled_capture.add_argument("--file-audit-output", type=Path)
    accept_controlled_capture.add_argument("--missing-files-output", type=Path)
    accept_controlled_capture.add_argument("--controlled-real-output", type=Path)
    accept_controlled_capture.add_argument("--sample-json", default="sample.json")
    accept_controlled_capture.add_argument("--objects-csv", default="objects.csv")
    accept_controlled_capture.add_argument("--frames-csv", default="frames.csv")
    accept_controlled_capture.add_argument("--annotations-csv", default="annotations.csv")
    accept_controlled_capture.add_argument("--actions-csv", default="actions.csv")
    accept_controlled_capture.add_argument(
        "--no-require-identity-ready",
        action="store_true",
        help="allow staging bundles without identity-stage readiness",
    )
    accept_controlled_capture.add_argument("--require-prediction-ready", action="store_true")
    accept_controlled_capture.add_argument("--require-intervention-ready", action="store_true")
    accept_controlled_capture.add_argument(
        "--no-require-gaussian-files",
        action="store_true",
        help="allow RGB-only capture bundles to pass the file audit",
    )
    accept_controlled_capture.add_argument("--check-artifact-refs", action="store_true")
    accept_controlled_capture.add_argument("--min-rgb-bytes", type=int, default=1)
    accept_controlled_capture.add_argument("--min-gaussian-bytes", type=int, default=1)
    accept_controlled_capture.add_argument("--hash-files", action="store_true")
    accept_controlled_capture.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    accept_controlled_capture.add_argument("--require-pass", action="store_true")
    accept_controlled_capture.set_defaults(
        handler=_object_state_accept_controlled_capture_bundle
    )
    validate_controlled_capture = object_state_subparsers.add_parser(
        "validate-controlled-capture",
        help="validate a frame-level controlled tabletop capture manifest",
    )
    validate_controlled_capture.add_argument("manifest", type=Path)
    validate_controlled_capture.add_argument("--summary-output", type=Path)
    validate_controlled_capture.add_argument("--controlled-real-output", type=Path)
    validate_controlled_capture.add_argument("--require-identity-ready", action="store_true")
    validate_controlled_capture.add_argument("--require-prediction-ready", action="store_true")
    validate_controlled_capture.add_argument("--require-intervention-ready", action="store_true")
    validate_controlled_capture.set_defaults(handler=_object_state_validate_controlled_capture)
    audit_controlled_capture_files = object_state_subparsers.add_parser(
        "audit-controlled-capture-files",
        help="check local files referenced by a controlled capture manifest",
    )
    audit_controlled_capture_files.add_argument("manifest", type=Path)
    audit_controlled_capture_files.add_argument(
        "--root",
        type=Path,
        help="root for relative frame refs; defaults to the manifest directory",
    )
    audit_controlled_capture_files.add_argument("--summary-output", type=Path)
    audit_controlled_capture_files.add_argument("--missing-files-output", type=Path)
    audit_controlled_capture_files.add_argument(
        "--no-require-gaussian-files",
        action="store_true",
        help="allow RGB-only capture bundles to pass the file audit",
    )
    audit_controlled_capture_files.add_argument(
        "--check-artifact-refs",
        action="store_true",
        help="also require sample.artifact_refs paths to exist",
    )
    audit_controlled_capture_files.add_argument(
        "--min-rgb-bytes",
        type=int,
        default=1,
        help="minimum byte size for each frame RGB file",
    )
    audit_controlled_capture_files.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
        help="minimum byte size for each frame Gaussian file",
    )
    audit_controlled_capture_files.add_argument(
        "--hash-files",
        action="store_true",
        help="include SHA256 hashes for valid frame RGB/Gaussian files",
    )
    audit_controlled_capture_files.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    audit_controlled_capture_files.add_argument("--require-pass", action="store_true")
    audit_controlled_capture_files.set_defaults(
        handler=_object_state_audit_controlled_capture_files
    )
    eval_controlled_identity = object_state_subparsers.add_parser(
        "eval-controlled-identity",
        help="score candidate identity tracks against a controlled capture manifest",
    )
    eval_controlled_identity.add_argument("capture_manifest", type=Path)
    eval_controlled_identity.add_argument("predictions", type=Path)
    eval_controlled_identity.add_argument("--summary-output", type=Path)
    eval_controlled_identity.add_argument("--controlled-real-output", type=Path)
    eval_controlled_identity.add_argument("--min-idf1", type=float, default=0.95)
    eval_controlled_identity.add_argument(
        "--min-track-retrieval-recall-at-1",
        type=float,
        default=0.95,
    )
    eval_controlled_identity.add_argument("--max-fragmentation-rate", type=float, default=0.05)
    eval_controlled_identity.add_argument(
        "--max-long-term-drift-rate",
        type=float,
        default=0.05,
    )
    eval_controlled_identity.add_argument("--max-swap-rate", type=float, default=0.0)
    eval_controlled_identity.add_argument(
        "--min-reconstruction-noise-robustness",
        type=float,
        default=0.95,
    )
    eval_controlled_identity.add_argument(
        "--min-reconstruction-noise-variants",
        type=int,
        default=2,
    )
    eval_controlled_identity.add_argument("--allow-identity-collapse", action="store_true")
    eval_controlled_identity.add_argument("--require-pass", action="store_true")
    eval_controlled_identity.set_defaults(handler=_object_state_eval_controlled_identity)
    eval_controlled_prediction = object_state_subparsers.add_parser(
        "eval-controlled-prediction",
        help=(
            "score candidate future pose predictions against a controlled capture "
            "manifest"
        ),
    )
    eval_controlled_prediction.add_argument("capture_manifest", type=Path)
    eval_controlled_prediction.add_argument("predictions", type=Path)
    eval_controlled_prediction.add_argument("--summary-output", type=Path)
    eval_controlled_prediction.add_argument("--controlled-real-output", type=Path)
    eval_controlled_prediction.add_argument("--max-state-ade", type=float, default=0.05)
    eval_controlled_prediction.add_argument(
        "--max-prediction-gap-vs-history-model",
        type=float,
        default=0.02,
    )
    eval_controlled_prediction.add_argument(
        "--max-error-ratio-vs-history-model",
        type=float,
        default=1.25,
    )
    eval_controlled_prediction.add_argument(
        "--min-prediction-count",
        type=int,
        default=1,
    )
    eval_controlled_prediction.add_argument("--require-pass", action="store_true")
    eval_controlled_prediction.set_defaults(
        handler=_object_state_eval_controlled_prediction
    )
    eval_controlled_intervention = object_state_subparsers.add_parser(
        "eval-controlled-intervention",
        help=(
            "score action-conditioned intervention predictions against a "
            "controlled capture manifest"
        ),
    )
    eval_controlled_intervention.add_argument("capture_manifest", type=Path)
    eval_controlled_intervention.add_argument("interventions", type=Path)
    eval_controlled_intervention.add_argument("--summary-output", type=Path)
    eval_controlled_intervention.add_argument("--controlled-real-output", type=Path)
    eval_controlled_intervention.add_argument(
        "--max-action-conditioned-ade",
        type=float,
        default=0.05,
    )
    eval_controlled_intervention.add_argument(
        "--min-counterfactual-outcome-accuracy",
        type=float,
        default=0.95,
    )
    eval_controlled_intervention.add_argument(
        "--max-wrong-direction-rate",
        type=float,
        default=0.0,
    )
    eval_controlled_intervention.add_argument(
        "--min-intervention-gain",
        type=float,
        default=0.0,
    )
    eval_controlled_intervention.add_argument(
        "--min-intervention-count",
        type=int,
        default=1,
    )
    eval_controlled_intervention.add_argument("--require-pass", action="store_true")
    eval_controlled_intervention.set_defaults(
        handler=_object_state_eval_controlled_intervention
    )
    export_identity_predictions = object_state_subparsers.add_parser(
        "export-identity-predictions",
        help=(
            "convert trainable kernel ObjectState frames into controlled identity "
            "prediction rows"
        ),
    )
    export_identity_predictions.add_argument("capture_manifest", type=Path)
    export_identity_predictions.add_argument("trainable_artifact", type=Path)
    export_identity_predictions.add_argument("--output", "-o", required=True, type=Path)
    export_identity_predictions.add_argument("--candidate-id")
    export_identity_predictions.add_argument(
        "--source",
        default="trainable_kernel_objectstate_nearest_pose_adapter",
    )
    export_identity_predictions.add_argument(
        "--artifact-ref",
        action="append",
        dest="artifact_refs",
        help=(
            "candidate artifact reference to store in predictions; defaults to the "
            "trainable artifact path"
        ),
    )
    export_identity_predictions.add_argument("--max-centroid-distance", type=float)
    export_identity_predictions.set_defaults(
        handler=_object_state_export_identity_predictions
    )
    controlled_identity_handoff = object_state_subparsers.add_parser(
        "controlled-identity-handoff",
        help=(
            "run capture + trainable ObjectState artifact through the Stage 1 "
            "controlled identity handoff"
        ),
    )
    controlled_identity_handoff.add_argument("capture_manifest", type=Path)
    controlled_identity_handoff.add_argument("trainable_artifact", type=Path)
    controlled_identity_handoff.add_argument("--output-dir", required=True, type=Path)
    controlled_identity_handoff.add_argument(
        "--capture-root",
        type=Path,
        help="root for relative capture frame refs; defaults to the manifest directory",
    )
    controlled_identity_handoff.add_argument("--candidate-id")
    controlled_identity_handoff.add_argument(
        "--source",
        default="trainable_kernel_objectstate_nearest_pose_adapter",
    )
    controlled_identity_handoff.add_argument(
        "--artifact-ref",
        action="append",
        dest="artifact_refs",
        help=(
            "candidate artifact reference to store in predictions; defaults to the "
            "trainable artifact path"
        ),
    )
    controlled_identity_handoff.add_argument("--max-centroid-distance", type=float)
    controlled_identity_handoff.add_argument("--min-idf1", type=float, default=0.95)
    controlled_identity_handoff.add_argument(
        "--min-track-retrieval-recall-at-1",
        type=float,
        default=0.95,
    )
    controlled_identity_handoff.add_argument("--max-fragmentation-rate", type=float, default=0.05)
    controlled_identity_handoff.add_argument(
        "--max-long-term-drift-rate",
        type=float,
        default=0.05,
    )
    controlled_identity_handoff.add_argument("--max-swap-rate", type=float, default=0.0)
    controlled_identity_handoff.add_argument(
        "--min-reconstruction-noise-robustness",
        type=float,
        default=0.95,
    )
    controlled_identity_handoff.add_argument(
        "--min-reconstruction-noise-variants",
        type=int,
        default=2,
    )
    controlled_identity_handoff.add_argument("--allow-identity-collapse", action="store_true")
    controlled_identity_handoff.add_argument(
        "--check-artifact-refs",
        action="store_true",
        help="also require sample.artifact_refs paths to exist before handoff pass",
    )
    controlled_identity_handoff.add_argument(
        "--min-rgb-bytes",
        type=int,
        default=1,
        help="minimum byte size for each frame RGB file",
    )
    controlled_identity_handoff.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
        help="minimum byte size for each frame Gaussian file",
    )
    controlled_identity_handoff.add_argument(
        "--hash-files",
        action="store_true",
        help="include SHA256 hashes for valid frame RGB/Gaussian files",
    )
    controlled_identity_handoff.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    controlled_identity_handoff.add_argument(
        "--min-candidate-artifact-bytes",
        type=int,
        default=1,
        help="minimum byte size for the trainable ObjectState artifact file",
    )
    controlled_identity_handoff.add_argument(
        "--hash-candidate-artifact",
        action="store_true",
        help="include a SHA256 hash for the trainable ObjectState artifact file",
    )
    controlled_identity_handoff.add_argument(
        "--min-identity-scenario-frames",
        type=int,
        default=3,
        help="minimum frame count for identity scenario challenge audit",
    )
    controlled_identity_handoff.add_argument(
        "--min-occlusion-fraction",
        type=float,
        default=0.5,
        help="occlusion_fraction threshold for identity scenario challenge audit",
    )
    controlled_identity_handoff.add_argument(
        "--min-view-conditions",
        type=int,
        default=2,
        help="minimum distinct frame.condition.view_id values for identity handoff",
    )
    controlled_identity_handoff.add_argument(
        "--min-lighting-conditions",
        type=int,
        default=2,
        help="minimum distinct frame.condition.lighting_id values for identity handoff",
    )
    controlled_identity_handoff.add_argument(
        "--min-camera-motion-m",
        type=float,
        default=0.01,
        help="minimum camera translation in meters from frame.condition.camera_pose",
    )
    controlled_identity_handoff.add_argument(
        "--synthetic-smoke-failed",
        action="store_true",
        help="mark the synthetic prerequisite smoke gate as failed",
    )
    controlled_identity_handoff.add_argument("--min-real-or-public-rows", type=int, default=1)
    controlled_identity_handoff.add_argument("--require-pass", action="store_true")
    controlled_identity_handoff.set_defaults(
        handler=_object_state_controlled_identity_handoff
    )
    controlled_identity_bundle_handoff = object_state_subparsers.add_parser(
        "controlled-identity-bundle-handoff",
        help=(
            "import a controlled capture bundle and run the Stage 1 "
            "controlled identity handoff"
        ),
    )
    controlled_identity_bundle_handoff.add_argument("bundle_root", type=Path)
    controlled_identity_bundle_handoff.add_argument("trainable_artifact", type=Path)
    controlled_identity_bundle_handoff.add_argument("--output-dir", required=True, type=Path)
    controlled_identity_bundle_handoff.add_argument("--sample-json", default="sample.json")
    controlled_identity_bundle_handoff.add_argument("--objects-csv", default="objects.csv")
    controlled_identity_bundle_handoff.add_argument("--frames-csv", default="frames.csv")
    controlled_identity_bundle_handoff.add_argument(
        "--annotations-csv",
        default="annotations.csv",
    )
    controlled_identity_bundle_handoff.add_argument("--actions-csv", default="actions.csv")
    controlled_identity_bundle_handoff.add_argument(
        "--require-prediction-ready",
        action="store_true",
        help="also require the imported bundle to be prediction-stage ready",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--require-intervention-ready",
        action="store_true",
        help="also require the imported bundle to be intervention-stage ready",
    )
    controlled_identity_bundle_handoff.add_argument("--candidate-id")
    controlled_identity_bundle_handoff.add_argument(
        "--source",
        default="trainable_kernel_objectstate_nearest_pose_adapter",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--artifact-ref",
        action="append",
        dest="artifact_refs",
        help=(
            "candidate artifact reference to store in predictions; defaults to the "
            "trainable artifact path"
        ),
    )
    controlled_identity_bundle_handoff.add_argument("--max-centroid-distance", type=float)
    controlled_identity_bundle_handoff.add_argument("--min-idf1", type=float, default=0.95)
    controlled_identity_bundle_handoff.add_argument(
        "--min-track-retrieval-recall-at-1",
        type=float,
        default=0.95,
    )
    controlled_identity_bundle_handoff.add_argument(
        "--max-fragmentation-rate",
        type=float,
        default=0.05,
    )
    controlled_identity_bundle_handoff.add_argument(
        "--max-long-term-drift-rate",
        type=float,
        default=0.05,
    )
    controlled_identity_bundle_handoff.add_argument("--max-swap-rate", type=float, default=0.0)
    controlled_identity_bundle_handoff.add_argument(
        "--min-reconstruction-noise-robustness",
        type=float,
        default=0.95,
    )
    controlled_identity_bundle_handoff.add_argument(
        "--min-reconstruction-noise-variants",
        type=int,
        default=2,
    )
    controlled_identity_bundle_handoff.add_argument(
        "--allow-identity-collapse",
        action="store_true",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--check-artifact-refs",
        action="store_true",
        help="also require sample.artifact_refs paths to exist before handoff pass",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--min-rgb-bytes",
        type=int,
        default=1,
        help="minimum byte size for each frame RGB file",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
        help="minimum byte size for each frame Gaussian file",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--hash-files",
        action="store_true",
        help="include SHA256 hashes for valid frame RGB/Gaussian files",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--min-candidate-artifact-bytes",
        type=int,
        default=1,
        help="minimum byte size for the trainable ObjectState artifact file",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--hash-candidate-artifact",
        action="store_true",
        help="include a SHA256 hash for the trainable ObjectState artifact file",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--min-identity-scenario-frames",
        type=int,
        default=3,
        help="minimum frame count for identity scenario challenge audit",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--min-occlusion-fraction",
        type=float,
        default=0.5,
        help="occlusion_fraction threshold for identity scenario challenge audit",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--min-view-conditions",
        type=int,
        default=2,
        help="minimum distinct frame.condition.view_id values for identity handoff",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--min-lighting-conditions",
        type=int,
        default=2,
        help="minimum distinct frame.condition.lighting_id values for identity handoff",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--min-camera-motion-m",
        type=float,
        default=0.01,
        help="minimum camera translation in meters from frame.condition.camera_pose",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--synthetic-smoke-failed",
        action="store_true",
        help="mark the synthetic prerequisite smoke gate as failed",
    )
    controlled_identity_bundle_handoff.add_argument(
        "--min-real-or-public-rows",
        type=int,
        default=1,
    )
    controlled_identity_bundle_handoff.add_argument("--require-pass", action="store_true")
    controlled_identity_bundle_handoff.set_defaults(
        handler=_object_state_controlled_identity_bundle_handoff
    )
    controlled_reality_bundle_handoff = object_state_subparsers.add_parser(
        "controlled-reality-bundle-handoff",
        help=(
            "import a controlled capture bundle and run identity, prediction, "
            "and intervention reality gates"
        ),
    )
    controlled_reality_bundle_handoff.add_argument("bundle_root", type=Path)
    controlled_reality_bundle_handoff.add_argument("trainable_artifact", type=Path)
    controlled_reality_bundle_handoff.add_argument("prediction_candidates", type=Path)
    controlled_reality_bundle_handoff.add_argument("intervention_candidates", type=Path)
    controlled_reality_bundle_handoff.add_argument(
        "--output-dir",
        required=True,
        type=Path,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--sample-json",
        default="sample.json",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--objects-csv",
        default="objects.csv",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--frames-csv",
        default="frames.csv",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--annotations-csv",
        default="annotations.csv",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--actions-csv",
        default="actions.csv",
    )
    controlled_reality_bundle_handoff.add_argument("--candidate-id")
    controlled_reality_bundle_handoff.add_argument(
        "--source",
        default="trainable_kernel_objectstate_nearest_pose_adapter",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--artifact-ref",
        action="append",
        dest="artifact_refs",
        help=(
            "candidate artifact reference to store in identity predictions; "
            "defaults to the trainable artifact path"
        ),
    )
    controlled_reality_bundle_handoff.add_argument("--max-centroid-distance", type=float)
    controlled_reality_bundle_handoff.add_argument("--min-idf1", type=float, default=0.95)
    controlled_reality_bundle_handoff.add_argument(
        "--min-track-retrieval-recall-at-1",
        type=float,
        default=0.95,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--max-fragmentation-rate",
        type=float,
        default=0.05,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--max-long-term-drift-rate",
        type=float,
        default=0.05,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--max-swap-rate",
        type=float,
        default=0.0,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-reconstruction-noise-robustness",
        type=float,
        default=0.95,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-reconstruction-noise-variants",
        type=int,
        default=2,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--allow-identity-collapse",
        action="store_true",
    )
    controlled_reality_bundle_handoff.add_argument("--max-state-ade", type=float, default=0.05)
    controlled_reality_bundle_handoff.add_argument(
        "--max-prediction-gap-vs-history-model",
        type=float,
        default=0.02,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--max-error-ratio-vs-history-model",
        type=float,
        default=1.25,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-prediction-count",
        type=int,
        default=1,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--max-action-conditioned-ade",
        type=float,
        default=0.05,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-counterfactual-outcome-accuracy",
        type=float,
        default=0.95,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--max-wrong-direction-rate",
        type=float,
        default=0.0,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-intervention-gain",
        type=float,
        default=0.0,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-intervention-count",
        type=int,
        default=1,
    )
    controlled_reality_bundle_handoff.add_argument(
        "--check-artifact-refs",
        action="store_true",
        help="also require sample.artifact_refs paths to exist before handoff pass",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-rgb-bytes",
        type=int,
        default=1,
        help="minimum byte size for each frame RGB file",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-gaussian-bytes",
        type=int,
        default=1,
        help="minimum byte size for each frame Gaussian file",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--hash-files",
        action="store_true",
        help="include SHA256 hashes for valid frame RGB/Gaussian files",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--no-require-frame-formats",
        action="store_true",
        help="skip RGB/Gaussian frame file format signature checks",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-candidate-artifact-bytes",
        type=int,
        default=1,
        help="minimum byte size for the trainable ObjectState artifact file",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--hash-candidate-artifact",
        action="store_true",
        help="include a SHA256 hash for the trainable ObjectState artifact file",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-identity-scenario-frames",
        type=int,
        default=3,
        help="minimum frame count for identity scenario challenge audit",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-occlusion-fraction",
        type=float,
        default=0.5,
        help="occlusion_fraction threshold for identity scenario challenge audit",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-view-conditions",
        type=int,
        default=2,
        help="minimum distinct frame.condition.view_id values for identity handoff",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-lighting-conditions",
        type=int,
        default=2,
        help="minimum distinct frame.condition.lighting_id values for identity handoff",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-camera-motion-m",
        type=float,
        default=0.01,
        help="minimum camera translation in meters from frame.condition.camera_pose",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--synthetic-smoke-failed",
        action="store_true",
        help="mark the synthetic prerequisite smoke gate as failed",
    )
    controlled_reality_bundle_handoff.add_argument(
        "--min-real-or-public-rows",
        type=int,
        default=3,
    )
    controlled_reality_bundle_handoff.add_argument("--require-pass", action="store_true")
    controlled_reality_bundle_handoff.set_defaults(
        handler=_object_state_controlled_reality_bundle_handoff
    )

    object_field = subparsers.add_parser(
        "object-field",
        help="initialize and inspect soft object-slot fields",
    )
    object_field_subparsers = object_field.add_subparsers(
        dest="object_field_command",
        required=True,
    )

    field_init = object_field_subparsers.add_parser(
        "init",
        help="warm-start a soft object field from Gaussian features",
    )
    field_init.add_argument("input", type=Path)
    field_init.add_argument("--output", "-o", required=True, type=Path)
    field_init.add_argument("--slots", "-k", type=int, required=True)
    field_init.add_argument("--ply-output", type=Path)
    field_init.add_argument("--seed", type=int, default=0)
    field_init.add_argument("--max-iter", type=int, default=100)
    field_init.add_argument("--confidence", type=float, default=0.92)
    field_init.add_argument("--spatial-weight", type=float, default=1.0)
    field_init.add_argument("--color-weight", type=float, default=0.5)
    field_init.add_argument("--opacity-weight", type=float, default=0.2)
    field_init.add_argument("--no-normalize", action="store_true")
    field_init.add_argument("--smoothness", action="store_true")
    field_init.add_argument("--neighbors", type=int, default=4)
    field_init.add_argument("--max-smooth-points", type=int, default=1024)
    field_init.add_argument("--colorize", action="store_true")
    field_init.add_argument("--rewrite-sh", action="store_true")
    field_init.add_argument("--ascii", action="store_true", help="write ASCII PLY")
    field_init.set_defaults(handler=_object_field_init)

    field_export = object_field_subparsers.add_parser(
        "export",
        help="export hard object_id labels from an object field",
    )
    field_export.add_argument("input", type=Path)
    field_export.add_argument("--field", required=True, type=Path)
    field_export.add_argument("--output", "-o", required=True, type=Path)
    field_export.add_argument("--object-id-field", default="object_id")
    field_export.add_argument(
        "--min-confidence",
        type=float,
        help=(
            "send Gaussians whose max object probability is below this threshold "
            "to an unknown/background object"
        ),
    )
    field_export.add_argument(
        "--unknown-object-id",
        type=int,
        help="object_id to use for low-confidence Gaussians; defaults to the slot count",
    )
    field_export.add_argument("--colorize", action="store_true")
    field_export.add_argument("--rewrite-sh", action="store_true")
    field_export.add_argument("--ascii", action="store_true", help="write ASCII PLY")
    field_export.set_defaults(handler=_object_field_export)

    field_stats = object_field_subparsers.add_parser(
        "stats",
        help="print object field metrics",
    )
    field_stats.add_argument("field", type=Path)
    field_stats.add_argument(
        "--min-confidence",
        type=float,
        help="include an unknown/background object count below this max-probability threshold",
    )
    field_stats.add_argument(
        "--unknown-object-id",
        type=int,
        help="object_id to use for low-confidence Gaussians; defaults to the slot count",
    )
    field_stats.set_defaults(handler=_object_field_stats)

    field_emergence = object_field_subparsers.add_parser(
        "emergence",
        help="compute object emergence observability metrics",
    )
    field_emergence.add_argument("field", type=Path)
    field_emergence.add_argument("--cloud", type=Path, help="Gaussian PLY for spatial compactness")
    field_emergence.add_argument("--reference", type=Path, help="reference Object Field for stability")
    field_emergence.add_argument("--output", "-o", type=Path)
    field_emergence.set_defaults(handler=_object_field_emergence)

    field_emergence_curve = object_field_subparsers.add_parser(
        "emergence-curve",
        help="benchmark emergence metrics over mask-vote training iterations",
    )
    field_emergence_curve.add_argument("input", type=Path)
    field_emergence_curve.add_argument("--field", required=True, type=Path)
    field_emergence_curve.add_argument("--masks", required=True, type=Path)
    field_emergence_curve.add_argument("--heldout-masks", type=Path)
    field_emergence_curve.add_argument("--output", "-o", required=True, type=Path)
    field_emergence_curve.add_argument("--csv-output", type=Path)
    field_emergence_curve.add_argument("--iterations", type=int, default=100)
    field_emergence_curve.add_argument("--learning-rate", type=float, default=0.5)
    field_emergence_curve.add_argument("--eval-every", type=int, default=10)
    field_emergence_curve.add_argument("--max-frames", type=int)
    field_emergence_curve.add_argument("--heldout-max-frames", type=int)
    field_emergence_curve.add_argument(
        "--render-size",
        type=int,
        default=128,
        help="max image dimension for render occlusion probe",
    )
    field_emergence_curve.add_argument(
        "--no-render-occlusion",
        action="store_true",
        help="skip image-space render occlusion delta and keep mask-proxy metrics only",
    )
    field_emergence_curve.set_defaults(handler=_object_field_emergence_curve)

    field_emergence_report = object_field_subparsers.add_parser(
        "emergence-report",
        help="render one or more emergence curve JSON files as an HTML report",
    )
    field_emergence_report.add_argument("curves", nargs="+", type=Path)
    field_emergence_report.add_argument("--output", "-o", required=True, type=Path)
    field_emergence_report.add_argument(
        "--label",
        action="append",
        help="scene label; repeat once per curve JSON",
    )
    field_emergence_report.add_argument(
        "--title",
        default="Object Emergence Benchmark",
    )
    field_emergence_report.set_defaults(handler=_object_field_emergence_report)

    field_emergence_benchmark = object_field_subparsers.add_parser(
        "emergence-benchmark",
        help="run a manifest-defined emergence curve benchmark suite",
    )
    field_emergence_benchmark.add_argument("manifest", type=Path)
    field_emergence_benchmark.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/benchmarks/semantic-smoke"),
    )
    field_emergence_benchmark.add_argument("--report", type=Path)
    field_emergence_benchmark.add_argument("--summary", type=Path)
    field_emergence_benchmark.add_argument(
        "--strict",
        action="store_true",
        help="exit with an error if any manifest threshold check fails",
    )
    field_emergence_benchmark.set_defaults(handler=_object_field_emergence_benchmark)

    field_nerf = object_field_subparsers.add_parser(
        "inspect-nerf",
        help="inspect a NeRF-style posed image dataset",
    )
    field_nerf.add_argument("dataset", type=Path)
    field_nerf.add_argument("--output", "-o", type=Path)
    field_nerf.set_defaults(handler=_object_field_inspect_nerf)

    field_vote = object_field_subparsers.add_parser(
        "vote-masks",
        help="project 2D masks to Gaussians and train object logits",
    )
    field_vote.add_argument("input", type=Path)
    field_vote.add_argument("--field", required=True, type=Path)
    field_vote.add_argument("--masks", required=True, type=Path)
    field_vote.add_argument("--output", "-o", required=True, type=Path)
    field_vote.add_argument("--summary-output", type=Path)
    field_vote.add_argument("--ply-output", type=Path)
    field_vote.add_argument("--iterations", type=int, default=100)
    field_vote.add_argument("--learning-rate", type=float, default=0.5)
    field_vote.add_argument("--max-frames", type=int)
    field_vote.add_argument(
        "--visibility-mode",
        choices=["projected", "depth-buffer"],
        default="projected",
        help="mask voting visibility filter; default keeps legacy projected voting",
    )
    field_vote.add_argument(
        "--depth-tolerance",
        type=float,
        default=0.0,
        help="world-depth tolerance for --visibility-mode depth-buffer",
    )
    field_vote.add_argument(
        "--background-slot",
        type=int,
        help=(
            "train this slot from projected Gaussians that are visible in a mask frame "
            "but do not hit any foreground mask"
        ),
    )
    field_vote.add_argument(
        "--background-weight",
        type=float,
        default=1.0,
        help="vote weight for --background-slot negative evidence",
    )
    field_vote.add_argument(
        "--min-confidence",
        type=float,
        help=(
            "when writing --ply-output, send Gaussians below this max-probability "
            "threshold to an unknown/background object"
        ),
    )
    field_vote.add_argument(
        "--unknown-object-id",
        type=int,
        help="object_id to use for low-confidence Gaussians; defaults to the slot count",
    )
    field_vote.add_argument("--colorize", action="store_true")
    field_vote.add_argument("--rewrite-sh", action="store_true")
    field_vote.add_argument("--ascii", action="store_true", help="write ASCII PLY")
    field_vote.set_defaults(handler=_object_field_vote_masks)

    field_vote_diagnostics = object_field_subparsers.add_parser(
        "vote-diagnostics",
        help="compare projected mask voting against depth-buffer visibility voting",
    )
    field_vote_diagnostics.add_argument("input", type=Path)
    field_vote_diagnostics.add_argument("--masks", required=True, type=Path)
    field_vote_diagnostics.add_argument("--slots", required=True, type=int)
    field_vote_diagnostics.add_argument("--output", "-o", type=Path)
    field_vote_diagnostics.add_argument("--max-frames", type=int)
    field_vote_diagnostics.add_argument(
        "--background-slot",
        type=int,
        help="optional slot for projected-but-unmatched background evidence",
    )
    field_vote_diagnostics.add_argument(
        "--background-weight",
        type=float,
        default=1.0,
        help="vote weight for --background-slot negative evidence",
    )
    field_vote_diagnostics.add_argument(
        "--depth-tolerance",
        type=float,
        default=0.0,
        help="world-depth tolerance for the depth-buffer diagnostic",
    )
    field_vote_diagnostics.set_defaults(handler=_object_field_vote_diagnostics)

    masks = subparsers.add_parser("masks", help="build mask manifests for Object Field voting")
    masks_subparsers = masks.add_subparsers(dest="masks_command", required=True)

    nerf_alpha = masks_subparsers.add_parser(
        "from-nerf-alpha",
        help="convert NeRF Synthetic RGBA alpha channels to mask manifest files",
    )
    nerf_alpha.add_argument("dataset", type=Path)
    nerf_alpha.add_argument("--output", "-o", required=True, type=Path)
    nerf_alpha.add_argument("--split", default="train")
    nerf_alpha.add_argument("--max-frames", type=int)
    nerf_alpha.add_argument("--slot", type=int, default=0)
    nerf_alpha.add_argument("--label", default="foreground")
    nerf_alpha.add_argument("--threshold", type=int, default=1)
    nerf_alpha.set_defaults(handler=_masks_from_nerf_alpha)

    nerf_alpha_fgbg = masks_subparsers.add_parser(
        "from-nerf-alpha-fgbg",
        help="convert NeRF Synthetic RGBA alpha to foreground/background masks plus ignore boundary",
    )
    nerf_alpha_fgbg.add_argument("dataset", type=Path)
    nerf_alpha_fgbg.add_argument("--output", "-o", required=True, type=Path)
    nerf_alpha_fgbg.add_argument("--split", default="train")
    nerf_alpha_fgbg.add_argument("--max-frames", type=int)
    nerf_alpha_fgbg.add_argument("--background-slot", type=int, default=0)
    nerf_alpha_fgbg.add_argument("--foreground-slot", type=int, default=1)
    nerf_alpha_fgbg.add_argument("--background-threshold", type=int, default=20)
    nerf_alpha_fgbg.add_argument("--foreground-threshold", type=int, default=200)
    nerf_alpha_fgbg.add_argument("--background-confidence", type=float, default=0.05)
    nerf_alpha_fgbg.add_argument("--foreground-confidence", type=float, default=1.0)
    nerf_alpha_fgbg.set_defaults(handler=_masks_from_nerf_alpha_fgbg)

    nerf_rgba_colors = masks_subparsers.add_parser(
        "from-nerf-rgba-colors",
        help="convert NeRF Synthetic Lego RGBA colors to multi-slot mask manifest files",
    )
    nerf_rgba_colors.add_argument("dataset", type=Path)
    nerf_rgba_colors.add_argument("--output", "-o", required=True, type=Path)
    nerf_rgba_colors.add_argument("--split", default="train")
    nerf_rgba_colors.add_argument("--max-frames", type=int)
    nerf_rgba_colors.add_argument("--alpha-threshold", type=int, default=16)
    nerf_rgba_colors.set_defaults(handler=_masks_from_nerf_rgba_colors)

    nerf_sam = masks_subparsers.add_parser(
        "from-nerf-sam",
        help="run optional Segment Anything automatic masks on NeRF-style images",
    )
    nerf_sam.add_argument("dataset", type=Path)
    nerf_sam.add_argument("--output", "-o", required=True, type=Path)
    nerf_sam.add_argument("--checkpoint", required=True, type=Path)
    nerf_sam.add_argument("--model-type", default="vit_b")
    nerf_sam.add_argument("--device", default="cpu")
    nerf_sam.add_argument("--split", default="train")
    nerf_sam.add_argument("--max-frames", type=int)
    nerf_sam.add_argument("--max-masks-per-frame", type=int, default=8)
    nerf_sam.add_argument("--min-area", type=int, default=1)
    nerf_sam.add_argument("--max-area-fraction", type=float, default=1.0)
    nerf_sam.add_argument("--max-image-size", type=int)
    nerf_sam.add_argument("--points-per-side", type=int, default=32)
    nerf_sam.add_argument("--pred-iou-thresh", type=float, default=0.88)
    nerf_sam.add_argument("--stability-score-thresh", type=float, default=0.95)
    nerf_sam.set_defaults(handler=_masks_from_nerf_sam)

    split_manifest = masks_subparsers.add_parser(
        "split-manifest",
        help="split an existing mask manifest into train and held-out frame manifests",
    )
    split_manifest.add_argument("source", type=Path)
    split_manifest.add_argument("--train-output", required=True, type=Path)
    split_manifest.add_argument("--heldout-output", required=True, type=Path)
    split_manifest.add_argument("--heldout-every", type=int, default=4)
    split_manifest.add_argument("--heldout-offset", type=int)
    split_manifest.set_defaults(handler=_masks_split_manifest)

    validate_manifest = masks_subparsers.add_parser(
        "validate",
        help="validate mask manifest image/mask shape, slots, overlap, empty masks, and ignore masks",
    )
    validate_manifest.add_argument("manifest", type=Path)
    validate_manifest.add_argument("--dataset", type=Path)
    validate_manifest.add_argument("--summary-output", type=Path)
    validate_manifest.add_argument("--max-overlap-fraction", type=float, default=0.0)
    validate_manifest.add_argument("--max-mask-area-fraction", type=float, default=0.98)
    validate_manifest.add_argument("--allow-empty", action="store_true")
    validate_manifest.add_argument("--strict", action="store_true")
    validate_manifest.add_argument("--max-report-frames", type=int, default=8)
    validate_manifest.set_defaults(handler=_masks_validate)

    score_clip = masks_subparsers.add_parser(
        "score-clip",
        help="score mask crops against text labels and cache CLIP scores into a mask manifest",
    )
    score_clip.add_argument("manifest", type=Path)
    score_clip.add_argument("--output", "-o", required=True, type=Path)
    score_clip.add_argument("--dataset", type=Path)
    score_clip.add_argument("--labels", nargs="+", default=[])
    score_clip.add_argument("--labels-file", type=Path)
    score_clip.add_argument(
        "--label-preset",
        action="append",
        choices=sorted(CLIP_LABEL_PRESETS),
        help="append a curated label set before labels-file / --labels de-duplication",
    )
    score_clip.add_argument("--summary-output", type=Path)
    score_clip.add_argument(
        "--backend",
        choices=["transformers", "hash"],
        default="transformers",
        help="use transformers for real CLIP inference; hash is a deterministic diagnostic backend",
    )
    score_clip.add_argument("--model", default="openai/clip-vit-base-patch32")
    score_clip.add_argument("--device")
    score_clip.add_argument("--max-frames", type=int)
    score_clip.add_argument("--max-masks", type=int)
    score_clip.add_argument("--crop-padding", type=float, default=0.05)
    score_clip.add_argument(
        "--background-fill",
        choices=["white", "black", "gray", "mean", "image"],
        default="white",
        help="fill non-mask pixels inside each crop before CLIP scoring",
    )
    score_clip.add_argument(
        "--prompt-template",
        action="append",
        help="CLIP text prompt template containing {label}; may be repeated",
    )
    score_clip.add_argument(
        "--background-labels",
        nargs="+",
        help="labels counted as background in naming quality gate",
    )
    score_clip.add_argument("--min-unique-top-labels", type=int, default=2)
    score_clip.add_argument("--max-top-label-fraction", type=float, default=0.75)
    score_clip.add_argument("--max-background-label-fraction", type=float, default=0.5)
    score_clip.add_argument(
        "--require-naming-quality",
        action="store_true",
        help="exit non-zero when the naming diversity/background gate fails",
    )
    score_clip.add_argument("--overwrite-scores", action="store_true")
    score_clip.set_defaults(handler=_masks_score_clip)

    align_slots = masks_subparsers.add_parser(
        "align-slots",
        help="rewrite a mask manifest with cross-view stable slots and optional CLIP-score names",
    )
    align_slots.add_argument("manifest", type=Path)
    align_slots.add_argument("--cloud", required=True, type=Path)
    align_slots.add_argument("--output", "-o", required=True, type=Path)
    align_slots.add_argument("--max-frames", type=int)
    align_slots.add_argument("--max-slots", type=int)
    align_slots.add_argument("--min-iou", type=float, default=0.05)
    align_slots.add_argument("--min-shared-gaussians", type=int, default=1)
    align_slots.add_argument("--min-mask-area", type=int, default=0)
    align_slots.add_argument("--min-mask-area-fraction", type=float, default=0.0)
    align_slots.add_argument(
        "--exclude-top-labels",
        nargs="+",
        help="drop masks whose top CLIP label matches one of these labels before slot alignment",
    )
    align_slots.add_argument(
        "--exclude-background-top-labels",
        action="store_true",
        help="drop masks whose top CLIP label is one of the configured background labels",
    )
    align_slots.add_argument(
        "--background-labels",
        nargs="+",
        help="labels counted as background for filtering and slot quality",
    )
    align_slots.add_argument("--min-named-slots", type=int, default=1)
    align_slots.add_argument("--min-unique-slot-labels", type=int, default=2)
    align_slots.add_argument("--max-slot-label-fraction", type=float, default=0.5)
    align_slots.add_argument("--max-background-slot-fraction", type=float, default=0.25)
    align_slots.add_argument(
        "--foreground-only-slot-names",
        action="store_true",
        help="do not allow configured background labels to become aligned slot names",
    )
    align_slots.add_argument(
        "--unique-slot-names",
        action="store_true",
        help="prefer an unused slot label when another candidate is available",
    )
    align_slots.add_argument(
        "--slot-name-diversity-penalty",
        type=float,
        default=0.0,
        help="divide repeated label scores by 1 + penalty * prior_slot_uses",
    )
    align_slots.add_argument(
        "--min-slot-support-gaussians",
        type=int,
        default=0,
        help="drop aligned slots with fewer Gaussian supports unless needed to keep min-balanced-slots",
    )
    align_slots.add_argument(
        "--min-slot-support-ratio",
        type=float,
        default=0.0,
        help="drop aligned slots whose support is below this fraction of the largest slot",
    )
    align_slots.add_argument(
        "--min-balanced-slots",
        type=int,
        default=1,
        help="minimum slots to keep even when support rebalance thresholds would drop them",
    )
    align_slots.add_argument(
        "--recover-foreground-coverage",
        action="store_true",
        help=(
            "keep foreground, non-background masks dropped by slot rebalance as coverage-only "
            "supervision by mapping them to a compatible kept slot"
        ),
    )
    align_slots.add_argument(
        "--require-slot-quality",
        action="store_true",
        help="exit non-zero when the slot-level naming quality gate fails",
    )
    align_slots.set_defaults(handler=_masks_align_slots)

    compare_baselines = masks_subparsers.add_parser(
        "compare-baselines",
        help="compare CLIP slot naming against color-mask, KMeans, alpha, and other baselines",
    )
    compare_baselines.add_argument(
        "--candidate",
        action="append",
        required=True,
        help="candidate evidence binding as name=path; repeat the same name to merge evidence",
    )
    compare_baselines.add_argument("--output", "-o", required=True, type=Path)
    compare_baselines.add_argument("--markdown-output", type=Path)
    compare_baselines.add_argument("--min-supervised-fraction", type=float, default=0.2)
    compare_baselines.add_argument("--max-vote-conflict-fraction", type=float, default=0.05)
    compare_baselines.add_argument("--min-slot-balance-score", type=float, default=0.01)
    compare_baselines.add_argument("--min-object-active-slots", type=int, default=2)
    compare_baselines.add_argument("--object-id-field", default="object_id")
    compare_baselines.add_argument(
        "--require-promotion-ready",
        action="store_true",
        help="exit non-zero unless at least one semantic candidate passes promotion policy",
    )
    compare_baselines.set_defaults(handler=_masks_compare_baselines)

    demo = subparsers.add_parser("demo", help="build reproducible ObjGauss demos")
    demo_subparsers = demo.add_subparsers(dest="demo_command", required=True)

    v1_closure = demo_subparsers.add_parser(
        "v1-closure",
        help="build the current ObjGauss v1 closed-loop acceptance demo",
    )
    v1_closure.add_argument("--input", type=Path, default=Path("public/samples/plush_objects.ply"))
    v1_closure.add_argument("--splat", type=Path, default=Path("public/samples/plush.splat"))
    v1_closure.add_argument("--output-dir", type=Path, default=Path("outputs/demos/v1-closure"))
    v1_closure.add_argument("--public-dir", type=Path, default=Path("public/samples"))
    v1_closure.add_argument("--no-public-copy", action="store_true")
    v1_closure.add_argument("--image-size", type=int, default=512)
    v1_closure.add_argument("--iterations", type=int, default=160)
    v1_closure.add_argument("--learning-rate", type=float, default=1.0)
    v1_closure.set_defaults(handler=_demo_v1_closure)

    verify_v1 = demo_subparsers.add_parser(
        "verify-v1-closure",
        help="verify the generated ObjGauss v1 closed-loop acceptance demo",
    )
    verify_v1.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("outputs/demos/v1-closure/v1-closure-manifest.json"),
    )
    verify_v1.add_argument("--asset-library", type=Path, default=Path("src/assetLibrary.js"))
    verify_v1.add_argument("--no-require-public-copy", action="store_true")
    verify_v1.set_defaults(handler=_demo_verify_v1_closure)

    plush_semantic = demo_subparsers.add_parser(
        "plush-semantic-closure",
        help="build a real Plush 3DGS closure demo from projected 2D color masks",
    )
    plush_semantic.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/assets/converted/plush.ply"),
    )
    plush_semantic.add_argument("--splat", type=Path, default=Path("public/samples/plush.splat"))
    plush_semantic.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/demos/plush-semantic-closure"),
    )
    plush_semantic.add_argument("--public-dir", type=Path, default=Path("public/samples"))
    plush_semantic.add_argument("--no-public-copy", action="store_true")
    plush_semantic.add_argument("--image-size", type=int, default=512)
    plush_semantic.add_argument("--iterations", type=int, default=160)
    plush_semantic.add_argument("--learning-rate", type=float, default=1.0)
    plush_semantic.set_defaults(handler=_demo_plush_semantic_closure)

    verify_plush_semantic = demo_subparsers.add_parser(
        "verify-plush-semantic-closure",
        help="verify the generated Plush semantic closure demo",
    )
    verify_plush_semantic.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("outputs/demos/plush-semantic-closure/plush-semantic-closure-manifest.json"),
    )
    verify_plush_semantic.add_argument("--asset-library", type=Path, default=Path("src/assetLibrary.js"))
    verify_plush_semantic.add_argument("--no-require-public-copy", action="store_true")
    verify_plush_semantic.add_argument("--min-views", type=int, default=2)
    verify_plush_semantic.set_defaults(handler=_demo_verify_plush_semantic_closure)

    lego_alpha = demo_subparsers.add_parser(
        "lego-alpha-closure",
        help="build a NeRF Lego alpha/color-mask ObjGauss closure proxy demo",
    )
    lego_alpha.add_argument(
        "--dataset",
        type=Path,
        default=Path("outputs/assets/training/nerf-synthetic-lego"),
    )
    lego_alpha.add_argument("--output-dir", type=Path, default=Path("outputs/demos/lego-alpha-closure"))
    lego_alpha.add_argument("--public-dir", type=Path, default=Path("public/samples"))
    lego_alpha.add_argument("--no-public-copy", action="store_true")
    lego_alpha.add_argument("--split", default="train")
    lego_alpha.add_argument("--max-frames", type=int, default=12)
    lego_alpha.add_argument("--sample-stride", type=int, default=8)
    lego_alpha.add_argument("--depth", type=float, default=4.0)
    lego_alpha.add_argument("--alpha-threshold", type=int, default=16)
    lego_alpha.add_argument("--iterations", type=int, default=160)
    lego_alpha.add_argument("--learning-rate", type=float, default=1.0)
    lego_alpha.set_defaults(handler=_demo_lego_alpha_closure)

    verify_lego_alpha = demo_subparsers.add_parser(
        "verify-lego-alpha-closure",
        help="verify the generated NeRF Lego alpha closure proxy demo",
    )
    verify_lego_alpha.add_argument(
        "manifest",
        nargs="?",
        type=Path,
        default=Path("outputs/demos/lego-alpha-closure/lego-alpha-closure-manifest.json"),
    )
    verify_lego_alpha.add_argument("--asset-library", type=Path, default=Path("src/assetLibrary.js"))
    verify_lego_alpha.add_argument("--no-require-public-copy", action="store_true")
    verify_lego_alpha.add_argument("--min-frames", type=int, default=2)
    verify_lego_alpha.set_defaults(handler=_demo_verify_lego_alpha_closure)

    audit_goal = demo_subparsers.add_parser(
        "audit-v1-goal",
        help="audit the current evidence against the ObjGauss v1 phase goal",
    )
    audit_goal.add_argument(
        "--v1-manifest",
        type=Path,
        default=Path("outputs/demos/v1-closure/v1-closure-manifest.json"),
    )
    audit_goal.add_argument(
        "--lego-manifest",
        type=Path,
        default=Path("outputs/demos/lego-alpha-closure/lego-alpha-closure-manifest.json"),
    )
    audit_goal.add_argument(
        "--semantic-manifest",
        type=Path,
        default=Path("outputs/demos/plush-semantic-closure/plush-semantic-closure-manifest.json"),
    )
    audit_goal.add_argument(
        "--trained-manifest",
        type=Path,
        default=Path("outputs/assets/gaussians/nerf-lego-trained/training-output-manifest.json"),
    )
    audit_goal.add_argument("--asset-library", type=Path, default=Path("src/assetLibrary.js"))
    audit_goal.add_argument("--allow-incomplete", action="store_true")
    audit_goal.set_defaults(handler=_demo_audit_v1_goal)

    training = subparsers.add_parser(
        "training",
        help="register external 3DGS training outputs for ObjGauss",
    )
    training_subparsers = training.add_subparsers(dest="training_command", required=True)
    register_output = training_subparsers.add_parser(
        "register-output",
        help="ingest a trained Gaussian PLY or splat and optionally run mask voting",
    )
    register_output.add_argument("input", type=Path)
    register_output.add_argument("--asset-id", required=True)
    register_output.add_argument("--output-dir", required=True, type=Path)
    register_output.add_argument("--dataset", type=Path)
    register_output.add_argument("--masks", type=Path)
    register_output.add_argument("--slots", type=int)
    register_output.add_argument("--public-dir", type=Path, default=Path("public/samples"))
    register_output.add_argument("--public-name")
    register_output.add_argument("--no-public-copy", action="store_true")
    register_output.add_argument("--iterations", type=int, default=100)
    register_output.add_argument("--learning-rate", type=float, default=0.5)
    register_output.add_argument(
        "--background-slot",
        type=int,
        help=(
            "train this Object Field slot from projected Gaussians that are visible "
            "but unmatched by foreground masks"
        ),
    )
    register_output.add_argument(
        "--background-weight",
        type=float,
        default=1.0,
        help="vote weight for --background-slot negative evidence",
    )
    register_output.add_argument(
        "--object-min-confidence",
        type=float,
        help=(
            "send trained Object Field assignments below this max-probability threshold "
            "to an unknown/background object in object_aware_gaussians.ply"
        ),
    )
    register_output.add_argument(
        "--unknown-object-id",
        type=int,
        help="object_id to use for low-confidence Gaussians; defaults to the slot count",
    )
    register_output.add_argument("--no-colorize", action="store_true")
    register_output.set_defaults(handler=_training_register_output)

    kernel_mvp = training_subparsers.add_parser(
        "kernel-mvp",
        help="run the dependency-free ObjGauss v1 trainable kernel smoke loop",
    )
    kernel_mvp.add_argument("--slots", type=int, default=2)
    kernel_mvp.add_argument("--iterations", type=int, default=40)
    kernel_mvp.add_argument("--learning-rate", type=float, default=0.35)
    kernel_mvp.add_argument("--render-weight", type=float, default=1.0)
    kernel_mvp.add_argument("--image-render-weight", type=float, default=0.0)
    kernel_mvp.add_argument("--image-renderer", choices=("point", "gsplat"), default="point")
    kernel_mvp.add_argument("--object-weight", type=float, default=1.0)
    kernel_mvp.add_argument("--temporal-weight", type=float, default=0.02)
    kernel_mvp.add_argument("--seed", type=int, default=0)
    kernel_mvp.add_argument("--record-every", type=int)
    kernel_mvp.add_argument("--summary-output", type=Path)
    kernel_mvp.add_argument("--require-loss-decrease", action="store_true")
    kernel_mvp.set_defaults(handler=_training_kernel_mvp)

    kernel_sample = training_subparsers.add_parser(
        "kernel-sample",
        help="run the trainable kernel MVP on a small Gaussian PLY sample",
    )
    kernel_sample.add_argument("input", type=Path)
    kernel_sample.add_argument("--slots", type=int)
    kernel_sample.add_argument("--frames", type=int, default=2)
    kernel_sample.add_argument("--max-points", type=int, default=24)
    kernel_sample.add_argument("--object-id-field", default="object_id")
    kernel_sample.add_argument("--temporal-offset", type=float, default=0.01)
    kernel_sample.add_argument("--bind-image-targets", action="store_true")
    kernel_sample.add_argument("--image-width", type=int, default=16)
    kernel_sample.add_argument("--image-height", type=int, default=16)
    kernel_sample.add_argument("--point-radius", type=int, default=1)
    kernel_sample.add_argument(
        "--visibility-policy",
        choices=("covered_pixels", "all_pixels"),
        default="covered_pixels",
    )
    kernel_sample.add_argument("--iterations", type=int, default=40)
    kernel_sample.add_argument("--learning-rate", type=float, default=0.35)
    kernel_sample.add_argument("--render-weight", type=float, default=1.0)
    kernel_sample.add_argument("--image-render-weight", type=float, default=0.0)
    kernel_sample.add_argument("--image-renderer", choices=("point", "gsplat"), default="point")
    kernel_sample.add_argument("--object-weight", type=float, default=1.0)
    kernel_sample.add_argument("--temporal-weight", type=float, default=0.02)
    kernel_sample.add_argument("--seed", type=int, default=0)
    kernel_sample.add_argument("--record-every", type=int)
    kernel_sample.add_argument("--summary-output", type=Path)
    kernel_sample.add_argument("--model-output", type=Path)
    kernel_sample.add_argument(
        "--quality-report-output",
        type=Path,
        help="write an objgauss-object-state-quality-report-v1 derived from --model-output",
    )
    kernel_sample.add_argument("--quality-report-id")
    kernel_sample.add_argument(
        "--manifest-output",
        type=Path,
        help="write an objgauss-model-artifact-manifest-v1 that exposes --model-output as browser-ready trainable_kernel",
    )
    kernel_sample.add_argument("--manifest-id")
    kernel_sample.add_argument("--manifest-asset-id")
    kernel_sample.add_argument("--manifest-name")
    kernel_sample.add_argument(
        "--manifest-license",
        default="local trainable kernel artifact; verify source license before release",
    )
    kernel_sample.add_argument("--require-loss-decrease", action="store_true")
    kernel_sample.set_defaults(handler=_training_kernel_sample)

    decoder_mvp = training_subparsers.add_parser(
        "decoder-mvp",
        help="train only ObjectState Gaussian decoder colors against image renderer loss",
    )
    decoder_mvp.add_argument("input", type=Path)
    decoder_mvp.add_argument("--solver-checkpoint", type=Path)
    decoder_mvp.add_argument("--slots", type=int)
    decoder_mvp.add_argument("--frames", type=int, default=2)
    decoder_mvp.add_argument("--max-points", type=int, default=24)
    decoder_mvp.add_argument("--object-id-field", default="object_id")
    decoder_mvp.add_argument("--temporal-offset", type=float, default=0.01)
    decoder_mvp.add_argument("--image-width", type=int, default=16)
    decoder_mvp.add_argument("--image-height", type=int, default=16)
    decoder_mvp.add_argument("--point-radius", type=int, default=1)
    decoder_mvp.add_argument(
        "--visibility-policy",
        choices=("covered_pixels", "all_pixels"),
        default="covered_pixels",
    )
    decoder_mvp.add_argument("--iterations", type=int, default=8)
    decoder_mvp.add_argument("--learning-rate", type=float, default=0.5)
    decoder_mvp.add_argument("--render-weight", type=float, default=0.0)
    decoder_mvp.add_argument("--image-render-weight", type=float, default=1.0)
    decoder_mvp.add_argument("--image-renderer", choices=("point", "gsplat"), default="point")
    decoder_mvp.add_argument("--gaussian-scale", type=float, default=0.5)
    decoder_mvp.add_argument("--gaussian-opacity", type=float, default=1.0)
    decoder_mvp.add_argument("--object-weight", type=float, default=0.0)
    decoder_mvp.add_argument("--temporal-weight", type=float, default=0.0)
    decoder_mvp.add_argument("--seed", type=int, default=0)
    decoder_mvp.add_argument("--record-every", type=int)
    decoder_mvp.add_argument("--summary-output", type=Path)
    decoder_mvp.add_argument("--require-loss-decrease", action="store_true")
    decoder_mvp.add_argument("--require-image-render-loss-decrease", action="store_true")
    decoder_mvp.set_defaults(handler=_training_decoder_mvp)

    solver_decoder_mvp = training_subparsers.add_parser(
        "solver-decoder-mvp",
        help="jointly train Object Emergence Solver assignment and decoder colors",
    )
    solver_decoder_mvp.add_argument("input", type=Path)
    solver_decoder_mvp.add_argument("--solver-checkpoint", type=Path)
    solver_decoder_mvp.add_argument("--resume-checkpoint", type=Path)
    solver_decoder_mvp.add_argument("--slots", type=int)
    solver_decoder_mvp.add_argument("--frames", type=int, default=2)
    solver_decoder_mvp.add_argument("--max-points", type=int, default=24)
    solver_decoder_mvp.add_argument("--object-id-field", default="object_id")
    solver_decoder_mvp.add_argument("--temporal-offset", type=float, default=0.01)
    solver_decoder_mvp.add_argument("--image-width", type=int, default=16)
    solver_decoder_mvp.add_argument("--image-height", type=int, default=16)
    solver_decoder_mvp.add_argument("--point-radius", type=int, default=1)
    solver_decoder_mvp.add_argument(
        "--visibility-policy",
        choices=("covered_pixels", "all_pixels"),
        default="covered_pixels",
    )
    solver_decoder_mvp.add_argument("--iterations", type=int, default=4)
    solver_decoder_mvp.add_argument("--solver-learning-rate", type=float, default=0.05)
    solver_decoder_mvp.add_argument("--decoder-learning-rate", type=float, default=0.5)
    solver_decoder_mvp.add_argument("--freeze-solver", action="store_true")
    solver_decoder_mvp.add_argument("--freeze-decoder-colors", action="store_true")
    solver_decoder_mvp.add_argument("--train-decoder-opacity", action="store_true")
    solver_decoder_mvp.add_argument("--freeze-decoder-opacity", action="store_true")
    solver_decoder_mvp.add_argument("--decoder-opacity-learning-rate", type=float, default=0.02)
    solver_decoder_mvp.add_argument("--decoder-opacity-init-logit", type=float, default=6.0)
    solver_decoder_mvp.add_argument("--train-decoder-scale", action="store_true")
    solver_decoder_mvp.add_argument("--freeze-decoder-scale", action="store_true")
    solver_decoder_mvp.add_argument("--decoder-scale-learning-rate", type=float, default=0.02)
    solver_decoder_mvp.add_argument("--decoder-scale-init-log-offset", type=float, default=0.0)
    solver_decoder_mvp.add_argument("--image-render-weight", type=float, default=1.0)
    solver_decoder_mvp.add_argument("--image-renderer", choices=("point", "gsplat"), default="point")
    solver_decoder_mvp.add_argument("--gaussian-scale", type=float, default=0.5)
    solver_decoder_mvp.add_argument("--gaussian-opacity", type=float, default=1.0)
    solver_decoder_mvp.add_argument("--solver-temperature", type=float)
    solver_decoder_mvp.add_argument("--object-weight", type=float, default=0.1)
    solver_decoder_mvp.add_argument("--entropy-weight", type=float, default=0.0)
    solver_decoder_mvp.add_argument("--balance-weight", type=float, default=0.0)
    solver_decoder_mvp.add_argument("--temporal-weight", type=float, default=0.0)
    solver_decoder_mvp.add_argument("--seed", type=int, default=0)
    solver_decoder_mvp.add_argument("--record-every", type=int)
    solver_decoder_mvp.add_argument("--loss-log-every", type=int)
    solver_decoder_mvp.add_argument("--checkpoint-every", type=int)
    solver_decoder_mvp.add_argument("--run-output-dir", type=Path)
    solver_decoder_mvp.add_argument("--tensorboard-logdir", type=Path)
    solver_decoder_mvp.add_argument("--vram-reserve-gb", type=int, default=1)
    solver_decoder_mvp.add_argument("--summary-output", type=Path)
    solver_decoder_mvp.add_argument("--checkpoint-output", type=Path)
    solver_decoder_mvp.add_argument("--include-weights", action="store_true")
    solver_decoder_mvp.add_argument("--include-assignments", action="store_true")
    solver_decoder_mvp.add_argument("--require-loss-decrease", action="store_true")
    solver_decoder_mvp.add_argument("--require-image-render-loss-decrease", action="store_true")
    solver_decoder_mvp.add_argument("--require-assignment-stability-not-degrade", action="store_true")
    solver_decoder_mvp.set_defaults(handler=_training_solver_decoder_mvp)

    eval_objectstate = training_subparsers.add_parser(
        "eval-objectstate",
        help="evaluate ObjectState stability from a solver-decoder joint checkpoint",
    )
    eval_objectstate.add_argument("input", type=Path)
    eval_objectstate.add_argument("--checkpoint", required=True, type=Path)
    eval_objectstate.add_argument("--slots", type=int)
    eval_objectstate.add_argument("--frames", type=int)
    eval_objectstate.add_argument("--max-points", type=int)
    eval_objectstate.add_argument("--object-id-field", default="object_id")
    eval_objectstate.add_argument("--temporal-offset", type=float, default=0.01)
    eval_objectstate.add_argument("--seed", type=int, default=0)
    eval_objectstate.add_argument("--solver-temperature", type=float)
    eval_objectstate.add_argument("--entropy-threshold", type=float, default=0.6)
    eval_objectstate.add_argument("--purity-threshold", type=float, default=0.8)
    eval_objectstate.add_argument("--collapse-mass-fraction", type=float, default=0.9)
    eval_objectstate.add_argument("--assignment-confidence-floor", type=float, default=0.4)
    eval_objectstate.add_argument("--summary-output", type=Path)
    eval_objectstate.add_argument("--require-pass", action="store_true")
    eval_objectstate.set_defaults(handler=_training_eval_objectstate)

    real_sample_handoff = training_subparsers.add_parser(
        "real-sample-v2-handoff",
        help="export real-sample v2 checkpoint, restore validation, and HTML effect preview",
    )
    real_sample_handoff.add_argument("input", type=Path)
    real_sample_handoff.add_argument("--slots", type=int)
    real_sample_handoff.add_argument("--frames", type=int, default=2)
    real_sample_handoff.add_argument("--max-points", type=int, default=24)
    real_sample_handoff.add_argument("--object-id-field", default="object_id")
    real_sample_handoff.add_argument("--temporal-offset", type=float, default=0.01)
    real_sample_handoff.add_argument("--image-width", type=int, default=12)
    real_sample_handoff.add_argument("--image-height", type=int, default=12)
    real_sample_handoff.add_argument("--point-radius", type=int, default=1)
    real_sample_handoff.add_argument(
        "--visibility-policy",
        choices=("covered_pixels", "all_pixels"),
        default="covered_pixels",
    )
    real_sample_handoff.add_argument("--iterations", type=int, default=100)
    real_sample_handoff.add_argument("--learning-rate", type=float, default=0.4)
    real_sample_handoff.add_argument(
        "--temperature-candidates",
        type=float,
        nargs="+",
        help="temperature sweep candidates; defaults to 1.0 0.75 0.5 0.35 0.25",
    )
    real_sample_handoff.add_argument("--baseline-temperature", type=float, default=1.0)
    real_sample_handoff.add_argument("--image-renderer", choices=("point", "gsplat"), default="point")
    real_sample_handoff.add_argument("--seed", type=int, default=4)
    real_sample_handoff.add_argument("--vram-reserve-gb", type=int, default=1)
    real_sample_handoff.add_argument("--summary-output", type=Path)
    real_sample_handoff.add_argument("--checkpoint-output", type=Path)
    real_sample_handoff.add_argument("--preview-output", type=Path)
    real_sample_handoff.add_argument("--require-pass", action="store_true")
    real_sample_handoff.set_defaults(handler=_training_real_sample_v2_handoff)

    real_sample_viewer_preview = training_subparsers.add_parser(
        "real-sample-v2-viewer-preview",
        help="project real-sample v2 checkpoint output onto a full Gaussian PLY for viewer debug",
    )
    real_sample_viewer_preview.add_argument("input", type=Path)
    real_sample_viewer_preview.add_argument("--preview-ply-output", required=True, type=Path)
    real_sample_viewer_preview.add_argument("--viewer-path")
    real_sample_viewer_preview.add_argument("--slots", type=int)
    real_sample_viewer_preview.add_argument("--frames", type=int, default=2)
    real_sample_viewer_preview.add_argument("--max-points", type=int, default=128)
    real_sample_viewer_preview.add_argument("--object-id-field", default="object_id")
    real_sample_viewer_preview.add_argument("--temporal-offset", type=float, default=0.01)
    real_sample_viewer_preview.add_argument("--image-width", type=int, default=12)
    real_sample_viewer_preview.add_argument("--image-height", type=int, default=12)
    real_sample_viewer_preview.add_argument("--point-radius", type=int, default=1)
    real_sample_viewer_preview.add_argument(
        "--visibility-policy",
        choices=("covered_pixels", "all_pixels"),
        default="covered_pixels",
    )
    real_sample_viewer_preview.add_argument("--iterations", type=int, default=100)
    real_sample_viewer_preview.add_argument("--learning-rate", type=float, default=0.4)
    real_sample_viewer_preview.add_argument(
        "--temperature-candidates",
        type=float,
        nargs="+",
        help="temperature sweep candidates; defaults to 1.0 0.75 0.5 0.35 0.25",
    )
    real_sample_viewer_preview.add_argument("--baseline-temperature", type=float, default=1.0)
    real_sample_viewer_preview.add_argument("--image-renderer", choices=("point", "gsplat"), default="point")
    real_sample_viewer_preview.add_argument("--seed", type=int, default=4)
    real_sample_viewer_preview.add_argument("--vram-reserve-gb", type=int, default=1)
    real_sample_viewer_preview.add_argument(
        "--assignment-feature-weight",
        type=float,
        default=REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT,
        help="full-cloud assignment feature cost weight; defaults to promoted weak-boundary value 2.0",
    )
    real_sample_viewer_preview.add_argument(
        "--assignment-position-weight",
        type=float,
        default=REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT,
        help="full-cloud assignment position cost weight; defaults to promoted weak-boundary value 1.0",
    )
    real_sample_viewer_preview.add_argument("--rewrite-sh", action="store_true")
    real_sample_viewer_preview.add_argument("--ascii", action="store_true", help="write ASCII PLY")
    real_sample_viewer_preview.add_argument("--summary-output", type=Path)
    real_sample_viewer_preview.add_argument("--require-pass", action="store_true")
    real_sample_viewer_preview.set_defaults(handler=_training_real_sample_v2_viewer_preview)

    real_sample_full_cloud_purity = training_subparsers.add_parser(
        "real-sample-v2-full-cloud-purity",
        help="scan real-sample v2 target coverage and export the best full-cloud viewer PLY",
    )
    real_sample_full_cloud_purity.add_argument("input", type=Path)
    real_sample_full_cloud_purity.add_argument("--preview-ply-output", required=True, type=Path)
    real_sample_full_cloud_purity.add_argument("--viewer-path")
    real_sample_full_cloud_purity.add_argument(
        "--max-point-candidates",
        type=int,
        nargs="+",
        default=[24, 64, 128],
        help="sample coverage candidates; defaults to 24 64 128",
    )
    real_sample_full_cloud_purity.add_argument("--slots", type=int)
    real_sample_full_cloud_purity.add_argument("--frames", type=int, default=2)
    real_sample_full_cloud_purity.add_argument("--object-id-field", default="object_id")
    real_sample_full_cloud_purity.add_argument("--temporal-offset", type=float, default=0.01)
    real_sample_full_cloud_purity.add_argument("--image-width", type=int, default=12)
    real_sample_full_cloud_purity.add_argument("--image-height", type=int, default=12)
    real_sample_full_cloud_purity.add_argument("--point-radius", type=int, default=1)
    real_sample_full_cloud_purity.add_argument(
        "--visibility-policy",
        choices=("covered_pixels", "all_pixels"),
        default="covered_pixels",
    )
    real_sample_full_cloud_purity.add_argument("--iterations", type=int, default=100)
    real_sample_full_cloud_purity.add_argument("--learning-rate", type=float, default=0.4)
    real_sample_full_cloud_purity.add_argument(
        "--temperature-candidates",
        type=float,
        nargs="+",
        help="temperature sweep candidates; defaults to 1.0 0.75 0.5 0.35 0.25",
    )
    real_sample_full_cloud_purity.add_argument("--baseline-temperature", type=float, default=1.0)
    real_sample_full_cloud_purity.add_argument("--image-renderer", choices=("point", "gsplat"), default="point")
    real_sample_full_cloud_purity.add_argument("--seed", type=int, default=4)
    real_sample_full_cloud_purity.add_argument("--vram-reserve-gb", type=int, default=1)
    real_sample_full_cloud_purity.add_argument("--rewrite-sh", action="store_true")
    real_sample_full_cloud_purity.add_argument("--ascii", action="store_true", help="write ASCII PLY")
    real_sample_full_cloud_purity.add_argument("--summary-output", type=Path)
    real_sample_full_cloud_purity.add_argument("--require-pass", action="store_true")
    real_sample_full_cloud_purity.set_defaults(handler=_training_real_sample_v2_full_cloud_purity)

    real_sample_segmentation_quality = training_subparsers.add_parser(
        "real-sample-v2-segmentation-quality",
        help="inspect 128-target real-sample v2 object segmentation confusion and uncertainty",
    )
    real_sample_segmentation_quality.add_argument("input", type=Path)
    real_sample_segmentation_quality.add_argument("--preview-ply-output", required=True, type=Path)
    real_sample_segmentation_quality.add_argument("--viewer-path")
    real_sample_segmentation_quality.add_argument("--slots", type=int)
    real_sample_segmentation_quality.add_argument("--frames", type=int, default=2)
    real_sample_segmentation_quality.add_argument("--max-points", type=int, default=128)
    real_sample_segmentation_quality.add_argument("--object-id-field", default="object_id")
    real_sample_segmentation_quality.add_argument("--temporal-offset", type=float, default=0.01)
    real_sample_segmentation_quality.add_argument("--image-width", type=int, default=12)
    real_sample_segmentation_quality.add_argument("--image-height", type=int, default=12)
    real_sample_segmentation_quality.add_argument("--point-radius", type=int, default=1)
    real_sample_segmentation_quality.add_argument(
        "--visibility-policy",
        choices=("covered_pixels", "all_pixels"),
        default="covered_pixels",
    )
    real_sample_segmentation_quality.add_argument("--iterations", type=int, default=100)
    real_sample_segmentation_quality.add_argument("--learning-rate", type=float, default=0.4)
    real_sample_segmentation_quality.add_argument(
        "--temperature-candidates",
        type=float,
        nargs="+",
        help="temperature sweep candidates; defaults to 1.0 0.75 0.5 0.35 0.25",
    )
    real_sample_segmentation_quality.add_argument("--baseline-temperature", type=float, default=1.0)
    real_sample_segmentation_quality.add_argument("--image-renderer", choices=("point", "gsplat"), default="point")
    real_sample_segmentation_quality.add_argument("--seed", type=int, default=4)
    real_sample_segmentation_quality.add_argument("--vram-reserve-gb", type=int, default=1)
    real_sample_segmentation_quality.add_argument("--rewrite-sh", action="store_true")
    real_sample_segmentation_quality.add_argument("--ascii", action="store_true", help="write ASCII PLY")
    real_sample_segmentation_quality.add_argument("--summary-output", type=Path)
    real_sample_segmentation_quality.add_argument("--require-pass", action="store_true")
    real_sample_segmentation_quality.set_defaults(handler=_training_real_sample_v2_segmentation_quality)

    real_sample_weak_boundary_opt = training_subparsers.add_parser(
        "real-sample-v2-weak-boundary-opt",
        help="try fixed-target cost-weight normalization for the real-sample v2 weak boundary",
    )
    real_sample_weak_boundary_opt.add_argument("input", type=Path)
    real_sample_weak_boundary_opt.add_argument("--preview-ply-output", required=True, type=Path)
    real_sample_weak_boundary_opt.add_argument("--viewer-path")
    real_sample_weak_boundary_opt.add_argument("--slots", type=int)
    real_sample_weak_boundary_opt.add_argument("--frames", type=int, default=2)
    real_sample_weak_boundary_opt.add_argument("--max-points", type=int, default=128)
    real_sample_weak_boundary_opt.add_argument("--solver-temperature", type=float, default=0.35)
    real_sample_weak_boundary_opt.add_argument("--candidate-feature-weight", type=float, default=2.0)
    real_sample_weak_boundary_opt.add_argument("--candidate-position-weight", type=float, default=1.0)
    real_sample_weak_boundary_opt.add_argument("--object-id-field", default="object_id")
    real_sample_weak_boundary_opt.add_argument("--temporal-offset", type=float, default=0.01)
    real_sample_weak_boundary_opt.add_argument("--image-width", type=int, default=12)
    real_sample_weak_boundary_opt.add_argument("--image-height", type=int, default=12)
    real_sample_weak_boundary_opt.add_argument("--point-radius", type=int, default=1)
    real_sample_weak_boundary_opt.add_argument(
        "--visibility-policy",
        choices=("covered_pixels", "all_pixels"),
        default="covered_pixels",
    )
    real_sample_weak_boundary_opt.add_argument("--iterations", type=int, default=100)
    real_sample_weak_boundary_opt.add_argument("--learning-rate", type=float, default=0.4)
    real_sample_weak_boundary_opt.add_argument("--baseline-temperature", type=float, default=1.0)
    real_sample_weak_boundary_opt.add_argument("--image-renderer", choices=("point", "gsplat"), default="point")
    real_sample_weak_boundary_opt.add_argument("--seed", type=int, default=4)
    real_sample_weak_boundary_opt.add_argument("--vram-reserve-gb", type=int, default=1)
    real_sample_weak_boundary_opt.add_argument("--rewrite-sh", action="store_true")
    real_sample_weak_boundary_opt.add_argument("--ascii", action="store_true", help="write ASCII PLY")
    real_sample_weak_boundary_opt.add_argument("--summary-output", type=Path)
    real_sample_weak_boundary_opt.add_argument("--require-pass", action="store_true")
    real_sample_weak_boundary_opt.set_defaults(handler=_training_real_sample_v2_weak_boundary_opt)

    real_sample_promoted_weights_cross_sample = training_subparsers.add_parser(
        "real-sample-v2-promoted-weights-cross-sample",
        help="compare promoted assignment weights against baseline weights on a second real sample",
    )
    real_sample_promoted_weights_cross_sample.add_argument("input", type=Path)
    real_sample_promoted_weights_cross_sample.add_argument(
        "--preview-ply-output",
        required=True,
        type=Path,
    )
    real_sample_promoted_weights_cross_sample.add_argument("--viewer-path")
    real_sample_promoted_weights_cross_sample.add_argument("--slots", type=int)
    real_sample_promoted_weights_cross_sample.add_argument("--frames", type=int, default=2)
    real_sample_promoted_weights_cross_sample.add_argument("--max-points", type=int, default=128)
    real_sample_promoted_weights_cross_sample.add_argument(
        "--solver-temperature",
        type=float,
        default=0.35,
    )
    real_sample_promoted_weights_cross_sample.add_argument(
        "--baseline-feature-weight",
        type=float,
        default=1.0,
    )
    real_sample_promoted_weights_cross_sample.add_argument(
        "--baseline-position-weight",
        type=float,
        default=1.0,
    )
    real_sample_promoted_weights_cross_sample.add_argument(
        "--promoted-feature-weight",
        type=float,
        default=REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT,
    )
    real_sample_promoted_weights_cross_sample.add_argument(
        "--promoted-position-weight",
        type=float,
        default=REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT,
    )
    real_sample_promoted_weights_cross_sample.add_argument(
        "--reference-sample",
        default="public/samples/lego_alpha_v1_objects.ply",
        help="sample where the promoted weights were first selected",
    )
    real_sample_promoted_weights_cross_sample.add_argument("--object-id-field", default="object_id")
    real_sample_promoted_weights_cross_sample.add_argument(
        "--temporal-offset",
        type=float,
        default=0.01,
    )
    real_sample_promoted_weights_cross_sample.add_argument("--image-width", type=int, default=12)
    real_sample_promoted_weights_cross_sample.add_argument("--image-height", type=int, default=12)
    real_sample_promoted_weights_cross_sample.add_argument("--point-radius", type=int, default=1)
    real_sample_promoted_weights_cross_sample.add_argument(
        "--visibility-policy",
        choices=("covered_pixels", "all_pixels"),
        default="covered_pixels",
    )
    real_sample_promoted_weights_cross_sample.add_argument("--iterations", type=int, default=100)
    real_sample_promoted_weights_cross_sample.add_argument("--learning-rate", type=float, default=0.4)
    real_sample_promoted_weights_cross_sample.add_argument(
        "--baseline-temperature",
        type=float,
        default=1.0,
    )
    real_sample_promoted_weights_cross_sample.add_argument(
        "--image-renderer",
        choices=("point", "gsplat"),
        default="point",
    )
    real_sample_promoted_weights_cross_sample.add_argument("--seed", type=int, default=4)
    real_sample_promoted_weights_cross_sample.add_argument("--vram-reserve-gb", type=int, default=1)
    real_sample_promoted_weights_cross_sample.add_argument("--rewrite-sh", action="store_true")
    real_sample_promoted_weights_cross_sample.add_argument(
        "--ascii",
        action="store_true",
        help="write ASCII PLY",
    )
    real_sample_promoted_weights_cross_sample.add_argument("--summary-output", type=Path)
    real_sample_promoted_weights_cross_sample.add_argument("--require-pass", action="store_true")
    real_sample_promoted_weights_cross_sample.set_defaults(
        handler=_training_real_sample_v2_promoted_weights_cross_sample
    )

    real_sample_sample_aware_weight_policy = training_subparsers.add_parser(
        "real-sample-v2-sample-aware-weight-policy",
        help="select baseline or promoted assignment weights per real sample using hard-boundary gates",
    )
    real_sample_sample_aware_weight_policy.add_argument("input", type=Path)
    real_sample_sample_aware_weight_policy.add_argument(
        "--preview-ply-output",
        required=True,
        type=Path,
    )
    real_sample_sample_aware_weight_policy.add_argument("--viewer-path")
    real_sample_sample_aware_weight_policy.add_argument("--slots", type=int)
    real_sample_sample_aware_weight_policy.add_argument("--frames", type=int, default=2)
    real_sample_sample_aware_weight_policy.add_argument("--max-points", type=int, default=128)
    real_sample_sample_aware_weight_policy.add_argument(
        "--solver-temperature",
        type=float,
        default=0.35,
    )
    real_sample_sample_aware_weight_policy.add_argument(
        "--baseline-feature-weight",
        type=float,
        default=1.0,
    )
    real_sample_sample_aware_weight_policy.add_argument(
        "--baseline-position-weight",
        type=float,
        default=1.0,
    )
    real_sample_sample_aware_weight_policy.add_argument(
        "--promoted-feature-weight",
        type=float,
        default=REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT,
    )
    real_sample_sample_aware_weight_policy.add_argument(
        "--promoted-position-weight",
        type=float,
        default=REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT,
    )
    real_sample_sample_aware_weight_policy.add_argument("--object-id-field", default="object_id")
    real_sample_sample_aware_weight_policy.add_argument(
        "--temporal-offset",
        type=float,
        default=0.01,
    )
    real_sample_sample_aware_weight_policy.add_argument("--image-width", type=int, default=12)
    real_sample_sample_aware_weight_policy.add_argument("--image-height", type=int, default=12)
    real_sample_sample_aware_weight_policy.add_argument("--point-radius", type=int, default=1)
    real_sample_sample_aware_weight_policy.add_argument(
        "--visibility-policy",
        choices=("covered_pixels", "all_pixels"),
        default="covered_pixels",
    )
    real_sample_sample_aware_weight_policy.add_argument("--iterations", type=int, default=100)
    real_sample_sample_aware_weight_policy.add_argument("--learning-rate", type=float, default=0.4)
    real_sample_sample_aware_weight_policy.add_argument(
        "--baseline-temperature",
        type=float,
        default=1.0,
    )
    real_sample_sample_aware_weight_policy.add_argument(
        "--image-renderer",
        choices=("point", "gsplat"),
        default="point",
    )
    real_sample_sample_aware_weight_policy.add_argument("--seed", type=int, default=4)
    real_sample_sample_aware_weight_policy.add_argument("--vram-reserve-gb", type=int, default=1)
    real_sample_sample_aware_weight_policy.add_argument("--rewrite-sh", action="store_true")
    real_sample_sample_aware_weight_policy.add_argument(
        "--ascii",
        action="store_true",
        help="write ASCII PLY",
    )
    real_sample_sample_aware_weight_policy.add_argument("--summary-output", type=Path)
    real_sample_sample_aware_weight_policy.add_argument("--require-pass", action="store_true")
    real_sample_sample_aware_weight_policy.set_defaults(
        handler=_training_real_sample_v2_sample_aware_weight_policy
    )

    bounded_normalization_cross_sample = training_subparsers.add_parser(
        "real-sample-v2-bounded-normalization-cross-sample",
        help="summarize sample-aware bounded normalization policy across real samples",
    )
    bounded_normalization_cross_sample.add_argument("inputs", nargs="+", type=Path)
    bounded_normalization_cross_sample.add_argument(
        "--sample-id",
        dest="sample_ids",
        action="append",
        help="sample id, repeated once per input when provided",
    )
    bounded_normalization_cross_sample.add_argument(
        "--viewer-path",
        dest="viewer_paths",
        action="append",
        help="viewer path, repeated once per input when provided",
    )
    bounded_normalization_cross_sample.add_argument("--min-samples", type=int, default=2)
    bounded_normalization_cross_sample.add_argument("--slots", type=int)
    bounded_normalization_cross_sample.add_argument("--frames", type=int, default=2)
    bounded_normalization_cross_sample.add_argument("--max-points", type=int, default=128)
    bounded_normalization_cross_sample.add_argument(
        "--solver-temperature",
        type=float,
        default=0.35,
    )
    bounded_normalization_cross_sample.add_argument(
        "--baseline-feature-weight",
        type=float,
        default=1.0,
    )
    bounded_normalization_cross_sample.add_argument(
        "--baseline-position-weight",
        type=float,
        default=1.0,
    )
    bounded_normalization_cross_sample.add_argument(
        "--promoted-feature-weight",
        type=float,
        default=REAL_SAMPLE_V2_PROMOTED_FEATURE_WEIGHT,
    )
    bounded_normalization_cross_sample.add_argument(
        "--promoted-position-weight",
        type=float,
        default=REAL_SAMPLE_V2_PROMOTED_POSITION_WEIGHT,
    )
    bounded_normalization_cross_sample.add_argument("--object-id-field", default="object_id")
    bounded_normalization_cross_sample.add_argument(
        "--temporal-offset",
        type=float,
        default=0.01,
    )
    bounded_normalization_cross_sample.add_argument("--image-width", type=int, default=12)
    bounded_normalization_cross_sample.add_argument("--image-height", type=int, default=12)
    bounded_normalization_cross_sample.add_argument("--point-radius", type=int, default=1)
    bounded_normalization_cross_sample.add_argument(
        "--visibility-policy",
        choices=("covered_pixels", "all_pixels"),
        default="covered_pixels",
    )
    bounded_normalization_cross_sample.add_argument("--iterations", type=int, default=100)
    bounded_normalization_cross_sample.add_argument("--learning-rate", type=float, default=0.4)
    bounded_normalization_cross_sample.add_argument(
        "--baseline-temperature",
        type=float,
        default=1.0,
    )
    bounded_normalization_cross_sample.add_argument(
        "--image-renderer",
        choices=("point", "gsplat"),
        default="point",
    )
    bounded_normalization_cross_sample.add_argument("--seed", type=int, default=4)
    bounded_normalization_cross_sample.add_argument("--vram-reserve-gb", type=int, default=1)
    bounded_normalization_cross_sample.add_argument("--rewrite-sh", action="store_true")
    bounded_normalization_cross_sample.add_argument("--summary-output", type=Path)
    bounded_normalization_cross_sample.add_argument("--require-pass", action="store_true")
    bounded_normalization_cross_sample.set_defaults(
        handler=_training_real_sample_v2_bounded_normalization_cross_sample
    )

    eval_assignment = training_subparsers.add_parser(
        "eval-assignment",
        help="evaluate assignment stability from an Object Emergence Solver checkpoint",
    )
    eval_assignment.add_argument("input", type=Path)
    eval_assignment.add_argument("--checkpoint", required=True, type=Path)
    eval_assignment.add_argument("--slots", type=int)
    eval_assignment.add_argument("--frames", type=int, default=2)
    eval_assignment.add_argument("--max-points", type=int)
    eval_assignment.add_argument("--object-id-field", default="object_id")
    eval_assignment.add_argument("--temporal-offset", type=float, default=0.01)
    eval_assignment.add_argument("--seed", type=int, default=0)
    eval_assignment.add_argument("--solver-temperature", type=float)
    eval_assignment.add_argument("--entropy-threshold", type=float, default=0.6)
    eval_assignment.add_argument("--purity-threshold", type=float, default=0.8)
    eval_assignment.add_argument("--collapse-mass-fraction", type=float, default=0.9)
    eval_assignment.add_argument("--assignment-confidence-floor", type=float, default=0.4)
    eval_assignment.add_argument("--id-stability-threshold", type=float, default=0.7)
    eval_assignment.add_argument("--temporal-drift-threshold", type=float)
    eval_assignment.add_argument("--summary-output", type=Path)
    eval_assignment.add_argument("--require-pass", action="store_true")
    eval_assignment.set_defaults(handler=_training_eval_assignment)

    object_emergence_solver = training_subparsers.add_parser(
        "object-emergence-solver",
        help="train the dependency-free Object Emergence Solver on object_id targets",
    )
    object_emergence_solver.add_argument("input", type=Path)
    object_emergence_solver.add_argument("--slots", type=int)
    object_emergence_solver.add_argument("--object-id-field", default="object_id")
    object_emergence_solver.add_argument("--max-points", type=int, default=64)
    object_emergence_solver.add_argument("--iterations", type=int, default=30)
    object_emergence_solver.add_argument("--learning-rate", type=float, default=0.35)
    object_emergence_solver.add_argument("--assignment-weight", type=float, default=1.0)
    object_emergence_solver.add_argument("--entropy-weight", type=float, default=0.01)
    object_emergence_solver.add_argument("--balance-weight", type=float, default=0.05)
    object_emergence_solver.add_argument("--temporal-weight", type=float, default=0.0)
    object_emergence_solver.add_argument("--finite-difference-epsilon", type=float, default=1e-3)
    object_emergence_solver.add_argument("--seed", type=int, default=0)
    object_emergence_solver.add_argument("--record-every", type=int)
    object_emergence_solver.add_argument("--summary-output", type=Path)
    object_emergence_solver.add_argument("--checkpoint-output", type=Path)
    object_emergence_solver.add_argument("--include-weights", action="store_true")
    object_emergence_solver.add_argument("--require-loss-decrease", action="store_true")
    object_emergence_solver.set_defaults(handler=_training_object_emergence_solver)

    renderer_loss_contract = training_subparsers.add_parser(
        "renderer-loss-contract",
        help="write the boundary between point-render smoke and real renderer loss",
    )
    renderer_loss_contract.add_argument("--kernel-summary", type=Path)
    renderer_loss_contract.add_argument("--output", "-o", type=Path)
    renderer_loss_contract.add_argument(
        "--target-renderer",
        default="differentiable-gaussian-image-renderer",
    )
    renderer_loss_contract.add_argument("--require-point-smoke-ready", action="store_true")
    renderer_loss_contract.set_defaults(handler=_training_renderer_loss_contract)

    write_bundle = training_subparsers.add_parser(
        "write-sample-bundle",
        help="write a traceable scene/object sample.json binding dataset, masks, and training output",
    )
    write_bundle.add_argument("--output", "-o", required=True, type=Path)
    write_bundle.add_argument("--sample-id", required=True)
    write_bundle.add_argument("--asset-id", required=True)
    write_bundle.add_argument("--dataset", required=True, type=Path)
    write_bundle.add_argument("--masks", required=True, type=Path)
    write_bundle.add_argument("--training-manifest", required=True, type=Path)
    write_bundle.add_argument("--split", default="train")
    write_bundle.set_defaults(handler=_training_write_sample_bundle)

    return parser


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--ascii", action="store_true", help="write ASCII PLY")


def _format_optional_float(value: object) -> str:
    return "-" if value is None else f"{float(value):.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
