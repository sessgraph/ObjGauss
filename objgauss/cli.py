from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
)
from objgauss.core.object_emergence_solver import (
    evidence_from_gaussian_cloud,
    object_id_targets_from_cloud,
    train_object_emergence_solver,
)
from objgauss.core.renderer_loss import renderer_loss_boundary_report
from objgauss.core.training_renderer import evaluate_training_renderer_loss
from objgauss.core.gsplat_training_renderer import evaluate_gsplat_training_renderer_loss
from objgauss.core.trainable_artifact import write_trainable_kernel_model_artifact
from objgauss.core.trainable_quality import write_trainable_quality_report
from objgauss.core.object_state_benchmark import write_object_state_stability_benchmark
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
        "gpu_policy": {
            "uses_gpu": False,
            "full_renderer_training": "suspended_until_torch_gsplat_cuda_available",
            "vram_reserve_gb": 1,
        },
    }
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
    print("gpu_used=false")
    print("vram_reserve_gb=1")
    if args.summary_output:
        write_json(args.summary_output, summary)
        print(f"summary={args.summary_output}")
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
