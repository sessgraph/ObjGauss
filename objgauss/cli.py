from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from objgauss.assets import list_assets, pull_asset
from objgauss.clustering import cluster_features, summarize_labels
from objgauss.demo import build_v1_closure_demo, verify_v1_closure_demo
from objgauss.features import extract_features
from objgauss.goal_audit import audit_v1_goal
from objgauss.mask_voting import (
    train_object_field_from_votes,
    training_summary,
    vote_masks_to_gaussians,
)
from objgauss.lego_verify import verify_lego_alpha_closure_demo
from objgauss.masks import (
    build_nerf_alpha_mask_manifest,
    build_nerf_rgba_color_mask_manifest,
    build_nerf_sam_mask_manifest,
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
from objgauss.splat import read_splat
from objgauss.training import register_training_output


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
    labeled = attach_hard_labels(cloud, field, object_id_field=args.object_id_field)
    if args.colorize:
        labeled = apply_object_colors(
            labeled,
            object_id_field=args.object_id_field,
            rewrite_sh=args.rewrite_sh,
        )
    write_ply(args.output, labeled, fmt=_output_format(args))
    print(f"exported {cloud.count} gaussians from {args.field} to {args.output}")
    _print_summary(field.labels())


def _object_field_stats(args: argparse.Namespace) -> None:
    field = load_object_field(args.field)
    metrics = object_field_metrics(field)
    print(f"gaussians={field.gaussian_count}")
    print(f"slots={field.slots}")
    _print_metrics(metrics)
    _print_summary(field.labels())


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
    print(f"projected={votes.projected}")
    print(f"matched={votes.matched}")
    print(f"supervised_gaussians={result.supervised_gaussians}")
    print(f"initial_loss={result.initial_loss:.6f}")
    print(f"final_loss={result.final_loss:.6f}")
    _print_metrics(object_field_metrics(result.field))
    _print_summary(result.field.labels())

    if args.summary_output:
        write_json(args.summary_output, training_summary(result))
        print(f"summary={args.summary_output}")

    if args.ply_output:
        labeled = attach_hard_labels(cloud, result.field)
        if args.colorize:
            labeled = apply_object_colors(labeled, rewrite_sh=args.rewrite_sh)
        write_ply(args.ply_output, labeled, fmt=_output_format(args))
        print(f"ply={args.ply_output}")


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
    if result.initial_loss is not None and result.final_loss is not None:
        print(f"initial_loss={result.initial_loss:.6f}")
        print(f"final_loss={result.final_loss:.6f}")


def _print_summary(labels: np.ndarray) -> None:
    for label, count in summarize_labels(labels):
        print(f"object_id={label} count={count}")


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
    field_export.add_argument("--colorize", action="store_true")
    field_export.add_argument("--rewrite-sh", action="store_true")
    field_export.add_argument("--ascii", action="store_true", help="write ASCII PLY")
    field_export.set_defaults(handler=_object_field_export)

    field_stats = object_field_subparsers.add_parser(
        "stats",
        help="print object field metrics",
    )
    field_stats.add_argument("field", type=Path)
    field_stats.set_defaults(handler=_object_field_stats)

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
    field_vote.add_argument("--colorize", action="store_true")
    field_vote.add_argument("--rewrite-sh", action="store_true")
    field_vote.add_argument("--ascii", action="store_true", help="write ASCII PLY")
    field_vote.set_defaults(handler=_object_field_vote_masks)

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
    nerf_sam.add_argument("--points-per-side", type=int, default=32)
    nerf_sam.add_argument("--pred-iou-thresh", type=float, default=0.88)
    nerf_sam.add_argument("--stability-score-thresh", type=float, default=0.95)
    nerf_sam.set_defaults(handler=_masks_from_nerf_sam)

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
    register_output.add_argument("--no-colorize", action="store_true")
    register_output.set_defaults(handler=_training_register_output)

    return parser


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--ascii", action="store_true", help="write ASCII PLY")


if __name__ == "__main__":
    raise SystemExit(main())
