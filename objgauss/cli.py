from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from objgauss.assets import list_assets, pull_asset
from objgauss.clustering import cluster_features, summarize_labels
from objgauss.features import extract_features
from objgauss.ply import read_ply, write_ply
from objgauss.segment import (
    apply_object_colors,
    assign_object_ids,
    filter_objects,
    parse_object_ids,
)
from objgauss.splat import read_splat


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
        clusters=args.clusters,
        force=args.force,
    )
    print(f"asset={result.asset.id} name={result.asset.name}")
    print(f"raw={result.raw_path}")
    print(f"converted={result.converted_path}")
    print(f"viewer_sample={result.output_path}")
    if result.raw_public_path:
        print(f"viewer_splat={result.raw_public_path}")
    print(f"gaussians={result.gaussian_count}")
    for label, count in result.object_counts:
        print(f"object_id={label} count={count}")


def _print_summary(labels: np.ndarray) -> None:
    for label, count in summarize_labels(labels):
        print(f"object_id={label} count={count}")


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
    assets_pull.add_argument("--clusters", type=int)
    assets_pull.add_argument("--force", action="store_true")
    assets_pull.set_defaults(handler=_assets_pull)

    return parser


def _add_io_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", "-o", required=True, type=Path)
    parser.add_argument("--ascii", action="store_true", help="write ASCII PLY")


if __name__ == "__main__":
    raise SystemExit(main())
