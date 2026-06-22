from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile
from urllib.request import urlretrieve

from objgauss.clustering import cluster_features, summarize_labels
from objgauss.features import extract_features
from objgauss.ply import write_ply
from objgauss.segment import assign_object_ids
from objgauss.splat import read_splat


@dataclass(frozen=True)
class AssetSource:
    id: str
    name: str
    category: str
    source_type: str
    status: str
    priority: str
    source_url: str
    license: str
    formats: tuple[str, ...]
    best_for: str
    local_path: str | None = None
    splat_path: str | None = None
    download_url: str | None = None
    raw_file_name: str | None = None
    output_file_name: str | None = None
    default_clusters: int | None = None
    pull_pipeline: str | None = None
    pipeline_stage: str = "来源登记"
    use_cases: tuple[str, ...] = ()


@dataclass(frozen=True)
class PulledAsset:
    asset: AssetSource
    raw_path: Path
    converted_path: Path
    output_path: Path
    raw_public_path: Path | None
    gaussian_count: int
    object_counts: tuple[tuple[int, int], ...]


ASSETS: tuple[AssetSource, ...] = (
    AssetSource(
        id="plush-3dgs-local",
        name="Plush 3DGS 示例",
        category="本地样例",
        source_type="gaussian",
        status="已接入",
        priority="P0",
        source_url="https://huggingface.co/cakewalk/splat-data/blob/main/plush.splat",
        download_url="https://huggingface.co/cakewalk/splat-data/resolve/main/plush.splat",
        license="来源许可混合，仅用于本地测试",
        formats=(".splat", ".ply", "object_id"),
        best_for="快速验证高斯点云加载、对象聚类色、删除/隔离预览。",
        local_path="/samples/plush_objects.ply",
        splat_path="/samples/plush.splat",
        raw_file_name="plush.splat",
        output_file_name="plush_objects.ply",
        default_clusters=6,
        pull_pipeline="splat-to-objgauss-ply",
        pipeline_stage="Demo 可用",
        use_cases=("Demo预览", "管线烟测"),
    ),
    AssetSource(
        id="arkitscenes",
        name="ARKitScenes",
        category="真实室内场景",
        source_type="scan",
        status="候选",
        priority="P0",
        source_url="https://github.com/apple/ARKitScenes",
        license="按 Apple 数据许可执行",
        formats=("RGB-D", "pose", "mesh", "3D bbox"),
        best_for="贴近手机 LiDAR 扫描输入，适合做房间和家具级对象化实验。",
        pipeline_stage="训练源",
        use_cases=("3DGS训练", "室内Demo"),
    ),
    AssetSource(
        id="scannet",
        name="ScanNet",
        category="真实室内场景",
        source_type="scan",
        status="候选",
        priority="P1",
        source_url="https://www.scan-net.org/",
        license="研究数据许可，需申请",
        formats=("RGB-D", "pose", "mesh", "semantic"),
        best_for="有语义/实例标注，适合验证场景到对象分组的准确性。",
        pipeline_stage="训练源",
        use_cases=("分割评估", "训练验证"),
    ),
    AssetSource(
        id="omniobject3d",
        name="OmniObject3D",
        category="对象级扫描",
        source_type="mesh",
        status="候选",
        priority="P0",
        source_url="https://omniobject3d.github.io/",
        license="按数据集条款执行",
        formats=("mesh", "point cloud", "multi-view"),
        best_for="高质量真实扫描对象，适合做单物体高斯化和对象级编辑。",
        pipeline_stage="训练源",
        use_cases=("对象训练", "对象Demo"),
    ),
    AssetSource(
        id="google-scanned-objects",
        name="Google Scanned Objects",
        category="对象级扫描",
        source_type="mesh",
        status="候选",
        priority="P1",
        source_url=(
            "https://research.google/blog/"
            "scanned-objects-by-google-research-a-dataset-of-3d-scanned-common-household-items/"
        ),
        license="CC-BY 4.0",
        formats=("OBJ", "SDF", "texture", "collider"),
        best_for="日用品和机器人仿真，适合后续 collider proxy / 物理代理。",
        pipeline_stage="训练源",
        use_cases=("物理代理", "对象Demo"),
    ),
    AssetSource(
        id="poly-haven",
        name="Poly Haven",
        category="可商用展示",
        source_type="mesh",
        status="候选",
        priority="P0",
        source_url="https://polyhaven.com/models",
        license="CC0",
        formats=("blend", "glTF", "textures"),
        best_for="快速搭展示 demo，许可干净，适合开源项目里的可复现素材。",
        pipeline_stage="Demo 素材",
        use_cases=("展示Demo", "可商用样例"),
    ),
    AssetSource(
        id="mipnerf360",
        name="Mip-NeRF 360",
        category="3DGS 训练集",
        source_type="images",
        status="候选",
        priority="P1",
        source_url="https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/",
        license="按原数据集条款执行",
        formats=("images", "COLMAP", "poses"),
        best_for="训练 3DGS 和评估新视角渲染质量。",
        pipeline_stage="训练源",
        use_cases=("3DGS训练", "渲染评估"),
    ),
    AssetSource(
        id="tanks-and-temples",
        name="Tanks and Temples",
        category="3DGS 训练集",
        source_type="images",
        status="候选",
        priority="P2",
        source_url="https://www.tanksandtemples.org/",
        license="按 benchmark 条款执行",
        formats=("images", "video", "ground truth"),
        best_for="复杂真实场景重建 benchmark，适合后期质量评估。",
        pipeline_stage="训练源",
        use_cases=("重建评估", "论文验证"),
    ),
)


def list_assets() -> tuple[AssetSource, ...]:
    return ASSETS


def get_asset(asset_id: str) -> AssetSource:
    for asset in ASSETS:
        if asset.id == asset_id:
            return asset
    valid = ", ".join(asset.id for asset in ASSETS)
    raise ValueError(f"unknown asset {asset_id!r}; valid assets: {valid}")


def pull_asset(
    asset_id: str,
    *,
    raw_dir: str | Path = "outputs/assets/raw",
    converted_dir: str | Path = "outputs/assets/converted",
    public_dir: str | Path = "public",
    clusters: int | None = None,
    force: bool = False,
) -> PulledAsset:
    asset = get_asset(asset_id)
    if asset.pull_pipeline != "splat-to-objgauss-ply":
        raise ValueError(
            f"{asset.name} is not automated yet; download from {asset.source_url}"
        )
    if not asset.download_url or not asset.raw_file_name or not asset.local_path:
        raise ValueError(f"{asset.name} is missing pull metadata")

    raw_path = Path(raw_dir) / asset.raw_file_name
    converted_path = Path(converted_dir) / raw_path.with_suffix(".ply").name
    output_path = Path(public_dir) / asset.local_path.lstrip("/")
    raw_public_path = (
        Path(public_dir) / asset.splat_path.lstrip("/") if asset.splat_path else None
    )
    k = clusters if clusters is not None else asset.default_clusters
    if k is None:
        raise ValueError(f"{asset.name} does not define a cluster count")

    _download(asset.download_url, raw_path, force=force)
    if raw_public_path and (force or not raw_public_path.exists()):
        raw_public_path.parent.mkdir(parents=True, exist_ok=True)
        copyfile(raw_path, raw_public_path)

    cloud = read_splat(raw_path)
    if force or not converted_path.exists():
        write_ply(converted_path, cloud, fmt="binary_little_endian")

    features = extract_features(cloud)
    result = cluster_features(
        features,
        clusters=k,
        seed=0,
        max_iter=100,
    )
    labeled = assign_object_ids(cloud, result.labels)
    write_ply(output_path, labeled, fmt="binary_little_endian")

    return PulledAsset(
        asset=asset,
        raw_path=raw_path,
        converted_path=converted_path,
        output_path=output_path,
        raw_public_path=raw_public_path,
        gaussian_count=labeled.count,
        object_counts=tuple(summarize_labels(result.labels)),
    )


def _download(url: str, path: Path, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    urlretrieve(url, path)
