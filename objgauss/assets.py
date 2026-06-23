from __future__ import annotations

import json
import math
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path
from shutil import copyfile, copyfileobj
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
    polyhaven_id: str | None = None
    resolution: str | None = None
    training_subdir: str | None = None


@dataclass(frozen=True)
class PulledAsset:
    asset: AssetSource
    raw_path: Path
    converted_path: Path | None = None
    output_path: Path | None = None
    raw_public_path: Path | None = None
    training_path: Path | None = None
    manifest_path: Path | None = None
    gaussian_count: int | None = None
    object_counts: tuple[tuple[int, int], ...] = ()
    downloaded_files: tuple[Path, ...] = ()


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
        id="polyhaven-school-chair-1k",
        name="Poly Haven School Chair 1K",
        category="可商用展示",
        source_type="mesh",
        status="已接入",
        priority="P0",
        source_url="https://polyhaven.com/a/SchoolChair_01",
        download_url="https://api.polyhaven.com/files/SchoolChair_01",
        license="CC0；API 仅用于非商用/研究拉取，需带 User-Agent",
        formats=(".gltf", ".bin", ".jpg", "CC0"),
        best_for="许可干净的单对象家具 Demo 输入，后续用于 mesh 多视角渲染和 3DGS 训练。",
        raw_file_name="polyhaven-school-chair-1k",
        output_file_name="asset-manifest.json",
        pull_pipeline="polyhaven-gltf",
        pipeline_stage="Demo 素材已自动化",
        use_cases=("展示Demo", "可商用样例", "mesh转3DGS"),
        polyhaven_id="SchoolChair_01",
        resolution="1k",
    ),
    AssetSource(
        id="nerf-synthetic-lego",
        name="NeRF Synthetic Lego 示例训练集",
        category="3DGS 训练集",
        source_type="images",
        status="已接入",
        priority="P0",
        source_url="https://github.com/bmild/nerf",
        download_url=(
            "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/"
            "ECCV20/nerf/nerf_example_data.zip"
        ),
        license="NeRF 官方示例数据；仅作为训练/研究素材使用",
        formats=("images", "transforms_train.json", "transforms_test.json"),
        best_for="ObjGauss v1 Object Field 的多视角训练烟测和跨视角一致性验证。",
        raw_file_name="nerf_example_data.zip",
        output_file_name="training-manifest.json",
        pull_pipeline="nerf-example-data",
        pipeline_stage="训练源已自动化",
        use_cases=("3DGS训练", "ObjectField烟测"),
        training_subdir="nerf_synthetic/lego",
    ),
    AssetSource(
        id="nerf-llff-fern",
        name="NeRF LLFF Fern 示例训练集",
        category="3DGS 训练集",
        source_type="images",
        status="已接入",
        priority="P0",
        source_url="https://github.com/bmild/nerf",
        download_url=(
            "http://cseweb.ucsd.edu/~viscomp/projects/LF/papers/"
            "ECCV20/nerf/nerf_example_data.zip"
        ),
        license="NeRF 官方示例数据；仅作为训练/研究素材使用",
        formats=("images", "COLMAP", "transforms_train.json"),
        best_for="第二个真实多视角/COLMAP Splatfacto 场景，用于跨场景 benchmark。",
        raw_file_name="nerf_example_data.zip",
        output_file_name="training-manifest.json",
        pull_pipeline="nerf-example-data",
        pipeline_stage="训练源已自动化",
        use_cases=("3DGS训练", "跨场景benchmark"),
        training_subdir="nerf_llff_data/fern",
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
    training_dir: str | Path = "outputs/assets/training",
    clusters: int | None = None,
    force: bool = False,
) -> PulledAsset:
    asset = get_asset(asset_id)
    if asset.pull_pipeline == "splat-to-objgauss-ply":
        return _pull_splat_asset(
            asset,
            raw_dir=raw_dir,
            converted_dir=converted_dir,
            public_dir=public_dir,
            clusters=clusters,
            force=force,
        )
    if asset.pull_pipeline == "polyhaven-gltf":
        return _pull_polyhaven_gltf(
            asset,
            raw_dir=raw_dir,
            converted_dir=converted_dir,
            force=force,
        )
    if asset.pull_pipeline == "nerf-example-data":
        return _pull_nerf_example_data(
            asset,
            raw_dir=raw_dir,
            training_dir=training_dir,
            converted_dir=converted_dir,
            force=force,
        )
    raise ValueError(f"{asset.name} is not automated yet; download from {asset.source_url}")


def _pull_splat_asset(
    asset: AssetSource,
    *,
    raw_dir: str | Path,
    converted_dir: str | Path,
    public_dir: str | Path,
    clusters: int | None,
    force: bool,
) -> PulledAsset:
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


def _pull_polyhaven_gltf(
    asset: AssetSource,
    *,
    raw_dir: str | Path,
    converted_dir: str | Path,
    force: bool,
) -> PulledAsset:
    if not asset.download_url or not asset.raw_file_name or not asset.output_file_name:
        raise ValueError(f"{asset.name} is missing pull metadata")
    if not asset.polyhaven_id or not asset.resolution:
        raise ValueError(f"{asset.name} is missing Poly Haven metadata")

    files = _fetch_json(asset.download_url)
    gltf_record = files["gltf"][asset.resolution]["gltf"]
    root = Path(raw_dir) / asset.raw_file_name
    downloaded = [_download_record(gltf_record, root, force=force)]
    entrypoint = downloaded[0]
    for relative_path, record in gltf_record.get("include", {}).items():
        downloaded.append(_download_record(record, root / relative_path, force=force))

    manifest_path = Path(converted_dir) / asset.id / asset.output_file_name
    manifest = {
        "asset_id": asset.id,
        "source": asset.source_url,
        "license": asset.license,
        "pipeline": asset.pull_pipeline,
        "polyhaven_id": asset.polyhaven_id,
        "resolution": asset.resolution,
        "root": str(root),
        "entrypoint": str(entrypoint),
        "files": [
            {
                "path": str(path),
                "size": path.stat().st_size if path.exists() else None,
            }
            for path in downloaded
        ],
    }
    _write_json(manifest_path, manifest)
    return PulledAsset(
        asset=asset,
        raw_path=root,
        converted_path=manifest_path,
        output_path=entrypoint,
        raw_public_path=None,
        manifest_path=manifest_path,
        object_counts=(),
        downloaded_files=tuple(downloaded),
    )


def _pull_nerf_example_data(
    asset: AssetSource,
    *,
    raw_dir: str | Path,
    training_dir: str | Path,
    converted_dir: str | Path,
    force: bool,
) -> PulledAsset:
    if not asset.download_url or not asset.raw_file_name or not asset.training_subdir:
        raise ValueError(f"{asset.name} is missing training pull metadata")

    raw_path = Path(raw_dir) / asset.raw_file_name
    _download(asset.download_url, raw_path, force=force)

    output_path = Path(training_dir) / asset.id
    extracted = _extract_zip_prefix(raw_path, asset.training_subdir, output_path, force=force)
    generated = _maybe_write_colmap_nerf_transforms(output_path, force=force)
    all_files = tuple(sorted({*extracted, *generated}))
    manifest_path = Path(converted_dir) / asset.id / (asset.output_file_name or "manifest.json")
    manifest = {
        "asset_id": asset.id,
        "source": asset.source_url,
        "license": asset.license,
        "pipeline": asset.pull_pipeline,
        "zip_path": str(raw_path),
        "training_path": str(output_path),
        "source_subdir": asset.training_subdir,
        "files": sorted(str(path.relative_to(output_path)) for path in all_files),
    }
    _write_json(manifest_path, manifest)
    return PulledAsset(
        asset=asset,
        raw_path=raw_path,
        converted_path=manifest_path,
        output_path=output_path,
        raw_public_path=None,
        training_path=output_path,
        manifest_path=manifest_path,
        object_counts=(),
        downloaded_files=all_files,
    )


def _download(url: str, path: Path, *, force: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not force:
        return
    request = Request(url, headers={"User-Agent": "ObjGauss/0.1 asset pipeline"})
    with urlopen(request, timeout=300) as response, path.open("wb") as output:
        copyfileobj(response, output)


def _download_record(record: dict[str, object], path: Path, *, force: bool) -> Path:
    url = record.get("url")
    if not isinstance(url, str):
        raise ValueError("download record is missing url")
    if path.suffix == "":
        path = path / _download_file_name(url)
    _download(url, path, force=force)
    return path


def _download_file_name(url: str) -> str:
    file_name = Path(urlparse(url).path).name
    if not file_name:
        raise ValueError(f"download url is missing a file name: {url}")
    return file_name


def _fetch_json(url: str) -> dict[str, object]:
    request = Request(
        url,
        headers={"User-Agent": "ObjGauss/0.1 ASSET-001 research pipeline"},
    )
    with urlopen(request, timeout=60) as response:
        return json.load(response)


def _extract_zip_prefix(
    zip_path: Path,
    prefix: str,
    output_path: Path,
    *,
    force: bool,
) -> tuple[Path, ...]:
    extracted: list[Path] = []
    prefix_parts = tuple(Path(prefix).parts)
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            member_path = Path(member.filename)
            prefix_index = _find_parts(member_path.parts, prefix_parts)
            if member.is_dir() or prefix_index is None:
                continue
            relative_parts = member_path.parts[prefix_index + len(prefix_parts) :]
            if not relative_parts or any(part in {"", ".", ".."} for part in relative_parts):
                continue
            target = output_path.joinpath(*relative_parts)
            if target.exists() and not force:
                extracted.append(target)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(member) as source, target.open("wb") as destination:
                copyfileobj(source, destination)
            extracted.append(target)
    if not extracted:
        raise ValueError(f"no files under {prefix!r} in {zip_path}")
    return tuple(extracted)


def _find_parts(parts: tuple[str, ...], needle: tuple[str, ...]) -> int | None:
    if not needle:
        return None
    for index in range(0, len(parts) - len(needle) + 1):
        if parts[index : index + len(needle)] == needle:
            return index
    return None


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _maybe_write_colmap_nerf_transforms(output_path: Path, *, force: bool) -> tuple[Path, ...]:
    sparse_dir = output_path / "sparse" / "0"
    cameras_path = sparse_dir / "cameras.bin"
    images_path = sparse_dir / "images.bin"
    transforms_path = output_path / "transforms_train.json"
    if not cameras_path.exists() or not images_path.exists():
        return ()
    if transforms_path.exists() and not force:
        return (transforms_path,)

    cameras = _read_colmap_cameras(cameras_path)
    images = _read_colmap_images(images_path)
    if not cameras:
        raise ValueError(f"no COLMAP cameras in {cameras_path}")
    if not images:
        raise ValueError(f"no COLMAP images in {images_path}")

    frames = []
    first_camera = cameras[images[0]["camera_id"]]
    intrinsics = _colmap_intrinsics(first_camera)
    for image in sorted(images, key=lambda item: str(item["name"])):
        image_path = output_path / "images" / str(image["name"])
        if not image_path.exists():
            continue
        c2w = _colmap_image_c2w(image)
        frames.append(
            {
                "file_path": str(image_path.relative_to(output_path)),
                "transform_matrix": c2w.tolist(),
                "colmap_image_id": int(image["image_id"]),
            }
        )

    if not frames:
        raise ValueError(f"COLMAP images did not match files under {output_path / 'images'}")

    camera_angle_x = 2.0 * math.atan(float(first_camera["width"]) / (2.0 * intrinsics["fl_x"]))
    camera_angle_y = 2.0 * math.atan(float(first_camera["height"]) / (2.0 * intrinsics["fl_y"]))
    payload = {
        "camera_model": str(first_camera["model"]),
        "camera_angle_x": camera_angle_x,
        "camera_angle_y": camera_angle_y,
        "fl_x": intrinsics["fl_x"],
        "fl_y": intrinsics["fl_y"],
        "cx": intrinsics["cx"],
        "cy": intrinsics["cy"],
        "w": int(first_camera["width"]),
        "h": int(first_camera["height"]),
        "source_type": "colmap-to-nerf-transforms",
        "frames": frames,
    }
    _write_json(transforms_path, payload)
    return (transforms_path,)


_COLMAP_CAMERA_MODELS: dict[int, tuple[str, int]] = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


def _read_colmap_cameras(path: Path) -> dict[int, dict[str, object]]:
    data = path.read_bytes()
    offset = 0
    (count,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    cameras: dict[int, dict[str, object]] = {}
    for _ in range(count):
        camera_id, model_id, width, height = struct.unpack_from("<iiQQ", data, offset)
        offset += struct.calcsize("<iiQQ")
        if model_id not in _COLMAP_CAMERA_MODELS:
            raise ValueError(f"unsupported COLMAP camera model id {model_id} in {path}")
        model, param_count = _COLMAP_CAMERA_MODELS[model_id]
        params = struct.unpack_from(f"<{param_count}d", data, offset)
        offset += struct.calcsize(f"<{param_count}d")
        cameras[int(camera_id)] = {
            "camera_id": int(camera_id),
            "model": model,
            "width": int(width),
            "height": int(height),
            "params": tuple(float(value) for value in params),
        }
    return cameras


def _read_colmap_images(path: Path) -> list[dict[str, object]]:
    data = path.read_bytes()
    offset = 0
    (count,) = struct.unpack_from("<Q", data, offset)
    offset += 8
    images: list[dict[str, object]] = []
    for _ in range(count):
        image_id = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        qvec = struct.unpack_from("<4d", data, offset)
        offset += struct.calcsize("<4d")
        tvec = struct.unpack_from("<3d", data, offset)
        offset += struct.calcsize("<3d")
        camera_id = struct.unpack_from("<i", data, offset)[0]
        offset += 4
        name, offset = _read_colmap_string(data, offset)
        points_count = struct.unpack_from("<Q", data, offset)[0]
        offset += 8 + int(points_count) * struct.calcsize("<ddq")
        images.append(
            {
                "image_id": int(image_id),
                "qvec": tuple(float(value) for value in qvec),
                "tvec": tuple(float(value) for value in tvec),
                "camera_id": int(camera_id),
                "name": name,
            }
        )
    return images


def _read_colmap_string(data: bytes, offset: int) -> tuple[str, int]:
    end = data.find(b"\x00", offset)
    if end < 0:
        raise ValueError("unterminated COLMAP image name")
    return data[offset:end].decode("utf-8"), end + 1


def _colmap_intrinsics(camera: dict[str, object]) -> dict[str, float]:
    params = tuple(float(value) for value in camera["params"])  # type: ignore[index]
    model = str(camera["model"])
    width = float(camera["width"])
    height = float(camera["height"])
    if model in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL_FISHEYE"}:
        fl_x = fl_y = params[0]
        cx = params[1]
        cy = params[2]
    else:
        fl_x = params[0]
        fl_y = params[1]
        cx = params[2]
        cy = params[3]
    return {
        "fl_x": float(fl_x),
        "fl_y": float(fl_y),
        "cx": float(cx if cx else width * 0.5),
        "cy": float(cy if cy else height * 0.5),
    }


def _colmap_image_c2w(image: dict[str, object]) -> "np.ndarray":
    import numpy as np

    qvec = np.asarray(image["qvec"], dtype=np.float64)
    tvec = np.asarray(image["tvec"], dtype=np.float64)
    rotation = _qvec_to_rotmat(qvec)
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ tvec
    opengl_from_colmap_camera = np.diag([1.0, -1.0, -1.0, 1.0])
    return c2w @ opengl_from_colmap_camera


def _qvec_to_rotmat(qvec: "np.ndarray") -> "np.ndarray":
    import numpy as np

    qw, qx, qy, qz = qvec
    return np.asarray(
        [
            [
                1.0 - 2.0 * qy * qy - 2.0 * qz * qz,
                2.0 * qx * qy - 2.0 * qw * qz,
                2.0 * qz * qx + 2.0 * qw * qy,
            ],
            [
                2.0 * qx * qy + 2.0 * qw * qz,
                1.0 - 2.0 * qx * qx - 2.0 * qz * qz,
                2.0 * qy * qz - 2.0 * qw * qx,
            ],
            [
                2.0 * qz * qx - 2.0 * qw * qy,
                2.0 * qy * qz + 2.0 * qw * qx,
                1.0 - 2.0 * qx * qx - 2.0 * qy * qy,
            ],
        ],
        dtype=np.float64,
    )
