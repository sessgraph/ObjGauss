from __future__ import annotations

import numpy as np

from objgauss.assets import get_asset, list_assets
from objgauss.cli import main
from objgauss.clustering import cluster_features
from objgauss.features import extract_features
from objgauss.gaussians import GaussianCloud
from objgauss.ply import read_ply, write_ply
from objgauss.segment import apply_object_colors, assign_object_ids, filter_objects
from objgauss.splat import read_splat


def test_cluster_colorize_and_filter_roundtrip(tmp_path):
    cloud = _synthetic_cloud()
    features = extract_features(cloud)
    result = cluster_features(features, clusters=2, seed=4, max_iter=50)

    labeled = assign_object_ids(cloud, result.labels)
    colored = apply_object_colors(labeled)
    assert {"object_id", "red", "green", "blue"}.issubset(colored.fields)

    output = tmp_path / "objects.ply"
    write_ply(output, colored, fmt="ascii")
    loaded = read_ply(output)

    assert loaded.count == cloud.count
    assert "object_id" in loaded.fields
    assert set(np.unique(loaded.vertices["object_id"])) == {0, 1}

    removed = filter_objects(loaded, {0}, mode="remove")
    assert 0 < removed.count < loaded.count
    assert set(np.unique(removed.vertices["object_id"])) == {1}


def test_rewrite_sh_object_colors():
    cloud = _synthetic_cloud(include_rgb=False, include_sh=True)
    labels = np.array([0, 0, 1, 1], dtype=np.int32)
    labeled = assign_object_ids(cloud, labels)
    colored = apply_object_colors(labeled, rewrite_sh=True)

    assert colored.vertices["f_dc_0"][0] != cloud.vertices["f_dc_0"][0]
    assert colored.vertices["f_dc_1"][2] != cloud.vertices["f_dc_1"][2]


def test_read_splat_as_gaussian_cloud(tmp_path):
    path = tmp_path / "tiny.splat"
    row = np.zeros(32, dtype=np.uint8)
    row[:24] = np.array([1.0, 2.0, 3.0, 0.1, 0.2, 0.3], dtype="<f4").view(np.uint8)
    row[24:28] = np.array([10, 20, 30, 128], dtype=np.uint8)
    row[28:32] = np.array([1, 2, 3, 4], dtype=np.uint8)
    path.write_bytes(row.tobytes())

    cloud = read_splat(path)

    assert cloud.count == 1
    assert cloud.vertices["x"][0] == 1.0
    assert cloud.vertices["red"][0] == 10
    assert np.isclose(cloud.vertices["opacity"][0], 128 / 255.0)


def test_asset_registry_has_pullable_sample():
    asset = get_asset("plush-3dgs-local")

    assert asset.download_url
    assert asset.local_path == "/samples/plush_objects.ply"
    assert asset.pull_pipeline == "splat-to-objgauss-ply"
    assert asset.pipeline_stage == "Demo 可用"
    assert "Demo预览" in asset.use_cases
    assert asset in list_assets()


def test_assets_list_cli_reports_pullable_sample(capsys):
    assert main(["assets", "list", "--pullable"]) == 0

    output = capsys.readouterr().out
    assert "plush-3dgs-local" in output
    assert "Demo 可用" in output
    assert "pull" in output


def _synthetic_cloud(*, include_rgb: bool = True, include_sh: bool = False) -> GaussianCloud:
    fields: list[tuple[str, str]] = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("opacity", "f4"),
    ]
    if include_rgb:
        fields.extend([("red", "u1"), ("green", "u1"), ("blue", "u1")])
    if include_sh:
        fields.extend([("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4")])

    vertices = np.zeros(4, dtype=np.dtype(fields))
    vertices["x"] = np.array([0.0, 0.1, 5.0, 5.2], dtype=np.float32)
    vertices["y"] = np.array([0.0, 0.1, 5.0, 5.1], dtype=np.float32)
    vertices["z"] = np.array([0.0, 0.0, 0.1, 0.1], dtype=np.float32)
    vertices["opacity"] = np.array([4.0, 4.0, 4.0, 4.0], dtype=np.float32)
    if include_rgb:
        vertices["red"] = np.array([220, 230, 20, 30], dtype=np.uint8)
        vertices["green"] = np.array([20, 30, 220, 230], dtype=np.uint8)
        vertices["blue"] = np.array([20, 25, 40, 35], dtype=np.uint8)
    if include_sh:
        vertices["f_dc_0"] = 0.0
        vertices["f_dc_1"] = 0.0
        vertices["f_dc_2"] = 0.0
    return GaussianCloud(vertices=vertices, source_format="ascii")
