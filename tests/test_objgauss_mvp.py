from __future__ import annotations

import json
import zipfile

import numpy as np

import objgauss.assets as asset_module
from objgauss.assets import get_asset, list_assets
from objgauss.cli import main
from objgauss.clustering import cluster_features
from objgauss.features import extract_features
from objgauss.gaussians import GaussianCloud
from objgauss.mask_voting import train_object_field_from_votes, vote_masks_to_gaussians
from objgauss.object_field import (
    ObjectField,
    field_from_labels,
    load_object_field,
    object_field_metrics,
    save_object_field,
)
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
    demo = get_asset("polyhaven-school-chair-1k")
    training = get_asset("nerf-synthetic-lego")

    assert asset.download_url
    assert asset.local_path == "/samples/plush_objects.ply"
    assert asset.splat_path == "/samples/plush.splat"
    assert asset.pull_pipeline == "splat-to-objgauss-ply"
    assert asset.pipeline_stage == "Demo 可用"
    assert "Demo预览" in asset.use_cases
    assert asset in list_assets()
    assert demo.pull_pipeline == "polyhaven-gltf"
    assert demo.polyhaven_id == "SchoolChair_01"
    assert demo.license.startswith("CC0")
    assert training.pull_pipeline == "nerf-example-data"
    assert training.training_subdir == "nerf_synthetic/lego"


def test_assets_list_cli_reports_pullable_sample(capsys):
    assert main(["assets", "list", "--pullable"]) == 0

    output = capsys.readouterr().out
    assert "plush-3dgs-local" in output
    assert "polyhaven-school-chair-1k" in output
    assert "nerf-synthetic-lego" in output
    assert "Demo 可用" in output
    assert "pull" in output


def test_polyhaven_pull_writes_gltf_manifest(tmp_path, monkeypatch):
    def fake_fetch_json(_url):
        return {
            "gltf": {
                "1k": {
                    "gltf": {
                        "url": "https://example.invalid/SchoolChair_01_1k.gltf",
                        "size": 10,
                        "include": {
                            "SchoolChair_01.bin": {
                                "url": "https://example.invalid/SchoolChair_01.bin",
                                "size": 4,
                            },
                            "textures/SchoolChair_01_diff_1k.jpg": {
                                "url": "https://example.invalid/SchoolChair_01_diff_1k.jpg",
                                "size": 6,
                            },
                        },
                    }
                }
            }
        }

    def fake_download(_url, path, *, force):
        path.parent.mkdir(parents=True, exist_ok=True)
        if force or not path.exists():
            path.write_bytes(b"asset")

    monkeypatch.setattr(asset_module, "_fetch_json", fake_fetch_json)
    monkeypatch.setattr(asset_module, "_download", fake_download)

    result = asset_module.pull_asset(
        "polyhaven-school-chair-1k",
        raw_dir=tmp_path / "raw",
        converted_dir=tmp_path / "converted",
    )

    assert result.output_path and result.output_path.name == "SchoolChair_01_1k.gltf"
    assert result.manifest_path and result.manifest_path.exists()
    assert (result.raw_path / "SchoolChair_01.bin").exists()
    assert (result.raw_path / "textures" / "SchoolChair_01_diff_1k.jpg").exists()
    assert len(result.downloaded_files) == 3


def test_nerf_pull_extracts_training_subset(tmp_path, monkeypatch):
    def fake_download(_url, path, *, force):
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and not force:
            return
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("nerf_example_data/nerf_synthetic/lego/transforms_train.json", "{}")
            archive.writestr("nerf_example_data/nerf_synthetic/lego/train/r_0.png", b"png")
            archive.writestr("nerf_example_data/nerf_synthetic/chair/transforms_train.json", "{}")

    monkeypatch.setattr(asset_module, "_download", fake_download)

    result = asset_module.pull_asset(
        "nerf-synthetic-lego",
        raw_dir=tmp_path / "raw",
        converted_dir=tmp_path / "converted",
        training_dir=tmp_path / "training",
    )

    assert result.training_path and result.training_path.name == "nerf-synthetic-lego"
    assert (result.training_path / "transforms_train.json").exists()
    assert (result.training_path / "train" / "r_0.png").exists()
    assert not (tmp_path / "training" / "chair").exists()
    assert result.manifest_path and result.manifest_path.exists()


def test_object_field_from_labels_has_soft_slots():
    field = field_from_labels(np.array([0, 1, 1, 0], dtype=np.int32), slots=2, confidence=0.9)
    probabilities = field.probabilities()
    metrics = object_field_metrics(field)

    assert probabilities.shape == (4, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.array_equal(field.labels(), np.array([0, 1, 1, 0], dtype=np.int32))
    assert 0.0 < metrics.entropy < np.log(2.0)
    assert metrics.active_slots == 2


def test_object_field_init_export_and_stats_cli(tmp_path, capsys):
    input_path = tmp_path / "gaussians.ply"
    field_path = tmp_path / "object_field"
    initialized_path = tmp_path / "initialized_objects.ply"
    exported_path = tmp_path / "exported_objects.ply"
    write_ply(input_path, _synthetic_cloud(), fmt="ascii")

    assert (
        main(
            [
                "object-field",
                "init",
                str(input_path),
                "--output",
                str(field_path),
                "--slots",
                "2",
                "--ply-output",
                str(initialized_path),
                "--colorize",
                "--smoothness",
                "--ascii",
            ]
        )
        == 0
    )

    init_output = capsys.readouterr().out
    field = load_object_field(field_path)
    initialized = read_ply(initialized_path)
    assert field.gaussian_count == 4
    assert field.slots == 2
    assert "entropy=" in init_output
    assert "smoothness=" in init_output
    assert {"object_id", "red", "green", "blue"}.issubset(initialized.fields)

    assert (
        main(
            [
                "object-field",
                "export",
                str(input_path),
                "--field",
                str(field_path),
                "--output",
                str(exported_path),
                "--colorize",
                "--ascii",
            ]
        )
        == 0
    )
    exported = read_ply(exported_path)
    assert "object_id" in exported.fields

    assert main(["object-field", "stats", str(field_path)]) == 0
    stats_output = capsys.readouterr().out
    assert "gaussians=4" in stats_output
    assert "slots=2" in stats_output
    assert "active_slots=2" in stats_output


def test_object_field_inspects_nerf_dataset(tmp_path, capsys):
    dataset = tmp_path / "nerf-synthetic-lego"
    (dataset / "train").mkdir(parents=True)
    (dataset / "train" / "r_0.png").write_bytes(b"png")
    transform = np.eye(4, dtype=float).tolist()
    (dataset / "transforms_train.json").write_text(
        (
            "{"
            '"camera_angle_x": 0.7,'
            '"frames": [{"file_path": "./train/r_0", "transform_matrix": '
            f"{transform}"
            "}]"
            "}"
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "nerf-summary.json"

    assert main(["object-field", "inspect-nerf", str(dataset), "--output", str(manifest)]) == 0

    output = capsys.readouterr().out
    assert "frames=1" in output
    assert "missing_images=0" in output
    assert "invalid_transforms=0" in output
    assert manifest.exists()


def test_mask_voting_trains_object_field_from_projected_rects(tmp_path):
    cloud = _camera_cloud()
    manifest = _write_rect_mask_manifest(tmp_path / "masks.json")
    field = ObjectField(np.zeros((cloud.count, 2), dtype=np.float32))

    votes = vote_masks_to_gaussians(cloud, manifest, slots=2)
    result = train_object_field_from_votes(field, votes, iterations=200, learning_rate=1.0)

    assert votes.frames == 1
    assert votes.projected == 4
    assert votes.supervised_gaussians == 4
    assert result.final_loss < result.initial_loss
    assert np.array_equal(result.field.labels(), np.array([0, 0, 1, 1], dtype=np.int32))


def test_object_field_vote_masks_cli_exports_summary_and_ply(tmp_path, capsys):
    cloud = _camera_cloud()
    input_path = tmp_path / "camera_cloud.ply"
    field_path = tmp_path / "field"
    output_field = tmp_path / "field_trained"
    output_ply = tmp_path / "field_trained.ply"
    summary_path = tmp_path / "summary.json"
    masks_path = _write_rect_mask_manifest(tmp_path / "masks.json")
    write_ply(input_path, cloud, fmt="ascii")
    save_object_field(field_path, ObjectField(np.zeros((cloud.count, 2), dtype=np.float32)))

    assert (
        main(
            [
                "object-field",
                "vote-masks",
                str(input_path),
                "--field",
                str(field_path),
                "--masks",
                str(masks_path),
                "--output",
                str(output_field),
                "--summary-output",
                str(summary_path),
                "--ply-output",
                str(output_ply),
                "--iterations",
                "200",
                "--learning-rate",
                "1.0",
                "--colorize",
                "--ascii",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    trained = load_object_field(output_field)
    exported = read_ply(output_ply)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "final_loss=" in output
    assert trained.labels().tolist() == [0, 0, 1, 1]
    assert {"object_id", "red", "green", "blue"}.issubset(exported.fields)
    assert summary["final_loss"] < summary["initial_loss"]
    assert summary["supervised_gaussians"] == 4


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


def _camera_cloud() -> GaussianCloud:
    vertices = np.zeros(
        4,
        dtype=np.dtype(
            [
                ("x", "f4"),
                ("y", "f4"),
                ("z", "f4"),
                ("opacity", "f4"),
                ("red", "u1"),
                ("green", "u1"),
                ("blue", "u1"),
            ]
        ),
    )
    vertices["x"] = np.array([-0.6, -0.3, 0.3, 0.6], dtype=np.float32)
    vertices["y"] = 0.0
    vertices["z"] = -2.0
    vertices["opacity"] = 1.0
    vertices["red"] = 128
    vertices["green"] = 128
    vertices["blue"] = 128
    return GaussianCloud(vertices=vertices, source_format="ascii")


def _write_rect_mask_manifest(path):
    payload = {
        "width": 100,
        "height": 100,
        "camera_angle_x": float(np.pi / 2.0),
        "frames": [
            {
                "transform_matrix": np.eye(4, dtype=float).tolist(),
                "masks": [
                    {"slot": 0, "label": "left", "rect": [0, 0, 50, 100]},
                    {"slot": 1, "label": "right", "rect": [50, 0, 100, 100]},
                ],
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path
