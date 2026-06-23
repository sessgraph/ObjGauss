from __future__ import annotations

import json
import struct
import zipfile
import zlib

import numpy as np
import pytest

import objgauss.assets as asset_module
from objgauss.assets import get_asset, list_assets
from objgauss.cli import main
from objgauss.clustering import cluster_features
from objgauss.emergence import object_emergence_curve, object_emergence_metrics
from objgauss.features import extract_features
from objgauss.gaussians import GaussianCloud
from objgauss.goal_audit import audit_v1_goal
from objgauss.mask_voting import train_object_field_from_votes, vote_masks_to_gaussians
from objgauss.masks import build_nerf_sam_mask_manifest
from objgauss.object_field import (
    ObjectField,
    field_from_labels,
    load_object_field,
    object_field_label_delta,
    object_field_metrics,
    save_object_field,
)
from objgauss.ply import read_ply, write_ply
from objgauss.render_probe import load_render_probe_frames
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


def test_object_field_label_delta_counts_mask_guidance_changes():
    initial = ObjectField(np.zeros((4, 2), dtype=np.float32))
    trained = field_from_labels(np.array([0, 1, 1, 0], dtype=np.int32), slots=2)

    delta = object_field_label_delta(initial, trained)

    assert delta.changed_gaussians == 2
    assert delta.changed_fraction == 0.5
    assert delta.initial_active_slots == 1
    assert delta.trained_active_slots == 2
    assert delta.as_dict()["changed_gaussians"] == 2


def test_object_emergence_metrics_track_assignment_spatial_and_stability():
    field = field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=0.99)
    permuted = field_from_labels(np.array([1, 1, 0, 0], dtype=np.int32), slots=2, confidence=0.99)
    positions = np.array(
        [[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [10.0, 0.0, 0.0], [10.1, 0.0, 0.0]],
        dtype=np.float32,
    )

    metrics = object_emergence_metrics(field, positions_xyz=positions, reference=permuted)

    assert metrics["assignment"]["assignment_confidence"] > 0.9
    assert np.isclose(metrics["assignment"]["effective_slots"], 2.0, atol=0.02)
    assert metrics["spatial"]["compactness_score"] > 0.99
    assert metrics["stability"]["adjusted_rand_index"] == 1.0
    assert metrics["stability"]["matched_label_agreement"] == 1.0
    assert metrics["object_emergence_score"]["score"] > 0.9
    assert metrics["object_emergence_score"]["complete"] is False
    assert "occlusion_effect" in metrics["object_emergence_score"]["missing_components"]


def test_object_emergence_curve_tracks_training_phase_metrics(tmp_path):
    cloud = _camera_cloud()
    masks_path = _write_rect_mask_manifest(tmp_path / "masks.json")
    votes = vote_masks_to_gaussians(cloud, masks_path, slots=2)
    field = ObjectField(np.zeros((cloud.count, 2), dtype=np.float32))
    positions = np.stack([cloud.vertices["x"], cloud.vertices["y"], cloud.vertices["z"]], axis=1)

    curve = object_emergence_curve(
        field,
        votes,
        positions_xyz=positions,
        cloud=cloud,
        render_frames=load_render_probe_frames(masks_path, max_size=32),
        iterations=20,
        learning_rate=1.0,
        eval_every=5,
    )

    points = curve["points"]
    assert [point["step"] for point in points] == [0, 5, 10, 15, 20]
    assert points[-1]["projection_loss"] < points[0]["projection_loss"]
    assert points[-1]["assignment_confidence"] > points[0]["assignment_confidence"]
    assert points[-1]["ari_to_initial"] == 0.0
    assert points[-1]["ari_to_previous"] == 1.0
    assert points[-1]["spatial_compactness_score"] > 0.9
    assert points[-1]["mask_proxy_occlusion_delta"]["mean_delta_loss"] > 0.0
    assert points[-1]["render_occlusion_delta"]["mean_delta_l1"] > 0.0
    assert points[-1]["object_emergence_score"]["components"]["occlusion_effect"] > 0.0
    assert curve["occlusion_delta_kind"] == "point_splat_render_l1"


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


def test_object_field_emergence_cli_outputs_partial_oes(tmp_path, capsys):
    input_path = tmp_path / "gaussians.ply"
    field_path = tmp_path / "object_field.npz"
    reference_path = tmp_path / "reference_field.npz"
    summary_path = tmp_path / "emergence.json"
    write_ply(input_path, _synthetic_cloud(), fmt="ascii")
    save_object_field(
        field_path,
        field_from_labels(np.array([0, 0, 1, 1], dtype=np.int32), slots=2, confidence=0.99),
    )
    save_object_field(
        reference_path,
        field_from_labels(np.array([1, 1, 0, 0], dtype=np.int32), slots=2, confidence=0.99),
    )

    assert (
        main(
            [
                "object-field",
                "emergence",
                str(field_path),
                "--cloud",
                str(input_path),
                "--reference",
                str(reference_path),
                "--output",
                str(summary_path),
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "object_emergence_score=" in output
    assert "object_emergence_complete=false" in output
    assert "missing_components=occlusion_effect" in output
    assert summary["stability"]["adjusted_rand_index"] == 1.0
    assert summary["object_emergence_score"]["complete"] is False


def test_object_field_emergence_curve_cli_writes_json_and_csv(tmp_path, capsys):
    cloud = _camera_cloud()
    input_path = tmp_path / "camera_cloud.ply"
    field_path = tmp_path / "field.npz"
    masks_path = _write_rect_mask_manifest(tmp_path / "masks.json")
    output_path = tmp_path / "curve.json"
    csv_path = tmp_path / "curve.csv"
    write_ply(input_path, cloud, fmt="ascii")
    save_object_field(field_path, ObjectField(np.zeros((cloud.count, 2), dtype=np.float32)))

    assert (
        main(
            [
                "object-field",
                "emergence-curve",
                str(input_path),
                "--field",
                str(field_path),
                "--masks",
                str(masks_path),
                "--output",
                str(output_path),
                "--csv-output",
                str(csv_path),
                "--iterations",
                "20",
                "--learning-rate",
                "1.0",
                "--eval-every",
                "10",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    curve = json.loads(output_path.read_text(encoding="utf-8"))
    csv_text = csv_path.read_text(encoding="utf-8")

    assert "final_projection_loss=" in output
    assert "final_mask_proxy_occlusion_mean_delta_loss=" in output
    assert "final_render_occlusion_mean_delta_l1=" in output
    assert [point["step"] for point in curve["points"]] == [0, 10, 20]
    assert curve["points"][-1]["projection_loss"] < curve["points"][0]["projection_loss"]
    assert curve["points"][-1]["mask_proxy_occlusion_delta"]["mean_delta_loss"] > 0.0
    assert curve["points"][-1]["render_occlusion_delta"]["mean_delta_l1"] > 0.0
    assert "mask_proxy_occlusion_mean_delta_loss" in csv_text
    assert "render_occlusion_mean_delta_l1" in csv_text


def test_object_field_emergence_report_cli_writes_html(tmp_path, capsys):
    cloud = _camera_cloud()
    input_path = tmp_path / "camera_cloud.ply"
    field_path = tmp_path / "field.npz"
    masks_path = _write_rect_mask_manifest(tmp_path / "masks.json")
    curve_path = tmp_path / "curve.json"
    report_path = tmp_path / "report.html"
    write_ply(input_path, cloud, fmt="ascii")
    save_object_field(field_path, ObjectField(np.zeros((cloud.count, 2), dtype=np.float32)))

    assert (
        main(
            [
                "object-field",
                "emergence-curve",
                str(input_path),
                "--field",
                str(field_path),
                "--masks",
                str(masks_path),
                "--output",
                str(curve_path),
                "--iterations",
                "10",
                "--learning-rate",
                "1.0",
                "--eval-every",
                "10",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "object-field",
                "emergence-report",
                str(curve_path),
                str(curve_path),
                "--label",
                "scene-a",
                "--label",
                "scene-b",
                "--output",
                str(report_path),
                "--title",
                "Test Benchmark",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    html_text = report_path.read_text(encoding="utf-8")
    assert "report=" in output
    assert "curves=2" in output
    assert "charts=" in output
    assert "scene-a" in html_text
    assert "scene-b" in html_text
    assert "Projection loss" in html_text
    assert "Render occlusion effect" in html_text
    assert "<svg" in html_text


def test_object_field_emergence_benchmark_cli_runs_manifest_suite(tmp_path, capsys):
    cloud = _camera_cloud()
    input_path = tmp_path / "camera_cloud.ply"
    field_path = tmp_path / "field.npz"
    masks_path = _write_rect_mask_manifest(tmp_path / "masks.json")
    manifest_path = tmp_path / "benchmark.json"
    output_dir = tmp_path / "benchmark-output"
    write_ply(input_path, cloud, fmt="ascii")
    save_object_field(field_path, ObjectField(np.zeros((cloud.count, 2), dtype=np.float32)))
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "object_emergence_benchmark",
                "title": "Test Benchmark",
                "root": ".",
                "defaults": {
                    "iterations": 10,
                    "learning_rate": 1.0,
                    "eval_every": 10,
                    "render_size": 32,
                },
                "thresholds": {
                    "require_projection_loss_decrease": True,
                    "min_points": 2,
                    "min_render_occlusion_effect_score": 0.001,
                },
                "scenes": [
                    {
                        "id": "camera-scene",
                        "label": "Camera scene",
                        "input": input_path.name,
                        "field": field_path.name,
                        "masks": masks_path.name,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                "object-field",
                "emergence-benchmark",
                str(manifest_path),
                "--output-dir",
                str(output_dir),
                "--strict",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    summary_path = output_dir / "summary.json"
    report_path = output_dir / "report.html"
    curve_path = output_dir / "camera-scene" / "curve.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert "passed=true" in output
    assert "scene=camera-scene passed=true" in output
    assert summary["passed"] is True
    assert summary["scenes"][0]["passed"] is True
    assert summary["scenes"][0]["checks"]
    assert report_path.exists()
    assert curve_path.exists()


def test_object_field_emergence_benchmark_reports_prepare_steps_for_missing_inputs(tmp_path, capsys):
    field_path = tmp_path / "field.npz"
    masks_path = _write_rect_mask_manifest(tmp_path / "masks.json")
    manifest_path = tmp_path / "benchmark.json"
    save_object_field(field_path, ObjectField(np.zeros((4, 2), dtype=np.float32)))
    manifest_path.write_text(
        json.dumps(
            {
                "kind": "object_emergence_benchmark",
                "root": ".",
                "help": "docs/benchmarks/semantic-smoke.md",
                "scenes": [
                    {
                        "id": "missing-scene",
                        "input": "missing.ply",
                        "field": field_path.name,
                        "masks": masks_path.name,
                        "prepare": ["uv run objgauss demo missing-scene"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "object-field",
                "emergence-benchmark",
                str(manifest_path),
                "--output-dir",
                str(tmp_path / "benchmark-output"),
            ]
        )
    assert exc.value.code == 2

    error = capsys.readouterr().err
    assert "scene missing-scene: missing input path:" in error
    assert "uv run objgauss demo missing-scene" in error
    assert "docs/benchmarks/semantic-smoke.md" in error


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


def test_masks_from_nerf_alpha_cli_writes_manifest_and_npy(tmp_path, capsys):
    dataset = tmp_path / "nerf-synthetic-lego"
    (dataset / "train").mkdir(parents=True)
    _write_rgba_png(
        dataset / "train" / "r_0.png",
        np.array(
            [
                [[10, 20, 30, 0], [40, 50, 60, 255]],
                [[70, 80, 90, 128], [100, 110, 120, 0]],
            ],
            dtype=np.uint8,
        ),
    )
    transform = np.eye(4, dtype=float).tolist()
    (dataset / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.7,
                "frames": [{"file_path": "./train/r_0", "transform_matrix": transform}],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "masks" / "mask-manifest.json"

    assert (
        main(
            [
                "masks",
                "from-nerf-alpha",
                str(dataset),
                "--output",
                str(manifest_path),
                "--threshold",
                "128",
                "--slot",
                "2",
                "--label",
                "lego",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    mask = np.load(manifest_path.parent / manifest["frames"][0]["masks"][0]["mask_path"])

    assert "foreground_pixels=2" in output
    assert manifest["source_type"] == "nerf-alpha"
    assert manifest["width"] == 2
    assert manifest["height"] == 2
    assert manifest["frames"][0]["image_path"] == "train/r_0.png"
    assert manifest["frames"][0]["masks"][0]["slot"] == 2
    assert manifest["frames"][0]["masks"][0]["label"] == "lego"
    assert mask.dtype == np.bool_
    assert mask.tolist() == [[False, True], [True, False]]


def test_masks_from_nerf_rgba_colors_cli_writes_multislot_manifest(tmp_path, capsys):
    dataset = tmp_path / "nerf-synthetic-lego"
    (dataset / "train").mkdir(parents=True)
    _write_rgba_png(
        dataset / "train" / "r_0.png",
        np.array(
            [
                [[220, 190, 20, 255], [210, 40, 30, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[230, 180, 25, 255], [30, 25, 20, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [160, 160, 150, 255], [150, 150, 140, 255]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [25, 25, 25, 255], [160, 160, 150, 255]],
            ],
            dtype=np.uint8,
        ),
    )
    transform = np.eye(4, dtype=float).tolist()
    (dataset / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.7,
                "frames": [{"file_path": "./train/r_0", "transform_matrix": transform}],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "masks" / "mask-manifest.json"

    assert (
        main(
            [
                "masks",
                "from-nerf-rgba-colors",
                str(dataset),
                "--output",
                str(manifest_path),
                "--alpha-threshold",
                "16",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    masks = manifest["frames"][0]["masks"]
    loaded_masks = {
        mask["label"]: np.load(manifest_path.parent / mask["mask_path"])
        for mask in masks
    }

    assert "masks=4" in output
    assert "slot=0 label=yellow pixels=2" in output
    assert "slot=1 label=red pixels=1" in output
    assert "slot=2 label=dark pixels=2" in output
    assert "slot=3 label=other pixels=3" in output
    assert manifest["source_type"] == "nerf-rgba-color-masks"
    assert manifest["width"] == 4
    assert manifest["height"] == 4
    assert [slot["label"] for slot in manifest["slots"]] == ["yellow", "red", "dark", "other"]
    assert {mask["slot"] for mask in masks} == {0, 1, 2, 3}
    assert loaded_masks["yellow"].dtype == np.bool_
    assert loaded_masks["yellow"].sum() == 2
    assert loaded_masks["red"].sum() == 1
    assert loaded_masks["dark"].sum() == 2
    assert loaded_masks["other"].sum() == 3


def test_masks_from_nerf_sam_writes_manifest_with_fake_generator(tmp_path):
    dataset = tmp_path / "nerf-synthetic-lego"
    (dataset / "train").mkdir(parents=True)
    _write_rgba_png(
        dataset / "train" / "r_0.png",
        np.array(
            [
                [[220, 190, 20, 255], [210, 40, 30, 255], [0, 0, 0, 255]],
                [[230, 180, 25, 255], [30, 25, 20, 255], [0, 0, 0, 255]],
                [[0, 0, 0, 255], [0, 0, 0, 255], [160, 160, 150, 255]],
            ],
            dtype=np.uint8,
        ),
    )
    transform = np.eye(4, dtype=float).tolist()
    (dataset / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": 0.7,
                "frames": [{"file_path": "./train/r_0", "transform_matrix": transform}],
            }
        ),
        encoding="utf-8",
    )
    manifest_path = tmp_path / "sam-masks" / "mask-manifest.json"

    result = build_nerf_sam_mask_manifest(
        dataset,
        output=manifest_path,
        checkpoint="not-used-by-fake-generator.pt",
        max_masks_per_frame=2,
        min_area=2,
        generator=_FakeSamGenerator(),
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    masks = manifest["frames"][0]["masks"]
    first = np.load(manifest_path.parent / masks[0]["mask_path"])
    second = np.load(manifest_path.parent / masks[1]["mask_path"])

    assert result.frames == 1
    assert result.masks == 2
    assert result.mask_pixels == 6
    assert manifest["source_type"] == "sam-automatic-mask-generator"
    assert manifest["sam"]["model_type"] == "vit_b"
    assert manifest["slots"] == [
        {"slot": 0, "label": "sam-area-rank-0"},
        {"slot": 1, "label": "sam-area-rank-1"},
    ]
    assert masks[0]["slot"] == 0
    assert masks[0]["label"] == "sam-area-rank-0"
    assert masks[0]["area"] == 4
    assert masks[0]["confidence"] == 0.91
    assert masks[0]["bbox"] == [0.0, 0.0, 2.0, 2.0]
    assert masks[1]["slot"] == 1
    assert masks[1]["area"] == 2
    assert first.dtype == np.bool_
    assert int(first.sum()) == 4
    assert int(second.sum()) == 2


def test_mask_voting_trains_object_field_from_projected_rects(tmp_path):
    cloud = _camera_cloud()
    manifest = _write_rect_mask_manifest(tmp_path / "masks.json")
    field = ObjectField(np.zeros((cloud.count, 2), dtype=np.float32))

    votes = vote_masks_to_gaussians(cloud, manifest, slots=2)
    result = train_object_field_from_votes(field, votes, iterations=200, learning_rate=1.0)
    vote_quality = result.vote_summary.as_dict()["vote_quality"]

    assert votes.frames == 1
    assert votes.projected == 4
    assert votes.supervised_gaussians == 4
    assert vote_quality["supervised_fraction"] == 1.0
    assert vote_quality["vote_conflict"]["gaussians"] == 0
    assert vote_quality["vote_conflict"]["normalized_target_entropy"] == 0.0
    assert [slot["winner_gaussians"] for slot in vote_quality["per_slot"]] == [2, 2]
    assert result.final_loss < result.initial_loss
    assert np.array_equal(result.field.labels(), np.array([0, 0, 1, 1], dtype=np.int32))


def test_mask_vote_quality_counts_conflicting_votes(tmp_path):
    payload = {
        "width": 100,
        "height": 100,
        "camera_angle_x": float(np.pi / 2.0),
        "frames": [
            {
                "transform_matrix": np.eye(4, dtype=float).tolist(),
                "masks": [
                    {"slot": 0, "label": "all-a", "rect": [0, 0, 100, 100]},
                    {"slot": 1, "label": "all-b", "rect": [0, 0, 100, 100]},
                ],
            }
        ],
    }
    manifest = tmp_path / "conflicting-masks.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    votes = vote_masks_to_gaussians(_camera_cloud(), manifest, slots=2)
    vote_quality = votes.as_dict()["vote_quality"]

    assert vote_quality["supervised_gaussians"] == 4
    assert vote_quality["vote_conflict"]["gaussians"] == 4
    assert vote_quality["vote_conflict"]["fraction"] == 1.0
    assert np.isclose(vote_quality["vote_conflict"]["normalized_target_entropy"], 1.0)
    assert vote_quality["target_confidence"]["mean"] == 0.5


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
    assert "supervised_fraction=1.000000" in output
    assert "vote_conflict_gaussians=0" in output
    assert trained.labels().tolist() == [0, 0, 1, 1]
    assert {"object_id", "red", "green", "blue"}.issubset(exported.fields)
    assert summary["final_loss"] < summary["initial_loss"]
    assert summary["supervised_gaussians"] == 4
    assert summary["vote_quality"]["supervised_fraction"] == 1.0
    assert summary["vote_quality"]["per_slot"][0]["winner_gaussians"] == 2


def test_training_register_output_ingests_external_gaussians_and_votes_masks(tmp_path, capsys):
    input_path = tmp_path / "external_trainer" / "point_cloud.ply"
    masks_path = _write_rect_mask_manifest(tmp_path / "masks.json")
    output_dir = tmp_path / "registered"
    public_dir = tmp_path / "public"
    write_ply(input_path, _camera_cloud(), fmt="ascii")

    assert (
        main(
            [
                "training",
                "register-output",
                str(input_path),
                "--asset-id",
                "nerf-lego-trained-output-local",
                "--output-dir",
                str(output_dir),
                "--dataset",
                str(tmp_path / "nerf-synthetic-lego"),
                "--masks",
                str(masks_path),
                "--public-dir",
                str(public_dir),
                "--public-name",
                "nerf_lego_trained",
                "--iterations",
                "120",
                "--learning-rate",
                "1.0",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    manifest = json.loads((output_dir / "training-output-manifest.json").read_text(encoding="utf-8"))
    registered = read_ply(output_dir / "gaussians.ply")
    splat = read_splat(output_dir / "gaussians.splat")
    object_ply = read_ply(output_dir / "object_aware_gaussians.ply")

    assert "manifest=" in output
    assert manifest["gaussian_source"] == "external_3dgs_training_output"
    assert manifest["input_format"] == "ply"
    assert manifest["asset_id"] == "nerf-lego-trained-output-local"
    assert manifest["acceptance"]["external_gaussian_loaded"] is True
    assert manifest["acceptance"]["viewer_splat_available"] is True
    assert manifest["acceptance"]["object_field_trained"] is True
    assert manifest["acceptance"]["mask_guidance_changed_object_field"] is True
    assert manifest["acceptance"]["mask_vote_quality_audit_available"] is True
    assert manifest["acceptance"]["projection_loss_decreased"] is True
    assert manifest["object_field_delta"]["changed_gaussians"] == 2
    assert manifest["training"]["final_loss"] < manifest["training"]["initial_loss"]
    assert manifest["training"]["vote_quality"]["per_slot"][0]["winner_gaussians"] == 2
    assert registered.count == 4
    assert splat.count == 4
    assert set(np.unique(object_ply.vertices["object_id"])) == {0, 1}
    assert (public_dir / "nerf_lego_trained.splat").exists()
    assert (public_dir / "nerf_lego_trained_objects.ply").exists()


def test_demo_v1_closure_builds_acceptance_artifacts(tmp_path, capsys):
    input_path = tmp_path / "objects.ply"
    splat_path = tmp_path / "scene.splat"
    output_dir = tmp_path / "demo"
    public_dir = tmp_path / "public"
    write_ply(input_path, _camera_cloud_with_object_ids(), fmt="ascii")
    splat_path.write_bytes(b"splat")

    assert (
        main(
            [
                "demo",
                "v1-closure",
                "--input",
                str(input_path),
                "--splat",
                str(splat_path),
                "--output-dir",
                str(output_dir),
                "--public-dir",
                str(public_dir),
                "--image-size",
                "96",
                "--iterations",
                "120",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    manifest = json.loads((output_dir / "v1-closure-manifest.json").read_text(encoding="utf-8"))
    exported = read_ply(output_dir / "plush_v1_objects.ply")

    assert "manifest=" in output
    assert manifest["acceptance"]["real_3dgs_scene_can_render"] is True
    assert manifest["acceptance"]["mask_guidance_changed_object_field"] is True
    assert manifest["acceptance"]["mask_vote_quality_audit_available"] is True
    assert manifest["acceptance"]["projection_loss_decreased"] is True
    assert manifest["object_field_delta"]["changed_gaussians"] == 2
    assert manifest["training"]["final_loss"] < manifest["training"]["initial_loss"]
    assert manifest["training"]["supervised_gaussians"] == 4
    assert manifest["training"]["vote_quality"]["supervised_fraction"] == 1.0
    assert (public_dir / "plush_v1_objects.ply").exists()
    assert set(np.unique(exported.vertices["object_id"])) == {0, 1}

    assert main(["demo", "verify-v1-closure", str(output_dir / "v1-closure-manifest.json")]) == 0
    verify_output = capsys.readouterr().out
    assert "passed=true" in verify_output
    assert "check=real_3dgs_scene status=pass" in verify_output
    assert "check=mask_vote_quality_audit_available status=pass" in verify_output
    assert "check=mask_guidance_changed_object_field status=pass" in verify_output
    assert "check=viewer_ply_exports_object_id status=pass" in verify_output


def test_demo_plush_semantic_closure_builds_real_splat_2d_mask_assets(tmp_path, capsys):
    input_path = tmp_path / "plush_raw.ply"
    splat_path = tmp_path / "scene.splat"
    output_dir = tmp_path / "demo"
    public_dir = tmp_path / "public"
    asset_library = tmp_path / "assetLibrary.js"
    _write_test_asset_library(asset_library)
    cloud = _camera_color_cloud()
    write_ply(input_path, cloud, fmt="ascii")
    splat_path.write_bytes(b"splat")

    assert (
        main(
            [
                "demo",
                "plush-semantic-closure",
                "--input",
                str(input_path),
                "--splat",
                str(splat_path),
                "--output-dir",
                str(output_dir),
                "--public-dir",
                str(public_dir),
                "--image-size",
                "96",
                "--iterations",
                "120",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    manifest = json.loads((output_dir / "plush-semantic-closure-manifest.json").read_text(encoding="utf-8"))
    exported = read_ply(output_dir / "plush_semantic_objects.ply")

    assert "manifest=" in output
    assert manifest["gaussian_source"] == "external_3dgs_splat"
    assert manifest["semantic_source"] == "projected_3dgs_color_masks"
    assert manifest["acceptance"]["real_3dgs_scene_can_render"] is True
    assert manifest["acceptance"]["mask_guidance_changed_object_field"] is True
    assert manifest["acceptance"]["mask_vote_quality_audit_available"] is True
    assert manifest["acceptance"]["projection_loss_decreased"] is True
    assert manifest["object_field_delta"]["changed_gaussians"] > 0
    assert manifest["training"]["final_loss"] < manifest["training"]["initial_loss"]
    assert manifest["training"]["supervised_gaussians"] == cloud.count
    assert manifest["training"]["vote_quality"]["slots"] == 4
    assert set(np.unique(exported.vertices["object_id"])) == {0, 1, 2, 3}
    assert exported.vertices["red"].tolist() == cloud.vertices["red"].tolist()
    assert (public_dir / "plush_semantic_objects.ply").exists()
    assert (public_dir / "plush_semantic.splat").exists()

    assert (
        main(
            [
                "demo",
                "verify-plush-semantic-closure",
                str(output_dir / "plush-semantic-closure-manifest.json"),
                "--asset-library",
                str(asset_library),
                "--min-views",
                "1",
            ]
        )
        == 0
    )
    verify_output = capsys.readouterr().out
    assert "passed=true" in verify_output
    assert "check=gaussian_source_is_real_3dgs status=pass" in verify_output
    assert "check=semantic_source_is_2d_color_masks status=pass" in verify_output
    assert "check=mask_vote_quality_audit_available status=pass" in verify_output
    assert "check=mask_guidance_changed_object_field status=pass" in verify_output
    assert "check=viewer_ply_exports_object_id status=pass" in verify_output


def test_demo_lego_alpha_closure_builds_proxy_assets(tmp_path, capsys):
    dataset = tmp_path / "nerf-synthetic-lego"
    (dataset / "train").mkdir(parents=True)
    _write_rgba_png(
        dataset / "train" / "r_0.png",
        np.array(
            [
                [[220, 190, 20, 255], [210, 40, 30, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[230, 180, 25, 255], [30, 25, 20, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [160, 160, 150, 255], [150, 150, 140, 255]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [25, 25, 25, 255], [160, 160, 150, 255]],
            ],
            dtype=np.uint8,
        ),
    )
    (dataset / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": float(np.pi / 2.0),
                "frames": [
                    {
                        "file_path": "./train/r_0",
                        "transform_matrix": np.eye(4, dtype=float).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "demo"
    public_dir = tmp_path / "public"

    assert (
        main(
            [
                "demo",
                "lego-alpha-closure",
                "--dataset",
                str(dataset),
                "--output-dir",
                str(output_dir),
                "--public-dir",
                str(public_dir),
                "--max-frames",
                "1",
                "--sample-stride",
                "1",
                "--depth",
                "2.0",
                "--iterations",
                "80",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    manifest = json.loads((output_dir / "lego-alpha-closure-manifest.json").read_text(encoding="utf-8"))
    exported = read_ply(output_dir / "lego_v1_objects.ply")
    splat = read_splat(output_dir / "lego_proxy.splat")

    assert "manifest=" in output
    assert manifest["acceptance"]["gaussian_proxy_saved"] is True
    assert manifest["acceptance"]["real_mask_manifest_saved"] is True
    assert manifest["acceptance"]["mask_guidance_changed_object_field"] is True
    assert manifest["acceptance"]["mask_vote_quality_audit_available"] is True
    assert manifest["object_field_delta"]["changed_gaussians"] > 0
    assert manifest["training"]["final_loss"] < manifest["training"]["initial_loss"]
    assert manifest["training"]["supervised_gaussians"] == exported.count
    assert manifest["training"]["vote_quality"]["slots"] == 4
    assert exported.count == 8
    assert splat.count == exported.count
    assert "object_id" in exported.fields
    assert (public_dir / "lego_alpha_v1_objects.ply").exists()
    assert (public_dir / "lego_alpha_proxy.splat").exists()

    assert (
        main(
            [
                "demo",
                "verify-lego-alpha-closure",
                str(output_dir / "lego-alpha-closure-manifest.json"),
                "--min-frames",
                "1",
            ]
        )
        == 0
    )
    verify_output = capsys.readouterr().out
    assert "passed=true" in verify_output
    assert "check=semantic_source_is_real_2d_masks status=pass" in verify_output
    assert "check=mask_vote_quality_audit_available status=pass" in verify_output
    assert "check=mask_guidance_changed_object_field status=pass" in verify_output
    assert "check=viewer_ply_exports_object_id status=pass" in verify_output


def test_goal_audit_reports_split_evidence_and_missing_unified_demo(tmp_path, capsys):
    v1_dir = tmp_path / "v1"
    lego_dir = tmp_path / "lego"
    public_dir = tmp_path / "public"
    asset_library = _write_test_asset_library(tmp_path / "assetLibrary.js")
    input_path = tmp_path / "objects.ply"
    splat_path = tmp_path / "scene.splat"
    write_ply(input_path, _camera_cloud_with_object_ids(), fmt="ascii")
    splat_path.write_bytes(b"splat")
    dataset = _write_small_lego_dataset(tmp_path / "nerf-synthetic-lego")

    assert (
        main(
            [
                "demo",
                "v1-closure",
                "--input",
                str(input_path),
                "--splat",
                str(splat_path),
                "--output-dir",
                str(v1_dir),
                "--public-dir",
                str(public_dir),
                "--image-size",
                "96",
                "--iterations",
                "80",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "demo",
                "lego-alpha-closure",
                "--dataset",
                str(dataset),
                "--output-dir",
                str(lego_dir),
                "--public-dir",
                str(public_dir),
                "--max-frames",
                "1",
                "--sample-stride",
                "1",
                "--depth",
                "2.0",
                "--iterations",
                "80",
            ]
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
            [
                "demo",
                "audit-v1-goal",
                "--v1-manifest",
                str(v1_dir / "v1-closure-manifest.json"),
                "--lego-manifest",
                str(lego_dir / "lego-alpha-closure-manifest.json"),
                "--semantic-manifest",
                str(tmp_path / "missing-semantic.json"),
                "--trained-manifest",
                str(tmp_path / "missing-training-output.json"),
                "--asset-library",
                str(asset_library),
                "--allow-incomplete",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "passed=false" in output
    assert "check=real_3dgs_scene_can_render status=pass" in output
    assert "check=mask_guidance_influences_object_field status=pass" in output
    assert "check=unified_real_3dgs_mask_demo_available status=fail" in output
    assert "completion_blockers=unified_real_3dgs_mask_demo_available" in output


def test_goal_audit_passes_when_unified_trained_manifest_exists(tmp_path):
    v1_dir = tmp_path / "v1"
    lego_dir = tmp_path / "lego"
    public_dir = tmp_path / "public"
    asset_library = _write_test_asset_library(tmp_path / "assetLibrary.js")
    input_path = tmp_path / "objects.ply"
    splat_path = tmp_path / "scene.splat"
    write_ply(input_path, _camera_cloud_with_object_ids(), fmt="ascii")
    splat_path.write_bytes(b"splat")
    dataset = _write_small_lego_dataset(tmp_path / "nerf-synthetic-lego")
    assert (
        main(
            [
                "demo",
                "v1-closure",
                "--input",
                str(input_path),
                "--splat",
                str(splat_path),
                "--output-dir",
                str(v1_dir),
                "--public-dir",
                str(public_dir),
                "--image-size",
                "96",
                "--iterations",
                "80",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "demo",
                "lego-alpha-closure",
                "--dataset",
                str(dataset),
                "--output-dir",
                str(lego_dir),
                "--public-dir",
                str(public_dir),
                "--max-frames",
                "1",
                "--sample-stride",
                "1",
                "--depth",
                "2.0",
                "--iterations",
                "80",
            ]
        )
        == 0
    )
    trained_manifest = tmp_path / "trained" / "training-output-manifest.json"
    trained_manifest.parent.mkdir(parents=True)
    trained_splat = trained_manifest.parent / "trained.splat"
    trained_objects = trained_manifest.parent / "trained_objects.ply"
    trained_splat.write_bytes(b"splat")
    write_ply(trained_objects, _camera_cloud_with_object_ids(), fmt="ascii")
    trained_manifest.write_text(
        json.dumps(
            {
                "gaussian_source": "external_3dgs_training_output",
                "splat_path": str(trained_splat),
                "object_ply": str(trained_objects),
                "object_field_delta": {"changed_gaussians": 2},
                "acceptance": {
                    "mask_guidance_changed_object_field": True,
                    "projection_loss_decreased": True,
                },
            }
        ),
        encoding="utf-8",
    )

    result = audit_v1_goal(
        v1_manifest=v1_dir / "v1-closure-manifest.json",
        lego_manifest=lego_dir / "lego-alpha-closure-manifest.json",
        semantic_manifest=tmp_path / "missing-semantic.json",
        trained_manifest=trained_manifest,
        asset_library_path=asset_library,
    )

    assert result.passed is True
    assert result.summary["completion_blockers"] == []


def test_goal_audit_passes_with_unified_semantic_demo(tmp_path):
    v1_dir = tmp_path / "v1"
    lego_dir = tmp_path / "lego"
    semantic_dir = tmp_path / "semantic"
    public_dir = tmp_path / "public"
    asset_library = _write_test_asset_library(tmp_path / "assetLibrary.js")
    splat_path = tmp_path / "scene.splat"
    splat_path.write_bytes(b"splat")
    dataset = _write_small_lego_dataset(tmp_path / "nerf-synthetic-lego")
    v1_input = tmp_path / "objects.ply"
    semantic_input = tmp_path / "plush_raw.ply"
    write_ply(v1_input, _camera_cloud_with_object_ids(), fmt="ascii")
    write_ply(semantic_input, _camera_color_cloud(), fmt="ascii")

    assert (
        main(
            [
                "demo",
                "v1-closure",
                "--input",
                str(v1_input),
                "--splat",
                str(splat_path),
                "--output-dir",
                str(v1_dir),
                "--public-dir",
                str(public_dir),
                "--image-size",
                "96",
                "--iterations",
                "80",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "demo",
                "lego-alpha-closure",
                "--dataset",
                str(dataset),
                "--output-dir",
                str(lego_dir),
                "--public-dir",
                str(public_dir),
                "--max-frames",
                "1",
                "--sample-stride",
                "1",
                "--depth",
                "2.0",
                "--iterations",
                "80",
            ]
        )
        == 0
    )
    assert (
        main(
            [
                "demo",
                "plush-semantic-closure",
                "--input",
                str(semantic_input),
                "--splat",
                str(splat_path),
                "--output-dir",
                str(semantic_dir),
                "--public-dir",
                str(public_dir),
                "--image-size",
                "96",
                "--iterations",
                "80",
            ]
        )
        == 0
    )

    result = audit_v1_goal(
        v1_manifest=v1_dir / "v1-closure-manifest.json",
        lego_manifest=lego_dir / "lego-alpha-closure-manifest.json",
        semantic_manifest=semantic_dir / "plush-semantic-closure-manifest.json",
        trained_manifest=tmp_path / "missing-training-output.json",
        asset_library_path=asset_library,
    )

    assert result.passed is True
    assert result.summary["current_evidence"] == "unified"
    assert result.summary["completion_blockers"] == []


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


def _camera_cloud_with_object_ids() -> GaussianCloud:
    cloud = _camera_cloud()
    fields = list(cloud.vertices.dtype.descr)
    fields.append(("object_id", "i4"))
    vertices = np.empty(cloud.count, dtype=np.dtype(fields))
    for name in cloud.fields:
        vertices[name] = cloud.vertices[name]
    vertices["object_id"] = np.array([0, 0, 1, 1], dtype=np.int32)
    return GaussianCloud(vertices=vertices, source_format="ascii")


def _camera_color_cloud() -> GaussianCloud:
    cloud = _camera_cloud()
    vertices = cloud.vertices.copy()
    vertices["red"] = np.array([220, 155, 30, 150], dtype=np.uint8)
    vertices["green"] = np.array([30, 145, 25, 150], dtype=np.uint8)
    vertices["blue"] = np.array([30, 75, 20, 150], dtype=np.uint8)
    return GaussianCloud(vertices=vertices, source_format="ascii")


def _write_small_lego_dataset(dataset):
    (dataset / "train").mkdir(parents=True)
    _write_rgba_png(
        dataset / "train" / "r_0.png",
        np.array(
            [
                [[220, 190, 20, 255], [210, 40, 30, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[230, 180, 25, 255], [30, 25, 20, 255], [0, 0, 0, 0], [0, 0, 0, 0]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [160, 160, 150, 255], [150, 150, 140, 255]],
                [[0, 0, 0, 0], [0, 0, 0, 0], [25, 25, 25, 255], [160, 160, 150, 255]],
            ],
            dtype=np.uint8,
        ),
    )
    (dataset / "transforms_train.json").write_text(
        json.dumps(
            {
                "camera_angle_x": float(np.pi / 2.0),
                "frames": [
                    {
                        "file_path": "./train/r_0",
                        "transform_matrix": np.eye(4, dtype=float).tolist(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return dataset


def _write_test_asset_library(path):
    path.write_text(
        """
        plush-v1-closure-local
        /samples/plush_v1_objects.ply
        /samples/plush.splat
        plush-semantic-closure-local
        /samples/plush_semantic_objects.ply
        /samples/plush_semantic.splat
        nerf-lego-alpha-closure-local
        /samples/lego_alpha_v1_objects.ply
        /samples/lego_alpha_proxy.splat
        """,
        encoding="utf-8",
    )
    return path


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


class _FakeSamGenerator:
    def generate(self, image: np.ndarray):
        assert image.shape == (3, 3, 3)
        large = np.array(
            [[True, True, False], [True, True, False], [False, False, False]],
            dtype=bool,
        )
        medium = np.array(
            [[False, False, False], [False, False, False], [True, True, False]],
            dtype=bool,
        )
        tiny = np.array(
            [[False, False, True], [False, False, False], [False, False, False]],
            dtype=bool,
        )
        return [
            {
                "segmentation": medium,
                "area": 2,
                "predicted_iou": 0.82,
                "bbox": [0, 2, 2, 1],
            },
            {
                "segmentation": tiny,
                "area": 1,
                "predicted_iou": 0.99,
                "bbox": [2, 0, 1, 1],
            },
            {
                "segmentation": large,
                "area": 4,
                "predicted_iou": 0.91,
                "bbox": [0, 0, 2, 2],
            },
        ]


def _write_rgba_png(path, pixels: np.ndarray) -> None:
    if pixels.ndim != 3 or pixels.shape[2] != 4 or pixels.dtype != np.uint8:
        raise ValueError("pixels must be uint8 RGBA")
    height, width, _channels = pixels.shape
    raw = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    png = b"\x89PNG\r\n\x1a\n"
    png += _png_chunk(
        b"IHDR",
        struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0),
    )
    png += _png_chunk(b"IDAT", zlib.compress(raw))
    png += _png_chunk(b"IEND", b"")
    path.write_bytes(png)


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    checksum = zlib.crc32(kind + data) & 0xFFFFFFFF
    return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", checksum)
