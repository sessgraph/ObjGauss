from __future__ import annotations

import json

import numpy as np

from objgauss.datasets.objectstate_multi_object_synthetic import (
    OBJECTSTATE_MULTI_OBJECT_DATASET_SCHEMA,
    build_multi_object_synthetic_dataset,
)
from objgauss.model_manifest import validate_model_artifact_manifest
from objgauss.pipelines.objectstate_multi_object_benchmark import (
    OBJECTSTATE_MULTI_OBJECT_RUN_SCHEMA,
    run_multi_object_instance_benchmark,
)
from objgauss.ply import read_ply


def test_multi_object_dataset_has_independent_instances_and_scene_split():
    first = build_multi_object_synthetic_dataset(
        scene_count=6,
        points_per_instance=64,
        heldout_stride=3,
        split_seed=0,
        seed=31,
    )
    second = build_multi_object_synthetic_dataset(
        scene_count=6,
        points_per_instance=64,
        heldout_stride=3,
        split_seed=0,
        seed=31,
    )
    manifest = first.as_dict()

    assert manifest["schema"] == OBJECTSTATE_MULTI_OBJECT_DATASET_SCHEMA
    assert manifest["source"]["type"] == "procedural_instance_authorship"
    assert manifest["source"]["target_derived_from_rgb"] is False
    assert "gt_instance_id" not in manifest["source"]["model_feature_fields"]
    assert manifest["split"]["scene_overlap_count"] == 0
    assert manifest["split"]["heldout_scene_ids"] == [0, 3]
    assert manifest["stress_contract"]["same_color_instance_ids"] == [0, 1]
    assert all(
        np.all(first.cloud.vertices[field] > 0)
        for field in ("scale_0", "scale_1", "scale_2")
    )
    assert manifest["instances"][0]["rgb"] == manifest["instances"][1]["rgb"]
    assert any(scene["contact"] for scene in manifest["scenes"] if scene["split"] == "heldout")
    assert any(
        scene["partial_observation"]
        for scene in manifest["scenes"]
        if scene["split"] == "heldout"
    )
    for scene_id in range(6):
        labels = np.unique(
            first.cloud.vertices["gt_instance_id"]
            [first.cloud.vertices["scene_id"] == scene_id]
        )
        np.testing.assert_array_equal(labels, np.asarray([0, 1, 2, 3]))
    np.testing.assert_array_equal(first.cloud.vertices, second.cloud.vertices)
    assert first.as_dict() == second.as_dict()


def test_multi_object_benchmark_writes_reviewable_raw_prediction_and_gt(tmp_path):
    viewer_dir = tmp_path / "public" / "models" / "multi-object-local"
    run = run_multi_object_instance_benchmark(
        tmp_path / "run",
        run_id="multi-object-test",
        scene_count=6,
        points_per_instance=64,
        heldout_stride=3,
        iterations=30,
        learning_rate=0.06,
        hidden_dim=10,
        seed=0,
        dataset_seed=41,
        viewer_dir=viewer_dir,
    )
    summary = run.summary

    assert summary["schema"] == OBJECTSTATE_MULTI_OBJECT_RUN_SCHEMA
    assert summary["benchmark"]["leakage_gate"]["passed"] is True
    assert summary["benchmark"]["split"]["scene_overlap_count"] == 0
    assert set(summary["benchmark"]["aggregate"]) == {
        "xyz_kmeans",
        "rgb_kmeans",
        "connected_components_3d",
        "objectstate_model_v0",
    }
    assert summary["benchmark"]["comparison"]["verdict"] in {
        "model_better_than_recorded_baselines",
        "model_tied_best_recorded_baseline",
        "model_not_better_than_recorded_baselines",
    }
    assert summary["claim_policy"]["failed_result_remains_visible"] is True
    assert len(run.scene_artifacts) == 2

    first_scene = run.scene_artifacts[0]
    raw = read_ply(run.output_dir / first_scene["raw"]["path"])
    prediction = read_ply(run.output_dir / first_scene["prediction"]["path"])
    ground_truth = read_ply(run.output_dir / first_scene["ground_truth"]["path"])
    assert "gt_instance_id" not in raw.fields
    assert "shape_id" not in raw.fields
    assert "object_id" not in raw.fields
    assert "object_id" in prediction.fields
    assert "object_id" in ground_truth.fields
    assert len(np.unique(ground_truth.vertices["object_id"])) == 4
    assert raw.count == prediction.count == ground_truth.count

    benchmark_on_disk = json.loads(run.benchmark_path.read_text(encoding="utf-8"))
    assert benchmark_on_disk == summary
    manifest = json.loads(run.manifest_path.read_text(encoding="utf-8"))
    assert validate_model_artifact_manifest(manifest, require_browser_ready=True).passed
    assert {artifact["role"] for artifact in manifest["artifacts"]} == {
        "quick_splat",
        "model_input",
        "object_edit",
        "ground_truth",
        "objectstate_model",
    }
    assert manifest["source"]["target_derived_from_rgb"] is False
    assert manifest["quality_evidence"][0]["comparison"] == summary["benchmark"]["comparison"]
    assert run.viewer_manifest_path == viewer_dir / "model-manifest.json"
    staged_manifest = json.loads(run.viewer_manifest_path.read_text(encoding="utf-8"))
    assert staged_manifest == manifest
    for artifact in staged_manifest["artifacts"]:
        assert (viewer_dir / artifact["path"]).is_file()
    training = json.loads(run.training_summary_path.read_text(encoding="utf-8"))
    assert training["split"]["field"] == "scene_id"
    assert training["split"]["policy"] == "deterministic_complete_scene_holdout"
    assert training["claim_policy"]["complete_scene_holdout"] is True
    assert "same_scene_heldout_frames_only" not in training["claim_policy"]
