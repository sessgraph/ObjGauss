from __future__ import annotations

import json

import numpy as np
import pytest

from objgauss.cli import main
from objgauss.datasets.objectstate_model_diagnostic_synthetic import (
    OBJECTSTATE_MODEL_DIAGNOSTIC_CASES,
    OBJECTSTATE_MODEL_DIAGNOSTIC_DATASET_SCHEMA,
    build_objectstate_model_diagnostic_dataset,
    diagnostic_semantic_proxy,
)
from objgauss.evaluation.objectstate_model_diagnostic import (
    OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES,
    OBJECTSTATE_MODEL_DIAGNOSTIC_SCHEMA,
    identity_swap_metrics,
    validate_objectstate_model_diagnostic,
)
from objgauss.pipelines.objectstate_model_diagnostic import (
    OBJECTSTATE_MODEL_DIAGNOSTIC_DATASET_BUNDLE_SCHEMA,
    OBJECTSTATE_MODEL_DIAGNOSTIC_RUN_SCHEMA,
)
from objgauss.pipelines.objectstate_model_v0 import (
    objectstate_model_v0_state_from_dict,
)


def test_diagnostic_dataset_has_named_hard_cases_and_layout_safe_split():
    first = build_objectstate_model_diagnostic_dataset(
        points_per_instance=64,
        seed=31,
    )
    second = build_objectstate_model_diagnostic_dataset(
        points_per_instance=64,
        seed=31,
    )
    manifest = first.as_dict()

    assert manifest["schema"] == OBJECTSTATE_MODEL_DIAGNOSTIC_DATASET_SCHEMA
    assert manifest["split"]["heldout_scene_ids"] == [0, 3, 6, 9, 12]
    assert manifest["split"]["scene_overlap_count"] == 0
    assert manifest["split"]["layout_overlap_count"] == 0
    assert set(manifest["hard_cases"]) == set(OBJECTSTATE_MODEL_DIAGNOSTIC_CASES)
    heldout_cases = {
        row["case"] for row in manifest["observations"] if row["split"] == "heldout"
    }
    assert heldout_cases == set(OBJECTSTATE_MODEL_DIAGNOSTIC_CASES)
    assert manifest["cross_view_pairs"] == [
        {
            "pair_id": "layout-103-cross-view",
            "layout_id": 103,
            "anchor_scene_id": 9,
            "target_scene_id": 12,
        }
    ]
    occlusion = next(
        row for row in manifest["observations"] if row["case"] == "cube_behind_cup"
    )
    assert 0 < occlusion["observed_point_counts"]["1"] < 32
    for scene_id in range(13):
        mask = first.cloud.vertices["scene_id"] == scene_id
        np.testing.assert_array_equal(
            np.unique(first.cloud.vertices["gt_instance_id"][mask]),
            np.asarray([0, 1, 2, 3]),
        )

    semantic = diagnostic_semantic_proxy(first.cloud)
    target = first.cloud.vertices["gt_instance_id"]
    assert np.array_equal(np.unique(semantic[target == 0], axis=0), [[1.0, 0.0, 0.0]])
    assert np.array_equal(np.unique(semantic[target == 1], axis=0), [[1.0, 0.0, 0.0]])
    np.testing.assert_array_equal(first.cloud.vertices, second.cloud.vertices)
    assert first.as_dict() == second.as_dict()


def test_identity_swap_metrics_freezes_anchor_mapping():
    anchor_target = np.asarray([0, 0, 1, 1], dtype=np.int64)
    anchor_predicted = np.asarray([4, 4, 9, 9], dtype=np.int64)
    target_target = np.asarray([0, 0, 1, 1], dtype=np.int64)
    target_predicted = np.asarray([9, 9, 4, 4], dtype=np.int64)

    result = identity_swap_metrics(
        anchor_predicted,
        anchor_target,
        target_predicted,
        target_target,
    )

    assert result["identity_count"] == 2
    assert result["slot_swap_count"] == 2
    assert result["slot_swap_rate"] == pytest.approx(1.0)
    assert result["unmapped_identity_count"] == 0


def test_model_diagnostic_cli_writes_all_seed_case_and_report_evidence(
    tmp_path,
    capsys,
):
    output = tmp_path / "diagnostic"

    main(
        [
            "training",
            "objectstate-model-diagnostic",
            "--output-dir",
            str(output),
            "--run-id",
            "diagnostic-test",
            "--points-per-instance",
            "64",
            "--iterations",
            "5",
            "--hidden-dim",
            "8",
            "--seeds",
            "0",
            "1",
            "2",
            "--dataset-seed",
            "41",
        ]
    )
    stdout = capsys.readouterr().out
    summary = json.loads((output / "diagnostic-summary.json").read_text(encoding="utf-8"))
    manifest = json.loads((output / "dataset-manifest.json").read_text(encoding="utf-8"))
    diagnostic = summary["diagnostic"]

    assert summary["schema"] == OBJECTSTATE_MODEL_DIAGNOSTIC_RUN_SCHEMA
    assert set(summary["datasets"]) == {"hard_case", "m2_original"}
    assert manifest["schema"] == OBJECTSTATE_MODEL_DIAGNOSTIC_DATASET_BUNDLE_SCHEMA
    assert set(manifest["datasets"]) == {"hard_case", "m2_original"}
    assert diagnostic["schema"] == OBJECTSTATE_MODEL_DIAGNOSTIC_SCHEMA
    assert validate_objectstate_model_diagnostic(diagnostic) == diagnostic
    assert diagnostic["leakage_gate"]["passed"] is True
    assert diagnostic["training_seed_policy"]["seeds"] == [0, 1, 2]
    assert diagnostic["training_seed_policy"]["best_seed_selection"] is False
    assert len(diagnostic["ablation_matrix"]) == 27
    assert set(diagnostic["hard_case"]["variants"]) == set(
        OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES
    )
    assert set(diagnostic["m2_reproduction"]["variants"]) == set(
        OBJECTSTATE_MODEL_DIAGNOSTIC_POLICIES
    )
    assert all(
        tuple(run["split"]["heldout_frame_ids"]) == (0, 3, 6, 9, 12)
        for runs in summary["training"]["runs"]["hard_case"].values()
        for run in runs
    )
    assert diagnostic["comparison"]["verdict"] in {
        "model_variant_beats_baselines_and_preserves_cross_view_identity",
        "per_frame_segmentation_gain_but_cross_view_identity_failed",
        "no_model_variant_beats_recorded_segmentation_baselines",
    }
    assert "verdict=" in stdout
    for name in (
        "dataset-manifest.json",
        "diagnostic-summary.json",
        "ablation-matrix.csv",
        "hard-case-matrix.csv",
        "error-taxonomy.csv",
        "report-artifact.json",
    ):
        assert (output / name).is_file()

    semantic_checkpoint = json.loads(
        (output / "models/hard_case/xyz_rgb_semantic/seed-0/checkpoint.json").read_text(
            encoding="utf-8"
        )
    )
    assert semantic_checkpoint["feature_order"][-3:] == [
        "semantic_cube",
        "semantic_cup",
        "semantic_tool",
    ]
    restored = objectstate_model_v0_state_from_dict(semantic_checkpoint)
    assert restored.config.input_dim == 10
    with pytest.raises(ValueError, match="require predict_features"):
        restored.predict(build_objectstate_model_diagnostic_dataset(points_per_instance=64).cloud)

    artifact = json.loads((output / "report-artifact.json").read_text(encoding="utf-8"))
    assert artifact["surface"] == "report"
    assert artifact["manifest"]["blocks"][0]["body"].startswith("# Why ObjectState")
    assert len(artifact["manifest"]["charts"]) == 2
    assert artifact["snapshot"]["status"] == "ready"
