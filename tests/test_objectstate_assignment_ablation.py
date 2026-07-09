from __future__ import annotations

import json

import numpy as np
import pytest

from objgauss.core.gaussian import GaussianCloud
from objgauss.core.objectstate_assignment_ablation import (
    OBJECTSTATE_ASSIGNMENT_ABLATION_SCHEMA,
    objectstate_assignment_ablation_summary,
    validate_objectstate_assignment_ablation_summary,
)


def test_assignment_ablation_finds_minimum_sufficient_evidence(tmp_path):
    summary = objectstate_assignment_ablation_summary(
        _color_separable_cloud(offset=0.0),
        _target_assignment(),
        _color_separable_cloud(offset=0.2),
        _target_assignment(),
        output_dir=tmp_path,
        sample_id="ablation-test",
        train_sample_id="train-color-scene",
        test_sample_id="heldout-color-scene",
        policies=("xyz", "xyz_rgb", "xyz_rgb_opacity"),
        iterations=120,
        learning_rate=0.4,
        assignment_weight=1.0,
        compactness_weight=0.0,
        seed=3,
        test_ari_floor=0.5,
        test_purity_floor=0.75,
    )

    assert summary["schema"] == OBJECTSTATE_ASSIGNMENT_ABLATION_SCHEMA
    assert summary["status"] == "objectstate_assignment_ablation_pass"
    assert summary["train_dataset"]["split"] == "train"
    assert summary["test_dataset"]["split"] == "test"
    assert [variant["policy"] for variant in summary["variants"]] == [
        "xyz",
        "xyz_rgb",
        "xyz_rgb_opacity",
    ]
    variants = {variant["policy"]: variant for variant in summary["variants"]}
    assert variants["xyz"]["feature_policy"]["uses_position_cost"] is True
    assert variants["xyz"]["feature_policy"]["uses_feature_cost"] is False
    assert variants["xyz_rgb"]["feature_policy"]["uses_color"] is True
    assert variants["xyz"]["test_after_metrics"]["ari"] < 0.5
    assert variants["xyz_rgb"]["test_after_metrics"]["ari"] == pytest.approx(1.0)
    assert variants["xyz_rgb"]["test_after_metrics"]["purity"] == pytest.approx(1.0)
    assert summary["ranking"]["minimum_sufficient_evidence"]["found"] is True
    assert summary["ranking"]["minimum_sufficient_evidence"]["policy"] == "xyz_rgb"
    assert summary["shortcut_diagnostics"]["requires_non_spatial_evidence"] is True
    assert summary["next_stage_gate"]["identity_gate_handoff_allowed"] is True
    assert summary["next_stage_gate"]["long_training_allowed"] is False
    assert summary["claim_policy"]["compares_minimum_sufficient_evidence"] is True
    assert validate_objectstate_assignment_ablation_summary(summary) == summary

    summary_path = tmp_path / "ablation-summary.json"
    assert summary_path.exists()
    assert json.loads(summary_path.read_text(encoding="utf-8"))["schema"] == (
        OBJECTSTATE_ASSIGNMENT_ABLATION_SCHEMA
    )
    assert (tmp_path / "xyz_rgb" / "assignment-generalization-final-state.json").exists()


def test_assignment_ablation_requires_semantic_feature_pair(tmp_path):
    with pytest.raises(ValueError, match="semantic ablation requires both"):
        objectstate_assignment_ablation_summary(
            _color_separable_cloud(offset=0.0),
            _target_assignment(),
            _color_separable_cloud(offset=0.2),
            _target_assignment(),
            output_dir=tmp_path,
            policies=("xyz_rgb_opacity_semantic",),
            semantic_train_features=np.ones((4, 2), dtype=np.float32),
        )


def _target_assignment() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
        ],
        dtype=np.float32,
    )


def _color_separable_cloud(*, offset: float) -> GaussianCloud:
    dtype = np.dtype(
        [
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "f4"),
            ("green", "f4"),
            ("blue", "f4"),
            ("opacity", "f4"),
        ]
    )
    vertices = np.zeros(4, dtype=dtype)
    vertices["x"] = [0.0 + offset, 1.0 + offset, 0.0 + offset, 1.0 + offset]
    vertices["y"] = [0.0, 0.0, 0.0, 0.0]
    vertices["z"] = [0.0, 0.0, 0.0, 0.0]
    vertices["red"] = [1.0, 1.0, 0.0, 0.0]
    vertices["green"] = [0.0, 0.0, 0.0, 0.0]
    vertices["blue"] = [0.0, 0.0, 1.0, 1.0]
    vertices["opacity"] = [1.0, 1.0, 1.0, 1.0]
    return GaussianCloud(vertices)
